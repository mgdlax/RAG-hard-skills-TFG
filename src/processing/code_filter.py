"""
Módulo de procesamiento de evidencias técnicas.

Pipeline por cada evidencia cruda (raw JSONL):
  1. metrics.score_evidence() calcula las 4 métricas estructurales (sin LLM).
  2. El LLM detecta qué skills aparecen en el contenido y genera una
     explicación + fragmento representativo para cada una.
  3. Se emite un bloque por skill detectada, todos con las mismas métricas.

Si enrich_with_llm=False, se guardan los bloques sin skill (útil para
depurar las métricas sin necesitar Ollama).
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from pydantic import ValidationError

from src.config import OLLAMA_URL, DEFAULT_LLM_MODEL
from src.processing.metrics import score_evidence
from src.processing.prompts import build_detection_prompt, DetectionResponse
from src.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_MODEL = DEFAULT_LLM_MODEL


# ─────────────────────────────────────────────────────────────────────────────
#  UTILIDADES JSONL
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(input_path: Path) -> List[Dict[str, Any]]:
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            clean = {k: v for k, v in record.items() if k != "_raw_text"}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  CLIENTE OLLAMA
# ─────────────────────────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["response"]


def _extract_raw_json(response_text: str) -> Optional[str]:
    """
    Extrae el texto JSON de la respuesta del LLM, ignorando texto extra
    y fences de Markdown que el modelo añade aunque se le pida que no.
    Devuelve la cadena JSON candidata o None si no se encuentra ningún objeto.
    """
    text = response_text.strip()

    # 1. Fences de Markdown: ```json...``` o ```...```
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)

    # 2. Primer { hasta el último } (ignora texto previo/posterior)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]

    return None


def _parse_detection_response(response_text: str) -> Optional[DetectionResponse]:
    """
    Convierte la respuesta de texto del LLM en un DetectionResponse validado.

    Intenta parsear el JSON y lo valida con Pydantic. Si el JSON tiene
    newlines literales dentro de strings (error frecuente en modelos locales),
    los repara antes de volver a intentarlo.
    Devuelve None si ningún intento tiene éxito.
    """
    raw = _extract_raw_json(response_text)
    if raw is None:
        return None

    # Intento 1: JSON tal cual
    try:
        data = json.loads(raw)
        return DetectionResponse.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        pass

    # Intento 2: reparar newlines literales dentro de valores string
    try:
        repaired = re.sub(
            r'("(?:[^"\\]|\\.)*")',
            lambda m: m.group(0).replace("\n", "\\n").replace("\r", "\\r"),
            raw,
        )
        data = json.loads(repaired)
        return DetectionResponse.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        log.debug(f"_parse_detection_response: fallo final — {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  DETECCIÓN + ENRIQUECIMIENTO (UN CALL POR EVIDENCIA)
# ─────────────────────────────────────────────────────────────────────────────

def detect_skills_with_llm(
    base_block: Dict[str, Any],
    model: str,
) -> List[Dict[str, Any]]:
    """
    Llama al LLM para detectar skills y generar explicaciones.

    Recibe el bloque base (con métricas ya calculadas) y devuelve
    una lista de bloques completos, uno por skill detectada.
    Cada bloque hereda las métricas del bloque base.

    Si el LLM falla o no detecta skills, devuelve lista vacía.
    """
    source = base_block.get("source", {})
    prompt = build_detection_prompt(
        repo=source.get("repo", ""),
        path=source.get("path", "") or source.get("artifact_type", ""),
        artifact_type=source.get("artifact_type", ""),
        content=base_block.get("_raw_text", "")[:6000],
    )

    t0 = time.time()
    try:
        response_text = call_ollama(prompt=prompt, model=model)
        elapsed = time.time() - t0
        parsed = _parse_detection_response(response_text)

        if parsed is None:
            log.warning(
                f"    LLM devolvió JSON inválido en "
                f"'{source.get('path', '')}' ({elapsed:.1f}s)"
            )
            return []

        if not parsed.skills:
            log.debug(
                f"    LLM: sin skills relevantes en "
                f"'{source.get('path', '')}' ({elapsed:.1f}s)"
            )
            return []

        log.debug(
            f"    LLM: {len(parsed.skills)} skills en "
            f"'{source.get('path', '')}' ({elapsed:.1f}s): "
            f"{[s.skill for s in parsed.skills]}"
        )

        return [
            {
                **base_block,
                "skill":             skill.skill,
                "explanation":       skill.explanation,
                "evidence_fragment": skill.evidence_fragment,
            }
            for skill in parsed.skills
        ]

    except Exception as e:
        elapsed = time.time() - t0
        log.warning(
            f"    LLM falló en '{source.get('path', '')}' "
            f"({elapsed:.1f}s): {e}"
        )
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def process_raw_user_file(
    input_path: Path,
    output_path: Path,
    model: str = DEFAULT_MODEL,
    limit_files: Optional[int] = None,
    enrich_with_llm: bool = True,
) -> List[Dict[str, Any]]:
    """
    Transforma el JSONL crudo en bloques puntuados e indexables.

    Por cada evidencia:
      1. Calcula métricas estructurales (determinístico, sin LLM)
      2. Llama al LLM para detectar skills + generar explicaciones
      3. Emite un bloque por skill detectada

    enrich_with_llm=False guarda bloques sin skill (útil para depurar métricas).
    """
    log.info(f"Iniciando procesamiento: {input_path.name}")
    log.info(f"  enrich_with_llm={enrich_with_llm}  model={model}  limit_files={limit_files}")

    raw_records = load_jsonl(input_path)
    log.info(f"  Evidencias crudas cargadas: {len(raw_records)}")

    if limit_files is not None:
        file_records  = [r for r in raw_records if r.get("artifact_type") == "file"]
        other_records = [r for r in raw_records if r.get("artifact_type") != "file"]
        # El recorte se aplica sobre el orden de extracción (recorrido del árbol
        # de directorios), no sobre la calidad de los ficheros, porque el scoring
        # aún no se ha calculado. Es una limitación de diseño aceptable para el TFG.
        records_to_process = file_records[:limit_files] + other_records
        log.info(
            f"  Límite aplicado: {limit_files} ficheros + "
            f"{len(other_records)} artefactos no-file = "
            f"{len(records_to_process)} total"
        )
    else:
        records_to_process = raw_records

    all_blocks: List[Dict[str, Any]] = []
    no_skills_count = 0
    total = len(records_to_process)
    t_start = time.time()

    for idx, record in enumerate(records_to_process, start=1):
        artifact_type = record.get("artifact_type", "")
        path = record.get("path", "") or artifact_type

        # ── Paso 1: métricas estructurales (sin LLM) ────────────────────────
        base_block = score_evidence(record)
        composite = base_block["scores"]["composite_score"]

        log.info(
            f"  [{idx:>3}/{total}] {artifact_type:<12} {path:<45} "
            f"composite={composite:.2f}"
        )

        if not enrich_with_llm:
            # Modo sin LLM: guardar bloque base sin skill (solo métricas)
            all_blocks.append(base_block)
            continue

        # ── Paso 2: detección de skills + explicaciones (LLM) ───────────────
        skill_blocks = detect_skills_with_llm(base_block, model=model)

        if not skill_blocks:
            no_skills_count += 1
            log.debug(f"    → sin skills detectadas")
            continue

        skill_names = [b["skill"] for b in skill_blocks]
        log.info(f"    → {len(skill_blocks)} skills: {skill_names}")
        all_blocks.extend(skill_blocks)

    elapsed = time.time() - t_start
    save_jsonl(all_blocks, output_path)

    log.info(f"Procesamiento completado en {elapsed:.1f}s")
    log.info(f"  Bloques generados:            {len(all_blocks)}")
    if enrich_with_llm:
        log.info(f"  Evidencias sin skills:        {no_skills_count}/{total}")
        log.info(f"  Skills únicas detectadas:     "
                 f"{len({b['skill'] for b in all_blocks if b.get('skill')})}")
    log.info(f"  Salida guardada en:           {output_path}")

    return all_blocks
