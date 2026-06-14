"""
Módulo de construcción del perfil técnico de un usuario.

Recibe los bloques puntuados del módulo de procesamiento (uno por skill/evidencia)
y los agrega en un perfil estructurado por skill. Calcula:

- evidence_count: número de evidencias que respaldan la skill
- avg_composite_score: calidad media de las evidencias
- evidence_diversity_score (nivel perfil): ratio de artifact_types distintos
  que respaldan la skill respecto al total posible (file, commit, issue, PR).
- confidence: categoría cualitativa basada en avg_composite_score × count
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from src.utils.logger import get_logger

log = get_logger(__name__)


def load_jsonl(input_path: Path) -> List[Dict[str, Any]]:
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _profile_level_diversity(artifact_types: set) -> float:
    """
    Diversity a nivel de perfil: cuántos tipos de artefacto distintos
    respaldan la skill. Máximo 4 tipos (file, commit, issue, pull_request).

    Diferencia respecto a evidence_diversity_score (nivel bloque):
    ese mide cuántos tipos de señal detectaron la skill dentro de UNA evidencia.
    Este mide cuántos tipos de fuente distintos la avalan en el perfil completo.
    """
    max_artifact_types = 4
    return round(len(artifact_types) / max_artifact_types, 4)


def _compute_confidence(avg_score: float, evidence_count: int, diversity_profile: float) -> str:
    """
    Categoría cualitativa para presentación en la respuesta RAG.
    Pondera la calidad media de las evidencias (60%), la cantidad de
    evidencias (25%) y la diversidad de tipos de artefacto (15%).
    """
    count_factor = min(evidence_count, 5) / 5

    weight = (0.60 * avg_score + 0.25 * count_factor + 0.15 * diversity_profile)
    if weight >= 0.65:
        return "alta"
    if weight >= 0.35:
        return "media"
    return "baja"


def build_user_profile(
    username: str,
    processed_input_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Construye el perfil técnico agregado del usuario a partir de
    los bloques puntuados generados por el módulo de procesamiento.

    La estructura resultante permite al módulo RAG consultar directamente
    qué skills tiene el usuario y con qué nivel de confianza, con
    trazabilidad completa a las fuentes.
    """
    log.info(f"Construyendo perfil técnico para '{username}'")
    records = load_jsonl(processed_input_path)
    log.info(f"  Bloques a agregar: {len(records)}")

    skill_scores: Dict[str, List[float]] = defaultdict(list)
    skill_repos: Dict[str, set] = defaultdict(set)
    skill_paths: Dict[str, set] = defaultdict(set)
    skill_artifact_types: Dict[str, set] = defaultdict(set)
    skill_explanations: Dict[str, List[str]] = defaultdict(list)
    skill_signals: Dict[str, List[str]] = defaultdict(list)

    total_blocks = len(records)

    for record in records:
        skill = record.get("skill", "").strip()
        if not skill:
            continue

        source = record.get("source", {})
        scores = record.get("scores", {})
        composite = scores.get("composite_score", 0.0)

        repo = source.get("repo", "")
        path = source.get("path", "")
        artifact_type = source.get("artifact_type", "")
        explanation = record.get("explanation", "")

        # El fragmento de código es la evidencia más concreta para el perfil
        fragment = record.get("evidence_fragment", "").strip()
        signal_values = [fragment] if fragment else []

        skill_scores[skill].append(composite)
        if repo:
            skill_repos[skill].add(repo)
        if path:
            skill_paths[skill].add(path)
        if artifact_type:
            skill_artifact_types[skill].add(artifact_type)
        if explanation:
            skill_explanations[skill].append(explanation)
        skill_signals[skill].extend(signal_values)

    hard_skills = []

    for skill, scores_list in skill_scores.items():
        evidence_count = len(scores_list)
        avg_composite = round(sum(scores_list) / evidence_count, 4)
        artifact_types = skill_artifact_types[skill]
        diversity_profile = _profile_level_diversity(artifact_types)

        hard_skills.append({
            "skill": skill,
            "confidence": _compute_confidence(avg_composite, evidence_count, diversity_profile),
            "evidence_count": evidence_count,
            "avg_composite_score": avg_composite,
            "evidence_diversity_score": diversity_profile,
            "evidence_repositories": sorted(skill_repos[skill]),
            "evidence_files": sorted(skill_paths[skill])[:8],  # máximo 8 para no saturar el perfil JSON
            "artifact_types": sorted(artifact_types),
            "sample_explanations": list(dict.fromkeys(
                e for e in skill_explanations[skill] if e
            ))[:3],
            "sample_signals": list(dict.fromkeys(
                s for s in skill_signals[skill] if s
            ))[:4],
        })

    hard_skills.sort(
        key=lambda s: (s["avg_composite_score"], s["evidence_count"]),
        reverse=True,
    )

    profile = {
        "user": username,
        "profile_source": str(processed_input_path),
        "total_skill_blocks": total_blocks,
        "unique_skills": len(hard_skills),
        "technical_profile": {
            "hard_skills": hard_skills
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    log.info(f"Perfil generado: {len(hard_skills)} skills únicas → {output_path}")

    top5 = hard_skills[:5]
    for s in top5:
        log.info(
            f"  {s['skill']:<28} score={s['avg_composite_score']:.2f}  "
            f"conf={s['confidence']:<5}  evidencias={s['evidence_count']}"
        )
    if len(hard_skills) > 5:
        log.debug(f"  ... y {len(hard_skills) - 5} skills más en el perfil completo")

    return profile
