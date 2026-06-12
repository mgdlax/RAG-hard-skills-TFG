"""
Modulo de generacion de respuesta RAG (Fase 6 del pipeline).

Recibe los candidatos rankeados con su evidencia y genera una respuesta
en lenguaje natural usando el LLM local (Ollama + Qwen2.5-Coder).

Rol del LLM: SINTETIZAR y NARRAR, nunca atribuir nuevas skills.
Toda la informacion factual (skills, scores, repos, URLs) proviene
del pipeline deterministico; el LLM solo redacta la respuesta final.

Degradacion elegante: si Ollama no esta disponible, _fallback_response()
genera una respuesta estructurada sin LLM basada en metadatos.
"""

import re
import requests
from typing import Any, Dict, List, Set, Tuple

from src.config import OLLAMA_URL, DEFAULT_LLM_MODEL
from src.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_MODEL = DEFAULT_LLM_MODEL

MAX_SKILLS_PER_CANDIDATE = 4   # Skills mostradas por candidato en el prompt
MAX_CANDIDATES_IN_PROMPT = 3   # Candidatos incluidos en el contexto RAG


def _extract_cited_reference_numbers(text: str) -> Set[int]:
    """
    Extrae los números de referencia que el LLM citó efectivamente en su respuesta.

    Ejemplo: "destaca en RAG [1] y LangChain [2][3]." → {1, 2, 3}
    """
    return {int(n) for n in re.findall(r"\[(\d+)\]", text)}


def _filter_references_used(
    references: List[Dict[str, Any]],
    used_nums: Set[int],
) -> List[Dict[str, Any]]:
    """
    Devuelve solo las referencias cuyo número aparece en used_nums.
    Mantiene el orden original de numeración.
    """
    return [r for r in references if r["num"] in used_nums]


def build_rag_prompt(
    query: str,
    candidates: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Construye el prompt RAG con los candidatos y su evidencia de GitHub.

    Cada evidencia recibe un número [N] que el LLM debe citar inline
    en su respuesta. Al final el LLM lista todas las referencias usadas,
    igual que en un informe académico o técnico.
    """
    # Construir lista numerada de evidencias
    references: List[Dict[str, Any]] = []
    context_lines: List[str] = []

    for candidate in candidates[:MAX_CANDIDATES_IN_PROMPT]:
        username = candidate["username"]
        score = candidate["final_score"]
        skills = candidate["matched_skills"][:MAX_SKILLS_PER_CANDIDATE]

        context_lines.append(
            f"## Candidato @{username} — puntuacion RAG: {score:.2f}"
        )

        for s in skills:
            ref_num = len(references) + 1
            references.append({
                "num":          ref_num,
                "username":     username,
                "skill":        s["skill"],
                "repo":         s.get("repo", ""),
                "path":         s.get("path", ""),
                "url":          s.get("url", ""),
                "artifact_type": s.get("artifact_type", ""),
                "score":        s["combined_score"],
            })

            context_lines.append(
                f"  [{ref_num}] Skill: {s['skill']} (score={s['combined_score']:.2f})"
            )
            if s.get("explanation"):
                context_lines.append(f"       {s['explanation']}")
            if s.get("evidence_fragment"):
                context_lines.append(f"       Código: {s['evidence_fragment'][:300]}")

        context_lines.append("")

    context = "\n".join(context_lines)

    prompt = (
        "Eres un asesor tecnico de RRHH. Basandote UNICAMENTE en la evidencia "
        "numerada que se te proporciona, responde la consulta en espanol.\n\n"
        f"CONSULTA:\n{query}\n\n"
        f"EVIDENCIA (cada item tiene un numero de referencia [N]):\n{context}\n"
        "INSTRUCCIONES ESTRICTAS:\n"
        "1. Escribe una respuesta narrativa en espanol de maximo 200 palabras.\n"
        "2. Cada vez que menciones una skill o un candidato, cita el numero de "
        "referencia correspondiente entre corchetes, por ejemplo: [1] o [2][3].\n"
        "3. Usa UNICAMENTE la informacion proporcionada. No inventes skills ni "
        "experiencias que no aparezcan en la evidencia.\n"
        "4. Despues de la respuesta narrativa, escribe exactamente la linea:\n"
        "   ## Referencias\n"
        "   Y lista SOLO las referencias que hayas citado en el texto, con el "
        "formato: [N] @usuario — skill — repo/fichero\n\n"
        "RESPUESTA:"
    )

    return prompt, references


def _build_references_section(references: List[Dict[str, Any]]) -> str:
    """
    Genera la sección de referencias determinista (siempre correcta,
    independientemente de lo que escriba el LLM).

    Se añade al final de la respuesta del LLM para garantizar que
    cada cita tenga repositorio, fichero y URL exactos.
    """
    lines = ["", "---", "## Referencias"]
    for ref in references:
        location = ref["repo"]
        if ref.get("path"):
            location += f" / {ref['path']}"

        line = f"[{ref['num']}] @{ref['username']} — **{ref['skill']}** — `{location}`"
        if ref.get("url"):
            line += f"  \n     <{ref['url']}>"
        lines.append(line)
    return "\n".join(lines)


def generate_response(
    query: str,
    candidates: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Genera la respuesta final del sistema RAG.

    El LLM produce una respuesta narrativa con citas [N] inline.
    La sección de referencias se añade de forma determinista al final
    para garantizar que repo, fichero y URL sean siempre correctos.

    Si Ollama no esta disponible, _fallback_response() produce la misma
    estructura sin LLM.

    Args:
        query:      Consulta original del reclutador.
        candidates: Lista rankeada de candidatos de ranking.py.
        model:      Modelo Ollama a usar.

    Returns:
        Respuesta con citas inline + sección de referencias al final.
    """
    if not candidates:
        return (
            "No se encontraron candidatos relevantes para esta consulta.\n"
            "Asegurate de que hay usuarios indexados: python -m src.main"
        )

    prompt, references = build_rag_prompt(query, candidates)

    try:
        log.debug(f"  Llamando a Ollama ({model})...")
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,    # Baja temperatura: respuestas mas factuales
                    "num_predict": 600,    # Margen extra para la seccion de referencias
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        llm_answer = response.json()["response"].strip()

        # Eliminar la sección ## Referencias que haya generado el LLM
        if "## Referencias" in llm_answer:
            llm_answer = llm_answer[:llm_answer.index("## Referencias")].rstrip()

        # Detectar qué referencias citó realmente el LLM
        cited_nums = _extract_cited_reference_numbers(llm_answer)
        if not cited_nums:
            log.warning("El LLM no citó ninguna referencia [N] — usando fallback estructurado")
            return _fallback_response(query, candidates, references)

        # La sección de referencias final solo muestra las realmente citadas
        used_references = _filter_references_used(references, cited_nums)
        answer = llm_answer + "\n" + _build_references_section(used_references)
        log.info(
            f"  Respuesta generada por LLM ({len(answer)} chars, "
            f"{len(used_references)}/{len(references)} referencias citadas)"
        )
        return answer

    except requests.exceptions.ConnectionError:
        log.warning("Ollama no esta en ejecucion -- usando respuesta estructurada")
        return _fallback_response(query, candidates, references)
    except Exception as e:
        log.warning(f"Error al llamar a Ollama ({e}) -- usando respuesta estructurada")
        return _fallback_response(query, candidates, references)


def _fallback_response(
    query: str,
    candidates: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
) -> str:
    """
    Respuesta estructurada sin LLM, con el mismo formato de citas [N].

    Garantiza que el sistema funcione aunque Ollama este caido.
    Toda la informacion proviene de los metadatos deterministicos del pipeline.
    """
    lines = [
        f"Resultados para: «{query}»",
        "",
        "No se pudo conectar con Ollama. Resultado estructurado:",
        "",
    ]

    # Respuesta narrativa minima con citas
    ref_idx = 0
    for rank, candidate in enumerate(candidates[:MAX_CANDIDATES_IN_PROMPT], 1):
        username = candidate["username"]
        score = candidate["final_score"]
        skills = candidate["matched_skills"][:MAX_SKILLS_PER_CANDIDATE]

        skill_names = []
        cites = []
        for s in skills:
            ref_idx += 1
            skill_names.append(s["skill"])
            cites.append(f"[{ref_idx}]")

        cite_str = "".join(cites)
        lines.append(
            f"#{rank} @{username} (score {score:.2f}) domina: "
            f"{', '.join(skill_names)} {cite_str}"
        )

    lines.append("")

    # Tabla comparativa
    sep = "-" * 52
    lines += [
        sep,
        f"{'Candidato':<22} {'Skills':<8} {'Score'}",
        sep,
    ]
    for c in candidates[:MAX_CANDIDATES_IN_PROMPT]:
        top = ", ".join(s["skill"] for s in c["matched_skills"][:3])
        lines.append(
            f"@{c['username']:<21} {c['skill_count']:<8} {c['final_score']:.3f}"
            f"  ({top})"
        )

    # Referencias deterministas (identicas al modo LLM)
    lines.append(_build_references_section(references))

    return "\n".join(lines)
