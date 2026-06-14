"""
services/rag_service.py — Pipeline RAG de recomendación técnica.

MODO REAL  (USE_MOCK = False, por defecto)
    Ejecuta el pipeline RAG real de tres fases:
      Fase 5a — retrieve_blocks()     → embedding + búsqueda en ChromaDB
      Fase 5b — rank_candidates()     → agrupación y scoring por usuario
      Fase 6  — generate_response()   → síntesis narrativa con Ollama
    Devuelve la respuesta textual + ranking + evidencias con el formato
    que espera el frontend.

MODO MOCK  (USE_MOCK = True)
    Genera respuestas, rankings y evidencias ficticios pero plausibles
    a partir de los perfiles cargados en session_state. Útil para
    desarrollar el frontend sin ChromaDB ni Ollama.

CONTRATO DE LA INTERFAZ
    ask_question(question, profiles) devuelve un Generator que hace yield de:
        (step_label: str, is_final: bool, result: dict | None)
    El último yield tiene is_final=True y result con:
        {
          "answer":   str,
          "ranking":  [{username, score, matched_skills}],
          "evidences":[{username, repo, path, skill, fragment, score}]
        }
"""

from __future__ import annotations

import random
import time
from typing import Generator, Optional

USE_MOCK: bool = False
"""Cambiar a True para simular el pipeline RAG sin ChromaDB ni Ollama."""

# Pasos del pipeline RAG tal como se muestran en la UI
RAG_STEPS: list[str] = [
    "Generando embedding de la consulta",
    "Recuperando evidencias relevantes de ChromaDB",
    "Calculando similitud semántica",
    "Agrupando evidencias por usuario",
    "Calculando ranking de candidatos",
    "Generando respuesta final con LLM",
]

_STEP_DELAYS: list[float] = [0.5, 0.7, 0.4, 0.3, 0.4, 1.0]

# Tiempo minimo (segundos) que cada paso informativo del modo real permanece
# visible en la UI. El retrieval y el ranking son casi instantaneos, asi que sin
# esta pausa los primeros pasos pasarian demasiado rapido para poder leerse.
_REAL_STEP_MIN_SECONDS = 0.9

_FRAGMENT_TEMPLATES: list[str] = [
    "Implementación de una cadena RAG con {skill} para recuperar y generar respuestas contextuales.",
    "Uso de {skill} para construir un agente conversacional con memoria persistente.",
    "Pipeline de indexación vectorial y recuperación semántica basado en {skill}.",
    "Integración de {skill} con modelos de embeddings para búsqueda semántica eficiente.",
    "Sistema de Q&A sobre documentos técnicos usando {skill} como backbone principal.",
    "Cadena de recuperación aumentada con re-ranking y filtrado usando {skill}.",
    "Orquestación de agentes LLM con herramientas y {skill} para tareas complejas.",
    "Fine-tuning y evaluación de modelos usando {skill} en pipelines reproducibles.",
    "Exposición de {skill} a través de una API REST documentada con OpenAPI.",
]

_REPO_PATTERNS: list[str] = [
    "{u}/rag-agent-project",
    "{u}/llm-toolkit",
    "{u}/langchain-demos",
    "{u}/ml-experiments",
    "{u}/vector-search",
    "{u}/chat-assistant",
    "{u}/llm-benchmarks",
    "{u}/semantic-search-api",
]

_PATH_PATTERNS: list[str] = [
    "src/rag_pipeline.py",
    "app/chains.py",
    "core/retrieval.py",
    "utils/embeddings.py",
    "notebooks/demo.ipynb",
    "backend/api.py",
    "scripts/index_data.py",
    "tests/test_pipeline.py",
]

_ANSWER_TEMPLATES: list[str] = [
    (
        "Basándome en las evidencias técnicas analizadas, **@{top}** es el perfil "
        "más adecuado para esta tarea (score: **{score:.3f}**). "
        "Sus repositorios muestran experiencia demostrable en {skills_str}, "
        "lo que lo posiciona como el candidato técnicamente más cualificado. "
        "Como segunda opción, **@{second}** presenta evidencias relevantes en "
        "{second_skills}, siendo una alternativa sólida."
    ),
    (
        "El análisis de las evidencias de GitHub muestra que **@{top}** tiene el "
        "perfil técnico más alineado con los requisitos planteados "
        "(score: **{score:.3f}**). Sus contribuciones demuestran dominio de "
        "{skills_str}. En segundo lugar, **@{second}** aporta experiencia "
        "contrastada en {second_skills}."
    ),
    (
        "Tras evaluar {n_profiles} perfiles técnicos mediante búsqueda semántica "
        "en ChromaDB, **@{top}** lidera el ranking con un score de **{score:.3f}**. "
        "Las evidencias de sus repositorios confirman conocimiento práctico de "
        "{skills_str}. **@{second}** ocupa la segunda posición gracias a su "
        "experiencia en {second_skills}."
    ),
]


# Utilidades

def _hold_step(started_at: float) -> None:
    """Mantiene el paso actual visible al menos _REAL_STEP_MIN_SECONDS segundos."""
    remaining = _REAL_STEP_MIN_SECONDS - (time.monotonic() - started_at)
    if remaining > 0:
        time.sleep(remaining)


# Interfaz pública

def ask_question(
    question: str,
    profiles: list[dict],
) -> Generator[tuple[str, bool, Optional[dict]], None, None]:
    """
    Ejecuta el pipeline RAG para una consulta en lenguaje natural.

    Args:
        question : Pregunta del usuario en lenguaje natural.
        profiles : Lista de perfiles cargados en st.session_state.profiles.

    Yields:
        (step_label, is_final, result)
          - step_label : Paso del pipeline en curso.
          - is_final   : True únicamente en el yield final.
          - result     : Dict con answer, ranking y evidences al terminar.
    """
    if USE_MOCK:
        yield from _mock_ask_question(question, profiles)
    else:
        yield from _real_ask_question(question, profiles)


# Implementación mock

def _mock_ask_question(
    question: str,
    profiles: list[dict],
) -> Generator[tuple[str, bool, Optional[dict]], None, None]:
    """Pipeline RAG simulado con scoring semántico aproximado."""
    rng = random.Random(hash(question) % 2**32)

    for step, base_delay in zip(RAG_STEPS, _STEP_DELAYS):
        time.sleep(rng.uniform(base_delay * 0.7, base_delay * 1.4))
        yield (step, False, None)

    if not profiles:
        yield ("Sin perfiles disponibles para rankear.", True, None)
        return

    # Scoring: ponderar por cuántas skills del perfil aparecen en la pregunta
    q_lower = question.lower()

    def relevance(profile: dict) -> float:
        skills = [s.lower() for s in profile.get("skills_detected", [])]
        hits   = sum(1 for s in skills if any(w in s for w in q_lower.split()))
        base   = rng.uniform(0.42, 0.62)
        return min(0.99, base + hits * 0.07)

    scored = sorted(profiles, key=relevance, reverse=True)

    # Construir ranking
    ranking: list[dict] = []
    for i, p in enumerate(scored[:5]):
        base_score = relevance(p) - i * rng.uniform(0.04, 0.09)
        score = round(max(0.28, min(0.99, base_score)), 3)
        skills = p.get("skills_detected", [])
        n_matched = rng.randint(min(2, len(skills)), min(4, len(skills))) if skills else 0
        matched = rng.sample(skills, k=n_matched) if skills else []
        ranking.append({
            "username":      p["username"],
            "score":         score,
            "matched_skills": matched,
        })

    # Construir evidencias (top 3 candidatos, 2-3 evidencias c/u)
    evidences: list[dict] = []
    for p in scored[:3]:
        skills = p.get("skills_detected", [])
        if not skills:
            continue
        for _ in range(rng.randint(2, 3)):
            skill    = rng.choice(skills)
            ev_score = round(rng.uniform(0.58, 0.97), 3)
            evidences.append({
                "username": p["username"],
                "repo":     rng.choice(_REPO_PATTERNS).format(u=p["username"]),
                "path":     rng.choice(_PATH_PATTERNS),
                "skill":    skill,
                "fragment": rng.choice(_FRAGMENT_TEMPLATES).format(skill=skill),
                "score":    ev_score,
            })

    evidences.sort(key=lambda e: e["score"], reverse=True)

    # Generar respuesta narrativa
    top    = ranking[0]
    second = ranking[1] if len(ranking) > 1 else ranking[0]
    tmpl   = rng.choice(_ANSWER_TEMPLATES)
    answer = tmpl.format(
        top           = top["username"],
        score         = top["score"],
        skills_str    = ", ".join(top["matched_skills"][:3]) or "diversas tecnologías",
        second        = second["username"],
        second_skills = ", ".join(second["matched_skills"][:2]) or "otras áreas",
        n_profiles    = len(profiles),
    )

    result = {
        "answer":    answer,
        "ranking":   ranking,
        "evidences": evidences[:8],
    }
    yield ("Respuesta generada correctamente", True, result)


# Implementación real

def _real_ask_question(
    question: str,
    profiles: list[dict],
) -> Generator[tuple[str, bool, Optional[dict]], None, None]:
    """
    Pipeline RAG real: ChromaDB → ranking → Ollama.

    Usa los módulos del backend directamente:
      - src.rag.retrieval.retrieve_blocks()
      - src.rag.ranking.rank_candidates()
      - src.rag.generation.generate_response()

    Cada paso se emite junto a su fase real de trabajo y se mantiene visible un
    mínimo de tiempo (_REAL_STEP_MIN_SECONDS) para que el proceso sea perceptible
    en la UI; el retrieval y el ranking son demasiado rápidos por sí solos.
    """
    try:
        from src.rag.retrieval import retrieve_blocks    # noqa: PLC0415
        from src.rag.ranking import rank_candidates      # noqa: PLC0415
        from src.rag.generation import generate_response  # noqa: PLC0415

        # Paso 1: embedding de la consulta
        t = time.monotonic()
        yield (RAG_STEPS[0], False, None)
        _hold_step(t)

        # Paso 2: recuperación de bloques en ChromaDB (trabajo real)
        t = time.monotonic()
        yield (RAG_STEPS[1], False, None)
        blocks = retrieve_blocks(question, n_results=30)
        _hold_step(t)

        # Paso 3: similitud semántica (calculada dentro del retrieval)
        t = time.monotonic()
        yield (RAG_STEPS[2], False, None)
        _hold_step(t)

        # Paso 4: agrupación de evidencias por usuario
        t = time.monotonic()
        yield (RAG_STEPS[3], False, None)
        _hold_step(t)

        # Paso 5: ranking de candidatos (trabajo real)
        t = time.monotonic()
        yield (RAG_STEPS[4], False, None)
        candidates = rank_candidates(blocks, top_k=5)
        _hold_step(t)

        # Paso 6: generación de la respuesta con el LLM (ya es lento de por sí)
        yield (RAG_STEPS[5], False, None)
        answer = generate_response(question, candidates)

        # Adaptar formato de candidatos al contrato del frontend
        ranking = [
            {
                "username":      c["username"],
                "score":         c["final_score"],
                "matched_skills": [s["skill"] for s in c["matched_skills"][:5]],
            }
            for c in candidates
        ]

        # Adaptar evidencias al contrato del frontend
        evidences: list[dict] = []
        for c in candidates[:3]:
            for skill_block in c["matched_skills"][:3]:
                evidences.append({
                    "username":    c["username"],
                    "repo":        skill_block.get("repo", ""),
                    "path":        skill_block.get("path", ""),
                    "artifact_type": skill_block.get("artifact_type", ""),
                    "skill":       skill_block.get("skill", ""),
                    "explanation": skill_block.get("explanation", ""),
                    "fragment":    skill_block.get("evidence_fragment", ""),
                    "score":       skill_block.get("combined_score", 0.0),
                })

        evidences.sort(key=lambda e: e["score"], reverse=True)

        result = {
            "answer":    answer,
            "ranking":   ranking,
            "evidences": evidences[:8],
        }
        yield (RAG_STEPS[-1], False, None)
        yield ("Respuesta generada correctamente", True, result)

    except Exception as exc:
        yield (f"Error en el pipeline RAG: {exc}", True, None)
