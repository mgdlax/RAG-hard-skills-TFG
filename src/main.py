"""
Pipeline principal de ingesta, procesamiento e indexación.

Ejecuta las 4 fases para un usuario de GitHub:
  1. Ingesta      — descarga ficheros, commits, issues y PRs
  2. Procesamiento — métricas estructurales + detección de skills con LLM
  3. Perfil        — agrega los bloques en un perfil técnico por skill
  4. Indexación    — vectoriza con minilm-code-search-512 y almacena en ChromaDB

Usuarios objetivo del TFG (empresa de IA/LLM):
  - hwchase17    : creador de LangChain y LangGraph
  - jerryjliu    : creador de LlamaIndex
  - pamelafox    : Microsoft, Azure OpenAI, demos Python/IA
  - NirDiamant   : tutoriales RAG y agentes con LangChain
  - Shubhamsaboo : apps de agentes IA con Streamlit y Groq

  Nota: estos 5 usuarios son los que usa el módulo de evaluación (src/evaluation).
  Para tests rápidos o usuarios alternativos, edita TARGET_USERS.

Uso:
  Cambia la variable 'username' o pasa el nombre como argumento:
    python -m src.main
    python -m src.main hwchase17
    python -m src.main --all   (indexa todos los TARGET_USERS)
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

from src.config import DEFAULT_LLM_MODEL
from src.utils.logger import setup_pipeline_logger, get_logger
from src.ingestion.github_client import get_github_client
from src.ingestion.extract_user import extract_user_profile_evidences
from src.processing.code_filter import process_raw_user_file
from src.profiling.build_profile import build_user_profile
from src.vectorization.indexer import index_user, get_index_stats


# Usuarios objetivo del sistema de recomendación.
# Para indexar todos: python -m src.main --all
TARGET_USERS = [
    "chitralputhran",
    "ranguy9304",
    "naurjhanvi",
    "yldzburhan",
    "HalemoGPA",
]


def _save_evidences(records: list, output_path: Path) -> None:
    """Serializa una lista de objetos GitHubEvidence (Pydantic) a JSONL.
    No confundir con save_jsonl de code_filter.py, que trabaja con dicts planos."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")


def run_pipeline(username: str) -> None:
    """
    Ejecuta el pipeline completo de ingesta a indexación para un usuario.
    Cada fase está cronometrada y rodeada de manejo de errores para que
    un fallo en una fase no corrompa los datos de otras ejecuciones.
    """
    max_repos            = 6
    max_files_per_repo   = 50
    max_commits_per_repo = 20
    max_issues_per_repo  = 10
    max_prs_per_repo     = 10
    limit_files          = 30   # ficheros .py/.ipynb a procesar con LLM
    model                = DEFAULT_LLM_MODEL

    setup_pipeline_logger(username=username)
    log = get_logger(__name__)
    t_total = time.time()

    log.info("=" * 55)
    log.info(f"PIPELINE — @{username}")
    log.info("=" * 55)
    log.info(
        f"repos={max_repos}  files/repo={max_files_per_repo}  "
        f"commits/repo={max_commits_per_repo}  model={model}"
    )

    # -------------------------------------------------------------------------
    # FASE 1: Ingesta desde GitHub
    # -------------------------------------------------------------------------
    log.info("-" * 55)
    log.info("FASE 1 — INGESTA")
    log.info("-" * 55)
    t0 = time.time()

    try:
        github = get_github_client()
        user = github.get_user(username)
        log.info(f"Conectado: {user.login} ({user.name})")
    except Exception as e:
        log.error(f"No se pudo conectar con GitHub: {e}", exc_info=True)
        return

    try:
        evidences = extract_user_profile_evidences(
            user=user,
            max_repos=max_repos,
            max_files_per_repo=max_files_per_repo,
            max_commits_per_repo=max_commits_per_repo,
            max_issues_per_repo=max_issues_per_repo,
            max_prs_per_repo=max_prs_per_repo,
        )
    except Exception as e:
        log.error(f"Error durante la extraccion: {e}", exc_info=True)
        return

    raw_path = Path("data/raw") / f"{username}.jsonl"
    _save_evidences(evidences, raw_path)

    counter = Counter(e.artifact_type for e in evidences)
    log.info(f"Evidencias: {len(evidences)}  {dict(counter)}")
    log.info(f"FASE 1 OK en {time.time() - t0:.1f}s")

    # -------------------------------------------------------------------------
    # FASE 2: Métricas estructurales + detección de skills con LLM
    # Para cada evidencia: score_evidence() calcula 4 métricas (recency,
    # authorship, artifact_weight, content_richness) y el LLM detecta las
    # skills presentes y genera una explicación. Se emite un bloque por skill.
    # -------------------------------------------------------------------------
    log.info("-" * 55)
    log.info("FASE 2 — PROCESAMIENTO (metricas + LLM)")
    log.info("-" * 55)
    t0 = time.time()

    processed_path = Path("data/procesado") / f"{username}_processed.jsonl"

    try:
        processed = process_raw_user_file(
            input_path=raw_path,
            output_path=processed_path,
            model=model,
            limit_files=limit_files,
            enrich_with_llm=True,
        )
    except Exception as e:
        log.error(f"Error durante el procesamiento: {e}", exc_info=True)
        return

    skills_found = {r.get("skill") for r in processed if r.get("skill")}
    log.info(f"FASE 2 OK en {time.time() - t0:.1f}s")
    log.info(f"  Skills detectadas ({len(skills_found)}): {sorted(skills_found)}")
    
    # -------------------------------------------------------------------------
    # FASE 3: Perfil tecnico agregado
    # -------------------------------------------------------------------------
    log.info("-" * 55)
    log.info("FASE 3 — PERFIL TECNICO")
    log.info("-" * 55)
    t0 = time.time()

    profile_path = Path("data/perfiles") / f"{username}_profile.json"

    try:
        build_user_profile(
            username=username,
            processed_input_path=processed_path,
            output_path=profile_path,
        )
    except Exception as e:
        log.error(f"Error generando el perfil: {e}", exc_info=True)
        return

    log.info(f"FASE 3 OK en {time.time() - t0:.1f}s")

    # -------------------------------------------------------------------------
    # FASE 4: Vectorizacion e indexacion en ChromaDB
    # -------------------------------------------------------------------------
    log.info("-" * 55)
    log.info("FASE 4 — INDEXACION EN CHROMADB")
    log.info("-" * 55)
    t0 = time.time()

    try:
        indexed = index_user(username=username, processed_path=processed_path)
    except Exception as e:
        log.error(f"Error durante la indexacion: {e}", exc_info=True)
        return

    stats = get_index_stats()
    log.info(f"FASE 4 OK en {time.time() - t0:.1f}s  ({indexed} bloques)")
    log.info(f"  Total en ChromaDB: {stats['total_blocks']} bloques")
    for u, count in stats["blocks_per_user"].items():
        n_skills = stats["unique_skills_per_user"].get(u, 0)
        log.info(f"    @{u}: {count} bloques, {n_skills} skills")

    # Resumen
    total = time.time() - t_total
    log.info("=" * 55)
    log.info(f"COMPLETADO en {total:.1f}s")
    log.info(f"  Evidencias: {len(evidences)}  |  Skills: {len(skills_found)}  |  Bloques: {indexed}")
    log.info(f"  Siguiente: python -m src.rag.query")
    log.info("=" * 55)


def main() -> None:
    # Se puede pasar el username como argumento: python -m src.main hwchase17
    # Si no se pasa ninguno, se usa el primero de la lista objetivo.
    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        username = TARGET_USERS[0]  # cambiar aquí para indexar otro usuario

    if username == "--all":
        for u in TARGET_USERS:
            run_pipeline(u)
    else:
        run_pipeline(username)


if __name__ == "__main__":
    main()
