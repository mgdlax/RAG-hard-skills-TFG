"""
Punto de entrada del sistema RAG de recomendacion tecnica.

Uso:
  # Consulta unica (ideal para demos y scripts):
  python -m src.rag.query "quien recomiendas para construir sistemas RAG?"

  # Modo interactivo (ideal para la defensa del TFG):
  python -m src.rag.query

Pipeline de una consulta:
  1. Retrieval  -> embed query + buscar en ChromaDB por similitud coseno
  2. Ranking    -> agrupar bloques por usuario + calcular final_score
  3. Generation -> formatear contexto + llamar a Ollama para respuesta narrativa

El sistema funciona aunque Ollama este apagado (respuesta estructurada fallback).
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from src.config import DEFAULT_LLM_MODEL
from src.utils.logger import setup_pipeline_logger, get_logger
from src.rag.retrieval import retrieve_blocks
from src.rag.ranking import rank_candidates
from src.rag.generation import generate_response
from src.rag.answer_evaluation import evaluate_answer_against_profiles
from src.vectorization.indexer import get_index_stats

DEFAULT_MODEL = DEFAULT_LLM_MODEL
N_BLOCKS = 30       # Bloques a recuperar de ChromaDB
TOP_K = 5           # Candidatos a rankear
PROFILES_DIR = Path("data/perfiles")  # Perfiles técnicos generados por src.profiling


def run_query(
    query: str,
    model: str = DEFAULT_MODEL,
    n_blocks: int = N_BLOCKS,
    top_k: int = TOP_K,
    verbose: bool = True,
    return_diagnostics: bool = False,
) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """
    Ejecuta el pipeline RAG completo para una consulta.

    Args:
        query:              Consulta en lenguaje natural.
        model:              Modelo Ollama para la fase de generacion.
        n_blocks:           Cuantos bloques recuperar de ChromaDB.
        top_k:              Maximo de candidatos a rankear.
        verbose:            Si True, loguea el detalle de cada fase.
        return_diagnostics: Si True, devuelve (response, diagnostics) en vez de solo response.

    Returns:
        Si return_diagnostics=False: respuesta RAG como string.
        Si return_diagnostics=True: tupla (respuesta, diccionario de diagnóstico).
    """
    log = get_logger(__name__)
    t_total = time.time()

    log.info("=" * 60)
    log.info(f"CONSULTA: '{query}'")
    log.info("=" * 60)

    # -- Fase 1: Retrieval ---------------------------------------------------
    log.info("- Fase 1: Retrieval semantico")
    t0 = time.time()
    blocks = retrieve_blocks(query, n_results=n_blocks)
    log.info(f"  -> {len(blocks)} bloques recuperados ({time.time()-t0:.2f}s)")

    if not blocks:
        empty_response = (
            "AVISO: ChromaDB esta vacio.\n"
            "Indexa usuarios primero:\n"
            "  python -m src.main"
        )
        if return_diagnostics:
            return empty_response, {"query": query, "n_blocks_retrieved": 0, "n_candidates": 0}
        return empty_response

    # -- Fase 2: Ranking -----------------------------------------------------
    log.info("- Fase 2: Ranking de candidatos")
    t0 = time.time()
    candidates = rank_candidates(blocks, top_k=top_k)
    log.info(f"  -> {len(candidates)} candidatos rankeados ({time.time()-t0:.2f}s)")

    if verbose:
        for c in candidates:
            top3 = [s["skill"] for s in c["matched_skills"][:3]]
            log.info(
                f"    @{c['username']:<22} score={c['final_score']:.3f}  "
                f"skills={top3}"
            )

    # -- Fase 3: Generacion --------------------------------------------------
    log.info("- Fase 3: Generacion de respuesta")
    t0 = time.time()
    response = generate_response(query, candidates, model=model)
    log.info(f"  -> Respuesta generada ({time.time()-t0:.2f}s)")

    # -- Fase 4: Evaluacion contra perfiles ----------------------------------
    log.info("- Fase 4: Evaluacion frente a perfiles tecnico")
    evaluation = evaluate_answer_against_profiles(response, candidates, PROFILES_DIR)

    # Resumen en log
    top_candidate = candidates[0] if candidates else {}
    consistency = evaluation.get("profile_consistency_score")
    consistency_str = f"{consistency:.2f}" if consistency is not None else "n/a"
    log.info(
        f"  -> Candidato top: @{top_candidate.get('username', '?')} | "
        f"consistencia_perfiles={consistency_str}"
    )

    latency = round(time.time() - t_total, 2)
    log.info(f"Pipeline completado en {latency}s")

    if not return_diagnostics:
        return response

    diagnostics: Dict[str, Any] = {
        "query": query,
        "n_blocks_retrieved": len(blocks),
        "n_candidates": len(candidates),
        "candidates": candidates,
        "profile_evaluation": evaluation,
        "latency_seconds": latency,
    }
    return response, diagnostics


def interactive_mode(model: str = DEFAULT_MODEL) -> None:
    """
    Modo interactivo: permite lanzar multiples consultas sin reiniciar.
    Ideal para demostrar el sistema en la defensa del TFG.
    """
    try:
        stats = get_index_stats()
    except Exception:
        stats = {"total_blocks": 0, "blocks_per_user": {}}

    total = stats["total_blocks"]
    users = list(stats["blocks_per_user"].keys())
    skills_info = stats.get("unique_skills_per_user", {})

    sep = "=" * 60
    print()
    print(sep)
    print("  SISTEMA RAG - RECOMENDACION TECNICA DE DESARROLLADORES")
    print(sep)
    print(f"  ChromaDB: {total} bloques indexados")
    print("  Usuarios disponibles:")
    for u in users:
        n_skills = skills_info.get(u, "?")
        n_blks = stats["blocks_per_user"].get(u, 0)
        print(f"    @{u} — {n_blks} bloques, {n_skills} skills unicas")
    print()
    print("  Ejemplos de consultas:")
    print("    Quien recomiendas para construir sistemas RAG?")
    print("    Quien tiene mas experiencia con LangChain y LangGraph?")
    print("    Quien sabe implementar agentes LLM con herramientas?")
    print("  Escribe 'salir' para terminar")
    print(sep)

    while True:
        print()
        try:
            query = input("Consulta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Sistema RAG detenido]")
            break

        if not query:
            continue
        if query.lower() in ("salir", "exit", "quit", "q"):
            print("[Sistema RAG detenido]")
            break

        print()
        response = run_query(query, model=model)
        print("\n" + "-" * 60)
        print(response)
        print("-" * 60)


def main(argv: Optional[list] = None) -> None:
    """
    Punto de entrada principal.

    Sin argumentos -> modo interactivo.
    Con argumentos -> consulta unica, imprime respuesta y sale.
    """
    setup_pipeline_logger(username="query")

    args = argv if argv is not None else sys.argv[1:]

    if args:
        query = " ".join(args)
        response = run_query(query)
        print("\n" + "-" * 60)
        print(response)
        print("-" * 60)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
