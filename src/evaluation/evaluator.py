"""
Evaluador del sistema RAG de recomendacion tecnica.

Ejecuta el pipeline RAG completo (retrieval + ranking) sobre el conjunto
de consultas del ground truth y calcula las metricas estandar de IR.

Uso:
  python -m src.evaluation.evaluator

  # Solo las consultas donde todos los usuarios relevantes estan indexados:
  python -m src.evaluation.evaluator --strict

Salida:
  - Tabla de resultados por consulta en consola
  - Resumen de metricas globales
  - Informe JSON en data/evaluation/report_<timestamp>.json

Metricas calculadas:
  Precision@1, Precision@3, Recall@3, MRR, NDCG@3

Usuarios objetivo del TFG:
  chitralputhran — RAG + LangGraph + Streamlit, apps de IA generativa
  ranguy9304     — RAG terminal con LangGraph, sistemas distribuidos
  naurjhanvi     — Edge AI + RAG + LSTM, despliegue con Docker/Kubernetes/FastAPI
  yldzburhan     — ML/data science, NLP, sistemas de recomendacion, LangChain
  HalemoGPA      — Deep learning, PyTorch, vision por computador, NLP

Para la memoria del TFG:
  Los valores de estas metricas constituyen el capitulo de evaluacion
  experimental. Un MRR > 0.7 y P@1 > 0.6 son resultados solidos para
  un sistema de recomendacion basado en evidencia de repositorios publicos.
"""

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import setup_pipeline_logger, get_logger
from src.rag.retrieval import retrieve_blocks
from src.rag.ranking import rank_candidates
from src.vectorization.indexer import get_index_stats
from src.evaluation.ground_truth import get_all_queries, get_required_users
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    mean_metric,
)

log = get_logger(__name__)

# Parametros del pipeline RAG usados durante la evaluacion
N_BLOCKS = 30
TOP_K_RANKING = 10   # Recuperar mas candidatos para calcular metricas hasta k=5


# ---------------------------------------------------------------------------
#  Comprobacion de usuarios disponibles
# ---------------------------------------------------------------------------

def _check_available_users(strict: bool = False) -> Tuple[List[str], List[str]]:
    """
    Comprueba que usuarios del ground truth estan indexados en ChromaDB.

    Returns:
        (available, missing): listas de usuarios presentes y ausentes.
    """
    stats = get_index_stats()
    indexed = set(stats["blocks_per_user"].keys())
    required = get_required_users()

    available = [u for u in required if u in indexed]
    missing = [u for u in required if u not in indexed]

    if missing:
        log.warning(
            f"Usuarios del ground truth NO indexados: {missing}\n"
            f"  Para indexarlos: edita 'username' en src/main.py y ejecuta "
            f"'python -m src.main' por cada uno.\n"
            f"  Las consultas que los requieren seran omitidas."
            if strict else
            f"Usuarios del ground truth NO indexados: {missing}\n"
            f"  La evaluacion continuara con los disponibles: {available}\n"
            f"  Algunas metricas (Recall) seran optimistas al no poder "
            f"penalizar la ausencia de estos usuarios."
        )

    return available, missing


# ---------------------------------------------------------------------------
#  Evaluacion de una sola consulta
# ---------------------------------------------------------------------------

def evaluate_query(
    query_data: Dict[str, Any],
    available_users: List[str],
    n_blocks: int = N_BLOCKS,
    top_k: int = TOP_K_RANKING,
) -> Optional[Dict[str, Any]]:
    """
    Ejecuta el pipeline RAG para una consulta y calcula sus metricas.

    Si ningun usuario relevante de la consulta esta disponible en ChromaDB,
    la consulta se omite (devuelve None).

    Args:
        query_data:      Entrada del ground truth.
        available_users: Usuarios actualmente indexados.
        n_blocks:        Bloques a recuperar de ChromaDB.
        top_k:           Candidatos a rankear.

    Returns:
        Dict con metricas y detalles, o None si la consulta se omite.
    """
    qid = query_data["id"]
    query = query_data["query"]
    relevant_set = set(query_data["relevant_users"])
    grades = query_data.get("grades", {})

    # Filtrar relevant_set a usuarios disponibles para evitar penalizar ausencias
    relevant_available = relevant_set & set(available_users)
    grades_available = {u: g for u, g in grades.items() if u in available_users}

    if not relevant_available:
        log.debug(
            f"  [{qid}] Omitida — ningun usuario relevante indexado "
            f"(relevantes: {relevant_set}, disponibles: {available_users})"
        )
        return None

    # -- Pipeline RAG --------------------------------------------------------
    t0 = time.time()
    blocks = retrieve_blocks(query, n_results=n_blocks)
    candidates = rank_candidates(blocks, top_k=top_k)
    elapsed = time.time() - t0

    retrieved_users = [c["username"] for c in candidates]

    # -- Metricas ------------------------------------------------------------
    p1 = precision_at_k(retrieved_users, relevant_available, k=1)
    p3 = precision_at_k(retrieved_users, relevant_available, k=3)
    r3 = recall_at_k(retrieved_users, relevant_available, k=3)
    rr = reciprocal_rank(retrieved_users, relevant_available)
    ndcg3 = ndcg_at_k(retrieved_users, grades_available, k=3)

    return {
        "query_id": qid,
        "category": query_data.get("category", ""),
        "query": query,
        "retrieved_top3": retrieved_users[:3],
        "retrieved_scores": [round(c["final_score"], 4) for c in candidates[:3]],
        "relevant_users": sorted(relevant_available),
        "missing_relevant": sorted(relevant_set - set(available_users)),
        "precision_at_1": round(p1, 4),
        "precision_at_3": round(p3, 4),
        "recall_at_3": round(r3, 4),
        "mrr": round(rr, 4),
        "ndcg_at_3": round(ndcg3, 4),
        "latency_s": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
#  Evaluacion completa
# ---------------------------------------------------------------------------

def run_evaluation(strict: bool = False) -> Dict[str, Any]:
    """
    Ejecuta la evaluacion completa sobre todas las consultas del ground truth.

    Args:
        strict: Si True, omite consultas donde ALGUN usuario relevante
                no esta indexado. Si False (default), ajusta la evaluacion
                a los usuarios disponibles.

    Returns:
        Diccionario completo con resultados por consulta y metricas globales.
    """
    log.info("=" * 60)
    log.info("EVALUACION DEL SISTEMA RAG")
    log.info("=" * 60)

    # -- Comprobacion de usuarios disponibles --------------------------------
    available, missing = _check_available_users(strict=strict)

    log.info(f"Usuarios indexados en ChromaDB: {available}")
    if missing:
        log.info(f"Usuarios ausentes (ground truth incompleto): {missing}")

    # -- Cargar consultas ----------------------------------------------------
    all_queries = get_all_queries()
    log.info(f"Consultas en el ground truth: {len(all_queries)}")
    log.info("-" * 60)

    # -- Evaluar consulta a consulta -----------------------------------------
    query_results: List[Dict[str, Any]] = []
    skipped = 0

    for q in all_queries:
        # En modo strict: omitir si alguno de los relevantes no esta disponible
        if strict:
            relevant_set = set(q["relevant_users"])
            if not relevant_set.issubset(set(available)):
                log.debug(f"  [{q['id']}] Omitida en modo strict")
                skipped += 1
                continue

        result = evaluate_query(q, available_users=available)

        if result is None:
            skipped += 1
            continue

        query_results.append(result)

        # Log por consulta
        top3_str = ", ".join(
            f"@{u}({s:.3f})"
            for u, s in zip(result["retrieved_top3"], result["retrieved_scores"])
        )
        relevant_str = ", ".join(f"@{u}" for u in result["relevant_users"])
        log.info(
            f"[{result['query_id']}] {result['query'][:55]:<55}\n"
            f"         Esperado: {relevant_str}\n"
            f"         Top-3:    {top3_str}\n"
            f"         P@1={result['precision_at_1']:.3f}  "
            f"P@3={result['precision_at_3']:.3f}  "
            f"R@3={result['recall_at_3']:.3f}  "
            f"MRR={result['mrr']:.3f}  "
            f"NDCG@3={result['ndcg_at_3']:.3f}  "
            f"({result['latency_s']:.2f}s)"
        )

    log.info("-" * 60)
    log.info(f"Consultas evaluadas: {len(query_results)} / {len(all_queries)}  "
             f"(omitidas: {skipped})")

    # -- Metricas globales ---------------------------------------------------
    global_metrics = _compute_global_metrics(query_results)
    _log_global_metrics(global_metrics, query_results)

    # -- Metricas por categoria ----------------------------------------------
    category_metrics = _compute_category_metrics(query_results)
    _log_category_metrics(category_metrics)

    # -- Construir informe ---------------------------------------------------
    report = {
        "timestamp": datetime.now().isoformat(),
        "indexed_users": available,
        "missing_users": missing,
        "strict_mode": strict,
        "total_queries": len(all_queries),
        "evaluated_queries": len(query_results),
        "skipped_queries": skipped,
        "global_metrics": global_metrics,
        "per_category_metrics": category_metrics,
        "per_query_results": query_results,
    }

    # Guardar informe JSON
    report_path = _save_report(report)
    log.info(f"Informe guardado en: {report_path}")

    return report


# ---------------------------------------------------------------------------
#  Agregacion de metricas
# ---------------------------------------------------------------------------

def _compute_global_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calcula la media de cada metrica sobre todas las consultas evaluadas."""
    if not results:
        return {}

    return {
        "precision_at_1": round(mean_metric([r["precision_at_1"] for r in results]), 4),
        "precision_at_3": round(mean_metric([r["precision_at_3"] for r in results]), 4),
        "recall_at_3": round(mean_metric([r["recall_at_3"] for r in results]), 4),
        "mrr": round(mean_metric([r["mrr"] for r in results]), 4),
        "ndcg_at_3": round(mean_metric([r["ndcg_at_3"] for r in results]), 4),
        "mean_latency_s": round(mean_metric([r["latency_s"] for r in results]), 3),
        "n_queries": len(results),
    }


def _compute_category_metrics(
    results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Agrupa y promedia metricas por categoria de consulta."""
    by_category: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    return {
        cat: _compute_global_metrics(cat_results)
        for cat, cat_results in sorted(by_category.items())
    }


# ---------------------------------------------------------------------------
#  Logging de resultados
# ---------------------------------------------------------------------------

def _log_global_metrics(metrics: Dict[str, float], results: List[Dict]) -> None:
    """Imprime el resumen de metricas globales."""
    if not metrics:
        log.warning("Sin resultados para calcular metricas globales")
        return

    log.info("=" * 60)
    log.info("METRICAS GLOBALES (media sobre todas las consultas evaluadas)")
    log.info("=" * 60)
    log.info(f"  Precision@1 : {metrics['precision_at_1']:.4f}  "
             f"(de cada 1 resultado, fraccion correctos)")
    log.info(f"  Precision@3 : {metrics['precision_at_3']:.4f}  "
             f"(de los 3 primeros, fraccion correctos)")
    log.info(f"  Recall@3    : {metrics['recall_at_3']:.4f}  "
             f"(de todos los relevantes, fraccion encontrados en top-3)")
    log.info(f"  MRR         : {metrics['mrr']:.4f}  "
             f"(rango inverso medio del primer resultado correcto)")
    log.info(f"  NDCG@3      : {metrics['ndcg_at_3']:.4f}  "
             f"(ganancia acumulada normalizada, considera orden y grado)")
    log.info(f"  Latencia    : {metrics['mean_latency_s']:.3f}s por consulta")
    log.info(f"  Consultas   : {metrics['n_queries']}")
    log.info("=" * 60)

    # Interpretacion cualitativa
    mrr = metrics["mrr"]
    p1 = metrics["precision_at_1"]
    if mrr >= 0.8 and p1 >= 0.7:
        log.info("  Interpretacion: EXCELENTE — el sistema es muy preciso")
    elif mrr >= 0.6 and p1 >= 0.5:
        log.info("  Interpretacion: BUENO — resultados solidos para TFG")
    elif mrr >= 0.4:
        log.info("  Interpretacion: ACEPTABLE — margen de mejora con mas usuarios")
    else:
        log.info("  Interpretacion: MEJORABLE — revisar ground truth e indexacion")


def _log_category_metrics(category_metrics: Dict[str, Dict[str, float]]) -> None:
    """Imprime metricas desglosadas por categoria."""
    if not category_metrics:
        return

    log.info("METRICAS POR CATEGORIA:")
    header = f"  {'Categoria':<20} {'P@1':>6} {'P@3':>6} {'R@3':>6} {'MRR':>6} {'NDCG@3':>8} {'N':>4}"
    log.info(header)
    log.info("  " + "-" * 58)

    for cat, m in category_metrics.items():
        log.info(
            f"  {cat:<20} "
            f"{m.get('precision_at_1', 0):>6.3f} "
            f"{m.get('precision_at_3', 0):>6.3f} "
            f"{m.get('recall_at_3', 0):>6.3f} "
            f"{m.get('mrr', 0):>6.3f} "
            f"{m.get('ndcg_at_3', 0):>8.3f} "
            f"{m.get('n_queries', 0):>4}"
        )


# ---------------------------------------------------------------------------
#  Guardado del informe
# ---------------------------------------------------------------------------

def _save_report(report: Dict[str, Any]) -> Path:
    """Guarda el informe completo en JSON con timestamp en el nombre."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"report_{ts}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return output_path


# ---------------------------------------------------------------------------
#  Punto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    setup_pipeline_logger(username="evaluation")

    strict = "--strict" in sys.argv
    if strict:
        log.info("Modo strict: solo se evaluan consultas con todos sus usuarios indexados")

    report = run_evaluation(strict=strict)

    # Salida final compacta para el terminal
    gm = report.get("global_metrics", {})
    if gm:
        print("\n" + "=" * 50)
        print("RESUMEN DE EVALUACION")
        print("=" * 50)
        print(f"  Consultas evaluadas : {gm.get('n_queries', 0)}")
        print(f"  Precision@1         : {gm.get('precision_at_1', 0):.4f}")
        print(f"  Precision@3         : {gm.get('precision_at_3', 0):.4f}")
        print(f"  Recall@3            : {gm.get('recall_at_3', 0):.4f}")
        print(f"  MRR                 : {gm.get('mrr', 0):.4f}")
        print(f"  NDCG@3              : {gm.get('ndcg_at_3', 0):.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()
