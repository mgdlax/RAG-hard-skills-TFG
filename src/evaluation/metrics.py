"""
Metricas de evaluacion para el sistema RAG.

Implementa las metricas estandar de recuperacion de informacion:
  - Precision@k  : fraccion de candidatos relevantes en los top-k resultados
  - Recall@k     : fraccion de relevantes encontrados entre todos los relevantes
  - MRR          : reciproco del rango del primer resultado correcto
  - NDCG@k       : ganancia acumulada descontada, considera orden y relevancia graduada

Todas las funciones son PURAS (sin estado, sin efectos laterales) para
facilitar su testeo y reutilizacion en diferentes contextos de evaluacion.

Referencias academicas:
  - Manning et al. "Introduction to Information Retrieval", cap. 8
  - Jarvelin & Kekalainen (2002). "Cumulated gain-based evaluation of IR"
"""

import math
from typing import Dict, List, Set


# ---------------------------------------------------------------------------
#  Precision@k
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Fraccion de los top-k candidatos recuperados que son relevantes.

    P@k = |{recuperados en top-k} interseccion {relevantes}| / k

    Args:
        retrieved: Lista de usernames ordenada por ranking del sistema.
        relevant:  Conjunto de usernames que son respuesta correcta.
        k:         Numero de resultados a considerar.

    Returns:
        Float en [0, 1]. 1.0 significa que todos los top-k son relevantes.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for u in top_k if u in relevant)
    return hits / k


# ---------------------------------------------------------------------------
#  Recall@k
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Fraccion de todos los relevantes que aparecen en los top-k.

    R@k = |{recuperados en top-k} interseccion {relevantes}| / |relevantes|

    Args:
        retrieved: Lista de usernames ordenada por ranking del sistema.
        relevant:  Conjunto de usernames que son respuesta correcta.
        k:         Numero de resultados a considerar.

    Returns:
        Float en [0, 1]. 1.0 significa que se encontraron todos los relevantes.
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for u in top_k if u in relevant)
    return hits / len(relevant)


# ---------------------------------------------------------------------------
#  MRR (Mean Reciprocal Rank)  — se calcula el RR individual aqui
# ---------------------------------------------------------------------------

def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Reciproco del rango del primer resultado correcto.

    RR = 1 / posicion_del_primer_relevante
    Si ningun relevante aparece en la lista, RR = 0.

    Args:
        retrieved: Lista de usernames ordenada por ranking del sistema.
        relevant:  Conjunto de usernames que son respuesta correcta.

    Returns:
        Float en [0, 1]. 1.0 si el primer resultado es correcto.

    Ejemplo:
        retrieved = ['A', 'B', 'C'], relevant = {'B'}  -> RR = 1/2 = 0.5
        retrieved = ['A', 'B', 'C'], relevant = {'A'}  -> RR = 1/1 = 1.0
        retrieved = ['A', 'B', 'C'], relevant = {'D'}  -> RR = 0.0
    """
    for rank, user in enumerate(retrieved, start=1):
        if user in relevant:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
#  NDCG@k
# ---------------------------------------------------------------------------

def ndcg_at_k(retrieved: List[str], grades: Dict[str, int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain a nivel k.

    Mide la calidad del ranking considerando:
      - Orden: aparecer en posicion 1 vale mas que en posicion 3.
      - Relevancia graduada: un experto (grade=3) aporta mas que
        un usuario basico (grade=1).

    Formula:
      DCG@k  = sum_{i=1}^{k} (2^grade_i - 1) / log2(i + 1)
      IDCG@k = DCG del ranking perfecto (relevantes ordenados por grade desc)
      NDCG@k = DCG@k / IDCG@k

    Args:
        retrieved: Lista de usernames ordenada por ranking del sistema.
        grades:    Dict {username: grade} donde grade es 0 (irrelevante),
                   1 (baja), 2 (media) o 3 (alta relevancia).
        k:         Numero de resultados a considerar.

    Returns:
        Float en [0, 1]. 1.0 significa ranking perfecto.
    """
    if k <= 0 or not grades:
        return 0.0

    def dcg(ranking: List[str]) -> float:
        total = 0.0
        for i, user in enumerate(ranking, start=1):
            grade = grades.get(user, 0)
            total += (2 ** grade - 1) / math.log2(i + 1)
        return total

    # Ranking ideal: todos los usuarios con grade > 0, ordenados de mayor a menor
    ideal_users = sorted(
        [u for u, g in grades.items() if g > 0],
        key=lambda u: grades[u],
        reverse=True,
    )

    dcg_val = dcg(retrieved[:k])
    idcg_val = dcg(ideal_users[:k])

    return dcg_val / idcg_val if idcg_val > 0 else 0.0


# ---------------------------------------------------------------------------
#  Agregacion: mean sobre un conjunto de consultas
# ---------------------------------------------------------------------------

def mean_metric(values: List[float]) -> float:
    """Media aritmetica, devuelve 0.0 para listas vacias."""
    return sum(values) / len(values) if values else 0.0
