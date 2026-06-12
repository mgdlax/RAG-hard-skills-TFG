"""
Modulo de recuperacion semantica (Fase 5 del pipeline RAG).

Convierte la consulta del usuario en un embedding usando el mismo modelo
que se utilizo durante la indexacion (minilm-code-search-512) y recupera los
bloques tecnicos mas relevantes de ChromaDB por similitud coseno.

La clave del diseno: el texto de los documentos indexados combina
skill + explanation + signals, lo que permite que consultas en lenguaje
natural como "quien sabe construir agentes LLM?" coincidan
semanticamente con bloques donde LangChain o LangGraph aparecen con
su contexto real de uso.
"""

from typing import Any, Dict, List, Optional

from src.config import EMBEDDING_MAX_TOKENS, EMBEDDING_OPTIMAL_TOKENS
from src.vectorization.embedder import embed_texts
from src.vectorization.indexer import get_collection
from src.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_N_RESULTS = 30
MIN_COMPOSITE_SCORE = 0.0   # Filtro desactivado por defecto; se puede subir a 0.30
MIN_SEMANTIC_SIMILARITY = 0.55  # Bloques por debajo de este umbral aportan ruido

# Longitud máxima recomendada para una query en caracteres.
# isuruwijesiri/all-MiniLM-L6-v2-code-search-512 soporta hasta 512 tokens.
# ~1000 chars de texto natural ≈ 256 tokens (rango óptimo estimado).
# Fuente: https://huggingface.co/isuruwijesiri/all-MiniLM-L6-v2-code-search-512
_QUERY_WARN_CHARS = 1000


def _limit_blocks_per_user(blocks: List[Dict[str, Any]], max_per_user: int = 10) -> List[Dict[str, Any]]:
    """
    Evita que un usuario con muchos bloques indexados monopolice la recuperación.
    Conserva el orden original (los primeros bloques son los más similares).
    """
    counts: Dict[str, int] = {}
    result: List[Dict[str, Any]] = []
    for block in blocks:
        user = block.get("username", "unknown")
        if counts.get(user, 0) < max_per_user:
            result.append(block)
            counts[user] = counts.get(user, 0) + 1
    return result


def retrieve_blocks(
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    username_filter: Optional[str] = None,
    min_score: float = MIN_COMPOSITE_SCORE,
) -> List[Dict[str, Any]]:
    """
    Recupera los bloques técnicos más relevantes para la consulta.

    Proceso:
      1. Embeds la consulta con minilm-code-search-512 (mismo modelo que indexación)
      2. Consulta ChromaDB por similitud coseno (HNSW index)
      3. Filtra opcionalmente por username y composite_score mínimo
      4. Devuelve bloques enriquecidos con semantic_similarity calculada

    Args:
        query:           Consulta en lenguaje natural del reclutador.
        n_results:       Cuántos bloques recuperar de ChromaDB.
        username_filter: Si se indica, restringe la búsqueda a ese usuario.
        min_score:       Umbral mínimo de composite_score (0.0 = sin filtro).

    Returns:
        Lista de dicts con todos los metadatos del bloque más:
          - semantic_distance:   distancia coseno ChromaDB [0, 2]
          - semantic_similarity: similitud normalizada [0, 1]
          - document_text:       texto vectorizado original
    """
    log.info(f"Query: '{query[:80]}'")

    # Advertir si la query supera el rango óptimo del modelo.
    # El modelo soporta hasta 512 tokens; queries muy largas se truncan
    # silenciosamente y pierden calidad semántica en los últimos tokens.
    if len(query) > _QUERY_WARN_CHARS:
        log.warning(
            f"Query de {len(query)} chars supera los ~{_QUERY_WARN_CHARS} chars "
            f"recomendados (≈{EMBEDDING_OPTIMAL_TOKENS} tokens óptimos del modelo). "
            f"Considera acortarla para evitar truncación silenciosa."
        )

    # ── 1. Embedding de la consulta ──────────────────────────────────────────
    query_embedding = embed_texts([query])[0]

    # ── 2. Construcción del filtro de metadatos ──────────────────────────────
    where: Optional[Dict] = None

    conditions = []
    if username_filter:
        conditions.append({"username": {"$eq": username_filter}})
    if min_score > 0.0:
        conditions.append({"composite_score": {"$gte": min_score}})

    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    # ── 3. Consulta a ChromaDB ───────────────────────────────────────────────
    collection = get_collection()
    total_in_collection = collection.count()

    if total_in_collection == 0:
        log.warning("ChromaDB está vacío. Ejecuta primero: python -m src.main")
        return []

    # Nunca pedir más bloques de los que existen
    n = min(n_results, total_in_collection)

    query_kwargs: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": n,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    # ── 4. Formateo de resultados ────────────────────────────────────────────
    blocks: List[Dict[str, Any]] = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        block = dict(meta)
        block["document_text"] = doc
        # ChromaDB usa distancia coseno en el rango [0, 2]:
        #   0 → vectores idénticos (máxima similitud)
        #   2 → vectores opuestos  (mínima similitud)
        # Conversión a similitud normalizada [0, 1]: similitud = 1 - distancia / 2
        #
        # Esto es correcto porque el modelo produce embeddings L2-normalizados
        # (normalize_embeddings=True), de modo que la distancia coseno de ChromaDB
        # equivale a: dist = 1 - cos(θ), con cos(θ) ∈ [-1, 1] → dist ∈ [0, 2].
        # Para embeddings de texto el coseno negativo es prácticamente imposible,
        # por lo que los valores reales oscilan entre ~0.0 y ~0.9.
        block["semantic_distance"] = round(float(dist), 4)
        block["semantic_similarity"] = round(1.0 - float(dist) / 2.0, 4)
        blocks.append(block)

    # ── 5. Filtro por similitud semántica mínima ─────────────────────────────
    n_before = len(blocks)
    blocks = [b for b in blocks if b["semantic_similarity"] >= MIN_SEMANTIC_SIMILARITY]
    if len(blocks) < n_before:
        log.info(
            f"  Filtro similitud ({MIN_SEMANTIC_SIMILARITY}): "
            f"{n_before} → {len(blocks)} bloques"
        )

    # ── 6. Limitar bloques por usuario ───────────────────────────────────────
    blocks = _limit_blocks_per_user(blocks)

    if blocks:
        sim_max = blocks[0]["semantic_similarity"]
        sim_min = blocks[-1]["semantic_similarity"]
        log.info(
            f"  Recuperados {len(blocks)} bloques  "
            f"(similitud: {sim_max:.3f}–{sim_min:.3f})"
        )
    else:
        log.info("  Sin resultados")

    return blocks
