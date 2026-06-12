from typing import List, Optional

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, EMBEDDING_MAX_TOKENS, EMBEDDING_OPTIMAL_TOKENS
from src.utils.logger import get_logger

MODEL_NAME = EMBEDDING_MODEL

log = get_logger(__name__)
_model: Optional[SentenceTransformer] = None

# Batch size para inferencia: 128 es un buen equilibrio entre velocidad y memoria
# para modelos de la familia MiniLM en CPU/GPU de consumo.
_ENCODE_BATCH_SIZE = 128


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info(f"Cargando modelo de embeddings: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Genera embeddings para una lista de textos con el modelo configurado en EMBEDDING_MODEL.

    - normalize_embeddings=True: aplica normalización L2 para que el producto escalar
      sea equivalente a la similitud coseno en el espacio de 384 dims. ChromaDB usa
      distancia coseno, por lo que esta normalización es obligatoria para coherencia
      entre indexación y recuperación.
    - batch_size=128: equilibrio entre velocidad y memoria en CPU/GPU de consumo.
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=_ENCODE_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def check_token_lengths(texts: List[str]) -> dict:
    """
    Diagnóstico: devuelve estadísticas de longitud en tokens para una lista de textos.

    Útil para verificar que los documentos a indexar no superan los límites
    del modelo antes de embeddearlos:
      - EMBEDDING_OPTIMAL_TOKENS (128): longitud de entrenamiento, rendimiento óptimo
      - EMBEDDING_MAX_TOKENS (256): límite duro, el modelo trunca silenciosamente

    Ejemplo de uso:
        stats = check_token_lengths(documents)
        print(stats)
        # {'total': 120, 'over_optimal': 45, 'over_max': 3,
        #  'max_tokens': 312, 'avg_tokens': 134.2}
    """
    model = get_model()
    tokenizer = model.tokenizer

    lengths = [
        len(tokenizer.encode(t, add_special_tokens=True, truncation=False))
        for t in texts
    ]

    over_optimal = sum(1 for l in lengths if l > EMBEDDING_OPTIMAL_TOKENS)
    over_max = sum(1 for l in lengths if l > EMBEDDING_MAX_TOKENS)

    stats = {
        "total": len(lengths),
        "over_optimal": over_optimal,   # superan 128 tokens (fuera del rango óptimo)
        "over_max": over_max,           # superan 256 tokens (truncación silenciosa)
        "max_tokens": max(lengths) if lengths else 0,
        "avg_tokens": round(sum(lengths) / len(lengths), 1) if lengths else 0,
    }

    if over_max > 0:
        log.warning(
            f"check_token_lengths: {over_max}/{len(lengths)} textos superan el límite "
            f"duro de {EMBEDDING_MAX_TOKENS} tokens y serán truncados silenciosamente."
        )
    elif over_optimal > 0:
        log.info(
            f"check_token_lengths: {over_optimal}/{len(lengths)} textos superan los "
            f"{EMBEDDING_OPTIMAL_TOKENS} tokens óptimos (máx={stats['max_tokens']}, "
            f"media={stats['avg_tokens']})."
        )

    return stats
