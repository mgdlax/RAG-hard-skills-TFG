"""
Módulo de indexación vectorial.

Adapta el nuevo formato de bloques puntuados (un bloque por skill/evidencia)
a documentos ChromaDB. El texto a vectorizar combina skill + explanation +
señales de detección para maximizar la calidad semántica del retrieval.
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb

from src.config import CHROMA_PATH, CHROMA_COLLECTION
from src.vectorization.embedder import embed_texts, check_token_lengths, MODEL_NAME
from src.utils.logger import get_logger

log = get_logger(__name__)

COLLECTION_NAME = CHROMA_COLLECTION


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _safe_id(text: str, max_len: int = 512) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", text)
    return sanitized[:max_len]


# Límites de caracteres para build_document_text(), calibrados para el modelo
# isuruwijesiri/all-MiniLM-L6-v2-code-search-512 (máx 512 tokens, óptimo ~256).
#
# Estimaciones conservadoras (el código es más denso que el texto natural):
#   - Texto natural:  ~1 token cada 4 chars  →  200 chars ≈  50 tokens
#   - Código fuente:  ~1 token cada 3 chars  →  300 chars ≈ 100 tokens
#
# Distribución del presupuesto (~160 tokens con estos límites):
#   "Skill: X | Evidence: " (fijo)  →  ~10 tokens
#   explanation                     →  ~50 tokens  (200 chars)
#   " | Code: "             (fijo)  →   ~5 tokens
#   fragment                        →  ~100 tokens (300 chars de código)
#   ─────────────────────────────────────────────
#   Total estimado                  → ~165 tokens  ✓ bien dentro del óptimo (256)
_EXPLANATION_MAX_CHARS = 200
_FRAGMENT_MAX_CHARS = 300


def build_document_text(block: Dict[str, Any]) -> str:
    """
    Construye el texto a vectorizar para un bloque técnico.

    Combina la skill detectada, la explicación del LLM y el fragmento
    de código representativo. Este texto es lo que hace que consultas
    como "quien sabe de LangChain" coincidan semánticamente con los
    bloques donde esa skill aparece con su contexto real de uso.

    Los límites de caracteres están calibrados para mantenerse dentro de los
    256 tokens óptimos del modelo de code-search (máx 512 tokens).
    Exceder el límite duro descarta el contenido más lejano del inicio,
    que suele ser el fragmento de código.
    Fuente: https://huggingface.co/isuruwijesiri/all-MiniLM-L6-v2-code-search-512
    """
    skill = block.get("skill", "")
    explanation = block.get("explanation", "")[:_EXPLANATION_MAX_CHARS]
    fragment = block.get("evidence_fragment", "")[:_FRAGMENT_MAX_CHARS]

    parts = [f"Skill: {skill}"]
    if explanation:
        parts.append(f"Evidence: {explanation}")
    if fragment:
        parts.append(f"Code: {fragment}")

    return " | ".join(parts)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _prepare_documents(
    records: List[Dict[str, Any]],
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Transforma los bloques puntuados en (ids, documents, metadatas)
    listos para ChromaDB. Cada bloque ya tiene username incluido.
    """
    ids = []
    documents = []
    metadatas = []

    for i, block in enumerate(records):
        doc_text = build_document_text(block)
        if not doc_text.strip():
            continue

        username = block.get("username", "")
        source = block.get("source", {})
        scores = block.get("scores", {})

        repo = source.get("repo", "")
        path = source.get("path", "") or ""
        skill = block.get("skill", "")

        raw_id = f"{username}__{skill}__{repo}__{path}__{i}"
        doc_id = _safe_id(raw_id)

        ids.append(doc_id)
        documents.append(doc_text)
        metadatas.append({
            # Identificación
            "username":      username,
            "skill":         skill,
            "repo":          repo,
            "path":          path,
            "artifact_type": source.get("artifact_type", ""),
            "url":           source.get("url", ""),
            # Métricas estructurales (nombres alineados con metrics.py)
            "composite_score":  float(scores.get("composite_score",  0.0)),
            "recency":          float(scores.get("recency",          0.0)),
            "authorship":       float(scores.get("authorship",       0.0)),
            "artifact_weight":  float(scores.get("artifact_weight",  0.0)),
            "content_richness": float(scores.get("content_richness", 0.0)),
            # Contenido legible
            "explanation":       block.get("explanation",       "")[:400],
            "evidence_fragment": block.get("evidence_fragment", "")[:800],
        })

    return ids, documents, metadatas


def index_user(
    username: str,
    processed_path: Path,
    batch_size: int = 64,
) -> int:
    """
    Indexa todos los bloques puntuados de un usuario en ChromaDB.
    Usa upsert para idempotencia: re-indexar no genera duplicados.
    Devuelve el número de bloques indexados.
    """
    if not processed_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo procesado: {processed_path}")

    records = _load_jsonl(processed_path)

    # Filtrar bloques del usuario indicado (por si el JSONL mezcla usuarios)
    # filtramos bloques que tienen una skill para limpiar ruido
    user_records = [
        r for r in records
        if r.get("username", "") == username and r.get("skill", "").strip()
    ]
    if not user_records:
        # Compatibilidad: si los registros no tienen username, usar todos
        user_records = records

    ids, documents, metadatas = _prepare_documents(user_records)

    if not documents:
        log.warning(f"Sin bloques para indexar en {processed_path.name}")
        return 0

    log.info(f"Indexando {len(documents)} bloques para '{username}'")
    log.info(f"  Modelo de embeddings: {MODEL_NAME}  batch_size={batch_size}")

    # Diagnóstico de longitud de tokens antes de embeddear.
    # Detecta documentos que superan el rango óptimo (256) o el límite duro (512)
    # del modelo de code-search para que puedan corregirse en build_document_text().
    token_stats = check_token_lengths(documents)
    log.info(
        f"  Token stats: avg={token_stats['avg_tokens']} "
        f"max={token_stats['max_tokens']} "
        f"over_optimal={token_stats['over_optimal']} "
        f"over_max={token_stats['over_max']}"
    )

    collection = get_collection()
    t_start = time.time()
    total_batches = (len(documents) + batch_size - 1) // batch_size

    for batch_num, start in enumerate(range(0, len(documents), batch_size), 1):
        end = start + batch_size
        batch_docs = documents[start:end]
        batch_meta = metadatas[start:end]
        batch_ids = ids[start:end]

        t_batch = time.time()
        embeddings = embed_texts(batch_docs)

        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_meta,
        )
        log.debug(
            f"  Batch {batch_num}/{total_batches}: "
            f"{len(batch_docs)} docs embeddados y almacenados "
            f"({time.time() - t_batch:.1f}s)"
        )

    elapsed = time.time() - t_start
    log.info(f"Indexación completada: {len(documents)} bloques en {elapsed:.1f}s")
    return len(documents)


def get_index_stats() -> Dict[str, Any]:
    collection = get_collection()
    total = collection.count()

    # collection.get sin límite trae todos los documentos de ChromaDB.
    # Para el volumen del TFG (5 usuarios, ~500-1000 bloques) es aceptable,
    # pero en producción habría que paginar o mantener un índice separado.
    all_items = collection.get(include=["metadatas"])
    users: Dict[str, int] = {}
    skills_per_user: Dict[str, set] = {}

    for meta in all_items["metadatas"]:
        user = meta.get("username", "unknown")
        skill = meta.get("skill", "")
        users[user] = users.get(user, 0) + 1
        skills_per_user.setdefault(user, set()).add(skill)

    return {
        "total_blocks": total,
        "blocks_per_user": users,
        "unique_skills_per_user": {u: len(s) for u, s in skills_per_user.items()},
    }
