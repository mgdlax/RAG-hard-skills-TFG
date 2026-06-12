"""
Configuración centralizada del sistema RAG.

Importar estas constantes desde cualquier módulo evita valores
duplicados y facilita cambiar el modelo o la URL en un solo lugar.
"""

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"

# Modelo por defecto para detección de skills y generación de respuestas.
# Cambia este valor para usar otro modelo sin modificar el resto del código.
DEFAULT_LLM_MODEL = "qwen2.5-coder:7b"

# ── Vectorización ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "isuruwijesiri/all-MiniLM-L6-v2-code-search-512"

# Límites del modelo isuruwijesiri/all-MiniLM-L6-v2-code-search-512
# Fuente: https://huggingface.co/isuruwijesiri/all-MiniLM-L6-v2-code-search-512
# Modelo MiniLM-L6-v2 (384 dims) afinado para búsqueda semántica de código.
#   - Límite duro: 512 tokens (indicado en el nombre del modelo)
#   - Longitud óptima estimada: 256 tokens (conservador para texto mixto + código)
EMBEDDING_MAX_TOKENS = 512
EMBEDDING_OPTIMAL_TOKENS = 256

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_PATH = "data/chromadb"
CHROMA_COLLECTION = "technical_blocks"
