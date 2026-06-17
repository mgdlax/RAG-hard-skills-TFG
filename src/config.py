"""
Configuración centralizada del sistema RAG.

Importar estas constantes desde cualquier módulo evita valores
duplicados y facilita cambiar modelos, URLs o parámetros en un solo lugar.
"""

# ============================================================================
# OLLAMA - Configuración del LLM local
# ============================================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_LLM_MODEL = "qwen2.5-coder:7b"
LLM_TEMPERATURE = 0.2
LLM_TIMEOUT_SECONDS = 120

# ============================================================================
# EMBEDDINGS - Modelo para vectorización de código
# ============================================================================
EMBEDDING_MODEL = "isuruwijesiri/all-MiniLM-L6-v2-code-search-512"
EMBEDDING_DIMENSIONS = 384
EMBEDDING_MAX_TOKENS = 512
EMBEDDING_OPTIMAL_TOKENS = 256

# ============================================================================
# CHROMADB - Vector store para búsqueda semántica
# ============================================================================
CHROMA_PATH = "data/chromadb"
CHROMA_COLLECTION = "technical_blocks"
CHROMA_SIMILARITY_THRESHOLD = 0.55

# ============================================================================
# RAG - Parámetros del pipeline de recuperación y ranking
# ============================================================================
RETRIEVAL_N_RESULTS = 30
RETRIEVAL_MAX_BLOCKS_PER_USER = 10
RANKING_TOP_K = 5
RANKING_SEMANTIC_WEIGHT = 0.60
RANKING_COMPOSITE_WEIGHT = 0.40
RANKING_SKILL_BONUS = 0.02

# ============================================================================
# GENERACIÓN - Parámetros para síntesis de respuestas
# ============================================================================
MAX_SKILLS_PER_CANDIDATE = 4
MAX_CANDIDATES_IN_PROMPT = 3
MAX_EVIDENCES_IN_RESPONSE = 10

# ============================================================================
# DETECCIÓN DE CÓDIGO - Heurísticas para identificar fragmentos técnicos
# ============================================================================
CODE_MARKERS = (
    "def ", "class ", "import ", "from ", "return ",
    "async def ", "try:", "except ", "@", "lambda",
    "function ", "const ", "let ", "var ", "export ",
    "public ", "private ", "protected ", "interface ",
    "<Component", "useState", "useEffect", "render()"
)

# ============================================================================
# PIPELINE - Límites de ingesta y procesamiento
# ============================================================================
PIPELINE_MAX_REPOS = 6
PIPELINE_MAX_FILES_PER_REPO = 50
PIPELINE_MAX_COMMITS_PER_REPO = 20
PIPELINE_MAX_ISSUES_PER_REPO = 10
PIPELINE_MAX_PRS_PER_REPO = 10
PIPELINE_MAX_FILES_TO_PROCESS_WITH_LLM = 30
