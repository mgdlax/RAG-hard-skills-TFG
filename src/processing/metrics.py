"""
Métricas estructurales de calidad de evidencia.

Sin listas de tuplas de tecnologías: la detección de skills corre a cargo
del LLM en code_filter.py. Este módulo responde únicamente a la pregunta
"¿qué tan valiosa es esta evidencia?" con 4 métricas independientes.

Métricas
--------
recency          ¿Qué tan reciente es la evidencia?
authorship       ¿Con qué seguridad se atribuye al usuario?
artifact_weight  ¿Qué tan valioso es este tipo de fichero/artefacto?
content_richness ¿Cuánta sustancia técnica contiene el texto?

La función principal `score_evidence` devuelve un único bloque por evidencia
(sin split por skill). code_filter.py añade el campo `skill` tras la llamada al LLM.
"""

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

COMPOSITE_WEIGHTS: Dict[str, float] = {
    "recency":          0.15,
    "authorship":       0.20,
    "artifact_weight":  0.30,
    "content_richness": 0.35,
}


#  MÉTRICA 1 · recency
#  ¿Qué tan reciente es la evidencia?
#
#  Decaimiento exponencial con semivida de 365 días:
#    hoy      → 1.0
#    1 año    → 0.50
#    2 años   → 0.25

def compute_recency(record: Dict[str, Any]) -> float:
    artifact_type = record.get("artifact_type", "")
    metadata = record.get("metadata", {})

    date_str: Optional[str] = None
    if artifact_type == "commit":
        date_str = metadata.get("date")
    elif artifact_type == "issue":
        date_str = metadata.get("created_at")
    elif artifact_type == "pull_request":
        date_str = metadata.get("merged_at") or metadata.get("created_at")
    elif artifact_type == "file":
        # repo_pushed_at es la fecha del último push al repo completo, no la
        # última modificación de este fichero concreto. Obtener la fecha real
        # del fichero requeriría una llamada extra a la API por cada uno, lo
        # que triplicaría el consumo de cuota. Se acepta como aproximación.
        date_str = metadata.get("repo_pushed_at")

    if not date_str:
        return 0.5  # sin fecha conocida → valor neutro

    try:
        date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        days_old = (datetime.now(timezone.utc) - date).days
        return round(math.exp(-0.693 * days_old / 365), 4)
    except (ValueError, TypeError):
        return 0.5


#  MÉTRICA 2 · authorship
#  ¿Con qué seguridad se atribuye la evidencia al usuario?
#
#  commit (filtrado por author en la API)  → 0.95  máxima seguridad
#  file en repo propio (username/repo)     → 0.90
#  file en repo ajeno (colaboración)       → 0.60
#  pull request abierta por el usuario     → 0.85
#  pull request de otro                    → 0.45
#  issue propio                            → 0.65
#  issue de otro                           → 0.35

def compute_authorship(record: Dict[str, Any]) -> float:
    username = record.get("username", "")
    artifact_type = record.get("artifact_type", "")
    repo = record.get("repo", "")
    author = record.get("author", "") or ""

    repo_owner = repo.split("/")[0] if "/" in repo else ""
    is_own_repo = repo_owner.lower() == username.lower()

    if artifact_type == "commit":
        return 0.95
    if artifact_type == "file":
        return 0.90 if is_own_repo else 0.60
    if artifact_type == "pull_request":
        return 0.85 if author.lower() == username.lower() else 0.45
    if artifact_type == "issue":
        return 0.65 if author.lower() == username.lower() else 0.35
    return 0.50


#  MÉTRICA 3 · artifact_weight
#  ¿Qué tan valioso es este tipo de fichero o artefacto?
#
#  Para ficheros, la valoración se basa en:
#    · Extensión: código ejecutable > configuración > documentación
#    · Nombre:    los manifiestos de dependencias siempre son valiosos
#    · Tests:     evidencian que el código llega a producción (+bonus)
#    · Profundidad en el árbol: ficheros muy anidados suelen ser auxiliares
#
#  Para artefactos no-file:
#    commit > pull_request > issue  (en términos de evidencia técnica directa)

# Ficheros que declaran dependencias explícitas → siempre valiosos.
# Sin extensión para Dockerfile porque split('.') daría ext vacío.
_MANIFESTS = {
    "requirements.txt", "pyproject.toml", "setup.py", "environment.yml",
    "package.json", "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    ".env.example",
}

# Peso base por extensión de fichero.
# Extensiones no listadas reciben 0.50 (valor neutro).
_EXT_WEIGHT = {
    "py":   0.85,   # Python — lenguaje principal en IA/ML
    "ts":   0.80,   # TypeScript
    "tsx":  0.78,
    "js":   0.75,   # JavaScript
    "jsx":  0.73,
    "yaml": 0.55,   # configuración (CI/CD, Docker Compose)
    "yml":  0.55,
    "md":   0.40,   # documentación
    "rst":  0.38,
}


def compute_artifact_weight(record: Dict[str, Any]) -> float:
    artifact_type = record.get("artifact_type", "")
    path = record.get("path", "") or ""

    if artifact_type == "commit":
        return 0.80   # autoría directa, mensaje describe qué cambió

    if artifact_type == "pull_request":
        return 0.60   # describe qué aportó, pero es texto no código

    if artifact_type == "issue":
        return 0.45   # útil para contexto, pero raramente es código directo

    if artifact_type == "file":
        filename = path.split("/")[-1]
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        depth = path.count("/")   # número de directorios hasta el fichero

        # Base por extensión (o neutro si no está en la tabla)
        base = _EXT_WEIGHT.get(ext, 0.50)

        # Los manifiestos declaran dependencias explícitas → alta señal
        if filename in _MANIFESTS:
            base = max(base, 0.75)

        # Tests → el código tiene cobertura, llega a producción.
        # Se comprueba solo en el nombre del fichero, no en el path completo,
        # para evitar falsos positivos como "my_test_project/main.py".
        filename_only = path.split("/")[-1]
        if any(kw in filename_only for kw in ("test_", "_test.", ".spec.", ".test.")):
            base = min(base + 0.10, 1.0)

        # Penalización por profundidad: ficheros muy anidados suelen ser
        # utilidades auxiliares o código generado automáticamente.
        # −0.03 por nivel, máximo −0.18 (6 niveles o más).
        depth_penalty = min(depth * 0.03, 0.18)

        return round(max(base - depth_penalty, 0.10), 4)

    return 0.50   # tipo desconocido → neutro


#  MÉTRICA 4 · content_richness
#  ¿Cuánta sustancia técnica tiene el contenido?
#
#  Para ficheros: número de líneas no vacías + presencia de definiciones
#  estructurales (funciones, clases). Un fichero con 3 líneas no aporta
#  suficiente contexto aunque sea Python puro.
#
#  Para commits / issues / PRs: longitud del texto y presencia de código
#  incrustado (bloques ```) o términos técnicos.

def compute_content_richness(record: Dict[str, Any]) -> float:
    artifact_type = record.get("artifact_type", "")
    text = record.get("text", "").strip()

    if not text:
        return 0.0

    if artifact_type == "file":
        return _file_richness(text)
    else:
        return _text_richness(text)


def _file_richness(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    n = len(lines)

    # Score progresivo por volumen de código
    if n < 5:
        score = 0.10
    elif n < 20:
        score = 0.40
    elif n < 80:
        score = 0.65
    elif n < 200:
        score = 0.80
    else:
        score = 0.90

    # Bonus: tiene definiciones estructurales → código real, no solo imports
    structural = re.search(
        r"^\s*(def |class |function |const |export |async def |@)",
        text, re.MULTILINE,
    )
    if structural:
        score = min(score + 0.10, 1.0)

    return round(score, 4)


def _text_richness(text: str) -> float:
    n = len(text)

    if n < 20:
        score = 0.10
    elif n < 100:
        score = 0.35
    elif n < 400:
        score = 0.60
    elif n < 1000:
        score = 0.75
    else:
        score = 0.85

    # Bonus: contiene fragmentos de código incrustados
    if "```" in text:
        score = min(score + 0.10, 1.0)

    return round(score, 4)


#  PUNTUACIÓN COMPUESTA

def compute_composite_score(scores: Dict[str, float]) -> float:
    return round(
        sum(COMPOSITE_WEIGHTS[k] * scores.get(k, 0.0) for k in COMPOSITE_WEIGHTS),
        4,
    )


#  FUNCIÓN PRINCIPAL

def score_evidence(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calcula las métricas estructurales de una evidencia.

    Devuelve UN único bloque por evidencia. El campo `skill` NO se rellena
    aquí: lo añade code_filter.py tras llamar al LLM.

    Estructura del bloque devuelto
    username   · usuario analizado
    source     · repo, path, artifact_type, url
    scores     · las 4 métricas + composite_score
    _raw_text  · texto original (campo interno, se elimina al guardar)
    """
    scores = {
        "recency":          compute_recency(record),
        "authorship":       compute_authorship(record),
        "artifact_weight":  compute_artifact_weight(record),
        "content_richness": compute_content_richness(record),
    }
    scores["composite_score"] = compute_composite_score(scores)

    return {
        "username": record.get("username", ""),
        "skill": "",           # se rellena en code_filter.py tras el LLM
        "source": {
            "repo":          record.get("repo", ""),
            "path":          record.get("path", "") or "",
            "artifact_type": record.get("artifact_type", ""),
            "url":           record.get("metadata", {}).get("url", ""),
        },
        "scores": scores,
        "explanation":       "",   # se rellena en code_filter.py
        "evidence_fragment": "",   # se rellena en code_filter.py
        "_raw_text":         record.get("text", "")[:8000],
    }
