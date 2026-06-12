"""
Extracción de artefactos desde un repositorio GitHub.

Descarga ficheros de código, commits, issues y pull requests de un repo.
El filtrado de ficheros está acotado a extensiones relevantes para proyectos
de IA/LLM: Python, notebooks Jupyter, JS/TS, configuración y documentación.
Se excluyen deliberadamente Java, Go, Rust y similares.
"""

import base64
import json
import time
from collections import Counter
from pathlib import PurePosixPath
from typing import List, Optional

from github.Repository import Repository
from github.ContentFile import ContentFile

from src.ingestion.schemas import GitHubEvidence
from src.utils.logger import get_logger

log = get_logger(__name__)


# Ficheros de manifiesto: se descargan siempre independientemente de extensión.
# Permiten detectar dependencias AI/LLM (openai, langchain, etc.) aunque el
# resto del repo no tenga código relevante.
DEPENDENCY_FILES = {
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "environment.yml",   # conda environments, habituales en proyectos ML
    "package.json",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".env.example",      # revela qué APIs usa el proyecto (OPENAI_API_KEY, etc.)
}

# Solo los READMEs raíz: suelen describir propósito, modelos y arquitectura.
DOC_FILES = {
    "README.md",
    "README.rst",
}

# Extensiones de código y configuración útiles para proyectos de IA/LLM.
ALLOWED_EXTENSIONS = {
    ".py",     # lenguaje dominante en ML, RAG, agentes y backends de IA
    ".ipynb",  # notebooks Jupyter: habituales en demos y tutoriales de IA/ML
    ".js",     # apps web, wrappers de LLMs, chatbots frontend
    ".ts",     # TypeScript: proyectos AI SDK, apps con tipado estricto
    ".tsx",    # componentes React para interfaces de chat/IA
    ".jsx",    # ídem sin tipos
    ".md",     # documentación interna de módulos, guías de uso
    ".rst",    # documentación estilo Sphinx, habitual en librerías Python
    ".yaml",   # configuración de despliegue, CI/CD, pipelines de modelos
    ".yml",    # ídem (extensión alternativa)
}

# Directorios que se saltan siempre: generados automáticamente, caches, etc.
# No contienen código escrito por el desarrollador.
SKIP_DIRS = {
    ".git",
    "node_modules",
    "venv", ".venv", "env",
    "__pycache__",
    "dist", "build",
    ".tox", ".eggs",
    ".mypy_cache", ".pytest_cache",
}


def _classify_file(path: str) -> str:
    """Devuelve la categoría de un fichero para los logs de resumen."""
    filename = PurePosixPath(path).name
    if filename in DEPENDENCY_FILES:
        return "manifiestos"
    if filename in DOC_FILES:
        return "documentación"
    ext = PurePosixPath(path).suffix.lower()
    return {
        ".py":    "Python",
        ".ipynb": "notebooks",
        ".js": "JavaScript", ".jsx": "JavaScript",
        ".ts": "TypeScript", ".tsx": "TypeScript",
        ".md": "documentación", ".rst": "documentación",
        ".yaml": "configuración", ".yml": "configuración",
    }.get(ext, "otros")


def is_relevant_file(path: str) -> bool:
    """
    Decide si un fichero debe descargarse y procesarse.

    La lógica de prioridad es:
      1. Manifiestos de dependencias → siempre sí
      2. Ficheros generados (lock files) → siempre no
      3. READMEs raíz → siempre sí
      4. Resto: depende de la extensión
    """
    filename = path.split("/")[-1]

    if filename in DEPENDENCY_FILES or filename in DOC_FILES:
        return True

    # Lock files: generados automáticamente, no aportan señal técnica.
    if filename in ("yarn.lock", "pnpm-lock.yaml", "package-lock.json",
                    "poetry.lock", "Pipfile.lock"):
        return False

    return any(path.endswith(ext) for ext in ALLOWED_EXTENSIONS)


def _extract_notebook_cells(raw: str) -> str:
    """
    Extrae el código fuente de las celdas de un notebook Jupyter.

    Los notebooks son JSON puro con metadatos, outputs y bloques base64
    que el LLM no necesita ver. Esta función extrae solo el código y el
    markdown que el desarrollador escribió, que es lo que aporta señal técnica.
    Si el JSON está corrupto devuelve el raw original sin modificar.
    """
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    parts = []
    for cell in nb.get("cells", []):
        source = cell.get("source", [])
        if isinstance(source, list):
            source = "".join(source)
        if not source.strip():
            continue
        if cell.get("cell_type") == "markdown":
            parts.append(f"# [markdown]\n{source}")
        else:
            parts.append(source)

    return "\n\n".join(parts) if parts else raw


def read_file_content(file: ContentFile) -> str:
    """
    Lee el contenido de un fichero desde la API de GitHub.

    Los notebooks (.ipynb) reciben tratamiento especial: en vez de devolver
    el JSON crudo (con outputs, metadatos y base64), se extraen solo las
    celdas de código y markdown que el desarrollador escribió.
    """
    if file.encoding == "base64":
        raw = base64.b64decode(file.content).decode("utf-8", errors="ignore")
    else:
        raw = file.decoded_content.decode("utf-8", errors="ignore")

    if file.path.endswith(".ipynb"):
        return _extract_notebook_cells(raw)
    return raw


def walk_repository_files(
    repo: Repository,
    username: str,
    path: str = "",
    max_files: int = 100,
    _repo_pushed_at: Optional[str] = None,
    _is_root: bool = True,
) -> List[GitHubEvidence]:
    """
    Recorre recursivamente un repositorio y descarga los ficheros relevantes.

    El parámetro _repo_pushed_at se propaga en la recursión para no hacer
    una llamada extra a la API por cada subdirectorio.
    """
    t0 = time.monotonic() if _is_root else None

    if _repo_pushed_at is None:
        _repo_pushed_at = repo.pushed_at.isoformat() if repo.pushed_at else None

    evidences: List[GitHubEvidence] = []

    try:
        contents = repo.get_contents(path)
    except Exception as e:
        log.warning(f"No se pudo acceder a {repo.full_name}/{path}: {e}")
        return []

    if not isinstance(contents, list):
        contents = [contents]

    for item in contents:
        if len(evidences) >= max_files:
            break

        if item.type == "dir":
            if item.name in SKIP_DIRS:
                continue
            evidences.extend(
                walk_repository_files(
                    repo, username, item.path,
                    max_files=max_files - len(evidences),
                    _repo_pushed_at=_repo_pushed_at,
                    _is_root=False,
                )
            )

        elif item.type == "file" and is_relevant_file(item.path):
            try:
                text = read_file_content(item)
                if not text.strip():
                    continue

                evidences.append(
                    GitHubEvidence(
                        username=username,
                        repo=repo.full_name,
                        artifact_type="file",
                        path=item.path,
                        author=None,
                        text=text[:20_000],
                        metadata={
                            "source": "github",
                            "url": item.html_url,
                            "size": item.size,
                            "repo_pushed_at": _repo_pushed_at,
                        },
                    )
                )
            except Exception as e:
                log.warning(f"No se pudo leer {item.path}: {e}")

    if _is_root:
        elapsed = time.monotonic() - t0
        counts = Counter(_classify_file(e.path) for e in evidences if e.path)
        breakdown = ", ".join(f"{cat}: {n}" for cat, n in sorted(counts.items()))
        log.info(
            f"[{repo.full_name}] {len(evidences)} ficheros descargados "
            f"({breakdown}) en {elapsed:.1f}s"
        )
    else:
        log.debug(f"walk({repo.full_name}, '{path}') -> {len(evidences)} ficheros")

    return evidences


def _take(paginated, n: int):
    """Itera un PaginatedList de PyGitHub hasta n elementos sin usar slice."""
    for i, item in enumerate(paginated):
        if i >= n:
            break
        yield item


def extract_commits(
    repo: Repository,
    username: str,
    max_commits: int = 50,
) -> List[GitHubEvidence]:
    """
    Extrae commits del autor especificado (filtrado por GitHub, no local).
    El mensaje del commit describe qué hizo el desarrollador y qué cambia.
    """
    evidences: List[GitHubEvidence] = []

    for commit in _take(repo.get_commits(author=username), max_commits):
        author_login = commit.author.login if commit.author else None

        evidences.append(
            GitHubEvidence(
                username=username,
                repo=repo.full_name,
                artifact_type="commit",
                path=None,
                author=author_login,
                text=commit.commit.message,
                metadata={
                    "source": "github",
                    "sha": commit.sha,
                    "url": commit.html_url,
                    "date": (
                        commit.commit.author.date.isoformat()
                        if commit.commit.author and commit.commit.author.date
                        else None
                    ),
                },
            )
        )

    log.debug(f"commits({repo.full_name}, author={username}) -> {len(evidences)}")
    return evidences


def extract_issues(
    repo: Repository,
    username: str,
    max_issues: int = 50,
) -> List[GitHubEvidence]:
    """
    Extrae issues creados por el usuario en este repositorio.
    El título y cuerpo de los issues revelan problemas técnicos y decisiones.

    Se filtra por creator=username directamente en la API para no descargar
    issues de otros participantes del repo (que no aportan evidencia sobre
    el usuario analizado y consumen cuota de API innecesariamente).
    """
    evidences: List[GitHubEvidence] = []

    for issue in _take(repo.get_issues(state="all", creator=username), max_issues):
        if issue.pull_request is not None:
            continue  # los PRs se recogen por separado con extract_pull_requests

        evidences.append(
            GitHubEvidence(
                username=username,
                repo=repo.full_name,
                artifact_type="issue",
                path=None,
                author=issue.user.login if issue.user else None,
                text=f"{issue.title or ''}\n\n{issue.body or ''}".strip(),
                metadata={
                    "source": "github",
                    "url": issue.html_url,
                    "state": issue.state,
                    "created_at": issue.created_at.isoformat() if issue.created_at else None,
                    "closed_at": issue.closed_at.isoformat() if issue.closed_at else None,
                },
            )
        )

    log.debug(f"issues({repo.full_name}, creator={username}) -> {len(evidences)}")
    return evidences


def extract_pull_requests(
    repo: Repository,
    username: str,
    max_prs: int = 50,
) -> List[GitHubEvidence]:
    """
    Extrae pull requests abiertos por el usuario en este repositorio.

    La API de GitHub no permite filtrar PRs por creador en el endpoint de lista
    (a diferencia de issues, que sí tiene creator=). Para compensarlo se
    descargan hasta 3× el límite pedido y se filtra localmente. En repos con
    muchos colaboradores esto garantiza encontrar los PRs del usuario aunque
    aparezcan tarde en la lista paginada.
    """
    evidences: List[GitHubEvidence] = []

    for pr in _take(repo.get_pulls(state="all"), max_prs * 3):
        if pr.user and pr.user.login.lower() != username.lower():
            continue

        evidences.append(
            GitHubEvidence(
                username=username,
                repo=repo.full_name,
                artifact_type="pull_request",
                path=None,
                author=pr.user.login if pr.user else None,
                text=f"{pr.title or ''}\n\n{pr.body or ''}".strip(),
                metadata={
                    "source": "github",
                    "url": pr.html_url,
                    "state": pr.state,
                    "created_at": pr.created_at.isoformat() if pr.created_at else None,
                    "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
                },
            )
        )

        if len(evidences) >= max_prs:
            break

    log.debug(f"pull_requests({repo.full_name}, user={username}) -> {len(evidences)}")
    return evidences
