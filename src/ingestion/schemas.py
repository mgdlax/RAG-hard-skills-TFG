"""
Esquema Pydantic para las evidencias crudas descargadas de GitHub.

Un objeto GitHubEvidence representa un artefacto concreto de un repositorio:
puede ser un fichero de codigo, un commit, un issue o un pull request.
El pipeline posterior (processing) convierte estas evidencias en bloques
con skills detectadas y metricas de calidad.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class GitHubEvidence(BaseModel):
    """Un artefacto tecnico descargado de GitHub."""

    username: str          # Login del desarrollador analizado
    repo: str              # full_name del repo (owner/repo)
    artifact_type: str     # "file" | "commit" | "issue" | "pull_request"
    path: Optional[str] = None      # Ruta del fichero (solo para artifact_type="file")
    author: Optional[str] = None    # Login del autor del artefacto
    text: str                       # Contenido o mensaje del artefacto
    metadata: Dict[str, Any] = Field(default_factory=dict)   # URL, fechas, sha, estado, etc.