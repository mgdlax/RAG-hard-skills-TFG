from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


#  Modelos Pydantic — contrato de la respuesta JSON del LLM

class SkillDetection(BaseModel):
    """Una skill detectada por el LLM en una evidencia."""

    skill: str = Field(..., min_length=1, max_length=80)
    explanation: str = Field(default="", max_length=300)
    # 1500 chars ≈ 15-20 líneas de código; el prompt pide "máximo 10 líneas"
    # pero siendo generosos evitamos que fragmentos legítimos fallen validación.
    evidence_fragment: str = Field(default="", max_length=1500)

    @field_validator("skill")
    @classmethod
    def skill_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("skill no puede ser una cadena vacía")
        return v

    @field_validator("explanation", "evidence_fragment", mode="before")
    @classmethod
    def coerce_none_to_empty(cls, v: object) -> str:
        """El LLM a veces devuelve null en vez de "" — lo normalizamos."""
        return "" if v is None else str(v)


class DetectionResponse(BaseModel):
    """Respuesta completa del prompt de detección de skills."""

    skills: list[SkillDetection] = Field(default_factory=list)

    @field_validator("skills", mode="before")
    @classmethod
    def coerce_none_to_list(cls, v: object) -> list:
        return [] if v is None else v


def build_detection_prompt(
    repo: str,
    path: str,
    artifact_type: str,
    content: str,
) -> str:
    """
    Prompt principal del pipeline de métricas.

    El LLM tiene DOS tareas:
      1. Detectar tecnologías/skills de IA y software explícitamente presentes.
      2. Verificar si realmente se usan en el artefacto, no solo si aparecen mencionadas.

    Restricciones clave:
      · Solo tecnologías explícitas en el contenido.
      · No inferir por nombre del repositorio, ruta o extensión.
      · En código fuente, un import aislado NO es suficiente.
      · Si no hay evidencia clara de uso real → lista vacía.
    """
    return f"""Eres un analizador estricto de código y artefactos técnicos para un sistema de perfiles técnicos de desarrolladores.

Repositorio: {repo}
Fichero/artefacto: {path} [{artifact_type}]

Contenido:
\"\"\"
{content}
\"\"\"

Tu tarea es identificar tecnologías, frameworks y librerías de IA/ML o desarrollo de software que aparezcan EXPLÍCITAMENTE en el contenido anterior y cuyo uso esté suficientemente evidenciado.

Reglas estrictas:
- No inventes tecnologías.
- No infieras tecnologías por el nombre del repositorio, el nombre del fichero, la ruta o la extensión.
- Solo incluye tecnologías que aparezcan literalmente en el contenido.
- Céntrate en tecnologías con valor para un perfil técnico: IA/LLM, ML, backend, APIs, testing, infraestructura, bases de datos, cloud, DevOps, frontend relevante, etc.
- Ignora librerías estándar o menciones triviales: os, sys, json, logging, pathlib, typing, datetime, re, math, collections, etc.
- Ignora menciones puramente textuales, futuras o hipotéticas, como TODO, ejemplos no ejecutados, comentarios vagos o frases del tipo "se podría usar X".
- Si el contenido no tiene tecnologías relevantes con evidencia clara, devuelve una lista vacía.

Criterio de uso real:
- En código fuente, NO basta con que una tecnología aparezca en un import.
- Solo incluye una tecnología si hay evidencia de que se usa realmente en el artefacto.
- Considera uso real cuando aparezcan llamadas, instanciaciones, decoradores, herencia, configuración, tipos, funciones, objetos o patrones propios de esa tecnología.
- Si una librería aparece importada pero no vuelve a usarse en el resto del contenido, NO la incluyas.
- Si una tecnología aparece solo en comentarios, docstrings o texto explicativo dentro de código, NO la incluyas salvo que también haya uso real en código ejecutable.
- En ficheros de dependencias o manifiestos, como requirements.txt, pyproject.toml, package.json, Dockerfile o docker-compose.yml, la declaración explícita de una dependencia o herramienta sí cuenta como evidencia.
- En documentación, como README o Markdown, incluye una tecnología solo si el texto describe claramente su instalación, configuración, ejecución o uso en el proyecto.

Para cada tecnología encontrada:
1. "skill": nombre canónico corto. Ejemplos: "LangChain", "FastAPI", "PyTorch", "Docker", "PostgreSQL".
2. "explanation": una frase de máximo 25 palabras explicando cómo se usa en este artefacto.
3. "evidence_fragment": fragmento literal del contenido que demuestre el uso real. Reglas:
   - Para código fuente (ficheros .py, .ts, .js, etc.): incluye el bloque completo donde aparece la tecnología, entre 3 y 10 líneas (funciones, clases, llamadas, configuraciones). Nunca devuelvas solo el import; incluye el contexto de uso.
   - Para manifiestos (requirements.txt, pyproject.toml, Dockerfile, etc.): incluye las líneas relevantes del fichero.
   - Para documentación/README: incluye el fragmento de texto o código de ejemplo que describa su uso.
   - Para issues/PRs: incluye el fragmento de texto más representativo del uso técnico descrito.
   - Nunca devuelvas un fragmento vacío si hay evidencia de uso real.

Antes de responder, revisa internamente cada tecnología candidata:
- ¿Aparece literalmente en el contenido?
- ¿Es una tecnología relevante para un perfil técnico?
- ¿Hay evidencia de uso real y no solo una mención aislada?
- ¿El fragmento citado demuestra ese uso?
Si alguna respuesta es "no", elimina esa tecnología.

Devuelve ÚNICAMENTE un JSON válido. Sin Markdown. Sin texto adicional.

{{
  "skills": [
    {{
      "skill": "nombre de la tecnología",
      "explanation": "una frase describiendo su uso real en este artefacto",
      "evidence_fragment": "fragmento literal que evidencia el uso real"
    }}
  ]
}}"""
