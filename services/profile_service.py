"""
services/profile_service.py — Gestión del pipeline de perfiles de GitHub.

MODO REAL  (USE_MOCK = False, por defecto)
    Delega en src.main.run_pipeline(), que ejecuta las 4 fases reales:
      Fase 1 — Ingesta desde GitHub (PyGithub)
      Fase 2 — Procesamiento LLM (detección de skills con Ollama)
      Fase 3 — Generación del perfil técnico agregado
      Fase 4 — Vectorización e indexación en ChromaDB
    Requiere: GITHUB_TOKEN en .env y Ollama en localhost:11434.

MODO MOCK  (USE_MOCK = True)
    Simula las fases del pipeline con tiempos realistas y genera
    un perfil técnico plausible a partir del username. No requiere
    token de GitHub ni Ollama. Útil para demos y desarrollo del frontend.

CONTRATO DE LA INTERFAZ
    add_profile(username) devuelve un Generator que hace yield de:
        (step_label: str, is_final: bool, result: dict | None)
    El último yield tiene is_final=True y result con el perfil completo.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Generator, Optional

USE_MOCK: bool = False
"""Cambiar a True para simular el pipeline sin GitHub, Ollama ni ChromaDB."""

# Pasos del pipeline tal como se muestran en la UI
PIPELINE_STEPS: list[str] = [
    "Conectando con la API de GitHub",
    "Obteniendo repositorios públicos",
    "Filtrando repositorios relevantes",
    "Extrayendo ficheros, commits, issues y pull requests",
    "Procesando evidencias técnicas",
    "Detectando hard-skills con LLM",
    "Generando perfil técnico agregado",
    "Indexando evidencias en ChromaDB",
]

# Pool de skills para generación mock
_SKILLS_POOL: list[str] = [
    "Python", "LangChain", "LangGraph", "RAG", "FastAPI", "ChromaDB",
    "OpenAI API", "Ollama", "LlamaIndex", "Transformers", "PyTorch",
    "TensorFlow", "Docker", "Kubernetes", "Streamlit", "PostgreSQL",
    "Redis", "Celery", "sentence-transformers", "FAISS", "Pinecone",
    "Weaviate", "Qdrant", "CrewAI", "AutoGen", "TypeScript", "REST API",
    "GraphQL", "CI/CD", "AsyncIO", "Pydantic", "SQLAlchemy",
]

# Demoras base (en segundos) para cada paso en mock
_STEP_DELAYS: list[float] = [0.3, 0.5, 0.4, 1.1, 0.9, 1.4, 0.6, 0.8]


# Interfaz pública

def add_profile(
    username: str,
) -> Generator[tuple[str, bool, Optional[dict]], None, None]:
    """
    Procesa el perfil técnico de un usuario de GitHub.

    Yields:
        (step_label, is_final, result)
          - step_label : Descripción del paso en curso.
          - is_final   : True únicamente en el yield final.
          - result     : None durante el proceso; dict con el perfil al terminar.
                         None también si ocurrió un error (is_final=True).
    """
    if USE_MOCK:
        yield from _mock_add_profile(username)
    else:
        yield from _real_add_profile(username)


# Implementación mock

def _mock_add_profile(
    username: str,
) -> Generator[tuple[str, bool, Optional[dict]], None, None]:
    """Simula el pipeline con datos generados y demoras realistas."""
    rng = random.Random(hash(username) % 2**32)

    for step, base_delay in zip(PIPELINE_STEPS, _STEP_DELAYS):
        time.sleep(rng.uniform(base_delay * 0.7, base_delay * 1.4))
        yield (step, False, None)

    # Perfil mock reproducible para el mismo username
    n_skills        = rng.randint(5, 9)
    skills          = rng.sample(_SKILLS_POOL, k=n_skills)
    repos_processed = rng.randint(4, 8)
    evidences_count = rng.randint(90, 230)

    result = {
        "username":               username,
        "status":                 "completed",
        "repositories_processed": repos_processed,
        "evidences_extracted":    evidences_count,
        "skills_detected":        skills,
        "profile_path":           f"data/profiles/{username}.json",
        "mock":                   True,
    }

    _persist_mock_profile(result)
    yield ("Perfil añadido correctamente", True, result)


# Implementación real

def _real_add_profile(
    username: str,
) -> Generator[tuple[str, bool, Optional[dict]], None, None]:
    """
    Llama al pipeline real (src.main.run_pipeline).

    NOTA: run_pipeline() ejecuta las 4 fases de forma síncrona.
    Los pasos previos al yield final son meramente informativos;
    el trabajo real (que puede durar varios minutos) ocurre en
    la llamada a run_pipeline().
    """
    # Mostrar los pasos mientras el pipeline trabaja en background
    # (en esta implementación síncrona, se muestran antes de la llamada)
    for step in PIPELINE_STEPS[:-1]:
        yield (step, False, None)

    try:
        from src.main import run_pipeline                        # noqa: PLC0415
        from src.vectorization.indexer import get_index_stats   # noqa: PLC0415

        run_pipeline(username)

        stats       = get_index_stats()
        user_blocks = stats["blocks_per_user"].get(username, 0)
        result      = _build_profile_summary(username, user_blocks)

        yield (PIPELINE_STEPS[-1], False, None)
        yield ("Perfil añadido correctamente", True, result)

    except Exception as exc:
        yield (f"Error en el pipeline: {exc}", True, None)


# Utilidades

def _persist_mock_profile(profile: dict) -> None:
    """Guarda el perfil mock en data/profiles/ para referencia."""
    path = Path(profile["profile_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)


def _build_profile_summary(username: str, indexed_blocks: int) -> dict:
    """
    Construye el resumen de perfil que consume el frontend a partir del
    JSON generado por src.profiling (data/perfiles/<username>_profile.json)
    y el número de bloques indexados en ChromaDB.

    Si el JSON no existe (usuario indexado en una ejecución antigua),
    devuelve el resumen con skills y repos vacíos en lugar de fallar.
    """
    profile_path = Path(f"data/perfiles/{username}_profile.json")

    skills: list[str] = []
    repos: set[str] = set()
    if profile_path.exists():
        try:
            with profile_path.open(encoding="utf-8") as f:
                data = json.load(f)
            hard_skills = data.get("technical_profile", {}).get("hard_skills", [])
            # Skills más sólidas primero (por número de evidencias y score medio)
            hard_skills.sort(
                key=lambda s: (s.get("evidence_count", 0), s.get("avg_composite_score", 0)),
                reverse=True,
            )
            skills = [s["skill"] for s in hard_skills[:12] if s.get("skill")]
            for s in hard_skills:
                repos.update(s.get("evidence_repositories", []))
        except (OSError, json.JSONDecodeError):
            pass

    return {
        "username":               username,
        "status":                 "completed",
        "repositories_processed": len(repos),
        "evidences_extracted":    indexed_blocks,
        "skills_detected":        skills,
        "profile_path":           str(profile_path),
        "mock":                   False,
    }


def load_indexed_profiles() -> list[dict]:
    """
    Reconstruye los perfiles ya indexados en ChromaDB sin re-ejecutar el
    pipeline. Permite que el frontend habilite el chat al arrancar con los
    datos persistidos de sesiones anteriores.

    Devuelve una lista de dicts con el mismo contrato que add_profile().
    """
    try:
        from src.vectorization.indexer import get_index_stats  # noqa: PLC0415
        stats = get_index_stats()
    except Exception:
        return []

    return [
        _build_profile_summary(username, blocks)
        for username, blocks in stats["blocks_per_user"].items()
    ]


def get_indexed_usernames() -> list[str]:
    """
    Devuelve los usernames disponibles.
    En modo mock lee data/profiles/; en modo real consulta ChromaDB.
    """
    if USE_MOCK:
        profiles_dir = Path("data/profiles")
        if not profiles_dir.exists():
            return []
        return [p.stem for p in profiles_dir.glob("*.json")]

    try:
        from src.vectorization.indexer import get_index_stats  # noqa: PLC0415
        return list(get_index_stats()["blocks_per_user"].keys())
    except Exception:
        return []
