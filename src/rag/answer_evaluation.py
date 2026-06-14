"""
Módulo de evaluación de la respuesta generada frente a perfiles técnicos (Fase 6b).

Evalúa la consistencia interna del sistema: comprueba si las skills mencionadas
en la respuesta del LLM están respaldadas por los perfiles técnicos agregados
del pipeline. NO es una validación de verdad profesional absoluta — los perfiles
son una "silver ground truth" construida automáticamente a partir de GitHub y
pueden tener errores o lagunas. Esta métrica mide coherencia interna del RAG.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.utils.logger import get_logger

log = get_logger(__name__)

# Palabras que indican afirmaciones fuertes sobre una skill.
# Si el perfil tiene esa skill con confidence="baja", hay riesgo de sobre-afirmar.
STRONG_CLAIM_WORDS = [
    "experto",
    "domina",
    "claramente",
    "muy sólido",
    "alta experiencia",
    "especialista",
]


# Carga y acceso a perfiles

def load_user_profile(profile_path: Path) -> Optional[Dict[str, Any]]:
    """Carga un perfil técnico JSON desde disco. Devuelve None si falla."""
    try:
        with profile_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"No se pudo cargar el perfil {profile_path}: {e}")
        return None


def build_profile_skill_index(profile: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Construye un índice skill_lowercase → datos de la skill.
    Facilita la búsqueda case-insensitive sin modificar los datos originales.
    """
    index: Dict[str, Dict[str, Any]] = {}
    hard_skills = (
        profile.get("technical_profile", {}).get("hard_skills", [])
    )
    for skill_data in hard_skills:
        key = skill_data.get("skill", "").lower().strip()
        if key:
            index[key] = skill_data
    return index


# Detección de skills en la respuesta

def extract_mentioned_skills(
    answer: str,
    candidate_profiles: Dict[str, Dict[str, Dict[str, Any]]],
) -> List[str]:
    """
    Detecta qué skills de los perfiles de candidatos aparecen en la respuesta.

    Usa comparación simple case-insensitive — no se necesita NLP avanzado para
    términos técnicos concretos (LangChain, RAG, embeddings, etc.).

    Args:
        answer:             Respuesta generada por el LLM.
        candidate_profiles: Mapa username → skill_index (de build_profile_skill_index).

    Returns:
        Lista de skills encontradas en la respuesta (sin duplicados).
    """
    answer_lower = answer.lower()
    found: Set[str] = set()

    for skill_index in candidate_profiles.values():
        for skill_key in skill_index:
            if skill_key in answer_lower:
                found.add(skill_key)

    return sorted(found)


def detect_unsupported_skills(
    answer: str,
    candidates: List[Dict[str, Any]],
    candidate_profiles: Dict[str, Dict[str, Dict[str, Any]]],
) -> List[str]:
    """
    Detecta skills que la respuesta atribuye a un candidato pero que no
    aparecen en el perfil técnico agregado de ese candidato.

    Se comparan las matched_skills del ranking (la evidencia que vio el LLM)
    contra el índice de skills del perfil del mismo usuario. Una discrepancia
    no implica necesariamente una invención del LLM (puede ser un alias o una
    variante terminológica), pero es una señal útil para revisión manual.
    """
    answer_lower = answer.lower()
    unsupported: Set[str] = set()

    for candidate in candidates:
        username = candidate.get("username", "")
        profile_index = candidate_profiles.get(username)
        if profile_index is None:
            continue

        for matched in candidate.get("matched_skills", []):
            skill = matched.get("skill", "") if isinstance(matched, dict) else str(matched)
            skill_key = skill.lower().strip()
            if not skill_key or skill_key not in answer_lower:
                continue
            if skill_key not in profile_index:
                unsupported.add(f"@{username}/{skill}")

    return sorted(unsupported)


def detect_overconfident_claims(
    answer: str,
    candidate_profiles: Dict[str, Dict[str, Dict[str, Any]]],
) -> List[str]:
    """
    Detecta posibles sobre-afirmaciones: el LLM usa lenguaje fuerte (STRONG_CLAIM_WORDS)
    sobre una skill que en el perfil tiene confidence="baja".

    Devuelve una lista de advertencias descriptivas (no bloquea la respuesta).
    """
    answer_lower = answer.lower()
    warnings: List[str] = []

    has_strong_language = any(w in answer_lower for w in STRONG_CLAIM_WORDS)
    if not has_strong_language:
        return warnings

    for username, skill_index in candidate_profiles.items():
        for skill_key, skill_data in skill_index.items():
            if skill_key not in answer_lower:
                continue
            if skill_data.get("confidence") == "baja":
                strong_words_found = [w for w in STRONG_CLAIM_WORDS if w in answer_lower]
                if strong_words_found:
                    warnings.append(
                        f"@{username}/{skill_data['skill']}: "
                        f"confidence=baja pero respuesta usa {strong_words_found}"
                    )

    return warnings


# Evaluación principal

def evaluate_answer_against_profiles(
    answer: str,
    candidates: List[Dict[str, Any]],
    profiles_dir: Path,
) -> Dict[str, Any]:
    """
    Evalúa la respuesta generada frente a los perfiles técnicos de los candidatos.

    La métrica profile_consistency_score = menciones compatibles / total menciones
    mide coherencia interna del pipeline, no validez profesional absoluta.
    Los perfiles son silver ground truth automática — pueden tener lagunas.

    Returns:
        Diccionario con:
          - profile_evaluation_available: bool
          - profile_consistency_score: float or None
          - mentioned_profile_skills: lista de skills del perfil encontradas en respuesta
          - unsupported_skills: skills en respuesta sin soporte en perfiles
          - overconfident_claims: advertencias de sobre-afirmación
    """
    empty_result: Dict[str, Any] = {
        "profile_evaluation_available": False,
        "profile_consistency_score": None,
        "mentioned_profile_skills": [],
        "unsupported_skills": [],
        "overconfident_claims": [],
    }

    if not profiles_dir.exists():
        log.debug(f"Directorio de perfiles no encontrado: {profiles_dir}")
        return empty_result

    # Cargar perfiles de los candidatos presentes
    candidate_profiles: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for candidate in candidates:
        username = candidate["username"]
        # Buscar archivo de perfil por nombre (varias convenciones posibles)
        for candidate_file in [
            profiles_dir / f"{username}.json",
            profiles_dir / f"{username}_profile.json",
            profiles_dir / f"profile_{username}.json",
        ]:
            if candidate_file.exists():
                profile = load_user_profile(candidate_file)
                if profile:
                    candidate_profiles[username] = build_profile_skill_index(profile)
                break

    if not candidate_profiles:
        log.debug("No se encontraron perfiles para los candidatos actuales")
        return empty_result

    mentioned = extract_mentioned_skills(answer, candidate_profiles)
    unsupported = detect_unsupported_skills(answer, candidates, candidate_profiles)
    overconfident = detect_overconfident_claims(answer, candidate_profiles)

    # Consistencia = proporción de menciones técnicas que tienen respaldo en perfiles
    total_mentions = len(mentioned) + len(unsupported)
    if total_mentions > 0:
        consistency_score = round(len(mentioned) / total_mentions, 4)
    else:
        consistency_score = None  # Sin menciones técnicas detectables

    log.info(
        f"  Evaluación perfiles: {len(mentioned)} skills con soporte, "
        f"{len(unsupported)} sin soporte, "
        f"consistencia={consistency_score}"
    )

    return {
        "profile_evaluation_available": True,
        "profile_consistency_score": consistency_score,
        "mentioned_profile_skills": mentioned,
        "unsupported_skills": unsupported,
        "overconfident_claims": overconfident,
    }
