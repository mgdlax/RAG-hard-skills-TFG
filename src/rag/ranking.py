"""
Módulo de ranking multi-usuario (Fase 5b del pipeline RAG).

Recibe los bloques recuperados por retrieval.py (mezclados de varios usuarios)
y los agrupa por username para producir una lista ordenada de candidatos.

Diseño del scoring:
  - Por bloque: combined_score = semantic_similarity * 0.40 + composite_score * 0.60
    (el composite_score es la métrica de calidad determinística del pipeline;
     la similitud semántica mide cuánto coincide con la consulta concreta)
  - Por skill: se queda con el bloque de mayor combined_score (evita duplicados)
  - Por usuario: media de los scores de todas las skills coincidentes
  - Bonus de amplitud: +0.02 por skill única, hasta +0.10
    (un candidato con 5 skills relevantes es mejor que uno con 1 muy buena)
"""

from collections import defaultdict
from typing import Any, Dict, List

SEMANTIC_WEIGHT = 0.40   # Peso de la similitud semántica con la consulta
COMPOSITE_WEIGHT = 0.60  # Peso de la calidad de evidencia (métricas del pipeline)
BREADTH_BONUS_PER_SKILL = 0.02
MAX_BREADTH_BONUS = 0.10

MIN_FINAL_SCORE = 0.55                     # Por debajo: confianza baja automática
MIN_SCORE_GAP_FOR_STRONG_RECOMMENDATION = 0.08  # Ventaja sobre el 2º para ser "alta"
MIN_SKILLS_FOR_CONFIDENCE = 2              # Menos skills → penalizar hacia baja


def attach_confidence(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Añade el campo "confidence" a cada candidato con valor "alta", "media" o "baja".

    Criterios (en orden de prioridad):
      1. final_score < MIN_FINAL_SCORE → "baja" (score insuficiente)
      2. Menos de MIN_SKILLS_FOR_CONFIDENCE skills coincidentes → "baja"
      3. Es el primero Y su ventaja sobre el 2º >= MIN_SCORE_GAP → "alta"
      4. Resto de casos → "media"
    """
    for i, c in enumerate(candidates):
        score = c["final_score"]
        n_skills = c.get("skill_count", 0)

        if score < MIN_FINAL_SCORE or n_skills < MIN_SKILLS_FOR_CONFIDENCE:
            c["confidence"] = "baja"
        elif i == 0 and len(candidates) >= 2:
            gap = score - candidates[1]["final_score"]
            c["confidence"] = "alta" if gap >= MIN_SCORE_GAP_FOR_STRONG_RECOMMENDATION else "media"
        else:
            c["confidence"] = "media"

    return candidates


def rank_candidates(
    blocks: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Rankea usuarios basándose en los bloques recuperados.

    Args:
        blocks: Lista de bloques de retrieve_blocks(), con campos
                username, skill, composite_score, semantic_similarity, etc.
        top_k:  Número máximo de candidatos a devolver.

    Returns:
        Lista de hasta top_k candidatos, cada uno con:
          - username
          - final_score        [0, 1]
          - matched_skills     lista de skills ordenadas por combined_score
          - skill_count        número de skills distintas que coinciden
          - block_count        total de bloques recuperados para este usuario
    """
    if not blocks:
        return []

    # Agrupar bloques por usuario
    user_blocks: Dict[str, List[Dict]] = defaultdict(list)
    for block in blocks:
        username = block.get("username", "unknown")
        user_blocks[username].append(block)

    candidates: List[Dict[str, Any]] = []

    for username, user_blks in user_blocks.items():
        # Por skill: quedarse con el bloque de mayor combined_score
        best_per_skill: Dict[str, Dict] = {}

        for b in user_blks:
            skill = b.get("skill", "")
            if not skill:
                continue

            combined = round(
                b.get("semantic_similarity", 0.0) * SEMANTIC_WEIGHT
                + b.get("composite_score", 0.0) * COMPOSITE_WEIGHT,
                4,
            )

            if skill not in best_per_skill or combined > best_per_skill[skill]["combined_score"]:
                best_per_skill[skill] = {
                    "skill": skill,
                    "combined_score": combined,
                    "semantic_similarity": round(b.get("semantic_similarity", 0.0), 4),
                    "composite_score": round(b.get("composite_score", 0.0), 4),
                    "explanation": b.get("explanation", ""),
                    "evidence_fragment": b.get("evidence_fragment", ""),
                    "artifact_type": b.get("artifact_type", ""),
                    "repo": b.get("repo", ""),
                    "path": b.get("path", ""),
                    "url": b.get("url", ""),
                    "recency":    round(b.get("recency",    0.0), 4),
                    "authorship": round(b.get("authorship", 0.0), 4),
                }

        if not best_per_skill:
            continue

        # Score de usuario: media + bonus de amplitud
        skill_scores = [s["combined_score"] for s in best_per_skill.values()]
        avg_score = sum(skill_scores) / len(skill_scores)

        breadth_bonus = min(len(best_per_skill) * BREADTH_BONUS_PER_SKILL, MAX_BREADTH_BONUS)
        final_score = round(min(avg_score + breadth_bonus, 1.0), 4)

        # Skills ordenadas de mayor a menor combined_score
        matched_skills = sorted(
            best_per_skill.values(),
            key=lambda s: s["combined_score"],
            reverse=True,
        )

        candidates.append({
            "username": username,
            "final_score": final_score,
            "matched_skills": matched_skills,
            "skill_count": len(best_per_skill),
            "block_count": len(user_blks),
        })

    # Ordenar candidatos y devolver top_k
    candidates.sort(key=lambda c: c["final_score"], reverse=True)
    candidates = candidates[:top_k]

    # Estimación de confianza por candidato
    attach_confidence(candidates)

    return candidates
