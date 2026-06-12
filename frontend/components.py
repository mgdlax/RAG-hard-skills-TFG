"""
frontend/components.py — Componentes visuales reutilizables.

Cada función recibe datos estructurados y los renderiza usando
st.markdown con HTML/CSS o widgets nativos de Streamlit.
"""

from __future__ import annotations

import streamlit as st

# ── Helpers internos ─────────────────────────────────────────────────────────

def _skill_badges_html(skills: list[str], variant: str = "light") -> str:
    """Genera el HTML con badges de skills."""
    css_class = "rag-skill-badge-light" if variant == "light" else "rag-skill-badge"
    badges = "".join(f'<span class="{css_class}">{s}</span>' for s in skills)
    return f'<div class="rag-skills-row">{badges}</div>'


def _rank_badge_class(position: int) -> str:
    if position == 1:
        return "gold"
    if position == 2:
        return "silver"
    if position == 3:
        return "bronze"
    return ""


def _score_fill_class(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _score_pct(score: float) -> int:
    return int(min(100, max(0, score * 100)))


# ── Componentes públicos ──────────────────────────────────────────────────────

def render_profile_card(profile: dict) -> None:
    """
    Tarjeta compacta de perfil para mostrar en la barra lateral.

    Args:
        profile: Dict con keys username, repositories_processed,
                 evidences_extracted, skills_detected, status.
    """
    username  = profile.get("username", "?")
    repos     = profile.get("repositories_processed", 0)
    evidences = profile.get("evidences_extracted", 0)
    skills    = profile.get("skills_detected", [])
    is_mock   = profile.get("mock", True)

    initials  = username[:2].upper()
    status_label = "simulado" if is_mock else "indexado"

    skills_html = _skill_badges_html(skills[:6])

    html = f"""
    <div class="rag-profile-card">
        <div class="rag-profile-header">
            <div class="rag-profile-avatar">{initials}</div>
            <div class="rag-profile-username">@{username}</div>
        </div>
        <div class="rag-profile-stats">
            <div class="rag-stat-item">
                <div class="rag-stat-value">{repos}</div>
                <div class="rag-stat-label">repos</div>
            </div>
            <div class="rag-stat-item">
                <div class="rag-stat-value">{evidences}</div>
                <div class="rag-stat-label">evidencias</div>
            </div>
            <div class="rag-stat-item">
                <div class="rag-stat-value">{len(skills)}</div>
                <div class="rag-stat-label">skills</div>
            </div>
        </div>
        {skills_html}
        <div style="margin-top: 0.5rem;">
            <span class="rag-status-pill">
                <span class="rag-status-dot"></span>
                {status_label}
            </span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_ranking(ranking: list[dict]) -> None:
    """
    Muestra el ranking de candidatos como lista con barra de score.

    Args:
        ranking: Lista de dicts con keys username, score, matched_skills.
    """
    if not ranking:
        return

    card_parts: list[str] = []
    for i, candidate in enumerate(ranking, start=1):
        username    = candidate.get("username", "?")
        score       = candidate.get("score", 0.0)
        matched     = candidate.get("matched_skills", [])
        pct         = _score_pct(score)
        fill_class  = _score_fill_class(score)
        badge_class = _rank_badge_class(i)
        top1_class  = "top-1" if i == 1 else ""
        skills_html = _skill_badges_html(matched[:4], variant="light")
        rank_label  = str(i)

        card_parts.append(
            f'<div class="rag-ranking-card {top1_class}">'
            f'<div class="rag-rank-badge {badge_class}">{rank_label}</div>'
            f'<div>'
            f'<div class="rag-candidate-username">@{username}</div>'
            f'<div class="rag-score-row">'
            f'<div class="rag-score-track">'
            f'<div class="rag-score-fill {fill_class}" style="width:{pct}%"></div>'
            f'</div>'
            f'<div class="rag-score-label">{score:.3f}</div>'
            f'</div>'
            f'{skills_html}'
            f'</div>'
            f'</div>'
        )

    cards_html = "".join(card_parts)
    html = (
        '<div class="rag-ranking-section">'
        '<div class="rag-section-title">Ranking de candidatos</div>'
        f'<div class="rag-ranking-list">{cards_html}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_evidences(evidences: list[dict]) -> None:
    """
    Muestra las evidencias recuperadas agrupadas en un expander.

    Args:
        evidences: Lista de dicts con keys username, repo, path,
                   skill, fragment, score.
    """
    if not evidences:
        return

    with st.expander(f"Evidencias utilizadas ({len(evidences)})", expanded=False):
        for ev in evidences:
            username = ev.get("username", "?")
            repo     = ev.get("repo", "")
            path     = ev.get("path", "")
            skill    = ev.get("skill", "")
            fragment = ev.get("fragment", "")
            score    = ev.get("score", 0.0)

            artifact_type = ev.get("artifact_type", "")
            explanation   = ev.get("explanation", "")
            type_label    = artifact_type if artifact_type else "file"

            code_markers = ("def ", "class ", "import ", "from ", "    ", "\n",
                            "= ", "()", "{", "}", "=>", "->", "return ", "self.")
            is_file_artifact = artifact_type in ("file", "")
            looks_like_code = is_file_artifact and fragment and any(
                marker in fragment for marker in code_markers
            )

            if looks_like_code:
                fragment_html = f'<pre class="rag-evidence-code"><code>{fragment}</code></pre>'
            elif fragment:
                fragment_html = f'<p class="rag-evidence-fragment">"{fragment}"</p>'
            else:
                fragment_html = ""

            explanation_html = (
                f'<p class="rag-evidence-explanation">{explanation}</p>'
                if explanation else ""
            )

            html = f"""
            <div class="rag-evidence-card">
                <div class="rag-evidence-meta">
                    <span class="rag-skill-badge-light">{skill}</span>
                    <span class="rag-evidence-repo">@{username} / {repo}</span>
                    <span class="rag-evidence-path">{path}</span>
                    <span class="rag-evidence-type">{type_label}</span>
                    <span class="rag-evidence-score">{score:.3f}</span>
                </div>
                {explanation_html}
                {fragment_html}
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)


def render_welcome_screen() -> None:
    """
    Pantalla de bienvenida cuando no hay mensajes en el chat.
    """
    examples = [
        "Que perfil recomiendas para construir un sistema RAG con LangChain?",
        "Quien tiene mas experiencia desarrollando agentes LLM con herramientas?",
        "Que candidato domina mejor FastAPI y arquitecturas de backend?",
        "Quien recomendarias para un proyecto de busqueda semantica con embeddings?",
    ]
    examples_html = "".join(
        f'<div class="rag-example-query">{q}</div>' for q in examples
    )

    html = f"""
    <div class="rag-welcome-wrapper">
        <div class="rag-welcome-icon">RAG GitHub Skills</div>
        <div class="rag-welcome-title">Recomendacion tecnica basada en evidencias</div>
        <div class="rag-welcome-desc">
            Analiza perfiles publicos de GitHub, extrae evidencias tecnicas de repositorios
            y recomienda que desarrollador encaja mejor con tu proyecto.
            Las respuestas se generan a partir de evidencias reales: codigo, commits, issues y pull requests.
        </div>
        <div class="rag-examples-title">Ejemplos de consultas</div>
        <div class="rag-examples-grid">
            {examples_html}
        </div>
        <div class="rag-welcome-hint">
            Añade uno o mas perfiles de GitHub en la barra lateral para empezar.
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_chat_disabled_hint() -> None:
    """Muestra un aviso cuando el chat esta deshabilitado por falta de perfiles."""
    html = """
    <div class="rag-disabled-hint">
        Añade al menos un perfil de GitHub para poder hacer preguntas.
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
