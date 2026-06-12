"""
frontend/layout.py — Funciones de layout de alto nivel.

Gestiona la cabecera de la app y los bloques estructurales
de la barra lateral (logo, título, métricas del sistema).
"""

from __future__ import annotations

import streamlit as st


# ── Cabecera principal ────────────────────────────────────────────────────────

def render_header() -> None:
    """
    Renderiza el título, subtítulo y badges de tecnología
    en la parte superior del área principal.
    """
    tech_chips = ["Python", "Streamlit", "ChromaDB", "Sentence-Transformers", "Ollama"]
    chips_html = "".join(
        f'<span class="rag-badge-chip">{t}</span>' for t in tech_chips
    )

    html = f"""
    <div class="rag-header">
        <h1 class="rag-header-title">RAG GitHub Skills</h1>
        <div class="rag-header-subtitle">
            Sistema de recomendacion tecnica basado en evidencias de GitHub.
            Analiza repositorios publicos, detecta hard-skills y encuentra
            el perfil mas adecuado para tu proyecto.
        </div>
        <div class="rag-header-badges">{chips_html}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ── Barra lateral ─────────────────────────────────────────────────────────────

def render_sidebar_header() -> None:
    """Logo y título de la barra lateral."""
    html = """
    <div style="padding: 1rem 0 0.5rem 0;">
        <div style="font-size:0.95rem; font-weight:600; color:#c8cdd6; letter-spacing:-0.01em;">
            RAG GitHub Skills
        </div>
        <div style="font-size:0.68rem; color:rgba(255,255,255,0.3); margin-top:0.15rem;">
            TFG — Ingenieria del Software
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_sidebar_metrics(profiles: list[dict]) -> None:
    """
    Muestra métricas del sistema al pie de la barra lateral.

    Args:
        profiles: Lista de perfiles actualmente en session_state.
    """
    n_profiles  = len(profiles)
    n_evidences = sum(p.get("evidences_extracted", 0) for p in profiles)
    has_real    = any(not p.get("mock", True) for p in profiles)
    chroma_status = "activo" if has_real else "simulado"
    chroma_color  = "#6dbf8a" if has_real else "#d4b44a"

    html = f"""
    <div class="rag-metrics-section">
        <div class="rag-metrics-title">Estado del sistema</div>
        <div class="rag-metrics-grid">
            <div class="rag-metric-box">
                <div class="rag-metric-value">{n_profiles}</div>
                <div class="rag-metric-label">Perfiles</div>
            </div>
            <div class="rag-metric-box">
                <div class="rag-metric-value">{n_evidences}</div>
                <div class="rag-metric-label">Evidencias</div>
            </div>
        </div>
        <div class="rag-chromadb-status" style="margin-top:0.5rem;">
            <div class="rag-chromadb-dot" style="background:{chroma_color};"></div>
            <div class="rag-chromadb-text">
                ChromaDB: <strong>{chroma_status}</strong>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
