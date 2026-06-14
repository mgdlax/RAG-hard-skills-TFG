"""
frontend/styles.py — Estilos CSS globales de la aplicación.

Inyecta un bloque <style> en el DOM de Streamlit con clases
personalizadas para todos los componentes visuales del frontend.
"""

import streamlit as st

# Paleta corporativa — austera, sin colores vivos
PRIMARY       = "#1e3a5f"
PRIMARY_LIGHT = "#e8eef6"
PRIMARY_DARK  = "#132844"
ACCENT        = "#2d5a8e"
BG_MAIN       = "#f4f5f7"
BG_CARD       = "#ffffff"
TEXT_MAIN     = "#1a1d23"
TEXT_MUTED    = "#6b7280"
BORDER        = "#dde1e7"
BORDER_LIGHT  = "#eaecf0"
SUCCESS       = "#2e7d52"
SUCCESS_LIGHT = "#e8f4ee"
WARNING       = "#8a6d00"
WARNING_LIGHT = "#faf3dc"
DANGER        = "#c0392b"

_CSS = f"""
<style>
/* ═══════════════════════════════════════════════════════
   TIPOGRAFÍA BASE
═══════════════════════════════════════════════════════ */
* {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}}

/* La regla universal anterior pisaria la fuente de los iconos de Streamlit
   (flecha de colapsar la barra lateral, flechas de los expanders), que entonces
   se mostrarian como texto literal del ligature. Se restaura su fuente. */
span[data-testid="stIconMaterial"],
[data-testid="stIconMaterial"],
.material-icons, .material-icons-outlined, .material-icons-round,
.material-symbols-rounded, .material-symbols-outlined, .material-symbols-sharp {{
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
}}

/* ═══════════════════════════════════════════════════════
   GLOBAL
═══════════════════════════════════════════════════════ */
.stApp {{
    background-color: {BG_MAIN};
}}

[data-testid="stHeader"] {{
    background-color: {BG_MAIN};
}}

/* ═══════════════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {{
    background: #1a2030;
    border-right: 1px solid #252e42;
}}
section[data-testid="stSidebar"] * {{
    color: #c8cdd6 !important;
}}
section[data-testid="stSidebar"] .stTextInput input {{
    background-color: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #e8eaed !important;
    border-radius: 4px !important;
}}
section[data-testid="stSidebar"] .stTextInput input::placeholder {{
    color: rgba(255,255,255,0.3) !important;
}}
section[data-testid="stSidebar"] label {{
    color: #8b95a3 !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    text-transform: none !important;
}}
section[data-testid="stSidebar"] .stButton > button {{
    background: {PRIMARY} !important;
    color: #e8eef6 !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    transition: background 0.15s ease !important;
    box-shadow: none !important;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: {ACCENT} !important;
    transform: none !important;
    box-shadow: none !important;
}}
section[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.08) !important;
}}

/* ═══════════════════════════════════════════════════════
   HEADER
═══════════════════════════════════════════════════════ */
.rag-header {{
    padding: 1.5rem 0 1.25rem 0;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 1.5rem;
}}
.rag-header-title {{
    font-size: 1.6rem;
    font-weight: 600;
    color: {TEXT_MAIN};
    margin: 0;
    line-height: 1.2;
    letter-spacing: -0.02em;
}}
.rag-header-subtitle {{
    color: {TEXT_MUTED};
    font-size: 0.88rem;
    margin-top: 0.35rem;
    line-height: 1.5;
}}
.rag-header-badges {{
    display: flex;
    gap: 0.4rem;
    margin-top: 0.75rem;
    flex-wrap: wrap;
}}
.rag-badge-chip {{
    background: {BG_CARD};
    color: {TEXT_MUTED};
    padding: 0.18rem 0.6rem;
    border: 1px solid {BORDER};
    border-radius: 3px;
    font-size: 0.7rem;
    font-weight: 400;
    letter-spacing: 0.01em;
}}

/* ═══════════════════════════════════════════════════════
   PROFILE CARDS (sidebar)
═══════════════════════════════════════════════════════ */
.rag-profile-card {{
    background: rgba(255,255,255,0.05);
    border-top: 1px solid rgba(255,255,255,0.08);
    padding: 0.6rem 0;
    margin-bottom: 0.25rem;
}}
.rag-profile-header {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}}
.rag-profile-avatar {{
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: {PRIMARY};
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.65rem;
    font-weight: 600;
    color: #c8d8ea;
    flex-shrink: 0;
}}
.rag-profile-username {{
    font-weight: 600;
    font-size: 0.85rem;
    color: #e0e4ea !important;
}}
.rag-profile-stats {{
    display: flex;
    gap: 1.2rem;
    margin-bottom: 0.5rem;
}}
.rag-stat-item {{
    text-align: left;
}}
.rag-stat-value {{
    font-size: 0.9rem;
    font-weight: 600;
    color: #9db3c8 !important;
}}
.rag-stat-label {{
    font-size: 0.63rem;
    color: rgba(255,255,255,0.35) !important;
    letter-spacing: 0.02em;
}}
.rag-status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(46,125,82,0.15);
    color: #6dbf8a !important;
    border-radius: 3px;
    padding: 0.18rem 0.5rem;
    font-size: 0.68rem;
    font-weight: 500;
}}
.rag-status-dot {{
    width: 5px;
    height: 5px;
    background: #6dbf8a;
    border-radius: 50%;
    animation: pulse 3s infinite;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.3; }}
}}

/* ═══════════════════════════════════════════════════════
   SKILL BADGES
═══════════════════════════════════════════════════════ */
.rag-skills-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    margin-top: 0.35rem;
}}
.rag-skill-badge {{
    background: rgba(30,58,95,0.3);
    color: #9db3c8 !important;
    border: 1px solid rgba(45,90,142,0.35);
    border-radius: 3px;
    padding: 0.12rem 0.4rem;
    font-size: 0.66rem;
    font-weight: 500;
    white-space: nowrap;
}}
.rag-skill-badge-light {{
    background: {PRIMARY_LIGHT};
    color: {PRIMARY} !important;
    border: 1px solid rgba(30,58,95,0.12);
    border-radius: 3px;
    padding: 0.15rem 0.45rem;
    font-size: 0.7rem;
    font-weight: 500;
    white-space: nowrap;
}}

/* ═══════════════════════════════════════════════════════
   SIDEBAR METRICS
═══════════════════════════════════════════════════════ */
.rag-metrics-section {{
    margin-top: 0.5rem;
}}
.rag-metrics-title {{
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    color: rgba(255,255,255,0.3) !important;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
}}
.rag-metrics-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.4rem;
}}
.rag-metric-box {{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 4px;
    padding: 0.5rem 0.6rem;
    text-align: center;
}}
.rag-metric-value {{
    font-size: 1.2rem;
    font-weight: 600;
    color: #9db3c8 !important;
    line-height: 1.1;
}}
.rag-metric-label {{
    font-size: 0.6rem;
    color: rgba(255,255,255,0.3) !important;
    margin-top: 0.1rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}
.rag-chromadb-status {{
    background: rgba(46,125,82,0.1);
    border: 1px solid rgba(46,125,82,0.2);
    border-radius: 4px;
    padding: 0.4rem 0.65rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.4rem;
}}
.rag-chromadb-dot {{
    width: 7px;
    height: 7px;
    background: {SUCCESS};
    border-radius: 50%;
    flex-shrink: 0;
}}
.rag-chromadb-text {{
    font-size: 0.7rem;
    color: #6dbf8a !important;
}}

/* ═══════════════════════════════════════════════════════
   WELCOME SCREEN
═══════════════════════════════════════════════════════ */
.rag-welcome-wrapper {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem 1rem;
    text-align: center;
    max-width: 680px;
    margin: 0 auto;
}}
.rag-welcome-icon {{
    font-size: 2rem;
    margin-bottom: 1rem;
    color: {PRIMARY};
    font-weight: 300;
    letter-spacing: -0.02em;
}}
.rag-welcome-title {{
    font-size: 1.4rem;
    font-weight: 600;
    color: {TEXT_MAIN};
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}}
.rag-welcome-desc {{
    color: {TEXT_MUTED};
    font-size: 0.88rem;
    line-height: 1.65;
    margin-bottom: 2rem;
    max-width: 520px;
}}
.rag-examples-title {{
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.07em;
    color: {TEXT_MUTED};
    margin-bottom: 0.75rem;
    text-transform: uppercase;
}}
.rag-examples-grid {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.4rem;
    width: 100%;
    max-width: 540px;
    margin-bottom: 2rem;
}}
.rag-example-query {{
    background: {BG_CARD};
    border-left: 2px solid {BORDER};
    padding: 0.65rem 0.9rem;
    font-size: 0.85rem;
    color: {TEXT_MAIN};
    text-align: left;
    cursor: default;
    transition: border-color 0.15s;
}}
.rag-example-query:hover {{
    border-left-color: {PRIMARY};
}}
.rag-welcome-hint {{
    background: {PRIMARY_LIGHT};
    border-left: 2px solid {PRIMARY};
    padding: 0.65rem 1rem;
    font-size: 0.83rem;
    color: {PRIMARY};
    font-weight: 400;
    text-align: left;
    width: 100%;
    max-width: 540px;
}}

/* ═══════════════════════════════════════════════════════
   BARRA DE PROGRESO PERSONALIZADA
═══════════════════════════════════════════════════════ */
.rag-progress-container {{
    padding: 1rem 0 0.75rem 0;
    width: 100%;
}}
.rag-progress-track {{
    width: 100%;
    height: 3px;
    background: {BORDER};
    border-radius: 0;
    overflow: hidden;
    margin-bottom: 0.6rem;
}}
.rag-progress-fill {{
    height: 100%;
    background: {PRIMARY};
    border-radius: 0;
    transition: width 0.4s ease;
}}
.rag-progress-label {{
    font-size: 0.78rem;
    color: {TEXT_MUTED};
    font-weight: 400;
    letter-spacing: 0;
}}
.rag-progress-step {{
    font-size: 0.72rem;
    color: rgba(107,114,128,0.6);
    margin-top: 0.2rem;
}}

/* ═══════════════════════════════════════════════════════
   RANKING DE CANDIDATOS
═══════════════════════════════════════════════════════ */
.rag-ranking-section {{
    margin-top: 1rem;
}}
.rag-section-title {{
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: {TEXT_MUTED};
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid {BORDER_LIGHT};
}}
.rag-ranking-list {{
    display: flex;
    flex-direction: column;
    gap: 0;
}}
.rag-ranking-card {{
    display: grid;
    grid-template-columns: 1.8rem 1fr auto;
    align-items: start;
    gap: 0.75rem;
    background: transparent;
    border-bottom: 1px solid {BORDER_LIGHT};
    padding: 0.7rem 0;
    transition: background 0.15s;
}}
.rag-ranking-card:first-child {{
    border-top: 1px solid {BORDER_LIGHT};
}}
.rag-ranking-card.top-1 {{
    background: {PRIMARY_LIGHT};
    padding: 0.7rem 0.75rem;
    margin: 0 -0.75rem;
}}
.rag-rank-badge {{
    width: 1.8rem;
    height: 1.8rem;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.78rem;
    font-weight: 600;
    color: {TEXT_MUTED};
    flex-shrink: 0;
    margin-top: 0.05rem;
}}
.rag-rank-badge.gold   {{ color: {PRIMARY}; font-weight: 700; }}
.rag-rank-badge.silver {{ color: {TEXT_MAIN}; }}
.rag-rank-badge.bronze {{ color: {TEXT_MUTED}; }}
.rag-candidate-username {{
    font-weight: 600;
    font-size: 0.88rem;
    color: {TEXT_MAIN};
    margin-bottom: 0.3rem;
}}
.rag-score-row {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
}}
.rag-score-track {{
    flex: 1;
    height: 3px;
    background: {BORDER};
    border-radius: 0;
    overflow: hidden;
}}
.rag-score-fill {{
    height: 100%;
    background: {PRIMARY};
    border-radius: 0;
}}
.rag-score-fill.high   {{ background: {SUCCESS}; }}
.rag-score-fill.medium {{ background: {PRIMARY}; }}
.rag-score-fill.low    {{ background: {TEXT_MUTED}; }}
.rag-score-label {{
    font-size: 0.75rem;
    font-weight: 600;
    color: {TEXT_MAIN};
    min-width: 2.8rem;
    text-align: right;
}}

/* ═══════════════════════════════════════════════════════
   EVIDENCIAS
═══════════════════════════════════════════════════════ */
.rag-evidence-section {{
    margin-top: 0.75rem;
}}
.rag-evidence-card {{
    background: transparent;
    border-left: 2px solid {BORDER};
    padding: 0.6rem 0.85rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.15s;
}}
.rag-evidence-card:hover {{
    border-left-color: {PRIMARY};
}}
.rag-evidence-meta {{
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 0.35rem;
}}
.rag-evidence-repo {{
    font-size: 0.76rem;
    color: {TEXT_MUTED};
    font-family: "SFMono-Regular", Consolas, monospace;
}}
.rag-evidence-path {{
    font-size: 0.73rem;
    color: {PRIMARY};
    font-family: "SFMono-Regular", Consolas, monospace;
    background: {PRIMARY_LIGHT};
    padding: 0.08rem 0.35rem;
    border-radius: 2px;
}}
.rag-evidence-score {{
    margin-left: auto;
    font-size: 0.7rem;
    font-weight: 600;
    color: {SUCCESS};
    background: {SUCCESS_LIGHT};
    padding: 0.12rem 0.45rem;
    border-radius: 2px;
}}
.rag-evidence-explanation {{
    font-size: 0.81rem;
    color: {TEXT_MUTED};
    line-height: 1.5;
    margin: 0.15rem 0 0.25rem 0;
}}
.rag-evidence-fragment {{
    font-size: 0.82rem;
    color: {TEXT_MAIN};
    line-height: 1.5;
    font-style: italic;
    margin: 0;
}}
.rag-evidence-code {{
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 3px;
    padding: 0.6rem 0.8rem;
    margin: 0.35rem 0 0 0;
    overflow-x: auto;
    font-size: 0.77rem;
    line-height: 1.55;
}}
.rag-evidence-code code {{
    color: #c9d1d9;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    white-space: pre;
}}
.rag-evidence-type {{
    font-size: 0.68rem;
    color: {TEXT_MUTED};
    background: {BG_MAIN};
    padding: 0.08rem 0.35rem;
    border-radius: 2px;
    font-family: monospace;
    border: 1px solid {BORDER};
}}

/* ═══════════════════════════════════════════════════════
   CHAT OVERRIDES
═══════════════════════════════════════════════════════ */
[data-testid="stChatMessage"] {{
    border-radius: 4px !important;
    margin-bottom: 0.4rem !important;
}}
[data-testid="stChatInput"] textarea {{
    border-radius: 4px !important;
    border-color: {BORDER} !important;
    font-size: 0.91rem !important;
}}
[data-testid="stChatInput"] textarea:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 2px rgba(30,58,95,0.1) !important;
}}

/* ═══════════════════════════════════════════════════════
   EXPANDERS
═══════════════════════════════════════════════════════ */
.streamlit-expanderHeader {{
    background: transparent !important;
    border-radius: 0 !important;
    font-size: 0.83rem !important;
    border-bottom: 1px solid {BORDER_LIGHT} !important;
}}

/* ═══════════════════════════════════════════════════════
   STATUS WIDGET (ocultar el widget nativo)
═══════════════════════════════════════════════════════ */
[data-testid="stStatusWidget"] {{
    border-radius: 4px !important;
    font-size: 0.83rem !important;
}}

/* ═══════════════════════════════════════════════════════
   DIVIDER
═══════════════════════════════════════════════════════ */
.rag-section-divider {{
    height: 1px;
    background: {BORDER_LIGHT};
    margin: 0.75rem 0;
}}

/* ═══════════════════════════════════════════════════════
   DISABLED INPUT HINT
═══════════════════════════════════════════════════════ */
.rag-disabled-hint {{
    background: {WARNING_LIGHT};
    border-left: 2px solid {WARNING};
    padding: 0.55rem 0.85rem;
    font-size: 0.82rem;
    color: #5a4700;
    text-align: left;
    margin-bottom: 0.5rem;
}}
</style>
"""


def inject_custom_css() -> None:
    """Inyecta los estilos CSS personalizados en el DOM de Streamlit."""
    st.markdown(_CSS, unsafe_allow_html=True)
