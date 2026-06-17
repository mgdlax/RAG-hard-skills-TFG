"""
app.py — Punto de entrada de la aplicación RAG GitHub Skills.

Responsabilidades:
  - Configurar la página y el estado de sesión.
  - Coordinar la barra lateral (añadir y listar perfiles).
  - Coordinar el área principal (chat RAG con historial).
  - Delegar lógica de negocio a services/ y componentes a frontend/.

Ejecución:
    streamlit run app.py
"""

import streamlit as st

from frontend.styles import inject_custom_css
from frontend.layout import render_header, render_sidebar_header, render_sidebar_metrics
from frontend.components import (
    render_profile_card,
    render_ranking,
    render_evidences,
    render_welcome_screen,
    render_chat_disabled_hint,
)
from services.profile_service import (
    add_profile,
    load_indexed_profiles,
    update_profiles,
    PIPELINE_STEPS,
)
from services.rag_service import ask_question, RAG_STEPS

# Configuración de página
st.set_page_config(
    page_title="RAG GitHub Skills",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

# Estado de sesión
_DEFAULTS: dict = {
    "messages":       [],
    "profiles":       [],
    "last_ranking":   [],
    "last_evidences": [],
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Precargar perfiles ya indexados en ChromaDB para habilitar el chat
# sin tener que re-ejecutar el pipeline en cada sesión.
if not st.session_state.profiles:
    st.session_state.profiles = load_indexed_profiles()


# Helper: barra de progreso personalizada

def _render_progress(step_index: int, total_steps: int, label: str) -> None:
    """Actualiza la barra de progreso y el texto descriptivo."""
    pct = int(min(100, (step_index / total_steps) * 100))
    st.markdown(
        f"""
        <div class="rag-progress-container">
            <div class="rag-progress-track">
                <div class="rag-progress-fill" style="width:{pct}%"></div>
            </div>
            <div class="rag-progress-label">{label}</div>
            <div class="rag-progress-step">Paso {step_index} de {total_steps}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# BARRA LATERAL
with st.sidebar:
    render_sidebar_header()
    st.markdown("---")

    # Añadir perfil
    st.markdown(
        '<div style="font-size:0.73rem;font-weight:500;letter-spacing:0.04em;'
        'color:#8b95a3;margin-bottom:0.4rem;text-transform:uppercase;">'
        "Añadir perfil de GitHub</div>",
        unsafe_allow_html=True,
    )

    github_user = st.text_input(
        "Usuario de GitHub",
        placeholder="ej. hwchase17",
        label_visibility="collapsed",
        key="github_user_input",
    )

    add_btn = st.button(
        "Añadir perfil",
        use_container_width=True,
        type="primary",
    )

    # Procesamiento al pulsar el botón
    if add_btn:
        username = github_user.strip()
        if not username:
            st.warning("Introduce un nombre de usuario de GitHub.")
        elif any(p["username"] == username for p in st.session_state.profiles):
            st.info(f"@{username} ya está procesado.")
        else:
            profile_result = None
            total = len(PIPELINE_STEPS)
            progress_slot = st.empty()

            for step_idx, (step_label, is_final, data) in enumerate(
                add_profile(username), start=1
            ):
                with progress_slot.container():
                    _render_progress(step_idx, total, step_label)

                if is_final:
                    profile_result = data
                    with progress_slot.container():
                        if data:
                            _render_progress(total, total, "Perfil indexado correctamente.")
                        else:
                            st.markdown(
                                '<div class="rag-disabled-hint">No se pudo procesar el perfil.</div>',
                                unsafe_allow_html=True,
                            )

            if profile_result:
                st.session_state.profiles.append(profile_result)
                progress_slot.empty()

    st.markdown("---")

    # Lista de perfiles procesados
    if st.session_state.profiles:
        st.markdown(
            '<div style="font-size:0.73rem;font-weight:500;letter-spacing:0.04em;'
            'color:#8b95a3;margin-bottom:0.6rem;text-transform:uppercase;">'
            f"Perfiles ({len(st.session_state.profiles)})</div>",
            unsafe_allow_html=True,
        )
        for profile in st.session_state.profiles:
            with st.expander(f"@{profile['username']}", expanded=False):
                render_profile_card(profile)

        # Botón para buscar cambios en GitHub y re-indexar lo que tenga actividad
        if st.button("Buscar cambios y actualizar", use_container_width=True):
            progress_slot = st.empty()
            resumen = None

            for mensaje, is_final, data in update_profiles():
                progress_slot.info(mensaje)
                if is_final:
                    resumen = data

            progress_slot.empty()
            if resumen is None:
                st.warning("No se pudieron actualizar los perfiles.")
            else:
                # Recargar los perfiles para reflejar los datos actualizados
                st.session_state.profiles = load_indexed_profiles()
                st.success(
                    f"{len(resumen['updated'])} actualizados, "
                    f"{len(resumen['skipped'])} sin cambios."
                )
                st.rerun()
    else:
        st.markdown(
            '<div style="font-size:0.78rem;color:rgba(255,255,255,0.28);'
            'padding:0.6rem 0;">Sin perfiles cargados.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Métricas del sistema
    render_sidebar_metrics(st.session_state.profiles)

    # Botón para limpiar el chat
    if st.session_state.messages:
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        if st.button("Limpiar conversacion", use_container_width=True):
            st.session_state.messages       = []
            st.session_state.last_ranking   = []
            st.session_state.last_evidences = []
            st.rerun()


# ÁREA PRINCIPAL
render_header()

# Historial de mensajes
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("ranking"):
                render_ranking(msg["ranking"])
            if msg.get("evidences"):
                render_evidences(msg["evidences"])

# Pantalla de bienvenida
if not st.session_state.messages:
    render_welcome_screen(has_profiles=bool(st.session_state.profiles))

# Aviso si el chat está deshabilitado
if not st.session_state.profiles:
    render_chat_disabled_hint()

# Input del chat
prompt = st.chat_input(
    "Haz una pregunta sobre los perfiles técnicos cargados...",
    disabled=not st.session_state.profiles,
)

if prompt:
    # 1. Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Añadir al historial
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Generar y mostrar respuesta del asistente
    with st.chat_message("assistant"):
        rag_result = None
        total = len(RAG_STEPS)
        progress_slot = st.empty()

        for step_idx, (step_label, is_final, data) in enumerate(
            ask_question(prompt, st.session_state.profiles), start=1
        ):
            with progress_slot.container():
                _render_progress(step_idx, total, step_label)

            if is_final:
                rag_result = data
                if data:
                    with progress_slot.container():
                        _render_progress(total, total, "Respuesta generada.")
                else:
                    with progress_slot.container():
                        st.markdown(
                            '<div class="rag-disabled-hint">Error en el pipeline RAG.</div>',
                            unsafe_allow_html=True,
                        )

        if rag_result:
            progress_slot.empty()
            st.markdown(rag_result["answer"])
            render_ranking(rag_result["ranking"])
            render_evidences(rag_result["evidences"])

    # 4. Guardar en el historial
    if rag_result:
        st.session_state.messages.append({
            "role":      "assistant",
            "content":   rag_result["answer"],
            "ranking":   rag_result["ranking"],
            "evidences": rag_result["evidences"],
        })
        st.session_state.last_ranking   = rag_result["ranking"]
        st.session_state.last_evidences = rag_result["evidences"]
