"""Live provider configuration and runtime health diagnostics."""

from __future__ import annotations

import streamlit as st
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.config import get_settings, redacted_database_url, update_env_file
from src.ui.components import (
    configure_page,
    page_header,
    render_database_error,
    render_flash,
    section_header,
    set_flash,
    status_badge_html,
)
from src.ui.data import CACHE_TTL_SECONDS, clear_ui_data_caches, get_ui_session_factory


def _index(options: list[str], value: str | None) -> int:
    cleaned = (value or "").lower()
    return options.index(cleaned) if cleaned in options else 0


def _provider_status(label: str, ready: bool) -> None:
    st.markdown(
        status_badge_html(
            f"{label}: {'Ready' if ready else 'Setup required'}",
            "good" if ready else "warn",
        ),
        unsafe_allow_html=True,
    )


def main() -> None:
    settings = get_settings()
    configure_page("Settings", settings)
    page_header(
        "Settings",
        "Manage provider readiness, runtime controls, and data connectivity.",
        eyebrow="Workspace settings",
    )
    render_flash()

    SessionFactory = get_ui_session_factory(settings.database_url)
    try:
        with st.spinner("Checking database health..."):
            with SessionFactory() as session:
                session.execute(text("SELECT 1"))
        database_state = "Connected"
        database_tone = "good"
    except SQLAlchemyError:
        database_state = "Unavailable"
        database_tone = "risk"

    environment, database, cache, mode = st.columns(4)
    environment.metric("Environment", settings.app_env.title())
    database.metric("Database", settings.database_url.split(":", 1)[0].upper())
    cache.metric("View cache", f"{CACHE_TTL_SECONDS}s")
    mode.metric(
        "Runtime mode",
        "Demo" if settings.demo_mode else ("Live" if settings.live_ready else "Setup"),
    )

    section_header(
        "Provider readiness",
        "Discovery uses local analysis and public community APIs when paid providers are unavailable.",
    )
    live_extraction_ready = bool(
        not settings.demo_mode and settings.discovery_ready
    )
    live_embedding_ready = bool(not settings.demo_mode and settings.embedding_ready)
    live_search_ready = settings.public_search_ready
    llm, embeddings, search, reddit = st.columns(4)
    with llm:
        _provider_status("Extraction", live_extraction_ready)
    with embeddings:
        _provider_status("Embeddings", live_embedding_ready)
    with search:
        _provider_status("Public search", live_search_ready)
    with reddit:
        _provider_status("Reddit (optional)", settings.reddit_ready)

    section_header("Database", "The active storage connection for this workspace.")
    status_column, address_column = st.columns([1, 4])
    status_column.markdown(
        status_badge_html(database_state, database_tone), unsafe_allow_html=True
    )
    address_column.text_input(
        "Configured database",
        value=redacted_database_url(settings.database_url),
        disabled=True,
    )
    if database_state == "Unavailable":
        render_database_error("Database health", settings)

    section_header(
        "Configuration",
        "Changes are stored locally in the ignored environment file.",
    )
    with st.form("runtime-configuration"):
        app_env = st.selectbox(
            "Environment",
            ["production", "development", "test"],
            index=_index(["production", "development", "test"], settings.app_env),
        )
        demo_mode = st.toggle("Demo mode", value=settings.demo_mode)

        st.markdown("**Language model**")
        first, second = st.columns(2)
        llm_provider = first.selectbox(
            "LLM provider",
            ["local", "openai", "mock"],
            index=_index(["local", "openai", "mock"], settings.llm_provider),
        )
        llm_model = second.text_input("LLM model", value=settings.llm_model)
        llm_api_key = first.text_input(
            "OpenAI API key",
            type="password",
            placeholder="Configured"
            if settings.llm_api_key
            else "Optional; local analysis remains available",
        )
        clear_llm_key = second.checkbox("Clear stored OpenAI key")

        st.markdown("**Embeddings and research**")
        first, second = st.columns(2)
        embedding_provider = first.selectbox(
            "Embedding provider",
            ["openai", "sentence_transformers", "deterministic"],
            index=_index(
                ["openai", "sentence_transformers", "deterministic"],
                settings.embedding_provider,
            ),
        )
        embedding_model = second.text_input(
            "Embedding model", value=settings.embedding_model
        )
        search_provider = first.selectbox(
            "Search provider",
            ["community", "tavily", "mock"],
            index=_index(
                ["community", "tavily", "mock"], settings.search_provider
            ),
        )
        search_api_key = second.text_input(
            "Tavily API key",
            type="password",
            placeholder="Configured"
            if settings.search_api_key
            else "Optional; community search remains available",
        )
        clear_search_key = second.checkbox("Clear stored Tavily key")

        st.markdown("**Reddit OAuth**")
        first, second = st.columns(2)
        reddit_client_id = first.text_input(
            "Client ID", value=settings.reddit_client_id or ""
        )
        reddit_client_secret = second.text_input(
            "Client secret",
            type="password",
            placeholder=(
                "Configured"
                if settings.reddit_client_secret
                else "Required for Reddit intake"
            ),
        )
        reddit_user_agent = first.text_input(
            "User agent", value=settings.reddit_user_agent
        )
        clear_reddit_secret = second.checkbox("Clear stored Reddit secret")

        with st.expander("Advanced runtime settings"):
            database_url = st.text_input(
                "New database URL",
                type="password",
                placeholder="Leave blank to keep the configured database",
            )
            first, second = st.columns(2)
            cluster_threshold = first.slider(
                "Clustering threshold",
                min_value=0.0,
                max_value=1.0,
                value=settings.cluster_similarity_threshold,
                step=0.01,
            )
            extraction_threshold = second.slider(
                "Minimum extraction confidence",
                min_value=0.0,
                max_value=1.0,
                value=settings.minimum_extraction_confidence,
                step=0.01,
            )
            max_results = first.number_input(
                "Maximum search results",
                min_value=1,
                max_value=100,
                value=settings.max_search_results,
            )
            search_depth = second.selectbox(
                "Search depth",
                ["basic", "advanced", "fast", "ultra-fast"],
                index=_index(
                    ["basic", "advanced", "fast", "ultra-fast"],
                    settings.search_depth,
                ),
            )

        saved = st.form_submit_button(
            "Save configuration", type="primary", use_container_width=True
        )

    if saved:
        updates = {
            "APP_ENV": app_env,
            "DEMO_MODE": demo_mode,
            "LLM_PROVIDER": llm_provider,
            "LLM_MODEL": llm_model.strip(),
            "EMBEDDING_PROVIDER": embedding_provider,
            "EMBEDDING_MODEL": embedding_model.strip(),
            "SEARCH_PROVIDER": search_provider,
            "SEARCH_DEPTH": search_depth,
            "CLUSTER_SIMILARITY_THRESHOLD": cluster_threshold,
            "MINIMUM_EXTRACTION_CONFIDENCE": extraction_threshold,
            "MAX_SEARCH_RESULTS": int(max_results),
            "REDDIT_CLIENT_ID": reddit_client_id.strip(),
            "REDDIT_USER_AGENT": reddit_user_agent.strip(),
        }
        if database_url.strip():
            updates["DATABASE_URL"] = database_url.strip()
        if llm_api_key.strip() or clear_llm_key:
            updates["LLM_API_KEY"] = "" if clear_llm_key else llm_api_key.strip()
        if search_api_key.strip() or clear_search_key:
            updates["SEARCH_API_KEY"] = (
                "" if clear_search_key else search_api_key.strip()
            )
        if reddit_client_secret.strip() or clear_reddit_secret:
            updates["REDDIT_CLIENT_SECRET"] = (
                "" if clear_reddit_secret else reddit_client_secret.strip()
            )
        try:
            update_env_file(updates)
            get_settings.cache_clear()
            get_ui_session_factory.clear()
            clear_ui_data_caches()
            set_flash("Configuration saved. Provider readiness has been refreshed.")
            st.rerun()
        except (OSError, ValueError) as exc:
            st.error(f"Configuration could not be saved: {exc}")

    section_header("Cached views")
    if st.button("Clear cached views", use_container_width=True):
        clear_ui_data_caches()
        st.success("Cached dashboard, opportunity, and evidence views were cleared.")


if __name__ == "__main__":
    main()
