"""Settings page shell for API and scoring configuration."""

import streamlit as st

from src.config import get_settings, redacted_database_url


settings = get_settings()

st.set_page_config(page_title="InSift Settings", page_icon="IS", layout="wide")
st.title("Settings")
st.text_input("Database URL", value=redacted_database_url(settings.database_url), disabled=True)
st.toggle("Demo mode", value=settings.demo_mode, disabled=True)
st.slider(
    "Clustering threshold",
    min_value=0.0,
    max_value=1.0,
    value=settings.cluster_similarity_threshold,
    disabled=True,
)
st.slider(
    "Minimum extraction confidence",
    min_value=0.0,
    max_value=1.0,
    value=settings.minimum_extraction_confidence,
    disabled=True,
)
st.text_input("Embedding provider", value=settings.embedding_provider, disabled=True)
st.number_input(
    "Maximum search results",
    min_value=1,
    max_value=100,
    value=settings.max_search_results,
    disabled=True,
)
st.text_input("Search provider", value=settings.search_provider or "mock", disabled=True)
st.text_input("Search depth", value=settings.search_depth, disabled=True)
