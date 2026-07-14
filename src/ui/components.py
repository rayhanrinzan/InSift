"""Reusable Streamlit components."""

from typing import Optional

import streamlit as st

from src.ui.formatting import format_score


def score_metric(label: str, score: Optional[float]) -> None:
    """Render a score metric with a consistent empty state."""

    st.metric(label, format_score(score))
