"""Sortable, filterable, and paginated opportunity view."""

from __future__ import annotations

import importlib

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from src import runtime as _runtime

_runtime = importlib.reload(_runtime)
_runtime.ensure_runtime_current()

from src.config import get_settings
from src.services.opportunity_service import RankedOpportunity
from src.ui.components import (
    configure_page,
    empty_state,
    page_header,
    page_size_control,
    paginate_items,
    render_database_error,
    render_pagination,
    score_bar_html,
    section_header,
    status_badge_html,
)
from src.ui.data import load_ranked_opportunities
from src.ui.formatting import format_datetime


def _render_rows(rows: tuple[RankedOpportunity, ...]) -> None:
    if not rows:
        empty_state(
            "No matching opportunities",
            "Adjust the filters or add more evidence to broaden the ranked set.",
        )
        return

    for index in range(0, len(rows), 2):
        columns = st.columns(2)
        for column, row in zip(columns, rows[index : index + 2]):
            with column:
                with st.container(border=True):
                    st.subheader(row.title)
                    st.caption(
                        f"{row.target_customer or 'Target customer not established'} | "
                        f"{row.independent_source_count} independent source(s) | "
                        f"{row.competitor_count} competitors"
                    )
                    stage_label = (
                        "Needs corroboration"
                        if row.pipeline_stage == "candidate"
                        else "Confirmed opportunity"
                    )
                    state, market = st.columns(2)
                    state.markdown(
                        status_badge_html(
                            stage_label,
                            "warn" if row.pipeline_stage == "candidate" else "good",
                        ),
                        unsafe_allow_html=True,
                    )
                    market.markdown(
                        status_badge_html(
                            row.market_check_label,
                            row.market_check_tone,
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown("**What this means**")
                    st.write(row.problem_summary)
                    st.markdown("**Product to test**")
                    st.write(row.product_hypothesis)
                    left, right = st.columns(2)
                    left.markdown(
                        score_bar_html("Problem", row.problem_score),
                        unsafe_allow_html=True,
                    )
                    right.markdown(
                        score_bar_html("Market gap", row.whitespace_score),
                        unsafe_allow_html=True,
                    )
                    left.markdown(
                        score_bar_html("Opportunity", row.opportunity_score),
                        unsafe_allow_html=True,
                    )
                    right.markdown(
                        score_bar_html("Confidence", row.confidence_score),
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Updated {format_datetime(row.last_updated)}")
                    if st.button(
                        "Open product brief",
                        key=f"open-{row.cluster_id}",
                        icon=":material/arrow_forward:",
                        use_container_width=True,
                    ):
                        st.session_state["selected_cluster_id"] = row.cluster_id
                        st.switch_page("pages/3_Opportunity_Details.py")


def main() -> None:
    """Render ranked opportunities, filters, sorting, and pagination."""

    settings = get_settings()
    configure_page("Opportunities", settings)
    page_header(
        "Opportunities",
        "Compare ranked problems, market gaps, and the confidence behind each signal.",
        eyebrow="Opportunity pipeline",
    )
    try:
        with st.spinner("Loading ranked opportunities..."):
            rows = list(load_ranked_opportunities(settings.database_url, limit=1000))
    except SQLAlchemyError:
        render_database_error("The opportunity list", settings)
        return

    with st.expander("Filters", expanded=False):
        first, second, third = st.columns(3)
        minimum_score = first.slider("Minimum opportunity score", 0, 100, 0)
        minimum_confidence = second.slider("Minimum confidence", 0, 100, 0)
        target_options = sorted(
            {row.target_customer for row in rows if row.target_customer}
        )
        target_customer = third.selectbox("Target customer", ["All", *target_options])
        first, second, third = st.columns(3)
        pain_options = sorted({pain for row in rows for pain in row.pain_types})
        pain_type = first.selectbox("Pain type", ["All", *pain_options])
        statuses = sorted({row.pipeline_stage for row in rows})
        selected_statuses = second.multiselect(
            "Pipeline stage", statuses, default=statuses
        )
        start_date = third.date_input("Updated on or after", value=None)

    filtered = [
        row
        for row in rows
        if (row.opportunity_score or 0) >= minimum_score
        and (row.confidence_score or 0) >= minimum_confidence
        and (target_customer == "All" or row.target_customer == target_customer)
        and (pain_type == "All" or pain_type in row.pain_types)
        and row.pipeline_stage in selected_statuses
        and (
            start_date is None
            or row.last_updated is None
            or row.last_updated.date() >= start_date
        )
    ]

    section_header(
        "Ranked pipeline",
        f"{len(filtered)} of {len(rows)} problem signals match the current view.",
    )
    sort_column, direction_column, size_column = st.columns([3, 1, 1])
    sort_key = sort_column.selectbox(
        "Sort by",
        [
            "Opportunity score",
            "Problem score",
            "Confidence",
            "Evidence",
            "Last updated",
        ],
    )
    descending = direction_column.toggle("Descending", value=True)
    with size_column:
        page_size = page_size_control("opportunities", default=10)
    key_functions = {
        "Opportunity score": lambda row: row.opportunity_score or 0,
        "Problem score": lambda row: row.problem_score or 0,
        "Confidence": lambda row: row.confidence_score or 0,
        "Evidence": lambda row: row.evidence_count,
        "Last updated": lambda row: row.last_updated.timestamp()
        if row.last_updated
        else 0,
    }
    filtered.sort(key=key_functions[sort_key], reverse=descending)

    page_number = int(st.session_state.get("opportunities-page", 1))
    page_slice = paginate_items(filtered, page=page_number, page_size=page_size)
    _render_rows(page_slice.items)
    if filtered:
        render_pagination(page_slice, "opportunities")


if __name__ == "__main__":
    main()
