"""Sortable, filterable, and paginated opportunity view."""

from __future__ import annotations

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from src.config import get_settings
from src.services.opportunity_service import RankedOpportunity
from src.ui.components import (
    configure_page,
    page_header,
    page_size_control,
    paginate_items,
    render_database_error,
    render_pagination,
    score_tone,
    status_badge_html,
)
from src.ui.data import load_ranked_opportunities
from src.ui.formatting import format_datetime, format_score


def _score_cell(column: object, label: str, value: float | None) -> None:
    column.markdown(
        status_badge_html(format_score(value), score_tone(value)),
        unsafe_allow_html=True,
    )
    column.caption(label)


def _render_rows(rows: tuple[RankedOpportunity, ...]) -> None:
    if not rows:
        st.info("No opportunities match the current filters.")
        return

    for row in rows:
        with st.container(border=True):
            summary, problem, whitespace, opportunity, confidence = st.columns(
                [3.5, 1, 1, 1, 1]
            )
            if summary.button(
                row.title,
                key=f"open-{row.cluster_id}",
                use_container_width=True,
            ):
                st.session_state["selected_cluster_id"] = row.cluster_id
                st.switch_page("pages/3_Opportunity_Details.py")
            summary.caption(
                f"{row.target_customer or 'Target customer not established'} | "
                f"{row.evidence_count} evidence item(s) | "
                f"{row.competitor_count} competitor(s) | "
                f"Updated {format_datetime(row.last_updated)}"
            )
            summary.markdown(
                status_badge_html(
                    row.research_status.replace("_", " ").title(),
                    "good" if row.research_status == "researched" else "neutral",
                ),
                unsafe_allow_html=True,
            )
            _score_cell(problem, "Problem", row.problem_score)
            _score_cell(whitespace, "White-space", row.whitespace_score)
            _score_cell(opportunity, "Opportunity", row.opportunity_score)
            _score_cell(confidence, "Confidence", row.confidence_score)


def main() -> None:
    """Render ranked opportunities, filters, sorting, and pagination."""

    settings = get_settings()
    configure_page("Opportunities", settings)
    page_header(
        "Opportunities",
        "Compare ranked problem clusters and open the evidence behind any score.",
        eyebrow="Ranked market signals",
    )
    try:
        with st.spinner("Loading ranked opportunities..."):
            rows = list(load_ranked_opportunities(settings.database_url, limit=1000))
    except SQLAlchemyError:
        render_database_error("The opportunity list", settings)
        return

    with st.expander("Filters", expanded=True):
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
        statuses = sorted({row.research_status for row in rows})
        selected_statuses = second.multiselect(
            "Research status", statuses, default=statuses
        )
        start_date = third.date_input("Updated on or after", value=None)

    filtered = [
        row
        for row in rows
        if (row.opportunity_score or 0) >= minimum_score
        and (row.confidence_score or 0) >= minimum_confidence
        and (target_customer == "All" or row.target_customer == target_customer)
        and (pain_type == "All" or pain_type in row.pain_types)
        and row.research_status in selected_statuses
        and (
            start_date is None
            or row.last_updated is None
            or row.last_updated.date() >= start_date
        )
    ]

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
    st.caption(f"{len(filtered)} opportunity result(s)")
    _render_rows(page_slice.items)
    if filtered:
        render_pagination(page_slice, "opportunities")


if __name__ == "__main__":
    main()
