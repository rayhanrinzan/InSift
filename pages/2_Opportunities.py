"""Sortable and filterable ranked opportunity view."""

from __future__ import annotations

from datetime import date

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from src.config import get_settings
from src.database.session import create_database_engine, create_session_factory
from src.services.opportunity_service import OpportunityService, RankedOpportunity
from src.ui.formatting import format_datetime, format_score


def _render_rows(rows: list[RankedOpportunity]) -> None:
    if not rows:
        st.info("No opportunities match the current filters.")
        return
    widths = [2.8, 1.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.8, 1.2]
    headers = st.columns(widths)
    for column, label in zip(
        headers,
        (
            "Opportunity",
            "Target customer",
            "Evidence",
            "Problem",
            "White-space",
            "Opportunity",
            "Confidence",
            "Competitors",
            "Updated",
        ),
    ):
        column.markdown(f"**{label}**")
    st.divider()
    for row in rows:
        columns = st.columns(widths)
        if columns[0].button(row.title, key=f"open-{row.cluster_id}", use_container_width=True):
            st.session_state["selected_cluster_id"] = row.cluster_id
            st.switch_page("pages/3_Opportunity_Details.py")
        columns[1].write(row.target_customer or "Unknown")
        columns[2].write(row.evidence_count)
        columns[3].write(format_score(row.problem_score))
        columns[4].write(format_score(row.whitespace_score))
        columns[5].write(format_score(row.opportunity_score))
        columns[6].write(format_score(row.confidence_score))
        columns[7].write(row.competitor_count)
        columns[8].write(format_datetime(row.last_updated))


def main() -> None:
    """Render ranked opportunities and filters."""

    st.set_page_config(page_title="InSift Opportunities", page_icon="IS", layout="wide")
    st.title("Opportunities")
    settings = get_settings()
    SessionFactory = create_session_factory(create_database_engine(settings))
    try:
        with SessionFactory() as session:
            rows = OpportunityService(session).ranked_opportunities(limit=1000)
    except SQLAlchemyError:
        st.error("The opportunity list is unavailable because the database could not be read.")
        return

    with st.expander("Filters", expanded=True):
        first, second, third = st.columns(3)
        minimum_score = first.slider("Minimum opportunity score", 0, 100, 0)
        minimum_confidence = second.slider("Minimum confidence", 0, 100, 0)
        target_options = sorted({row.target_customer for row in rows if row.target_customer})
        target_customer = third.selectbox("Target customer", ["All", *target_options])
        first, second, third = st.columns(3)
        pain_options = sorted({pain for row in rows for pain in row.pain_types})
        pain_type = first.selectbox("Pain type", ["All", *pain_options])
        statuses = sorted({row.research_status for row in rows})
        selected_statuses = second.multiselect("Research status", statuses, default=statuses)
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

    left, right = st.columns([3, 1])
    sort_key = left.selectbox(
        "Sort by",
        ["Opportunity score", "Problem score", "Confidence", "Evidence", "Last updated"],
        label_visibility="collapsed",
    )
    descending = right.toggle("Descending", value=True)
    key_functions = {
        "Opportunity score": lambda row: row.opportunity_score or 0,
        "Problem score": lambda row: row.problem_score or 0,
        "Confidence": lambda row: row.confidence_score or 0,
        "Evidence": lambda row: row.evidence_count,
        "Last updated": lambda row: row.last_updated.timestamp() if row.last_updated else 0,
    }
    filtered.sort(key=key_functions[sort_key], reverse=descending)
    st.caption(f"{len(filtered)} opportunity result(s)")
    _render_rows(filtered)


if __name__ == "__main__":
    main()
