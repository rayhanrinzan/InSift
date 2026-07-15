"""InSift Streamlit overview dashboard."""

from __future__ import annotations

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from src.config import get_settings
from src.services.opportunity_service import RankedOpportunity
from src.ui.components import (
    configure_page,
    page_header,
    render_database_error,
    render_page_link,
    status_badge_html,
)
from src.ui.data import DashboardSnapshot, EvidenceSummary, load_dashboard_snapshot
from src.ui.formatting import format_datetime, format_score


def _render_opportunities(opportunities: tuple[RankedOpportunity, ...]) -> None:
    """Render compact, responsive opportunity rows."""

    if not opportunities:
        st.info("No scored opportunities are available yet.")
        render_page_link(
            "pages/1_Discover.py", label="Open Discover", route="/Discover"
        )
        return

    for opportunity in opportunities:
        with st.container(border=True):
            summary, problem, whitespace, opportunity_score, confidence = st.columns(
                [3.4, 1, 1, 1, 1]
            )
            if summary.button(
                opportunity.title,
                key=f"dashboard-open-{opportunity.cluster_id}",
                use_container_width=True,
            ):
                st.session_state["selected_cluster_id"] = opportunity.cluster_id
                st.switch_page("pages/3_Opportunity_Details.py")
            target = opportunity.target_customer or "Target customer not established"
            summary.caption(
                f"{target} | {opportunity.evidence_count} evidence item(s) | "
                f"{opportunity.competitor_count} competitor(s)"
            )
            summary.markdown(
                status_badge_html(
                    opportunity.research_status.replace("_", " ").title(),
                    "good"
                    if opportunity.research_status == "researched"
                    else "neutral",
                ),
                unsafe_allow_html=True,
            )
            problem.markdown(f"**{format_score(opportunity.problem_score)}**")
            problem.caption("Problem")
            whitespace.markdown(f"**{format_score(opportunity.whitespace_score)}**")
            whitespace.caption("White-space")
            opportunity_score.markdown(
                f"**{format_score(opportunity.opportunity_score)}**"
            )
            opportunity_score.caption("Opportunity")
            confidence.markdown(f"**{format_score(opportunity.confidence_score)}**")
            confidence.caption("Confidence")


def _render_recent_evidence(evidence_items: tuple[EvidenceSummary, ...]) -> None:
    """Render recent evidence as a scan-friendly activity list."""

    if not evidence_items:
        st.info("No evidence has been collected yet.")
        return

    for item in evidence_items:
        title, source, state, collected = st.columns([3.5, 1.5, 1, 1.5])
        title.write(item.title or item.problem_statement or "Untitled discussion")
        title.caption((item.problem_statement or item.raw_text)[:150])
        source.write(item.community or item.platform)
        state.markdown(
            status_badge_html(
                "Accepted" if item.contains_problem else "Review",
                "good" if item.contains_problem else "warn",
            ),
            unsafe_allow_html=True,
        )
        collected.write(format_datetime(item.collected_at))
        st.divider()


def _render_metrics(snapshot: DashboardSnapshot) -> None:
    metrics = snapshot.metrics
    coverage = (
        (metrics.researched_opportunity_count / metrics.cluster_count) * 100
        if metrics.cluster_count
        else 0.0
    )
    evidence, clusters, researched, coverage_column = st.columns(4)
    evidence.metric("Evidence items", metrics.evidence_count)
    clusters.metric("Opportunity clusters", metrics.cluster_count)
    researched.metric("Researched", metrics.researched_opportunity_count)
    coverage_column.metric("Research coverage", f"{coverage:.0f}%")


def main() -> None:
    """Render the overview dashboard."""

    settings = get_settings()
    configure_page("Overview", settings)
    page_header(
        "InSift",
        "Evidence-backed opportunities ranked by problem strength, market gaps, and confidence.",
        eyebrow="Opportunity intelligence",
    )

    if not settings.demo_mode and not settings.live_ready:
        st.warning(
            "Production mode is active, but one or more live providers still need "
            "credentials. Existing data remains available while setup is completed."
        )
        render_page_link(
            "pages/4_Settings.py",
            label="Complete live setup",
            route="/Settings",
            use_container_width=False,
        )

    try:
        with st.spinner("Loading the latest opportunity signals..."):
            snapshot = load_dashboard_snapshot(settings.database_url)
    except SQLAlchemyError:
        render_database_error("The overview", settings)
        return

    _render_metrics(snapshot)

    heading, mode = st.columns([4, 1])
    heading.subheader("Highest-ranked opportunities")
    mode.markdown(
        status_badge_html(
            "30-second cache",
            "neutral",
        ),
        unsafe_allow_html=True,
    )
    _render_opportunities(snapshot.opportunities)

    st.subheader("Recent ingestion activity")
    _render_recent_evidence(snapshot.recent_evidence)


if __name__ == "__main__":
    main()
