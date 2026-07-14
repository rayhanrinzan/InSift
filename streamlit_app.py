"""InSift Streamlit home page."""

from __future__ import annotations

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from src.config import get_settings, redacted_database_url
from src.database.models import EvidenceItem
from src.database.session import create_database_engine, create_session_factory
from src.services.opportunity_service import OpportunityService, RankedOpportunity
from src.ui.formatting import format_datetime, format_score


def _render_opportunities(opportunities: list[RankedOpportunity]) -> None:
    """Render opportunity rows without Streamlit's Arrow-backed dataframe."""

    if not opportunities:
        st.info("No opportunities have been seeded yet.")
        return

    header = st.columns([3, 2, 1, 1, 1, 1, 1, 1.5])
    header[0].markdown("**Opportunity**")
    header[1].markdown("**Target customer**")
    header[2].markdown("**Evidence**")
    header[3].markdown("**Problem**")
    header[4].markdown("**Opportunity**")
    header[5].markdown("**Confidence**")
    header[6].markdown("**Competitors**")
    header[7].markdown("**Last updated**")
    st.divider()

    for opportunity in opportunities:
        cols = st.columns([3, 2, 1, 1, 1, 1, 1, 1.5])
        cols[0].write(opportunity.title)
        cols[1].write(opportunity.target_customer or "Unknown")
        cols[2].write(str(opportunity.evidence_count))
        cols[3].write(format_score(opportunity.problem_score))
        cols[4].write(format_score(opportunity.opportunity_score))
        cols[5].write(format_score(opportunity.confidence_score))
        cols[6].write(str(opportunity.competitor_count))
        cols[7].write(format_datetime(opportunity.last_updated))


def _render_recent_evidence(evidence_items: list[EvidenceItem]) -> None:
    """Render recent evidence without requiring pandas or pyarrow."""

    if not evidence_items:
        st.info("No evidence items yet.")
        return

    header = st.columns([3, 2, 1, 1.5])
    header[0].markdown("**Title**")
    header[1].markdown("**Community**")
    header[2].markdown("**Problem**")
    header[3].markdown("**Collected**")
    st.divider()

    for item in evidence_items:
        cols = st.columns([3, 2, 1, 1.5])
        cols[0].write(item.title or "Untitled")
        cols[1].write(item.community or item.platform)
        cols[2].write("Yes" if item.contains_problem else "No")
        cols[3].write(format_datetime(item.collected_at))


def main() -> None:
    """Render the home dashboard."""

    settings = get_settings()
    st.set_page_config(page_title="InSift", page_icon="IS", layout="wide")

    st.title("InSift")
    st.write(
        "Evidence-backed startup opportunity discovery from real online discussions."
    )

    try:
        engine = create_database_engine(settings)
        SessionFactory = create_session_factory(engine)
        with SessionFactory() as session:
            service = OpportunityService(session)
            metrics = service.dashboard_metrics()
            opportunities = service.ranked_opportunities(limit=10)
            recent_evidence = service.recent_evidence(limit=5)
    except SQLAlchemyError:
        st.error(
            "The database is not ready yet. Run `python scripts/initialize_database.py` "
            "and then `python scripts/seed_demo_data.py`."
        )
        st.caption(f"Configured database: {redacted_database_url(settings.database_url)}")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Evidence items", metrics.evidence_count)
    col2.metric("Opportunity clusters", metrics.cluster_count)
    col3.metric("Scored opportunities", metrics.researched_opportunity_count)

    st.subheader("Highest-ranked opportunities")
    _render_opportunities(opportunities)

    st.subheader("Recent ingestion activity")
    _render_recent_evidence(recent_evidence)

    st.caption(
        "Demo mode is on." if settings.demo_mode else "Demo mode is off."
    )


if __name__ == "__main__":
    main()
