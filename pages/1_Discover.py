"""Manual and CSV evidence discovery workflow."""

from __future__ import annotations

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from src.config import get_settings
from src.database.models import EvidenceItem
from src.database.repositories import EvidenceRepository
from src.database.session import create_database_engine, create_session_factory
from src.extraction.problem_extractor import ExtractionError
from src.ingestion.manual import IngestionError, manual_submission, parse_csv_submissions
from src.services.discovery_service import DiscoveryResult, build_discovery_service
from src.ui.formatting import format_score


def _render_result(result: DiscoveryResult) -> None:
    """Render one ingestion outcome without exposing internal exceptions."""

    if result.duplicate:
        st.info("This discussion was already ingested, so the existing record was reused.")
    elif result.accepted and result.assignment and result.score:
        label = "Created" if result.assignment.created else "Matched"
        st.success(
            f"Accepted. {label} cluster: {result.assignment.cluster.title} "
            f"({result.assignment.similarity_score:.2f} similarity)."
        )
        cols = st.columns(3)
        cols[0].metric("Extraction confidence", f"{result.extraction.confidence * 100:.0f}")
        cols[1].metric("Opportunity score", format_score(result.score.opportunity_score))
        cols[2].metric("Confidence score", format_score(result.score.confidence_score))
    else:
        st.warning(
            "Rejected as problem evidence. The text is still stored for review, but it was "
            "not clustered or scored."
        )


def _render_evidence_review(items: list[EvidenceItem]) -> None:
    """Show accepted and rejected extraction records."""

    accepted = [item for item in items if item.contains_problem]
    rejected = [item for item in items if not item.contains_problem]
    accepted_tab, rejected_tab = st.tabs(
        [f"Accepted ({len(accepted)})", f"Rejected ({len(rejected)})"]
    )
    for tab, records in ((accepted_tab, accepted), (rejected_tab, rejected)):
        with tab:
            if not records:
                st.caption("No records in this review queue.")
            for item in records:
                label = item.problem_statement or item.title or "No problem detected"
                with st.expander(label[:110]):
                    st.write(item.raw_text)
                    if item.problem_statement:
                        st.markdown(f"**Extracted problem:** {item.problem_statement}")
                    st.caption(
                        f"Confidence: {item.extraction_confidence * 100:.0f} | "
                        f"Pain types: {', '.join(item.pain_types or []) or 'none'}"
                    )
                    if item.source_url:
                        st.link_button("Open source", item.source_url)


def main() -> None:
    """Render the Discover page."""

    st.set_page_config(page_title="InSift Discover", page_icon="IS", layout="wide")
    st.title("Discover")
    settings = get_settings()
    engine = create_database_engine(settings)
    SessionFactory = create_session_factory(engine)

    manual_tab, csv_tab = st.tabs(["Paste discussion", "Upload CSV"])
    with manual_tab:
        with st.form("manual-discovery"):
            discussion = st.text_area("Discussion text", height=220)
            left, right = st.columns(2)
            title = left.text_input("Title (optional)")
            source_url = right.text_input("Source URL (optional)")
            left, right = st.columns(2)
            source_author = left.text_input("Author (optional)")
            community = right.text_input("Community or site (optional)")
            submitted = st.form_submit_button("Extract and score", type="primary")
        if submitted:
            try:
                submission = manual_submission(
                    discussion,
                    source_url=source_url or None,
                    title=title or None,
                    source_author=source_author or None,
                    community=community or None,
                )
                with st.spinner("Extracting evidence and updating opportunities..."):
                    with SessionFactory() as session:
                        result = build_discovery_service(session, settings).process(submission)
                _render_result(result)
            except (IngestionError, ExtractionError) as exc:
                st.error(str(exc))
            except SQLAlchemyError:
                st.error("The database could not save this discussion. Check the database setup.")

    with csv_tab:
        uploaded = st.file_uploader("CSV file", type=["csv"])
        max_rows = st.number_input("Maximum rows", min_value=1, max_value=1000, value=50)
        if st.button("Ingest CSV", type="primary", disabled=uploaded is None):
            try:
                submissions = parse_csv_submissions(
                    uploaded.getvalue(), max_rows=int(max_rows)  # type: ignore[union-attr]
                )
                with st.spinner(f"Processing {len(submissions)} discussion(s)..."):
                    with SessionFactory() as session:
                        results = build_discovery_service(session, settings).process_many(submissions)
                accepted = sum(result.accepted and not result.duplicate for result in results)
                rejected = sum(not result.accepted and not result.duplicate for result in results)
                duplicates = sum(result.duplicate for result in results)
                st.success(
                    f"Processed {len(results)} row(s): {accepted} accepted, "
                    f"{rejected} rejected, {duplicates} duplicate(s)."
                )
            except (IngestionError, ExtractionError) as exc:
                st.error(str(exc))
            except SQLAlchemyError:
                st.error("The database could not save this CSV batch. Check the database setup.")

    st.subheader("Extraction review")
    try:
        with SessionFactory() as session:
            recent = EvidenceRepository(session).list_recent(limit=50)
        _render_evidence_review(recent)
    except SQLAlchemyError:
        st.error("Extraction review is unavailable because the database could not be read.")


if __name__ == "__main__":
    main()
