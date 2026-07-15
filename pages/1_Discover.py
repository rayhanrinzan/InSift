"""Manual and CSV evidence discovery workflow."""

from __future__ import annotations

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from src.clustering.embeddings import EmbeddingError
from src.config import get_settings
from src.extraction.problem_extractor import ExtractionError
from src.ingestion.manual import (
    IngestionError,
    manual_submission,
    parse_csv_submissions,
)
from src.ingestion.reddit import RedditIngestionError, build_reddit_client
from src.services.discovery_service import DiscoveryResult, build_discovery_service
from src.ui.components import (
    configure_page,
    page_header,
    page_size_control,
    paginate_items,
    render_database_error,
    render_page_link,
    render_pagination,
    status_badge_html,
)
from src.ui.data import (
    EvidenceSummary,
    clear_ui_data_caches,
    get_ui_session_factory,
    load_evidence_review,
)
from src.ui.formatting import format_datetime, format_score


def _render_result(result: DiscoveryResult) -> None:
    """Render one ingestion outcome without exposing internal exceptions."""

    if result.duplicate:
        st.info("This discussion already exists. InSift reused the stored record.")
    elif result.accepted and result.assignment and result.score:
        label = "Created" if result.assignment.created else "Matched"
        st.success(
            f"Accepted. {label} cluster: {result.assignment.cluster.title} "
            f"({result.assignment.similarity_score:.2f} similarity)."
        )
        extraction, opportunity, confidence = st.columns(3)
        extraction.metric(
            "Extraction confidence", f"{result.extraction.confidence * 100:.0f}"
        )
        opportunity.metric(
            "Opportunity score", format_score(result.score.opportunity_score)
        )
        confidence.metric(
            "Confidence score", format_score(result.score.confidence_score)
        )
    else:
        st.warning(
            "Stored for review, but rejected as problem evidence and excluded from scoring."
        )


def _render_review_group(
    records: list[EvidenceSummary],
    *,
    key: str,
) -> None:
    if not records:
        st.info("This review queue is empty.")
        return

    toolbar, sizing = st.columns([4, 1])
    toolbar.caption(f"{len(records)} evidence record(s)")
    with sizing:
        page_size = page_size_control(key, default=10)
    page_number = int(st.session_state.get(f"{key}-page", 1))
    page_slice = paginate_items(records, page=page_number, page_size=page_size)

    for item in page_slice.items:
        label = item.problem_statement or item.title or "No problem detected"
        with st.expander(label[:110]):
            state, collected = st.columns([3, 1])
            state.markdown(
                status_badge_html(
                    "Accepted" if item.contains_problem else "Rejected",
                    "good" if item.contains_problem else "warn",
                ),
                unsafe_allow_html=True,
            )
            collected.caption(format_datetime(item.collected_at))
            st.write(item.raw_text)
            if item.problem_statement:
                st.markdown(f"**Extracted problem:** {item.problem_statement}")
            st.caption(
                f"Confidence: {item.extraction_confidence * 100:.0f} | "
                f"Pain types: {', '.join(item.pain_types) or 'none'}"
            )
            if item.source_url:
                st.link_button("Open source", item.source_url)
    render_pagination(page_slice, key)


def _render_evidence_review(items: tuple[EvidenceSummary, ...]) -> None:
    """Show accepted and rejected extraction records with bounded pages."""

    accepted = [item for item in items if item.contains_problem]
    rejected = [item for item in items if not item.contains_problem]
    accepted_tab, rejected_tab = st.tabs(
        [f"Accepted ({len(accepted)})", f"Rejected ({len(rejected)})"]
    )
    with accepted_tab:
        _render_review_group(accepted, key="accepted-review")
    with rejected_tab:
        _render_review_group(rejected, key="rejected-review")


def _process_batch(
    submissions: list,
    *,
    SessionFactory: object,
    label: str,
) -> list[DiscoveryResult]:
    """Process a bounded source batch with shared progress and cache handling."""

    results: list[DiscoveryResult] = []
    with st.status(f"Processing {len(submissions)} {label}", expanded=True) as status:
        progress = st.progress(0, text="Preparing batch")
        with SessionFactory() as session:  # type: ignore[operator]
            service = build_discovery_service(session, get_settings())
            for index, submission in enumerate(submissions, start=1):
                results.append(service.process(submission))
                progress.progress(
                    int((index / len(submissions)) * 100),
                    text=f"Processed item {index} of {len(submissions)}",
                )
        clear_ui_data_caches()
        status.update(
            label=f"Processed {len(results)} {label}",
            state="complete",
            expanded=False,
        )
    accepted = sum(result.accepted and not result.duplicate for result in results)
    rejected = sum(not result.accepted and not result.duplicate for result in results)
    duplicates = sum(result.duplicate for result in results)
    st.success(f"{accepted} accepted, {rejected} rejected, {duplicates} duplicate(s).")
    return results


def main() -> None:
    """Render the Discover page."""

    settings = get_settings()
    configure_page("Discover", settings)
    page_header(
        "Discover",
        "Collect a discussion, verify the extracted problem, and update opportunity clusters.",
        eyebrow="Evidence intake",
    )
    SessionFactory = get_ui_session_factory(settings.database_url)

    if not settings.discovery_ready:
        st.warning(
            "Live extraction is not configured. Add an OpenAI key and embedding "
            "provider in Settings before processing new evidence."
        )
        render_page_link(
            "pages/4_Settings.py",
            label="Open Settings",
            route="/Settings",
            use_container_width=False,
        )

    manual_tab, reddit_tab, csv_tab = st.tabs(
        ["Paste discussion", "Reddit", "Upload CSV"]
    )
    with manual_tab:
        with st.form("manual-discovery"):
            discussion = st.text_area(
                "Discussion text",
                height=220,
                placeholder="Paste one first-person complaint or workflow discussion...",
            )
            left, right = st.columns(2)
            title = left.text_input("Title (optional)")
            source_url = right.text_input("Source URL (optional)")
            left, right = st.columns(2)
            source_author = left.text_input("Author (optional)")
            community = right.text_input("Community or site (optional)")
            submitted = st.form_submit_button(
                "Extract and score",
                type="primary",
                use_container_width=True,
                disabled=not settings.discovery_ready,
            )
        if submitted:
            try:
                with st.status("Processing discussion", expanded=True) as status:
                    status.write("Validating the submitted source")
                    submission = manual_submission(
                        discussion,
                        source_url=source_url or None,
                        title=title or None,
                        source_author=source_author or None,
                        community=community or None,
                    )
                    status.write("Extracting evidence and assigning a cluster")
                    with SessionFactory() as session:
                        result = build_discovery_service(session, settings).process(
                            submission
                        )
                    clear_ui_data_caches()
                    status.update(
                        label="Discussion processed", state="complete", expanded=False
                    )
                _render_result(result)
            except (IngestionError, ExtractionError, EmbeddingError) as exc:
                st.error(f"The discussion could not be processed: {exc}")
            except SQLAlchemyError:
                render_database_error("Discussion ingestion", settings)

    with reddit_tab:
        with st.form("reddit-discovery"):
            source_mode = st.radio(
                "Reddit source",
                ["Post URL", "Subreddit", "Keywords"],
                horizontal=True,
            )
            reddit_url = ""
            subreddit = ""
            keywords = ""
            subreddit_filter = ""
            if source_mode == "Post URL":
                reddit_url = st.text_input(
                    "Reddit post URL",
                    placeholder="https://www.reddit.com/r/.../comments/...",
                )
            elif source_mode == "Subreddit":
                subreddit = st.text_input("Subreddit", placeholder="smallbusiness")
            else:
                keywords = st.text_input(
                    "Keywords", placeholder="manual invoicing takes hours"
                )
                subreddit_filter = st.text_input(
                    "Subreddit filter (optional)", placeholder="Entrepreneur"
                )
            result_limit = st.number_input(
                "Maximum items", min_value=1, max_value=100, value=25
            )
            reddit_submitted = st.form_submit_button(
                "Collect and extract",
                type="primary",
                use_container_width=True,
                disabled=not (settings.discovery_ready and settings.reddit_ready),
            )
        if not settings.reddit_ready:
            st.info("Reddit OAuth credentials are required for Reddit collection.")
        if reddit_submitted:
            try:
                client = build_reddit_client(settings)
                if source_mode == "Post URL":
                    submissions = client.submissions_from_url(
                        reddit_url, max_results=int(result_limit)
                    )
                elif source_mode == "Subreddit":
                    submissions = client.submissions_from_subreddit(
                        subreddit, max_results=int(result_limit)
                    )
                else:
                    submissions = client.submissions_from_keywords(
                        keywords,
                        subreddit=subreddit_filter or None,
                        max_results=int(result_limit),
                    )
                _process_batch(
                    submissions,
                    SessionFactory=SessionFactory,
                    label="Reddit item(s)",
                )
            except (RedditIngestionError, ExtractionError, EmbeddingError) as exc:
                st.error(f"Reddit collection could not complete: {exc}")
            except SQLAlchemyError:
                render_database_error("Reddit ingestion", settings)

    with csv_tab:
        file_column, limit_column = st.columns([3, 1])
        uploaded = file_column.file_uploader("CSV file", type=["csv"])
        max_rows = limit_column.number_input(
            "Maximum rows", min_value=1, max_value=1000, value=50
        )
        if st.button(
            "Ingest CSV",
            type="primary",
            disabled=uploaded is None or not settings.discovery_ready,
            use_container_width=True,
        ):
            try:
                submissions = parse_csv_submissions(
                    uploaded.getvalue(), max_rows=int(max_rows)  # type: ignore[union-attr]
                )
                _process_batch(
                    submissions,
                    SessionFactory=SessionFactory,
                    label="CSV row(s)",
                )
            except (IngestionError, ExtractionError, EmbeddingError) as exc:
                st.error(f"The CSV batch could not be processed: {exc}")
            except SQLAlchemyError:
                render_database_error("CSV ingestion", settings)

    st.subheader("Extraction review")
    try:
        with st.spinner("Loading the review queue..."):
            recent = load_evidence_review(settings.database_url)
        _render_evidence_review(recent)
    except SQLAlchemyError:
        render_database_error("The extraction review", settings)


if __name__ == "__main__":
    main()
