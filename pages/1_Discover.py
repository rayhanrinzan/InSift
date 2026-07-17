"""Automated, manual, CSV, and Reddit evidence discovery workflows."""

from __future__ import annotations

import importlib

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from src import runtime as _runtime

_runtime = importlib.reload(_runtime)
_runtime.ensure_runtime_current()

from src.clustering.embeddings import EmbeddingError
from src.config import Settings, get_settings
from src.extraction.opportunity_synthesizer import (
    OpportunitySynthesisError,
    build_opportunity_synthesizer,
)
from src.extraction.problem_extractor import ExtractionError
from src.ingestion.manual import (
    IngestionError,
    manual_submission,
    parse_csv_submissions,
)
from src.ingestion.reddit import RedditIngestionError, build_reddit_client
from src.ingestion.web import (
    WEB_SOURCE_LABELS,
    WebEvidenceCandidate,
    WebEvidenceDiscoveryService,
)
from src.research.competitor_search import SearchProviderError
from src.research.public_discussion_search import (
    build_public_discussion_search_provider,
)
from src.services.discovery_service import DiscoveryResult, build_discovery_service
from src.services.problem_scout_service import (
    SCOUT_FOCUS_LABELS,
    DiscoveredOpportunity,
    LiveScoutConfigurationError,
    ProblemScoutRun,
    ProblemScoutService,
)
from src.ui.components import (
    configure_page,
    page_header,
    page_size_control,
    paginate_items,
    render_database_error,
    render_pagination,
    section_header,
    status_badge_html,
)
from src.ui.data import (
    EvidenceSummary,
    clear_ui_data_caches,
    get_ui_session_factory,
    load_evidence_review,
)
from src.ui.formatting import format_datetime, format_score
from src.ui.navigation import render_page_link


SOURCE_PLATFORM_LABELS = {
    "github": "GitHub issue",
    "hacker_news": "Hacker News",
    "stack_exchange": "Stack Exchange",
    "support_community": "Product support community",
    "product_review": "Product review",
    "reddit": "Reddit",
    "web": "Public web",
}


def _source_platform_label(value: str) -> str:
    return SOURCE_PLATFORM_LABELS.get(value, value.replace("_", " ").title())


def _render_result(result: DiscoveryResult) -> None:
    """Render one ingestion outcome without exposing internal exceptions."""

    if result.duplicate:
        st.info("This discussion already exists. FlowSift AI reused the stored record.")
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
        corroborated = result.assignment.cluster.independent_source_count >= 2
        if st.button(
            "Open this opportunity" if corroborated else "Open saved signal",
            key=f"open-ingested-{result.assignment.cluster.id}",
            type="primary",
        ):
            st.session_state["selected_cluster_id"] = result.assignment.cluster.id
            st.switch_page("pages/3_Opportunity_Details.py")
        if not corroborated:
            st.info(
                "This signal is now visible throughout the product. One more "
                "independent discussion supporting the same problem will promote it "
                "to a confirmed opportunity."
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
    promoted_ids = {
        result.assignment.cluster.id
        for result in results
        if result.assignment
        and result.assignment.cluster.independent_source_count >= 2
        and result.assignment.cluster.status != "archived"
    }
    if promoted_ids:
        render_page_link(
            "pages/2_Opportunities.py",
            label="View updated opportunities",
            route="/Opportunities",
            use_container_width=False,
        )
    return results


def _render_web_candidates(
    candidates: list[WebEvidenceCandidate],
    *,
    discovery_ready: bool,
    SessionFactory: object,
) -> None:
    """Render a review-first web evidence selection workflow."""

    if not candidates:
        st.info("No attributable discussions matched this search. Try a broader topic.")
        return

    section_header(
        "Review sources",
        "Select credible discussions to extract. Every result keeps its original URL.",
    )
    selected: list[WebEvidenceCandidate] = []
    for index, candidate in enumerate(candidates):
        selection_key = f"web-evidence-{index}-{candidate.url}"
        with st.container(border=True):
            checked = st.checkbox(
                candidate.title,
                value=True,
                key=selection_key,
            )
            source, relevance, action = st.columns([2.5, 1, 1])
            source.caption(candidate.domain)
            relevance.caption(f"Search relevance {candidate.score * 100:.0f}")
            action.link_button(
                "Open source",
                candidate.url,
                use_container_width=True,
            )
            st.write(candidate.preview)
            if checked:
                selected.append(candidate)

    if not discovery_ready:
        st.info(
            "Sources can be searched now. Configure extraction and embeddings to "
            "ingest the selected discussions."
        )
    if st.button(
        f"Extract {len(selected)} selected source(s)",
        type="primary",
        use_container_width=True,
        disabled=not selected or not discovery_ready,
    ):
        _process_batch(
            [candidate.to_submission() for candidate in selected],
            SessionFactory=SessionFactory,
            label="web source(s)",
        )
        st.session_state.pop("web-evidence-candidates", None)


def _live_extraction_ready(settings: Settings) -> bool:
    return bool(not settings.demo_mode and settings.discovery_ready)


def _live_search_ready(settings: Settings) -> bool:
    return not settings.demo_mode


def _live_scout_ready(settings: Settings) -> bool:
    return _live_search_ready(settings) and _live_extraction_ready(settings)


def _render_discovered_opportunity(opportunity: DiscoveredOpportunity) -> None:
    """Render one persisted opportunity and its real supporting sources."""

    with st.container(border=True):
        heading, source_count = st.columns([4, 1])
        heading.subheader(opportunity.title)
        heading.caption(opportunity.target_customer)
        source_count.metric("Sources", opportunity.independent_source_count)

        st.markdown("**Problem**")
        st.write(opportunity.problem_summary)
        workaround, product = st.columns(2)
        workaround.markdown("**Current workaround**")
        workaround.write(opportunity.current_workaround)
        product.markdown("**Product direction**")
        product.write(opportunity.proposed_solution)

        problem, opportunity_score, confidence = st.columns(3)
        problem.metric("Problem", format_score(opportunity.problem_score))
        opportunity_score.metric(
            "Opportunity", format_score(opportunity.opportunity_score)
        )
        confidence.metric("Confidence", format_score(opportunity.confidence_score))

        with st.expander(f"Evidence ({len(opportunity.sources)} public sources)"):
            for index, source in enumerate(opportunity.sources):
                source_heading, source_action = st.columns([4, 1])
                source_heading.markdown(f"**{source.title}**")
                source_details = [
                    _source_platform_label(source.source_type),
                    source.domain,
                ]
                if source.source_author:
                    source_details.append(f"by {source.source_author}")
                if source.engagement_count:
                    source_details.append(f"{source.engagement_count} interactions")
                source_heading.caption(" | ".join(source_details))
                source_action.link_button(
                    "Open source",
                    source.url,
                    use_container_width=True,
                )
                st.write(source.excerpt)
                if index < len(opportunity.sources) - 1:
                    st.divider()

        if st.button(
            "Open full opportunity",
            key=f"open-scouted-{opportunity.cluster_id}",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["selected_cluster_id"] = opportunity.cluster_id
            st.switch_page("pages/3_Opportunity_Details.py")


def _render_scout_run(run: ProblemScoutRun) -> None:
    """Render promoted leads or an honest no-corroboration result."""

    st.caption(
        f"{run.new_source_count} new public discussion(s) processed | "
        f"{run.accepted_count} contained usable problem evidence | "
        f"{run.duplicate_count} previously seen skipped | "
        f"{run.search_query_count} searches run"
    )
    if run.source_breakdown:
        st.caption(
            "Source coverage: "
            + " | ".join(
                f"{_source_platform_label(platform)} {count}"
                for platform, count in run.source_breakdown
            )
        )
    if not run.outcomes:
        st.info(
            "Scan complete. No new qualifying discussions were found in this "
            "batch, and previously stored sources were not replayed as new results. "
            "Run the next batch to rotate into different markets and workflows."
        )
        render_page_link(
            "pages/2_Opportunities.py",
            label="View existing opportunity pipeline",
            route="/Opportunities",
            use_container_width=False,
        )
        return
    if not run.opportunities:
        st.warning(
            "No repeated problem was supported by at least two independent public "
            "discussions in this batch. New evidence is still visible as a signal "
            "throughout the product, but it is not labeled a confirmed opportunity."
        )

        accepted = [
            outcome
            for outcome in run.outcomes
            if outcome.result.accepted and outcome.result.assignment is not None
        ]
        if accepted:
            section_header(
                "New signals saved",
                "These findings now appear on Overview, Opportunities, and Opportunity details.",
            )
            shown_clusters: set[str] = set()
            for outcome in accepted:
                assignment = outcome.result.assignment
                if assignment is None or assignment.cluster.id in shown_clusters:
                    continue
                shown_clusters.add(assignment.cluster.id)
                with st.container(border=True):
                    title, action = st.columns([4, 1])
                    title.markdown(
                        f"**{outcome.result.extraction.problem_statement or outcome.source.evidence.title}**"
                    )
                    title.caption(outcome.source.segment.label)
                    action.link_button(
                        "Open source",
                        outcome.source.evidence.url,
                        use_container_width=True,
                    )
                    if st.button(
                        "Open signal details",
                        key=f"open-signal-{assignment.cluster.id}",
                        use_container_width=True,
                    ):
                        st.session_state["selected_cluster_id"] = assignment.cluster.id
                        st.switch_page("pages/3_Opportunity_Details.py")

        render_page_link(
            "pages/2_Opportunities.py",
            label="View opportunity pipeline",
            route="/Opportunities",
            use_container_width=False,
        )
        return

    section_header(
        "Evidence-backed opportunities",
        f"{len(run.opportunities)} repeated problem(s) persisted to the shared pipeline.",
    )
    for opportunity in run.opportunities:
        _render_discovered_opportunity(opportunity)
    render_page_link(
        "pages/2_Opportunities.py",
        label="View all opportunities",
        route="/Opportunities",
        use_container_width=False,
    )


def _render_opportunity_scout(
    *,
    settings: Settings,
    SessionFactory: object,
) -> None:
    """Render the no-prompt automated discovery workflow."""

    label_to_focus = {label: key for key, label in SCOUT_FOCUS_LABELS.items()}
    selected_focus_label = st.selectbox(
        "Market focus",
        list(label_to_focus),
        index=0,
    )
    focus = label_to_focus[selected_focus_label]
    st.caption(
        "Scanning GitHub issues, Hacker News, Stack Exchange, and public product-support communities. "
        "Reddit is not used by Opportunity Scout."
    )
    stored_focus = st.session_state.get("problem-scout-focus")
    has_results = stored_focus == focus and "problem-scout-run" in st.session_state
    scan_submitted = st.button(
        "Scan next batch" if has_results else "Scan for opportunities",
        key=f"problem-scout-scan-{focus}",
        type="primary",
        use_container_width=True,
        disabled=not _live_scout_ready(settings),
    )
    if not _live_scout_ready(settings):
        st.warning(
            "Automatic discovery only runs against real public sources. Turn Demo "
            "mode off and configure a working embedding provider to use it."
        )
        render_page_link(
            "pages/4_Settings.py",
            label="Open Settings",
            route="/Settings",
            use_container_width=False,
        )
    if scan_submitted:
        scan_index_key = f"problem-scout-index-{focus}"
        scan_index = int(st.session_state.get(scan_index_key, 0))
        st.session_state.pop("problem-scout-run", None)
        try:
            with st.status("Discovering repeated customer problems", expanded=True) as status:
                progress = st.progress(0, text="Selecting customer segments")

                def update_progress(value: float, message: str) -> None:
                    progress.progress(int(value * 100), text=message)

                with SessionFactory() as session:  # type: ignore[operator]
                    scout = ProblemScoutService(
                        build_public_discussion_search_provider(settings),
                        build_discovery_service(session, settings),
                        build_opportunity_synthesizer(settings),
                        search_depth=settings.search_depth,
                    )
                    run = scout.run(
                        focus=focus,
                        segment_limit=4,
                        results_per_segment=12,
                        offset=scan_index * 4,
                        scan_round=scan_index,
                        progress_callback=update_progress,
                    )
                clear_ui_data_caches()
                st.session_state["problem-scout-run"] = run
                st.session_state["problem-scout-focus"] = focus
                st.session_state[scan_index_key] = scan_index + 1
                status.update(
                    label=(
                        f"Saved {run.new_source_count} new source(s); found "
                        f"{len(run.opportunities)} confirmed opportunity lead(s)"
                    ),
                    state="complete",
                    expanded=False,
                )
        except (
            IngestionError,
            LiveScoutConfigurationError,
            SearchProviderError,
            ExtractionError,
            OpportunitySynthesisError,
            EmbeddingError,
        ) as exc:
            st.error(f"Opportunity scouting could not complete: {exc}")
        except SQLAlchemyError:
            render_database_error("Opportunity scouting", settings)

    if stored_focus == focus or scan_submitted:
        scout_run = st.session_state.get("problem-scout-run")
        if isinstance(scout_run, ProblemScoutRun):
            _render_scout_run(scout_run)


def _render_topic_search(
    *,
    settings: Settings,
    SessionFactory: object,
) -> None:
    """Render evidence search for users who already have a direction."""

    with st.form("web-discovery"):
        topic = st.text_input(
            "Market, workflow, or problem",
            placeholder="patient referral follow-up, vendor renewals, manual invoicing...",
        )
        target_customer = st.text_input(
            "Target customer (optional)",
            placeholder="independent clinics, small finance teams, agencies...",
        )
        label_to_key = {label: key for key, label in WEB_SOURCE_LABELS.items()}
        selected_labels = st.multiselect(
            "Public sources",
            list(label_to_key),
            default=[
                WEB_SOURCE_LABELS["forums"],
                WEB_SOURCE_LABELS["issues"],
                WEB_SOURCE_LABELS["reviews"],
            ],
        )
        result_limit = st.number_input(
            "Maximum sources", min_value=1, max_value=100, value=15
        )
        web_submitted = st.form_submit_button(
            "Find customer discussions",
            type="primary",
            use_container_width=True,
            disabled=not _live_search_ready(settings),
        )
    if not _live_search_ready(settings):
        st.warning(
            "Public web search is disabled in Demo mode. Simulated search results "
            "are never shown as sources."
        )
    if web_submitted:
        try:
            with st.status("Searching public sources", expanded=True) as status:
                status.write("Building evidence-oriented search queries")
                service = WebEvidenceDiscoveryService(
                    build_public_discussion_search_provider(settings),
                    search_depth=settings.search_depth,
                )
                status.write("Searching and deduplicating attributable results")
                candidates = service.discover(
                    topic,
                    target_customer=target_customer or None,
                    source_types=tuple(
                        label_to_key[label] for label in selected_labels
                    ),
                    max_results=int(result_limit),
                )
                st.session_state["web-evidence-candidates"] = candidates
                status.update(
                    label=f"Found {len(candidates)} source(s)",
                    state="complete",
                    expanded=False,
                )
        except (IngestionError, SearchProviderError) as exc:
            st.error(f"Public web discovery could not complete: {exc}")
    web_candidates = st.session_state.get("web-evidence-candidates")
    if web_candidates is not None:
        _render_web_candidates(
            web_candidates,
            discovery_ready=_live_extraction_ready(settings),
            SessionFactory=SessionFactory,
        )


def main() -> None:
    """Render the Discover page."""

    settings = get_settings()
    configure_page("Discover", settings)
    page_header(
        "Discover",
        "Find sourced customer pain and turn it into ranked market opportunities.",
        eyebrow="Evidence intake",
    )
    SessionFactory = get_ui_session_factory(settings.database_url)

    if settings.demo_mode:
        st.warning(
            "Demo mode is on. Automatic public-source discovery is disabled because "
            "demo search results are fictional. Turn Demo mode off for real discovery."
        )
    elif not settings.discovery_ready:
        st.warning(
            "Evidence analysis is not configured. Select a local or OpenAI analysis "
            "provider and a working embedding provider in Settings."
        )
        render_page_link(
            "pages/4_Settings.py",
            label="Open Settings",
            route="/Settings",
            use_container_width=False,
        )

    web_tab, manual_tab, csv_tab, reddit_tab = st.tabs(
        ["Opportunity scout", "Paste discussion", "Upload CSV", "Reddit (optional)"]
    )
    with web_tab:
        discovery_mode = st.segmented_control(
            "Discovery mode",
            ["Scout for me", "Search a topic"],
            default="Scout for me",
            label_visibility="collapsed",
        )
        if discovery_mode == "Search a topic":
            _render_topic_search(
                settings=settings,
                SessionFactory=SessionFactory,
            )
        else:
            _render_opportunity_scout(
                settings=settings,
                SessionFactory=SessionFactory,
            )

    with manual_tab:
        with st.form("manual-discovery"):
            discussion = st.text_area(
                "Discussion text",
                height=220,
                placeholder=(
                    "Paste a first-person complaint, workaround, or workflow discussion..."
                ),
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

    section_header(
        "Extraction review",
        "Verify what was accepted as problem evidence and what needs review.",
    )
    try:
        with st.spinner("Loading the review queue..."):
            recent = load_evidence_review(settings.database_url)
        _render_evidence_review(recent)
    except SQLAlchemyError:
        render_database_error("The extraction review", settings)


if __name__ == "__main__":
    main()
