"""Opportunity research, scoring, evidence, and correction details."""

from __future__ import annotations

import importlib
import re
from html import escape
from typing import Any

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from src import runtime as _runtime

_runtime = importlib.reload(_runtime)
_runtime.ensure_runtime_current()

from src.config import get_settings
from src.database.models import (
    OpportunityCluster,
    OpportunityScore,
    SearchQuery,
    UserFeedback,
)
from src.database.repositories import (
    ClusterRepository,
    FeedbackRepository,
    ResearchRepository,
    ScoreRepository,
)
from src.ingestion.source_urls import is_placeholder_source_url, source_identity
from src.research.competitor_search import SearchProviderError
from src.scoring.opportunity_score import OpportunityScorer
from src.services.correction_service import build_correction_service
from src.services.opportunity_brief_service import (
    OpportunityBrief,
    build_opportunity_brief,
)
from src.services.research_service import build_research_service
from src.ui.components import (
    configure_page,
    empty_state,
    page_header,
    page_size_control,
    paginate_items,
    render_database_error,
    render_flash,
    render_pagination,
    score_bar_html,
    score_tone,
    section_header,
    set_flash,
    status_badge_html,
)
from src.ui.data import clear_ui_data_caches, get_ui_session_factory
from src.ui.formatting import format_datetime, format_score
from src.ui.navigation import render_page_link


def _plain_summary(value: str | None, fallback: str, limit: int = 320) -> str:
    cleaned = re.sub(r"#{1,6}\s*", "", value or "")
    cleaned = re.sub(r"[*_`]+", "", cleaned)
    cleaned = " ".join(cleaned.split()) or fallback
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 3].rsplit(" ", 1)[0] + "..."
    return escape(cleaned)


EXPLANATION_LABELS = {
    "problem_score": "Problem Score",
    "pain_severity": "Pain Severity",
    "problem_frequency": "Problem Frequency",
    "willingness_to_pay": "Willingness to Pay",
    "evidence_quality": "Evidence Quality",
    "whitespace": "White-Space",
    "unmet_customer_need": "Unmet Customer Need",
    "differentiation_potential": "Differentiation Potential",
    "competitor_weakness": "Competitor Weakness",
    "niche_specificity": "Niche Specificity",
    "low_direct_competitor_density": "Low Direct-Competitor Density",
    "build_feasibility": "Build Feasibility",
    "market_accessibility": "Market Accessibility",
    "opportunity": "Opportunity Score",
    "confidence": "Confidence",
}
PAIN_TYPES = [
    "time",
    "labor",
    "cost",
    "lost_revenue",
    "risk",
    "compliance",
    "coordination",
    "data_entry",
    "poor_user_experience",
    "lack_of_visibility",
    "integration",
    "repetitive_work",
]
RELATIONSHIP_TYPES = ["direct", "adjacent", "substitute", "irrelevant"]


def _problem_score(score: OpportunityScore) -> float:
    return float(
        (score.explanation_json or {}).get("problem_score", {}).get("score", 0.0)
    )


def _render_score_explanations(explanations: dict[str, Any]) -> None:
    for key, label in EXPLANATION_LABELS.items():
        component = explanations.get(key)
        if not component:
            continue
        score = component.get("score")
        with st.expander(f"{label}: {format_score(score)}"):
            st.markdown(
                status_badge_html(format_score(score), score_tone(score)),
                unsafe_allow_html=True,
            )
            st.write(component.get("reason") or "No explanation is available.")
            inputs = component.get("inputs") or {}
            if inputs:
                st.caption(
                    "Inputs: "
                    + ", ".join(f"{name}={value}" for name, value in inputs.items())
                )


def _render_overview(
    cluster: OpportunityCluster,
    score: OpportunityScore | None,
) -> None:
    brief = build_opportunity_brief(cluster)
    is_candidate = cluster.independent_source_count < 2
    state, updated = st.columns([3, 1])
    state.markdown(
        status_badge_html(
            "Needs corroboration" if is_candidate else "Confirmed opportunity",
            "warn" if is_candidate else "good",
        ),
        unsafe_allow_html=True,
    )
    updated.caption(f"Updated {format_datetime(cluster.updated_at)}")
    if is_candidate:
        st.warning(
            "Early signal: only one independent source supports this problem. Use the "
            "brief to plan validation, but do not treat the product direction as proven."
        )

    section_header(
        "Problem, in plain English",
        "What is happening, who experiences it, and why it matters.",
    )
    st.write(brief.plain_english)
    target, workaround, evidence = st.columns(3)
    target.markdown("**Who has this problem**")
    target.write(brief.core_user)
    workaround.markdown("**What they use today**")
    workaround.write(brief.current_workaround)
    evidence.markdown("**Evidence strength**")
    evidence.write(brief.evidence_strength)
    st.info(brief.business_impact)

    _render_market_assessment(brief)

    section_header(
        "Product to test",
        "A narrow hypothesis that directly addresses the documented workflow.",
    )
    st.write(brief.product_hypothesis)
    st.markdown("**Core user flow**")
    for index, step in enumerate(brief.core_workflow, start=1):
        st.write(f"{index}. {step}")

    section_header(
        "MVP scope",
        "Only the capabilities needed to prove the workflow and outcome.",
    )
    for index in range(0, len(brief.mvp_features), 2):
        columns = st.columns(2)
        for column, feature in zip(columns, brief.mvp_features[index : index + 2]):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{feature.name}**")
                    st.write(feature.purpose)

    section_header(
        "Four-week validation and build plan",
        "Validate demand before investing in a broad product.",
    )
    for phase in brief.build_phases:
        with st.container(border=True):
            phase_label, phase_title = st.columns([1, 4])
            phase_label.markdown(
                status_badge_html(phase.week, "neutral"),
                unsafe_allow_html=True,
            )
            phase_title.markdown(f"**{phase.title}**")
            for action in phase.actions:
                st.write(f"- {action}")
            st.caption(f"Exit criteria: {phase.exit_criteria}")

    section_header("Build boundaries", "What to implement first and what to delay.")
    with st.expander("Technical starting point", expanded=True):
        for step in brief.technical_start:
            st.write(f"- {step}")
    with st.expander("Do not build yet"):
        for item in brief.excluded_scope:
            st.write(f"- {item}")
    st.success(f"Pilot success threshold: {brief.success_metric}")

    section_header("Scorecard", "Quality and confidence are evaluated separately.")
    if score is None:
        empty_state(
            "No score available",
            "Recompute scores after linking evidence to this opportunity.",
        )
    else:
        problem, whitespace, opportunity, confidence = st.columns(4)
        problem.markdown(
            score_bar_html("Problem", _problem_score(score)), unsafe_allow_html=True
        )
        whitespace.markdown(
            score_bar_html("Market gap", score.whitespace_score),
            unsafe_allow_html=True,
        )
        opportunity.markdown(
            score_bar_html("Opportunity", score.opportunity_score),
            unsafe_allow_html=True,
        )
        confidence.markdown(
            score_bar_html("Confidence", score.confidence_score),
            unsafe_allow_html=True,
        )
        if score.opportunity_score >= 65 and score.confidence_score < 50:
            st.warning(
                "The score is promising, but confidence is limited. Add independent "
                "evidence before treating the ranking as reliable."
            )


def _render_market_assessment(brief: OpportunityBrief) -> None:
    """Show the market verdict before recommending product work."""

    competition = brief.competition
    section_header(
        "Existing-solution check",
        "Direct products, adjacent products, and current substitutes.",
    )
    st.markdown(
        status_badge_html(competition.label, competition.tone),
        unsafe_allow_html=True,
    )
    st.write(competition.summary)
    if competition.status == "required":
        st.warning(competition.recommendation)
    elif competition.tone == "risk":
        st.error(competition.recommendation)
    else:
        st.info(competition.recommendation)
    direct, adjacent, substitutes = st.columns(3)
    direct.metric("Direct", competition.direct_count)
    adjacent.metric("Adjacent", competition.adjacent_count)
    substitutes.metric("Substitutes", competition.substitute_count)
    if competition.gaps:
        st.markdown("**Gaps supported by research**")
        for gap in competition.gaps:
            st.write(f"- {gap}")
    st.caption(
        "A web search can find evidence of existing solutions, but it cannot prove that "
        "no solution exists. Customer interviews must confirm why current options fail."
    )


def _render_evidence(cluster: OpportunityCluster) -> None:
    links = sorted(
        (
            link
            for link in cluster.evidence_links
            if not is_placeholder_source_url(link.evidence_item.source_url)
        ),
        key=lambda link: link.evidence_item.collected_at,
        reverse=True,
    )
    evidence_items = [link.evidence_item for link in links]
    source_ids = {
        identity
        for item in evidence_items
        if (
            identity := source_identity(
                item.source_url,
                item.source_external_id,
                item.id,
            )
        )
    }
    linked, authors, sources = st.columns(3)
    linked.metric("Linked items", len({item.id for item in evidence_items}))
    authors.metric(
        "Independent authors",
        len({item.source_author for item in evidence_items if item.source_author}),
    )
    sources.metric("Independent sources", len(source_ids))
    st.caption(
        f"Evidence range: {format_datetime(cluster.first_seen_at)} to "
        f"{format_datetime(cluster.last_seen_at)}"
    )
    if not links:
        empty_state(
            "No linked evidence",
            "This opportunity needs source evidence before it can be validated.",
        )
        return

    toolbar, sizing = st.columns([4, 1])
    toolbar.caption(f"{len(links)} linked evidence item(s)")
    with sizing:
        page_size = page_size_control(f"evidence-{cluster.id}", default=10)
    key = f"evidence-{cluster.id}"
    page_number = int(st.session_state.get(f"{key}-page", 1))
    page_slice = paginate_items(links, page=page_number, page_size=page_size)
    for link in page_slice.items:
        item = link.evidence_item
        quote = (item.metadata_json or {}).get("evidence_quote") or item.raw_text
        with st.expander(item.title or item.problem_statement or "Evidence item"):
            st.write(f'"{quote}"')
            st.caption(
                f"Similarity: {link.similarity_score:.2f} | "
                f"Author: {item.source_author or 'unknown'} | "
                f"Source: {item.community or item.platform} | "
                f"Collected: {format_datetime(item.collected_at)}"
            )
            if item.source_url:
                st.link_button("Open source", item.source_url)
    render_pagination(page_slice, key)


def _render_competitors(cluster: OpportunityCluster) -> None:
    visible = [
        item for item in cluster.competitors if item.relationship_type != "irrelevant"
    ]
    if not visible:
        empty_state(
            "No existing-solution check yet",
            "Run the market check to map direct products, adjacent products, and substitutes before building.",
        )
        return

    relationships = sorted({item.relationship_type for item in visible})
    selected = st.multiselect(
        "Relationship type",
        relationships,
        default=relationships,
        key=f"competitor-filter-{cluster.id}",
    )
    filtered = [item for item in visible if item.relationship_type in selected]
    toolbar, sizing = st.columns([4, 1])
    toolbar.caption(f"{len(filtered)} competitor result(s)")
    with sizing:
        page_size = page_size_control(f"competitors-{cluster.id}", default=10)
    key = f"competitors-{cluster.id}"
    page_number = int(st.session_state.get(f"{key}-page", 1))
    page_slice = paginate_items(filtered, page=page_number, page_size=page_size)

    for competitor in page_slice.items:
        with st.container(border=True):
            product_column, type_column = st.columns([4, 1])
            product = competitor.product_name or competitor.company_name or "Unknown"
            if competitor.url:
                product_column.link_button(product, competitor.url)
            else:
                product_column.subheader(product)
            type_column.markdown(
                status_badge_html(competitor.relationship_type.title(), "neutral"),
                unsafe_allow_html=True,
            )
            problem, gap = st.columns(2)
            problem.markdown(
                "<p><strong>Problem solved</strong><br>"
                f"{_plain_summary(competitor.problem_solved, 'Unknown problem')}</p>",
                unsafe_allow_html=True,
            )
            gap.markdown(
                "<p><strong>Supported gap</strong><br>"
                f"{_plain_summary(competitor.possible_gap, 'No supported gap yet')}</p>",
                unsafe_allow_html=True,
            )
            if competitor.weaknesses:
                st.caption("Weaknesses: " + ", ".join(competitor.weaknesses))
    render_pagination(page_slice, key)


def _render_research_history(queries: list[SearchQuery], cluster_id: str) -> None:
    section_header("Search history")
    if not queries:
        st.info("No competitor queries have been run.")
        return
    key = f"query-history-{cluster_id}"
    page_slice = paginate_items(
        queries,
        page=int(st.session_state.get(f"{key}-page", 1)),
        page_size=10,
    )
    for query in page_slice.items:
        query_text, query_state = st.columns([4, 1])
        query_text.write(query.query_text)
        tone = "good" if query.status == "completed" else "warn"
        query_state.markdown(
            status_badge_html(query.status.title(), tone), unsafe_allow_html=True
        )
        detail = f"{query.result_count} result(s)"
        if query.error_message:
            detail += f" | {query.error_message}"
        st.caption(detail)
        st.divider()
    render_pagination(page_slice, key)


def _render_feedback_history(feedback: list[UserFeedback], cluster_id: str) -> None:
    section_header("Correction history")
    if not feedback:
        st.info("No user corrections have been recorded.")
        return
    key = f"feedback-history-{cluster_id}"
    page_slice = paginate_items(
        feedback,
        page=int(st.session_state.get(f"{key}-page", 1)),
        page_size=10,
    )
    for item in page_slice.items:
        st.write(f"{item.entity_type}: {item.field_name}")
        st.caption(
            f"{item.original_value or 'null'} -> {item.corrected_value or 'null'} | "
            f"{format_datetime(item.created_at)}"
        )
        st.divider()
    render_pagination(page_slice, key)


def _render_corrections(
    cluster: OpportunityCluster,
    all_clusters: list[OpportunityCluster],
    SessionFactory: Any,
    settings: Any,
) -> None:
    customer_tab, evidence_tab, competitor_tab, cluster_tab = st.tabs(
        ["Target customer", "Evidence", "Competitors", "Merge or split"]
    )

    with customer_tab:
        with st.form(f"target-customer-{cluster.id}"):
            target = st.text_input(
                "Target customer", value=cluster.target_customer or ""
            )
            submitted = st.form_submit_button(
                "Save target customer", type="primary", use_container_width=True
            )
        if submitted:
            with SessionFactory() as session:
                build_correction_service(session, settings).update_target_customer(
                    cluster.id, target
                )
            clear_ui_data_caches()
            set_flash("Target customer updated and scores recomputed.")
            st.rerun()

    with evidence_tab:
        evidence_items = [link.evidence_item for link in cluster.evidence_links]
        if not evidence_items:
            st.info("This opportunity has no linked evidence.")
        else:
            evidence_id = st.selectbox(
                "Evidence item",
                [item.id for item in evidence_items],
                format_func=lambda item_id: next(
                    (item.problem_statement or item.title or item.id)
                    for item in evidence_items
                    if item.id == item_id
                )[:100],
                key=f"evidence-correction-select-{cluster.id}",
            )
            item = next(value for value in evidence_items if value.id == evidence_id)
            with st.form(f"evidence-correction-{item.id}"):
                contains_problem = st.checkbox(
                    "Contains a real problem", value=item.contains_problem
                )
                problem_statement = st.text_area(
                    "Problem statement", value=item.problem_statement or ""
                )
                affected_user = st.text_input(
                    "Affected user", value=item.affected_user or ""
                )
                workaround = st.text_area(
                    "Current workaround", value=item.current_workaround or ""
                )
                pain_types = st.multiselect(
                    "Pain types", PAIN_TYPES, default=item.pain_types or []
                )
                first, second, third = st.columns(3)
                severity = first.slider(
                    "Severity", 0.0, 1.0, float(item.severity_score), 0.05
                )
                frequency = second.slider(
                    "Frequency", 0.0, 1.0, float(item.frequency_signal), 0.05
                )
                willingness = third.slider(
                    "Willingness to pay",
                    0.0,
                    1.0,
                    float(item.willingness_to_pay_score),
                    0.05,
                )
                evidence_submitted = st.form_submit_button(
                    "Save evidence correction", type="primary", use_container_width=True
                )
            if evidence_submitted:
                with SessionFactory() as session:
                    build_correction_service(session, settings).correct_evidence(
                        item.id,
                        contains_problem=contains_problem,
                        problem_statement=problem_statement,
                        affected_user=affected_user,
                        current_workaround=workaround,
                        pain_types=pain_types,
                        severity_score=severity,
                        frequency_signal=frequency,
                        willingness_to_pay_score=willingness,
                    )
                clear_ui_data_caches()
                set_flash("Evidence correction saved and affected scores recomputed.")
                st.rerun()

    with competitor_tab:
        if not cluster.competitors:
            st.info("No stored competitors are available to reclassify.")
        else:
            competitor_id = st.selectbox(
                "Competitor",
                [item.id for item in cluster.competitors],
                format_func=lambda item_id: next(
                    (item.product_name or item.company_name or item.id)
                    for item in cluster.competitors
                    if item.id == item_id
                ),
                key=f"competitor-correction-select-{cluster.id}",
            )
            selected = next(
                item for item in cluster.competitors if item.id == competitor_id
            )
            with st.form(f"competitor-correction-{selected.id}"):
                relationship = st.selectbox(
                    "Relationship type",
                    RELATIONSHIP_TYPES,
                    index=RELATIONSHIP_TYPES.index(selected.relationship_type),
                )
                competitor_submitted = st.form_submit_button(
                    "Save classification", type="primary", use_container_width=True
                )
            if competitor_submitted:
                with SessionFactory() as session:
                    build_correction_service(session, settings).reclassify_competitor(
                        selected.id, relationship
                    )
                clear_ui_data_caches()
                set_flash("Competitor classification updated and scores recomputed.")
                st.rerun()

    with cluster_tab:
        merge_targets = [
            item
            for item in all_clusters
            if item.id != cluster.id and item.status != "archived"
        ]
        if merge_targets:
            with st.form(f"merge-cluster-{cluster.id}"):
                target_id = st.selectbox(
                    "Merge this cluster into",
                    [item.id for item in merge_targets],
                    format_func=lambda item_id: next(
                        item.title for item in merge_targets if item.id == item_id
                    ),
                )
                merge_submitted = st.form_submit_button(
                    "Merge cluster", use_container_width=True
                )
            if merge_submitted:
                with SessionFactory() as session:
                    build_correction_service(session, settings).merge_clusters(
                        cluster.id, target_id
                    )
                clear_ui_data_caches()
                st.session_state["selected_cluster_id"] = target_id
                set_flash("Clusters merged and affected scores recomputed.")
                st.rerun()
        else:
            st.info("No other active cluster is available for merging.")

        evidence_items = [link.evidence_item for link in cluster.evidence_links]
        if len(evidence_items) >= 2:
            with st.form(f"split-cluster-{cluster.id}"):
                split_ids = st.multiselect(
                    "Evidence to move into a new cluster",
                    [item.id for item in evidence_items],
                    format_func=lambda item_id: next(
                        (item.problem_statement or item.title or item.id)
                        for item in evidence_items
                        if item.id == item_id
                    )[:100],
                )
                split_title = st.text_input("New cluster title (optional)")
                split_submitted = st.form_submit_button(
                    "Split selected evidence", use_container_width=True
                )
            if split_submitted:
                with SessionFactory() as session:
                    new_cluster = build_correction_service(
                        session, settings
                    ).split_cluster(cluster.id, split_ids, title=split_title or None)
                clear_ui_data_caches()
                st.session_state["selected_cluster_id"] = new_cluster.id
                set_flash("Evidence split into a new cluster and scores recomputed.")
                st.rerun()


def main() -> None:
    """Render one selected cluster and all research and correction controls."""

    settings = get_settings()
    configure_page("Opportunity details", settings)
    page_header(
        "Product opportunity brief",
        "Understand the problem, check existing solutions, and follow a validation-first build plan.",
        eyebrow="Problem to product",
    )
    render_flash()
    SessionFactory = get_ui_session_factory(settings.database_url)

    try:
        with SessionFactory() as session:
            clusters = ClusterRepository(session).list_pipeline(limit=1000)
        if not clusters:
            empty_state(
                "No opportunity selected",
                "Add evidence in Discover to create the first opportunity cluster.",
            )
            render_page_link(
                "pages/1_Discover.py",
                label="Add evidence",
                route="/Discover",
                use_container_width=True,
            )
            return

        ids = [cluster.id for cluster in clusters]
        selected = st.session_state.get("selected_cluster_id")
        selected_index = ids.index(selected) if selected in ids else 0
        selected_id = st.selectbox(
            "Opportunity",
            ids,
            index=selected_index,
            format_func=lambda cluster_id: next(
                cluster.title for cluster in clusters if cluster.id == cluster_id
            ),
        )
        st.session_state["selected_cluster_id"] = selected_id
        selected_preview = next(
            cluster for cluster in clusters if cluster.id == selected_id
        )

        research, recompute = st.columns([2, 1])
        if recompute.button(
            "Refresh scores",
            icon=":material/refresh:",
            use_container_width=True,
        ):
            with st.status("Recomputing scores", expanded=True) as status:
                status.write("Loading evidence and competitor inputs")
                with SessionFactory() as session:
                    OpportunityScorer(session).score_cluster(selected_id)
                clear_ui_data_caches()
                status.update(
                    label="Scores recomputed", state="complete", expanded=False
                )
            set_flash("Scores recomputed from the latest stored evidence.")
            st.rerun()

        research_ready = settings.research_ready
        research_label = (
            "Refresh existing-solution check"
            if selected_preview.status == "researched"
            else "Check existing solutions"
        )
        if research.button(
            research_label,
            type="primary",
            icon=":material/manage_search:",
            use_container_width=True,
            disabled=not research_ready,
        ):
            with st.status("Researching competitors", expanded=True) as status:
                progress = st.progress(0, text="Preparing competitor research")

                def update_progress(value: float, message: str) -> None:
                    progress.progress(int(value * 100), text=message)

                with SessionFactory() as session:
                    outcome = build_research_service(
                        session, settings
                    ).research_cluster(
                        selected_id,
                        progress_callback=update_progress,
                    )
                clear_ui_data_caches()
                status.update(
                    label="Competitor research complete",
                    state="complete",
                    expanded=False,
                )
            set_flash(
                f"Checked {len(outcome.queries)} market searches and stored "
                f"{len(outcome.competitors)} relevant existing solution(s)."
            )
            st.rerun()
        if not research_ready:
            research.info(
                "Configure Tavily in Settings to check existing solutions. The "
                "product plan remains provisional until this check runs."
            )

        with SessionFactory() as session:
            cluster = ClusterRepository(session).get(selected_id)
            latest_score = ScoreRepository(session).latest_for_cluster(selected_id)
            queries = ResearchRepository(session).list_queries_for_cluster(selected_id)
            entity_ids = {selected_id}
            if cluster:
                entity_ids.update(
                    link.evidence_item_id for link in cluster.evidence_links
                )
                entity_ids.update(item.id for item in cluster.competitors)
            feedback = [
                item
                for item in FeedbackRepository(session).list_recent(limit=500)
                if item.entity_id in entity_ids
            ]
        if cluster is None:
            st.error("The selected opportunity no longer exists.")
            return

        st.header(cluster.title)
        (
            overview_tab,
            evidence_tab,
            competitor_tab,
            scoring_tab,
            correction_tab,
            history_tab,
        ) = st.tabs(
            [
                "Product brief",
                "Evidence",
                "Existing solutions",
                "Scoring",
                "Corrections",
                "History",
            ]
        )
        with overview_tab:
            _render_overview(cluster, latest_score)
        with evidence_tab:
            _render_evidence(cluster)
        with competitor_tab:
            _render_competitors(cluster)
        with scoring_tab:
            if latest_score:
                _render_score_explanations(latest_score.explanation_json or {})
            else:
                st.info("This opportunity has not been scored yet.")
        with correction_tab:
            _render_corrections(cluster, clusters, SessionFactory, settings)
        with history_tab:
            _render_research_history(queries, cluster.id)
            _render_feedback_history(feedback, cluster.id)
    except SearchProviderError as exc:
        st.error(f"Competitor research could not complete: {exc}")
    except SQLAlchemyError:
        render_database_error("Opportunity details", settings)
    except ValueError as exc:
        st.error(f"The requested update could not be completed: {exc}")


if __name__ == "__main__":
    main()
