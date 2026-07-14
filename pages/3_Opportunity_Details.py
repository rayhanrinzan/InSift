"""Opportunity research, scoring, evidence, and correction details."""

from __future__ import annotations

from typing import Any

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from src.config import get_settings
from src.database.models import OpportunityCluster, OpportunityScore, SearchQuery, UserFeedback
from src.database.repositories import (
    ClusterRepository,
    FeedbackRepository,
    ResearchRepository,
    ScoreRepository,
)
from src.database.session import create_database_engine, create_session_factory
from src.research.competitor_search import SearchProviderError
from src.scoring.opportunity_score import OpportunityScorer
from src.services.correction_service import build_correction_service
from src.services.research_service import build_research_service
from src.ui.formatting import format_datetime, format_score


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
    return float((score.explanation_json or {}).get("problem_score", {}).get("score", 0.0))


def _render_explanations(explanations: dict[str, Any]) -> None:
    for key, label in EXPLANATION_LABELS.items():
        component = explanations.get(key)
        if not component:
            continue
        with st.expander(f"{label}: {format_score(component.get('score'))}"):
            st.write(component.get("reason") or "No explanation is available.")
            inputs = component.get("inputs") or {}
            if inputs:
                st.caption(
                    "Inputs: "
                    + ", ".join(f"{name}={value}" for name, value in inputs.items())
                )


def _render_cluster(cluster: OpportunityCluster, score: OpportunityScore | None) -> None:
    st.header(cluster.title)
    st.write(cluster.problem_summary)
    summary_columns = st.columns(3)
    summary_columns[0].markdown(f"**Target user**  \n{cluster.target_customer or 'Unknown'}")
    summary_columns[1].markdown(
        f"**Current workaround**  \n{cluster.current_workaround or 'Not established'}"
    )
    summary_columns[2].markdown(
        f"**Proposed MVP**  \n{cluster.proposed_solution or 'Not generated'}"
    )

    if score:
        cards = st.columns(4)
        cards[0].metric("Problem Score", format_score(_problem_score(score)))
        cards[1].metric("White-Space", format_score(score.whitespace_score))
        cards[2].metric("Opportunity Score", format_score(score.opportunity_score))
        cards[3].metric("Confidence", format_score(score.confidence_score))
        if score.opportunity_score >= 65 and score.confidence_score < 50:
            st.warning(
                "This opportunity looks promising, but confidence remains limited. Add more "
                "independent evidence before relying on the ranking."
            )
        st.subheader("Score explanations")
        _render_explanations(score.explanation_json or {})
    else:
        st.info("This cluster has not been scored yet.")

    st.subheader("Evidence")
    links = sorted(
        cluster.evidence_links,
        key=lambda link: link.evidence_item.collected_at,
        reverse=True,
    )
    metrics = st.columns(3)
    metrics[0].metric("Linked items", cluster.evidence_count)
    metrics[1].metric("Independent authors", cluster.independent_author_count)
    metrics[2].metric("Independent sources", cluster.independent_source_count)
    st.caption(
        f"Evidence range: {format_datetime(cluster.first_seen_at)} to "
        f"{format_datetime(cluster.last_seen_at)}"
    )
    for link in links:
        item = link.evidence_item
        quote = (item.metadata_json or {}).get("evidence_quote") or item.raw_text
        with st.expander(item.title or item.problem_statement or "Evidence item"):
            st.write(f'"{quote}"')
            st.caption(
                f"Similarity: {link.similarity_score:.2f} | "
                f"Author: {item.source_author or 'unknown'} | "
                f"Source: {item.community or item.platform}"
            )
            if item.source_url:
                st.link_button("Open source", item.source_url)

    st.subheader("Competitors")
    visible = [item for item in cluster.competitors if item.relationship_type != "irrelevant"]
    if not visible:
        st.caption("No relevant competitors are stored yet.")
    if visible:
        headers = st.columns([2, 1, 2, 2])
        for column, label in zip(headers, ("Product", "Type", "Problem", "Supported gap")):
            column.markdown(f"**{label}**")
    for competitor in visible:
        columns = st.columns([2, 1, 2, 2])
        product = competitor.product_name or competitor.company_name or "Unknown"
        if competitor.url:
            columns[0].link_button(product, competitor.url)
        else:
            columns[0].write(product)
        columns[1].write(competitor.relationship_type.title())
        columns[2].write(competitor.problem_solved or "Unknown problem")
        columns[3].write(competitor.possible_gap or "No supported gap yet")

    st.subheader("Risks")
    st.write(
        "Evidence may not represent the wider market. Search coverage, classification "
        "confidence, build feasibility, and customer access should be reviewed before acting."
    )


def _render_research_history(queries: list[SearchQuery]) -> None:
    with st.expander(f"Search history ({len(queries)})"):
        if not queries:
            st.caption("No competitor queries have been run.")
        for query in queries[:30]:
            st.write(query.query_text)
            detail = f"{query.status.title()} | {query.result_count} result(s)"
            if query.error_message:
                detail += f" | {query.error_message}"
            st.caption(detail)


def _render_feedback_history(feedback: list[UserFeedback]) -> None:
    with st.expander(f"Correction history ({len(feedback)})"):
        if not feedback:
            st.caption("No user corrections have been recorded.")
        for item in feedback[:50]:
            st.write(f"{item.entity_type}: {item.field_name}")
            st.caption(
                f"{item.original_value or 'null'} -> {item.corrected_value or 'null'} | "
                f"{format_datetime(item.created_at)}"
            )


def _render_corrections(
    cluster: OpportunityCluster,
    all_clusters: list[OpportunityCluster],
    SessionFactory: Any,
    settings: Any,
) -> None:
    st.subheader("Corrections")
    with st.expander("Edit target customer"):
        with st.form(f"target-customer-{cluster.id}"):
            target = st.text_input("Target customer", value=cluster.target_customer or "")
            submitted = st.form_submit_button("Save target customer")
        if submitted:
            with SessionFactory() as session:
                build_correction_service(session, settings).update_target_customer(
                    cluster.id, target
                )
            st.rerun()

    with st.expander("Correct extracted evidence"):
        evidence_items = [link.evidence_item for link in cluster.evidence_links]
        if not evidence_items:
            st.caption("This cluster has no linked evidence.")
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
                evidence_submitted = st.form_submit_button("Save evidence correction")
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
                st.rerun()

    with st.expander("Reclassify competitor"):
        if not cluster.competitors:
            st.caption("No stored competitors are available to reclassify.")
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
            selected = next(item for item in cluster.competitors if item.id == competitor_id)
            with st.form(f"competitor-correction-{selected.id}"):
                relationship = st.selectbox(
                    "Relationship type",
                    RELATIONSHIP_TYPES,
                    index=RELATIONSHIP_TYPES.index(selected.relationship_type),
                )
                competitor_submitted = st.form_submit_button("Save classification")
            if competitor_submitted:
                with SessionFactory() as session:
                    build_correction_service(session, settings).reclassify_competitor(
                        selected.id, relationship
                    )
                st.rerun()

    with st.expander("Merge or split cluster"):
        merge_targets = [
            item for item in all_clusters if item.id != cluster.id and item.status != "archived"
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
                merge_submitted = st.form_submit_button("Merge cluster")
            if merge_submitted:
                with SessionFactory() as session:
                    build_correction_service(session, settings).merge_clusters(
                        cluster.id, target_id
                    )
                st.session_state["selected_cluster_id"] = target_id
                st.rerun()
        else:
            st.caption("No other active cluster is available for merging.")

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
                split_submitted = st.form_submit_button("Split selected evidence")
            if split_submitted:
                with SessionFactory() as session:
                    new_cluster = build_correction_service(session, settings).split_cluster(
                        cluster.id, split_ids, title=split_title or None
                    )
                st.session_state["selected_cluster_id"] = new_cluster.id
                st.rerun()


def main() -> None:
    """Render one selected cluster and all Phase 5-7 controls."""

    st.set_page_config(page_title="InSift Opportunity Details", page_icon="IS", layout="wide")
    st.title("Opportunity Details")
    settings = get_settings()
    SessionFactory = create_session_factory(create_database_engine(settings))

    try:
        with SessionFactory() as session:
            clusters = ClusterRepository(session).list(limit=1000)
        if not clusters:
            st.info("No opportunity clusters exist yet. Add evidence on the Discover page.")
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

        left, right = st.columns(2)
        if left.button("Recompute scores", type="primary", use_container_width=True):
            with SessionFactory() as session:
                OpportunityScorer(session).score_cluster(selected_id)
            st.rerun()
        if right.button("Research competitors", use_container_width=True):
            with st.spinner("Searching and classifying competitors..."):
                with SessionFactory() as session:
                    outcome = build_research_service(session, settings).research_cluster(
                        selected_id
                    )
            st.success(
                f"Ran {len(outcome.queries)} queries and stored "
                f"{len(outcome.competitors)} relevant competitor(s)."
            )
            st.rerun()

        with SessionFactory() as session:
            cluster = ClusterRepository(session).get(selected_id)
            latest_score = ScoreRepository(session).latest_for_cluster(selected_id)
            queries = ResearchRepository(session).list_queries_for_cluster(selected_id)
            entity_ids = {selected_id}
            if cluster:
                entity_ids.update(link.evidence_item_id for link in cluster.evidence_links)
                entity_ids.update(item.id for item in cluster.competitors)
            feedback = [
                item
                for item in FeedbackRepository(session).list_recent(limit=200)
                if item.entity_id in entity_ids
            ]
        if cluster is not None:
            _render_cluster(cluster, latest_score)
            _render_research_history(queries)
            _render_corrections(cluster, clusters, SessionFactory, settings)
            _render_feedback_history(feedback)
    except SearchProviderError as exc:
        st.error(str(exc))
    except SQLAlchemyError:
        st.error("Opportunity details are unavailable because the database could not be updated.")
    except ValueError as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
