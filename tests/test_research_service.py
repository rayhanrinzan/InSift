"""Tests for query generation and complete demo competitor research."""

from sqlalchemy.orm import Session

from src.database.repositories import (
    ClusterRepository,
    EvidenceRepository,
    ResearchRepository,
)
from src.research.competitor_classifier import CompetitorClassifier
from src.research.competitor_search import MockSearchProvider, canonical_url
from src.research.query_generator import generate_competitor_queries
from src.services.research_service import ResearchService


def _cluster(session: Session):
    evidence = EvidenceRepository(session).create(
        platform="manual",
        source_external_id="research-evidence",
        source_author="clinic-ops-user",
        community="clinic-ops",
        raw_text="We still use Excel for referral follow-up and miss patients.",
        contains_problem=True,
        extraction_confidence=0.9,
        problem_statement="Clinics manually track patient referral follow-up in spreadsheets",
        affected_user="small clinic operations teams",
        current_workaround="Excel spreadsheets",
        pain_types=["time", "risk"],
        severity_score=0.8,
        frequency_signal=0.7,
        willingness_to_pay_score=0.4,
        metadata_json={"evidence_quote": "We miss patients."},
    )
    cluster = ClusterRepository(session).create(
        title="Clinic referral follow-up",
        problem_summary=evidence.problem_statement,
        target_customer=evidence.affected_user,
        current_workaround=evidence.current_workaround,
        proposed_solution="A focused follow-up queue",
    )
    ClusterRepository(session).link_evidence(cluster.id, evidence.id, 1.0)
    return cluster


def test_query_generation_is_broad_and_deduplicated(db_session: Session) -> None:
    queries = generate_competitor_queries(_cluster(db_session))

    assert len(queries) == len({query.casefold() for query in queries})
    assert any("site:producthunt.com" in query for query in queries)
    assert any("site:g2.com" in query for query in queries)
    assert any("alternative to" in query for query in queries)


def test_demo_research_persists_queries_and_filters_irrelevant_results(
    db_session: Session,
) -> None:
    cluster = _cluster(db_session)
    outcome = ResearchService(
        db_session,
        MockSearchProvider(),
        CompetitorClassifier(),
        max_results=10,
    ).research_cluster(cluster.id)

    stored_queries = ResearchRepository(db_session).list_queries_for_cluster(cluster.id)
    urls = [item.url for item in outcome.competitors]
    assert outcome.run.status == "completed"
    assert len(stored_queries) == len(generate_competitor_queries(cluster))
    assert all(query.status == "completed" for query in stored_queries)
    assert outcome.irrelevant_result_count >= 1
    assert all(item.relationship_type != "irrelevant" for item in outcome.competitors)
    assert len(urls) == len(set(urls))
    assert canonical_url("https://www.airtable.com/?utm_source=test") == "https://airtable.com"
    assert outcome.score is not None
    assert "unmet_customer_need" in outcome.score.explanation_json
