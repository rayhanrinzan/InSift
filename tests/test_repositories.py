"""Tests for Phase 1 repository behavior."""

from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from src.database.models import FeedbackType, RelationshipType
from src.database.repositories import (
    ClusterRepository,
    CompetitorRepository,
    EvidenceRepository,
    FeedbackRepository,
    ScoreRepository,
)


def test_evidence_creation(db_session: Session) -> None:
    repo = EvidenceRepository(db_session)

    evidence = repo.create(
        platform="manual",
        source_external_id="source-1",
        raw_text="Manual process takes forever.",
        contains_problem=True,
        problem_statement="Manual workflow is slow.",
        pain_types=["time", "repetitive_work"],
    )

    assert evidence.id
    assert repo.count() == 1
    assert repo.find_by_source(source_external_id="source-1") == evidence


def test_cluster_creation_and_evidence_linking(db_session: Session) -> None:
    evidence_repo = EvidenceRepository(db_session)
    cluster_repo = ClusterRepository(db_session)
    evidence = evidence_repo.create(
        platform="manual",
        source_external_id="source-2",
        source_author="author-a",
        source_url="https://example.com/a",
        raw_text="We still use spreadsheets for intake.",
        contains_problem=True,
    )
    cluster = cluster_repo.create(
        title="Spreadsheet intake tracking",
        problem_summary="Teams track intake in spreadsheets.",
    )

    cluster_repo.link_evidence(cluster.id, evidence.id, 0.91)
    cluster_repo.link_evidence(cluster.id, evidence.id, 0.95)
    refreshed = cluster_repo.get(cluster.id)

    assert refreshed is not None
    assert refreshed.evidence_count == 1
    assert refreshed.independent_author_count == 1
    assert refreshed.independent_source_count == 1
    assert refreshed.evidence_links[0].similarity_score == 0.95


def test_competitor_persistence(db_session: Session) -> None:
    cluster = ClusterRepository(db_session).create(
        title="Renewal tracking",
        problem_summary="Vendor renewals are missed.",
    )
    competitor = CompetitorRepository(db_session).create(
        cluster_id=cluster.id,
        company_name="Airtable",
        product_name="Airtable",
        url="https://www.airtable.com/",
        relationship_type=RelationshipType.SUBSTITUTE.value,
        features=["tables", "automations"],
        strengths=["flexible"],
        weaknesses=["not purpose-built"],
        source_evidence={"query": "vendor renewal spreadsheet"},
    )

    competitors = CompetitorRepository(db_session).list_for_cluster(cluster.id)

    assert competitor.id
    assert len(competitors) == 1
    assert competitors[0].relationship_type == "substitute"


def test_score_versioning_and_latest_lookup(db_session: Session) -> None:
    cluster = ClusterRepository(db_session).create(
        title="Follow-up queue",
        problem_summary="Follow-up work is manually coordinated.",
    )
    scores = ScoreRepository(db_session)
    scores.create(
        cluster_id=cluster.id,
        pain_severity_score=60,
        problem_frequency_score=50,
        willingness_to_pay_score=40,
        evidence_quality_score=70,
        whitespace_score=55,
        build_feasibility_score=80,
        market_accessibility_score=65,
        opportunity_score=61,
        confidence_score=45,
        scoring_version="v1",
        explanation_json={"summary": "Initial score"},
        created_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    newest = scores.create(
        cluster_id=cluster.id,
        pain_severity_score=70,
        problem_frequency_score=55,
        willingness_to_pay_score=45,
        evidence_quality_score=75,
        whitespace_score=60,
        build_feasibility_score=82,
        market_accessibility_score=68,
        opportunity_score=66,
        confidence_score=50,
        scoring_version="v2",
        explanation_json={"summary": "Updated score"},
        created_at=datetime.now(timezone.utc),
    )

    assert scores.count() == 2
    assert scores.latest_for_cluster(cluster.id) == newest


def test_feedback_persistence(db_session: Session) -> None:
    feedback = FeedbackRepository(db_session).create(
        entity_type="evidence_item",
        entity_id="example-id",
        field_name="contains_problem",
        original_value="true",
        corrected_value="false",
        feedback_type=FeedbackType.CORRECTION.value,
    )

    assert feedback.id
    assert feedback.feedback_type == "correction"
