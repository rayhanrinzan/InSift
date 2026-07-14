"""Tests for auditable corrections, merge/split, and rescoring."""

from sqlalchemy.orm import Session

from src.clustering.clusterer import IncrementalClusterer
from src.clustering.embeddings import DeterministicEmbeddingProvider
from src.database.repositories import (
    ClusterRepository,
    CompetitorRepository,
    EvidenceRepository,
    FeedbackRepository,
    ScoreRepository,
)
from src.services.correction_service import CorrectionService


def _service(session: Session) -> CorrectionService:
    return CorrectionService(
        session,
        IncrementalClusterer(session, DeterministicEmbeddingProvider(), threshold=0.78),
    )


def _evidence(session: Session, external_id: str, statement: str):
    return EvidenceRepository(session).create(
        platform="manual",
        source_external_id=external_id,
        source_author=f"author-{external_id}",
        community="operations",
        raw_text=statement,
        contains_problem=True,
        extraction_confidence=0.8,
        problem_statement=statement,
        affected_user="operations managers",
        current_workaround="spreadsheets",
        pain_types=["time"],
        severity_score=0.6,
        frequency_signal=0.5,
        willingness_to_pay_score=0.2,
        metadata_json={},
    )


def test_target_customer_correction_is_audited_and_rescored(db_session: Session) -> None:
    item = _evidence(db_session, "target", "Manual intake takes hours")
    cluster = ClusterRepository(db_session).create(
        title="Manual intake", problem_summary=item.problem_statement
    )
    ClusterRepository(db_session).link_evidence(cluster.id, item.id, 1.0)

    _service(db_session).update_target_customer(cluster.id, "clinic operations managers")

    refreshed = ClusterRepository(db_session).get(cluster.id)
    history = FeedbackRepository(db_session).list_for_entity(
        "opportunity_cluster", cluster.id
    )
    assert refreshed is not None and refreshed.target_customer == "clinic operations managers"
    assert history[0].field_name == "target_customer"
    assert ScoreRepository(db_session).latest_for_cluster(cluster.id) is not None


def test_marking_evidence_irrelevant_removes_link_and_archives_empty_cluster(
    db_session: Session,
) -> None:
    item = _evidence(db_session, "irrelevant", "Manual intake takes hours")
    cluster = ClusterRepository(db_session).create(
        title="Manual intake", problem_summary=item.problem_statement
    )
    ClusterRepository(db_session).link_evidence(cluster.id, item.id, 1.0)

    _service(db_session).correct_evidence(
        item.id,
        contains_problem=False,
        problem_statement=item.problem_statement,
        affected_user=item.affected_user,
        current_workaround=item.current_workaround,
        pain_types=item.pain_types,
        severity_score=item.severity_score,
        frequency_signal=item.frequency_signal,
        willingness_to_pay_score=item.willingness_to_pay_score,
    )

    refreshed = ClusterRepository(db_session).get(cluster.id)
    assert refreshed is not None
    assert refreshed.status == "archived"
    assert refreshed.evidence_count == 0
    assert FeedbackRepository(db_session).list_for_entity("evidence_item", item.id)


def test_competitor_reclassification_persists_user_override(db_session: Session) -> None:
    item = _evidence(db_session, "competitor", "Manual intake takes hours")
    cluster = ClusterRepository(db_session).create(
        title="Manual intake", problem_summary=item.problem_statement
    )
    ClusterRepository(db_session).link_evidence(cluster.id, item.id, 1.0)
    competitor = CompetitorRepository(db_session).create(
        cluster_id=cluster.id,
        product_name="Generic Forms",
        url="https://forms.example",
        relationship_type="adjacent",
        source_evidence={},
    )

    _service(db_session).reclassify_competitor(competitor.id, "direct")

    refreshed = CompetitorRepository(db_session).get(competitor.id)
    assert refreshed is not None and refreshed.relationship_type == "direct"
    assert refreshed.source_evidence["user_corrected_relationship"] is True
    assert FeedbackRepository(db_session).list_for_entity("competitor", competitor.id)


def test_split_then_merge_moves_evidence_and_preserves_audit(db_session: Session) -> None:
    first = _evidence(db_session, "first", "Manual clinic intake takes hours")
    second = _evidence(db_session, "second", "Vendor renewal spreadsheets are stale")
    source = ClusterRepository(db_session).create(
        title="Mixed workflows", problem_summary=first.problem_statement
    )
    ClusterRepository(db_session).link_evidence(source.id, first.id, 1.0)
    ClusterRepository(db_session).link_evidence(source.id, second.id, 0.4)
    service = _service(db_session)

    split = service.split_cluster(source.id, [second.id], title="Vendor renewals")
    remaining = ClusterRepository(db_session).get(source.id)
    assert remaining is not None and remaining.evidence_count == 1
    assert ClusterRepository(db_session).get(split.id).evidence_count == 1

    merged = service.merge_clusters(split.id, source.id)
    archived_split = ClusterRepository(db_session).get(split.id)
    assert merged.evidence_count == 2
    assert archived_split is not None and archived_split.status == "archived"
    history = FeedbackRepository(db_session).list_for_entity(
        "opportunity_cluster", split.id
    )
    assert any(item.field_name == "merged_into" for item in history)
