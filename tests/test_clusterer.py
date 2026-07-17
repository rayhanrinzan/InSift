"""Tests for incremental centroid-based clustering."""

from sqlalchemy.orm import Session

from src.clustering.clusterer import IncrementalClusterer
from src.database.repositories import EvidenceRepository


class FixedEmbeddingProvider:
    """Predictable embeddings for cluster boundary tests."""

    def embed(self, text: str) -> list[float]:
        normalized = text.lower()
        if "vendor" in normalized:
            return [0.0, 1.0]
        if "referral" in normalized:
            return [0.98, 0.2]
        return [1.0, 0.0]


def _evidence(session: Session, external_id: str, problem: str):
    return EvidenceRepository(session).create(
        platform="manual",
        source_external_id=external_id,
        source_author=f"author-{external_id}",
        community=f"community-{external_id}",
        raw_text=problem,
        contains_problem=True,
        extraction_confidence=0.8,
        problem_statement=problem,
        pain_types=["repetitive_work"],
        severity_score=0.7,
        frequency_signal=0.6,
    )


def test_similar_problems_merge(db_session: Session) -> None:
    clusterer = IncrementalClusterer(db_session, FixedEmbeddingProvider(), threshold=0.8)
    first = clusterer.assign(
        _evidence(db_session, "one", "Clinics manually track patient follow-up")
    )
    second = clusterer.assign(
        _evidence(db_session, "two", "Referral follow-up is manually tracked by clinics")
    )

    assert second.cluster.id == first.cluster.id
    assert second.created is False
    assert second.cluster.evidence_count == 2


def test_dissimilar_problems_create_separate_clusters(db_session: Session) -> None:
    clusterer = IncrementalClusterer(db_session, FixedEmbeddingProvider(), threshold=0.8)
    first = clusterer.assign(_evidence(db_session, "one", "Clinic follow-up is manual"))
    second = clusterer.assign(_evidence(db_session, "two", "Vendor renewals are missed"))

    assert second.cluster.id != first.cluster.id
    assert second.created is True


def test_cluster_centroid_updates(db_session: Session) -> None:
    clusterer = IncrementalClusterer(db_session, FixedEmbeddingProvider(), threshold=0.8)
    assignment = clusterer.assign(_evidence(db_session, "one", "Clinic follow-up is manual"))
    clusterer.assign(_evidence(db_session, "two", "Referral follow-up is manual"))

    centroid = clusterer.centroid_for_cluster(assignment.cluster.id)

    assert centroid == [0.99, 0.1]


def test_duplicate_evidence_does_not_inflate_counts(db_session: Session) -> None:
    clusterer = IncrementalClusterer(db_session, FixedEmbeddingProvider(), threshold=0.8)
    item = _evidence(db_session, "one", "Clinic follow-up is manual")
    first = clusterer.assign(item)
    second = clusterer.assign(item)

    assert second.cluster.id == first.cluster.id
    assert second.cluster.evidence_count == 1
    assert len(second.cluster.evidence_links) == 1


def test_scout_workflow_anchor_joins_source_wording_variants(
    db_session: Session,
) -> None:
    first_item = _evidence(
        db_session,
        "source-one",
        "The tracking app cannot display customer shipment status",
    )
    second_item = _evidence(
        db_session,
        "source-two",
        "Staff send an Excel order file to suppliers every evening",
    )
    first_item.metadata_json = {"scout_workflow_topic": "order tracking customer emails"}
    second_item.metadata_json = {"scout_workflow_topic": "order tracking customer emails"}
    db_session.commit()
    clusterer = IncrementalClusterer(
        db_session,
        FixedEmbeddingProvider(),
        threshold=0.8,
    )

    first = clusterer.assign(first_item)
    second = clusterer.assign(second_item)

    assert second.cluster.id == first.cluster.id
    assert second.similarity_score >= 0.86
    assert second.created is False
