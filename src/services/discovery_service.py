"""Application service for ingestion, extraction, clustering, and scoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

from src.clustering.clusterer import ClusterAssignment, IncrementalClusterer
from src.clustering.embeddings import build_embedding_provider
from src.config import Settings
from src.database.models import EvidenceItem, OpportunityScore
from src.database.repositories import EvidenceRepository
from src.extraction.problem_extractor import (
    ProblemExtractor,
    build_problem_extraction_provider,
)
from src.extraction.schemas import ExtractedProblem
from src.ingestion.schemas import SourceSubmission
from src.logging_config import log_event
from src.scoring.opportunity_score import OpportunityScorer


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscoveryResult:
    """Outcome of processing one submitted discussion."""

    evidence: EvidenceItem
    extraction: ExtractedProblem
    accepted: bool
    duplicate: bool
    assignment: Optional[ClusterAssignment] = None
    score: Optional[OpportunityScore] = None


class DiscoveryService:
    """Coordinate the complete Phase 2-4 discovery workflow."""

    def __init__(
        self,
        session: Session,
        extractor: ProblemExtractor,
        clusterer: IncrementalClusterer,
        scorer: OpportunityScorer,
        minimum_confidence: float = 0.45,
    ) -> None:
        self.session = session
        self.extractor = extractor
        self.clusterer = clusterer
        self.scorer = scorer
        self.minimum_confidence = minimum_confidence
        self.evidence = EvidenceRepository(session)

    def process(self, submission: SourceSubmission) -> DiscoveryResult:
        """Extract, persist, cluster, and score one normalized source."""

        existing = self.evidence.find_by_source(
            source_url=submission.source_url,
            source_external_id=submission.source_external_id,
        )
        if existing is not None:
            extraction = self._extraction_from_evidence(existing)
            return DiscoveryResult(
                evidence=existing,
                extraction=extraction,
                accepted=existing.contains_problem,
                duplicate=True,
            )

        log_event(
            logger,
            logging.INFO,
            "extraction_request",
            {"platform": submission.platform, "text_length": len(submission.raw_text)},
        )
        extraction = self.extractor.extract(submission.raw_text)
        accepted = bool(
            extraction.has_usable_problem
            and extraction.confidence >= self.minimum_confidence
        )
        metadata = dict(submission.metadata_json)
        if extraction.evidence_quote:
            metadata["evidence_quote"] = extraction.evidence_quote
        stored = self.evidence.create(
            platform=submission.platform,
            source_url=submission.source_url,
            source_external_id=submission.source_external_id,
            source_author=submission.source_author,
            published_at=submission.published_at,
            community=submission.community,
            title=submission.title,
            raw_text=submission.raw_text,
            engagement_score=submission.engagement_score,
            contains_problem=accepted,
            extraction_confidence=extraction.confidence,
            problem_statement=extraction.problem_statement,
            affected_user=extraction.affected_user,
            current_workaround=extraction.current_workaround,
            pain_types=list(extraction.pain_types),
            severity_score=extraction.severity_score,
            frequency_signal=extraction.frequency_signal,
            willingness_to_pay_score=extraction.willingness_to_pay_score,
            metadata_json=metadata,
        )

        assignment = None
        score = None
        if accepted:
            assignment = self.clusterer.assign(stored)
            score = self.scorer.score_cluster(assignment.cluster.id)
        log_event(
            logger,
            logging.INFO,
            "ingestion_run",
            {
                "evidence_item_id": stored.id,
                "accepted": accepted,
                "cluster_id": assignment.cluster.id if assignment else None,
            },
        )
        return DiscoveryResult(stored, extraction, accepted, False, assignment, score)

    def process_many(
        self, submissions: list[SourceSubmission]
    ) -> list[DiscoveryResult]:
        """Process a batch of normalized submissions in order."""

        return [self.process(submission) for submission in submissions]

    @staticmethod
    def _extraction_from_evidence(evidence: EvidenceItem) -> ExtractedProblem:
        return ExtractedProblem(
            contains_real_problem=evidence.contains_problem,
            problem_statement=evidence.problem_statement,
            affected_user=evidence.affected_user,
            current_workaround=evidence.current_workaround,
            pain_types=evidence.pain_types or [],
            severity_score=evidence.severity_score,
            frequency_signal=evidence.frequency_signal,
            willingness_to_pay_score=evidence.willingness_to_pay_score,
            evidence_quote=(evidence.metadata_json or {}).get("evidence_quote"),
            confidence=evidence.extraction_confidence,
        )


def build_discovery_service(session: Session, settings: Settings) -> DiscoveryService:
    """Build the default workflow with configuration-driven dependencies."""

    extractor = ProblemExtractor(build_problem_extraction_provider(settings))
    clusterer = IncrementalClusterer(
        session,
        build_embedding_provider(settings),
        threshold=settings.cluster_similarity_threshold,
    )
    return DiscoveryService(
        session,
        extractor,
        clusterer,
        OpportunityScorer(session),
        minimum_confidence=settings.minimum_extraction_confidence,
    )
