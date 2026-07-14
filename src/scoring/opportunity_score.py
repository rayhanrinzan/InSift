"""Initial explainable Opportunity Score calculation and persistence."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from sqlalchemy.orm import Session

from src.database.models import Competitor, EvidenceItem, OpportunityScore
from src.database.repositories import ClusterRepository, ResearchRepository, ScoreRepository
from src.logging_config import log_event
from src.scoring.confidence_score import calculate_confidence_score
from src.scoring.problem_score import calculate_problem_score, clamp_score
from src.scoring.schemas import (
    OpportunityScoreWeights,
    OpportunityScoringResult,
    ScoreComponent,
)
from src.scoring.whitespace_score import (
    calculate_whitespace_score,
    initial_whitespace_score,
)


logger = logging.getLogger(__name__)


def calculate_opportunity_score(
    evidence_items: Sequence[EvidenceItem],
    competitors: Sequence[Competitor] = (),
    weights: OpportunityScoreWeights | None = None,
    *,
    research_complete: bool = False,
    successful_query_count: int = 0,
    target_customer: str | None = None,
    proposed_solution: str | None = None,
) -> OpportunityScoringResult:
    """Calculate opportunity components with researched white-space when available."""

    weights = weights or OpportunityScoreWeights()
    problem = calculate_problem_score(evidence_items)
    whitespace_breakdown = None
    if research_complete:
        whitespace_breakdown = calculate_whitespace_score(
            evidence_items,
            competitors,
            target_customer=target_customer,
            proposed_solution=proposed_solution,
            successful_query_count=successful_query_count,
        )
        whitespace = whitespace_breakdown.whitespace_score
    else:
        whitespace = initial_whitespace_score()
    feasibility = ScoreComponent(
        score=50.0,
        reason="Neutral placeholder until the proposed MVP receives a feasibility review.",
        inputs={"review_complete": False},
    )
    accessibility = ScoreComponent(
        score=50.0,
        reason="Neutral placeholder until customer acquisition channels are researched.",
        inputs={"research_complete": False},
    )
    confidence = calculate_confidence_score(
        evidence_items,
        competitors,
        successful_query_count=successful_query_count,
        research_complete=research_complete,
    )
    opportunity = clamp_score(
        (weights.pain_severity * problem.pain_severity.score)
        + (weights.problem_frequency * problem.problem_frequency.score)
        + (weights.willingness_to_pay * problem.willingness_to_pay.score)
        + (weights.evidence_quality * problem.evidence_quality.score)
        + (weights.whitespace * whitespace.score)
        + (weights.build_feasibility * feasibility.score)
        + (weights.market_accessibility * accessibility.score)
    )
    opportunity_component = ScoreComponent(
        score=opportunity,
        reason=(
            "Weighted from evidence-backed problem signals plus neutral placeholders for "
            "white-space, build feasibility, and market accessibility pending later research."
        ),
    )
    explanations = {
        "problem_score": problem.problem_score.dict(),
        "pain_severity": problem.pain_severity.dict(),
        "problem_frequency": problem.problem_frequency.dict(),
        "willingness_to_pay": problem.willingness_to_pay.dict(),
        "evidence_quality": problem.evidence_quality.dict(),
        "whitespace": whitespace.dict(),
        "build_feasibility": feasibility.dict(),
        "market_accessibility": accessibility.dict(),
        "opportunity": opportunity_component.dict(),
        "confidence": confidence.dict(),
    }
    if whitespace_breakdown is not None:
        explanations.update(
            {
                "unmet_customer_need": whitespace_breakdown.unmet_customer_need.dict(),
                "differentiation_potential": whitespace_breakdown.differentiation_potential.dict(),
                "competitor_weakness": whitespace_breakdown.competitor_weakness.dict(),
                "niche_specificity": whitespace_breakdown.niche_specificity.dict(),
                "low_direct_competitor_density": whitespace_breakdown.low_direct_competitor_density.dict(),
            }
        )
    return OpportunityScoringResult(
        pain_severity_score=problem.pain_severity.score,
        problem_frequency_score=problem.problem_frequency.score,
        willingness_to_pay_score=problem.willingness_to_pay.score,
        evidence_quality_score=problem.evidence_quality.score,
        whitespace_score=whitespace.score,
        build_feasibility_score=feasibility.score,
        market_accessibility_score=accessibility.score,
        opportunity_score=opportunity,
        confidence_score=confidence.score,
        explanation_json=explanations,
    )


class OpportunityScorer:
    """Score one cluster and persist a versioned result."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.clusters = ClusterRepository(session)
        self.scores = ScoreRepository(session)
        self.research = ResearchRepository(session)

    def score_cluster(self, cluster_id: str) -> OpportunityScore:
        """Recompute and save the current score for one cluster."""

        cluster = self.clusters.get(cluster_id)
        if cluster is None:
            raise ValueError("Cluster does not exist.")
        evidence_items = [link.evidence_item for link in cluster.evidence_links]
        successful_queries = self.research.successful_query_count(cluster_id)
        latest_run = self.research.latest_run(cluster_id)
        research_complete = bool(
            latest_run
            and latest_run.status in {"completed", "partial"}
            and successful_queries
        )
        result = calculate_opportunity_score(
            evidence_items,
            cluster.competitors,
            research_complete=research_complete,
            successful_query_count=successful_queries,
            target_customer=cluster.target_customer,
            proposed_solution=cluster.proposed_solution,
        )
        stored = self.scores.create(
            cluster_id=cluster_id,
            scoring_version="phase6-v1" if research_complete else "phase4-v1",
            **result.dict(),
        )
        log_event(
            logger,
            logging.INFO,
            "score_calculation",
            {
                "cluster_id": cluster_id,
                "opportunity_score": result.opportunity_score,
                "confidence_score": result.confidence_score,
                "scoring_version": "phase6-v1" if research_complete else "phase4-v1",
            },
        )
        return stored
