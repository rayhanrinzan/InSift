"""Read services for opportunity dashboard data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database.models import (
    Competitor,
    EvidenceItem,
    OpportunityCluster,
    OpportunityScore,
)
from src.database.repositories import ClusterRepository


@dataclass(frozen=True)
class DashboardMetrics:
    """Aggregate counts shown on the home page."""

    evidence_count: int
    cluster_count: int
    researched_opportunity_count: int


@dataclass(frozen=True)
class RankedOpportunity:
    """Dashboard row for a ranked opportunity."""

    cluster_id: str
    title: str
    target_customer: Optional[str]
    evidence_count: int
    problem_score: Optional[float]
    whitespace_score: Optional[float]
    opportunity_score: Optional[float]
    confidence_score: Optional[float]
    competitor_count: int
    pain_types: tuple[str, ...]
    research_status: str
    last_updated: Optional[datetime]


class OpportunityService:
    """High-level opportunity read operations."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.clusters = ClusterRepository(session)

    def dashboard_metrics(self) -> DashboardMetrics:
        """Return high-level dashboard counts."""

        evidence_count = int(self.session.scalar(select(func.count(EvidenceItem.id))) or 0)
        cluster_count = int(self.session.scalar(select(func.count(OpportunityCluster.id))) or 0)
        researched = int(
            self.session.scalar(
                select(func.count(OpportunityCluster.id)).where(
                    OpportunityCluster.status == "researched"
                )
            )
            or 0
        )
        return DashboardMetrics(
            evidence_count=evidence_count,
            cluster_count=cluster_count,
            researched_opportunity_count=researched,
        )

    def ranked_opportunities(self, limit: int = 10) -> list[RankedOpportunity]:
        """Return clusters with their latest score, ordered by opportunity score."""

        rows: list[RankedOpportunity] = []
        clusters = self.clusters.list(limit=100)
        for cluster in clusters:
            latest_score = max(
                cluster.scores,
                key=lambda score: score.created_at,
                default=None,
            )
            competitor_count = int(
                self.session.scalar(
                    select(func.count(Competitor.id)).where(
                        Competitor.cluster_id == cluster.id
                    )
                )
                or 0
            )
            rows.append(
                RankedOpportunity(
                    cluster_id=cluster.id,
                    title=cluster.title,
                    target_customer=cluster.target_customer,
                    evidence_count=cluster.evidence_count,
                    problem_score=self._problem_score(latest_score),
                    whitespace_score=(
                        latest_score.whitespace_score if latest_score else None
                    ),
                    opportunity_score=(
                        latest_score.opportunity_score if latest_score else None
                    ),
                    confidence_score=(
                        latest_score.confidence_score if latest_score else None
                    ),
                    competitor_count=competitor_count,
                    pain_types=tuple(
                        sorted(
                            {
                                pain
                                for link in cluster.evidence_links
                                for pain in (link.evidence_item.pain_types or [])
                            }
                        )
                    ),
                    research_status=cluster.status,
                    last_updated=cluster.updated_at,
                )
            )
        return sorted(
            rows,
            key=lambda row: (row.opportunity_score is not None, row.opportunity_score or 0.0),
            reverse=True,
        )[:limit]

    def recent_evidence(self, limit: int = 5) -> list[EvidenceItem]:
        """Return recent evidence for dashboard activity."""

        statement = (
            select(EvidenceItem)
            .order_by(EvidenceItem.collected_at.desc())
            .limit(limit)
        )
        return list(self.session.execute(statement).scalars())

    @staticmethod
    def _problem_score(score: Optional[OpportunityScore]) -> Optional[float]:
        if score is None:
            return None
        stored = (score.explanation_json or {}).get("problem_score", {}).get("score")
        if stored is not None:
            return float(stored)
        return round(
            (0.35 * score.pain_severity_score)
            + (0.25 * score.problem_frequency_score)
            + (0.20 * score.willingness_to_pay_score)
            + (0.20 * score.evidence_quality_score),
            2,
        )
