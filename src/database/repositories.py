"""Repository layer for database persistence."""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import Select, delete, func, select
from sqlalchemy.orm import Session, selectinload

from src.database.models import (
    ClusterEvidence,
    Competitor,
    EvidenceItem,
    OpportunityCluster,
    OpportunityScore,
    ResearchRun,
    ResearchStatus,
    SearchQuery,
    UserFeedback,
    utc_now,
)
from src.ingestion.source_urls import is_placeholder_source_url, source_identity


class EvidenceRepository:
    """Persistence operations for evidence items."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, **data: Any) -> EvidenceItem:
        """Create and persist an evidence item."""

        evidence = EvidenceItem(**data)
        self.session.add(evidence)
        self.session.commit()
        self.session.refresh(evidence)
        return evidence

    def get(self, evidence_id: str) -> Optional[EvidenceItem]:
        """Return an evidence item by ID."""

        return self.session.get(EvidenceItem, evidence_id)

    def save(self, evidence: EvidenceItem) -> EvidenceItem:
        """Persist changes to an existing evidence item."""

        self.session.add(evidence)
        self.session.commit()
        self.session.refresh(evidence)
        return evidence

    def find_by_source(
        self,
        source_url: Optional[str] = None,
        source_external_id: Optional[str] = None,
    ) -> Optional[EvidenceItem]:
        """Find an evidence item by source URL or external source ID."""

        statement: Select[tuple[EvidenceItem]] = select(EvidenceItem)
        if source_external_id:
            statement = statement.where(
                EvidenceItem.source_external_id == source_external_id
            )
        elif source_url:
            statement = statement.where(EvidenceItem.source_url == source_url)
        else:
            return None
        return self.session.execute(statement).scalars().first()

    def list_recent(self, limit: int = 20) -> list[EvidenceItem]:
        """Return recent evidence items."""

        statement = (
            select(EvidenceItem).order_by(EvidenceItem.collected_at.desc()).limit(limit)
        )
        return list(self.session.execute(statement).scalars())

    def list_problem_evidence(self, limit: int = 1000) -> list[EvidenceItem]:
        """Return accepted problem evidence in collection order."""

        statement = (
            select(EvidenceItem)
            .where(EvidenceItem.contains_problem.is_(True))
            .order_by(EvidenceItem.collected_at.asc())
            .limit(limit)
        )
        return list(self.session.execute(statement).scalars())

    def list_source_urls(self) -> list[str]:
        """Return every stored non-empty source URL for search deduplication."""

        statement = select(EvidenceItem.source_url).where(
            EvidenceItem.source_url.is_not(None)
        )
        return [
            source_url
            for source_url in self.session.execute(statement).scalars()
            if source_url
        ]

    def list_visible_recent(self, limit: int = 20) -> list[EvidenceItem]:
        """Return recent evidence without fake or local source URLs."""

        statement = select(EvidenceItem).order_by(EvidenceItem.collected_at.desc())
        items = self.session.execute(statement).scalars()
        return [
            item for item in items if not is_placeholder_source_url(item.source_url)
        ][:limit]

    def count_visible(self) -> int:
        """Count evidence that can be shown without presenting fake sources."""

        statement = select(EvidenceItem.source_url)
        return sum(
            not is_placeholder_source_url(source_url)
            for source_url in self.session.execute(statement).scalars()
        )

    def count(self) -> int:
        """Return the total number of evidence items."""

        return int(self.session.scalar(select(func.count(EvidenceItem.id))) or 0)


class ClusterRepository:
    """Persistence operations for opportunity clusters."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, **data: Any) -> OpportunityCluster:
        """Create and persist an opportunity cluster."""

        cluster = OpportunityCluster(**data)
        self.session.add(cluster)
        self.session.commit()
        self.session.refresh(cluster)
        return cluster

    def get(self, cluster_id: str) -> Optional[OpportunityCluster]:
        """Return a cluster by ID with related records preloaded."""

        statement = (
            select(OpportunityCluster)
            .where(OpportunityCluster.id == cluster_id)
            .options(
                selectinload(OpportunityCluster.evidence_links).selectinload(
                    ClusterEvidence.evidence_item
                ),
                selectinload(OpportunityCluster.competitors),
                selectinload(OpportunityCluster.scores),
            )
        )
        return self.session.execute(statement).scalars().first()

    def list(self, limit: int = 100) -> list[OpportunityCluster]:
        """Return clusters ordered by most recently updated."""

        statement = (
            select(OpportunityCluster)
            .options(selectinload(OpportunityCluster.scores))
            .order_by(OpportunityCluster.updated_at.desc())
            .limit(limit)
        )
        return list(self.session.execute(statement).scalars())

    def list_with_evidence(self, limit: int = 1000) -> list[OpportunityCluster]:
        """Return clusters with evidence records preloaded."""

        statement = (
            select(OpportunityCluster)
            .options(
                selectinload(OpportunityCluster.evidence_links).selectinload(
                    ClusterEvidence.evidence_item
                )
            )
            .order_by(OpportunityCluster.created_at.asc())
            .limit(limit)
        )
        return list(self.session.execute(statement).scalars())

    def list_promoted(self, limit: int = 1000) -> list[OpportunityCluster]:
        """Return corroborated, user-facing opportunity clusters."""

        statement = (
            select(OpportunityCluster)
            .where(
                OpportunityCluster.status != "archived",
                OpportunityCluster.independent_source_count >= 2,
            )
            .options(
                selectinload(OpportunityCluster.scores),
                selectinload(OpportunityCluster.evidence_links).selectinload(
                    ClusterEvidence.evidence_item
                ),
                selectinload(OpportunityCluster.competitors),
            )
            .order_by(OpportunityCluster.updated_at.desc())
        )
        clusters = list(self.session.execute(statement).scalars())
        return [
            cluster
            for cluster in clusters
            if len(
                {
                    identity
                    for link in cluster.evidence_links
                    if (
                        identity := source_identity(
                            link.evidence_item.source_url,
                            link.evidence_item.source_external_id,
                            link.evidence_item.id,
                        )
                    )
                }
            )
            >= 2
        ][:limit]

    def list_pipeline(self, limit: int = 1000) -> list[OpportunityCluster]:
        """Return every real-source problem signal visible in the product pipeline."""

        statement = (
            select(OpportunityCluster)
            .options(
                selectinload(OpportunityCluster.scores),
                selectinload(OpportunityCluster.evidence_links).selectinload(
                    ClusterEvidence.evidence_item
                ),
                selectinload(OpportunityCluster.competitors),
            )
            .order_by(OpportunityCluster.updated_at.desc())
        )
        clusters = list(self.session.execute(statement).scalars())
        visible: list[OpportunityCluster] = []
        for cluster in clusters:
            evidence_items = [
                link.evidence_item
                for link in cluster.evidence_links
                if link.evidence_item.contains_problem
                and not is_placeholder_source_url(link.evidence_item.source_url)
            ]
            if not evidence_items:
                continue
            scout_candidate = all(
                (item.metadata_json or {}).get("scout_segment")
                for item in evidence_items
            )
            if cluster.status == "archived" and not scout_candidate:
                continue
            visible.append(cluster)
            if len(visible) >= limit:
                break
        return visible

    def save(self, cluster: OpportunityCluster) -> OpportunityCluster:
        """Persist changes to a cluster summary."""

        self.session.add(cluster)
        self.session.commit()
        self.session.refresh(cluster)
        return cluster

    def unlink_evidence(self, cluster_id: str, evidence_item_id: str) -> bool:
        """Remove one cluster-evidence link and refresh aggregate counts."""

        link = self.session.get(
            ClusterEvidence,
            {"cluster_id": cluster_id, "evidence_item_id": evidence_item_id},
        )
        if link is None:
            return False
        self.session.delete(link)
        self.session.commit()
        self.session.expire_all()
        self._recompute_counts(cluster_id)
        return True

    def cluster_ids_for_evidence(self, evidence_item_id: str) -> list[str]:
        """Return every cluster currently linked to an evidence item."""

        statement = select(ClusterEvidence.cluster_id).where(
            ClusterEvidence.evidence_item_id == evidence_item_id
        )
        return list(self.session.execute(statement).scalars())

    def recompute_counts(self, cluster_id: str) -> None:
        """Public aggregate refresh used by correction workflows."""

        self._recompute_counts(cluster_id)

    def count(self) -> int:
        """Return the total number of clusters."""

        return int(self.session.scalar(select(func.count(OpportunityCluster.id))) or 0)

    def link_evidence(
        self,
        cluster_id: str,
        evidence_item_id: str,
        similarity_score: float,
    ) -> ClusterEvidence:
        """Link evidence to a cluster and recompute cluster evidence counts."""

        existing = self.session.get(
            ClusterEvidence,
            {"cluster_id": cluster_id, "evidence_item_id": evidence_item_id},
        )
        if existing:
            existing.similarity_score = similarity_score
            link = existing
        else:
            link = ClusterEvidence(
                cluster_id=cluster_id,
                evidence_item_id=evidence_item_id,
                similarity_score=similarity_score,
            )
            self.session.add(link)
        self.session.commit()
        self.session.expire_all()
        self._recompute_counts(cluster_id)
        self.session.refresh(link)
        return link

    def _recompute_counts(self, cluster_id: str) -> None:
        cluster = self.get(cluster_id)
        if not cluster:
            return
        evidence_items = [link.evidence_item for link in cluster.evidence_links]
        cluster.evidence_count = len({item.id for item in evidence_items})
        cluster.independent_author_count = len(
            {item.source_author for item in evidence_items if item.source_author}
        )
        cluster.independent_source_count = len(
            {
                item.source_url or item.source_external_id or item.id
                for item in evidence_items
            }
        )
        collected_dates = [
            item.collected_at for item in evidence_items if item.collected_at
        ]
        if collected_dates:
            cluster.first_seen_at = min(collected_dates)
            cluster.last_seen_at = max(collected_dates)
        else:
            cluster.first_seen_at = None
            cluster.last_seen_at = None
        self.session.add(cluster)
        self.session.commit()


class CompetitorRepository:
    """Persistence operations for competitors."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, **data: Any) -> Competitor:
        """Create and persist a competitor classification."""

        competitor = Competitor(**data)
        self.session.add(competitor)
        self.session.commit()
        self.session.refresh(competitor)
        return competitor

    def get(self, competitor_id: str) -> Optional[Competitor]:
        """Return one competitor classification."""

        return self.session.get(Competitor, competitor_id)

    def find_by_url(self, cluster_id: str, url: str) -> Optional[Competitor]:
        """Find an existing competitor by canonical URL within a cluster."""

        statement = select(Competitor).where(
            Competitor.cluster_id == cluster_id,
            Competitor.url == url,
        )
        return self.session.execute(statement).scalars().first()

    def save(self, competitor: Competitor) -> Competitor:
        """Persist a corrected or enriched competitor."""

        self.session.add(competitor)
        self.session.commit()
        self.session.refresh(competitor)
        return competitor

    def list_for_cluster(self, cluster_id: str) -> list[Competitor]:
        """Return competitors for a cluster."""

        statement = select(Competitor).where(Competitor.cluster_id == cluster_id)
        return list(self.session.execute(statement).scalars())

    def count_for_cluster(self, cluster_id: str) -> int:
        """Return competitor count for a cluster."""

        statement = select(func.count(Competitor.id)).where(
            Competitor.cluster_id == cluster_id
        )
        return int(self.session.scalar(statement) or 0)

    def delete_stale_for_cluster(
        self,
        cluster_id: str,
        *,
        keep_ids: set[str],
    ) -> int:
        """Remove old generated results while preserving user-corrected records."""

        deleted = 0
        for competitor in self.list_for_cluster(cluster_id):
            if competitor.id in keep_ids:
                continue
            if (competitor.source_evidence or {}).get(
                "user_corrected_relationship", False
            ):
                continue
            self.session.delete(competitor)
            deleted += 1
        if deleted:
            self.session.commit()
        return deleted


class ScoreRepository:
    """Persistence operations for opportunity scores."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, **data: Any) -> OpportunityScore:
        """Create and persist a versioned opportunity score."""

        score = OpportunityScore(**data)
        self.session.add(score)
        self.session.commit()
        self.session.refresh(score)
        return score

    def latest_for_cluster(self, cluster_id: str) -> Optional[OpportunityScore]:
        """Return the newest score for a cluster."""

        statement = (
            select(OpportunityScore)
            .where(OpportunityScore.cluster_id == cluster_id)
            .order_by(OpportunityScore.created_at.desc())
            .limit(1)
        )
        return self.session.execute(statement).scalars().first()

    def count(self) -> int:
        """Return the total number of stored score records."""

        return int(self.session.scalar(select(func.count(OpportunityScore.id))) or 0)


class FeedbackRepository:
    """Persistence operations for user feedback."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, **data: Any) -> UserFeedback:
        """Create and persist a feedback record."""

        feedback = UserFeedback(**data)
        self.session.add(feedback)
        self.session.commit()
        self.session.refresh(feedback)
        return feedback

    def list_for_entity(self, entity_type: str, entity_id: str) -> list[UserFeedback]:
        """Return feedback history in reverse chronological order."""

        statement = (
            select(UserFeedback)
            .where(
                UserFeedback.entity_type == entity_type,
                UserFeedback.entity_id == entity_id,
            )
            .order_by(UserFeedback.created_at.desc())
        )
        return list(self.session.execute(statement).scalars())

    def list_recent(self, limit: int = 100) -> list[UserFeedback]:
        """Return recent correction history across entity types."""

        statement = (
            select(UserFeedback).order_by(UserFeedback.created_at.desc()).limit(limit)
        )
        return list(self.session.execute(statement).scalars())


class ResearchRepository:
    """Persistence operations for research runs and generated queries."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_run(self, cluster_id: str, provider: str) -> ResearchRun:
        """Create a running research record."""

        run = ResearchRun(
            cluster_id=cluster_id,
            provider=provider,
            status=ResearchStatus.RUNNING.value,
        )
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        return run

    def create_query(
        self, research_run_id: str, cluster_id: str, query_text: str
    ) -> SearchQuery:
        """Persist a generated query before it is executed."""

        query = SearchQuery(
            research_run_id=research_run_id,
            cluster_id=cluster_id,
            query_text=query_text,
            status=ResearchStatus.RUNNING.value,
        )
        self.session.add(query)
        self.session.commit()
        self.session.refresh(query)
        return query

    def finish_query(
        self,
        query: SearchQuery,
        *,
        result_count: int,
        error_message: Optional[str] = None,
    ) -> SearchQuery:
        """Store one query's terminal outcome."""

        query.result_count = result_count
        query.error_message = error_message
        query.status = (
            ResearchStatus.FAILED.value
            if error_message
            else ResearchStatus.COMPLETED.value
        )
        query.completed_at = utc_now()
        self.session.add(query)
        self.session.commit()
        self.session.refresh(query)
        return query

    def finish_run(
        self,
        run: ResearchRun,
        *,
        query_count: int,
        result_count: int,
        relevant_competitor_count: int,
        failed_query_count: int = 0,
        error_message: Optional[str] = None,
    ) -> ResearchRun:
        """Store aggregate research results and terminal status."""

        run.query_count = query_count
        run.result_count = result_count
        run.relevant_competitor_count = relevant_competitor_count
        run.error_message = error_message
        if error_message and failed_query_count >= query_count:
            run.status = ResearchStatus.FAILED.value
        elif failed_query_count:
            run.status = ResearchStatus.PARTIAL.value
        else:
            run.status = ResearchStatus.COMPLETED.value
        run.completed_at = utc_now()
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        return run

    def list_queries_for_cluster(self, cluster_id: str) -> list[SearchQuery]:
        """Return all queries used for a cluster."""

        statement = (
            select(SearchQuery)
            .where(SearchQuery.cluster_id == cluster_id)
            .order_by(SearchQuery.created_at.desc())
        )
        return list(self.session.execute(statement).scalars())

    def successful_query_count(self, cluster_id: str) -> int:
        """Return the count of successfully executed queries."""

        statement = select(func.count(SearchQuery.id)).where(
            SearchQuery.cluster_id == cluster_id,
            SearchQuery.status == ResearchStatus.COMPLETED.value,
        )
        return int(self.session.scalar(statement) or 0)

    def latest_run(self, cluster_id: str) -> Optional[ResearchRun]:
        """Return the newest research run for a cluster."""

        statement = (
            select(ResearchRun)
            .where(ResearchRun.cluster_id == cluster_id)
            .order_by(ResearchRun.created_at.desc())
            .limit(1)
        )
        return self.session.execute(statement).scalars().first()
