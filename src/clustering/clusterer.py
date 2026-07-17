"""Incremental semantic clustering for extracted problem evidence."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from src.clustering.embeddings import EmbeddingProvider
from src.clustering.similarity import cosine_similarity, mean_embedding
from src.database.models import EvidenceItem, OpportunityCluster
from src.database.repositories import ClusterRepository, EvidenceRepository
from src.logging_config import log_event


logger = logging.getLogger(__name__)


class ClusteringError(RuntimeError):
    """Raised when evidence cannot be assigned to a cluster."""


@dataclass(frozen=True)
class ClusterAssignment:
    """Result of assigning one evidence item."""

    cluster: OpportunityCluster
    similarity_score: float
    created: bool


@dataclass(frozen=True)
class ClusterSummary:
    """Evidence-backed summary derived from all records in a cluster."""

    normalized_problem: str
    target_user: Optional[str]
    current_workaround: Optional[str]
    common_pain_types: list[str]
    independent_source_count: int
    independent_author_count: int
    first_seen_at: Optional[datetime]
    last_seen_at: Optional[datetime]
    representative_excerpts: list[str]


class IncrementalClusterer:
    """Assign evidence to the closest centroid or create a new cluster."""

    def __init__(
        self,
        session: Session,
        embedding_provider: EmbeddingProvider,
        threshold: float = 0.78,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Clustering threshold must be between 0 and 1.")
        self.session = session
        self.embedding_provider = embedding_provider
        self.threshold = threshold
        self.evidence = EvidenceRepository(session)
        self.clusters = ClusterRepository(session)

    def assign(self, evidence_item: EvidenceItem) -> ClusterAssignment:
        """Assign accepted evidence and refresh the resulting cluster summary."""

        if not evidence_item.contains_problem or not evidence_item.problem_statement:
            raise ClusteringError("Only accepted evidence with a problem statement can be clustered.")
        embedding = self._ensure_embedding(evidence_item)

        best_cluster: OpportunityCluster | None = None
        best_similarity = -1.0
        for cluster in self.clusters.list_with_evidence():
            centroid = self._cluster_centroid(cluster)
            if centroid is None or len(centroid) != len(embedding):
                continue
            similarity = cosine_similarity(embedding, centroid)
            if self._shares_scout_workflow(evidence_item, cluster):
                similarity = max(similarity, 0.86)
            if similarity > best_similarity:
                best_cluster = cluster
                best_similarity = similarity

        created = best_cluster is None or best_similarity < self.threshold
        if created:
            best_cluster = self.clusters.create(
                title=self._title_from_problem(evidence_item.problem_statement),
                problem_summary=evidence_item.problem_statement,
                target_customer=evidence_item.affected_user,
                current_workaround=evidence_item.current_workaround,
                proposed_solution=self._conservative_solution(evidence_item),
            )
            best_similarity = 1.0

        self.clusters.link_evidence(best_cluster.id, evidence_item.id, best_similarity)
        refreshed = self.refresh_summary(best_cluster.id)
        log_event(
            logger,
            logging.INFO,
            "cluster_assignment",
            {
                "cluster_id": refreshed.id,
                "evidence_item_id": evidence_item.id,
                "similarity": round(best_similarity, 4),
                "created": created,
            },
        )
        return ClusterAssignment(refreshed, best_similarity, created)

    def refresh_summary(self, cluster_id: str) -> OpportunityCluster:
        """Recompute stored summary fields from linked evidence."""

        cluster = self.clusters.get(cluster_id)
        if cluster is None:
            raise ClusteringError("Cluster does not exist.")
        summary = self.summarize(cluster)
        cluster.problem_summary = summary.normalized_problem
        cluster.target_customer = summary.target_user
        cluster.current_workaround = summary.current_workaround
        cluster.title = self._title_from_problem(summary.normalized_problem)
        if not cluster.proposed_solution and cluster.evidence_links:
            cluster.proposed_solution = self._conservative_solution(
                cluster.evidence_links[0].evidence_item
            )
        return self.clusters.save(cluster)

    def summarize(self, cluster: OpportunityCluster) -> ClusterSummary:
        """Build a traceable cluster summary without inventing missing fields."""

        items = [link.evidence_item for link in cluster.evidence_links]
        if not items:
            raise ClusteringError("Cannot summarize a cluster without evidence.")
        problem_statements = [item.problem_statement for item in items if item.problem_statement]
        target_users = [item.affected_user for item in items if item.affected_user]
        workarounds = [item.current_workaround for item in items if item.current_workaround]
        pain_counts = Counter(pain for item in items for pain in (item.pain_types or []))
        excerpts = []
        for item in items:
            quote = (item.metadata_json or {}).get("evidence_quote")
            excerpt = quote or item.raw_text
            cleaned = " ".join(str(excerpt).split())
            if cleaned and cleaned not in excerpts:
                excerpts.append(cleaned[:280])

        return ClusterSummary(
            normalized_problem=self._most_representative(problem_statements)
            or cluster.problem_summary,
            target_user=self._most_common(target_users),
            current_workaround=self._most_common(workarounds),
            common_pain_types=[pain for pain, _ in pain_counts.most_common(5)],
            independent_source_count=cluster.independent_source_count,
            independent_author_count=cluster.independent_author_count,
            first_seen_at=cluster.first_seen_at,
            last_seen_at=cluster.last_seen_at,
            representative_excerpts=excerpts[:5],
        )

    def centroid_for_cluster(self, cluster_id: str) -> list[float]:
        """Return the current centroid, primarily for diagnostics and tests."""

        cluster = self.clusters.get(cluster_id)
        if cluster is None:
            raise ClusteringError("Cluster does not exist.")
        centroid = self._cluster_centroid(cluster)
        if centroid is None:
            raise ClusteringError("Cluster has no embeddable evidence.")
        return centroid

    def _cluster_centroid(self, cluster: OpportunityCluster) -> list[float] | None:
        vectors: list[list[float]] = []
        changed = False
        for link in cluster.evidence_links:
            item = link.evidence_item
            if not item.problem_statement:
                continue
            if not item.embedding:
                item.embedding = self.embedding_provider.embed(
                    self._embedding_text(item)
                )
                self.session.add(item)
                changed = True
            vectors.append(item.embedding)
        if changed:
            self.session.commit()
        return mean_embedding(vectors) if vectors else None

    def _ensure_embedding(self, evidence_item: EvidenceItem) -> list[float]:
        if not evidence_item.embedding:
            evidence_item.embedding = self.embedding_provider.embed(
                self._embedding_text(evidence_item)
            )
            self.evidence.save(evidence_item)
        return evidence_item.embedding

    @staticmethod
    def _embedding_text(evidence_item: EvidenceItem) -> str:
        problem = evidence_item.problem_statement or ""
        workflow_topic = str(
            (evidence_item.metadata_json or {}).get("scout_workflow_topic") or ""
        ).strip()
        return f"{workflow_topic}. {problem}" if workflow_topic else problem

    @staticmethod
    def _shares_scout_workflow(
        evidence_item: EvidenceItem,
        cluster: OpportunityCluster,
    ) -> bool:
        workflow_topic = str(
            (evidence_item.metadata_json or {}).get("scout_workflow_topic") or ""
        ).strip()
        if not workflow_topic:
            return False
        return any(
            str(
                (link.evidence_item.metadata_json or {}).get("scout_workflow_topic")
                or ""
            ).strip()
            == workflow_topic
            for link in cluster.evidence_links
        )

    @staticmethod
    def _most_common(values: list[str]) -> str | None:
        if not values:
            return None
        return Counter(values).most_common(1)[0][0]

    @staticmethod
    def _most_representative(values: list[str]) -> str | None:
        if not values:
            return None
        frequencies = Counter(value.strip() for value in values)
        return max(frequencies, key=lambda value: (frequencies[value], -len(value)))

    @staticmethod
    def _title_from_problem(problem: str) -> str:
        cleaned = " ".join(problem.strip().rstrip(".!?").split())
        if len(cleaned) > 90:
            cleaned = f"{cleaned[:87].rstrip()}..."
        return cleaned[:1].upper() + cleaned[1:]

    @staticmethod
    def _conservative_solution(evidence_item: EvidenceItem) -> str:
        customer = evidence_item.affected_user or "the affected users"
        return (
            f"A focused workflow tool for {customer} that addresses the documented "
            "manual steps; feasibility and market demand still require validation."
        )
