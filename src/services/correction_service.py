"""Auditable user corrections with cluster maintenance and rescoring."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from sqlalchemy.orm import Session

from src.clustering.clusterer import IncrementalClusterer
from src.clustering.embeddings import build_embedding_provider
from src.config import Settings
from src.database.models import EvidenceItem, OpportunityCluster, RelationshipType
from src.database.repositories import (
    ClusterRepository,
    CompetitorRepository,
    EvidenceRepository,
    FeedbackRepository,
)
from src.logging_config import log_event
from src.research.competitor_search import canonical_url
from src.scoring.opportunity_score import OpportunityScorer


logger = logging.getLogger(__name__)


def _audit_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple, bool, int, float)):
        return json.dumps(value, sort_keys=True)
    return str(value)


class CorrectionService:
    """Apply corrections while preserving original values and dependent calculations."""

    def __init__(self, session: Session, clusterer: IncrementalClusterer) -> None:
        self.session = session
        self.clusterer = clusterer
        self.evidence = EvidenceRepository(session)
        self.clusters = ClusterRepository(session)
        self.competitors = CompetitorRepository(session)
        self.feedback = FeedbackRepository(session)
        self.scorer = OpportunityScorer(session)

    def correct_evidence(
        self,
        evidence_id: str,
        *,
        contains_problem: bool,
        problem_statement: Optional[str],
        affected_user: Optional[str],
        current_workaround: Optional[str],
        pain_types: list[str],
        severity_score: float,
        frequency_signal: float,
        willingness_to_pay_score: float,
    ) -> EvidenceItem:
        """Correct structured extraction fields and update affected clusters."""

        item = self.evidence.get(evidence_id)
        if item is None:
            raise ValueError("Evidence item does not exist.")
        if contains_problem and not (problem_statement or "").strip():
            raise ValueError("Accepted evidence needs a problem statement.")
        cluster_ids = self.clusters.cluster_ids_for_evidence(evidence_id)
        original_problem_statement = item.problem_statement
        changes = {
            "contains_problem": contains_problem,
            "problem_statement": (problem_statement or "").strip() or None,
            "affected_user": (affected_user or "").strip() or None,
            "current_workaround": (current_workaround or "").strip() or None,
            "pain_types": sorted(set(pain_types)),
            "severity_score": severity_score,
            "frequency_signal": frequency_signal,
            "willingness_to_pay_score": willingness_to_pay_score,
        }
        for field_name, corrected_value in changes.items():
            original_value = getattr(item, field_name)
            if original_value == corrected_value:
                continue
            self.feedback.create(
                entity_type="evidence_item",
                entity_id=item.id,
                field_name=field_name,
                original_value=_audit_value(original_value),
                corrected_value=_audit_value(corrected_value),
                feedback_type="correction" if contains_problem else "rejection",
            )
            setattr(item, field_name, corrected_value)
        if original_problem_statement != changes["problem_statement"]:
            item.embedding = None
        item.extraction_confidence = 1.0
        stored = self.evidence.save(item)

        if not contains_problem:
            for cluster_id in cluster_ids:
                self.clusters.unlink_evidence(cluster_id, evidence_id)
                self._refresh_or_archive(cluster_id)
        elif cluster_ids:
            for cluster_id in cluster_ids:
                self.clusterer.refresh_summary(cluster_id)
                self.scorer.score_cluster(cluster_id)
        else:
            assignment = self.clusterer.assign(stored)
            self.scorer.score_cluster(assignment.cluster.id)
        self._log("evidence_item", evidence_id, "structured_extraction")
        return stored

    def update_target_customer(self, cluster_id: str, target_customer: str) -> OpportunityCluster:
        """Correct a cluster target customer and rescore it."""

        cluster = self._cluster(cluster_id)
        cleaned = target_customer.strip()
        if not cleaned:
            raise ValueError("Target customer cannot be empty.")
        self._record_change(
            "opportunity_cluster", cluster.id, "target_customer", cluster.target_customer, cleaned
        )
        cluster.target_customer = cleaned
        stored = self.clusters.save(cluster)
        self.scorer.score_cluster(cluster_id)
        self._log("opportunity_cluster", cluster_id, "target_customer")
        return stored

    def reclassify_competitor(self, competitor_id: str, relationship_type: str) -> None:
        """Correct a competitor relationship and preserve it on later research reruns."""

        allowed = {item.value for item in RelationshipType}
        if relationship_type not in allowed:
            raise ValueError("Unsupported competitor relationship type.")
        competitor = self.competitors.get(competitor_id)
        if competitor is None:
            raise ValueError("Competitor does not exist.")
        self._record_change(
            "competitor",
            competitor.id,
            "relationship_type",
            competitor.relationship_type,
            relationship_type,
        )
        competitor.relationship_type = relationship_type
        source_evidence = dict(competitor.source_evidence or {})
        source_evidence["user_corrected_relationship"] = True
        competitor.source_evidence = source_evidence
        self.competitors.save(competitor)
        self.scorer.score_cluster(competitor.cluster_id)
        self._log("competitor", competitor_id, "relationship_type")

    def merge_clusters(self, source_cluster_id: str, target_cluster_id: str) -> OpportunityCluster:
        """Move unique evidence and competitors into a target, then archive the source."""

        if source_cluster_id == target_cluster_id:
            raise ValueError("Choose two different clusters to merge.")
        source = self._cluster(source_cluster_id)
        target = self._cluster(target_cluster_id)
        target_urls = {
            canonical_url(item.url) for item in target.competitors if item.url
        }
        target_products = {
            item.product_name.casefold() for item in target.competitors if item.product_name
        }
        for link in list(source.evidence_links):
            self.clusters.link_evidence(
                target.id, link.evidence_item_id, link.similarity_score
            )
            self.clusters.unlink_evidence(source.id, link.evidence_item_id)
        for competitor in list(source.competitors):
            duplicate = bool(
                (competitor.url and canonical_url(competitor.url) in target_urls)
                or (
                    competitor.product_name
                    and competitor.product_name.casefold() in target_products
                )
            )
            if duplicate:
                continue
            competitor.cluster_id = target.id
            self.competitors.save(competitor)
            if competitor.url:
                target_urls.add(canonical_url(competitor.url))
            if competitor.product_name:
                target_products.add(competitor.product_name.casefold())
        source.status = "archived"
        self.clusters.save(source)
        self.feedback.create(
            entity_type="opportunity_cluster",
            entity_id=source.id,
            field_name="merged_into",
            original_value=source.id,
            corrected_value=target.id,
            feedback_type="correction",
        )
        refreshed = self.clusterer.refresh_summary(target.id)
        self.scorer.score_cluster(target.id)
        self._log("opportunity_cluster", source.id, "merged_into")
        return refreshed

    def split_cluster(
        self,
        source_cluster_id: str,
        evidence_ids: list[str],
        *,
        title: Optional[str] = None,
    ) -> OpportunityCluster:
        """Move selected evidence into a new independently scored cluster."""

        source = self._cluster(source_cluster_id)
        selected = set(evidence_ids)
        links = [link for link in source.evidence_links if link.evidence_item_id in selected]
        if not links:
            raise ValueError("Select at least one evidence item to split.")
        first = links[0].evidence_item
        new_cluster = self.clusters.create(
            title=(title or first.problem_statement or "Split opportunity")[:255],
            problem_summary=first.problem_statement or source.problem_summary,
            target_customer=first.affected_user or source.target_customer,
            current_workaround=first.current_workaround,
            proposed_solution=source.proposed_solution,
        )
        for link in links:
            self.clusters.link_evidence(
                new_cluster.id, link.evidence_item_id, link.similarity_score
            )
            self.clusters.unlink_evidence(source.id, link.evidence_item_id)
        refreshed_new = self.clusterer.refresh_summary(new_cluster.id)
        if title and title.strip():
            refreshed_new.title = title.strip()[:255]
            refreshed_new = self.clusters.save(refreshed_new)
        self._refresh_or_archive(source.id)
        self.scorer.score_cluster(new_cluster.id)
        self.feedback.create(
            entity_type="opportunity_cluster",
            entity_id=source.id,
            field_name="split_evidence",
            original_value=_audit_value(sorted(selected)),
            corrected_value=new_cluster.id,
            feedback_type="correction",
        )
        self._log("opportunity_cluster", source.id, "split_evidence")
        return refreshed_new

    def _refresh_or_archive(self, cluster_id: str) -> None:
        cluster = self._cluster(cluster_id)
        if not cluster.evidence_links:
            cluster.status = "archived"
            self.clusters.save(cluster)
            return
        self.clusterer.refresh_summary(cluster_id)
        self.scorer.score_cluster(cluster_id)

    def _cluster(self, cluster_id: str) -> OpportunityCluster:
        cluster = self.clusters.get(cluster_id)
        if cluster is None:
            raise ValueError("Cluster does not exist.")
        return cluster

    def _record_change(
        self,
        entity_type: str,
        entity_id: str,
        field_name: str,
        original_value: Any,
        corrected_value: Any,
    ) -> None:
        if original_value == corrected_value:
            return
        self.feedback.create(
            entity_type=entity_type,
            entity_id=entity_id,
            field_name=field_name,
            original_value=_audit_value(original_value),
            corrected_value=_audit_value(corrected_value),
            feedback_type="correction",
        )

    @staticmethod
    def _log(entity_type: str, entity_id: str, field_name: str) -> None:
        log_event(
            logger,
            logging.INFO,
            "user_correction",
            {"entity_type": entity_type, "entity_id": entity_id, "field_name": field_name},
        )


def build_correction_service(session: Session, settings: Settings) -> CorrectionService:
    """Build correction dependencies from centralized settings."""

    clusterer = IncrementalClusterer(
        session,
        build_embedding_provider(settings),
        threshold=settings.cluster_similarity_threshold,
    )
    return CorrectionService(session, clusterer)
