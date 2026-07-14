"""SQLAlchemy models for InSift's core entities."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base


def new_id() -> str:
    """Return a compact UUID string for portable primary keys."""

    return str(uuid.uuid4())


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


class RelationshipType(str, Enum):
    """Allowed competitor relationship classifications."""

    DIRECT = "direct"
    ADJACENT = "adjacent"
    SUBSTITUTE = "substitute"
    IRRELEVANT = "irrelevant"


class OpportunityStatus(str, Enum):
    """Lifecycle status for an opportunity cluster."""

    NEW = "new"
    RESEARCHED = "researched"
    ARCHIVED = "archived"


class FeedbackType(str, Enum):
    """Supported user feedback categories."""

    CORRECTION = "correction"
    REJECTION = "rejection"
    NOTE = "note"


class ResearchStatus(str, Enum):
    """Lifecycle status for competitor research operations."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class EvidenceItem(Base):
    """A single source item containing potential problem evidence."""

    __tablename__ = "evidence_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    platform: Mapped[str] = mapped_column(String(80), nullable=False, default="manual")
    source_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    source_external_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_author: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    community: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    engagement_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    contains_problem: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    extraction_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    problem_statement: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    affected_user: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    current_workaround: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pain_types: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    severity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    frequency_signal: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    willingness_to_pay_score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    embedding: Mapped[Optional[list[float]]] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    cluster_links: Mapped[list[ClusterEvidence]] = relationship(
        "ClusterEvidence", back_populates="evidence_item", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_evidence_items_source_url", "source_url"),
        Index("ix_evidence_items_source_external_id", "source_external_id"),
        Index("ix_evidence_items_contains_problem", "contains_problem"),
    )


class OpportunityCluster(Base):
    """A semantic cluster of related problem evidence."""

    __tablename__ = "opportunity_clusters"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    problem_summary: Mapped[str] = mapped_column(Text, nullable=False)
    target_customer: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    current_workaround: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    proposed_solution: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evidence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    independent_author_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    independent_source_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(
        String(40), nullable=False, default=OpportunityStatus.NEW.value
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    evidence_links: Mapped[list[ClusterEvidence]] = relationship(
        "ClusterEvidence", back_populates="cluster", cascade="all, delete-orphan"
    )
    competitors: Mapped[list[Competitor]] = relationship(
        "Competitor", back_populates="cluster", cascade="all, delete-orphan"
    )
    scores: Mapped[list[OpportunityScore]] = relationship(
        "OpportunityScore", back_populates="cluster", cascade="all, delete-orphan"
    )
    research_runs: Mapped[list[ResearchRun]] = relationship(
        "ResearchRun", back_populates="cluster", cascade="all, delete-orphan"
    )
    search_queries: Mapped[list[SearchQuery]] = relationship(
        "SearchQuery", back_populates="cluster", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_opportunity_clusters_status", "status"),)


class ClusterEvidence(Base):
    """Association between a cluster and the evidence supporting it."""

    __tablename__ = "cluster_evidence"

    cluster_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("opportunity_clusters.id"), primary_key=True
    )
    evidence_item_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("evidence_items.id"), primary_key=True
    )
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    cluster: Mapped[OpportunityCluster] = relationship(
        "OpportunityCluster", back_populates="evidence_links"
    )
    evidence_item: Mapped[EvidenceItem] = relationship(
        "EvidenceItem", back_populates="cluster_links"
    )


class Competitor(Base):
    """A product, substitute, or irrelevant search result for a cluster."""

    __tablename__ = "competitors"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    cluster_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("opportunity_clusters.id"), nullable=False
    )
    company_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    product_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    relationship_type: Mapped[str] = mapped_column(String(40), nullable=False)
    target_customer: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    problem_solved: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    features: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    pricing_position: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    strengths: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    weaknesses: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    possible_gap: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    classification_confidence: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    source_evidence: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    cluster: Mapped[OpportunityCluster] = relationship(
        "OpportunityCluster", back_populates="competitors"
    )

    __table_args__ = (
        Index("ix_competitors_cluster_id", "cluster_id"),
        UniqueConstraint("cluster_id", "url", name="uq_competitor_cluster_url"),
    )


class ResearchRun(Base):
    """One auditable competitor-research execution for a cluster."""

    __tablename__ = "research_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    cluster_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("opportunity_clusters.id"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(
        String(40), nullable=False, default=ResearchStatus.PENDING.value
    )
    query_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    result_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    relevant_competitor_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    cluster: Mapped[OpportunityCluster] = relationship(
        "OpportunityCluster", back_populates="research_runs"
    )
    queries: Mapped[list[SearchQuery]] = relationship(
        "SearchQuery", back_populates="research_run", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_research_runs_cluster_id", "cluster_id"),)


class SearchQuery(Base):
    """A generated search query and its execution outcome."""

    __tablename__ = "search_queries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    research_run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("research_runs.id"), nullable=False
    )
    cluster_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("opportunity_clusters.id"), nullable=False
    )
    query_text: Mapped[str] = mapped_column(String(700), nullable=False)
    status: Mapped[str] = mapped_column(
        String(40), nullable=False, default=ResearchStatus.PENDING.value
    )
    result_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    research_run: Mapped[ResearchRun] = relationship(
        "ResearchRun", back_populates="queries"
    )
    cluster: Mapped[OpportunityCluster] = relationship(
        "OpportunityCluster", back_populates="search_queries"
    )

    __table_args__ = (
        Index("ix_search_queries_cluster_id", "cluster_id"),
        UniqueConstraint(
            "research_run_id", "query_text", name="uq_search_query_run_text"
        ),
    )


class OpportunityScore(Base):
    """Versioned score breakdown for an opportunity cluster."""

    __tablename__ = "opportunity_scores"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    cluster_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("opportunity_clusters.id"), nullable=False
    )
    pain_severity_score: Mapped[float] = mapped_column(Float, nullable=False)
    problem_frequency_score: Mapped[float] = mapped_column(Float, nullable=False)
    willingness_to_pay_score: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    whitespace_score: Mapped[float] = mapped_column(Float, nullable=False)
    build_feasibility_score: Mapped[float] = mapped_column(Float, nullable=False)
    market_accessibility_score: Mapped[float] = mapped_column(Float, nullable=False)
    opportunity_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    scoring_version: Mapped[str] = mapped_column(String(80), nullable=False, default="v1")
    explanation_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    cluster: Mapped[OpportunityCluster] = relationship(
        "OpportunityCluster", back_populates="scores"
    )

    __table_args__ = (Index("ix_opportunity_scores_cluster_id", "cluster_id"),)


class UserFeedback(Base):
    """Auditable user correction or note on an AI-generated field."""

    __tablename__ = "user_feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    entity_type: Mapped[str] = mapped_column(String(80), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(36), nullable=False)
    field_name: Mapped[str] = mapped_column(String(120), nullable=False)
    original_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    corrected_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    feedback_type: Mapped[str] = mapped_column(
        String(80), nullable=False, default=FeedbackType.CORRECTION.value
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    __table_args__ = (Index("ix_user_feedback_entity", "entity_type", "entity_id"),)
