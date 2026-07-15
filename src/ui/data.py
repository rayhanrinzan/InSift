"""Cached read models for the Streamlit application."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import streamlit as st
from sqlalchemy.orm import sessionmaker

from src.config import get_settings
from src.database.models import EvidenceItem
from src.database.repositories import EvidenceRepository
from src.database.session import (
    create_database_engine,
    create_session_factory,
    initialize_database,
)
from src.services.opportunity_service import (
    DashboardMetrics,
    OpportunityService,
    RankedOpportunity,
)


CACHE_TTL_SECONDS = 30


@dataclass(frozen=True)
class EvidenceSummary:
    """Serializable evidence fields used by list and review screens."""

    id: str
    title: Optional[str]
    platform: str
    community: Optional[str]
    source_url: Optional[str]
    source_author: Optional[str]
    raw_text: str
    contains_problem: bool
    problem_statement: Optional[str]
    extraction_confidence: float
    pain_types: tuple[str, ...]
    collected_at: Optional[datetime]


@dataclass(frozen=True)
class DashboardSnapshot:
    """Cached home-page aggregates and recent activity."""

    metrics: DashboardMetrics
    opportunities: tuple[RankedOpportunity, ...]
    recent_evidence: tuple[EvidenceSummary, ...]


def _evidence_summary(item: EvidenceItem) -> EvidenceSummary:
    return EvidenceSummary(
        id=item.id,
        title=item.title,
        platform=item.platform,
        community=item.community,
        source_url=item.source_url,
        source_author=item.source_author,
        raw_text=item.raw_text,
        contains_problem=item.contains_problem,
        problem_statement=item.problem_statement,
        extraction_confidence=float(item.extraction_confidence),
        pain_types=tuple(item.pain_types or []),
        collected_at=item.collected_at,
    )


@st.cache_resource(show_spinner=False)
def get_ui_session_factory(database_url: str) -> sessionmaker:
    """Initialize and reuse one engine and session factory per database URL."""

    settings = get_settings().copy(update={"database_url": database_url})
    engine = create_database_engine(settings)
    initialize_database(engine)
    return create_session_factory(engine)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_dashboard_snapshot(database_url: str) -> DashboardSnapshot:
    """Load cached metrics, top opportunities, and recent evidence."""

    SessionFactory = get_ui_session_factory(database_url)
    with SessionFactory() as session:
        service = OpportunityService(session)
        return DashboardSnapshot(
            metrics=service.dashboard_metrics(),
            opportunities=tuple(service.ranked_opportunities(limit=10)),
            recent_evidence=tuple(
                _evidence_summary(item) for item in service.recent_evidence(limit=8)
            ),
        )


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_ranked_opportunities(
    database_url: str,
    *,
    limit: int = 1000,
) -> tuple[RankedOpportunity, ...]:
    """Load cached opportunity rows for filtering and pagination."""

    SessionFactory = get_ui_session_factory(database_url)
    with SessionFactory() as session:
        return tuple(OpportunityService(session).ranked_opportunities(limit=limit))


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_evidence_review(
    database_url: str,
    *,
    limit: int = 500,
) -> tuple[EvidenceSummary, ...]:
    """Load cached accepted and rejected evidence summaries."""

    SessionFactory = get_ui_session_factory(database_url)
    with SessionFactory() as session:
        items = EvidenceRepository(session).list_recent(limit=limit)
        return tuple(_evidence_summary(item) for item in items)


def clear_ui_data_caches() -> None:
    """Invalidate read snapshots after any persisted write."""

    load_dashboard_snapshot.clear()
    load_ranked_opportunities.clear()
    load_evidence_review.clear()
