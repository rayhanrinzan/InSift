"""Evidence confidence scoring kept separate from opportunity quality."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from datetime import datetime, timezone

from src.database.models import Competitor, EvidenceItem
from src.scoring.problem_score import clamp_score
from src.scoring.schemas import ScoreComponent


def calculate_confidence_score(
    evidence_items: Sequence[EvidenceItem],
    competitors: Sequence[Competitor] = (),
    *,
    successful_query_count: int = 0,
    research_complete: bool = False,
) -> ScoreComponent:
    """Calculate confidence from independence, reliability, recency, and agreement."""

    if not evidence_items:
        return ScoreComponent(score=0, reason="No evidence is available to support confidence.")

    independent_sources = {
        item.source_url or item.source_external_id or item.id for item in evidence_items
    }
    authors = {item.source_author for item in evidence_items if item.source_author}
    communities = {
        item.community or item.platform for item in evidence_items if item.community or item.platform
    }
    volume = min(len(independent_sources) / 5, 1.0) * 100
    author_diversity = min(len(authors) / 5, 1.0) * 100
    source_diversity = min(len(communities) / 3, 1.0) * 100
    extraction = (
        sum(item.extraction_confidence for item in evidence_items) / len(evidence_items) * 100
    )

    timestamps = [item.published_at or item.collected_at for item in evidence_items]
    latest = max(timestamps)
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)
    age_days = max(0, (datetime.now(timezone.utc) - latest).days)
    recency = max(0.0, 100.0 - min(age_days, 365) / 365 * 100)

    signals = [
        (item.severity_score + item.frequency_signal) / 2 for item in evidence_items
    ]
    agreement = 100.0 if len(signals) == 1 else max(0.0, 100 - statistics.pstdev(signals) * 150)
    classification_confidence = (
        sum(item.classification_confidence for item in competitors) / len(competitors) * 100
        if competitors
        else 0.0
    )
    search_coverage = min(successful_query_count / 8, 1.0) * 100
    research = (
        (0.6 * classification_confidence) + (0.4 * search_coverage)
        if research_complete
        else 0.0
    )
    score = clamp_score(
        (0.20 * volume)
        + (0.15 * author_diversity)
        + (0.15 * source_diversity)
        + (0.20 * extraction)
        + (0.10 * recency)
        + (0.10 * agreement)
        + (0.10 * research)
    )
    research_note = (
        f"{len(competitors)} classified result(s) and {successful_query_count} successful "
        "query execution(s) contribute to coverage."
        if research_complete
        else "Competitor research is not yet included, which limits confidence."
    )
    return ScoreComponent(
        score=score,
        reason=(
            f"Supported by {len(independent_sources)} independent source(s), "
            f"{len(authors)} author(s), and {len(communities)} community/site group(s). "
            f"{research_note}"
        ),
        inputs={
            "independent_sources": len(independent_sources),
            "independent_authors": len(authors),
            "communities_or_platforms": len(communities),
            "mean_extraction_confidence": round(extraction, 2),
            "evidence_age_days": age_days,
            "source_agreement": round(agreement, 2),
            "competitor_records": len(competitors),
            "successful_search_queries": successful_query_count,
            "competitor_classification_confidence": round(
                classification_confidence, 2
            ),
        },
    )
