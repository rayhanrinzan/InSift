"""Problem Score and evidence-quality calculations."""

from __future__ import annotations

from collections.abc import Sequence

from src.database.models import EvidenceItem
from src.scoring.schemas import ProblemScoreBreakdown, ProblemScoreWeights, ScoreComponent


def clamp_score(value: float) -> float:
    """Clamp and round a score to the 0-100 range."""

    return round(max(0.0, min(100.0, value)), 2)


def _average(items: Sequence[EvidenceItem], field_name: str) -> float:
    if not items:
        return 0.0
    return sum(float(getattr(item, field_name) or 0.0) for item in items) / len(items)


def calculate_evidence_quality(evidence_items: Sequence[EvidenceItem]) -> ScoreComponent:
    """Score extraction reliability, source independence, and attribution."""

    if not evidence_items:
        return ScoreComponent(score=0, reason="No supporting evidence is linked.")
    extraction = _average(evidence_items, "extraction_confidence") * 100
    unique_sources = len(
        {
            item.source_url or item.source_external_id or item.id
            for item in evidence_items
        }
    )
    unique_authors = len({item.source_author for item in evidence_items if item.source_author})
    independence = min(unique_sources / 3, 1.0) * 60 + min(unique_authors / 3, 1.0) * 40
    attribution = (
        sum(bool(item.source_url or item.source_author or item.community) for item in evidence_items)
        / len(evidence_items)
        * 100
    )
    score = clamp_score((0.60 * extraction) + (0.25 * independence) + (0.15 * attribution))
    return ScoreComponent(
        score=score,
        reason=(
            f"Based on {len(evidence_items)} linked item(s), {unique_sources} independent "
            f"source(s), {unique_authors} named author(s), and {extraction:.0f}% mean "
            "extraction confidence."
        ),
        inputs={
            "evidence_items": len(evidence_items),
            "independent_sources": unique_sources,
            "independent_authors": unique_authors,
            "mean_extraction_confidence": round(extraction, 2),
        },
    )


def calculate_problem_score(
    evidence_items: Sequence[EvidenceItem],
    weights: ProblemScoreWeights | None = None,
) -> ProblemScoreBreakdown:
    """Calculate the weighted Problem Score with explanations."""

    weights = weights or ProblemScoreWeights()
    count = len(evidence_items)
    unique_sources = len(
        {item.source_url or item.source_external_id or item.id for item in evidence_items}
    )
    severity = clamp_score(_average(evidence_items, "severity_score") * 100)
    raw_frequency = _average(evidence_items, "frequency_signal") * 100
    corroboration = min(unique_sources / 5, 1.0) * 100
    frequency = clamp_score((0.70 * raw_frequency) + (0.30 * corroboration))
    willingness = clamp_score(_average(evidence_items, "willingness_to_pay_score") * 100)
    evidence_quality = calculate_evidence_quality(evidence_items)

    severity_component = ScoreComponent(
        score=severity,
        reason=f"Mean evidence-backed severity across {count} linked item(s).",
        inputs={"linked_items": count},
    )
    frequency_component = ScoreComponent(
        score=frequency,
        reason=(
            f"Combines explicit recurrence signals with corroboration across "
            f"{unique_sources} independent source(s)."
        ),
        inputs={"mean_frequency_signal": round(raw_frequency, 2), "independent_sources": unique_sources},
    )
    willingness_component = ScoreComponent(
        score=willingness,
        reason=(
            "Uses only explicit payment, cost, or money-loss signals found during extraction; "
            "missing signals remain low."
        ),
        inputs={"linked_items": count},
    )
    problem_score = clamp_score(
        (weights.pain_severity * severity)
        + (weights.problem_frequency * frequency)
        + (weights.willingness_to_pay * willingness)
        + (weights.evidence_quality * evidence_quality.score)
    )
    return ProblemScoreBreakdown(
        pain_severity=severity_component,
        problem_frequency=frequency_component,
        willingness_to_pay=willingness_component,
        evidence_quality=evidence_quality,
        problem_score=ScoreComponent(
            score=problem_score,
            reason=(
                "Weighted from pain severity (35%), problem frequency (25%), willingness "
                "to pay (20%), and evidence quality (20%)."
            ),
        ),
    )
