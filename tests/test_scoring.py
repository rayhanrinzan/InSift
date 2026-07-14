"""Tests for explainable Phase 4 scoring."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.database.models import Competitor, EvidenceItem
from src.scoring.confidence_score import calculate_confidence_score
from src.scoring.opportunity_score import calculate_opportunity_score
from src.scoring.schemas import ProblemScoreWeights


def _evidence(
    index: int,
    *,
    severity: float = 0.6,
    frequency: float = 0.5,
    willingness: float = 0.3,
) -> EvidenceItem:
    return EvidenceItem(
        id=f"evidence-{index}",
        platform="manual",
        source_external_id=f"source-{index}",
        source_author=f"author-{index}",
        community=f"community-{index % 3}",
        raw_text="Manual work takes hours every week.",
        collected_at=datetime.now(timezone.utc),
        contains_problem=True,
        extraction_confidence=0.8,
        problem_statement="Manual work takes hours",
        pain_types=["time", "repetitive_work"],
        severity_score=severity,
        frequency_signal=frequency,
        willingness_to_pay_score=willingness,
        metadata_json={},
    )


def test_scores_remain_between_zero_and_one_hundred() -> None:
    result = calculate_opportunity_score([_evidence(1)])

    numeric_scores = [
        result.pain_severity_score,
        result.problem_frequency_score,
        result.willingness_to_pay_score,
        result.evidence_quality_score,
        result.whitespace_score,
        result.opportunity_score,
        result.confidence_score,
    ]
    assert all(0 <= score <= 100 for score in numeric_scores)


def test_weight_totals_are_validated() -> None:
    with pytest.raises(ValidationError, match="must total 1.0"):
        ProblemScoreWeights(
            pain_severity=0.5,
            problem_frequency=0.5,
            willingness_to_pay=0.5,
            evidence_quality=0.5,
        )


def test_higher_pain_increases_opportunity_score() -> None:
    low = calculate_opportunity_score([_evidence(1, severity=0.1)])
    high = calculate_opportunity_score([_evidence(2, severity=0.95)])

    assert high.opportunity_score > low.opportunity_score


def test_more_competitors_do_not_automatically_reduce_initial_whitespace() -> None:
    item = _evidence(1)
    competitor = Competitor(
        cluster_id="cluster",
        relationship_type="direct",
        classification_confidence=0.8,
    )

    without_competitor = calculate_opportunity_score([item])
    with_competitor = calculate_opportunity_score([item], [competitor])

    assert without_competitor.whitespace_score == 50
    assert with_competitor.whitespace_score == 50


def test_low_evidence_reduces_confidence() -> None:
    low = calculate_confidence_score([_evidence(1)])
    high = calculate_confidence_score([_evidence(index) for index in range(1, 6)])

    assert high.score > low.score


def test_scoring_explanations_are_generated_for_every_component() -> None:
    result = calculate_opportunity_score([_evidence(1)])

    expected = {
        "problem_score",
        "pain_severity",
        "problem_frequency",
        "willingness_to_pay",
        "evidence_quality",
        "whitespace",
        "build_feasibility",
        "market_accessibility",
        "opportunity",
        "confidence",
    }
    assert set(result.explanation_json) == expected
    assert all(value["reason"] for value in result.explanation_json.values())


def test_no_competitors_after_research_is_not_automatically_high_whitespace() -> None:
    result = calculate_opportunity_score(
        [_evidence(1, severity=0.9)],
        research_complete=True,
        successful_query_count=10,
        target_customer="clinic operations managers",
        proposed_solution="A follow-up queue",
    )

    assert result.whitespace_score < 60


def test_supported_competitor_gaps_can_create_high_whitespace() -> None:
    competitors = [
        Competitor(
            cluster_id="cluster",
            relationship_type="direct",
            classification_confidence=0.9,
            weaknesses=["opaque pricing", "limited integrations"],
            possible_gap="Affordable workflow for independent clinics",
        ),
        Competitor(
            cluster_id="cluster",
            relationship_type="adjacent",
            classification_confidence=0.85,
            weaknesses=["not focused on follow-up ownership"],
            possible_gap="Post-intake accountability",
        ),
    ]

    result = calculate_opportunity_score(
        [_evidence(index, severity=0.85) for index in range(1, 4)],
        competitors,
        research_complete=True,
        successful_query_count=10,
        target_customer="independent clinic operations managers",
        proposed_solution="A focused follow-up queue",
    )

    assert result.whitespace_score >= 65
    assert "competitor_weakness" in result.explanation_json
    assert "low_direct_competitor_density" in result.explanation_json
