"""Tests for deterministic filtering and structured extraction."""

from typing import Any

import pytest

from src.extraction.problem_extractor import (
    DeterministicMockExtractionProvider,
    ExtractionError,
    ProblemExtractor,
)


class StaticProvider:
    """Provider test double returning one configured value."""

    def __init__(self, value: Any) -> None:
        self.value = value
        self.called = False

    def extract_problem(self, text: str, prompt: str) -> Any:
        self.called = True
        return self.value


def test_real_problem_detected() -> None:
    extractor = ProblemExtractor(DeterministicMockExtractionProvider())

    result = extractor.extract(
        "As a clinic manager, we still use spreadsheets for follow-up. "
        "It takes hours every week and referrals get missed."
    )

    assert result.contains_real_problem is True
    assert result.problem_statement
    assert result.evidence_quote in (
        "As a clinic manager, we still use spreadsheets for follow-up.",
        "It takes hours every week and referrals get missed.",
    )
    assert "time" in result.pain_types
    assert result.confidence >= 0.45


def test_non_problem_is_rejected_before_provider_call() -> None:
    provider = StaticProvider({"contains_real_problem": True})
    result = ProblemExtractor(provider).extract("I launched a new product today and it is blue.")

    assert result.contains_real_problem is False
    assert provider.called is False


def test_missing_optional_fields_are_handled() -> None:
    provider = StaticProvider({"contains_real_problem": True, "confidence": 0.7})

    result = ProblemExtractor(provider).extract("Our manual process is frustrating.")

    assert result.problem_statement is None
    assert result.pain_types == []
    assert result.has_usable_problem is False


def test_invalid_llm_json_raises_safe_error() -> None:
    extractor = ProblemExtractor(StaticProvider("{not valid JSON"))

    with pytest.raises(ExtractionError, match="invalid JSON"):
        extractor.extract("This manual process takes forever.")


def test_low_confidence_extraction_is_preserved_for_review() -> None:
    result = ProblemExtractor(
        StaticProvider(
            {
                "contains_real_problem": True,
                "problem_statement": "A manual workflow is slow",
                "confidence": 0.12,
            }
        )
    ).extract("The manual workflow is slow and frustrating.")

    assert result.has_usable_problem is True
    assert result.confidence == 0.12
