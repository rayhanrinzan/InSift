"""Tests for conservative competitor relationship classification."""

from src.research.competitor_classifier import CompetitorClassifier
from src.research.schemas import CompetitorResearchContext, SearchResult


CONTEXT = CompetitorResearchContext(
    title="Clinic referral follow-up",
    problem_summary="Clinics manually track patient referral follow-up",
    target_customer="small clinic operations managers",
)


def test_direct_competitor_classified_correctly() -> None:
    result = SearchResult(
        title="ReferralFlow for clinic operations managers",
        url="https://referralflow.example",
        snippet="Automates patient referral follow-up for small clinic operations managers.",
    )

    classification = CompetitorClassifier().classify(CONTEXT, result)

    assert classification.relationship_type == "direct"
    assert classification.similarity_score >= 0.7


def test_adjacent_competitor_classified_correctly() -> None:
    result = SearchResult(
        title="Patient Intake Forms",
        url="https://intake.example",
        snippet="Digital patient intake paperwork for healthcare practices.",
    )

    classification = CompetitorClassifier().classify(CONTEXT, result)

    assert classification.relationship_type == "adjacent"


def test_spreadsheet_is_classified_as_substitute() -> None:
    result = SearchResult(
        title="Microsoft Excel",
        url="https://microsoft.com/excel",
        snippet="A spreadsheet used to manually track referrals and follow-up.",
    )

    classification = CompetitorClassifier().classify(CONTEXT, result)

    assert classification.relationship_type == "substitute"


def test_irrelevant_result_is_rejected() -> None:
    result = SearchResult(
        title="Bread recipe collection",
        url="https://recipes.example/bread",
        snippet="Recipes for sourdough bread and pastries.",
    )

    classification = CompetitorClassifier().classify(CONTEXT, result)

    assert classification.relationship_type == "irrelevant"
