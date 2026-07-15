"""Tests for conservative competitor relationship classification."""

from src.research.competitor_classifier import (
    CompetitorClassifier,
    ResilientCompetitorClassifier,
    is_product_candidate,
)
from src.research.competitor_search import SearchProviderError
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


def test_live_classifier_falls_back_after_provider_failure() -> None:
    class FailingClassifier:
        calls = 0

        def classify(self, context, result):
            del context, result
            self.calls += 1
            raise SearchProviderError("OpenAI rate limit reached.")

    primary = FailingClassifier()
    classifier = ResilientCompetitorClassifier(primary, CompetitorClassifier())
    result = SearchResult(
        title="Microsoft Excel",
        url="https://microsoft.com/excel",
        snippet="A spreadsheet used to manually track referrals and follow-up.",
    )

    first = classifier.classify(CONTEXT, result)
    second = classifier.classify(CONTEXT, result)

    assert first.relationship_type == "substitute"
    assert second.relationship_type == "substitute"
    assert primary.calls == 1


def test_product_candidate_filter_rejects_roundups_and_social_posts() -> None:
    roundup = SearchResult(
        title="The Best Order Tracking Tools for eCommerce",
        url="https://ecommercetech.io/categories/order-tracking",
        snippet="A directory of order tracking tools.",
    )
    social = SearchResult(
        title="My favorite tracking app",
        url="https://www.reddit.com/r/ecommerce/comments/123/tracking/",
        snippet="A discussion about tracking software.",
    )
    company_blog = SearchResult(
        title="Kintone: the Excel Alternative That'll Boost Collaboration",
        url="https://blog.kintone.com/company-news/kintone-excel-alternative",
        snippet="A company article about replacing spreadsheets.",
    )
    use_case = SearchResult(
        title="Build ecommerce order tracking software with AI",
        url="https://glideapps.com/use-cases/ecommerce-order-tracking-software",
        snippet="Use a general app builder to create tracking software.",
    )
    research_paper = SearchResult(
        title="SaaS order tracking architecture",
        url="https://researchgate.net/figure/saas-order-tracking_fig1",
        snippet="A figure from an academic paper.",
    )
    product = SearchResult(
        title="TrackFlow Reviews 2026",
        url="https://www.g2.com/products/trackflow/reviews",
        snippet="Order tracking automation software for ecommerce teams.",
    )

    assert not is_product_candidate(roundup)
    assert not is_product_candidate(social)
    assert not is_product_candidate(company_blog)
    assert not is_product_candidate(use_case)
    assert not is_product_candidate(research_paper)
    assert is_product_candidate(product)


def test_generic_email_product_is_not_counted_for_order_tracking() -> None:
    context = CompetitorResearchContext(
        title="Manual order tracking emails take hours every day",
        problem_summary="An ecommerce team manually answers repetitive order tracking emails.",
        target_customer="ecommerce operations teams",
    )
    generic_email = SearchResult(
        title="Atomic Mail Agentic",
        url="https://producthunt.com/products/atomic-mail-agentic",
        snippet="Let software agents read, send, and react to email autonomously.",
    )
    actual_match = SearchResult(
        title="TrackFlow order tracking communication",
        url="https://trackflow.example",
        snippet="Order tracking communication automation for ecommerce operations teams.",
    )

    assert (
        CompetitorClassifier().classify(context, generic_email).relationship_type
        == "irrelevant"
    )
    assert (
        CompetitorClassifier().classify(context, actual_match).relationship_type
        == "direct"
    )
