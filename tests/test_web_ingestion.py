"""Tests for public web evidence discovery and normalization."""

import pytest

from src.ingestion.manual import IngestionError
from src.ingestion.web import (
    WebEvidenceDiscoveryService,
    candidate_from_search_result,
    generate_evidence_queries,
)
from src.research.competitor_search import MockSearchProvider
from src.research.schemas import SearchResult
from src.services.problem_scout_service import (
    build_problem_query,
    select_customer_segments,
)


class StaticSearchProvider:
    """Return the same bounded results for every generated query."""

    name = "static"

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.queries: list[str] = []

    def search(
        self,
        query: str,
        *,
        max_results: int,
        search_depth: str,
    ) -> list[SearchResult]:
        del search_depth
        self.queries.append(query)
        return self.results[:max_results]


def test_evidence_queries_target_selected_public_sources() -> None:
    queries = generate_evidence_queries(
        "manual invoicing",
        target_customer="small agencies",
        source_types=("forums", "issues"),
    )

    assert len(queries) == 2
    assert all('"manual invoicing"' in query for query in queries)
    assert all('"small agencies"' in query for query in queries)
    assert any("forum community discussion" in query for query in queries)
    assert any("public issue report" in query for query in queries)
    assert all(len(query) < 400 for query in queries)


def test_evidence_queries_require_topic_and_source() -> None:
    with pytest.raises(IngestionError, match="Enter a market"):
        generate_evidence_queries(" ")
    with pytest.raises(IngestionError, match="at least one source"):
        generate_evidence_queries("billing", source_types=())


def test_automated_queries_use_broad_unquoted_terms() -> None:
    segment = select_customer_segments("healthcare", limit=1)[0]
    query = build_problem_query(segment)

    assert "patient referral follow-up" not in query
    assert "clinic operations" in query
    assert segment.search_terms not in query
    assert "site:" not in query
    assert "reddit" not in query.lower()
    assert len(query) < 400


def test_web_discovery_deduplicates_urls_and_preserves_queries() -> None:
    provider = StaticSearchProvider(
        [
            SearchResult(
                title="Manual invoicing complaint",
                url=(
                    "https://news.ycombinator.com/item?id=42&utm_source=test"
                ),
                snippet="This manual process takes hours every week.",
                score=0.8,
            ),
            SearchResult(
                title="Manual invoicing complaint",
                url="https://news.ycombinator.com/item?id=42",
                snippet="This manual process takes hours every week and is frustrating.",
                score=0.9,
            ),
        ]
    )

    candidates = WebEvidenceDiscoveryService(provider).discover(
        "manual invoicing",
        source_types=("forums", "issues"),
        max_results=10,
    )

    assert len(candidates) == 1
    assert candidates[0].score == 0.9
    assert len(candidates[0].source_queries) == 2
    assert len(provider.queries) == 2


def test_candidate_becomes_attributable_submission() -> None:
    candidate = candidate_from_search_result(
        SearchResult(
            title="A recurring clinic problem",
            url="https://community.example/clinic-problem",
            snippet="We still use Excel and the manual process takes hours.",
            score=0.92,
        ),
        query="clinic workflow customer complaint",
    )

    assert candidate is not None
    submission = candidate.to_submission()
    assert submission.platform == "web"
    assert submission.source_url == "https://community.example/clinic-problem"
    assert submission.community == "community.example"
    assert submission.metadata_json["ingestion_method"] == "web_search"


def test_candidate_preserves_source_native_attribution() -> None:
    candidate = candidate_from_search_result(
        SearchResult(
            title="Manual inventory reconciliation loses adjustments",
            url="https://github.com/example/inventory/issues/42",
            snippet="The manual reconciliation fails and creates inventory errors.",
            content=(
                "Our warehouse team manually reconciles inventory every day. "
                "The current process fails when two adjustments overlap, and staff "
                "must repair the spreadsheet by hand."
            ),
            score=0.88,
            metadata={
                "source_platform": "github",
                "source_kind": "issue",
                "source_author": "warehouse-user",
                "published_at": "2026-06-01T10:30:00Z",
                "engagement_count": 7,
            },
        ),
        query="inventory reconciliation problem workaround",
    )

    assert candidate is not None
    submission = candidate.to_submission()
    assert submission.platform == "github"
    assert submission.source_author == "warehouse-user"
    assert submission.published_at is not None
    assert submission.engagement_score == 7
    assert submission.metadata_json["source_kind"] == "issue"


def test_demo_search_returns_problem_evidence() -> None:
    results = MockSearchProvider().search(
        "clinic referral customer complaint discussion manual process",
        max_results=10,
        search_depth="basic",
    )

    assert results
    assert any("takes hours" in result.snippet.lower() for result in results)


def test_customer_segment_selection_rotates_across_markets() -> None:
    first_scan = select_customer_segments("all", limit=4, offset=0)
    second_scan = select_customer_segments("all", limit=4, offset=4)
    healthcare = select_customer_segments("healthcare", limit=8)

    assert len(first_scan) == 4
    assert len({segment.focus for segment in first_scan}) == 4
    assert {segment.key for segment in first_scan}.isdisjoint(
        {segment.key for segment in second_scan}
    )
    assert healthcare
    assert all(segment.focus == "healthcare" for segment in healthcare)
