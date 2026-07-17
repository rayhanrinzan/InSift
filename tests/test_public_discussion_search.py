"""Tests for real public search and paid-provider fallback behavior."""

from __future__ import annotations

import json
from typing import Any

from src.research.competitor_search import (
    SearchProviderError,
    TavilySearchProvider,
)
from src.research.public_discussion_search import (
    CommunityAPISearchProvider,
    ResilientPublicSearchProvider,
    _plain_query,
    is_supported_discussion_url,
)
from src.research.schemas import SearchResult


class StubResponse:
    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode("utf-8")
        self.headers: dict[str, str] = {}

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "StubResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_community_search_returns_attributable_real_discussions() -> None:
    def opener(request: Any, *, timeout: float) -> StubResponse:
        del timeout
        if "hn.algolia.com" in request.full_url:
            return StubResponse(
                {
                    "hits": [
                        {
                            "objectID": "12345",
                            "title": "Manual deployment tracking",
                            "story_text": (
                                "We copy-paste deployment status every day and the "
                                "manual process takes hours."
                            ),
                            "points": 12,
                        }
                    ]
                }
            )
        return StubResponse(
            {
                "items": [
                    {
                        "title": "How can we replace a repetitive release spreadsheet?",
                        "body": (
                            "Our team still uses Excel and missed handoffs are "
                            "frustrating every week."
                        ),
                        "link": "https://stackoverflow.com/questions/42/example",
                        "score": 8,
                    }
                ]
            }
        )

    results = CommunityAPISearchProvider(opener=opener).search(
        "release manager manual tracking",
        max_results=5,
        search_depth="basic",
    )

    assert {result.url for result in results} == {
        "https://news.ycombinator.com/item?id=12345",
        "https://stackoverflow.com/questions/42/example",
    }
    assert all(result.content for result in results)


def test_resilient_search_uses_public_fallback_after_tavily_failure() -> None:
    class FailingPrimary:
        name = "tavily"

        def search(self, query: str, *, max_results: int, search_depth: str):
            del query, max_results, search_depth
            raise SearchProviderError("Tavily rejected the search request.")

    class StaticFallback:
        name = "community_apis"

        def search(self, query: str, *, max_results: int, search_depth: str):
            del query, max_results, search_depth
            return [
                SearchResult(
                    title="Real discussion",
                    url="https://news.ycombinator.com/item?id=99",
                    snippet="This manual process takes hours every week.",
                    score=0.8,
                )
            ]

    provider = ResilientPublicSearchProvider(FailingPrimary(), StaticFallback())

    assert provider.search("workflow", max_results=3, search_depth="basic")[0].url == (
        "https://news.ycombinator.com/item?id=99"
    )


def test_community_fallback_reduces_scout_query_to_workflow_terms() -> None:
    query = (
        "ecommerce operations manager merchant order tracking customer emails "
        "manual takes hours frustrating reddit"
    )

    assert _plain_query(query) == (
        "ecommerce operations order tracking customer emails"
    )


def test_tavily_request_uses_domain_filter() -> None:
    requests: list[Any] = []

    def opener(request: Any, *, timeout: float) -> StubResponse:
        del timeout
        requests.append(request)
        return StubResponse({"results": []})

    provider = TavilySearchProvider(
        "tvly-test",
        include_domains=("reddit.com", "news.ycombinator.com"),
        opener=opener,
    )
    provider.search("short customer complaint", max_results=3, search_depth="basic")

    payload = json.loads(requests[0].data.decode("utf-8"))
    assert payload["include_domains"] == ["reddit.com", "news.ycombinator.com"]
    assert len(payload["query"]) < 400


def test_supported_discussion_urls_reject_generic_domain_pages() -> None:
    assert is_supported_discussion_url(
        "https://www.reddit.com/r/healthIT/comments/abc/workflow/"
    )
    assert is_supported_discussion_url(
        "https://github.com/example/project/issues/42"
    )
    assert is_supported_discussion_url(
        "https://community.airtable.com/t/automation-keeps-failing/12345"
    )
    assert not is_supported_discussion_url("https://www.reddit.com/r/healthIT/")
    assert not is_supported_discussion_url("https://community.airtable.com/latest")
    assert not is_supported_discussion_url("https://example.org/discussion")


def test_community_search_returns_detailed_github_issue_metadata() -> None:
    def opener(request: Any, *, timeout: float) -> StubResponse:
        del timeout
        if "hn.algolia.com" in request.full_url:
            return StubResponse({"hits": []})
        if "stackexchange.com" in request.full_url:
            return StubResponse({"items": []})
        return StubResponse(
            {
                "items": [
                    {
                        "title": "Inventory reconciliation fails after imports",
                        "html_url": "https://github.com/example/inventory/issues/42",
                        "body": (
                            "Our inventory team manually repairs spreadsheet errors "
                            "after every import. The reconciliation fails when two "
                            "warehouse adjustments overlap and there is no workaround."
                        ),
                        "comments": 4,
                        "created_at": "2026-06-01T10:30:00Z",
                        "user": {"login": "operator"},
                        "repository_url": "https://api.github.com/repos/example/inventory",
                        "labels": [{"name": "bug"}],
                        "reactions": {"total_count": 3},
                    }
                ]
            }
        )

    results = CommunityAPISearchProvider(opener=opener).search(
        "inventory reconciliation problem workaround",
        max_results=5,
        search_depth="basic",
    )

    assert len(results) == 1
    assert results[0].metadata["source_platform"] == "github"
    assert results[0].metadata["source_author"] == "operator"
    assert results[0].metadata["engagement_count"] == 7
