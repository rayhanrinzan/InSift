"""Resilient real-source search for public customer discussions."""

from __future__ import annotations

import gzip
import html
import json
import re
import socket
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from src.config import Settings
from src.research.competitor_search import (
    SearchProvider,
    SearchProviderError,
    TavilySearchProvider,
    canonical_url,
)
from src.research.schemas import SearchResult


PUBLIC_DISCUSSION_DOMAINS: tuple[str, ...] = (
    "reddit.com",
    "news.ycombinator.com",
    "indiehackers.com",
    "stackoverflow.com",
    "github.com",
    "g2.com",
    "capterra.com",
    "trustpilot.com",
)


def is_supported_discussion_url(url: str) -> bool:
    """Return whether a URL identifies an attributable discussion or review."""

    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path = parsed.path.lower().rstrip("/")
    if host == "reddit.com" or host.endswith(".reddit.com"):
        return "/comments/" in path
    if host == "news.ycombinator.com":
        return path == "/item" and bool(parsed.query)
    if host == "stackoverflow.com" or host.endswith(".stackoverflow.com"):
        return "/questions/" in path
    if host == "github.com" or host.endswith(".github.com"):
        return bool(re.search(r"/[^/]+/[^/]+/issues/\d+$", path))
    if host == "indiehackers.com" or host.endswith(".indiehackers.com"):
        return path.startswith(("/post/", "/product/"))
    if host == "trustpilot.com" or host.endswith(".trustpilot.com"):
        return path.startswith("/review/")
    if host in {"g2.com", "capterra.com"} or host.endswith(
        (".g2.com", ".capterra.com")
    ):
        return bool(path and path != "/")
    return False


class CommunityAPISearchProvider:
    """Search public Hacker News and Stack Exchange APIs without credentials."""

    name = "community_apis"

    def __init__(
        self,
        *,
        timeout_seconds: float = 12.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.opener = opener

    def search(
        self,
        query: str,
        *,
        max_results: int,
        search_depth: str,
    ) -> list[SearchResult]:
        """Return deduplicated discussions from available community APIs."""

        del search_depth
        cleaned_query = _plain_query(query)
        results: list[SearchResult] = []
        successful_sources = 0
        for search in (self._search_hacker_news, self._search_stack_overflow):
            try:
                results.extend(search(cleaned_query, max_results=max_results))
                successful_sources += 1
            except SearchProviderError:
                continue
        if not successful_sources:
            raise SearchProviderError(
                "Public community search is temporarily unavailable."
            )

        unique: dict[str, SearchResult] = {}
        for result in results:
            key = canonical_url(result.url)
            existing = unique.get(key)
            if existing is None or result.score > existing.score:
                unique[key] = result
        return sorted(
            unique.values(),
            key=lambda item: (item.score, len(item.content or item.snippet)),
            reverse=True,
        )[:max_results]

    def _search_hacker_news(
        self,
        query: str,
        *,
        max_results: int,
    ) -> list[SearchResult]:
        params = urlencode(
            {
                "query": query,
                "hitsPerPage": min(50, max(5, max_results * 3)),
            }
        )
        payload = self._get_json(f"https://hn.algolia.com/api/v1/search?{params}")
        hits = payload.get("hits", [])
        if not isinstance(hits, list):
            return []

        results: list[SearchResult] = []
        for index, item in enumerate(hits):
            if not isinstance(item, dict):
                continue
            item_id = item.get("story_id") or item.get("objectID")
            if not item_id:
                continue
            title = item.get("title") or item.get("story_title") or "Hacker News discussion"
            body = _plain_text(
                item.get("comment_text") or item.get("story_text") or title
            )
            if not body:
                continue
            points = max(0, int(item.get("points") or 0))
            score = min(0.92, 0.78 - (index * 0.015) + min(points, 100) / 1_000)
            results.append(
                SearchResult(
                    title=_plain_text(title),
                    url=f"https://news.ycombinator.com/item?id={item_id}",
                    snippet=body[:1_000],
                    content=body[:20_000],
                    score=max(0.35, score),
                )
            )
        return results[:max_results]

    def _search_stack_overflow(
        self,
        query: str,
        *,
        max_results: int,
    ) -> list[SearchResult]:
        params = urlencode(
            {
                "q": query,
                "site": "stackoverflow",
                "pagesize": min(50, max(5, max_results * 2)),
                "order": "desc",
                "sort": "relevance",
                "filter": "withbody",
            }
        )
        payload = self._get_json(
            f"https://api.stackexchange.com/2.3/search/advanced?{params}"
        )
        items = payload.get("items", [])
        if not isinstance(items, list):
            return []

        results: list[SearchResult] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict) or not item.get("link"):
                continue
            title = _plain_text(item.get("title") or "Stack Overflow discussion")
            body = _plain_text(item.get("body") or title)
            if not body:
                continue
            votes = max(0, int(item.get("score") or 0))
            score = min(0.88, 0.72 - (index * 0.015) + min(votes, 100) / 1_000)
            results.append(
                SearchResult(
                    title=title,
                    url=str(item["link"]),
                    snippet=body[:1_000],
                    content=body[:20_000],
                    score=max(0.35, score),
                )
            )
        return results[:max_results]

    def _get_json(self, url: str) -> dict[str, Any]:
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "User-Agent": "FlowSiftAI/1.0 public-evidence-search",
            },
        )
        try:
            with self.opener(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
                encoding = response.headers.get("Content-Encoding", "")
            if encoding.lower() == "gzip":
                raw = gzip.decompress(raw)
            payload = json.loads(raw.decode("utf-8"))
        except (HTTPError, URLError, socket.timeout, TimeoutError) as exc:
            raise SearchProviderError("A public community API could not be reached.") from exc
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SearchProviderError("A public community API returned invalid data.") from exc
        if not isinstance(payload, dict):
            raise SearchProviderError("A public community API returned invalid data.")
        return payload


class ResilientPublicSearchProvider:
    """Prefer Tavily and transparently fall back to credential-free public APIs."""

    name = "resilient_public_search"

    def __init__(self, primary: SearchProvider, fallback: SearchProvider) -> None:
        self.primary = primary
        self.fallback = fallback
        self.primary_available = True

    def search(
        self,
        query: str,
        *,
        max_results: int,
        search_depth: str,
    ) -> list[SearchResult]:
        if self.primary_available:
            try:
                results = self.primary.search(
                    query,
                    max_results=max_results,
                    search_depth=search_depth,
                )
                if results:
                    return results
            except SearchProviderError:
                self.primary_available = False
        return self.fallback.search(
            query,
            max_results=max_results,
            search_depth=search_depth,
        )


def build_public_discussion_search_provider(settings: Settings) -> SearchProvider:
    """Build real public search with Tavily enhancement when it is available."""

    fallback = CommunityAPISearchProvider()
    if (
        not settings.demo_mode
        and (settings.search_provider or "").lower() == "tavily"
        and settings.search_api_key
    ):
        primary = TavilySearchProvider(
            settings.search_api_key.get_secret_value(),
            include_domains=PUBLIC_DISCUSSION_DOMAINS,
        )
        return ResilientPublicSearchProvider(primary, fallback)
    return fallback


def _plain_query(query: str) -> str:
    """Convert search-engine syntax into a concise community API query."""

    cleaned = re.sub(r"site:[^\s)]+", " ", query, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bOR\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[()\"]", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    if "first hand customer complaint" in cleaned.lower():
        segment_anchor = cleaned.split()[0]
        return f"{segment_anchor} spreadsheet"
    return cleaned[:180]


def _plain_text(value: Any) -> str:
    """Normalize small HTML excerpts returned by community APIs."""

    without_tags = re.sub(r"<[^>]+>", " ", str(value or ""))
    return " ".join(html.unescape(without_tags).split())
