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
    "news.ycombinator.com",
    "github.com",
    "stackoverflow.com",
    "stackexchange.com",
    "serverfault.com",
    "superuser.com",
    "askubuntu.com",
    "indiehackers.com",
    "community.shopify.com",
    "community.hubspot.com",
    "quickbooks.intuit.com",
    "community.zapier.com",
    "community.airtable.com",
    "community.make.com",
    "community.atlassian.com",
    "community.open-emr.org",
    "g2.com",
    "capterra.com",
    "trustpilot.com",
)

STACK_EXCHANGE_HOSTS = {
    "stackoverflow.com",
    "serverfault.com",
    "superuser.com",
    "askubuntu.com",
}

SUPPORT_COMMUNITY_HOSTS = {
    "community.shopify.com",
    "community.hubspot.com",
    "quickbooks.intuit.com",
    "community.zapier.com",
    "community.airtable.com",
    "community.make.com",
    "community.atlassian.com",
    "community.open-emr.org",
}

STACK_EXCHANGE_SITE_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("employee", "hiring", "recruit", "workplace", "onboarding"), "workplace"),
    (("accounting", "bookkeeping", "invoice", "payment", "insurance"), "money"),
    (("construction", "contractor", "project", "subcontractor"), "pm"),
    (("ecommerce", "e-commerce", "inventory", "merchant", "order"), "softwarerecs"),
)

GENERATED_ISSUE_TITLE_MARKERS = (
    "problem:",
    "epic-",
    "[epic]",
    "roadmap:",
    "tracking:",
    "kanban",
    "implementation plan",
    "migration plan",
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
    if (
        host in STACK_EXCHANGE_HOSTS
        or host.endswith(".stackoverflow.com")
        or host.endswith(".stackexchange.com")
    ):
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
    if host in SUPPORT_COMMUNITY_HOSTS:
        return bool(
            re.search(
                r"(?:/t/[^/]+/\d+|/td-p/\d+|/qaq-p/\d+|/[^/]+-\d{4,}(?:/|$)|/\d{5,}(?:/|$))",
                path,
            )
        )
    return False


class CommunityAPISearchProvider:
    """Search source-native public APIs without requiring user credentials."""

    name = "community_apis"

    def __init__(
        self,
        *,
        timeout_seconds: float = 12.0,
        github_token: str | None = None,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.github_token = github_token
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
        for search in (
            self._search_hacker_news,
            self._search_stack_exchange,
            self._search_github_issues,
        ):
            try:
                results.extend(search(cleaned_query, max_results=max_results))
                successful_sources += 1
            except SearchProviderError:
                continue
        if not successful_sources:
            raise SearchProviderError(
                "Public community search is temporarily unavailable."
            )

        return _rank_diverse_results(results, max_results=max_results)

    def _search_hacker_news(
        self,
        query: str,
        *,
        max_results: int,
    ) -> list[SearchResult]:
        search_query = _compact_search_terms(query)
        params = urlencode(
            {
                "query": search_query,
                "tags": "(ask_hn,comment)",
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
            object_id = item.get("objectID")
            story_id = item.get("story_id") or object_id
            if not story_id or not object_id:
                continue
            title = item.get("title") or item.get("story_title") or "Hacker News discussion"
            body = _plain_text(
                item.get("comment_text") or item.get("story_text") or title
            )
            if not body:
                continue
            points = max(0, int(item.get("points") or 0))
            replies = max(0, int(item.get("num_comments") or 0))
            score = min(
                0.92,
                0.67
                - (index * 0.01)
                + min(points, 100) / 1_000
                + min(len(body), 2_000) / 20_000,
            )
            anchor = f"#{object_id}" if item.get("comment_text") else ""
            results.append(
                SearchResult(
                    title=_plain_text(title),
                    url=f"https://news.ycombinator.com/item?id={story_id}{anchor}",
                    snippet=body[:1_000],
                    content=body[:20_000],
                    score=max(0.35, score),
                    metadata={
                        "source_platform": "hacker_news",
                        "source_kind": "comment" if anchor else "question",
                        "source_author": item.get("author"),
                        "published_at": item.get("created_at"),
                        "engagement_count": points + replies,
                    },
                )
            )
        return results[:max_results]

    def _search_stack_exchange(
        self,
        query: str,
        *,
        max_results: int,
    ) -> list[SearchResult]:
        site = _stack_exchange_site(query)
        params = urlencode(
            {
                "q": query,
                "site": site,
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
            answers = max(0, int(item.get("answer_count") or 0))
            score = min(
                0.9,
                0.65
                - (index * 0.01)
                + min(votes, 100) / 1_000
                + min(answers, 10) / 100
                + min(len(body), 2_000) / 25_000,
            )
            owner = item.get("owner") if isinstance(item.get("owner"), dict) else {}
            results.append(
                SearchResult(
                    title=title,
                    url=str(item["link"]),
                    snippet=body[:1_000],
                    content=body[:20_000],
                    score=max(0.35, score),
                    metadata={
                        "source_platform": "stack_exchange",
                        "source_kind": "question",
                        "source_author": owner.get("display_name"),
                        "published_at": item.get("creation_date"),
                        "engagement_count": votes + answers,
                        "stack_exchange_site": site,
                    },
                )
            )
        return results[:max_results]

    def _search_github_issues(
        self,
        query: str,
        *,
        max_results: int,
    ) -> list[SearchResult]:
        search_query = _compact_search_terms(query)
        params = urlencode(
            {
                "q": f"{search_query} in:title,body is:issue -label:duplicate",
                "per_page": min(50, max(5, max_results * 2)),
                "sort": "comments",
                "order": "desc",
            }
        )
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        payload = self._get_json(
            f"https://api.github.com/search/issues?{params}",
            headers=headers,
        )
        items = payload.get("items", [])
        if not isinstance(items, list):
            return []

        results: list[SearchResult] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict) or item.get("pull_request"):
                continue
            url = item.get("html_url")
            title = _plain_text(item.get("title") or "GitHub issue")
            body = _plain_text(item.get("body"))
            if not url or not body or _looks_generated_issue(title, body):
                continue
            comments = max(0, int(item.get("comments") or 0))
            reactions = item.get("reactions")
            reaction_count = (
                max(0, int(reactions.get("total_count") or 0))
                if isinstance(reactions, dict)
                else 0
            )
            score = min(
                0.92,
                0.66
                - (index * 0.01)
                + min(comments, 20) / 100
                + min(reaction_count, 20) / 200
                + min(len(body), 3_000) / 30_000,
            )
            user = item.get("user") if isinstance(item.get("user"), dict) else {}
            labels = item.get("labels") if isinstance(item.get("labels"), list) else []
            repository_url = str(item.get("repository_url") or "")
            repository = "/".join(repository_url.rstrip("/").split("/")[-2:])
            results.append(
                SearchResult(
                    title=title,
                    url=str(url),
                    snippet=body[:1_000],
                    content=body[:20_000],
                    score=max(0.35, score),
                    metadata={
                        "source_platform": "github",
                        "source_kind": "issue",
                        "source_author": user.get("login"),
                        "published_at": item.get("created_at"),
                        "engagement_count": comments + reaction_count,
                        "reply_count": comments,
                        "repository": repository,
                        "labels": [
                            label.get("name")
                            for label in labels
                            if isinstance(label, dict) and label.get("name")
                        ],
                    },
                )
            )
        return results[:max_results]

    def _get_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        request_headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "User-Agent": "FlowSiftAI/1.0 public-evidence-search",
        }
        request_headers.update(headers or {})
        request = Request(
            url,
            headers=request_headers,
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
    """Merge Tavily coverage with source-native APIs and tolerate either failing."""

    name = "resilient_public_search"

    def __init__(self, primary: SearchProvider, fallback: SearchProvider) -> None:
        self.primary = primary
        self.fallback = fallback

    def search(
        self,
        query: str,
        *,
        max_results: int,
        search_depth: str,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        errors: list[SearchProviderError] = []
        for provider in (self.primary, self.fallback):
            try:
                results.extend(
                    provider.search(
                        query,
                        max_results=max_results,
                        search_depth=search_depth,
                    )
                )
            except SearchProviderError as exc:
                errors.append(exc)
        if not results and errors:
            raise errors[0]
        return _rank_diverse_results(results, max_results=max_results)


def build_public_discussion_search_provider(settings: Settings) -> SearchProvider:
    """Build real public search with Tavily enhancement when it is available."""

    fallback = CommunityAPISearchProvider(
        github_token=(
            settings.github_api_token.get_secret_value()
            if settings.github_api_token
            else None
        )
    )
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
    lowered = cleaned.lower()
    topic_marker = lowered.find("customer complaint discussion")
    if topic_marker >= 0:
        return cleaned[:topic_marker].strip()[:120]
    if "first hand customer complaint" in cleaned.lower():
        segment_anchor = cleaned.split()[0]
        return f"{segment_anchor} spreadsheet"
    scout_lens_starts = (
        "manual takes hours",
        "spreadsheet repetitive",
        "scaling backlog",
        "copy paste missed",
        "expensive software",
        "tracking coordination wish",
        "problem workaround",
        "feature request manual",
        "cannot automate repetitive",
        "errors missed handoffs",
        "too expensive workaround",
        "takes hours every week",
    )
    lens_positions = [
        lowered.find(lens) for lens in scout_lens_starts if lens in lowered
    ]
    if lens_positions:
        workflow_tokens = cleaned[: min(lens_positions)].split()
        selected = [*workflow_tokens[:2], *workflow_tokens[-4:]]
        return " ".join(dict.fromkeys(selected))
    return cleaned[:180]


def _compact_search_terms(query: str, *, max_terms: int = 4) -> str:
    """Keep source-native searches broad enough to return useful evidence."""

    tokens = _plain_query(query).split()
    if len(tokens) <= max_terms:
        return " ".join(tokens)
    selected = [tokens[0], *tokens[-(max_terms - 1) :]]
    return " ".join(dict.fromkeys(selected))


def _stack_exchange_site(query: str) -> str:
    lowered = query.lower()
    for markers, site in STACK_EXCHANGE_SITE_RULES:
        if any(marker in lowered for marker in markers):
            return site
    return "softwarerecs"


def _looks_generated_issue(title: str, body: str) -> bool:
    lowered_title = title.lower().strip()
    if any(marker in lowered_title for marker in GENERATED_ISSUE_TITLE_MARKERS):
        return True
    lowered_body = body.lower()
    return "<!-- openhax-kanban-sync" in lowered_body or (
        "## kanban metadata" in lowered_body and "migrated" in lowered_body
    )


def _rank_diverse_results(
    results: list[SearchResult],
    *,
    max_results: int,
) -> list[SearchResult]:
    """Deduplicate results while preventing one host from filling the batch."""

    unique: dict[str, SearchResult] = {}
    for result in results:
        key = canonical_url(result.url)
        existing = unique.get(key)
        if existing is None or (result.score, len(result.content or result.snippet)) > (
            existing.score,
            len(existing.content or existing.snippet),
        ):
            unique[key] = result

    ranked = sorted(
        unique.values(),
        key=lambda item: (item.score, len(item.content or item.snippet)),
        reverse=True,
    )
    selected: list[SearchResult] = []
    host_counts: dict[str, int] = {}
    for result in ranked:
        host = (urlsplit(result.url).hostname or "").lower().removeprefix("www.")
        if host_counts.get(host, 0) >= 3:
            continue
        selected.append(result)
        host_counts[host] = host_counts.get(host, 0) + 1
        if len(selected) >= max_results:
            break
    return selected


def _plain_text(value: Any) -> str:
    """Normalize small HTML excerpts returned by community APIs."""

    without_tags = re.sub(r"<[^>]+>", " ", str(value or ""))
    return " ".join(html.unescape(without_tags).split())
