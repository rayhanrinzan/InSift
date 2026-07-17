"""Public web evidence discovery and normalization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from math import ceil
import re
from urllib.parse import urlsplit

from src.ingestion.manual import IngestionError, build_source_external_id
from src.ingestion.schemas import SourceSubmission
from src.ingestion.source_urls import is_public_source_url
from src.research.competitor_search import SearchProvider, canonical_url
from src.research.public_discussion_search import (
    SUPPORT_COMMUNITY_HOSTS,
    is_supported_discussion_url,
)
from src.research.schemas import SearchResult


MAX_EVIDENCE_TEXT_CHARS = 20_000

SOURCE_PAGE_BOILERPLATE = (
    "Skip to main content",
    "Open menu",
    "Open navigation",
    "Go to Reddit Home",
    "Get the Reddit app",
    "Get App",
    "Log in to Reddit",
    "Log In",
    "Expand user menu",
    "Open settings menu",
)

WEB_SOURCE_LABELS: dict[str, str] = {
    "forums": "Forums & communities",
    "issues": "Issue trackers",
    "reviews": "Product reviews",
    "web": "Broad web",
}

SOURCE_FILTERS: dict[str, str] = {
    "forums": "forum community discussion",
    "issues": "public issue report",
    "reviews": "software product review",
    "web": "",
}

PAIN_SIGNALS = (
    '"manual process" OR "takes hours" OR frustrating OR expensive OR '
    'workaround OR repetitive OR "wish there was"'
)


@dataclass(frozen=True)
class WebEvidenceCandidate:
    """One attributable public result that can be reviewed before ingestion."""

    title: str
    url: str
    domain: str
    raw_text: str
    snippet: str
    score: float
    source_queries: tuple[str, ...]
    source_platform: str = "web"
    source_kind: str = "discussion"
    source_author: str | None = None
    published_at: datetime | None = None
    engagement_count: int = 0

    @property
    def preview(self) -> str:
        """Return a concise result excerpt for the review UI."""

        text = self.snippet or self.raw_text
        return " ".join(text.split())[:600]

    def to_submission(self) -> SourceSubmission:
        """Normalize this candidate for the shared discovery pipeline."""

        return SourceSubmission(
            platform=self.source_platform,
            raw_text=self.raw_text,
            source_url=self.url,
            source_external_id=build_source_external_id(self.raw_text, self.url),
            source_author=self.source_author,
            published_at=self.published_at,
            community=self.domain,
            title=self.title,
            engagement_score=float(self.engagement_count or self.score),
            metadata_json={
                "ingestion_method": "web_search",
                "search_queries": list(self.source_queries),
                "search_score": self.score,
                "search_snippet": self.snippet[:1_000],
                "source_platform": self.source_platform,
                "source_kind": self.source_kind,
                "engagement_count": self.engagement_count,
            },
        )


def generate_evidence_queries(
    topic: str,
    *,
    target_customer: str | None = None,
    source_types: tuple[str, ...] = ("forums", "issues", "reviews"),
) -> list[str]:
    """Build bounded queries aimed at first-hand pain and workaround evidence."""

    cleaned_topic = " ".join(topic.strip().split()).replace('"', "")
    cleaned_customer = " ".join((target_customer or "").strip().split()).replace(
        '"', ""
    )
    if not cleaned_topic:
        raise IngestionError("Enter a market, workflow, or problem to search.")
    if len(cleaned_topic) > 100 or len(cleaned_customer) > 80:
        raise IngestionError(
            "Keep the search topic under 100 characters and target customer under 80."
        )
    if not source_types:
        raise IngestionError("Select at least one source type.")

    unknown = set(source_types) - set(SOURCE_FILTERS)
    if unknown:
        raise IngestionError(f"Unsupported web source type: {sorted(unknown)[0]}.")

    audience = f' "{cleaned_customer}"' if cleaned_customer else ""
    queries: list[str] = []
    for source_type in source_types:
        source_filter = SOURCE_FILTERS[source_type]
        query = (
            f'"{cleaned_topic}"{audience} customer complaint discussion '
            f"({PAIN_SIGNALS}) {source_filter}"
        )
        queries.append(" ".join(query.split()))
    return queries


class WebEvidenceDiscoveryService:
    """Search, deduplicate, and rank public evidence candidates."""

    def __init__(
        self,
        provider: SearchProvider,
        *,
        search_depth: str = "basic",
    ) -> None:
        self.provider = provider
        self.search_depth = search_depth

    def discover(
        self,
        topic: str,
        *,
        target_customer: str | None = None,
        source_types: tuple[str, ...] = ("forums", "issues", "reviews"),
        max_results: int = 15,
    ) -> list[WebEvidenceCandidate]:
        """Return a bounded set of unique, attributable search results."""

        if not 1 <= max_results <= 100:
            raise IngestionError("Maximum web results must be between 1 and 100.")
        queries = generate_evidence_queries(
            topic,
            target_customer=target_customer,
            source_types=source_types,
        )
        per_query = min(10, max(3, ceil(max_results / len(queries))))
        candidates: dict[str, WebEvidenceCandidate] = {}
        require_discussion_url = "web" not in source_types

        for query in queries:
            for result in self.provider.search(
                query,
                max_results=per_query,
                search_depth=self.search_depth,
            ):
                candidate = candidate_from_search_result(result, query=query)
                if candidate is None:
                    continue
                if not is_public_source_url(candidate.url):
                    continue
                if require_discussion_url and not is_supported_discussion_url(
                    candidate.url
                ):
                    continue
                key = canonical_url(candidate.url)
                existing = candidates.get(key)
                if existing is None:
                    candidates[key] = candidate
                    continue
                queries_seen = tuple(
                    dict.fromkeys((*existing.source_queries, *candidate.source_queries))
                )
                preferred = (
                    candidate
                    if (candidate.score, len(candidate.raw_text))
                    > (existing.score, len(existing.raw_text))
                    else existing
                )
                candidates[key] = replace(preferred, source_queries=queries_seen)

        return sorted(
            candidates.values(),
            key=lambda item: (item.score, len(item.raw_text)),
            reverse=True,
        )[:max_results]


def candidate_from_search_result(
    result: SearchResult,
    *,
    query: str,
) -> WebEvidenceCandidate | None:
    """Convert one search result into a reviewable evidence candidate."""

    url = result.url.strip()
    if not url.startswith(("http://", "https://")):
        return None
    raw_text = _clean_source_text(result.content or result.snippet)
    if not raw_text:
        return None
    raw_text = raw_text[:MAX_EVIDENCE_TEXT_CHARS].rstrip()
    snippet = _clean_source_text(result.snippet)
    domain = urlsplit(url).netloc.lower().removeprefix("www.")
    if not domain:
        return None
    metadata = result.metadata or {}
    return WebEvidenceCandidate(
        title=result.title.strip() or "Untitled public discussion",
        url=url,
        domain=domain,
        raw_text=raw_text,
        snippet=snippet,
        score=float(result.score),
        source_queries=(query,),
        source_platform=str(
            metadata.get("source_platform") or _source_platform_from_domain(domain)
        ),
        source_kind=str(metadata.get("source_kind") or "discussion"),
        source_author=_optional_text(metadata.get("source_author")),
        published_at=_source_datetime(metadata.get("published_at")),
        engagement_count=_non_negative_int(metadata.get("engagement_count")),
    )


def _clean_source_text(value: str | None) -> str:
    """Remove common search-indexed page chrome from public discussion text."""

    text = str(value or "")
    for trailing_section in (
        "Related topics | Topic |",
        "Powered by Discourse",
        "Similar topics | Topic |",
    ):
        text = text.split(trailing_section, 1)[0]
    for phrase in SOURCE_PAGE_BOILERPLATE:
        text = text.replace(phrase, " ")
    text = re.sub(r"(?:^|\s)#{1,6}\s*", " ", text)
    text = re.sub(r"\bTitle:\s*", "", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def _source_platform_from_domain(domain: str) -> str:
    if domain == "github.com" or domain.endswith(".github.com"):
        return "github"
    if domain == "news.ycombinator.com":
        return "hacker_news"
    if domain in {"stackoverflow.com", "serverfault.com", "superuser.com", "askubuntu.com"}:
        return "stack_exchange"
    if domain.endswith(".stackexchange.com"):
        return "stack_exchange"
    if domain in SUPPORT_COMMUNITY_HOSTS:
        return "support_community"
    if domain in {"g2.com", "capterra.com", "trustpilot.com"}:
        return "product_review"
    if domain == "reddit.com" or domain.endswith(".reddit.com"):
        return "reddit"
    return "web"


def _source_datetime(value: object) -> datetime | None:
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _optional_text(value: object) -> str | None:
    cleaned = str(value or "").strip()
    return cleaned or None


def _non_negative_int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
