"""Public web evidence discovery and normalization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from urllib.parse import urlsplit

from src.ingestion.manual import IngestionError, build_source_external_id
from src.ingestion.schemas import SourceSubmission
from src.research.competitor_search import SearchProvider, canonical_url
from src.research.schemas import SearchResult


MAX_EVIDENCE_TEXT_CHARS = 20_000

WEB_SOURCE_LABELS: dict[str, str] = {
    "forums": "Forums & communities",
    "issues": "Issue trackers",
    "reviews": "Product reviews",
    "web": "Broad web",
}

SOURCE_FILTERS: dict[str, str] = {
    "forums": (
        "(site:news.ycombinator.com OR site:indiehackers.com OR "
        "site:stackoverflow.com OR site:discourse.org)"
    ),
    "issues": "site:github.com/issues",
    "reviews": "(site:g2.com OR site:capterra.com OR site:trustpilot.com)",
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

    @property
    def preview(self) -> str:
        """Return a concise result excerpt for the review UI."""

        text = self.snippet or self.raw_text
        return " ".join(text.split())[:600]

    def to_submission(self) -> SourceSubmission:
        """Normalize this candidate for the shared discovery pipeline."""

        return SourceSubmission(
            platform="web",
            raw_text=self.raw_text,
            source_url=self.url,
            source_external_id=build_source_external_id(self.raw_text, self.url),
            community=self.domain,
            title=self.title,
            engagement_score=self.score,
            metadata_json={
                "ingestion_method": "web_search",
                "search_queries": list(self.source_queries),
                "search_score": self.score,
                "search_snippet": self.snippet[:1_000],
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
    if len(cleaned_topic) > 300 or len(cleaned_customer) > 200:
        raise IngestionError("Search topic or target customer is too long.")
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

        for query in queries:
            for result in self.provider.search(
                query,
                max_results=per_query,
                search_depth=self.search_depth,
            ):
                candidate = candidate_from_search_result(result, query=query)
                if candidate is None:
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
    raw_text = (result.content or result.snippet).strip()
    if not raw_text:
        return None
    raw_text = raw_text[:MAX_EVIDENCE_TEXT_CHARS].rstrip()
    domain = urlsplit(url).netloc.lower().removeprefix("www.")
    if not domain:
        return None
    return WebEvidenceCandidate(
        title=result.title.strip() or "Untitled public discussion",
        url=url,
        domain=domain,
        raw_text=raw_text,
        snippet=result.snippet.strip(),
        score=float(result.score),
        source_queries=(query,),
    )
