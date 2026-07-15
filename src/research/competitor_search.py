"""Search provider interfaces, Tavily integration, and demo results."""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Callable
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.config import Settings
from src.research.schemas import SearchResult


class SearchProviderError(RuntimeError):
    """Base error for safe search-provider failures."""


class SearchAuthenticationError(SearchProviderError):
    """Permanent invalid-credential failure that must not be retried."""


class SearchRateLimitError(SearchProviderError):
    """Search provider rate-limit failure after retries."""


class SearchTimeoutError(SearchProviderError):
    """Search provider timeout after retries."""


class SearchProvider(Protocol):
    """Interchangeable competitor search provider."""

    name: str

    def search(
        self, query: str, *, max_results: int, search_depth: str
    ) -> list[SearchResult]:
        """Return normalized search results."""


def canonical_url(url: str) -> str:
    """Remove fragments, tracking parameters, and trailing slashes for deduplication."""

    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    parts = urlsplit(url.strip())
    query = urlencode(
        [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_")
        ]
    )
    path = parts.path.rstrip("/") or ""
    host = parts.netloc.lower().removeprefix("www.")
    return urlunsplit((parts.scheme.lower(), host, path, query, ""))


class TavilySearchProvider:
    """Minimal Tavily Search API client with bounded exponential backoff."""

    name = "tavily"

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 15.0,
        max_attempts: int = 3,
        sleep_fn: Callable[[float], None] = time.sleep,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        if not api_key.strip():
            raise SearchAuthenticationError("A Tavily API key is required.")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.sleep_fn = sleep_fn
        self.opener = opener

    def search(
        self, query: str, *, max_results: int, search_depth: str
    ) -> list[SearchResult]:
        """Execute a Tavily `/search` request and normalize its results."""

        payload = json.dumps(
            {
                "query": query,
                "search_depth": search_depth,
                "max_results": max_results,
                "include_raw_content": True,
            }
        ).encode("utf-8")
        request = Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return [self._normalize(item) for item in body.get("results", [])]
            except HTTPError as exc:
                if exc.code in {401, 403}:
                    raise SearchAuthenticationError(
                        "The Tavily API key was rejected."
                    ) from exc
                if exc.code == 400:
                    raise SearchProviderError(
                        "Tavily rejected the search request."
                    ) from exc
                last_error = exc
                if exc.code != 429 and exc.code < 500:
                    raise SearchProviderError(
                        f"Tavily search failed with HTTP {exc.code}."
                    ) from exc
            except (TimeoutError, socket.timeout, URLError) as exc:
                last_error = exc
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise SearchProviderError(
                    "Tavily returned an invalid response."
                ) from exc
            if attempt < self.max_attempts - 1:
                self.sleep_fn(0.5 * (2**attempt))

        if isinstance(last_error, HTTPError) and last_error.code == 429:
            raise SearchRateLimitError("Tavily rate limit reached. Try again later.")
        raise SearchTimeoutError("Tavily search timed out after multiple attempts.")

    @staticmethod
    def _normalize(item: dict[str, Any]) -> SearchResult:
        return SearchResult(
            title=str(item.get("title") or "Untitled result"),
            url=str(item.get("url") or ""),
            snippet=str(item.get("content") or ""),
            content=item.get("raw_content"),
            score=max(0.0, min(1.0, float(item.get("score") or 0.0))),
        )


class MockSearchProvider:
    """Deterministic search provider for a complete offline workflow."""

    name = "mock"

    def search(
        self, query: str, *, max_results: int, search_depth: str
    ) -> list[SearchResult]:
        """Return stable domain-specific direct, adjacent, substitute, and noise results."""

        del search_depth
        normalized = query.lower()
        if "customer complaint discussion" in normalized:
            return self._evidence_results(normalized)[:max_results]
        if any(
            term in normalized for term in ("clinic", "patient", "referral", "intake")
        ):
            results = self._clinic_results()
        elif any(term in normalized for term in ("vendor", "renewal", "contract")):
            results = self._renewal_results()
        else:
            results = self._workflow_results()
        return results[:max_results]

    @staticmethod
    def _evidence_results(query: str) -> list[SearchResult]:
        """Return attributable demo discussions for the web discovery workflow."""

        if any(term in query for term in ("clinic", "patient", "referral")):
            return [
                SearchResult(
                    title="Referral follow-up is eating our week",
                    url="https://community.example/clinic-referral-follow-up",
                    snippet=(
                        "As a clinic manager, we still use Excel for referral follow-up "
                        "every day. The manual process takes hours and missed reminders "
                        "put patients at risk."
                    ),
                    score=0.94,
                ),
                SearchResult(
                    title="Small practice intake workaround",
                    url="https://issues.example/small-practice-intake",
                    snippet=(
                        "Our staff copy-paste patient intake details between systems every "
                        "week. It is repetitive, frustrating, and errors are easy to miss."
                    ),
                    score=0.86,
                ),
            ]
        return [
            SearchResult(
                title="Operations workflow still depends on spreadsheets",
                url="https://community.example/manual-operations-workflow",
                snippet=(
                    "We still use spreadsheets and copy-paste updates across systems every "
                    "week. This manual process takes hours and the team wishes there was a "
                    "simpler tool."
                ),
                score=0.91,
            ),
            SearchResult(
                title="Recurring handoffs are difficult to track",
                url="https://issues.example/recurring-handoffs",
                snippet=(
                    "Our operations team manually coordinates handoffs through inbox "
                    "reminders every day. The repetitive work is frustrating and missed "
                    "updates create risk."
                ),
                score=0.83,
            ),
        ]

    @staticmethod
    def _clinic_results() -> list[SearchResult]:
        return [
            SearchResult(
                title="CareQueue Referral Follow-up",
                url="https://carequeue.example/referrals",
                snippet="Referral and patient follow-up queues for small outpatient clinics.",
                score=0.94,
                metadata={
                    "company_name": "CareQueue",
                    "product_name": "CareQueue Referral Follow-up",
                    "relationship_type": "direct",
                    "target_customer": "small clinic operations teams",
                    "problem_solved": "patient intake and referral follow-up",
                    "features": [
                        "follow-up queue",
                        "owner reminders",
                        "status tracking",
                    ],
                    "pricing_position": "paid SaaS",
                    "strengths": [
                        "healthcare-specific workflow",
                        "clear accountability",
                    ],
                    "weaknesses": ["limited EHR integrations", "pricing is opaque"],
                    "possible_gap": "A lighter low-cost workflow for independent clinics.",
                },
            ),
            SearchResult(
                title="IntakeForms Pro",
                url="https://intakeforms.example",
                snippet="Digital patient intake forms and appointment paperwork for practices.",
                score=0.76,
                metadata={
                    "relationship_type": "adjacent",
                    "target_customer": "healthcare practices",
                    "problem_solved": "digital patient intake forms",
                    "features": ["forms", "signatures", "patient messaging"],
                    "strengths": ["fast digital intake"],
                    "weaknesses": ["follow-up ownership is not central"],
                    "possible_gap": "Post-intake referral accountability remains manual.",
                },
            ),
            SearchResult(
                title="Airtable",
                url="https://airtable.com/?utm_source=demo",
                snippet="Flexible spreadsheets, databases, and automations for general teams.",
                score=0.62,
                metadata={
                    "relationship_type": "substitute",
                    "target_customer": "general operations teams",
                    "problem_solved": "custom tracking workflows",
                    "features": ["tables", "views", "automations"],
                    "strengths": ["flexible"],
                    "weaknesses": ["not purpose-built for clinical follow-up"],
                    "possible_gap": "Clinics must design and maintain the workflow themselves.",
                },
            ),
            SearchResult(
                title="What does follow-up mean?",
                url="https://dictionary.example/follow-up",
                snippet="A dictionary definition and grammar guide.",
                score=0.21,
                metadata={"relationship_type": "irrelevant"},
            ),
        ]

    @staticmethod
    def _renewal_results() -> list[SearchResult]:
        return [
            SearchResult(
                title="RenewalPilot",
                url="https://renewalpilot.example",
                snippet="Vendor contract dates, owners, reminders, and renewal spend for small teams.",
                score=0.92,
                metadata={
                    "relationship_type": "direct",
                    "target_customer": "small business finance leads",
                    "problem_solved": "vendor renewal tracking",
                    "features": ["renewal calendar", "owner reminders", "spend notes"],
                    "strengths": ["purpose-built renewal workflow"],
                    "weaknesses": ["no inbox ingestion", "limited reporting"],
                    "possible_gap": "Automatic extraction from scattered inbox reminders.",
                },
            ),
            SearchResult(
                title="ProcureBoard",
                url="https://procureboard.example",
                snippet="Enterprise procurement and supplier management suite.",
                score=0.68,
                metadata={
                    "relationship_type": "adjacent",
                    "target_customer": "enterprise procurement",
                    "problem_solved": "supplier procurement management",
                    "features": ["supplier portal", "approvals"],
                    "strengths": ["broad procurement coverage"],
                    "weaknesses": ["heavy setup for small businesses"],
                    "possible_gap": "Simple renewal visibility without a procurement suite.",
                },
            ),
            SearchResult(
                title="Microsoft Excel",
                url="https://microsoft.com/excel",
                snippet="General-purpose spreadsheet used for contract and renewal lists.",
                score=0.57,
                metadata={
                    "relationship_type": "substitute",
                    "problem_solved": "manual renewal lists",
                    "features": ["spreadsheets", "formulas"],
                    "strengths": ["widely available"],
                    "weaknesses": ["stale data and scattered reminders"],
                    "possible_gap": "Automated ownership and due-date alerts.",
                },
            ),
            SearchResult(
                title="Renewable energy market news",
                url="https://news.example/renewable-market",
                snippet="News about renewable energy markets.",
                score=0.18,
                metadata={"relationship_type": "irrelevant"},
            ),
        ]

    @staticmethod
    def _workflow_results() -> list[SearchResult]:
        return [
            SearchResult(
                title="WorkflowPilot",
                url="https://workflowpilot.example",
                snippet="Configurable workflow automation for operations teams.",
                score=0.72,
                metadata={
                    "relationship_type": "adjacent",
                    "target_customer": "operations teams",
                    "problem_solved": "general workflow automation",
                    "features": ["tasks", "rules", "reminders"],
                    "strengths": ["flexible automation"],
                    "weaknesses": ["not tailored to the specific workflow"],
                    "possible_gap": "A focused workflow with less configuration.",
                },
            ),
            SearchResult(
                title="Google Sheets",
                url="https://workspace.google.com/sheets",
                snippet="General spreadsheet used for manual tracking.",
                score=0.5,
                metadata={"relationship_type": "substitute"},
            ),
        ]


def build_search_provider(settings: Settings) -> SearchProvider:
    """Build demo or Tavily search from centralized configuration."""

    if settings.demo_mode or (settings.search_provider or "").lower() == "mock":
        return MockSearchProvider()
    if (settings.search_provider or "").lower() == "tavily":
        if settings.search_api_key is None:
            raise SearchAuthenticationError("SEARCH_API_KEY is required for Tavily.")
        return TavilySearchProvider(settings.search_api_key.get_secret_value())
    raise SearchProviderError(
        "Configure SEARCH_PROVIDER=tavily with an API key, or enable DEMO_MODE."
    )
