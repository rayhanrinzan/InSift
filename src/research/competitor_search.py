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
        include_domains: tuple[str, ...] = (),
        timeout_seconds: float = 15.0,
        max_attempts: int = 3,
        sleep_fn: Callable[[float], None] = time.sleep,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        if not api_key.strip():
            raise SearchAuthenticationError("A Tavily API key is required.")
        self.api_key = api_key
        self.include_domains = include_domains
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.sleep_fn = sleep_fn
        self.opener = opener

    def search(
        self, query: str, *, max_results: int, search_depth: str
    ) -> list[SearchResult]:
        """Execute a Tavily `/search` request and normalize its results."""

        request_body: dict[str, Any] = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_raw_content": True,
        }
        if self.include_domains:
            request_body["include_domains"] = list(self.include_domains)
        payload = json.dumps(request_body).encode("utf-8")
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
                    detail = self._error_detail(exc)
                    raise SearchProviderError(
                        "Tavily rejected the search request"
                        f"{f': {detail}' if detail else '.'}"
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

    @staticmethod
    def _error_detail(error: HTTPError) -> str:
        """Return a bounded provider explanation without credential data."""

        try:
            payload = json.loads(error.read().decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return ""
        if not isinstance(payload, dict):
            return ""
        detail = payload.get("detail") or payload.get("message") or payload.get("error")
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("detail")
        return " ".join(str(detail or "").split())[:240]


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

        fixture_groups = (
            (
                ("clinic", "patient", "referral", "authorization"),
                (
                    (
                        "Referral follow-up is eating our week",
                        "https://community.example/clinic-referral-follow-up",
                        "As a clinic manager, we still use Excel for referral follow-up "
                        "every day. The manual process takes hours and missed reminders "
                        "put patients at risk.",
                        0.94,
                    ),
                    (
                        "Small practice intake workaround",
                        "https://issues.example/small-practice-intake",
                        "Our staff copy-paste patient intake details between systems every "
                        "week. It is repetitive, frustrating, and errors are easy to miss.",
                        0.86,
                    ),
                ),
            ),
            (
                ("client document", "accounting", "content approval"),
                (
                    (
                        "Month-end is mostly chasing client documents",
                        "https://accounting.example/client-document-chasing",
                        "Our accounting team spends hours every month emailing clients for "
                        "missing statements. We track reminders in a spreadsheet and still "
                        "discover missing files at the deadline.",
                        0.93,
                    ),
                    (
                        "A better way to track client requests",
                        "https://community.example/client-request-tracker",
                        "We copy every client request into a shared sheet and manually send "
                        "follow-ups. The repetitive work is frustrating during tax season.",
                        0.84,
                    ),
                ),
            ),
            (
                (
                    "maintenance request",
                    "property manager",
                    "change order",
                    "technician",
                ),
                (
                    (
                        "Maintenance updates disappear across calls and texts",
                        "https://property.example/maintenance-coordination",
                        "Our property managers manually copy maintenance requests from "
                        "emails and texts into a spreadsheet. Tenants call repeatedly "
                        "because status updates are easy to miss.",
                        0.92,
                    ),
                    (
                        "Vendor follow-up takes hours every week",
                        "https://community.example/property-vendor-follow-up",
                        "Coordinating vendors, tenants, and owners takes hours every week. "
                        "We wish there was one simple place for approvals and updates.",
                        0.85,
                    ),
                ),
            ),
            (
                ("return", "inventory", "purchase order", "warehouse"),
                (
                    (
                        "Return exceptions live in three different systems",
                        "https://commerce.example/return-exceptions",
                        "Our ecommerce team copy-pastes return details between the help "
                        "desk, store, and warehouse. Refund exceptions are manual and take "
                        "hours to reconcile each week.",
                        0.91,
                    ),
                    (
                        "Inventory discrepancy spreadsheet keeps growing",
                        "https://issues.example/inventory-reconciliation",
                        "We reconcile inventory discrepancies in a shared spreadsheet every "
                        "day. It is repetitive, frustrating, and stock errors delay orders.",
                        0.83,
                    ),
                ),
            ),
            (
                ("interview", "onboarding", "compliance training", "hiring"),
                (
                    (
                        "Interview feedback still requires constant reminders",
                        "https://recruiting.example/interview-feedback",
                        "Our recruiters manually chase interview feedback in chat after "
                        "every panel. Hiring decisions take days longer when one reviewer "
                        "forgets to submit notes.",
                        0.9,
                    ),
                    (
                        "Onboarding handoffs are difficult to track",
                        "https://community.example/onboarding-handoffs",
                        "We coordinate onboarding through email and a spreadsheet. The same "
                        "reminders are sent every week and missed tasks frustrate new hires.",
                        0.82,
                    ),
                ),
            ),
        )
        for keywords, fixtures in fixture_groups:
            if any(keyword in query for keyword in keywords):
                return [
                    SearchResult(title=title, url=url, snippet=snippet, score=score)
                    for title, url, snippet, score in fixtures
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
            )
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
