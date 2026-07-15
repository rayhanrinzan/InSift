"""Conservative direct, adjacent, substitute, and irrelevant classification."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol
from urllib.parse import urlsplit

from pydantic import ValidationError

from src.config import Settings
from src.extraction.competitor_extractor import extract_product_fields
from src.providers.openai import OpenAIClient, OpenAIProviderError
from src.research.competitor_search import SearchProviderError
from src.research.schemas import (
    CompetitorClassification,
    CompetitorResearchContext,
    SearchResult,
)


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "software",
    "tool",
    "tools",
    "teams",
    "users",
}
PAIN_NARRATIVE_WORDS = {
    "daily",
    "day",
    "drowning",
    "hour",
    "hours",
    "manual",
    "minimum",
    "need",
    "repetitive",
    "spending",
    "task",
    "tasks",
}
SUBSTITUTE_MARKERS = {
    "airtable",
    "consultant",
    "excel",
    "google sheets",
    "manual",
    "notion",
    "spreadsheet",
    "spreadsheets",
    "whatsapp",
    "agency",
}
NON_PRODUCT_HOSTS = {
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "medium.com",
    "reddit.com",
    "researchgate.net",
    "youtube.com",
}
CONTENT_PATH_MARKERS = (
    "/article/",
    "/articles/",
    "/blog/",
    "/blogs/",
    "/categories/",
    "/category/",
    "/company-news/",
    "/guide/",
    "/guides/",
    "/figure/",
    "/figures/",
    "/glossary/",
    "/news/",
    "/post/",
    "/posts/",
    "/resources/",
    "/use-case/",
    "/use-cases/",
)
PRODUCT_LANGUAGE = (
    "app",
    "automation",
    "dashboard",
    "manage",
    "management",
    "platform",
    "software",
    "solution",
    "spreadsheet",
    "tracking",
    "workflow",
)


class CompetitorClassificationProvider(Protocol):
    """Classify one normalized search result against an opportunity."""

    def classify(
        self,
        context: CompetitorResearchContext,
        result: SearchResult,
    ) -> CompetitorClassification:
        """Return a grounded competitor classification."""


NULLABLE_STRING_SCHEMA = {"anyOf": [{"type": "string"}, {"type": "null"}]}
COMPETITOR_CLASSIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "company_name": NULLABLE_STRING_SCHEMA,
        "product_name": NULLABLE_STRING_SCHEMA,
        "relationship_type": {
            "type": "string",
            "enum": ["direct", "adjacent", "substitute", "irrelevant"],
        },
        "target_customer": NULLABLE_STRING_SCHEMA,
        "problem_solved": NULLABLE_STRING_SCHEMA,
        "features": {"type": "array", "items": {"type": "string"}},
        "pricing_position": NULLABLE_STRING_SCHEMA,
        "similarity_score": {"type": "number"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "possible_gap": NULLABLE_STRING_SCHEMA,
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": [
        "company_name",
        "product_name",
        "relationship_type",
        "target_customer",
        "problem_solved",
        "features",
        "pricing_position",
        "similarity_score",
        "strengths",
        "weaknesses",
        "possible_gap",
        "confidence",
        "reasoning",
    ],
}

COMPETITOR_CLASSIFICATION_PROMPT = """
Classify a web search result against one documented startup opportunity. A direct
competitor serves the same core customer and problem. An adjacent competitor overlaps
the customer or workflow but not both. A substitute is a manual process, spreadsheet,
general-purpose tool, employee, consultant, or agency used instead. Mark a result
irrelevant when the supplied title, snippet, and content do not support a meaningful
relationship. Use only supplied evidence. Do not invent features, pricing, weaknesses,
or gaps; use null or empty arrays when unsupported. Keep reasoning concise.
""".strip()


def _tokens(value: str | None) -> set[str]:
    return set(_normalized_terms(value))


def _normalized_terms(value: str | None) -> list[str]:
    if not value:
        return []
    replacements = {
        "clinics": "clinic",
        "emails": "email",
        "managers": "manager",
        "orders": "order",
        "patients": "patient",
        "referrals": "referral",
        "renewals": "renewal",
        "vendors": "vendor",
        "tracking": "track",
        "followup": "follow-up",
        "healthcare": "clinic",
        "practice": "clinic",
        "practices": "clinic",
    }
    ordered = re.findall(r"[a-z0-9]+", value.lower().replace("follow up", "follow-up"))
    return [
        replacements.get(token, token)
        for token in ordered
        if token not in STOP_WORDS
        and token not in PAIN_NARRATIVE_WORDS
        and not token.isdigit()
    ]


def _phrases(value: str | None) -> set[str]:
    terms = _normalized_terms(value)
    return {f"{left} {right}" for left, right in zip(terms, terms[1:])}


def _coverage(reference: set[str], candidate: set[str]) -> float:
    if not reference or not candidate:
        return 0.0
    return len(reference & candidate) / len(reference)


def is_product_candidate(result: SearchResult) -> bool:
    """Return whether a search result represents a product, not content about one."""

    parsed = urlsplit(result.url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    root_host = _root_host(host)
    path = parsed.path.lower().rstrip("/")
    if root_host == "producthunt.com":
        return path.startswith("/products/")
    if root_host == "g2.com":
        return path.startswith("/products/")
    if root_host == "capterra.com":
        return path.startswith("/p/") and "/compare/" not in path
    if root_host in NON_PRODUCT_HOSTS:
        return False
    if any(marker in f"{path}/" for marker in CONTENT_PATH_MARKERS):
        return False
    if re.search(r"/(?:19|20)\d{2}/(?:0?[1-9]|1[0-2])/", f"{path}/"):
        return False
    title = result.title.lower()
    if re.search(r"\b(?:best|top)\s+\d+\b|\b\d+\s+(?:best|top)\b", title):
        return False
    if any(
        marker in title
        for marker in (
            "alternatives to",
            "best alternatives",
            "best tools",
            "how to ",
            "tips to ",
            "what is ",
        )
    ):
        return False
    if path in {"", "/"}:
        return True
    searchable = f"{result.title} {result.snippet}".lower()
    return any(marker in searchable for marker in PRODUCT_LANGUAGE)


def product_identity(
    classification: CompetitorClassification,
    url: str,
) -> str:
    """Return a stable product-level key for cross-query deduplication."""

    root_host = _root_host((urlsplit(url).hostname or "").lower())
    name = classification.product_name or classification.company_name or root_host
    normalized = re.sub(
        r"\b(?:alternatives|and|cons|details|features|more|pricing|pros|reviews|software|20\d{2})\b",
        " ",
        name.casefold(),
    )
    normalized = " ".join(re.findall(r"[a-z0-9]+", normalized))
    return normalized or root_host


def _root_host(host: str) -> str:
    cleaned = host.removeprefix("www.")
    parts = cleaned.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else cleaned


class CompetitorClassifier:
    """Classify a result only when customer/problem evidence supports it."""

    def classify(
        self,
        context: CompetitorResearchContext,
        result: SearchResult,
    ) -> CompetitorClassification:
        """Return an evidence-backed relationship classification."""

        fields = extract_product_fields(result)
        candidate_text = " ".join(
            str(value or "")
            for value in (
                result.title,
                result.snippet,
                fields.get("target_customer"),
                fields.get("problem_solved"),
            )
        )
        candidate_tokens = _tokens(candidate_text)
        target_tokens = _tokens(context.target_customer)
        problem_text = f"{context.title} {context.problem_summary}"
        problem_tokens = _tokens(problem_text)
        target_matches = target_tokens & candidate_tokens
        problem_matches = problem_tokens & candidate_tokens
        target_overlap = _coverage(target_tokens, candidate_tokens)
        problem_overlap = _coverage(problem_tokens, candidate_tokens)
        phrase_overlap = bool(_phrases(problem_text) & _phrases(candidate_text))
        explicit_type = result.metadata.get("relationship_type")
        identity_text = " ".join(
            str(value or "")
            for value in (
                result.title,
                fields.get("company_name"),
                fields.get("product_name"),
                urlsplit(result.url).hostname,
            )
        ).lower()
        if explicit_type in {"direct", "adjacent", "substitute", "irrelevant"}:
            relationship_type = str(explicit_type)
            reasoning = "The deterministic demo result includes an explicit relationship fixture."
        elif any(marker in identity_text for marker in SUBSTITUTE_MARKERS):
            relationship_type = "substitute"
            reasoning = (
                "The result is a manual or general-purpose alternative used instead."
            )
        elif target_overlap >= 0.25 and problem_overlap >= 0.25 and phrase_overlap:
            relationship_type = "direct"
            reasoning = "The result overlaps the target customer and shares a specific workflow phrase."
        elif phrase_overlap or len(target_matches) >= 2 or len(problem_matches) >= 2:
            relationship_type = "adjacent"
            reasoning = "The result has meaningful customer or workflow overlap, but not enough evidence to call it direct."
        else:
            relationship_type = "irrelevant"
            reasoning = "The result lacks meaningful overlap with the customer and core problem."

        base_similarity = (target_overlap + problem_overlap) / 2
        similarity_by_type = {
            "direct": max(0.72, base_similarity),
            "adjacent": max(0.45, min(0.75, base_similarity)),
            "substitute": max(0.30, min(0.65, base_similarity)),
            "irrelevant": min(0.25, base_similarity),
        }
        confidence = (
            0.9
            if explicit_type
            else min(
                0.92, 0.55 + abs(target_overlap - 0.2) + abs(problem_overlap - 0.2)
            )
        )
        possible_gap = fields.get("possible_gap")
        if relationship_type in {"adjacent", "substitute"} and not possible_gap:
            possible_gap = (
                "The result is not purpose-built for the full documented workflow."
            )

        return CompetitorClassification(
            company_name=fields.get("company_name"),
            product_name=fields.get("product_name"),
            relationship_type=relationship_type,  # type: ignore[arg-type]
            target_customer=fields.get("target_customer"),
            problem_solved=(
                fields.get("problem_solved")
                or (
                    result.snippet.strip()[:320]
                    if relationship_type != "irrelevant"
                    else None
                )
            ),
            features=fields.get("features") or [],
            pricing_position=fields.get("pricing_position"),
            similarity_score=round(min(1.0, similarity_by_type[relationship_type]), 3),
            strengths=fields.get("strengths") or [],
            weaknesses=fields.get("weaknesses") or [],
            possible_gap=possible_gap,
            confidence=round(min(1.0, confidence), 3),
            reasoning=reasoning,
        )


class OpenAICompetitorClassifier:
    """Use OpenAI structured output to classify live search results."""

    def __init__(self, client: OpenAIClient) -> None:
        self.client = client

    def classify(
        self,
        context: CompetitorResearchContext,
        result: SearchResult,
    ) -> CompetitorClassification:
        input_payload = {
            "opportunity": context.dict(),
            "search_result": {
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "content": (result.content or "")[:8000],
            },
        }
        try:
            payload = self.client.structured_response(
                schema_name="competitor_classification",
                schema=COMPETITOR_CLASSIFICATION_SCHEMA,
                instructions=COMPETITOR_CLASSIFICATION_PROMPT,
                input_text=json.dumps(input_payload, ensure_ascii=True),
            )
            return CompetitorClassification.parse_obj(payload)
        except ValidationError as exc:
            raise SearchProviderError(
                "OpenAI returned an invalid competitor classification."
            ) from exc
        except OpenAIProviderError as exc:
            raise SearchProviderError(str(exc)) from exc


class ResilientCompetitorClassifier:
    """Use local classification when the configured LLM is unavailable."""

    def __init__(
        self,
        primary: CompetitorClassificationProvider,
        fallback: CompetitorClassificationProvider,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.primary_available = True

    def classify(
        self,
        context: CompetitorResearchContext,
        result: SearchResult,
    ) -> CompetitorClassification:
        if self.primary_available:
            try:
                return self.primary.classify(context, result)
            except SearchProviderError:
                self.primary_available = False
        return self.fallback.classify(context, result)


def build_competitor_classifier(settings: Settings) -> CompetitorClassificationProvider:
    """Build deterministic demo or live OpenAI competitor classification."""

    provider = (settings.llm_provider or "").lower()
    if settings.demo_mode or provider in {
        "",
        "local",
        "mock",
        "deterministic",
        "rule_based",
    }:
        return CompetitorClassifier()
    if provider == "openai":
        if not settings.llm_api_key:
            return CompetitorClassifier()
        return ResilientCompetitorClassifier(
            OpenAICompetitorClassifier(
                OpenAIClient(
                    settings.llm_api_key.get_secret_value(),
                    model=settings.llm_model,
                    base_url=settings.openai_base_url,
                )
            ),
            CompetitorClassifier(),
        )
    raise SearchProviderError(
        "Configure LLM_PROVIDER=openai with an API key, or enable demo mode."
    )
