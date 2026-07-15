"""Conservative direct, adjacent, substitute, and irrelevant classification."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

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
SUBSTITUTE_MARKERS = {
    "airtable",
    "consultant",
    "excel",
    "google sheets",
    "manual",
    "notion",
    "spreadsheet",
    "spreadsheets",
    "agency",
}


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
    if not value:
        return set()
    replacements = {
        "clinics": "clinic",
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
    values = set(
        re.findall(r"[a-z0-9]+", value.lower().replace("follow up", "follow-up"))
    )
    return {
        replacements.get(token, token) for token in values if token not in STOP_WORDS
    }


def _coverage(reference: set[str], candidate: set[str]) -> float:
    if not reference or not candidate:
        return 0.0
    return len(reference & candidate) / max(1, min(len(reference), len(candidate)))


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
                result.content,
                fields.get("target_customer"),
                fields.get("problem_solved"),
            )
        )
        candidate_tokens = _tokens(candidate_text)
        target_overlap = _coverage(_tokens(context.target_customer), candidate_tokens)
        problem_overlap = _coverage(
            _tokens(f"{context.title} {context.problem_summary}"), candidate_tokens
        )
        explicit_type = result.metadata.get("relationship_type")
        if explicit_type in {"direct", "adjacent", "substitute", "irrelevant"}:
            relationship_type = str(explicit_type)
            reasoning = "The deterministic demo result includes an explicit relationship fixture."
        elif any(marker in candidate_text.lower() for marker in SUBSTITUTE_MARKERS):
            relationship_type = "substitute"
            reasoning = (
                "The result is a manual or general-purpose alternative used instead."
            )
        elif target_overlap >= 0.28 and problem_overlap >= 0.28:
            relationship_type = "direct"
            reasoning = "The result overlaps both the target customer and core problem."
        elif target_overlap >= 0.15 or problem_overlap >= 0.15:
            relationship_type = "adjacent"
            reasoning = (
                "The result overlaps the customer or workflow, but not both strongly."
            )
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
            problem_solved=fields.get("problem_solved"),
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


def build_competitor_classifier(settings: Settings) -> CompetitorClassificationProvider:
    """Build deterministic demo or live OpenAI competitor classification."""

    provider = (settings.llm_provider or "").lower()
    if settings.demo_mode or provider == "mock":
        return CompetitorClassifier()
    if provider == "openai":
        if not settings.llm_api_key:
            raise SearchProviderError(
                "LLM_API_KEY is required for competitor classification."
            )
        return OpenAICompetitorClassifier(
            OpenAIClient(
                settings.llm_api_key.get_secret_value(),
                model=settings.llm_model,
                base_url=settings.openai_base_url,
            )
        )
    raise SearchProviderError(
        "Configure LLM_PROVIDER=openai with an API key, or enable demo mode."
    )
