"""Evidence pre-filtering and structured problem extraction."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import ValidationError

from src.config import Settings
from src.extraction.prompts import PROBLEM_EXTRACTION_PROMPT
from src.extraction.schemas import ExtractedProblem, PainType
from src.providers.openai import OpenAIClient, OpenAIProviderError


class ExtractionError(RuntimeError):
    """Raised when an extraction provider returns unusable structured output."""


class ProblemExtractionProvider(Protocol):
    """Interface implemented by mock and external structured-output providers."""

    def extract_problem(self, text: str, prompt: str) -> Any:
        """Return an ExtractedProblem, mapping, or JSON string."""


NULLABLE_STRING_SCHEMA = {
    "anyOf": [{"type": "string"}, {"type": "null"}],
}

EXTRACTED_PROBLEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "contains_real_problem": {"type": "boolean"},
        "problem_statement": NULLABLE_STRING_SCHEMA,
        "affected_user": NULLABLE_STRING_SCHEMA,
        "current_workaround": NULLABLE_STRING_SCHEMA,
        "pain_types": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "time",
                    "labor",
                    "cost",
                    "lost_revenue",
                    "risk",
                    "compliance",
                    "coordination",
                    "data_entry",
                    "poor_user_experience",
                    "lack_of_visibility",
                    "integration",
                    "repetitive_work",
                ],
            },
        },
        "severity_score": {"type": "number"},
        "frequency_signal": {"type": "number"},
        "willingness_to_pay_score": {"type": "number"},
        "evidence_quote": NULLABLE_STRING_SCHEMA,
        "confidence": {"type": "number"},
    },
    "required": [
        "contains_real_problem",
        "problem_statement",
        "affected_user",
        "current_workaround",
        "pain_types",
        "severity_score",
        "frequency_signal",
        "willingness_to_pay_score",
        "evidence_quote",
        "confidence",
    ],
}


class OpenAIProblemExtractionProvider:
    """Use OpenAI structured output for evidence-grounded extraction."""

    def __init__(self, client: OpenAIClient) -> None:
        self.client = client

    def extract_problem(self, text: str, prompt: str) -> dict[str, Any]:
        try:
            return self.client.structured_response(
                schema_name="extracted_problem",
                schema=EXTRACTED_PROBLEM_SCHEMA,
                instructions=prompt,
                input_text=f"SOURCE DISCUSSION:\n{text}",
            )
        except OpenAIProviderError as exc:
            raise ExtractionError(str(exc)) from exc


class ResilientProblemExtractionProvider:
    """Use OpenAI when available and local evidence rules when it is not."""

    def __init__(
        self,
        primary: ProblemExtractionProvider,
        fallback: ProblemExtractionProvider,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.primary_available = True

    def extract_problem(self, text: str, prompt: str) -> Any:
        if not self.primary_available:
            return self.fallback.extract_problem(text, prompt)
        try:
            return self.primary.extract_problem(text, prompt)
        except ExtractionError:
            self.primary_available = False
            return self.fallback.extract_problem(text, prompt)


EVIDENCE_PHRASES = (
    "waste hours",
    "takes forever",
    "manual process",
    "still use excel",
    "still use spreadsheets",
    "i hate using",
    "too expensive",
    "there has to be a better way",
    "does anyone know a tool",
    "how do you manage",
    "wish there was",
    "we currently pay",
    "we lose money",
    "repetitive",
    "tedious",
    "painful",
    "frustrating",
    "can't",
    "doesn't work",
    "not work",
    "not working",
    "won't work",
    "keeps failing",
)

SUPPORTING_PATTERNS = (
    r"\b(hours?|days?|weeks?)\b.*\b(every|each|per)\b",
    r"\b(miss|missed|losing|lost|error|errors|risk)\b",
    r"\b(manually|spreadsheet|excel|copy[- ]?paste|workaround)\b",
    r"\b(slow|difficult|expensive|annoying|broken)\b",
    r"\bneed (?:a|an|better)|looking for (?:a|an)|any tool\b",
)


class EvidencePreFilter:
    """Cheap deterministic screen that reduces unnecessary provider calls."""

    def should_extract(self, text: str) -> bool:
        """Return true when text contains at least one meaningful pain signal."""

        normalized = " ".join(text.lower().split())
        if any(phrase in normalized for phrase in EVIDENCE_PHRASES):
            return True
        return any(re.search(pattern, normalized) for pattern in SUPPORTING_PATTERNS)


class ProblemExtractor:
    """Run pre-filtering and validate a provider's structured output."""

    def __init__(
        self,
        provider: ProblemExtractionProvider,
        pre_filter: EvidencePreFilter | None = None,
    ) -> None:
        self.provider = provider
        self.pre_filter = pre_filter or EvidencePreFilter()

    def extract(self, text: str) -> ExtractedProblem:
        """Extract a grounded problem or return a deterministic rejection."""

        cleaned = text.strip()
        if not cleaned:
            raise ExtractionError("Discussion text cannot be empty.")
        if not self.pre_filter.should_extract(cleaned):
            return ExtractedProblem(contains_real_problem=False, confidence=0.9)

        raw_result = self.provider.extract_problem(cleaned, PROBLEM_EXTRACTION_PROMPT)
        return self._parse_result(raw_result)

    @staticmethod
    def _parse_result(raw_result: Any) -> ExtractedProblem:
        if isinstance(raw_result, ExtractedProblem):
            return raw_result
        payload: Mapping[str, Any] | Any = raw_result
        if isinstance(raw_result, str):
            candidate = raw_result.strip()
            if candidate.startswith("```"):
                candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate)
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError as exc:
                raise ExtractionError(
                    "The extraction provider returned invalid JSON."
                ) from exc
        if not isinstance(payload, Mapping):
            raise ExtractionError(
                "The extraction provider returned an unsupported response."
            )
        try:
            return ExtractedProblem.parse_obj(payload)
        except ValidationError as exc:
            raise ExtractionError(
                "The extraction response did not match the schema."
            ) from exc


PAIN_PATTERNS: dict[PainType, tuple[str, ...]] = {
    "time": ("hours", "takes forever", "slow", "time-consuming"),
    "labor": ("manual", "manually", "staff", "labor"),
    "cost": ("expensive", "cost", "overpay", "paying"),
    "lost_revenue": ("lose money", "lost revenue", "missed revenue", "overpay"),
    "risk": (
        "risk",
        "missed",
        "easy to miss",
        "error",
        "fails",
        "failing",
        "broken",
        "can't",
        "doesn't work",
        "not work",
        "not working",
        "won't work",
        "cannot",
    ),
    "compliance": ("compliance", "audit", "regulation"),
    "coordination": ("coordinate", "handoff", "inboxes", "reminders", "follow-up"),
    "data_entry": (
        "copy-paste",
        "copy paste",
        "re-enter",
        "data entry",
        "spreadsheet",
        "excel",
    ),
    "poor_user_experience": ("hate using", "painful", "frustrating", "annoying"),
    "lack_of_visibility": ("visibility", "stale", "don't know", "cannot see"),
    "integration": ("integrate", "between systems", "sync", "api"),
    "repetitive_work": ("repetitive", "tedious", "every week", "every day"),
}


class DeterministicMockExtractionProvider:
    """Conservative local extractor grounded in explicit source text."""

    def extract_problem(self, text: str, prompt: str) -> ExtractedProblem:
        """Derive conservative structured fields from explicit text signals."""

        del prompt
        normalized = text.lower()
        pain_types: list[PainType] = [
            pain_type
            for pain_type, phrases in PAIN_PATTERNS.items()
            if any(phrase in normalized for phrase in phrases)
        ]
        evidence_quote = self._representative_sentence(text)
        if not pain_types or evidence_quote is None:
            return ExtractedProblem(contains_real_problem=False, confidence=0.7)

        affected_user = self._affected_user(text)
        workaround = self._workaround(text)
        severity = min(0.95, 0.38 + (0.08 * len(pain_types)))
        if any(marker in normalized for marker in ("lose money", "missed", "risk")):
            severity = min(0.95, severity + 0.12)
        frequency = 0.2
        if any(
            marker in normalized for marker in ("every day", "every week", "always")
        ):
            frequency = 0.75
        elif any(
            marker in normalized for marker in ("repetitive", "tedious", "currently")
        ):
            frequency = 0.5
        willingness = 0.1
        if any(
            marker in normalized for marker in ("currently pay", "we pay", "expensive")
        ):
            willingness = 0.65
        elif any(
            marker in normalized for marker in ("lose money", "lost revenue", "overpay")
        ):
            willingness = 0.5

        return ExtractedProblem(
            contains_real_problem=True,
            problem_statement=self._problem_statement(text),
            affected_user=affected_user,
            current_workaround=workaround,
            pain_types=pain_types,
            severity_score=round(severity, 2),
            frequency_signal=frequency,
            willingness_to_pay_score=willingness,
            evidence_quote=evidence_quote,
            confidence=min(0.92, 0.58 + (0.04 * len(pain_types))),
        )

    @staticmethod
    def _representative_sentence(text: str) -> str | None:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text)]
        scored: list[tuple[int, str]] = []
        for sentence in sentences:
            normalized = sentence.lower()
            score = sum(
                phrase in normalized
                for phrases in PAIN_PATTERNS.values()
                for phrase in phrases
            ) + sum(phrase in normalized for phrase in EVIDENCE_PHRASES)
            if score:
                scored.append((score, sentence))
        return max(scored, default=(0, ""), key=lambda item: item[0])[1] or None

    @staticmethod
    def _problem_statement(text: str) -> str:
        """Keep a concise source-grounded statement for reliable clustering."""

        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text)]
        relevant = []
        for sentence in sentences:
            normalized = sentence.lower()
            if any(
                marker in normalized
                for marker in (
                    "our app",
                    "recommend you use",
                    "book a demo",
                    "free trial",
                    "we built",
                    "shopify app store",
                    "related topics",
                )
            ):
                continue
            if any(
                phrase in normalized
                for phrases in PAIN_PATTERNS.values()
                for phrase in phrases
            ) or any(phrase in normalized for phrase in EVIDENCE_PHRASES):
                relevant.append(sentence)
            if len(relevant) >= 3:
                break
        statement = " ".join(relevant) or text.strip()
        statement = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", statement)
        return statement[:420].rstrip()

    @staticmethod
    def _affected_user(text: str) -> str | None:
        match = re.search(r"\bas (?:a|an) ([^,.!?]{2,80})", text, flags=re.IGNORECASE)
        return match.group(1).strip() if match else None

    @staticmethod
    def _workaround(text: str) -> str | None:
        normalized = text.lower()
        for marker in (
            "spreadsheet",
            "excel",
            "copy-paste",
            "copy paste",
            "manual process",
            "inbox reminders",
        ):
            if marker in normalized:
                return f"Uses {marker} according to the source text"
        return None


def build_problem_extraction_provider(settings: Settings) -> ProblemExtractionProvider:
    """Build extraction with a local path that never requires paid API quota."""

    provider = (settings.llm_provider or "").lower()
    local = DeterministicMockExtractionProvider()
    if settings.demo_mode or provider in {"", "local", "mock"}:
        return local
    if provider == "openai" and not settings.llm_api_key:
        return local
    if provider == "openai":
        primary = OpenAIProblemExtractionProvider(
            OpenAIClient(
                settings.llm_api_key.get_secret_value(),
                model=settings.llm_model,
                base_url=settings.openai_base_url,
            )
        )
        return ResilientProblemExtractionProvider(primary, local)
    if provider in {"deterministic", "rule_based"}:
        return DeterministicMockExtractionProvider()
    raise ExtractionError(
        "Configure LLM_PROVIDER as local or openai."
    )
