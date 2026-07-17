"""Embedding provider interfaces and local implementations."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from src.config import Settings
from src.providers.openai import OpenAIClient, OpenAIProviderError


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot produce a vector."""


class EmbeddingProvider(Protocol):
    """Interface for semantic embedding providers."""

    def embed(self, text: str) -> list[float]:
        """Return one embedding for the supplied text."""


CONCEPT_GROUPS: tuple[tuple[str, ...], ...] = (
    ("spreadsheet", "spreadsheets", "excel", "sheet"),
    (
        "manual",
        "manually",
        "copy paste",
        "copy-paste",
        "repetitive",
        "tedious",
        "hours",
        "time-consuming",
        "time consuming",
    ),
    ("clinic", "clinics", "patient", "patients", "referral", "ehr", "healthcare"),
    ("follow up", "follow-up", "followup", "tracking", "queue"),
    ("vendor", "vendors", "contract", "renewal", "renewals"),
    ("invoice", "invoices", "billing", "payment", "payments"),
    ("inbox", "inboxes", "email", "emails", "reminder", "reminders"),
    ("intake", "onboarding", "form", "forms"),
    ("schedule", "scheduling", "calendar", "appointment"),
    (
        "hours",
        "slow",
        "forever",
        "time-consuming",
        "time consuming",
        "manual",
        "manually",
        "tedious",
        "repetitive",
    ),
    ("cost", "expensive", "overpay", "money", "revenue"),
    ("miss", "missed", "risk", "error", "errors"),
    ("coordinate", "coordination", "handoff", "owner", "status"),
    ("visibility", "stale", "reporting", "dashboard"),
    ("integrate", "integration", "sync", "systems", "api"),
    (
        "order tracking",
        "tracking number",
        "shipping update",
        "shipment status",
        "delivery status",
    ),
    (
        "inventory reconciliation",
        "inventory discrepancy",
        "stock count",
        "stock level",
        "warehouse inventory",
    ),
    ("return", "returns", "refund", "refunds", "chargeback"),
    ("purchase order", "supplier delivery", "supplier follow up", "procurement"),
    ("client document", "content approval", "asset approval", "scope change"),
    ("interview scheduling", "candidate screening", "interview feedback"),
    ("maintenance request", "tenant communication", "vendor scheduling"),
    ("change order", "subcontractor", "jobsite", "construction schedule"),
    ("policy renewal", "insurance quote", "claim status", "carrier quote"),
    ("technician dispatch", "service call", "service quote", "field service"),
    ("production schedule", "quality issue", "material planning", "factory"),
    ("audit preparation", "compliance evidence", "policy review", "regulatory report"),
)

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "this",
    "to",
    "use",
    "using",
    "we",
    "when",
    "with",
    "every",
}


class DeterministicEmbeddingProvider:
    """Small offline semantic hash embedding for demo mode.

    Known workflow concepts receive stable dimensions while remaining terms are
    feature-hashed. It is deterministic and requires no model download.
    """

    def __init__(self, dimensions: int = 96) -> None:
        if dimensions <= len(CONCEPT_GROUPS):
            raise ValueError("Embedding dimensions must exceed concept dimensions.")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        """Return a normalized deterministic embedding."""

        normalized = " ".join(text.lower().split())
        if not normalized:
            raise EmbeddingError("Cannot embed empty text.")
        vector = [0.0] * self.dimensions
        for index, terms in enumerate(CONCEPT_GROUPS):
            if any(term in normalized for term in terms):
                vector[index] = 3.0

        tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", normalized)
            if len(token) > 2 and token not in STOP_WORDS
        }
        hashed_dimensions = self.dimensions - len(CONCEPT_GROUPS)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = len(CONCEPT_GROUPS) + (value % hashed_dimensions)
            vector[index] += 0.35

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0.0:
            raise EmbeddingError("Text did not produce embedding features.")
        return [value / magnitude for value in vector]


class SentenceTransformerEmbeddingProvider:
    """Lazy local Sentence Transformers embedding provider."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def embed(self, text: str) -> list[float]:
        """Load the configured model on first use and return its embedding."""

        if not text.strip():
            raise EmbeddingError("Cannot embed empty text.")
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                raise EmbeddingError(
                    "The local semantic embedding model is unavailable."
                ) from exc
        try:
            vector = self._model.encode(text, normalize_embeddings=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise EmbeddingError(
                "The local semantic embedding model could not encode the text."
            ) from exc
        return [float(value) for value in vector]


class OpenAIEmbeddingProvider:
    """Create semantic embeddings with the configured OpenAI model."""

    def __init__(self, client: OpenAIClient, model_name: str) -> None:
        self.client = client
        self.model_name = model_name

    def embed(self, text: str) -> list[float]:
        try:
            return self.client.embedding(text, model=self.model_name)
        except OpenAIProviderError as exc:
            raise EmbeddingError(str(exc)) from exc


class ResilientEmbeddingProvider:
    """Use a configured embedding provider with a dimension-safe local fallback."""

    def __init__(self, primary: EmbeddingProvider) -> None:
        self.primary = primary
        self.primary_available = True
        self.fallback: DeterministicEmbeddingProvider | None = None

    def embed(self, text: str) -> list[float]:
        if self.primary_available:
            try:
                vector = self.primary.embed(text)
                if self.fallback is None:
                    self.fallback = DeterministicEmbeddingProvider(len(vector))
                return vector
            except EmbeddingError:
                self.primary_available = False
        if self.fallback is None:
            self.fallback = DeterministicEmbeddingProvider()
        return self.fallback.embed(text)


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Build the configured provider, using deterministic embeddings in demo mode."""

    if settings.demo_mode or settings.embedding_provider.lower() in {
        "mock",
        "deterministic",
    }:
        return DeterministicEmbeddingProvider()
    if settings.embedding_provider.lower() == "sentence_transformers":
        return ResilientEmbeddingProvider(
            SentenceTransformerEmbeddingProvider(settings.embedding_model)
        )
    if settings.embedding_provider.lower() == "openai":
        if not settings.llm_api_key:
            return DeterministicEmbeddingProvider()
        return ResilientEmbeddingProvider(
            OpenAIEmbeddingProvider(
                OpenAIClient(
                    settings.llm_api_key.get_secret_value(),
                    model=settings.llm_model,
                    base_url=settings.openai_base_url,
                ),
                settings.embedding_model,
            )
        )
    raise EmbeddingError(
        f"Unsupported embedding provider: {settings.embedding_provider}"
    )
