"""Tests for live OpenAI provider boundaries without paid API calls."""

from __future__ import annotations

import io
import json
from urllib.error import HTTPError

import pytest

from src.clustering.embeddings import (
    EmbeddingError,
    ResilientEmbeddingProvider,
)
from src.config import Settings
from src.extraction.problem_extractor import (
    DeterministicMockExtractionProvider,
    ExtractionError,
    OpenAIProblemExtractionProvider,
    ResilientProblemExtractionProvider,
    build_problem_extraction_provider,
)
from src.providers.openai import (
    OpenAIAuthenticationError,
    OpenAIClient,
)


class StubResponse:
    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "StubResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_structured_response_uses_strict_schema_and_disables_storage() -> None:
    requests = []

    def opener(request: object, *, timeout: float) -> StubResponse:
        del timeout
        requests.append(request)
        return StubResponse({"output_text": '{"accepted": true}'})

    client = OpenAIClient("secret", model="test-model", opener=opener)
    result = client.structured_response(
        schema_name="result",
        schema={
            "type": "object",
            "properties": {"accepted": {"type": "boolean"}},
            "required": ["accepted"],
            "additionalProperties": False,
        },
        instructions="Return the classification.",
        input_text="source",
    )

    assert result == {"accepted": True}
    payload = json.loads(requests[0].data.decode("utf-8"))
    assert payload["store"] is False
    assert payload["text"]["format"]["strict"] is True
    assert payload["model"] == "test-model"


def test_embedding_response_is_normalized_to_floats() -> None:
    client = OpenAIClient(
        "secret",
        model="test-model",
        opener=lambda request, timeout: StubResponse(
            {"data": [{"embedding": [0, 0.5, 1]}]}
        ),
    )

    assert client.embedding("problem", model="embedding-model") == [0.0, 0.5, 1.0]


def test_authentication_failure_is_not_retried_or_leaked() -> None:
    calls = 0

    def opener(request: object, *, timeout: float) -> StubResponse:
        nonlocal calls
        del request, timeout
        calls += 1
        raise HTTPError("https://api.openai.com", 401, "no", {}, io.BytesIO())

    client = OpenAIClient("top-secret", model="test-model", opener=opener)

    with pytest.raises(OpenAIAuthenticationError, match="rejected") as error:
        client.embedding("problem", model="embedding-model")

    assert calls == 1
    assert "top-secret" not in str(error.value)


def test_rate_limit_retries_before_succeeding() -> None:
    calls = 0
    delays: list[float] = []

    def opener(request: object, *, timeout: float) -> StubResponse:
        nonlocal calls
        del request, timeout
        calls += 1
        if calls == 1:
            raise HTTPError("https://api.openai.com", 429, "slow", {}, io.BytesIO())
        return StubResponse({"data": [{"embedding": [0.25]}]})

    client = OpenAIClient(
        "secret",
        model="test-model",
        opener=opener,
        sleeper=delays.append,
    )

    assert client.embedding("problem", model="embedding-model") == [0.25]
    assert calls == 2
    assert delays == [0.5]


def test_live_extraction_builder_requires_and_uses_openai() -> None:
    settings = Settings(
        demo_mode=False,
        llm_provider="openai",
        llm_api_key="secret",
        llm_model="test-model",
    )

    provider = build_problem_extraction_provider(settings)

    assert isinstance(provider, ResilientProblemExtractionProvider)
    assert isinstance(provider.primary, OpenAIProblemExtractionProvider)
    assert isinstance(provider.fallback, DeterministicMockExtractionProvider)


def test_openai_extraction_failure_falls_back_to_grounded_local_rules() -> None:
    class RateLimitedProvider:
        def extract_problem(self, text: str, prompt: str):
            del text, prompt
            raise ExtractionError("OpenAI rate limit reached.")

    provider = ResilientProblemExtractionProvider(
        RateLimitedProvider(),
        DeterministicMockExtractionProvider(),
    )

    result = provider.extract_problem(
        "As a clinic manager, our manual process takes hours every week.",
        "extract",
    )

    assert result.contains_real_problem is True
    assert "manual process" in (result.problem_statement or "").lower()


def test_embedding_failure_switches_to_local_vectors_for_the_batch() -> None:
    class RejectedEmbeddingProvider:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, text: str) -> list[float]:
            del text
            self.calls += 1
            raise EmbeddingError("OpenAI rejected the configured API key.")

    primary = RejectedEmbeddingProvider()
    provider = ResilientEmbeddingProvider(primary)

    first = provider.embed("manual clinic referral follow-up")
    second = provider.embed("clinic referral spreadsheet tracking")

    assert len(first) == len(second) == 96
    assert primary.calls == 1
