"""Tests for live OpenAI provider boundaries without paid API calls."""

from __future__ import annotations

import io
import json
from urllib.error import HTTPError

import pytest

from src.config import Settings
from src.extraction.problem_extractor import (
    OpenAIProblemExtractionProvider,
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

    assert isinstance(provider, OpenAIProblemExtractionProvider)
