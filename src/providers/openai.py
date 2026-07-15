"""Minimal OpenAI Responses and Embeddings API client."""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Callable, Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OpenAIProviderError(RuntimeError):
    """Base error for safe OpenAI provider failures."""


class OpenAIAuthenticationError(OpenAIProviderError):
    """Raised when OpenAI rejects the configured credential."""


class OpenAIRateLimitError(OpenAIProviderError):
    """Raised after bounded retries for an OpenAI rate limit."""


class OpenAIResponseError(OpenAIProviderError):
    """Raised when OpenAI returns an unusable response."""


OpenFunction = Callable[..., Any]
SleepFunction = Callable[[float], None]


class OpenAIClient:
    """Call OpenAI over HTTPS without coupling the domain layer to an SDK."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 45.0,
        max_attempts: int = 3,
        opener: OpenFunction = urlopen,
        sleeper: SleepFunction = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise OpenAIAuthenticationError("An OpenAI API key is required.")
        if not model.strip():
            raise OpenAIProviderError("An OpenAI model is required.")
        self._api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, max_attempts)
        self._opener = opener
        self._sleeper = sleeper

    def structured_response(
        self,
        *,
        schema_name: str,
        schema: Mapping[str, Any],
        instructions: str,
        input_text: str,
    ) -> dict[str, Any]:
        """Return one strict JSON object from the Responses API."""

        payload = {
            "model": self.model,
            "instructions": instructions,
            "input": input_text,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": dict(schema),
                    "strict": True,
                }
            },
        }
        response = self._post_json("/responses", payload)
        text = self._extract_output_text(response)
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise OpenAIResponseError(
                "OpenAI returned structured output that could not be decoded."
            ) from exc
        if not isinstance(value, dict):
            raise OpenAIResponseError(
                "OpenAI returned the wrong structured output type."
            )
        return value

    def embedding(self, text: str, *, model: str) -> list[float]:
        """Return one embedding vector for text."""

        if not text.strip():
            raise OpenAIProviderError("Cannot embed empty text.")
        response = self._post_json(
            "/embeddings",
            {"model": model, "input": text, "encoding_format": "float"},
        )
        try:
            vector = response["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenAIResponseError(
                "OpenAI returned an invalid embedding response."
            ) from exc
        if not isinstance(vector, list) or not vector:
            raise OpenAIResponseError("OpenAI returned an empty embedding.")
        try:
            return [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise OpenAIResponseError(
                "OpenAI returned a non-numeric embedding."
            ) from exc

    def _post_json(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{endpoint}",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "InSift/1.0",
            },
            method="POST",
        )
        last_was_rate_limit = False
        for attempt in range(self.max_attempts):
            try:
                with self._opener(request, timeout=self.timeout_seconds) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise OpenAIResponseError(
                        "OpenAI returned an invalid response body."
                    )
                return decoded
            except HTTPError as exc:
                if exc.code in {401, 403}:
                    raise OpenAIAuthenticationError(
                        "OpenAI rejected the configured API key."
                    ) from exc
                last_was_rate_limit = exc.code == 429
                if exc.code != 429 and not 500 <= exc.code < 600:
                    raise OpenAIProviderError(
                        f"OpenAI rejected the request with HTTP {exc.code}."
                    ) from exc
            except (URLError, socket.timeout, TimeoutError) as exc:
                last_was_rate_limit = False
                if attempt + 1 >= self.max_attempts:
                    raise OpenAIProviderError(
                        "OpenAI could not be reached after multiple attempts."
                    ) from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise OpenAIResponseError(
                    "OpenAI returned an invalid response body."
                ) from exc

            if attempt + 1 < self.max_attempts:
                self._sleeper(0.5 * (2**attempt))

        if last_was_rate_limit:
            raise OpenAIRateLimitError("OpenAI rate limit reached. Try again later.")
        raise OpenAIProviderError("OpenAI failed after multiple attempts.")

    @staticmethod
    def _extract_output_text(response: Mapping[str, Any]) -> str:
        direct = response.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct
        output = response.get("output")
        if not isinstance(output, list):
            raise OpenAIResponseError("OpenAI returned no structured output.")
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                if part.get("type") == "refusal":
                    raise OpenAIResponseError(
                        "OpenAI declined to process the submitted source."
                    )
                text = part.get("text")
                if part.get("type") == "output_text" and isinstance(text, str):
                    return text
        raise OpenAIResponseError("OpenAI returned no structured output.")
