"""Tests for live readiness and local settings persistence."""

from pathlib import Path

import pytest

from src.config import Settings, update_env_file


def test_live_readiness_requires_all_external_credentials() -> None:
    partial = Settings(
        demo_mode=False,
        llm_provider="openai",
        llm_api_key="openai-key",
        embedding_provider="openai",
        search_provider="tavily",
        search_api_key="tavily-key",
    )
    ready = partial.copy(
        update={
            "reddit_client_id": "reddit-id",
            "reddit_client_secret": "reddit-secret",
        }
    )

    assert partial.live_ready is False
    assert ready.live_ready is True


def test_env_update_preserves_unmodified_secret(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# local settings\nLLM_API_KEY=keep-me\nDEMO_MODE=true\n",
        encoding="utf-8",
    )

    update_env_file(
        {
            "DEMO_MODE": False,
            "LLM_MODEL": "gpt-test",
            "REDDIT_USER_AGENT": "InSift/1.0 by test-user",
        },
        env_path=env_path,
    )

    result = env_path.read_text(encoding="utf-8")
    assert "# local settings" in result
    assert "LLM_API_KEY=keep-me" in result
    assert "DEMO_MODE=false" in result
    assert "LLM_MODEL=gpt-test" in result
    assert 'REDDIT_USER_AGENT="InSift/1.0 by test-user"' in result


def test_env_update_rejects_multiline_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported characters"):
        update_env_file({"LLM_MODEL": "bad\nvalue"}, env_path=tmp_path / ".env")
