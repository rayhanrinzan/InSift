"""Tests for live readiness and local settings persistence."""

from pathlib import Path

import pytest

from src.config import Settings, update_env_file


def test_live_readiness_treats_reddit_as_optional() -> None:
    settings = Settings(
        demo_mode=False,
        llm_provider="openai",
        llm_api_key="openai-key",
        embedding_provider="openai",
        search_provider="tavily",
        search_api_key="tavily-key",
    )

    assert settings.live_ready is True
    assert settings.reddit_ready is False
    without_paid_search = settings.copy(update={"search_api_key": None})
    assert without_paid_search.live_ready is True
    assert without_paid_search.public_search_ready is True
    assert without_paid_search.research_ready is False


def test_default_live_workflow_requires_no_paid_credentials() -> None:
    settings = Settings(_env_file=None, demo_mode=False)

    assert settings.llm_provider == "local"
    assert settings.embedding_provider == "deterministic"
    assert settings.search_provider == "community"
    assert settings.live_ready is True


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
            "REDDIT_USER_AGENT": "FlowSiftAI/1.0 by test-user",
        },
        env_path=env_path,
    )

    result = env_path.read_text(encoding="utf-8")
    assert "# local settings" in result
    assert "LLM_API_KEY=keep-me" in result
    assert "DEMO_MODE=false" in result
    assert "LLM_MODEL=gpt-test" in result
    assert 'REDDIT_USER_AGENT="FlowSiftAI/1.0 by test-user"' in result


def test_env_update_rejects_multiline_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported characters"):
        update_env_file({"LLM_MODEL": "bad\nvalue"}, env_path=tmp_path / ".env")
