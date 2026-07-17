"""Centralized application configuration."""

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import Optional

from pydantic import BaseSettings, Field, SecretStr, validator


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or `.env`."""

    app_env: str = Field("development", env="APP_ENV")
    database_url: str = Field("sqlite:///flowsift.db", env="DATABASE_URL")
    llm_provider: Optional[str] = Field("local", env="LLM_PROVIDER")
    llm_api_key: Optional[SecretStr] = Field(None, env="LLM_API_KEY")
    llm_model: str = Field("gpt-5.6-luna", env="LLM_MODEL")
    openai_base_url: str = Field("https://api.openai.com/v1", env="OPENAI_BASE_URL")
    embedding_provider: str = Field("deterministic", env="EMBEDDING_PROVIDER")
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    search_provider: Optional[str] = Field("community", env="SEARCH_PROVIDER")
    search_api_key: Optional[SecretStr] = Field(None, env="SEARCH_API_KEY")
    github_api_token: Optional[SecretStr] = Field(None, env="GITHUB_TOKEN")
    search_depth: str = Field("basic", env="SEARCH_DEPTH")
    cluster_similarity_threshold: float = Field(
        0.78, env="CLUSTER_SIMILARITY_THRESHOLD", ge=0.0, le=1.0
    )
    minimum_extraction_confidence: float = Field(
        0.45, env="MINIMUM_EXTRACTION_CONFIDENCE", ge=0.0, le=1.0
    )
    max_search_results: int = Field(10, env="MAX_SEARCH_RESULTS", ge=1, le=100)
    reddit_client_id: Optional[str] = Field(None, env="REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[SecretStr] = Field(None, env="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = Field(
        "FlowSiftAI/1.0 by configured-user", env="REDDIT_USER_AGENT"
    )
    demo_mode: bool = Field(False, env="DEMO_MODE")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    @validator(
        "llm_provider",
        "search_provider",
        "reddit_client_id",
        pre=True,
        allow_reuse=True,
    )
    def empty_string_to_none(cls, value: Optional[str]) -> Optional[str]:
        """Treat blank optional provider names as missing values."""

        if isinstance(value, str) and not value.strip():
            return None
        return value

    @validator("search_depth", allow_reuse=True)
    def search_depth_must_be_supported(cls, value: str) -> str:
        """Validate search depth values supported by Tavily."""

        cleaned = value.strip().lower()
        if cleaned not in {"basic", "advanced", "fast", "ultra-fast"}:
            raise ValueError(
                "SEARCH_DEPTH must be basic, advanced, fast, or ultra-fast."
            )
        return cleaned

    @property
    def llm_ready(self) -> bool:
        provider = (self.llm_provider or "").lower()
        return provider in {"", "local", "mock", "deterministic", "rule_based"} or (
            provider == "openai"
        )

    @property
    def embedding_ready(self) -> bool:
        provider = self.embedding_provider.lower()
        if self.demo_mode or provider in {
            "mock",
            "deterministic",
            "sentence_transformers",
        }:
            return True
        return provider == "openai"

    @property
    def search_ready(self) -> bool:
        """Return whether Tavily-backed competitor research is configured."""

        provider = (self.search_provider or "").lower()
        return (
            self.demo_mode
            or provider == "mock"
            or bool(provider == "tavily" and self.search_api_key)
        )

    @property
    def public_search_ready(self) -> bool:
        """Credential-free community APIs keep public discovery available."""

        return not self.demo_mode

    @property
    def research_ready(self) -> bool:
        """Return whether optional Tavily competitor research is available."""

        return bool(
            not self.demo_mode
            and (self.search_provider or "").lower() == "tavily"
            and self.search_api_key
        )

    @property
    def reddit_ready(self) -> bool:
        return bool(self.reddit_client_id and self.reddit_client_secret)

    @property
    def discovery_ready(self) -> bool:
        return self.llm_ready and self.embedding_ready

    @property
    def live_ready(self) -> bool:
        return bool(
            not self.demo_mode
            and self.discovery_ready
            and self.public_search_ready
        )

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()


def redacted_database_url(database_url: str) -> str:
    """Return a database URL safe for logs and UI display."""

    if "@" not in database_url or "://" not in database_url:
        return database_url
    scheme, rest = database_url.split("://", 1)
    credentials, host = rest.rsplit("@", 1)
    if ":" not in credentials:
        return database_url
    username = credentials.split(":", 1)[0]
    return f"{scheme}://{username}:***@{host}"


EDITABLE_ENV_KEYS = {
    "APP_ENV",
    "DATABASE_URL",
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_MODEL",
    "OPENAI_BASE_URL",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "SEARCH_PROVIDER",
    "SEARCH_API_KEY",
    "GITHUB_TOKEN",
    "SEARCH_DEPTH",
    "CLUSTER_SIMILARITY_THRESHOLD",
    "MINIMUM_EXTRACTION_CONFIDENCE",
    "MAX_SEARCH_RESULTS",
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "REDDIT_USER_AGENT",
    "DEMO_MODE",
}


def update_env_file(updates: dict[str, Any], env_path: str | Path = ".env") -> None:
    """Atomically update approved local settings without returning secret values."""

    unknown = set(updates) - EDITABLE_ENV_KEYS
    if unknown:
        raise ValueError(f"Unsupported setting: {sorted(unknown)[0]}")
    normalized: dict[str, str] = {}
    for key, value in updates.items():
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        if "\n" in rendered or "\r" in rendered or "\x00" in rendered:
            raise ValueError(f"{key} contains unsupported characters.")
        normalized[key] = rendered

    path = Path(env_path)
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    output: list[str] = []
    replaced: set[str] = set()
    for line in existing:
        candidate = line.strip()
        if not candidate or candidate.startswith("#") or "=" not in line:
            output.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in normalized:
            output.append(f"{key}={_dotenv_value(normalized[key])}")
            replaced.add(key)
        else:
            output.append(line)
    for key, value in normalized.items():
        if key not in replaced:
            output.append(f"{key}={_dotenv_value(value)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as temporary:
            temporary.write("\n".join(output).rstrip() + "\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _dotenv_value(value: str) -> str:
    """Quote dotenv values only when special characters require it."""

    if value and all(
        character.isalnum() or character in "_./:+-" for character in value
    ):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
