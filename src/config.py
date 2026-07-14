"""Centralized application configuration."""

from functools import lru_cache
from typing import Optional

from pydantic import BaseSettings, Field, SecretStr, validator


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or `.env`."""

    app_env: str = Field("development", env="APP_ENV")
    database_url: str = Field("sqlite:///insift.db", env="DATABASE_URL")
    llm_provider: Optional[str] = Field(None, env="LLM_PROVIDER")
    llm_api_key: Optional[SecretStr] = Field(None, env="LLM_API_KEY")
    embedding_provider: str = Field("sentence_transformers", env="EMBEDDING_PROVIDER")
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    search_provider: Optional[str] = Field(None, env="SEARCH_PROVIDER")
    search_api_key: Optional[SecretStr] = Field(None, env="SEARCH_API_KEY")
    search_depth: str = Field("basic", env="SEARCH_DEPTH")
    cluster_similarity_threshold: float = Field(
        0.78, env="CLUSTER_SIMILARITY_THRESHOLD", ge=0.0, le=1.0
    )
    minimum_extraction_confidence: float = Field(
        0.45, env="MINIMUM_EXTRACTION_CONFIDENCE", ge=0.0, le=1.0
    )
    max_search_results: int = Field(10, env="MAX_SEARCH_RESULTS", ge=1, le=100)
    demo_mode: bool = Field(True, env="DEMO_MODE")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    @validator("llm_provider", "search_provider", pre=True, allow_reuse=True)
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
            raise ValueError("SEARCH_DEPTH must be basic, advanced, fast, or ultra-fast.")
        return cleaned

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
