"""Pydantic schemas for ingestion payloads."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, validator


class SourceSubmission(BaseModel):
    """A normalized discussion submitted for ingestion."""

    platform: str = Field(default="manual")
    raw_text: str
    source_url: Optional[str] = None
    source_external_id: Optional[str] = None
    source_author: Optional[str] = None
    published_at: Optional[datetime] = None
    community: Optional[str] = None
    title: Optional[str] = None
    engagement_score: float = Field(default=0.0, ge=0.0)
    metadata_json: dict[str, Any] = Field(default_factory=dict)

    @validator("raw_text", allow_reuse=True)
    def text_must_not_be_blank(cls, value: str) -> str:
        """Reject empty discussions before extraction."""

        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Discussion text cannot be empty.")
        return cleaned

    @validator("source_url", allow_reuse=True)
    def source_url_must_be_http(cls, value: Optional[str]) -> Optional[str]:
        """Accept only attributable HTTP(S) source links."""

        if value is None or not value.strip():
            return None
        cleaned = value.strip()
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError("Source URL must start with http:// or https://.")
        return cleaned
