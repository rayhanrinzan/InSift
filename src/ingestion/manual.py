"""Manual text and CSV ingestion adapters."""

from __future__ import annotations

import csv
import hashlib
import io
from typing import Optional

from pydantic import ValidationError

from src.ingestion.schemas import SourceSubmission


TEXT_COLUMNS = ("raw_text", "text", "body", "discussion", "content")


class IngestionError(ValueError):
    """Raised when submitted source data cannot be normalized."""


def build_source_external_id(raw_text: str, source_url: Optional[str] = None) -> str:
    """Return a stable identifier used to prevent duplicate ingestion."""

    normalized = " ".join(raw_text.lower().split())
    digest = hashlib.sha256(f"{source_url or ''}|{normalized}".encode("utf-8")).hexdigest()
    return f"submitted-{digest[:24]}"


def manual_submission(
    raw_text: str,
    *,
    source_url: Optional[str] = None,
    title: Optional[str] = None,
    source_author: Optional[str] = None,
    community: Optional[str] = None,
) -> SourceSubmission:
    """Normalize a manually pasted discussion."""

    try:
        return SourceSubmission(
            platform="manual",
            raw_text=raw_text,
            source_url=source_url,
            source_external_id=build_source_external_id(raw_text, source_url),
            title=title or None,
            source_author=source_author or None,
            community=community or None,
            metadata_json={"ingestion_method": "manual"},
        )
    except ValidationError as exc:
        raise IngestionError(exc.errors()[0]["msg"]) from exc


def parse_csv_submissions(
    payload: bytes | str,
    *,
    max_rows: Optional[int] = None,
) -> list[SourceSubmission]:
    """Parse source discussions from a UTF-8 CSV payload.

    Supported text columns are ``raw_text``, ``text``, ``body``, ``discussion``,
    and ``content``. Optional source fields use the database field names.
    """

    if isinstance(payload, bytes):
        try:
            csv_text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise IngestionError("CSV must be UTF-8 encoded.") from exc
    else:
        csv_text = payload

    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        if not reader.fieldnames:
            raise IngestionError("CSV is missing a header row.")
        normalized_headers = {name.strip().lower(): name for name in reader.fieldnames}
        text_column = next(
            (normalized_headers[name] for name in TEXT_COLUMNS if name in normalized_headers),
            None,
        )
        if text_column is None:
            expected = ", ".join(TEXT_COLUMNS)
            raise IngestionError(f"CSV needs one text column: {expected}.")

        submissions: list[SourceSubmission] = []
        for row_number, row in enumerate(reader, start=2):
            if max_rows is not None and len(submissions) >= max_rows:
                break
            cleaned: dict[str, str] = {
                key.strip().lower(): (value or "").strip()
                for key, value in row.items()
                if key is not None
            }
            raw_text = (row.get(text_column) or "").strip()
            if not raw_text:
                continue
            source_url = cleaned.get("source_url") or cleaned.get("url") or None
            external_id = cleaned.get("source_external_id") or build_source_external_id(
                raw_text, source_url
            )
            try:
                submissions.append(
                    SourceSubmission(
                        platform=cleaned.get("platform") or "csv",
                        raw_text=raw_text,
                        source_url=source_url,
                        source_external_id=external_id,
                        source_author=cleaned.get("source_author")
                        or cleaned.get("author")
                        or None,
                        published_at=cleaned.get("published_at") or None,
                        community=cleaned.get("community") or None,
                        title=cleaned.get("title") or None,
                        engagement_score=float(cleaned.get("engagement_score") or 0.0),
                        metadata_json={"ingestion_method": "csv", "csv_row": row_number},
                    )
                )
            except (ValidationError, ValueError) as exc:
                message = (
                    exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
                )
                raise IngestionError(f"CSV row {row_number}: {message}") from exc
    except csv.Error as exc:
        raise IngestionError(f"Malformed CSV: {exc}") from exc

    if not submissions:
        raise IngestionError("CSV contains no non-empty discussion rows.")
    return submissions
