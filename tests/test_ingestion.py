"""Tests for manual and CSV source adapters."""

import pytest

from src.ingestion.manual import (
    IngestionError,
    manual_submission,
    parse_csv_submissions,
)


def test_manual_submission_has_stable_duplicate_id() -> None:
    first = manual_submission("This manual process takes forever.")
    second = manual_submission("  This   manual process takes forever.  ")

    assert first.source_external_id == second.source_external_id


def test_csv_ingestion_supports_text_and_source_fields() -> None:
    rows = parse_csv_submissions(
        "text,source_url,author,community\n"
        '"We still use Excel and it takes hours",https://example.com/1,sam,ops\n'
    )

    assert len(rows) == 1
    assert rows[0].source_author == "sam"
    assert rows[0].community == "ops"


def test_csv_without_text_column_is_rejected() -> None:
    with pytest.raises(IngestionError, match="needs one text column"):
        parse_csv_submissions("name,url\nExample,https://example.com\n")
