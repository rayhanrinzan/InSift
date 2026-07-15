"""Tests for bounded Reddit OAuth source ingestion."""

from __future__ import annotations

import json

import pytest

from src.ingestion.reddit import RedditClient, RedditIngestionError


class StubResponse:
    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "StubResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _listing(children: list[dict[str, object]]) -> dict[str, object]:
    return {"data": {"children": children}}


def test_post_url_collects_attributed_post_and_comment() -> None:
    post = {
        "kind": "t3",
        "data": {
            "name": "t3_abc123",
            "title": "Manual billing takes forever",
            "selftext": "We lose hours every week copying invoices.",
            "author": "operator",
            "subreddit": "smallbusiness",
            "score": 12,
            "created_utc": 1_700_000_000,
            "permalink": "/r/smallbusiness/comments/abc123/example/",
        },
    }
    comment = {
        "kind": "t1",
        "data": {
            "name": "t1_reply",
            "body": "Our spreadsheet workaround is frustrating.",
            "author": "owner",
            "subreddit": "smallbusiness",
            "score": 4,
            "created_utc": 1_700_000_100,
            "permalink": "/r/smallbusiness/comments/abc123/example/reply/",
            "replies": "",
        },
    }

    def opener(request: object, *, timeout: float) -> StubResponse:
        del timeout
        if request.full_url.endswith("/api/v1/access_token"):
            return StubResponse({"access_token": "token"})
        return StubResponse([_listing([post]), _listing([comment])])

    client = RedditClient("client", "secret", "InSift test", opener=opener)
    submissions = client.submissions_from_url(
        "https://www.reddit.com/r/smallbusiness/comments/abc123/example/",
        max_results=2,
    )

    assert len(submissions) == 2
    assert submissions[0].source_external_id == "t3_abc123"
    assert submissions[0].community == "smallbusiness"
    assert submissions[1].source_external_id == "t1_reply"
    assert submissions[1].metadata_json["source_kind"] == "comment"
    assert submissions[1].source_url.startswith("https://www.reddit.com/")


def test_keyword_search_uses_a_bounded_listing() -> None:
    calls: list[str] = []
    child = {
        "kind": "t3",
        "data": {
            "name": "t3_result",
            "title": "A painful manual process",
            "selftext": "This takes hours every week.",
            "permalink": "/r/ops/comments/result/example/",
        },
    }

    def opener(request: object, *, timeout: float) -> StubResponse:
        del timeout
        calls.append(request.full_url)
        if request.full_url.endswith("/api/v1/access_token"):
            return StubResponse({"access_token": "token"})
        return StubResponse(_listing([child]))

    client = RedditClient("client", "secret", "InSift test", opener=opener)
    submissions = client.submissions_from_keywords(
        "manual workflow", subreddit="operations", max_results=500
    )

    assert len(submissions) == 1
    assert "/r/operations/search?" in calls[-1]
    assert "limit=100" in calls[-1]
    assert "restrict_sr=on" in calls[-1]


def test_non_reddit_url_is_rejected_before_network_access() -> None:
    client = RedditClient("client", "secret", "InSift test")

    with pytest.raises(RedditIngestionError, match="valid reddit.com"):
        client.submissions_from_url("https://example.com/comments/abc123")
