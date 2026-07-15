"""Bounded Reddit OAuth ingestion with source attribution."""

from __future__ import annotations

import base64
import json
import re
import socket
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

from pydantic import ValidationError

from src.config import Settings
from src.ingestion.schemas import SourceSubmission


class RedditIngestionError(RuntimeError):
    """Base error for safe Reddit ingestion failures."""


class RedditAuthenticationError(RedditIngestionError):
    """Raised when Reddit rejects configured OAuth credentials."""


OpenFunction = Callable[..., Any]
SleepFunction = Callable[[float], None]
MAX_SOURCE_TEXT_CHARS = 12_000
REDDIT_HOSTS = {
    "reddit.com",
    "www.reddit.com",
    "old.reddit.com",
    "redd.it",
    "www.redd.it",
}


class RedditClient:
    """Read a small number of attributable Reddit posts and comments via OAuth."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        *,
        timeout_seconds: float = 20.0,
        max_attempts: int = 3,
        opener: OpenFunction = urlopen,
        sleeper: SleepFunction = time.sleep,
    ) -> None:
        if not client_id.strip() or not client_secret.strip():
            raise RedditAuthenticationError("Reddit OAuth credentials are required.")
        if not user_agent.strip():
            raise RedditIngestionError("A descriptive Reddit user agent is required.")
        self._client_id = client_id.strip()
        self._client_secret = client_secret.strip()
        self.user_agent = user_agent.strip()
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, max_attempts)
        self._opener = opener
        self._sleeper = sleeper
        self._access_token: str | None = None

    def submissions_from_url(
        self, reddit_url: str, *, max_results: int = 25
    ) -> list[SourceSubmission]:
        """Fetch one Reddit post and a bounded set of its visible comments."""

        post_id = self._post_id_from_url(reddit_url)
        payload = self._get_json(
            f"/comments/{quote(post_id)}",
            {"limit": self._bounded_limit(max_results), "depth": 2, "raw_json": 1},
        )
        if not isinstance(payload, list) or not payload:
            raise RedditIngestionError("Reddit returned no post data for that URL.")

        submissions: list[SourceSubmission] = []
        post_children = self._listing_children(payload[0])
        if post_children:
            post = self._post_submission(post_children[0])
            if post is not None:
                submissions.append(post)
        if len(payload) > 1:
            for child in self._walk_comment_children(
                self._listing_children(payload[1])
            ):
                if len(submissions) >= self._bounded_limit(max_results):
                    break
                comment = self._comment_submission(child)
                if comment is not None:
                    submissions.append(comment)
        if not submissions:
            raise RedditIngestionError("The Reddit post contains no ingestible text.")
        return submissions

    def submissions_from_subreddit(
        self, subreddit: str, *, max_results: int = 25
    ) -> list[SourceSubmission]:
        """Fetch a bounded newest-post listing for one subreddit."""

        cleaned = subreddit.strip().removeprefix("r/").strip("/")
        if not re.fullmatch(r"[A-Za-z0-9_]{2,21}", cleaned):
            raise RedditIngestionError(
                "Enter a valid subreddit name such as smallbusiness."
            )
        payload = self._get_json(
            f"/r/{quote(cleaned)}/new",
            {"limit": self._bounded_limit(max_results), "raw_json": 1},
        )
        return self._post_listing_submissions(payload, max_results=max_results)

    def submissions_from_keywords(
        self,
        keywords: str,
        *,
        subreddit: str | None = None,
        max_results: int = 25,
    ) -> list[SourceSubmission]:
        """Search public Reddit posts for a bounded keyword query."""

        query = " ".join(keywords.split())
        if len(query) < 2:
            raise RedditIngestionError("Enter at least two keyword characters.")
        path = "/search"
        cleaned_subreddit = None
        if subreddit and subreddit.strip():
            cleaned_subreddit = subreddit.strip().removeprefix("r/").strip("/")
            if not re.fullmatch(r"[A-Za-z0-9_]{2,21}", cleaned_subreddit):
                raise RedditIngestionError("Enter a valid subreddit filter.")
            path = f"/r/{quote(cleaned_subreddit)}/search"
        payload = self._get_json(
            path,
            {
                "q": query,
                "limit": self._bounded_limit(max_results),
                "sort": "relevance",
                "type": "link",
                "restrict_sr": "on" if cleaned_subreddit else "off",
                "raw_json": 1,
            },
        )
        return self._post_listing_submissions(payload, max_results=max_results)

    def _get_access_token(self) -> str:
        if self._access_token:
            return self._access_token
        credentials = base64.b64encode(
            f"{self._client_id}:{self._client_secret}".encode("utf-8")
        ).decode("ascii")
        request = Request(
            "https://www.reddit.com/api/v1/access_token",
            data=urlencode({"grant_type": "client_credentials"}).encode("ascii"),
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": self.user_agent,
            },
            method="POST",
        )
        payload = self._open_json(request, authentication_request=True)
        token = payload.get("access_token") if isinstance(payload, Mapping) else None
        if not isinstance(token, str) or not token:
            raise RedditAuthenticationError(
                "Reddit did not issue an OAuth access token."
            )
        self._access_token = token
        return token

    def _get_json(self, path: str, params: Mapping[str, Any]) -> Any:
        token = self._get_access_token()
        request = Request(
            f"https://oauth.reddit.com{path}?{urlencode(params)}",
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": self.user_agent,
            },
        )
        try:
            return self._open_json(request)
        except RedditAuthenticationError:
            self._access_token = None
            raise

    def _open_json(
        self, request: Request, *, authentication_request: bool = False
    ) -> Any:
        last_was_rate_limit = False
        for attempt in range(self.max_attempts):
            try:
                with self._opener(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                if exc.code in {401, 403}:
                    label = (
                        "OAuth credentials"
                        if authentication_request
                        else "access token"
                    )
                    raise RedditAuthenticationError(
                        f"Reddit rejected the configured {label}."
                    ) from exc
                if exc.code == 404:
                    raise RedditIngestionError(
                        "The requested Reddit source was not found."
                    ) from exc
                last_was_rate_limit = exc.code == 429
                if exc.code != 429 and not 500 <= exc.code < 600:
                    raise RedditIngestionError(
                        f"Reddit rejected the request with HTTP {exc.code}."
                    ) from exc
            except (URLError, socket.timeout, TimeoutError) as exc:
                last_was_rate_limit = False
                if attempt + 1 >= self.max_attempts:
                    raise RedditIngestionError(
                        "Reddit could not be reached after multiple attempts."
                    ) from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RedditIngestionError(
                    "Reddit returned an invalid response."
                ) from exc
            if attempt + 1 < self.max_attempts:
                self._sleeper(0.5 * (2**attempt))
        if last_was_rate_limit:
            raise RedditIngestionError("Reddit rate limit reached. Try again later.")
        raise RedditIngestionError("Reddit failed after multiple attempts.")

    def _post_listing_submissions(
        self, payload: Any, *, max_results: int
    ) -> list[SourceSubmission]:
        submissions = [
            submission
            for child in self._listing_children(payload)
            if (submission := self._post_submission(child)) is not None
        ][: self._bounded_limit(max_results)]
        if not submissions:
            raise RedditIngestionError(
                "Reddit returned no text posts for that request."
            )
        return submissions

    @staticmethod
    def _listing_children(payload: Any) -> list[Mapping[str, Any]]:
        if not isinstance(payload, Mapping):
            return []
        data = payload.get("data")
        children = data.get("children") if isinstance(data, Mapping) else None
        return [child for child in children or [] if isinstance(child, Mapping)]

    @classmethod
    def _walk_comment_children(
        cls, children: Iterable[Mapping[str, Any]]
    ) -> Iterable[Mapping[str, Any]]:
        for child in children:
            if child.get("kind") != "t1":
                continue
            yield child
            data = child.get("data")
            replies = data.get("replies") if isinstance(data, Mapping) else None
            if isinstance(replies, Mapping):
                yield from cls._walk_comment_children(cls._listing_children(replies))

    @classmethod
    def _post_submission(cls, child: Mapping[str, Any]) -> SourceSubmission | None:
        if child.get("kind") != "t3":
            return None
        data = child.get("data")
        if not isinstance(data, Mapping):
            return None
        title = cls._clean_text(data.get("title"))
        body = cls._clean_text(data.get("selftext"))
        raw_text = "\n\n".join(value for value in (title, body) if value)
        return cls._source_submission(data, raw_text, title=title, kind="post")

    @classmethod
    def _comment_submission(cls, child: Mapping[str, Any]) -> SourceSubmission | None:
        data = child.get("data")
        if not isinstance(data, Mapping):
            return None
        body = cls._clean_text(data.get("body"))
        return cls._source_submission(data, body, title=None, kind="comment")

    @classmethod
    def _source_submission(
        cls,
        data: Mapping[str, Any],
        raw_text: str,
        *,
        title: str | None,
        kind: str,
    ) -> SourceSubmission | None:
        if not raw_text:
            return None
        permalink = str(data.get("permalink") or "")
        source_url = f"https://www.reddit.com{permalink}" if permalink else None
        external_id = str(data.get("name") or data.get("id") or "") or None
        try:
            return SourceSubmission(
                platform="reddit",
                source_url=source_url,
                source_external_id=external_id,
                source_author=cls._optional_text(data.get("author")),
                published_at=cls._published_at(data.get("created_utc")),
                community=cls._optional_text(data.get("subreddit")),
                title=title,
                raw_text=raw_text[:MAX_SOURCE_TEXT_CHARS],
                engagement_score=max(0.0, float(data.get("score") or 0.0)),
                metadata_json={"ingestion_method": "reddit_oauth", "source_kind": kind},
            )
        except (ValidationError, TypeError, ValueError):
            return None

    @staticmethod
    def _clean_text(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        cleaned = " ".join(value.split()).strip()
        return "" if cleaned in {"[deleted]", "[removed]"} else cleaned

    @classmethod
    def _optional_text(cls, value: Any) -> str | None:
        return cls._clean_text(value) or None

    @staticmethod
    def _published_at(value: Any) -> datetime | None:
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            return None

    @staticmethod
    def _bounded_limit(value: int) -> int:
        return min(100, max(1, int(value)))

    @staticmethod
    def _post_id_from_url(value: str) -> str:
        parsed = urlsplit(value.strip())
        host = parsed.netloc.lower().split(":", 1)[0]
        if parsed.scheme not in {"http", "https"} or host not in REDDIT_HOSTS:
            raise RedditIngestionError("Enter a valid reddit.com post URL.")
        if host.endswith("redd.it"):
            candidate = parsed.path.strip("/").split("/", 1)[0]
        else:
            match = re.search(r"/comments/([a-z0-9]+)", parsed.path, re.IGNORECASE)
            candidate = match.group(1) if match else ""
        if not re.fullmatch(r"[a-z0-9]+", candidate, re.IGNORECASE):
            raise RedditIngestionError("The Reddit URL does not contain a post ID.")
        return candidate


def build_reddit_client(settings: Settings) -> RedditClient:
    """Build Reddit OAuth ingestion from centralized settings."""

    if not settings.reddit_client_id or not settings.reddit_client_secret:
        raise RedditAuthenticationError(
            "Configure Reddit OAuth credentials in Settings before collecting Reddit data."
        )
    return RedditClient(
        settings.reddit_client_id,
        settings.reddit_client_secret.get_secret_value(),
        settings.reddit_user_agent,
    )
