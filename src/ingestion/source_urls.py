"""Source URL validation shared by ingestion and user-facing reads."""

from __future__ import annotations

from typing import Optional
from urllib.parse import urlsplit, urlunsplit


_PLACEHOLDER_HOSTS = {"example.com", "example.net", "example.org"}
_PLACEHOLDER_SUFFIXES = (".example", ".invalid", ".localhost", ".test")


def is_placeholder_source_url(url: Optional[str]) -> bool:
    """Return whether a supplied source URL is local, fake, or unusable."""

    if not url:
        return False
    parsed = urlsplit(url.strip())
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme not in {"http", "https"} or not host:
        return True
    if host == "localhost" or host in _PLACEHOLDER_HOSTS:
        return True
    return host.endswith(_PLACEHOLDER_SUFFIXES)


def is_public_source_url(url: Optional[str]) -> bool:
    """Return whether a URL can be presented as a real public source."""

    return bool(url) and not is_placeholder_source_url(url)


def source_identity(
    source_url: Optional[str],
    source_external_id: Optional[str],
    fallback_id: str,
) -> Optional[str]:
    """Return a stable identity for non-placeholder evidence."""

    if is_placeholder_source_url(source_url):
        return None
    if not source_url:
        return source_external_id or fallback_id

    parsed = urlsplit(source_url.strip())
    host = (parsed.hostname or "").lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), host, path, parsed.query, ""))
