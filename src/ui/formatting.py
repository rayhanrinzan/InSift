"""Formatting helpers for UI display."""

from datetime import datetime
from typing import Optional


def format_score(score: Optional[float]) -> str:
    """Format a score for compact display."""

    if score is None:
        return "Not scored"
    return f"{score:.0f}"


def format_datetime(value: Optional[datetime]) -> str:
    """Format a timestamp for display."""

    if value is None:
        return "Unknown"
    return value.strftime("%Y-%m-%d %H:%M")
