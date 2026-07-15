"""Deterministic product-field extraction from search results."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from src.research.schemas import SearchResult


KNOWN_FEATURES = (
    "automation",
    "dashboard",
    "forms",
    "integrations",
    "reminders",
    "reporting",
    "spreadsheets",
    "status tracking",
    "tasks",
    "workflow",
)


def extract_product_fields(result: SearchResult) -> dict[str, object]:
    """Extract conservative product fields without inventing absent details."""

    metadata = result.metadata
    host = urlsplit(result.url).netloc.removeprefix("www.").split(":", 1)[0]
    host_parts = host.split(".")
    company_slug = (
        host_parts[-2] if len(host_parts) >= 2 else (host_parts[0] if host else "")
    )
    inferred_company = company_slug.replace("-", " ").title() or None
    title_name = re.split(
        r"\s+(?:Reviews?|Pros and Cons|Software Pricing|Pricing|Alternatives)\b|\s*[|:]\s*|\s+-\s+",
        result.title,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    if len(title_name.split()) > 8 and inferred_company:
        title_name = inferred_company
    product_name = metadata.get("product_name") or title_name or inferred_company
    searchable = f"{result.title} {result.snippet} {result.content or ''}".lower()
    features = metadata.get("features") or [
        feature for feature in KNOWN_FEATURES if feature in searchable
    ]
    return {
        "company_name": metadata.get("company_name") or inferred_company,
        "product_name": product_name or None,
        "target_customer": metadata.get("target_customer"),
        "problem_solved": metadata.get("problem_solved"),
        "description": result.snippet or result.content,
        "features": list(features),
        "pricing_position": metadata.get("pricing_position"),
        "strengths": list(metadata.get("strengths") or []),
        "weaknesses": list(metadata.get("weaknesses") or []),
        "possible_gap": metadata.get("possible_gap"),
    }
