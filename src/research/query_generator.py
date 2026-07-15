"""Generate broad, deduplicated competitor search queries."""

from __future__ import annotations

import re

from src.database.models import OpportunityCluster
from src.services.opportunity_brief_service import opportunity_workflow


def _compact(value: str, word_limit: int = 12) -> str:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", value)
    return " ".join(words[:word_limit])


def _search_category(workflow: str) -> str:
    cleaned = workflow.replace("-", " ")
    cleaned = re.sub(r"\b(?:communication|coordination)\b", " ", cleaned)
    return " ".join(cleaned.split()) or workflow


def generate_competitor_queries(cluster: OpportunityCluster) -> list[str]:
    """Generate exact-workflow, customer, substitute, and directory queries."""

    customer = _compact(cluster.target_customer or "affected teams", word_limit=7)
    workaround = _compact(cluster.current_workaround or "manual workflow", word_limit=8)
    workflow = _compact(
        opportunity_workflow(cluster).replace("-", " "),
        word_limit=9,
    )
    category = _search_category(workflow)
    candidates = [
        f'"{category}" software for "{customer}"',
        f'"{workflow}" automation "{customer}"',
        f'"{category}" workflow software',
        f'"{workflow}" alternative to "{workaround}"',
        f'site:producthunt.com/products "{category}"',
        f'site:g2.com/products "{category}"',
        f'site:capterra.com/p "{category}"',
    ]
    seen: set[str] = set()
    queries: list[str] = []
    for candidate in candidates:
        cleaned = " ".join(candidate.split())[:700]
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            queries.append(cleaned)
    return queries
