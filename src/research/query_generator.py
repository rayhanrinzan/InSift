"""Generate broad, deduplicated competitor search queries."""

from __future__ import annotations

import re

from src.database.models import OpportunityCluster


def _compact(value: str, word_limit: int = 12) -> str:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", value)
    return " ".join(words[:word_limit])


def generate_competitor_queries(cluster: OpportunityCluster) -> list[str]:
    """Generate exact-workflow, customer, substitute, and directory queries."""

    problem = _compact(cluster.problem_summary)
    customer = _compact(cluster.target_customer or "affected teams", word_limit=7)
    workaround = _compact(cluster.current_workaround or "manual workflow", word_limit=8)
    workflow = _compact(cluster.title, word_limit=9)
    candidates = [
        f"software for {problem}",
        f"{customer} {workflow} automation",
        f"{problem} SaaS",
        f"alternative to {workaround}",
        f"{customer} software {problem}",
        f"{workflow} tools and platforms",
        f"site:producthunt.com {workflow}",
        f"site:ycombinator.com/companies {workflow}",
        f"site:g2.com {workflow}",
        f"site:capterra.com {workflow}",
        f"site:github.com {workflow}",
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
