"""Evidence-backed five-component White-Space Score."""

from __future__ import annotations

from collections.abc import Sequence

from src.database.models import Competitor, EvidenceItem
from src.scoring.problem_score import clamp_score
from src.scoring.schemas import ScoreComponent, WhiteSpaceScoreBreakdown


def initial_whitespace_score() -> ScoreComponent:
    """Return a neutral score when competitor research has not run."""

    return ScoreComponent(
        score=50.0,
        reason=(
            "Neutral placeholder: competitor gaps and unmet need have not been researched. "
            "No-competitor evidence is not treated as positive."
        ),
        inputs={"research_complete": False},
    )


def calculate_whitespace_score(
    evidence_items: Sequence[EvidenceItem],
    competitors: Sequence[Competitor],
    *,
    target_customer: str | None,
    proposed_solution: str | None,
    successful_query_count: int,
) -> WhiteSpaceScoreBreakdown:
    """Calculate white-space from supported need, gaps, weaknesses, niche, and density."""

    relevant = [item for item in competitors if item.relationship_type != "irrelevant"]
    direct = [item for item in relevant if item.relationship_type == "direct"]
    supported_gaps = [item.possible_gap for item in relevant if item.possible_gap]
    weaknesses = [weakness for item in relevant for weakness in (item.weaknesses or [])]
    mean_severity = (
        sum(item.severity_score for item in evidence_items) / len(evidence_items) * 100
        if evidence_items
        else 0.0
    )
    gap_coverage = len(supported_gaps) / len(relevant) * 100 if relevant else 0.0
    if relevant:
        unmet_score = clamp_score((0.55 * mean_severity) + (0.45 * gap_coverage))
    else:
        unmet_score = clamp_score(min(45.0, mean_severity * 0.5))
    excerpt = ""
    if evidence_items:
        excerpt = str(
            (evidence_items[0].metadata_json or {}).get("evidence_quote")
            or evidence_items[0].problem_statement
            or ""
        )[:140]
    unmet = ScoreComponent(
        score=unmet_score,
        reason=(
            f"Evidence reports mean severity of {mean_severity:.0f}; "
            f"{len(supported_gaps)} of {len(relevant)} relevant competitor(s) have a stored gap."
            + (f' Representative evidence: "{excerpt}"' if excerpt else "")
        ),
        inputs={
            "mean_pain_severity": round(mean_severity, 2),
            "supported_competitor_gaps": len(supported_gaps),
            "relevant_competitors": len(relevant),
        },
    )

    if relevant:
        differentiation_value = 35 + (45 * len(supported_gaps) / len(relevant))
        if proposed_solution:
            differentiation_value += 5
    else:
        differentiation_value = 30
    differentiation = ScoreComponent(
        score=clamp_score(differentiation_value),
        reason=(
            f"Differentiation is supported by {len(supported_gaps)} stored gap(s), not by "
            "competitor absence alone."
        ),
        inputs={"supported_gaps": supported_gaps[:5], "has_proposed_solution": bool(proposed_solution)},
    )

    if relevant:
        weakness_value = 30 + min(55, (len(weaknesses) / len(relevant)) * 28)
    else:
        weakness_value = 25
    competitor_weakness = ScoreComponent(
        score=clamp_score(weakness_value),
        reason=(
            f"Stored classifications identify {len(weaknesses)} concrete weakness(es) across "
            f"{len(relevant)} relevant competitor(s)."
        ),
        inputs={"weaknesses": weaknesses[:8]},
    )

    customer_specific = bool(
        target_customer
        and target_customer.lower() not in {"users", "teams", "businesses", "unknown"}
    )
    pain_types = {pain for item in evidence_items for pain in (item.pain_types or [])}
    independent_sources = len(
        {item.source_url or item.source_external_id or item.id for item in evidence_items}
    )
    niche_value = (
        (55 if customer_specific else 25)
        + min(25, len(pain_types) * 5)
        + min(20, independent_sources * 5)
    )
    niche = ScoreComponent(
        score=clamp_score(niche_value),
        reason=(
            f"The opportunity has {'a specific' if customer_specific else 'a broad or missing'} "
            f"customer definition, {len(pain_types)} pain type(s), and "
            f"{independent_sources} independent source(s)."
        ),
        inputs={
            "target_customer": target_customer,
            "pain_types": sorted(pain_types),
            "independent_sources": independent_sources,
        },
    )

    if not relevant:
        density_value = 45 if successful_query_count else 50
    else:
        density_value = max(20, 80 - (15 * len(direct)))
    density = ScoreComponent(
        score=clamp_score(density_value),
        reason=(
            f"{len(direct)} direct competitor(s) were found across "
            f"{successful_query_count} successful query execution(s). This component is only "
            "10% of white-space and never makes no results automatically attractive."
        ),
        inputs={
            "direct_competitors": len(direct),
            "relevant_competitors": len(relevant),
            "successful_queries": successful_query_count,
        },
    )

    overall = clamp_score(
        (0.30 * unmet.score)
        + (0.25 * differentiation.score)
        + (0.20 * competitor_weakness.score)
        + (0.15 * niche.score)
        + (0.10 * density.score)
    )
    return WhiteSpaceScoreBreakdown(
        unmet_customer_need=unmet,
        differentiation_potential=differentiation,
        competitor_weakness=competitor_weakness,
        niche_specificity=niche,
        low_direct_competitor_density=density,
        whitespace_score=ScoreComponent(
            score=overall,
            reason=(
                "Weighted from unmet need (30%), differentiation (25%), competitor weakness "
                "(20%), niche specificity (15%), and direct-competitor density (10%)."
            ),
        ),
    )
