"""Actionable product briefs derived from stored opportunity evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.database.models import Competitor, EvidenceItem, OpportunityCluster


@dataclass(frozen=True)
class BriefFeature:
    """One bounded MVP capability and the job it performs."""

    name: str
    purpose: str


@dataclass(frozen=True)
class BuildPhase:
    """One week in the validation-first product plan."""

    week: str
    title: str
    actions: tuple[str, ...]
    exit_criteria: str


@dataclass(frozen=True)
class CompetitionAssessment:
    """Conservative interpretation of stored competitor research."""

    status: str
    label: str
    tone: str
    summary: str
    recommendation: str
    direct_count: int
    adjacent_count: int
    substitute_count: int
    gaps: tuple[str, ...]


@dataclass(frozen=True)
class OpportunityBrief:
    """Plain-language problem explanation and a validation-first build plan."""

    workflow: str
    core_user: str
    problem_statement: str
    plain_english: str
    business_impact: str
    current_workaround: str
    evidence_strength: str
    product_hypothesis: str
    core_workflow: tuple[str, ...]
    mvp_features: tuple[BriefFeature, ...]
    build_phases: tuple[BuildPhase, ...]
    technical_start: tuple[str, ...]
    excluded_scope: tuple[str, ...]
    success_metric: str
    competition: CompetitionAssessment


WORKFLOW_LABELS: tuple[tuple[str, str], ...] = (
    ("month end close", "month-end close"),
    ("month-end close", "month-end close"),
    ("order tracking", "order-tracking communication"),
    ("order fulfillment", "order fulfillment"),
    ("interview scheduling", "interview scheduling"),
    ("referral", "patient referral follow-up"),
    ("schedule", "schedule utilization"),
    ("insurance authorization", "insurance authorization tracking"),
    ("returns", "returns and refund exceptions"),
    ("inventory", "inventory reconciliation"),
    ("invoice", "invoice processing"),
    ("onboarding", "employee onboarding"),
    ("maintenance request", "maintenance request coordination"),
    ("renewal", "renewal follow-up"),
)

IMPACT_LABELS: dict[str, str] = {
    "time": "consumes staff time",
    "labor": "adds avoidable manual work",
    "cost": "raises operating cost",
    "lost_revenue": "can delay or lose revenue",
    "risk": "creates missed work and preventable mistakes",
    "compliance": "creates audit or compliance exposure",
    "coordination": "causes handoff and ownership failures",
    "data_entry": "requires duplicate data entry and reconciliation",
    "poor_user_experience": "creates a poor customer or staff experience",
    "lack_of_visibility": "leaves the team without reliable status visibility",
    "integration": "forces work across disconnected systems",
    "repetitive_work": "repeats the same low-value steps",
}


def build_opportunity_brief(cluster: OpportunityCluster) -> OpportunityBrief:
    """Build a useful brief without claiming evidence the product does not have."""

    evidence = _accepted_evidence(cluster)
    workflow = _workflow_label(cluster)
    core_user = cluster.target_customer or _affected_user(evidence)
    problem_statement = _problem_statement(cluster, evidence)
    pain_types = _pain_types(evidence)
    impacts = [IMPACT_LABELS[pain] for pain in pain_types if pain in IMPACT_LABELS]
    impact_text = (
        _join_phrases(impacts[:3])
        or "creates operational drag that still needs to be quantified"
    )
    workaround = _workaround(cluster, evidence)
    workaround_clause = (
        f"The documented workaround is {workaround}."
        if workaround != "Not documented yet"
        else "The current workaround is not documented yet and must be established in interviews."
    )
    plain_english = (
        f"This is an operational {workflow} problem for {core_user}. {problem_statement} "
        f"{workaround_clause} In practical terms, it {impact_text}."
    )
    evidence_strength = (
        f"Confirmed pattern across {cluster.independent_source_count} independent sources."
        if cluster.independent_source_count >= 2
        else "Early signal from one source. Treat the product direction as provisional until another user reports the same workflow problem."
    )
    product_hypothesis = (
        f"Test a lightweight {workflow} workspace for {core_user}. It should put every "
        "case in one queue, assign clear ownership, automate the routine steps, and "
        "surface exceptions before they become missed work."
    )
    features = _mvp_features(workflow, pain_types)
    competition = assess_competition(cluster)
    return OpportunityBrief(
        workflow=workflow,
        core_user=core_user,
        problem_statement=problem_statement,
        plain_english=plain_english,
        business_impact=(
            f"The evidence indicates that this problem {impact_text}. Quantify the "
            "baseline in interviews before estimating market size or ROI."
        ),
        current_workaround=workaround,
        evidence_strength=evidence_strength,
        product_hypothesis=product_hypothesis,
        core_workflow=(
            f"Capture or import each {workflow} item",
            "Assign an owner, status, and due date",
            "Run one rule for the repetitive step",
            "Escalate exceptions that need human judgment",
            "Measure completion time, errors, and unresolved items",
        ),
        mvp_features=features,
        build_phases=_build_phases(core_user, workflow),
        technical_start=(
            "Model five core records: organization, user, work item, status event, and automation rule.",
            "Build one responsive work-queue screen with filters, ownership, due dates, and an exception view.",
            "Use a relational database and background jobs for reminders, imports, and retries.",
            "Integrate with only the single system that pilot users already rely on; support CSV first when an API is unavailable.",
            "Record every status change and the baseline outcome metrics needed to prove time or error reduction.",
        ),
        excluded_scope=(
            "No broad all-in-one operations platform",
            "No native mobile app before the workflow is proven",
            "No generic AI assistant without a measured task to automate",
            "No billing, advanced permissions, or custom reporting in the first pilot",
        ),
        success_metric=(
            f"A pilot team completes real {workflow} work with at least 30% less handling "
            "time or 30% fewer missed/error cases, and at least one team agrees to pay or signs a letter of intent."
        ),
        competition=competition,
    )


def opportunity_workflow(cluster: OpportunityCluster) -> str:
    """Return the concise workflow name used in product and market research."""

    return _workflow_label(cluster)


def assess_competition(cluster: OpportunityCluster) -> CompetitionAssessment:
    """Translate competitor records into a conservative build decision."""

    competitors = [
        item for item in cluster.competitors if item.relationship_type != "irrelevant"
    ]
    direct = [item for item in competitors if item.relationship_type == "direct"]
    adjacent = [item for item in competitors if item.relationship_type == "adjacent"]
    substitutes = [
        item for item in competitors if item.relationship_type == "substitute"
    ]
    gaps = tuple(
        dict.fromkeys(
            item.possible_gap.strip()
            for item in competitors
            if item.possible_gap and item.possible_gap.strip()
        )
    )[:3]
    if cluster.status != "researched":
        return CompetitionAssessment(
            status="required",
            label="Market check required",
            tone="warn",
            summary=(
                "Existing products have not been checked yet. The build plan is a "
                "hypothesis, not permission to start coding."
            ),
            recommendation=(
                "Run the existing-solution check before building. It searches the open web, "
                "product directories, and substitutes for this exact user and workflow."
            ),
            direct_count=0,
            adjacent_count=0,
            substitute_count=0,
            gaps=(),
        )

    if direct:
        names = _competitor_names(direct[:3])
        strong_overlap = max(item.similarity_score for item in direct) >= 0.82
        return CompetitionAssessment(
            status="crowded" if len(direct) >= 3 or strong_overlap else "differentiate",
            label=(
                "Crowded: do not build the generic version"
                if len(direct) >= 3 or strong_overlap
                else "Direct competitor found"
            ),
            tone="risk" if len(direct) >= 3 or strong_overlap else "warn",
            summary=(
                f"The market check found {len(direct)} direct product(s) for the same "
                f"customer and workflow{f': {names}' if names else ''}."
            ),
            recommendation=(
                "Do not copy the generic solution. Interview users of these products and "
                "proceed only if a specific underserved segment or workflow gap is repeated."
            ),
            direct_count=len(direct),
            adjacent_count=len(adjacent),
            substitute_count=len(substitutes),
            gaps=gaps,
        )

    return CompetitionAssessment(
        status="potential_gap",
        label="No direct match found",
        tone="good",
        summary=(
            "The completed market check found no product classified as a direct match. "
            f"It did find {len(adjacent)} adjacent product(s) and "
            f"{len(substitutes)} substitute(s)."
        ),
        recommendation=(
            "This is a possible gap, not proof that no solution exists. Validate the gap "
            "with customer interviews and review the adjacent products before funding a full build."
        ),
        direct_count=0,
        adjacent_count=len(adjacent),
        substitute_count=len(substitutes),
        gaps=gaps,
    )


def _accepted_evidence(cluster: OpportunityCluster) -> list[EvidenceItem]:
    return [
        link.evidence_item
        for link in cluster.evidence_links
        if link.evidence_item.contains_problem
    ]


def _workflow_label(cluster: OpportunityCluster) -> str:
    text = f"{cluster.title} {cluster.problem_summary}".lower()
    for marker, label in WORKFLOW_LABELS:
        if marker in text:
            return label
    cleaned = re.sub(r"[^a-z0-9\s-]", " ", cluster.title.lower())
    cleaned = re.sub(
        r"\b(?:need|software|tool|problem|help|using|workflow)\b", " ", cleaned
    )
    words = " ".join(cleaned.split()).split()[:7]
    return " ".join(words) or "documented workflow"


def _problem_statement(
    cluster: OpportunityCluster,
    evidence: list[EvidenceItem],
) -> str:
    source = next(
        (
            item.problem_statement
            for item in evidence
            if item.problem_statement and item.problem_statement.strip()
        ),
        cluster.problem_summary,
    )
    text = re.sub(r"#{1,6}\s*", "", source or "")
    text = re.sub(r"Skip to main content", "", text, flags=re.IGNORECASE)
    text = " ".join(text.split())
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    pain_markers = (
        "hour",
        "manual",
        "spreadsheet",
        "error",
        "miss",
        "risk",
        "slow",
        "wait",
        "hole",
        "overwhelmed",
        "repetitive",
        "difficult",
        "frustrating",
    )
    useful = [
        sentence
        for sentence in sentences
        if any(marker in sentence.lower() for marker in pain_markers)
    ]
    selected = useful[:2] or sentences[:2]
    unique = list(dict.fromkeys(sentence.casefold() for sentence in selected))
    lookup = {sentence.casefold(): sentence for sentence in selected}
    summary = " ".join(lookup[key] for key in unique)
    if len(summary) > 520:
        summary = summary[:517].rsplit(" ", 1)[0] + "..."
    return (
        summary
        or "The source describes an operational problem that needs clarification."
    )


def _affected_user(evidence: list[EvidenceItem]) -> str:
    return next(
        (
            item.affected_user
            for item in evidence
            if item.affected_user and item.affected_user.strip()
        ),
        "the affected operations team",
    )


def _pain_types(evidence: list[EvidenceItem]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(pain for item in evidence for pain in (item.pain_types or []))
    )


def _workaround(
    cluster: OpportunityCluster,
    evidence: list[EvidenceItem],
) -> str:
    value = cluster.current_workaround or next(
        (
            item.current_workaround
            for item in evidence
            if item.current_workaround and item.current_workaround.strip()
        ),
        None,
    )
    if not value:
        return "Not documented yet"
    cleaned = " ".join(value.split()).rstrip(".")
    lowered = cleaned.lower()
    if lowered.startswith("uses ") and "according to the source text" in lowered:
        tool = re.sub(
            r"^uses\s+|\s+according to the source text$",
            "",
            lowered,
        )
        article = "an" if tool[:1] in "aeiou" else "a"
        return f"{article} {tool}"
    return cleaned


def _mvp_features(
    workflow: str,
    pain_types: tuple[str, ...],
) -> tuple[BriefFeature, ...]:
    features: list[BriefFeature] = [
        BriefFeature(
            "Shared work queue",
            f"Capture every {workflow} item with an owner, status, due date, and source record.",
        )
    ]

    def add(name: str, purpose: str) -> None:
        if all(item.name != name for item in features):
            features.append(BriefFeature(name, purpose))

    if "data_entry" in pain_types:
        add(
            "Import and reconciliation",
            "Import the existing spreadsheet or CSV, validate required fields, and show mismatches that need review.",
        )
    if "coordination" in pain_types:
        add(
            "Ownership and handoffs",
            "Assign work, record handoffs, and escalate overdue items without relying on inbox reminders.",
        )
    if set(pain_types) & {"time", "labor", "repetitive_work"}:
        add(
            "One high-volume automation",
            "Automate the single repeated update, reminder, or reconciliation step that consumes the most time.",
        )
    if set(pain_types) & {"risk", "compliance"}:
        add(
            "Exception alerts and audit trail",
            "Flag missing, late, or inconsistent work and preserve a history of who changed what.",
        )
    if "lack_of_visibility" in pain_types:
        add(
            "Status and bottleneck view",
            "Show unresolved work, aging items, and the stage where work is getting stuck.",
        )
    if "poor_user_experience" in pain_types:
        add(
            "Simple status communication",
            "Send a clear status update without requiring staff to answer the same question manually.",
        )
    if "integration" in pain_types:
        add(
            "Single system connection",
            "Read and write the minimum fields needed from the pilot team's existing system of record.",
        )
    add(
        "Exception view",
        "Keep routine work quiet and give the user one place to resolve cases that need judgment.",
    )
    add(
        "Outcome measurement",
        "Track handling time, missed items, and errors so the pilot can prove whether the product works.",
    )
    return tuple(features[:5])


def _build_phases(core_user: str, workflow: str) -> tuple[BuildPhase, ...]:
    return (
        BuildPhase(
            "Week 1",
            "Prove the workflow",
            (
                f"Interview five {core_user} who personally perform {workflow} work.",
                "Collect the real spreadsheet, inbox thread, form, or checklist they use today.",
                "Measure cases per week, handling time per case, error rate, and the cost of a miss.",
            ),
            "At least three of five users describe the same workflow and quantify meaningful recurring pain.",
        ),
        BuildPhase(
            "Week 2",
            "Test the solution without building it",
            (
                "Create a clickable work-queue prototype using the user's real fields and statuses.",
                "Run a concierge test: manually perform the proposed automation for two users.",
                "Show the competitor list and ask why the user does not already use those products.",
            ),
            "Two users provide real data for a pilot and prefer the focused workflow over their current option.",
        ),
        BuildPhase(
            "Week 3",
            "Build the narrow MVP",
            (
                "Implement import, work queue, ownership, status history, and one automation.",
                "Add the exception view and only the first required integration.",
                "Instrument handling time, completion, errors, and unresolved work from day one.",
            ),
            "The product completes at least ten real cases end to end without a hidden manual workaround.",
        ),
        BuildPhase(
            "Week 4",
            "Run a paid-pilot decision",
            (
                "Pilot with two teams for one real operating cycle.",
                "Compare time and errors against the Week 1 baseline.",
                "Ask for payment or a signed letter of intent; record objections and missing requirements.",
            ),
            "Continue only with measurable improvement and a payment signal; otherwise narrow, reposition, or stop.",
        ),
    )


def _competitor_names(competitors: list[Competitor]) -> str:
    return ", ".join(
        filter(
            None,
            (item.product_name or item.company_name for item in competitors),
        )
    )


def _join_phrases(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"
