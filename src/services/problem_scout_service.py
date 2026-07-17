"""Real-source problem scouting that persists results end to end."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

from src.database.models import OpportunityCluster, OpportunityScore
from src.database.repositories import ClusterRepository
from src.extraction.opportunity_synthesizer import (
    OpportunitySynthesisProvider,
)
from src.ingestion.manual import IngestionError
from src.ingestion.schemas import SourceSubmission
from src.ingestion.source_urls import is_public_source_url
from src.ingestion.web import (
    WebEvidenceCandidate,
    candidate_from_search_result,
)
from src.research.competitor_search import SearchProvider, canonical_url
from src.research.public_discussion_search import is_supported_discussion_url
from src.services.discovery_service import DiscoveryResult, DiscoveryService
from src.services.opportunity_brief_service import build_opportunity_brief


SCOUT_FOCUS_LABELS: dict[str, str] = {
    "all": "Any market",
    "healthcare": "Healthcare",
    "professional_services": "Professional services",
    "field_services": "Property & field services",
    "commerce": "Commerce & supply chain",
    "people_ops": "Hiring & workplace operations",
}


class LiveScoutConfigurationError(IngestionError):
    """Raised when a non-live provider is used for public-source discovery."""


@dataclass(frozen=True)
class CustomerSegment:
    """A customer role used to seed evidence searches without assuming a problem."""

    key: str
    label: str
    search_terms: str
    focus: str


CUSTOMER_SEGMENTS: tuple[CustomerSegment, ...] = (
    CustomerSegment(
        "clinic-operations",
        "Independent clinic operations",
        "clinic manager practice administrator",
        "healthcare",
    ),
    CustomerSegment(
        "accounting-firms",
        "Small accounting firms",
        "accountant bookkeeping firm owner",
        "professional_services",
    ),
    CustomerSegment(
        "property-management",
        "Property managers",
        "property manager landlord operations",
        "field_services",
    ),
    CustomerSegment(
        "ecommerce-operations",
        "Ecommerce operations teams",
        "ecommerce operations manager merchant",
        "commerce",
    ),
    CustomerSegment(
        "recruiting-teams",
        "Recruiting teams",
        "recruiter talent acquisition manager",
        "people_ops",
    ),
    CustomerSegment(
        "therapy-practices",
        "Independent therapy practices",
        "therapy practice owner office manager",
        "healthcare",
    ),
    CustomerSegment(
        "marketing-agencies",
        "Marketing agencies",
        "marketing agency owner account manager",
        "professional_services",
    ),
    CustomerSegment(
        "construction-teams",
        "Small construction teams",
        "construction project manager subcontractor",
        "field_services",
    ),
    CustomerSegment(
        "distributors",
        "Small distributors",
        "wholesale distributor operations manager",
        "commerce",
    ),
    CustomerSegment(
        "small-hr-teams",
        "Small HR teams",
        "HR manager people operations small business",
        "people_ops",
    ),
    CustomerSegment(
        "insurance-brokers",
        "Independent insurance brokers",
        "insurance broker agency owner operations",
        "professional_services",
    ),
    CustomerSegment(
        "field-service-businesses",
        "Local field-service businesses",
        "field service manager home service business owner",
        "field_services",
    ),
    CustomerSegment(
        "manufacturers",
        "Small manufacturers",
        "small manufacturer operations production manager",
        "commerce",
    ),
    CustomerSegment(
        "dental-practices",
        "Independent dental practices",
        "dental practice manager office administrator",
        "healthcare",
    ),
    CustomerSegment(
        "regulated-businesses",
        "Regulated small businesses",
        "compliance manager regulated small business",
        "people_ops",
    ),
)

SCOUT_RELEVANCE_TERMS: dict[str, tuple[str, ...]] = {
    "clinic-operations": ("clinic", "practice", "patient", "medical office"),
    "accounting-firms": ("accountant", "accounting", "bookkeeping", "bookkeeper"),
    "property-management": (
        "property manager",
        "property management",
        "landlord",
        "tenant",
    ),
    "ecommerce-operations": ("ecommerce", "e-commerce", "merchant", "online store"),
    "recruiting-teams": ("recruiter", "recruiting", "talent acquisition", "candidate"),
    "therapy-practices": ("therapist", "therapy practice", "counselor", "client"),
    "marketing-agencies": ("marketing agency", "agency owner", "client campaign"),
    "construction-teams": ("construction", "contractor", "subcontractor", "jobsite"),
    "distributors": ("distributor", "wholesale", "warehouse", "inventory"),
    "small-hr-teams": (
        "hr manager",
        "human resources",
        "people operations",
        "employee",
    ),
    "insurance-brokers": (
        "insurance broker",
        "insurance agency",
        "policyholder",
        "carrier",
    ),
    "field-service-businesses": (
        "field service",
        "home service",
        "technician",
        "service call",
    ),
    "manufacturers": ("manufacturer", "manufacturing", "production", "factory"),
    "dental-practices": ("dental practice", "dentist", "dental office", "patient"),
    "regulated-businesses": ("compliance", "regulated", "audit", "regulation"),
}

SCOUT_QUERY_ANCHORS: dict[str, str] = {
    "clinic-operations": "clinic operations",
    "accounting-firms": "accounting firm",
    "property-management": "property management",
    "ecommerce-operations": "ecommerce operations",
    "recruiting-teams": "recruiting",
    "therapy-practices": "therapy practice",
    "marketing-agencies": "marketing agency",
    "construction-teams": "construction project management",
    "distributors": "wholesale distribution",
    "small-hr-teams": "HR operations",
    "insurance-brokers": "insurance agency",
    "field-service-businesses": "field service business",
    "manufacturers": "manufacturing operations",
    "dental-practices": "dental practice",
    "regulated-businesses": "small business compliance",
}

SCOUT_SEARCH_LENSES: tuple[str, ...] = (
    "problem workaround",
    "feature request manual",
    "cannot automate repetitive",
    "errors missed handoffs",
    "too expensive workaround",
    "takes hours every week",
)

SCOUT_WORKFLOW_TOPICS: dict[str, tuple[str, ...]] = {
    "clinic-operations": (
        "patient referral follow up",
        "insurance authorization tracking",
        "patient intake paperwork",
        "appointment reminder coordination",
    ),
    "accounting-firms": (
        "chasing client documents",
        "month end reconciliation",
        "invoice and payment follow up",
        "bookkeeping data entry",
    ),
    "property-management": (
        "maintenance request coordination",
        "tenant communication",
        "vendor scheduling follow up",
        "inspection tracking",
    ),
    "ecommerce-operations": (
        "order tracking customer emails",
        "returns refunds exceptions",
        "inventory fulfillment reconciliation",
        "supplier purchase order coordination",
    ),
    "recruiting-teams": (
        "interview feedback reminders",
        "interview scheduling",
        "applicant screening backlog",
        "offer approval coordination",
    ),
    "therapy-practices": (
        "client intake paperwork",
        "insurance billing follow up",
        "appointment scheduling reminders",
        "clinical note administration",
    ),
    "marketing-agencies": (
        "client content approvals",
        "campaign reporting",
        "asset and feedback tracking",
        "scope change requests",
    ),
    "construction-teams": (
        "change order tracking",
        "subcontractor scheduling",
        "jobsite progress updates",
        "invoice approval coordination",
    ),
    "distributors": (
        "purchase order tracking",
        "inventory discrepancy reconciliation",
        "customer order status",
        "supplier follow up",
    ),
    "small-hr-teams": (
        "employee onboarding tasks",
        "leave request tracking",
        "performance review reminders",
        "policy acknowledgement tracking",
    ),
    "insurance-brokers": (
        "policy renewal follow up",
        "carrier quote comparison",
        "client document collection",
        "claims status communication",
    ),
    "field-service-businesses": (
        "technician scheduling dispatch",
        "customer appointment updates",
        "service quote follow up",
        "job completion paperwork",
    ),
    "manufacturers": (
        "production schedule tracking",
        "quality issue reporting",
        "supplier delivery follow up",
        "inventory material planning",
    ),
    "dental-practices": (
        "insurance verification",
        "patient recall follow up",
        "treatment plan coordination",
        "appointment cancellation filling",
    ),
    "regulated-businesses": (
        "compliance evidence collection",
        "audit preparation tracking",
        "policy review reminders",
        "regulatory reporting workflow",
    ),
}

GENERIC_WORKFLOW_TERMS = {
    "client",
    "customer",
    "employee",
    "patient",
    "supplier",
    "tenant",
    "vendor",
}

FIRST_HAND_MARKERS = (" i ", " i'm ", " i've ", " my ", " we ", " we're ", " our ")
FIRST_HAND_PAIN_MARKERS = (
    "manual",
    "spreadsheet",
    "excel",
    "copy-paste",
    "copy paste",
    "takes hours",
    "waste hours",
    "frustrating",
    "struggling",
    "killing our",
    "difficult",
    "missed",
    "errors",
    "problem",
    "painful",
)
OPERATIONAL_PAIN_MARKERS = (
    "manual",
    "manually",
    "spreadsheet",
    "excel",
    "copy-paste",
    "copy paste",
    "takes hours",
    "hours every",
    "every day",
    "every week",
    "repetitive",
    "backlog",
    "overwhelmed",
    "missed",
    "errors",
    "expensive",
    "slow",
    "frustrating",
    "struggling",
    "difficult",
    "painful",
    "tedious",
    "annoying",
    "broken",
    "cannot",
    "can't",
    "doesn't work",
    "not work",
    "not working",
    "won't work",
    "fails",
    "failing",
)
SOLICITATION_MARKERS = (
    "doing some research",
    "would love to hear",
    "what's the most frustrating",
    "what is the most frustrating",
    "what repetitive tasks",
    "i help small businesses",
    "system we built",
    "we built this",
    "offer my help",
    "book a demo",
    "dm me",
    "case study",
    "reviews reveal",
    "analyzed thousands",
    "analysed thousands",
    "real problems worth solving",
    "people i talked to",
    "person i talked to",
    "direct quote from",
    "market research",
    "who is hiring",
    "we're hiring",
    "we are hiring",
    "looking for a principal engineer",
    "apply at",
)

ISSUE_PROBLEM_MARKERS = (
    "actual behavior",
    "expected behavior",
    "bug",
    "broken",
    "cannot",
    "can't",
    "doesn't work",
    "error",
    "fails",
    "feature request",
    "limitation",
    "missing",
    "not work",
    "not working",
    "won't work",
    "problem",
    "reproduction",
    "workaround",
)

PLANNING_DOCUMENT_MARKERS = (
    "## acceptance criteria",
    "## dependencies",
    "## implementation plan",
    "## scope when picked up",
    "est. man-days",
    "migrated-to-kanban",
)


@dataclass(frozen=True)
class SourcedDiscussion:
    """A real public result associated with the neutral segment that found it."""

    evidence: WebEvidenceCandidate
    segment: CustomerSegment
    workflow_topic: str

    def to_submission(self) -> SourceSubmission:
        submission = self.evidence.to_submission()
        metadata = {
            **submission.metadata_json,
            "scout_segment": self.segment.key,
            "scout_segment_label": self.segment.label,
            "scout_workflow_topic": self.workflow_topic,
        }
        source_text = f"{self.evidence.title}. {submission.raw_text}"[:20_000]
        return submission.copy(
            update={"raw_text": source_text, "metadata_json": metadata}
        )


@dataclass(frozen=True)
class ProblemScoutOutcome:
    """One sourced discussion and its persisted discovery result."""

    source: SourcedDiscussion
    result: DiscoveryResult


@dataclass(frozen=True)
class ProblemScoutRun:
    """Complete persisted outcome of one automatic scan."""

    segments: tuple[CustomerSegment, ...]
    outcomes: tuple[ProblemScoutOutcome, ...]
    opportunities: tuple["DiscoveredOpportunity", ...]
    previously_seen_count: int = 0
    search_query_count: int = 0

    @property
    def accepted_count(self) -> int:
        return sum(outcome.result.accepted for outcome in self.outcomes)

    @property
    def rejected_count(self) -> int:
        return sum(
            not outcome.result.accepted and not outcome.result.duplicate
            for outcome in self.outcomes
        )

    @property
    def duplicate_count(self) -> int:
        return self.previously_seen_count + sum(
            outcome.result.duplicate for outcome in self.outcomes
        )

    @property
    def new_source_count(self) -> int:
        return sum(not outcome.result.duplicate for outcome in self.outcomes)

    @property
    def source_breakdown(self) -> tuple[tuple[str, int], ...]:
        counts = Counter(
            outcome.source.evidence.source_platform for outcome in self.outcomes
        )
        return tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


@dataclass(frozen=True)
class OpportunitySourceSummary:
    """Serializable public evidence shown with a discovered opportunity."""

    title: str
    url: str
    domain: str
    excerpt: str
    source_type: str = "web"
    source_author: str | None = None
    engagement_count: int = 0


@dataclass(frozen=True)
class DiscoveredOpportunity:
    """A promoted database opportunity backed by repeated public evidence."""

    cluster_id: str
    title: str
    problem_summary: str
    target_customer: str
    current_workaround: str
    proposed_solution: str
    evidence_count: int
    independent_source_count: int
    problem_score: float
    opportunity_score: float
    confidence_score: float
    synthesis_reasoning: str
    synthesis_confidence: float
    sources: tuple[OpportunitySourceSummary, ...]


def select_customer_segments(
    focus: str = "all",
    *,
    limit: int = 4,
    offset: int = 0,
) -> tuple[CustomerSegment, ...]:
    """Select rotating customer roles without preselecting their problems."""

    if focus not in SCOUT_FOCUS_LABELS:
        raise IngestionError("Select a supported market focus.")
    if not 1 <= limit <= 8:
        raise IngestionError("Problem scans must include between 1 and 8 segments.")
    segments = [
        segment
        for segment in CUSTOMER_SEGMENTS
        if focus == "all" or segment.focus == focus
    ]
    if not segments:
        return ()
    start = offset % len(segments)
    rotated = segments[start:] + segments[:start]
    return tuple(rotated[:limit])


def build_problem_query(
    segment: CustomerSegment,
    *,
    scan_round: int = 0,
    attempt: int = 0,
) -> str:
    """Build a rotating natural-language query for first-hand workflow pain."""

    topic = _workflow_topic(segment, scan_round=scan_round, attempt=attempt)
    lens = SCOUT_SEARCH_LENSES[((scan_round * 2) + attempt) % len(SCOUT_SEARCH_LENSES)]
    query = f'{SCOUT_QUERY_ANCHORS[segment.key]} "{topic}" {lens}'
    return " ".join(query.split())


def _workflow_topic(
    segment: CustomerSegment,
    *,
    scan_round: int,
    attempt: int,
) -> str:
    topics = SCOUT_WORKFLOW_TOPICS[segment.key]
    return topics[(scan_round + attempt) % len(topics)]


def _matches_segment(
    evidence: WebEvidenceCandidate,
    segment: CustomerSegment,
) -> bool:
    text = " ".join(
        (evidence.title, evidence.url, evidence.raw_text[:3_000], evidence.snippet)
    ).lower()
    return any(term in text for term in SCOUT_RELEVANCE_TERMS[segment.key])


def _matches_workflow(evidence: WebEvidenceCandidate, topic: str) -> bool:
    """Match a concrete workflow term alongside an operational pain signal."""

    text = " ".join((evidence.title, evidence.snippet[:1_200])).lower()
    topic_terms = {
        (token[:-1] if token.endswith("s") and len(token) > 4 else token)
        for token in re.findall(r"[a-z0-9]+", topic.lower())
        if len(token) >= 4 and token not in GENERIC_WORKFLOW_TERMS
    }
    for passage in re.split(r"(?<=[.!?])\s+", text):
        passage_terms = {
            token[:-1] if token.endswith("s") and len(token) > 4 else token
            for token in re.findall(r"[a-z0-9]+", passage)
        }
        if topic_terms & passage_terms and any(
            marker in passage for marker in OPERATIONAL_PAIN_MARKERS
        ):
            return True
    return False


def _is_scoutable_discussion(evidence: WebEvidenceCandidate) -> bool:
    """Exclude search landing pages that are not first-person evidence."""

    parsed = urlsplit(evidence.url)
    path = parsed.path.lower()
    title = evidence.title.lower()
    if "/compare/" in path:
        return False
    if evidence.source_platform == "github" and title.startswith(
        ("problem:", "epic-", "[epic]", "roadmap:", "tracking:")
    ):
        return False
    return not (title.startswith("compare ") or "pricing, alternatives & more" in title)


def _contains_first_hand_problem(evidence: WebEvidenceCandidate) -> bool:
    excerpt = evidence.snippet or evidence.raw_text[:3_000]
    text = " ".join((evidence.title, excerpt)).lower()
    padded = f" {text} "
    if any(marker in padded for marker in SOLICITATION_MARKERS):
        return False
    sentences = [
        f" {sentence.strip()} " for sentence in re.split(r"(?<=[.!?])\s+|\n+", padded)
    ]
    same_sentence_signal = any(
        any(subject in sentence for subject in FIRST_HAND_MARKERS)
        and any(pain in sentence for pain in OPERATIONAL_PAIN_MARKERS)
        for sentence in sentences
    )
    if same_sentence_signal:
        return True
    first_hand = any(marker in padded for marker in FIRST_HAND_MARKERS)
    pain_signal_count = sum(marker in padded for marker in FIRST_HAND_PAIN_MARKERS)
    return first_hand and pain_signal_count >= 1


def _contains_source_native_problem(evidence: WebEvidenceCandidate) -> bool:
    """Recognize explicit issue and support requests without requiring first person."""

    text = " ".join((evidence.title, evidence.raw_text[:6_000])).lower()
    if evidence.source_platform == "github":
        planning_markers = sum(marker in text for marker in PLANNING_DOCUMENT_MARKERS)
        if planning_markers >= 2:
            return False
        return any(marker in text for marker in ISSUE_PROBLEM_MARKERS) and any(
            marker in text for marker in OPERATIONAL_PAIN_MARKERS
        )
    if evidence.source_platform in {"stack_exchange", "support_community"}:
        return _contains_first_hand_problem(evidence) or (
            any(marker in text for marker in ISSUE_PROBLEM_MARKERS)
            and any(marker in text for marker in OPERATIONAL_PAIN_MARKERS)
        )
    return _contains_first_hand_problem(evidence)


def _has_enough_source_detail(evidence: WebEvidenceCandidate) -> bool:
    """Reject snippets that are too thin to support an explainable product claim."""

    text_length = len(" ".join(evidence.raw_text.split()))
    if evidence.source_platform in {"github", "stack_exchange"}:
        minimum = 180
    elif evidence.source_platform in {"hacker_news", "reddit"}:
        minimum = 80
    else:
        minimum = 120
    return text_length >= minimum


def _source_token_set(evidence: WebEvidenceCandidate) -> set[str]:
    fingerprint_text = f"{evidence.title} {evidence.snippet[:1_200]}"
    return {
        token
        for token in re.findall(r"[a-z0-9]+", fingerprint_text.lower())
        if len(token) >= 4
    }


def _is_near_duplicate_source(
    evidence: WebEvidenceCandidate,
    accepted_token_sets: list[set[str]],
) -> bool:
    tokens = _source_token_set(evidence)
    if not tokens:
        return True
    return any(
        len(tokens & existing) / max(1, min(len(tokens), len(existing))) >= 0.78
        for existing in accepted_token_sets
    )


class ProblemScoutService:
    """Search real sources, extract problems, and persist resulting opportunities."""

    def __init__(
        self,
        provider: SearchProvider,
        discovery: DiscoveryService,
        synthesizer: OpportunitySynthesisProvider,
        *,
        search_depth: str = "basic",
        minimum_independent_sources: int = 2,
        minimum_synthesis_confidence: float = 0.55,
    ) -> None:
        if provider.name == "mock":
            raise LiveScoutConfigurationError(
                "Public problem scouting requires a real-source search provider."
            )
        if minimum_independent_sources < 2:
            raise ValueError("Opportunity promotion requires at least two sources.")
        self.provider = provider
        self.discovery = discovery
        self.synthesizer = synthesizer
        self.search_depth = search_depth
        self.minimum_independent_sources = minimum_independent_sources
        self.minimum_synthesis_confidence = minimum_synthesis_confidence
        self.clusters = ClusterRepository(discovery.session)

    def run(
        self,
        *,
        focus: str = "all",
        segment_limit: int = 4,
        results_per_segment: int = 2,
        offset: int = 0,
        scan_round: int = 0,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> ProblemScoutRun:
        """Run one complete search-to-database discovery cycle."""

        if not 1 <= results_per_segment <= 20:
            raise IngestionError(
                "Results per customer segment must be between 1 and 20."
            )
        segments = select_customer_segments(
            focus,
            limit=segment_limit,
            offset=offset,
        )
        self._restore_scout_candidates()
        sources, previously_seen_count, search_query_count = self._search_sources(
            segments,
            results_per_segment=results_per_segment,
            scan_round=scan_round,
            progress_callback=progress_callback,
        )
        outcomes: list[ProblemScoutOutcome] = []
        touched_cluster_ids: set[str] = set()
        total = max(1, len(sources))
        for index, source in enumerate(sources, start=1):
            if progress_callback:
                progress_callback(
                    0.4 + (index / total) * 0.4,
                    f"Extracting problem evidence {index} of {len(sources)}",
                )
            result = self.discovery.process(source.to_submission())
            outcomes.append(ProblemScoutOutcome(source=source, result=result))
            cluster_ids = self._cluster_ids_for_result(result)
            touched_cluster_ids.update(cluster_ids)
            for cluster_id in cluster_ids:
                self._mark_pipeline_stage(cluster_id)

        opportunities = self._promote_opportunities(
            touched_cluster_ids,
            progress_callback=progress_callback,
        )
        if progress_callback:
            progress_callback(
                1.0,
                (
                    f"Saved {len(opportunities)} evidence-backed opportunity lead(s)"
                    if opportunities
                    else "Saved evidence; no repeated problem met the promotion threshold"
                ),
            )
        return ProblemScoutRun(
            segments=segments,
            outcomes=tuple(outcomes),
            opportunities=tuple(opportunities),
            previously_seen_count=previously_seen_count,
            search_query_count=search_query_count,
        )

    def _search_sources(
        self,
        segments: tuple[CustomerSegment, ...],
        *,
        results_per_segment: int,
        scan_round: int,
        progress_callback: Callable[[float, str], None] | None,
    ) -> tuple[list[SourcedDiscussion], int, int]:
        stored_urls = {
            canonical_url(url) for url in self.discovery.evidence.list_source_urls()
        }
        seen_urls: set[str] = set(stored_urls)
        previously_seen_urls: set[str] = set()
        sources: list[SourcedDiscussion] = []
        accepted_token_sets: list[set[str]] = []
        query_count = 0
        total = max(1, len(segments))
        for index, segment in enumerate(segments, start=1):
            if progress_callback:
                progress_callback(
                    (index - 1) / total * 0.4,
                    f"Searching public discussions for {segment.label}",
                )
            segment_source_count = 0
            domain_counts: dict[str, int] = {}
            for attempt in range(2):
                query = build_problem_query(
                    segment,
                    scan_round=scan_round,
                    attempt=attempt,
                )
                workflow_topic = _workflow_topic(
                    segment,
                    scan_round=scan_round,
                    attempt=attempt,
                )
                query_count += 1
                results = self.provider.search(
                    query,
                    max_results=results_per_segment,
                    search_depth=self.search_depth,
                )
                for result in results:
                    evidence = candidate_from_search_result(result, query=query)
                    if evidence is None:
                        continue
                    if not is_public_source_url(evidence.url):
                        continue
                    if not is_supported_discussion_url(evidence.url):
                        continue
                    if not _is_scoutable_discussion(evidence):
                        continue
                    if not _has_enough_source_detail(evidence):
                        continue
                    if not _matches_segment(evidence, segment):
                        continue
                    url_key = canonical_url(evidence.url)
                    if url_key in stored_urls:
                        previously_seen_urls.add(url_key)
                        continue
                    if url_key in seen_urls:
                        continue
                    if not _matches_workflow(evidence, workflow_topic):
                        continue
                    if not _contains_source_native_problem(evidence):
                        continue
                    if _is_near_duplicate_source(evidence, accepted_token_sets):
                        continue
                    domain = urlsplit(evidence.url).netloc.lower().removeprefix("www.")
                    if domain_counts.get(domain, 0) >= 2:
                        continue
                    seen_urls.add(url_key)
                    accepted_token_sets.append(_source_token_set(evidence))
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                    sources.append(
                        SourcedDiscussion(
                            evidence=evidence,
                            segment=segment,
                            workflow_topic=workflow_topic,
                        )
                    )
                    segment_source_count += 1
                if segment_source_count >= 3:
                    break
        return sources, len(previously_seen_urls), query_count

    def _cluster_ids_for_result(self, result: DiscoveryResult) -> set[str]:
        if not result.accepted:
            return set()
        if result.assignment is not None:
            return {result.assignment.cluster.id}
        return set(self.clusters.cluster_ids_for_evidence(result.evidence.id))

    def _mark_pipeline_stage(self, cluster_id: str) -> None:
        cluster = self.clusters.get(cluster_id)
        if (
            cluster is not None
            and cluster.independent_source_count < self.minimum_independent_sources
            and self._is_scout_only(cluster)
            and cluster.status != "researched"
        ):
            cluster.status = "candidate"
            self._refresh_candidate_identity(cluster)
            self.clusters.save(cluster)

    def _restore_scout_candidates(self) -> None:
        """Bring one-source scout records created by older releases into the pipeline."""

        for cluster in self.clusters.list_with_evidence(limit=10_000):
            if (
                cluster.independent_source_count < self.minimum_independent_sources
                and self._is_scout_only(cluster)
                and cluster.status != "researched"
            ):
                if cluster.status == "archived":
                    cluster.status = "candidate"
                self._refresh_candidate_identity(cluster)
                self.clusters.save(cluster)

    @staticmethod
    def _refresh_candidate_identity(cluster: OpportunityCluster) -> None:
        """Use scout metadata and source titles to label an early pipeline signal."""

        items = [link.evidence_item for link in cluster.evidence_links]
        if not cluster.target_customer:
            cluster.target_customer = next(
                (
                    str((item.metadata_json or {}).get("scout_segment_label"))
                    for item in items
                    if (item.metadata_json or {}).get("scout_segment_label")
                ),
                None,
            )

        source_title = next(
            (
                item.title
                for item in items
                if item.title and "heart of the internet" not in item.title.lower()
            ),
            None,
        )
        candidate = source_title or cluster.problem_summary
        cleaned = re.sub(r"(?:^|\s)#{1,6}\s*", " ", candidate)
        cleaned = re.sub(r"\s*:\s*r/[A-Za-z0-9_]+.*$", "", cleaned)
        cleaned = cleaned.replace("Skip to main content", " ")
        cleaned = " ".join(cleaned.split()).strip(" .:-")
        if cleaned:
            cluster.title = (
                f"{cleaned[:87].rstrip()}..." if len(cleaned) > 90 else cleaned
            )
        cluster.proposed_solution = build_opportunity_brief(cluster).product_hypothesis

    def _promote_opportunities(
        self,
        cluster_ids: set[str],
        *,
        progress_callback: Callable[[float, str], None] | None,
    ) -> list[DiscoveredOpportunity]:
        eligible: list[OpportunityCluster] = []
        for cluster_id in sorted(cluster_ids):
            cluster = self.clusters.get(cluster_id)
            if (
                cluster is not None
                and cluster.independent_source_count >= self.minimum_independent_sources
            ):
                eligible.append(cluster)

        opportunities: list[DiscoveredOpportunity] = []
        total = max(1, len(eligible))
        for index, cluster in enumerate(eligible, start=1):
            if progress_callback:
                progress_callback(
                    0.82 + (index / total) * 0.16,
                    f"Validating repeated problem {index} of {len(eligible)}",
                )
            evidence_items = [
                link.evidence_item
                for link in cluster.evidence_links
                if link.evidence_item.contains_problem
            ]
            draft = self.synthesizer.synthesize(cluster, evidence_items)
            if (
                not draft.supported
                or draft.confidence < self.minimum_synthesis_confidence
            ):
                if cluster.status != "researched" and self._is_scout_only(cluster):
                    cluster.status = "candidate"
                    self.clusters.save(cluster)
                continue

            cluster.title = draft.title
            cluster.problem_summary = draft.problem_summary
            cluster.target_customer = draft.target_customer
            cluster.current_workaround = draft.current_workaround
            cluster.proposed_solution = draft.proposed_solution
            if cluster.status in {"archived", "candidate"} and self._is_scout_only(
                cluster
            ):
                cluster.status = "new"
            self.clusters.save(cluster)
            score = self.discovery.scorer.score_cluster(cluster.id)
            refreshed = self.clusters.get(cluster.id)
            if refreshed is None:
                continue
            opportunities.append(
                self._serialize_opportunity(
                    refreshed,
                    score,
                    reasoning=draft.reasoning,
                    synthesis_confidence=draft.confidence,
                )
            )

        return sorted(
            opportunities,
            key=lambda opportunity: opportunity.opportunity_score,
            reverse=True,
        )

    @staticmethod
    def _is_scout_only(cluster: OpportunityCluster) -> bool:
        items = [link.evidence_item for link in cluster.evidence_links]
        return bool(items) and all(
            (item.metadata_json or {}).get("scout_segment") for item in items
        )

    @staticmethod
    def _serialize_opportunity(
        cluster: OpportunityCluster,
        score: OpportunityScore,
        *,
        reasoning: str,
        synthesis_confidence: float,
    ) -> DiscoveredOpportunity:
        sources: list[OpportunitySourceSummary] = []
        seen_urls: set[str] = set()
        for link in cluster.evidence_links:
            item = link.evidence_item
            if not is_public_source_url(item.source_url):
                continue
            url_key = canonical_url(item.source_url)
            if url_key in seen_urls:
                continue
            seen_urls.add(url_key)
            quote = (item.metadata_json or {}).get("evidence_quote") or item.raw_text
            sources.append(
                OpportunitySourceSummary(
                    title=item.title or item.problem_statement or "Public discussion",
                    url=item.source_url,
                    domain=item.community or urlsplit(item.source_url).netloc,
                    excerpt=" ".join(str(quote).split())[:500],
                    source_type=item.platform,
                    source_author=item.source_author,
                    engagement_count=int(
                        (item.metadata_json or {}).get("engagement_count") or 0
                    ),
                )
            )

        problem_score = float(
            (score.explanation_json or {}).get("problem_score", {}).get("score", 0.0)
        )
        return DiscoveredOpportunity(
            cluster_id=cluster.id,
            title=cluster.title,
            problem_summary=cluster.problem_summary,
            target_customer=cluster.target_customer or "Not established",
            current_workaround=cluster.current_workaround or "Not established",
            proposed_solution=cluster.proposed_solution or "Not generated",
            evidence_count=cluster.evidence_count,
            independent_source_count=cluster.independent_source_count,
            problem_score=problem_score,
            opportunity_score=float(score.opportunity_score),
            confidence_score=float(score.confidence_score),
            synthesis_reasoning=reasoning,
            synthesis_confidence=synthesis_confidence,
            sources=tuple(sources),
        )
