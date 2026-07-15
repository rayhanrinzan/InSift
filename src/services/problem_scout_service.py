"""Real-source problem scouting that persists results end to end."""

from __future__ import annotations

import re
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
    "small-hr-teams": ("hr manager", "human resources", "people operations", "employee"),
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
)


@dataclass(frozen=True)
class SourcedDiscussion:
    """A real public result associated with the neutral segment that found it."""

    evidence: WebEvidenceCandidate
    segment: CustomerSegment

    def to_submission(self) -> SourceSubmission:
        submission = self.evidence.to_submission()
        metadata = {
            **submission.metadata_json,
            "scout_segment": self.segment.key,
            "scout_segment_label": self.segment.label,
        }
        return submission.copy(update={"metadata_json": metadata})


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
        return sum(outcome.result.duplicate for outcome in self.outcomes)


@dataclass(frozen=True)
class OpportunitySourceSummary:
    """Serializable public evidence shown with a discovered opportunity."""

    title: str
    url: str
    domain: str
    excerpt: str


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


def build_problem_query(segment: CustomerSegment) -> str:
    """Build a broad query for first-hand pain, workarounds, and repeated work."""

    query = (
        f"{segment.search_terms} first hand customer complaint "
        "manual repetitive workaround spreadsheet follow-up hours frustrating"
    )
    return " ".join(query.split())


def _matches_segment(
    evidence: WebEvidenceCandidate,
    segment: CustomerSegment,
) -> bool:
    text = " ".join(
        (evidence.title, evidence.url, evidence.raw_text[:3_000], evidence.snippet)
    ).lower()
    return any(term in text for term in SCOUT_RELEVANCE_TERMS[segment.key])


def _contains_first_hand_problem(evidence: WebEvidenceCandidate) -> bool:
    text = " ".join(
        (evidence.title, evidence.raw_text[:3_000], evidence.snippet)
    ).lower()
    padded = f" {text} "
    if any(marker in padded for marker in SOLICITATION_MARKERS):
        return False
    sentences = [
        f" {sentence.strip()} "
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", padded)
    ]
    return any(
        any(subject in sentence for subject in FIRST_HAND_MARKERS)
        and any(pain in sentence for pain in FIRST_HAND_PAIN_MARKERS)
        for sentence in sentences
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
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> ProblemScoutRun:
        """Run one complete search-to-database discovery cycle."""

        if not 1 <= results_per_segment <= 10:
            raise IngestionError("Results per customer segment must be between 1 and 10.")
        segments = select_customer_segments(
            focus,
            limit=segment_limit,
            offset=offset,
        )
        sources = self._search_sources(
            segments,
            results_per_segment=results_per_segment,
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
            if result.assignment and result.assignment.created:
                self._hide_uncorroborated_cluster(result.assignment.cluster.id)

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
        )

    def _search_sources(
        self,
        segments: tuple[CustomerSegment, ...],
        *,
        results_per_segment: int,
        progress_callback: Callable[[float, str], None] | None,
    ) -> list[SourcedDiscussion]:
        seen_urls: set[str] = set()
        sources: list[SourcedDiscussion] = []
        total = max(1, len(segments))
        for index, segment in enumerate(segments, start=1):
            if progress_callback:
                progress_callback(
                    (index - 1) / total * 0.4,
                    f"Searching public discussions for {segment.label}",
                )
            query = build_problem_query(segment)
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
                if not _matches_segment(evidence, segment):
                    continue
                if not _contains_first_hand_problem(evidence):
                    continue
                url_key = canonical_url(evidence.url)
                if url_key in seen_urls:
                    continue
                seen_urls.add(url_key)
                sources.append(SourcedDiscussion(evidence=evidence, segment=segment))
        return sources

    def _cluster_ids_for_result(self, result: DiscoveryResult) -> set[str]:
        if not result.accepted:
            return set()
        if result.assignment is not None:
            return {result.assignment.cluster.id}
        return set(self.clusters.cluster_ids_for_evidence(result.evidence.id))

    def _hide_uncorroborated_cluster(self, cluster_id: str) -> None:
        cluster = self.clusters.get(cluster_id)
        if (
            cluster is not None
            and cluster.independent_source_count < self.minimum_independent_sources
            and self._is_scout_only(cluster)
        ):
            cluster.status = "archived"
            self.clusters.save(cluster)

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
                and cluster.independent_source_count
                >= self.minimum_independent_sources
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
                    cluster.status = "archived"
                    self.clusters.save(cluster)
                continue

            cluster.title = draft.title
            cluster.problem_summary = draft.problem_summary
            cluster.target_customer = draft.target_customer
            cluster.current_workaround = draft.current_workaround
            cluster.proposed_solution = draft.proposed_solution
            if cluster.status == "archived" and self._is_scout_only(cluster):
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
                )
            )

        problem_score = float(
            (score.explanation_json or {})
            .get("problem_score", {})
            .get("score", 0.0)
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
