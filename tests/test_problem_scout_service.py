"""End-to-end tests for evidence-backed automatic problem discovery."""

import pytest
from sqlalchemy.orm import Session

from src.clustering.clusterer import IncrementalClusterer
from src.clustering.embeddings import DeterministicEmbeddingProvider
from src.database.repositories import ClusterRepository
from src.extraction.opportunity_synthesizer import (
    DeterministicOpportunitySynthesizer,
    OpportunitySynthesisError,
    ResilientOpportunitySynthesizer,
)
from src.extraction.problem_extractor import (
    DeterministicMockExtractionProvider,
    ProblemExtractor,
)
from src.research.competitor_search import MockSearchProvider
from src.research.schemas import SearchResult
from src.scoring.opportunity_score import OpportunityScorer
from src.services.discovery_service import DiscoveryService
from src.services.opportunity_service import OpportunityService
from src.services.problem_scout_service import (
    LiveScoutConfigurationError,
    ProblemScoutService,
)


class StaticLiveSearchProvider:
    """Return controlled public results while exercising the live-only contract."""

    name = "tavily"

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.queries: list[str] = []

    def search(
        self,
        query: str,
        *,
        max_results: int,
        search_depth: str,
    ) -> list[SearchResult]:
        del search_depth
        self.queries.append(query)
        return self.results[:max_results]


def _discovery(session: Session) -> DiscoveryService:
    return DiscoveryService(
        session,
        ProblemExtractor(DeterministicMockExtractionProvider()),
        IncrementalClusterer(
            session,
            DeterministicEmbeddingProvider(),
            threshold=0.60,
        ),
        OpportunityScorer(session),
    )


def _repeated_problem_results() -> list[SearchResult]:
    return [
        SearchResult(
            title="Referral follow-up consumes hours each week",
            url="https://www.reddit.com/r/healthcare/comments/abc123/referral_followup/",
            snippet=(
                "As a clinic manager, we still use spreadsheets to track referral "
                "follow-up. The manual process takes hours every week and missed "
                "referrals create risk."
            ),
            score=0.94,
        ),
        SearchResult(
            title="Tracking referrals in Excel keeps failing",
            url="https://news.ycombinator.com/item?id=42424242",
            snippet=(
                "As a practice administrator, referral follow-up is tracked in Excel. "
                "We copy-paste status every day, and errors are easy to miss."
            ),
            score=0.89,
        ),
    ]


def test_scout_promotes_one_persisted_opportunity_across_pages(
    db_session: Session,
) -> None:
    provider = StaticLiveSearchProvider(_repeated_problem_results())
    run = ProblemScoutService(
        provider,
        _discovery(db_session),
        DeterministicOpportunitySynthesizer(),
    ).run(focus="healthcare", segment_limit=1, results_per_segment=2)

    assert provider.queries
    assert len(run.outcomes) == 2
    assert len(run.opportunities) == 1
    lead = run.opportunities[0]
    assert {source.url for source in lead.sources} == {
        result.url for result in _repeated_problem_results()
    }

    stored = ClusterRepository(db_session).get(lead.cluster_id)
    assert stored is not None
    assert stored.status == "new"
    assert stored.title == lead.title
    assert stored.problem_summary == lead.problem_summary
    assert stored.proposed_solution == lead.proposed_solution
    assert stored.independent_source_count == 2

    ranked = OpportunityService(db_session).ranked_opportunities(limit=10)
    details = ClusterRepository(db_session).list_promoted(limit=10)
    assert [row.cluster_id for row in ranked] == [lead.cluster_id]
    assert [cluster.id for cluster in details] == [lead.cluster_id]


def test_scout_keeps_one_off_evidence_out_of_opportunity_pages(
    db_session: Session,
) -> None:
    provider = StaticLiveSearchProvider(_repeated_problem_results()[:1])
    run = ProblemScoutService(
        provider,
        _discovery(db_session),
        DeterministicOpportunitySynthesizer(),
    ).run(focus="healthcare", segment_limit=1, results_per_segment=1)

    assert len(run.outcomes) == 1
    assert run.opportunities == ()
    assert OpportunityService(db_session).ranked_opportunities(limit=10) == []
    clusters = ClusterRepository(db_session).list(limit=10)
    assert len(clusters) == 1
    assert clusters[0].status == "archived"


def test_scout_uses_local_synthesis_when_openai_is_rate_limited(
    db_session: Session,
) -> None:
    class RateLimitedSynthesizer:
        def synthesize(self, cluster, evidence_items):
            del cluster, evidence_items
            raise OpportunitySynthesisError("OpenAI rate limit reached.")

    run = ProblemScoutService(
        StaticLiveSearchProvider(_repeated_problem_results()),
        _discovery(db_session),
        ResilientOpportunitySynthesizer(
            RateLimitedSynthesizer(),
            DeterministicOpportunitySynthesizer(),
        ),
    ).run(focus="healthcare", segment_limit=1, results_per_segment=2)

    assert len(run.opportunities) == 1
    assert "referral" in run.opportunities[0].problem_summary.lower()


def test_scout_rejects_mock_search_and_placeholder_sources(
    db_session: Session,
) -> None:
    with pytest.raises(LiveScoutConfigurationError, match="real-source"):
        ProblemScoutService(
            MockSearchProvider(),
            _discovery(db_session),
            DeterministicOpportunitySynthesizer(),
        )

    provider = StaticLiveSearchProvider(
        [
            SearchResult(
                title="Invented source",
                url="https://community.example/not-real",
                snippet="This manual process takes hours every week.",
                score=0.99,
            )
        ]
    )
    run = ProblemScoutService(
        provider,
        _discovery(db_session),
        DeterministicOpportunitySynthesizer(),
    ).run(focus="healthcare", segment_limit=1, results_per_segment=1)

    assert run.outcomes == ()
    assert run.opportunities == ()
    assert ClusterRepository(db_session).list(limit=10) == []


def test_scout_rejects_generic_pages_solicitations_and_vendor_posts(
    db_session: Session,
) -> None:
    provider = StaticLiveSearchProvider(
        [
            SearchResult(
                title="Small business discussions",
                url="https://www.reddit.com/r/smallbusiness/",
                snippet="What repetitive tasks take the most time?",
                score=0.95,
            ),
            SearchResult(
                title="Tell me about your clinic problems",
                url="https://www.reddit.com/r/healthcare/comments/research/problems/",
                snippet=(
                    "I am doing some research and would love to hear what manual "
                    "clinic work is frustrating."
                ),
                score=0.92,
            ),
            SearchResult(
                title="We automated a clinic",
                url="https://www.reddit.com/r/SaaS/comments/vendor/clinic_automation/",
                snippet=(
                    "Here is the system we built for a clinic with a manual "
                    "spreadsheet process."
                ),
                score=0.90,
            ),
            SearchResult(
                title="Referral handoffs keep getting missed",
                url="https://www.reddit.com/r/healthIT/comments/real/referrals/",
                snippet=(
                    "As a clinic manager, our referral spreadsheet is manual and "
                    "we keep missing follow-up every week."
                ),
                score=0.88,
            ),
        ]
    )

    run = ProblemScoutService(
        provider,
        _discovery(db_session),
        DeterministicOpportunitySynthesizer(),
    ).run(focus="healthcare", segment_limit=1, results_per_segment=4)

    assert [outcome.source.evidence.url for outcome in run.outcomes] == [
        "https://www.reddit.com/r/healthIT/comments/real/referrals/"
    ]
