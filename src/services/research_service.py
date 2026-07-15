"""Competitor research orchestration with query and result traceability."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from sqlalchemy.orm import Session

from src.config import Settings
from src.database.models import Competitor, OpportunityScore, ResearchRun, SearchQuery
from src.database.repositories import (
    ClusterRepository,
    CompetitorRepository,
    ResearchRepository,
)
from src.logging_config import log_event
from src.research.competitor_classifier import (
    CompetitorClassificationProvider,
    build_competitor_classifier,
    is_product_candidate,
    product_identity,
)
from src.research.competitor_search import (
    SearchAuthenticationError,
    SearchProvider,
    SearchProviderError,
    build_search_provider,
    canonical_url,
)
from src.research.query_generator import generate_competitor_queries
from src.research.schemas import CompetitorResearchContext, SearchResult
from src.scoring.opportunity_score import OpportunityScorer


logger = logging.getLogger(__name__)

ResearchProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class ResearchOutcome:
    """Completed competitor research summary."""

    run: ResearchRun
    queries: list[SearchQuery]
    competitors: list[Competitor]
    irrelevant_result_count: int
    score: OpportunityScore | None


class ResearchService:
    """Generate queries, search, classify, deduplicate, persist, and rescore."""

    def __init__(
        self,
        session: Session,
        provider: SearchProvider,
        classifier: CompetitorClassificationProvider,
        *,
        max_results: int = 10,
        search_depth: str = "basic",
    ) -> None:
        self.session = session
        self.provider = provider
        self.classifier = classifier
        self.max_results = max_results
        self.search_depth = search_depth
        self.clusters = ClusterRepository(session)
        self.competitors = CompetitorRepository(session)
        self.research = ResearchRepository(session)

    def research_cluster(
        self,
        cluster_id: str,
        progress_callback: Optional[ResearchProgressCallback] = None,
    ) -> ResearchOutcome:
        """Run a complete research cycle for one opportunity cluster."""

        self._report_progress(progress_callback, 0.03, "Loading opportunity context")
        cluster = self.clusters.get(cluster_id)
        if cluster is None:
            raise ValueError("Cluster does not exist.")
        generated_queries = generate_competitor_queries(cluster)
        self._report_progress(
            progress_callback,
            0.08,
            f"Generated {len(generated_queries)} research queries",
        )
        run = self.research.create_run(cluster_id, self.provider.name)
        query_records = [
            self.research.create_query(run.id, cluster_id, query)
            for query in generated_queries
        ]
        unique_results: dict[str, tuple[SearchResult, list[str]]] = {}
        total_results = 0
        failed_queries = 0
        permanent_error: str | None = None

        for index, query_record in enumerate(query_records):
            query_progress = 0.1 + (0.5 * (index / max(1, len(query_records))))
            self._report_progress(
                progress_callback,
                query_progress,
                f"Searching query {index + 1} of {len(query_records)}",
            )
            if permanent_error:
                self.research.finish_query(
                    query_record,
                    result_count=0,
                    error_message="Not executed after permanent provider failure.",
                )
                failed_queries += 1
                continue
            log_event(
                logger,
                logging.INFO,
                "search_query",
                {"cluster_id": cluster_id, "query": query_record.query_text},
            )
            try:
                results = self.provider.search(
                    query_record.query_text,
                    max_results=self.max_results,
                    search_depth=self.search_depth,
                )
                total_results += len(results)
                self.research.finish_query(query_record, result_count=len(results))
                for result in results:
                    if not result.url:
                        continue
                    key = canonical_url(result.url)
                    if key in unique_results:
                        unique_results[key][1].append(query_record.query_text)
                    else:
                        unique_results[key] = (result, [query_record.query_text])
            except SearchAuthenticationError as exc:
                permanent_error = str(exc)
                failed_queries += 1
                self.research.finish_query(
                    query_record, result_count=0, error_message=permanent_error
                )
            except SearchProviderError as exc:
                failed_queries += 1
                self.research.finish_query(
                    query_record, result_count=0, error_message=str(exc)
                )

        self._report_progress(
            progress_callback,
            0.62,
            f"Classifying {len(unique_results)} unique search results",
        )

        context = CompetitorResearchContext(
            title=cluster.title,
            problem_summary=cluster.problem_summary,
            target_customer=cluster.target_customer,
            current_workaround=cluster.current_workaround,
            proposed_solution=cluster.proposed_solution,
        )
        persisted: list[Competitor] = []
        irrelevant_count = 0
        existing_records = self.competitors.list_for_cluster(cluster_id)
        seen_products: set[str] = set()
        relationship_counts = {"direct": 0, "adjacent": 0, "substitute": 0}
        relationship_limits = {"direct": 4, "adjacent": 5, "substitute": 3}
        for result_index, (url, (result, source_queries)) in enumerate(
            unique_results.items()
        ):
            classification_progress = 0.62 + (
                0.27 * ((result_index + 1) / max(1, len(unique_results)))
            )
            self._report_progress(
                progress_callback,
                classification_progress,
                f"Classifying result {result_index + 1} of {len(unique_results)}",
            )
            if not is_product_candidate(result):
                irrelevant_count += 1
                continue
            classification = self.classifier.classify(context, result)
            log_event(
                logger,
                logging.INFO,
                "competitor_classification",
                {
                    "cluster_id": cluster_id,
                    "url": url,
                    "relationship_type": classification.relationship_type,
                    "confidence": classification.confidence,
                },
            )
            if classification.relationship_type == "irrelevant":
                irrelevant_count += 1
                continue
            identity = product_identity(classification, url)
            relationship = classification.relationship_type
            if (
                identity in seen_products
                or relationship_counts[relationship]
                >= relationship_limits[relationship]
            ):
                irrelevant_count += 1
                continue
            seen_products.add(identity)
            relationship_counts[relationship] += 1
            existing = next(
                (
                    item
                    for item in existing_records
                    if (item.url and canonical_url(item.url) == url)
                    or (
                        item.product_name
                        and classification.product_name
                        and item.product_name.casefold()
                        == classification.product_name.casefold()
                    )
                ),
                None,
            )
            source_evidence = {
                "search_title": result.title,
                "snippet": result.snippet,
                "content_excerpt": (result.content or "")[:1000],
                "queries": sorted(set(source_queries)),
                "search_score": result.score,
                "classification_reasoning": classification.reasoning,
                "research_run_id": run.id,
            }
            data = {
                "cluster_id": cluster_id,
                "company_name": classification.company_name,
                "product_name": classification.product_name,
                "url": url,
                "relationship_type": classification.relationship_type,
                "target_customer": classification.target_customer,
                "problem_solved": classification.problem_solved,
                "description": result.snippet or result.content,
                "features": classification.features,
                "pricing_position": classification.pricing_position,
                "similarity_score": classification.similarity_score,
                "strengths": classification.strengths,
                "weaknesses": classification.weaknesses,
                "possible_gap": classification.possible_gap,
                "classification_confidence": classification.confidence,
                "source_evidence": source_evidence,
            }
            if existing is None:
                stored = self.competitors.create(**data)
                existing_records.append(stored)
            else:
                user_corrected = (existing.source_evidence or {}).get(
                    "user_corrected_relationship", False
                )
                corrected_type = existing.relationship_type
                for key, value in data.items():
                    if key not in {"cluster_id", "relationship_type"}:
                        setattr(existing, key, value)
                existing.relationship_type = (
                    corrected_type
                    if user_corrected
                    else classification.relationship_type
                )
                if user_corrected:
                    existing.source_evidence["user_corrected_relationship"] = True
                stored = self.competitors.save(existing)
            persisted.append(stored)

        if failed_queries < len(query_records):
            self.competitors.delete_stale_for_cluster(
                cluster_id,
                keep_ids={item.id for item in persisted},
            )
            self.session.expire_all()

        run_error = permanent_error
        finished_run = self.research.finish_run(
            run,
            query_count=len(query_records),
            result_count=total_results,
            relevant_competitor_count=len(persisted),
            failed_query_count=failed_queries,
            error_message=run_error,
        )
        if failed_queries < len(query_records):
            self._report_progress(
                progress_callback, 0.93, "Recomputing researched opportunity scores"
            )
            cluster.status = "researched"
            self.clusters.save(cluster)
            score = OpportunityScorer(self.session).score_cluster(cluster_id)
        else:
            score = None
        if permanent_error:
            raise SearchAuthenticationError(permanent_error)
        self._report_progress(progress_callback, 1.0, "Competitor research complete")
        return ResearchOutcome(
            run=finished_run,
            queries=query_records,
            competitors=persisted,
            irrelevant_result_count=irrelevant_count,
            score=score,
        )

    @staticmethod
    def _report_progress(
        callback: Optional[ResearchProgressCallback],
        progress: float,
        message: str,
    ) -> None:
        if callback is not None:
            callback(min(1.0, max(0.0, progress)), message)


def build_research_service(session: Session, settings: Settings) -> ResearchService:
    """Build configured research dependencies."""

    return ResearchService(
        session,
        build_search_provider(settings),
        build_competitor_classifier(settings),
        max_results=settings.max_search_results,
        search_depth=settings.search_depth,
    )
