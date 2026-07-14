"""Recompute scores for every opportunity cluster."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.database.repositories import ClusterRepository
from src.database.session import create_database_engine, create_session_factory
from src.scoring.opportunity_score import OpportunityScorer


def main() -> None:
    """Score all clusters and print their current rankings."""

    settings = get_settings()
    SessionFactory = create_session_factory(create_database_engine(settings))
    with SessionFactory() as session:
        cluster_ids = [cluster.id for cluster in ClusterRepository(session).list(limit=10000)]
        scorer = OpportunityScorer(session)
        scores = [scorer.score_cluster(cluster_id) for cluster_id in cluster_ids]
    for score in sorted(scores, key=lambda item: item.opportunity_score, reverse=True):
        print(
            f"{score.cluster_id}: opportunity={score.opportunity_score:.1f}, "
            f"confidence={score.confidence_score:.1f}"
        )
    print(f"Scored {len(scores)} opportunity cluster(s).")


if __name__ == "__main__":
    main()
