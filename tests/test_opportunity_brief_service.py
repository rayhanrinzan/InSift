"""Tests for actionable problem, product, and competition briefs."""

from sqlalchemy.orm import Session

from src.database.repositories import (
    ClusterRepository,
    CompetitorRepository,
    EvidenceRepository,
)
from src.services.opportunity_brief_service import build_opportunity_brief


def _accounting_cluster(session: Session):
    evidence = EvidenceRepository(session).create(
        platform="web",
        source_url="https://www.reddit.com/r/Accounting/comments/brief/month_end/",
        source_external_id="brief-month-end",
        raw_text=(
            "Our month-end close takes 17 days because we manually reconcile "
            "transactions in Excel."
        ),
        contains_problem=True,
        extraction_confidence=0.9,
        problem_statement=(
            "Month-end close takes 17 days because the team manually reconciles "
            "transactions in Excel."
        ),
        affected_user="small accounting firms",
        current_workaround="Excel spreadsheets",
        pain_types=["time", "labor", "data_entry", "risk"],
        severity_score=0.8,
        frequency_signal=0.7,
        willingness_to_pay_score=0.4,
    )
    cluster = ClusterRepository(session).create(
        title="Need Month End Close Software Tool",
        problem_summary=evidence.problem_statement,
        target_customer="Small accounting firms",
        current_workaround="Uses spreadsheet according to the source text",
        proposed_solution=None,
        status="candidate",
    )
    ClusterRepository(session).link_evidence(cluster.id, evidence.id, 1.0)
    return ClusterRepository(session).get(cluster.id)


def test_brief_explains_problem_and_provides_bounded_build_plan(
    db_session: Session,
) -> None:
    cluster = _accounting_cluster(db_session)
    assert cluster is not None

    brief = build_opportunity_brief(cluster)

    assert brief.workflow == "month-end close"
    assert "Small accounting firms" in brief.plain_english
    assert brief.current_workaround == "a spreadsheet"
    assert "month-end close workspace" in brief.product_hypothesis
    assert any(
        feature.name == "Import and reconciliation" for feature in brief.mvp_features
    )
    assert len(brief.build_phases) == 4
    assert "30%" in brief.success_metric
    assert brief.competition.status == "required"


def test_brief_blocks_generic_build_when_direct_competitor_exists(
    db_session: Session,
) -> None:
    cluster = _accounting_cluster(db_session)
    assert cluster is not None
    CompetitorRepository(db_session).create(
        cluster_id=cluster.id,
        company_name="CloseCo",
        product_name="CloseFlow",
        url="https://closeflow.example",
        relationship_type="direct",
        target_customer="small accounting firms",
        problem_solved="month-end close reconciliation",
        similarity_score=0.9,
        possible_gap="Existing tools require enterprise implementation.",
    )
    cluster.status = "researched"
    ClusterRepository(db_session).save(cluster)
    refreshed = ClusterRepository(db_session).get(cluster.id)
    assert refreshed is not None

    assessment = build_opportunity_brief(refreshed).competition

    assert assessment.status == "crowded"
    assert assessment.direct_count == 1
    assert "CloseFlow" in assessment.summary
    assert "Do not copy" in assessment.recommendation
    assert assessment.gaps == ("Existing tools require enterprise implementation.",)
