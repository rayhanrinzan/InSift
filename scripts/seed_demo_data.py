"""Seed deterministic demo data for local exploration."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import get_settings
from src.database.models import (
    Competitor,
    EvidenceItem,
    OpportunityCluster,
    OpportunityScore,
    RelationshipType,
)
from src.database.repositories import (
    ClusterRepository,
    CompetitorRepository,
    EvidenceRepository,
    ResearchRepository,
    ScoreRepository,
)
from src.database.session import create_database_engine, create_session_factory, initialize_database
from src.logging_config import log_event, setup_logging
from src.scoring.opportunity_score import OpportunityScorer
from src.research.competitor_classifier import CompetitorClassifier
from src.research.competitor_search import MockSearchProvider
from src.services.research_service import ResearchService


logger = logging.getLogger(__name__)


DEMO_EVIDENCE: list[dict[str, Any]] = [
    {
        "platform": "manual",
        "source_external_id": "demo-clinic-followup-1",
        "source_author": "ops_manager_17",
        "community": "r/clinicops",
        "title": "Patient intake follow-up still lives in spreadsheets",
        "raw_text": (
            "We still use Excel to track patient intake follow-ups. It takes hours "
            "every week and it is painful when a referral gets missed."
        ),
        "contains_problem": True,
        "extraction_confidence": 0.92,
        "problem_statement": (
            "Small clinics manually track patient intake follow-ups in spreadsheets."
        ),
        "affected_user": "clinic operations managers",
        "current_workaround": "Excel sheets and calendar reminders",
        "pain_types": ["time", "coordination", "risk", "repetitive_work"],
        "severity_score": 0.82,
        "frequency_signal": 0.74,
        "willingness_to_pay_score": 0.48,
        "metadata_json": {"demo": True, "evidence_quote": "It takes hours every week"},
    },
    {
        "platform": "manual",
        "source_external_id": "demo-clinic-followup-2",
        "source_author": "frontdesk_lead",
        "community": "r/healthcareadmin",
        "title": "Referral follow-up is too manual",
        "raw_text": (
            "Does anyone know a tool for referral follow-up? We copy notes between "
            "forms, spreadsheets, and the EHR. It is tedious and easy to miss people."
        ),
        "contains_problem": True,
        "extraction_confidence": 0.89,
        "problem_statement": (
            "Clinic teams copy referral follow-up details across forms, spreadsheets, "
            "and EHR notes."
        ),
        "affected_user": "clinic front-desk teams",
        "current_workaround": "manual copy-paste between forms, spreadsheets, and EHR",
        "pain_types": ["data_entry", "coordination", "poor_user_experience"],
        "severity_score": 0.76,
        "frequency_signal": 0.68,
        "willingness_to_pay_score": 0.42,
        "metadata_json": {"demo": True, "evidence_quote": "Does anyone know a tool"},
    },
    {
        "platform": "manual",
        "source_external_id": "demo-vendor-renewals-1",
        "source_author": "agency_finance",
        "community": "r/smallbusiness",
        "title": "Vendor renewal tracking is a mess",
        "raw_text": (
            "We lose money when vendor renewals sneak up on us. The spreadsheet is "
            "always stale and reminders live in three different inboxes."
        ),
        "contains_problem": True,
        "extraction_confidence": 0.87,
        "problem_statement": (
            "Small businesses miss vendor renewals because tracking is split across "
            "stale spreadsheets and inbox reminders."
        ),
        "affected_user": "small business finance leads",
        "current_workaround": "spreadsheet plus inbox reminders",
        "pain_types": ["lost_revenue", "lack_of_visibility", "coordination"],
        "severity_score": 0.79,
        "frequency_signal": 0.52,
        "willingness_to_pay_score": 0.62,
        "metadata_json": {"demo": True, "evidence_quote": "We lose money"},
    },
]


def _get_cluster_by_title(session: Session, title: str) -> Optional[OpportunityCluster]:
    return session.execute(
        select(OpportunityCluster).where(OpportunityCluster.title == title)
    ).scalars().first()


def _get_competitor_by_url(
    session: Session, cluster_id: str, url: str
) -> Optional[Competitor]:
    return session.execute(
        select(Competitor).where(
            Competitor.cluster_id == cluster_id,
            Competitor.url == url,
        )
    ).scalars().first()


def _get_demo_score(session: Session, cluster_id: str) -> Optional[OpportunityScore]:
    return session.execute(
        select(OpportunityScore).where(
            OpportunityScore.cluster_id == cluster_id,
            OpportunityScore.scoring_version == "demo-v1",
        )
    ).scalars().first()


def seed(session: Session) -> None:
    """Insert deterministic demo evidence, clusters, competitors, and scores."""

    evidence_repo = EvidenceRepository(session)
    cluster_repo = ClusterRepository(session)
    competitor_repo = CompetitorRepository(session)
    score_repo = ScoreRepository(session)

    evidence_items: list[EvidenceItem] = []
    for payload in DEMO_EVIDENCE:
        existing = evidence_repo.find_by_source(
            source_external_id=payload["source_external_id"]
        )
        evidence_items.append(existing or evidence_repo.create(**payload))

    clinic_cluster = _get_cluster_by_title(
        session, "Clinic intake follow-up automation"
    ) or cluster_repo.create(
        title="Clinic intake follow-up automation",
        problem_summary=(
            "Small clinic teams manually coordinate intake and referral follow-up "
            "across spreadsheets, forms, inboxes, and EHR notes."
        ),
        target_customer="small clinic operations teams",
        current_workaround="Excel, calendar reminders, inbox notes, and EHR copy-paste",
        proposed_solution=(
            "A lightweight follow-up queue that reconciles form submissions, owners, "
            "due dates, and status changes without replacing the EHR."
        ),
        status="researched",
    )
    vendor_cluster = _get_cluster_by_title(
        session, "Vendor renewal visibility for small businesses"
    ) or cluster_repo.create(
        title="Vendor renewal visibility for small businesses",
        problem_summary=(
            "Small businesses miss or overpay vendor renewals because contract dates "
            "and reminders are scattered across spreadsheets and inboxes."
        ),
        target_customer="small business finance leads",
        current_workaround="stale spreadsheets and inbox reminders",
        proposed_solution=(
            "A renewal tracker that centralizes vendor dates, owners, reminders, "
            "and spend notes."
        ),
        status="new",
    )

    cluster_repo.link_evidence(clinic_cluster.id, evidence_items[0].id, 0.93)
    cluster_repo.link_evidence(clinic_cluster.id, evidence_items[1].id, 0.87)
    cluster_repo.link_evidence(vendor_cluster.id, evidence_items[2].id, 0.91)

    competitors = [
        {
            "cluster_id": clinic_cluster.id,
            "company_name": "FormDr",
            "product_name": "FormDr",
            "url": "https://www.formdr.com/",
            "relationship_type": RelationshipType.ADJACENT.value,
            "target_customer": "healthcare practices",
            "problem_solved": "digital patient intake forms and workflows",
            "description": "Digital intake and form automation for medical offices.",
            "features": ["intake forms", "workflow automation", "patient communication"],
            "pricing_position": "paid SaaS",
            "similarity_score": 0.63,
            "strengths": ["healthcare-specific", "form workflow coverage"],
            "weaknesses": ["not focused on referral follow-up visibility"],
            "possible_gap": "Follow-up accountability after intake appears less central.",
            "classification_confidence": 0.76,
            "source_evidence": {"demo": True, "query": "clinic referral follow up software"},
        },
        {
            "cluster_id": clinic_cluster.id,
            "company_name": "Airtable",
            "product_name": "Airtable",
            "url": "https://www.airtable.com/",
            "relationship_type": RelationshipType.SUBSTITUTE.value,
            "target_customer": "general operations teams",
            "problem_solved": "flexible spreadsheet-database workflows",
            "description": "General-purpose database/spreadsheet tooling.",
            "features": ["tables", "views", "automations"],
            "pricing_position": "freemium to paid SaaS",
            "similarity_score": 0.44,
            "strengths": ["flexible", "fast to customize"],
            "weaknesses": ["not purpose-built for healthcare follow-up workflows"],
            "possible_gap": "Teams still need to design their own clinical process.",
            "classification_confidence": 0.81,
            "source_evidence": {"demo": True, "query": "spreadsheet patient follow up"},
        },
    ]
    for competitor in competitors:
        if competitor["url"] and _get_competitor_by_url(
            session, competitor["cluster_id"], competitor["url"]
        ):
            continue
        competitor_repo.create(**competitor)

    if not _get_demo_score(session, clinic_cluster.id):
        score_repo.create(
            cluster_id=clinic_cluster.id,
            pain_severity_score=80.0,
            problem_frequency_score=71.0,
            willingness_to_pay_score=45.0,
            evidence_quality_score=76.0,
            whitespace_score=64.0,
            build_feasibility_score=72.0,
            market_accessibility_score=58.0,
            opportunity_score=68.0,
            confidence_score=54.0,
            scoring_version="demo-v1",
            explanation_json={
                "pain_severity": {
                    "score": 80,
                    "reason": "Users describe hours of recurring manual work and missed referrals.",
                },
                "problem_frequency": {
                    "score": 71,
                    "reason": "The demo evidence spans two independent authors and communities.",
                },
                "white_space": {
                    "score": 64,
                    "reason": (
                        "Adjacent products exist, but the stored evidence points to a "
                        "specific follow-up accountability gap."
                    ),
                },
                "confidence": {
                    "score": 54,
                    "reason": "Evidence is directionally useful but intentionally limited in demo mode.",
                },
            },
        )

    research_repo = ResearchRepository(session)
    research_service = ResearchService(
        session,
        MockSearchProvider(),
        CompetitorClassifier(),
        max_results=10,
    )
    scorer = OpportunityScorer(session)
    for cluster in (clinic_cluster, vendor_cluster):
        if research_repo.successful_query_count(cluster.id) == 0:
            research_service.research_cluster(cluster.id)
            continue
        latest = ScoreRepository(session).latest_for_cluster(cluster.id)
        if latest is None or latest.scoring_version != "phase6-v1":
            scorer.score_cluster(cluster.id)


def main() -> None:
    """Run the demo seed."""

    settings = get_settings()
    setup_logging(settings)
    engine = create_database_engine(settings)
    initialize_database(engine)
    SessionFactory = create_session_factory(engine)
    with SessionFactory() as session:
        seed(session)
    log_event(logger, logging.INFO, "demo_data_seeded", {"demo_mode": settings.demo_mode})
    print("Seeded demo data.")


if __name__ == "__main__":
    main()
