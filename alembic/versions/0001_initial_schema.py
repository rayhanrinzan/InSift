"""Create initial InSift schema.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-07-14
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply the initial schema."""

    op.create_table(
        "evidence_items",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("platform", sa.String(length=80), nullable=False),
        sa.Column("source_url", sa.String(length=500), nullable=True),
        sa.Column("source_external_id", sa.String(length=255), nullable=True),
        sa.Column("source_author", sa.String(length=255), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("collected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("community", sa.String(length=255), nullable=True),
        sa.Column("title", sa.String(length=500), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("engagement_score", sa.Float(), nullable=False),
        sa.Column("contains_problem", sa.Boolean(), nullable=False),
        sa.Column("extraction_confidence", sa.Float(), nullable=False),
        sa.Column("problem_statement", sa.Text(), nullable=True),
        sa.Column("affected_user", sa.String(length=255), nullable=True),
        sa.Column("current_workaround", sa.Text(), nullable=True),
        sa.Column("pain_types", sa.JSON(), nullable=False),
        sa.Column("severity_score", sa.Float(), nullable=False),
        sa.Column("frequency_signal", sa.Float(), nullable=False),
        sa.Column("willingness_to_pay_score", sa.Float(), nullable=False),
        sa.Column("embedding", sa.JSON(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_evidence_items_contains_problem",
        "evidence_items",
        ["contains_problem"],
    )
    op.create_index(
        "ix_evidence_items_source_external_id",
        "evidence_items",
        ["source_external_id"],
    )
    op.create_index("ix_evidence_items_source_url", "evidence_items", ["source_url"])

    op.create_table(
        "opportunity_clusters",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("problem_summary", sa.Text(), nullable=False),
        sa.Column("target_customer", sa.String(length=255), nullable=True),
        sa.Column("current_workaround", sa.Text(), nullable=True),
        sa.Column("proposed_solution", sa.Text(), nullable=True),
        sa.Column("evidence_count", sa.Integer(), nullable=False),
        sa.Column("independent_author_count", sa.Integer(), nullable=False),
        sa.Column("independent_source_count", sa.Integer(), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_opportunity_clusters_status",
        "opportunity_clusters",
        ["status"],
    )

    op.create_table(
        "cluster_evidence",
        sa.Column("cluster_id", sa.String(length=36), nullable=False),
        sa.Column("evidence_item_id", sa.String(length=36), nullable=False),
        sa.Column("similarity_score", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["cluster_id"], ["opportunity_clusters.id"]),
        sa.ForeignKeyConstraint(["evidence_item_id"], ["evidence_items.id"]),
        sa.PrimaryKeyConstraint("cluster_id", "evidence_item_id"),
    )

    op.create_table(
        "competitors",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("cluster_id", sa.String(length=36), nullable=False),
        sa.Column("company_name", sa.String(length=255), nullable=True),
        sa.Column("product_name", sa.String(length=255), nullable=True),
        sa.Column("url", sa.String(length=500), nullable=True),
        sa.Column("relationship_type", sa.String(length=40), nullable=False),
        sa.Column("target_customer", sa.String(length=255), nullable=True),
        sa.Column("problem_solved", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("pricing_position", sa.String(length=255), nullable=True),
        sa.Column("similarity_score", sa.Float(), nullable=False),
        sa.Column("strengths", sa.JSON(), nullable=False),
        sa.Column("weaknesses", sa.JSON(), nullable=False),
        sa.Column("possible_gap", sa.Text(), nullable=True),
        sa.Column("classification_confidence", sa.Float(), nullable=False),
        sa.Column("source_evidence", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["cluster_id"], ["opportunity_clusters.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("cluster_id", "url", name="uq_competitor_cluster_url"),
    )
    op.create_index("ix_competitors_cluster_id", "competitors", ["cluster_id"])

    op.create_table(
        "opportunity_scores",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("cluster_id", sa.String(length=36), nullable=False),
        sa.Column("pain_severity_score", sa.Float(), nullable=False),
        sa.Column("problem_frequency_score", sa.Float(), nullable=False),
        sa.Column("willingness_to_pay_score", sa.Float(), nullable=False),
        sa.Column("evidence_quality_score", sa.Float(), nullable=False),
        sa.Column("whitespace_score", sa.Float(), nullable=False),
        sa.Column("build_feasibility_score", sa.Float(), nullable=False),
        sa.Column("market_accessibility_score", sa.Float(), nullable=False),
        sa.Column("opportunity_score", sa.Float(), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False),
        sa.Column("scoring_version", sa.String(length=80), nullable=False),
        sa.Column("explanation_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["cluster_id"], ["opportunity_clusters.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_opportunity_scores_cluster_id",
        "opportunity_scores",
        ["cluster_id"],
    )

    op.create_table(
        "user_feedback",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("entity_type", sa.String(length=80), nullable=False),
        sa.Column("entity_id", sa.String(length=36), nullable=False),
        sa.Column("field_name", sa.String(length=120), nullable=False),
        sa.Column("original_value", sa.Text(), nullable=True),
        sa.Column("corrected_value", sa.Text(), nullable=True),
        sa.Column("feedback_type", sa.String(length=80), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_user_feedback_entity",
        "user_feedback",
        ["entity_type", "entity_id"],
    )


def downgrade() -> None:
    """Revert the initial schema."""

    op.drop_index("ix_user_feedback_entity", table_name="user_feedback")
    op.drop_table("user_feedback")
    op.drop_index("ix_opportunity_scores_cluster_id", table_name="opportunity_scores")
    op.drop_table("opportunity_scores")
    op.drop_index("ix_competitors_cluster_id", table_name="competitors")
    op.drop_table("competitors")
    op.drop_table("cluster_evidence")
    op.drop_index("ix_opportunity_clusters_status", table_name="opportunity_clusters")
    op.drop_table("opportunity_clusters")
    op.drop_index("ix_evidence_items_source_url", table_name="evidence_items")
    op.drop_index("ix_evidence_items_source_external_id", table_name="evidence_items")
    op.drop_index("ix_evidence_items_contains_problem", table_name="evidence_items")
    op.drop_table("evidence_items")
