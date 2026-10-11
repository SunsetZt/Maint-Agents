"""Add governed OpenSearch search profiles and A/B evaluation evidence."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0036_opensearch_search_profiles"
down_revision: str | None = "0035_graphrag_causal_evidence"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge_search_profiles",
        sa.Column("search_profile_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("source_index_release_id", sa.String(128), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("backend", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("is_active_shadow", sa.Boolean(), nullable=False),
        sa.Column("index_name", sa.String(255), nullable=False),
        sa.Column("schema_version", sa.String(64), nullable=False),
        sa.Column("manifest_hash", sa.String(128), nullable=True),
        sa.Column("document_count", sa.Integer(), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column("gate_results_json", sa.JSON(), nullable=False),
        sa.Column("failure_codes", sa.JSON(), nullable=False),
        sa.Column("created_by_subject_id", sa.String(128), nullable=False),
        sa.Column("evaluated_by_subject_id", sa.String(128), nullable=True),
        sa.Column("activated_by_subject_id", sa.String(128), nullable=True),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("state_version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(["source_index_release_id"], ["index_releases.release_id"]),
        sa.UniqueConstraint(
            "tenant_id", "name", "version", name="uq_knowledge_search_profile_version"
        ),
    )
    op.create_index(
        "uq_active_knowledge_search_profile",
        "knowledge_search_profiles",
        ["tenant_id", "name"],
        unique=True,
        postgresql_where=sa.text("is_active_shadow"),
        sqlite_where=sa.text("is_active_shadow = 1"),
    )
    op.create_table(
        "knowledge_search_projections",
        sa.Column("projection_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("search_profile_id", sa.String(128), nullable=False),
        sa.Column("citation_id", sa.String(128), nullable=False),
        sa.Column("chunk_id", sa.String(128), nullable=False),
        sa.Column("document_id", sa.String(128), nullable=False),
        sa.Column("document_version_id", sa.String(128), nullable=False),
        sa.Column("content_checksum", sa.String(128), nullable=False),
        sa.Column("source_checksum", sa.String(128), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(
            ["search_profile_id"], ["knowledge_search_profiles.search_profile_id"]
        ),
        sa.ForeignKeyConstraint(["citation_id"], ["citation_anchors.citation_id"]),
        sa.ForeignKeyConstraint(["chunk_id"], ["knowledge_chunks.chunk_id"]),
        sa.ForeignKeyConstraint(["document_id"], ["knowledge_documents.document_id"]),
        sa.ForeignKeyConstraint(
            ["document_version_id"], ["knowledge_document_versions.document_version_id"]
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "search_profile_id",
            "citation_id",
            name="uq_knowledge_search_projection_citation",
        ),
    )
    op.create_index(
        "ix_knowledge_search_projection_source",
        "knowledge_search_projections",
        ["tenant_id", "document_version_id"],
    )
    op.create_table(
        "knowledge_search_evaluations",
        sa.Column("evaluation_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("search_profile_id", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("policy_version", sa.String(128), nullable=False),
        sa.Column("cases_json", sa.JSON(), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column("gate_results_json", sa.JSON(), nullable=False),
        sa.Column("failure_codes", sa.JSON(), nullable=False),
        sa.Column("requested_by_subject_id", sa.String(128), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(
            ["search_profile_id"], ["knowledge_search_profiles.search_profile_id"]
        ),
    )
    for table in (
        "knowledge_search_profiles",
        "knowledge_search_projections",
        "knowledge_search_evaluations",
    ):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            "USING (tenant_id = current_setting('app.tenant_id', true)) "
            "WITH CHECK (tenant_id = current_setting('app.tenant_id', true))"
        )


def downgrade() -> None:
    op.drop_table("knowledge_search_evaluations")
    op.drop_table("knowledge_search_projections")
    op.drop_table("knowledge_search_profiles")
