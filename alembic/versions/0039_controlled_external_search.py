"""Add tenant-governed external search policies and reference evidence."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0039_controlled_external_search"
down_revision: str | None = "0038_search_profile_rebuild_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "external_search_policies",
        sa.Column("policy_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("allowed_domains", sa.JSON(), nullable=False),
        sa.Column("official_domains", sa.JSON(), nullable=False),
        sa.Column("max_results", sa.Integer(), nullable=False),
        sa.Column("updated_by_subject_id", sa.String(128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.UniqueConstraint("tenant_id", name="uq_external_search_policy_tenant"),
    )
    op.create_table(
        "external_search_queries",
        sa.Column("external_search_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("query_hash", sa.String(64), nullable=False),
        sa.Column("use_case", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("provider_request_id", sa.String(128), nullable=True),
        sa.Column("requested_by_subject_id", sa.String(128), nullable=False),
        sa.Column("result_count", sa.Integer(), nullable=False),
        sa.Column("blocked_result_count", sa.Integer(), nullable=False),
        sa.Column("usage_credits", sa.Integer(), nullable=True),
        sa.Column("failure_code", sa.String(128), nullable=True),
        sa.Column("usage_conclusion", sa.String(48), nullable=False),
        sa.Column("conclusion_reason", sa.Text(), nullable=True),
        sa.Column("concluded_by_subject_id", sa.String(128), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
    )
    op.create_index(
        "ix_external_search_query_status",
        "external_search_queries",
        ["tenant_id", "status", "updated_at"],
    )
    op.create_index(
        "ix_external_search_query_requested",
        "external_search_queries",
        ["tenant_id", "requested_by_subject_id"],
    )
    op.create_table(
        "external_search_references",
        sa.Column("external_reference_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("external_search_id", sa.String(128), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("url", sa.String(2048), nullable=False),
        sa.Column("url_hash", sa.String(64), nullable=False),
        sa.Column("domain", sa.String(255), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("relevance_score", sa.Float(), nullable=False),
        sa.Column("trust_level", sa.String(32), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("guardrail_policy_version", sa.String(128), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(
            ["external_search_id"], ["external_search_queries.external_search_id"]
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "external_search_id",
            "rank",
            name="uq_external_search_reference_rank",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "external_search_id",
            "url_hash",
            name="uq_external_search_reference_url",
        ),
    )
    for table in (
        "external_search_policies",
        "external_search_queries",
        "external_search_references",
    ):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            "USING (tenant_id = current_setting('app.tenant_id', true)) "
            "WITH CHECK (tenant_id = current_setting('app.tenant_id', true))"
        )


def downgrade() -> None:
    op.drop_table("external_search_references")
    op.drop_table("external_search_queries")
    op.drop_table("external_search_policies")
