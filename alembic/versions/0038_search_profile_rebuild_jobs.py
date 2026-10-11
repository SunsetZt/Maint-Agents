"""Add durable Temporal jobs for OpenSearch profile rebuilds."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0038_search_profile_rebuild_jobs"
down_revision: str | None = "0037_search_profile_embedding_binding"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge_search_rebuild_jobs",
        sa.Column("rebuild_job_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("search_profile_id", sa.String(128), nullable=False),
        sa.Column("workflow_id", sa.String(255), nullable=False),
        sa.Column("expected_profile_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("progress_percent", sa.Integer(), nullable=False),
        sa.Column("completed_items", sa.Integer(), nullable=False),
        sa.Column("total_items", sa.Integer(), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("requested_by_subject_id", sa.String(128), nullable=False),
        sa.Column("failure_code", sa.String(128), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint(
            "progress_percent >= 0 AND progress_percent <= 100",
            name="ck_knowledge_search_rebuild_progress",
        ),
        sa.CheckConstraint(
            "completed_items >= 0 AND total_items >= 0",
            name="ck_knowledge_search_rebuild_items",
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(
            ["search_profile_id"], ["knowledge_search_profiles.search_profile_id"]
        ),
        sa.UniqueConstraint(
            "tenant_id", "workflow_id", name="uq_knowledge_search_rebuild_workflow"
        ),
    )
    op.create_index(
        "uq_active_knowledge_search_rebuild",
        "knowledge_search_rebuild_jobs",
        ["tenant_id", "search_profile_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('QUEUED', 'RUNNING')"),
        sqlite_where=sa.text("status IN ('QUEUED', 'RUNNING')"),
    )
    op.create_index(
        "ix_knowledge_search_rebuild_status",
        "knowledge_search_rebuild_jobs",
        ["tenant_id", "status", "updated_at"],
    )
    op.execute("ALTER TABLE knowledge_search_rebuild_jobs ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE knowledge_search_rebuild_jobs FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY knowledge_search_rebuild_jobs_tenant_isolation "
        "ON knowledge_search_rebuild_jobs "
        "USING (tenant_id = current_setting('app.tenant_id', true)) "
        "WITH CHECK (tenant_id = current_setting('app.tenant_id', true))"
    )


def downgrade() -> None:
    op.drop_table("knowledge_search_rebuild_jobs")
