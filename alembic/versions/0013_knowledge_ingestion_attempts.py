"""Add durable knowledge-ingestion processing attempts."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013_knowledge_ingestion_attempts"
down_revision: str | None = "0012_knowledge_file_ingestion"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "knowledge_ingestion_jobs",
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.alter_column("knowledge_ingestion_jobs", "attempt_count", server_default=None)
    op.create_table(
        "knowledge_ingestion_attempts",
        sa.Column("attempt_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "ingestion_id",
            sa.String(length=128),
            sa.ForeignKey("knowledge_ingestion_jobs.ingestion_id"),
            nullable=False,
        ),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("workflow_id", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("failure_reason", sa.String(length=255), nullable=True),
        sa.Column("parser_version", sa.String(length=128), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "ingestion_id",
            "attempt_number",
            name="uq_knowledge_ingestion_attempt_number",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "workflow_id",
            name="uq_knowledge_ingestion_attempt_workflow",
        ),
    )
    op.create_index(
        "ix_knowledge_ingestion_attempts_tenant_id",
        "knowledge_ingestion_attempts",
        ["tenant_id"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE knowledge_ingestion_attempts ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE knowledge_ingestion_attempts FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY knowledge_ingestion_attempts_tenant_isolation "
        f"ON knowledge_ingestion_attempts USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_table("knowledge_ingestion_attempts")
    op.drop_column("knowledge_ingestion_jobs", "attempt_count")
