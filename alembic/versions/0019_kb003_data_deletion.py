"""Add the KB-003 knowledge deletion workflow audit record."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0019_kb003_data_deletion"
down_revision: str | None = "0018_m7_predictive_maintenance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge_deletion_requests",
        sa.Column("deletion_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("workflow_id", sa.String(length=255), nullable=False),
        sa.Column("document_id", sa.String(length=128), nullable=False),
        sa.Column("document_version_id", sa.String(length=128), nullable=False),
        sa.Column("trigger", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.String(length=1024), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("target_source_checksum", sa.String(length=128), nullable=False),
        sa.Column("target_content_checksum", sa.String(length=128), nullable=False),
        sa.Column("scope_json", sa.JSON(), nullable=False),
        sa.Column("impact_json", sa.JSON(), nullable=False),
        sa.Column("verification_json", sa.JSON(), nullable=False),
        sa.Column("failure_reason", sa.String(length=1024), nullable=True),
        sa.Column("requested_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
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
            "workflow_id",
            name="uq_knowledge_deletion_workflow",
        ),
    )
    op.create_index(
        "ix_knowledge_deletion_requests_tenant_id",
        "knowledge_deletion_requests",
        ["tenant_id"],
    )
    op.create_index(
        "ix_knowledge_deletion_target",
        "knowledge_deletion_requests",
        ["tenant_id", "document_version_id", "requested_at"],
    )
    op.create_index(
        "ix_knowledge_deletion_status",
        "knowledge_deletion_requests",
        ["tenant_id", "status", "requested_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE knowledge_deletion_requests ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE knowledge_deletion_requests FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY knowledge_deletion_requests_tenant_isolation "
        f"ON knowledge_deletion_requests USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index("ix_knowledge_deletion_status", table_name="knowledge_deletion_requests")
    op.drop_index("ix_knowledge_deletion_target", table_name="knowledge_deletion_requests")
    op.drop_index(
        "ix_knowledge_deletion_requests_tenant_id",
        table_name="knowledge_deletion_requests",
    )
    op.drop_table("knowledge_deletion_requests")
