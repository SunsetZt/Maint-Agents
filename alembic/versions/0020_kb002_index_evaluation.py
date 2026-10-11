"""Add durable KB-002 index evaluation and release ownership."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0020_kb002_index_evaluation"
down_revision: str | None = "0019_kb003_data_deletion"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "index_releases",
        sa.Column("created_by_subject_id", sa.String(length=128), nullable=True),
    )
    op.create_table(
        "knowledge_index_evaluations",
        sa.Column("evaluation_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "release_id",
            sa.String(length=128),
            sa.ForeignKey("index_releases.release_id"),
            nullable=False,
        ),
        sa.Column("workflow_id", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("policy_version", sa.String(length=128), nullable=False),
        sa.Column("evaluator_version", sa.String(length=128), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column("gate_results_json", sa.JSON(), nullable=False),
        sa.Column("failure_codes", sa.JSON(), nullable=False),
        sa.Column("requested_by_subject_id", sa.String(length=128), nullable=False),
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
            name="uq_knowledge_index_evaluation_workflow",
        ),
    )
    op.create_index(
        "ix_knowledge_index_evaluations_tenant_id",
        "knowledge_index_evaluations",
        ["tenant_id"],
    )
    op.create_index(
        "ix_knowledge_index_evaluations_release",
        "knowledge_index_evaluations",
        ["tenant_id", "release_id", "created_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE knowledge_index_evaluations ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE knowledge_index_evaluations FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY knowledge_index_evaluations_tenant_isolation "
        f"ON knowledge_index_evaluations USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index(
        "ix_knowledge_index_evaluations_release",
        table_name="knowledge_index_evaluations",
    )
    op.drop_index(
        "ix_knowledge_index_evaluations_tenant_id",
        table_name="knowledge_index_evaluations",
    )
    op.drop_table("knowledge_index_evaluations")
    op.drop_column("index_releases", "created_by_subject_id")
