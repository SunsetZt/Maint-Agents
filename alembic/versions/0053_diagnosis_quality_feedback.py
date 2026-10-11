"""Add immutable diagnosis quality feedback facts."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0053_diagnosis_quality_feedback"
down_revision: str | None = "0052_enterprise_purchase_requisition_delivery"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "diagnosis_quality_feedback",
        sa.Column("feedback_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("request_digest", sa.String(128), nullable=False),
        sa.Column("diagnosis_run_id", sa.String(128), nullable=False),
        sa.Column("incident_id", sa.String(128), nullable=False),
        sa.Column("asset_id", sa.String(128), nullable=False),
        sa.Column("agent_run_id", sa.String(128), nullable=False),
        sa.Column("diagnosis_status", sa.String(32), nullable=False),
        sa.Column("diagnosis_version", sa.Integer(), nullable=False),
        sa.Column("report_digest", sa.String(128), nullable=False),
        sa.Column("manifest_digest", sa.String(128), nullable=False),
        sa.Column("model_release_id", sa.String(128), nullable=True),
        sa.Column("prompt_bundle_id", sa.String(128), nullable=True),
        sa.Column("index_release_id", sa.String(128), nullable=True),
        sa.Column("context_hash", sa.String(128), nullable=True),
        sa.Column("verdict", sa.String(32), nullable=False),
        sa.Column("issue_codes", sa.JSON(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("content_digest", sa.String(128), nullable=False),
        sa.Column("submitted_by_subject_id", sa.String(128), nullable=False),
        sa.Column("submitted_by_role", sa.String(64), nullable=False),
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
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(
            ["diagnosis_run_id"], ["diagnosis_runs.diagnosis_run_id"]
        ),
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.incident_id"]),
        sa.ForeignKeyConstraint(["asset_id"], ["assets.asset_id"]),
        sa.UniqueConstraint(
            "tenant_id",
            "diagnosis_run_id",
            "submitted_by_subject_id",
            name="uq_diagnosis_feedback_subject_run",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "submitted_by_subject_id",
            "idempotency_key",
            name="uq_diagnosis_feedback_subject_idempotency",
        ),
        sa.CheckConstraint(
            "verdict IN ('HELPFUL', 'PARTIALLY_HELPFUL', 'NOT_HELPFUL', 'UNSAFE')",
            name="ck_diagnosis_feedback_verdict",
        ),
        sa.CheckConstraint(
            "diagnosis_status IN ('COMPLETED', 'NEEDS_INFORMATION')",
            name="ck_diagnosis_feedback_terminal_status",
        ),
    )
    op.create_index(
        "ix_diagnosis_quality_feedback_tenant_id",
        "diagnosis_quality_feedback",
        ["tenant_id"],
    )
    op.create_index(
        "ix_diagnosis_feedback_incident_created",
        "diagnosis_quality_feedback",
        ["tenant_id", "incident_id", "created_at", "feedback_id"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE diagnosis_quality_feedback ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE diagnosis_quality_feedback FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY diagnosis_quality_feedback_tenant_isolation "
        "ON diagnosis_quality_feedback "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index(
        "ix_diagnosis_feedback_incident_created",
        table_name="diagnosis_quality_feedback",
    )
    op.drop_index(
        "ix_diagnosis_quality_feedback_tenant_id",
        table_name="diagnosis_quality_feedback",
    )
    op.drop_table("diagnosis_quality_feedback")
