"""Add governed multi-agent maintenance planning councils."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0075_multi_agent_maintenance_planning"
down_revision: str | None = "0074_rul_outcome_calibration"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "maintenance_planning_councils",
        sa.Column("council_id", sa.String(length=128), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column(
            "incident_id",
            sa.String(length=128),
            sa.ForeignKey("incidents.incident_id"),
            nullable=False,
        ),
        sa.Column(
            "diagnosis_run_id",
            sa.String(length=128),
            sa.ForeignKey("diagnosis_runs.diagnosis_run_id"),
            nullable=False,
        ),
        sa.Column(
            "asset_id", sa.String(length=128), sa.ForeignKey("assets.asset_id"), nullable=False
        ),
        sa.Column("diagnosis_version", sa.Integer(), nullable=False),
        sa.Column("diagnosis_report_digest", sa.String(length=128), nullable=False),
        sa.Column("diagnosis_manifest_digest", sa.String(length=128), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("request_hash", sa.String(length=128), nullable=False),
        sa.Column("workflow_id", sa.String(length=255), nullable=False),
        sa.Column("model_alias", sa.String(length=128), nullable=False),
        sa.Column("model_release_id", sa.String(length=128), nullable=False),
        sa.Column("model_manifest_hash", sa.String(length=128), nullable=False),
        sa.Column("prompt_bundle_id", sa.String(length=128), nullable=False),
        sa.Column("prompt_bundle_hash", sa.String(length=128), nullable=False),
        sa.Column("response_schema_version", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("stage", sa.String(length=64), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("plan_json", sa.JSON(), nullable=True),
        sa.Column("result_digest", sa.String(length=128), nullable=True),
        sa.Column("coordinator_context_hash", sa.String(length=128), nullable=True),
        sa.Column("failure_code", sa.String(length=128), nullable=True),
        sa.Column("requested_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("reviewed_by_subject_id", sa.String(length=128), nullable=True),
        sa.Column("review_decision", sa.String(length=32), nullable=True),
        sa.Column("review_reason", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "tenant_id", "diagnosis_run_id", name="uq_maintenance_council_diagnosis"
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "requested_by_subject_id",
            "idempotency_key",
            name="uq_maintenance_council_idempotency",
        ),
        sa.CheckConstraint(
            "status IN ('QUEUED', 'RUNNING', 'REVIEW_PENDING', 'ACCEPTED', 'REJECTED', 'FAILED')",
            name="ck_maintenance_council_status",
        ),
        sa.CheckConstraint(
            "attempt_count >= 0 AND version >= 1 AND diagnosis_version >= 1",
            name="ck_maintenance_council_versions",
        ),
    )
    op.create_index(
        "ix_maintenance_planning_councils_tenant_id", "maintenance_planning_councils", ["tenant_id"]
    )
    op.create_index(
        "ix_maintenance_council_status",
        "maintenance_planning_councils",
        ["tenant_id", "status", "updated_at"],
    )

    op.create_table(
        "maintenance_planning_contributions",
        sa.Column("contribution_id", sa.String(length=128), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column(
            "council_id",
            sa.String(length=128),
            sa.ForeignKey("maintenance_planning_councils.council_id"),
            nullable=False,
        ),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("agent_role", sa.String(length=32), nullable=False),
        sa.Column("inference_request_id", sa.String(length=128), nullable=False),
        sa.Column("input_digest", sa.String(length=128), nullable=False),
        sa.Column("output_json", sa.JSON(), nullable=False),
        sa.Column("output_digest", sa.String(length=128), nullable=False),
        sa.Column("model_release_id", sa.String(length=128), nullable=False),
        sa.Column("model_manifest_hash", sa.String(length=128), nullable=False),
        sa.Column("context_hash", sa.String(length=128), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "council_id",
            "attempt_number",
            "agent_role",
            name="uq_maintenance_contribution_role_attempt",
        ),
        sa.UniqueConstraint(
            "tenant_id", "inference_request_id", name="uq_maintenance_contribution_inference"
        ),
        sa.CheckConstraint(
            "agent_role IN ('SAFETY', 'PARTS', 'DISPATCH', 'COORDINATOR')",
            name="ck_maintenance_contribution_role",
        ),
        sa.CheckConstraint("attempt_number >= 1", name="ck_maintenance_contribution_attempt"),
    )
    op.create_index(
        "ix_maintenance_planning_contributions_tenant_id",
        "maintenance_planning_contributions",
        ["tenant_id"],
    )
    op.create_index(
        "ix_maintenance_contribution_council",
        "maintenance_planning_contributions",
        ["tenant_id", "council_id", "attempt_number"],
    )

    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table in ("maintenance_planning_councils", "maintenance_planning_contributions"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    op.drop_index(
        "ix_maintenance_contribution_council", table_name="maintenance_planning_contributions"
    )
    op.drop_index(
        "ix_maintenance_planning_contributions_tenant_id",
        table_name="maintenance_planning_contributions",
    )
    op.drop_table("maintenance_planning_contributions")
    op.drop_index("ix_maintenance_council_status", table_name="maintenance_planning_councils")
    op.drop_index(
        "ix_maintenance_planning_councils_tenant_id", table_name="maintenance_planning_councils"
    )
    op.drop_table("maintenance_planning_councils")
