"""Add governed activation and rollback for maintenance-planning councils."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0081_maintenance_planning_activation"
down_revision: str | None = "0080_maintenance_planning_value_evaluation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "maintenance_planning_activations",
        sa.Column("activation_id", sa.String(length=128), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column(
            "evaluation_id",
            sa.String(length=128),
            sa.ForeignKey("maintenance_planning_evaluation_runs.evaluation_id"),
            nullable=False,
        ),
        sa.Column(
            "suite_id",
            sa.String(length=128),
            sa.ForeignKey("maintenance_planning_evaluation_suites.suite_id"),
            nullable=False,
        ),
        sa.Column("target_environment", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("request_hash", sa.String(length=128), nullable=False),
        sa.Column("evaluation_decision", sa.String(length=48), nullable=False),
        sa.Column("evaluation_report_hash", sa.String(length=128), nullable=False),
        sa.Column("evaluation_policy_hash", sa.String(length=128), nullable=False),
        sa.Column("suite_manifest_hash", sa.String(length=128), nullable=False),
        sa.Column("activation_policy_hash", sa.String(length=128), nullable=False),
        sa.Column("request_reason", sa.Text(), nullable=False),
        sa.Column("requested_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("decided_by_subject_id", sa.String(length=128), nullable=True),
        sa.Column("decision", sa.String(length=16), nullable=True),
        sa.Column("decision_reason", sa.Text(), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rolled_back_by_subject_id", sa.String(length=128), nullable=True),
        sa.Column("rollback_reason", sa.Text(), nullable=True),
        sa.Column("rolled_back_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "requested_by_subject_id",
            "idempotency_key",
            name="uq_maintenance_planning_activation_idempotency",
        ),
        sa.CheckConstraint(
            "target_environment IN ('PROJECT_STAGING', 'PRODUCTION')",
            name="ck_maintenance_planning_activation_target",
        ),
        sa.CheckConstraint(
            "status IN ('PENDING_APPROVAL', 'ACTIVE', 'REJECTED', 'ROLLED_BACK')",
            name="ck_maintenance_planning_activation_status",
        ),
        sa.CheckConstraint(
            "decision IS NULL OR decision IN ('APPROVE', 'REJECT')",
            name="ck_maintenance_planning_activation_decision",
        ),
        sa.CheckConstraint(
            "version > 0",
            name="ck_maintenance_planning_activation_version",
        ),
    )
    op.create_index(
        "ix_maintenance_planning_activations_tenant_id",
        "maintenance_planning_activations",
        ["tenant_id"],
    )
    op.create_index(
        "ix_maintenance_planning_activation_status",
        "maintenance_planning_activations",
        ["tenant_id", "target_environment", "status", "updated_at"],
    )
    op.create_index(
        "uq_active_maintenance_planning_activation",
        "maintenance_planning_activations",
        ["tenant_id", "target_environment"],
        unique=True,
        postgresql_where=sa.text("status = 'ACTIVE'"),
    )

    op.add_column(
        "maintenance_planning_councils",
        sa.Column("activation_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "maintenance_planning_councils",
        sa.Column("activation_policy_hash", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "maintenance_planning_councils",
        sa.Column("activation_target_environment", sa.String(length=32), nullable=True),
    )
    op.create_foreign_key(
        "fk_maintenance_planning_council_activation",
        "maintenance_planning_councils",
        "maintenance_planning_activations",
        ["activation_id"],
        ["activation_id"],
    )
    op.create_index(
        "ix_maintenance_planning_council_activation",
        "maintenance_planning_councils",
        ["tenant_id", "activation_id"],
    )

    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE maintenance_planning_activations ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE maintenance_planning_activations FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY maintenance_planning_activations_tenant_isolation "
        "ON maintenance_planning_activations "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index(
        "ix_maintenance_planning_council_activation",
        table_name="maintenance_planning_councils",
    )
    op.drop_constraint(
        "fk_maintenance_planning_council_activation",
        "maintenance_planning_councils",
        type_="foreignkey",
    )
    op.drop_column("maintenance_planning_councils", "activation_target_environment")
    op.drop_column("maintenance_planning_councils", "activation_policy_hash")
    op.drop_column("maintenance_planning_councils", "activation_id")
    op.drop_index(
        "uq_active_maintenance_planning_activation",
        table_name="maintenance_planning_activations",
    )
    op.drop_index(
        "ix_maintenance_planning_activation_status",
        table_name="maintenance_planning_activations",
    )
    op.drop_index(
        "ix_maintenance_planning_activations_tenant_id",
        table_name="maintenance_planning_activations",
    )
    op.drop_table("maintenance_planning_activations")
