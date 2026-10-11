"""Add governed enterprise model release onboarding and auto-Shadow intent."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0077_enterprise_model_release_onboarding"
down_revision: str | None = "0076_governed_device_family_profiles"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "enterprise_model_release_onboardings",
        sa.Column("onboarding_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "release_id",
            sa.String(length=128),
            sa.ForeignKey("model_releases.release_id"),
            nullable=False,
        ),
        sa.Column(
            "baseline_release_id",
            sa.String(length=128),
            sa.ForeignKey("model_releases.release_id"),
            nullable=False,
        ),
        sa.Column("component", sa.String(length=16), nullable=False),
        sa.Column(
            "candidate_experiment_id",
            sa.String(length=128),
            sa.ForeignKey("training_experiments.experiment_id"),
            nullable=False,
        ),
        sa.Column(
            "evaluation_id",
            sa.String(length=128),
            sa.ForeignKey("model_evaluation_runs.evaluation_id"),
            nullable=False,
        ),
        sa.Column("evidence_chain_sha256", sa.String(length=64), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("auto_shadow_enabled", sa.Boolean(), nullable=False),
        sa.Column("deployment_plan_json", sa.JSON(), nullable=False),
        sa.Column("deployment_plan_hash", sa.String(length=64), nullable=False),
        sa.Column("automation_status", sa.String(length=32), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.String(length=255), nullable=True),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("requested_by_subject_id", sa.String(length=128), nullable=False),
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
            "release_id",
            name="uq_enterprise_release_onboarding_release",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "requested_by_subject_id",
            "idempotency_key",
            name="uq_enterprise_release_onboarding_idempotency",
        ),
        sa.CheckConstraint(
            "component IN ('LLM', 'VLM', 'RUL')",
            name="ck_enterprise_release_onboarding_component",
        ),
        sa.CheckConstraint(
            "automation_status IN "
            "('WAITING_APPROVAL', 'SHADOW_REQUESTED', 'CANCELLED', 'BLOCKED', 'DISABLED')",
            name="ck_enterprise_release_onboarding_status",
        ),
        sa.CheckConstraint(
            "attempt_count >= 0 AND version >= 1",
            name="ck_enterprise_release_onboarding_versions",
        ),
    )
    op.create_index(
        "ix_enterprise_model_release_onboardings_tenant_id",
        "enterprise_model_release_onboardings",
        ["tenant_id"],
    )
    op.create_index(
        "ix_enterprise_release_onboarding_status",
        "enterprise_model_release_onboardings",
        ["tenant_id", "automation_status", "auto_shadow_enabled"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    table = "enterprise_model_release_onboardings"
    policy_name = f"{table}_tenant_isolation"[:63]
    op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY {policy_name} ON {table} "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index(
        "ix_enterprise_release_onboarding_status",
        table_name="enterprise_model_release_onboardings",
    )
    op.drop_index(
        "ix_enterprise_model_release_onboardings_tenant_id",
        table_name="enterprise_model_release_onboardings",
    )
    op.drop_table("enterprise_model_release_onboardings")
