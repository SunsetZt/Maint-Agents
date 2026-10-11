"""Add OPS-COST-001 versioned cost policy and inference correlation fields."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0028_ops_cost_attribution"
down_revision: str | None = "0027_iam005_tenant_administration"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for name in (
        "agent_run_id",
        "diagnosis_run_id",
        "prompt_bundle_id",
        "index_release_id",
    ):
        op.add_column(
            "model_inference_requests",
            sa.Column(name, sa.String(length=128), nullable=True),
        )
    op.create_index(
        "ix_model_inference_cost_window",
        "model_inference_requests",
        ["tenant_id", "created_at", "resolved_release_id"],
    )
    op.create_index(
        "ix_model_inference_diagnosis",
        "model_inference_requests",
        ["tenant_id", "diagnosis_run_id"],
    )

    op.create_table(
        "cost_policies",
        sa.Column("policy_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("effective_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("prompt_tokens_per_million_usd", sa.Float(), nullable=False),
        sa.Column("completion_tokens_per_million_usd", sa.Float(), nullable=False),
        sa.Column("gpu_hourly_cost_usd", sa.Float(), nullable=False),
        sa.Column("monthly_budget_usd", sa.Float(), nullable=False),
        sa.Column("scenario_budgets_json", sa.JSON(), nullable=False),
        sa.Column("warning_ratio", sa.Float(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("updated_by_subject_id", sa.String(length=128), nullable=False),
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
        sa.UniqueConstraint("tenant_id", "version", name="uq_cost_policy_tenant_version"),
    )
    op.create_index("ix_cost_policies_tenant_id", "cost_policies", ["tenant_id"])
    op.create_index(
        "ix_cost_policy_effective",
        "cost_policies",
        ["tenant_id", "effective_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE cost_policies ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE cost_policies FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY cost_policies_tenant_isolation ON cost_policies "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index("ix_cost_policy_effective", table_name="cost_policies")
    op.drop_index("ix_cost_policies_tenant_id", table_name="cost_policies")
    op.drop_table("cost_policies")
    op.drop_index(
        "ix_model_inference_diagnosis",
        table_name="model_inference_requests",
    )
    op.drop_index(
        "ix_model_inference_cost_window",
        table_name="model_inference_requests",
    )
    for name in (
        "index_release_id",
        "prompt_bundle_id",
        "diagnosis_run_id",
        "agent_run_id",
    ):
        op.drop_column("model_inference_requests", name)
