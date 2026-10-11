"""Add blinded net-benefit evaluation for multi-agent maintenance planning."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0080_maintenance_planning_value_evaluation"
down_revision: str | None = "0079_enterprise_asr_model_asset_imports"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "maintenance_planning_evaluation_suites",
        sa.Column("suite_id", sa.String(length=128), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("suite_version", sa.String(length=64), nullable=False),
        sa.Column("evidence_tier", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("cases_json", sa.JSON(), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("manifest_hash", sa.String(length=128), nullable=False),
        sa.Column("created_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("retired_by_subject_id", sa.String(length=128), nullable=True),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "name",
            "suite_version",
            name="uq_maintenance_planning_eval_suite_version",
        ),
        sa.CheckConstraint(
            "evidence_tier IN ('PROJECT_STAGING_GOLD', 'ENTERPRISE_GOLD')",
            name="ck_maintenance_planning_eval_suite_tier",
        ),
        sa.CheckConstraint(
            "status IN ('FROZEN', 'RETIRED')",
            name="ck_maintenance_planning_eval_suite_status",
        ),
        sa.CheckConstraint(
            "sample_count > 0 AND version > 0",
            name="ck_maintenance_planning_eval_suite_counts",
        ),
    )
    op.create_index(
        "ix_maintenance_planning_evaluation_suites_tenant_id",
        "maintenance_planning_evaluation_suites",
        ["tenant_id"],
    )

    op.create_table(
        "maintenance_planning_evaluation_runs",
        sa.Column("evaluation_id", sa.String(length=128), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column(
            "suite_id",
            sa.String(length=128),
            sa.ForeignKey("maintenance_planning_evaluation_suites.suite_id"),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("request_hash", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("policy_version", sa.String(length=128), nullable=False),
        sa.Column("policy_hash", sa.String(length=128), nullable=False),
        sa.Column("policy_json", sa.JSON(), nullable=False),
        sa.Column("pairs_json", sa.JSON(), nullable=False),
        sa.Column("required_judgments_per_case", sa.Integer(), nullable=False),
        sa.Column("aggregate_metrics_json", sa.JSON(), nullable=False),
        sa.Column("gate_results_json", sa.JSON(), nullable=False),
        sa.Column("decision", sa.String(length=48), nullable=False),
        sa.Column("report_hash", sa.String(length=128), nullable=True),
        sa.Column("failure_reason", sa.String(length=255), nullable=True),
        sa.Column("requested_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
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
            name="uq_maintenance_planning_eval_run_idempotency",
        ),
        sa.CheckConstraint(
            "status IN ('JUDGING', 'COMPLETED', 'FAILED')",
            name="ck_maintenance_planning_eval_run_status",
        ),
        sa.CheckConstraint(
            "decision IN ('PENDING', 'KEEP_SINGLE_AGENT', "
            "'PROJECT_CANDIDATE_ELIGIBLE', 'PRODUCTION_CANDIDATE_ELIGIBLE')",
            name="ck_maintenance_planning_eval_run_decision",
        ),
        sa.CheckConstraint(
            "required_judgments_per_case >= 2 AND version > 0",
            name="ck_maintenance_planning_eval_run_counts",
        ),
    )
    op.create_index(
        "ix_maintenance_planning_evaluation_runs_tenant_id",
        "maintenance_planning_evaluation_runs",
        ["tenant_id"],
    )
    op.create_index(
        "ix_maintenance_planning_eval_run_status",
        "maintenance_planning_evaluation_runs",
        ["tenant_id", "status", "updated_at"],
    )

    op.create_table(
        "maintenance_planning_blind_judgments",
        sa.Column("judgment_id", sa.String(length=128), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column(
            "evaluation_id",
            sa.String(length=128),
            sa.ForeignKey("maintenance_planning_evaluation_runs.evaluation_id"),
            nullable=False,
        ),
        sa.Column("case_id", sa.String(length=128), nullable=False),
        sa.Column("judge_subject_id", sa.String(length=128), nullable=False),
        sa.Column("blind_assignment_digest", sa.String(length=128), nullable=False),
        sa.Column("variant_a_scores_json", sa.JSON(), nullable=False),
        sa.Column("variant_b_scores_json", sa.JSON(), nullable=False),
        sa.Column("preferred_variant", sa.String(length=8), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("score_digest", sa.String(length=128), nullable=False),
        sa.Column("is_adjudication", sa.Boolean(), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "evaluation_id",
            "case_id",
            "judge_subject_id",
            name="uq_maintenance_planning_blind_judge_case",
        ),
        sa.CheckConstraint(
            "preferred_variant IN ('A', 'B', 'TIE')",
            name="ck_maintenance_planning_blind_judgment_preference",
        ),
    )
    op.create_index(
        "ix_maintenance_planning_blind_judgments_tenant_id",
        "maintenance_planning_blind_judgments",
        ["tenant_id"],
    )
    op.create_index(
        "ix_maintenance_planning_blind_judgment_case",
        "maintenance_planning_blind_judgments",
        ["tenant_id", "evaluation_id", "case_id", "submitted_at"],
    )

    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table in (
        "maintenance_planning_evaluation_suites",
        "maintenance_planning_evaluation_runs",
        "maintenance_planning_blind_judgments",
    ):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    op.drop_index(
        "ix_maintenance_planning_blind_judgment_case",
        table_name="maintenance_planning_blind_judgments",
    )
    op.drop_index(
        "ix_maintenance_planning_blind_judgments_tenant_id",
        table_name="maintenance_planning_blind_judgments",
    )
    op.drop_table("maintenance_planning_blind_judgments")
    op.drop_index(
        "ix_maintenance_planning_eval_run_status",
        table_name="maintenance_planning_evaluation_runs",
    )
    op.drop_index(
        "ix_maintenance_planning_evaluation_runs_tenant_id",
        table_name="maintenance_planning_evaluation_runs",
    )
    op.drop_table("maintenance_planning_evaluation_runs")
    op.drop_index(
        "ix_maintenance_planning_evaluation_suites_tenant_id",
        table_name="maintenance_planning_evaluation_suites",
    )
    op.drop_table("maintenance_planning_evaluation_suites")
