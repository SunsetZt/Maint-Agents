"""Add reproducible training experiment and deterministic evaluation governance."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_m4_training_evaluation"
down_revision: str | None = "0003_m3_data_feedback"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> tuple[sa.Column[sa.DateTime], sa.Column[sa.DateTime]]:
    return (
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
    )


def _tenant_id() -> sa.Column[str]:
    return sa.Column(
        "tenant_id",
        sa.String(length=64),
        sa.ForeignKey("tenants.id"),
        nullable=False,
    )


def _install_rls(table_name: str) -> None:
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute(f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {table_name} FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY {table_name}_tenant_isolation ON {table_name} "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def upgrade() -> None:
    op.create_table(
        "training_experiments",
        sa.Column("experiment_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("comparison_group_id", sa.String(length=128), nullable=False),
        sa.Column("method", sa.String(length=32), nullable=False),
        sa.Column("task_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column(
            "dataset_snapshot_id",
            sa.String(length=128),
            sa.ForeignKey("dataset_snapshots.snapshot_id"),
            nullable=False,
        ),
        sa.Column("dataset_manifest_hash", sa.String(length=128), nullable=False),
        sa.Column("base_model_id", sa.String(length=255), nullable=False),
        sa.Column("base_model_digest", sa.String(length=128), nullable=False),
        sa.Column("tokenizer_digest", sa.String(length=128), nullable=False),
        sa.Column("chat_template_digest", sa.String(length=128), nullable=False),
        sa.Column("git_commit", sa.String(length=128), nullable=False),
        sa.Column("container_digest", sa.String(length=255), nullable=False),
        sa.Column("training_config", sa.JSON(), nullable=False),
        sa.Column("config_hash", sa.String(length=128), nullable=False),
        sa.Column("distributed_profile", sa.JSON(), nullable=False),
        sa.Column("random_seeds", sa.JSON(), nullable=False),
        sa.Column("hardware_topology", sa.JSON(), nullable=False),
        sa.Column("mlflow_experiment_name", sa.String(length=255), nullable=False),
        sa.Column("mlflow_run_id", sa.String(length=128), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("cost_summary", sa.JSON(), nullable=False),
        sa.Column("license_status", sa.String(length=32), nullable=False),
        sa.Column("created_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_reason", sa.String(length=255), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id", "idempotency_key", name="uq_training_experiment_idempotency"
        ),
        sa.UniqueConstraint("tenant_id", "mlflow_run_id", name="uq_training_experiment_mlflow_run"),
    )
    op.create_index("ix_training_experiments_tenant_id", "training_experiments", ["tenant_id"])
    op.create_index(
        "ix_training_experiments_group_status",
        "training_experiments",
        ["tenant_id", "comparison_group_id", "status"],
    )

    op.create_table(
        "training_artifacts",
        sa.Column("artifact_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "experiment_id",
            sa.String(length=128),
            sa.ForeignKey("training_experiments.experiment_id"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("object_key", sa.String(length=1024), nullable=False),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id", "experiment_id", "object_key", name="uq_training_artifact_key"
        ),
    )
    op.create_index("ix_training_artifacts_tenant_id", "training_artifacts", ["tenant_id"])

    op.create_table(
        "evaluation_suites",
        sa.Column("suite_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column(
            "source_snapshot_id",
            sa.String(length=128),
            sa.ForeignKey("dataset_snapshots.snapshot_id"),
            nullable=False,
        ),
        sa.Column("tier", sa.String(length=32), nullable=False),
        sa.Column("purpose", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("manifest_hash", sa.String(length=128), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("slice_counts", sa.JSON(), nullable=False),
        sa.Column("created_by_subject_id", sa.String(length=128), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id", "name", "version", name="uq_evaluation_suite_name_version"
        ),
    )
    op.create_index("ix_evaluation_suites_tenant_id", "evaluation_suites", ["tenant_id"])

    op.create_table(
        "evaluation_policies",
        sa.Column("policy_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("primary_metric", sa.String(length=128), nullable=False),
        sa.Column("hard_gates", sa.JSON(), nullable=False),
        sa.Column("thresholds", sa.JSON(), nullable=False),
        sa.Column("policy_hash", sa.String(length=128), nullable=False),
        sa.Column("created_by_subject_id", sa.String(length=128), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id", "name", "version", name="uq_evaluation_policy_name_version"
        ),
    )
    op.create_index("ix_evaluation_policies_tenant_id", "evaluation_policies", ["tenant_id"])

    op.create_table(
        "model_evaluation_runs",
        sa.Column("evaluation_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column(
            "candidate_experiment_id",
            sa.String(length=128),
            sa.ForeignKey("training_experiments.experiment_id"),
            nullable=False,
        ),
        sa.Column(
            "baseline_experiment_id",
            sa.String(length=128),
            sa.ForeignKey("training_experiments.experiment_id"),
            nullable=False,
        ),
        sa.Column(
            "suite_id",
            sa.String(length=128),
            sa.ForeignKey("evaluation_suites.suite_id"),
            nullable=False,
        ),
        sa.Column(
            "policy_id",
            sa.String(length=128),
            sa.ForeignKey("evaluation_policies.policy_id"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("primary_metric", sa.String(length=128), nullable=False),
        sa.Column("candidate_score", sa.Float(), nullable=False),
        sa.Column("baseline_score", sa.Float(), nullable=False),
        sa.Column("quality_delta", sa.Float(), nullable=False),
        sa.Column("ci_low", sa.Float(), nullable=False),
        sa.Column("ci_high", sa.Float(), nullable=False),
        sa.Column("latency_improvement", sa.Float(), nullable=False),
        sa.Column("cost_improvement", sa.Float(), nullable=False),
        sa.Column("hard_gate_results", sa.JSON(), nullable=False),
        sa.Column("slice_metrics", sa.JSON(), nullable=False),
        sa.Column("report_hash", sa.String(length=128), nullable=False),
        sa.Column("triggered_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("failure_reason", sa.String(length=255), nullable=True),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "idempotency_key", name="uq_model_evaluation_idempotency"),
    )
    op.create_index("ix_model_evaluation_runs_tenant_id", "model_evaluation_runs", ["tenant_id"])
    op.create_index(
        "ix_model_evaluation_runs_candidate",
        "model_evaluation_runs",
        ["tenant_id", "candidate_experiment_id", "completed_at"],
    )

    for table_name in (
        "training_experiments",
        "training_artifacts",
        "evaluation_suites",
        "evaluation_policies",
        "model_evaluation_runs",
    ):
        _install_rls(table_name)


def downgrade() -> None:
    op.drop_index("ix_model_evaluation_runs_candidate", table_name="model_evaluation_runs")
    op.drop_index("ix_model_evaluation_runs_tenant_id", table_name="model_evaluation_runs")
    op.drop_table("model_evaluation_runs")
    op.drop_index("ix_evaluation_policies_tenant_id", table_name="evaluation_policies")
    op.drop_table("evaluation_policies")
    op.drop_index("ix_evaluation_suites_tenant_id", table_name="evaluation_suites")
    op.drop_table("evaluation_suites")
    op.drop_index("ix_training_artifacts_tenant_id", table_name="training_artifacts")
    op.drop_table("training_artifacts")
    op.drop_index("ix_training_experiments_group_status", table_name="training_experiments")
    op.drop_index("ix_training_experiments_tenant_id", table_name="training_experiments")
    op.drop_table("training_experiments")
