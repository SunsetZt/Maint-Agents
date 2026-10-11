"""Add independent evaluation jobs and immutable case-level evidence artifacts."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_m4_evaluation_runner"
down_revision: str | None = "0004_m4_training_evaluation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> tuple[sa.Column[sa.DateTime], sa.Column[sa.DateTime]]:
    return (
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )


def _tenant_id() -> sa.Column[str]:
    return sa.Column(
        "tenant_id", sa.String(length=64), sa.ForeignKey("tenants.id"), nullable=False
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
        "model_evaluation_jobs",
        sa.Column("job_id", sa.String(length=128), primary_key=True),
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
        sa.Column("target_profile", sa.String(length=32), nullable=False),
        sa.Column("runner_git_commit", sa.String(length=128), nullable=False),
        sa.Column("container_digest", sa.String(length=255), nullable=False),
        sa.Column("runner_config", sa.JSON(), nullable=False),
        sa.Column("config_hash", sa.String(length=128), nullable=False),
        sa.Column(
            "result_evaluation_id",
            sa.String(length=128),
            sa.ForeignKey("model_evaluation_runs.evaluation_id"),
            nullable=True,
        ),
        sa.Column("created_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_reason", sa.String(length=255), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id", "idempotency_key", name="uq_model_eval_job_idempotency"
        ),
    )
    op.create_index("ix_model_evaluation_jobs_tenant_id", "model_evaluation_jobs", ["tenant_id"])
    op.create_index(
        "ix_model_evaluation_jobs_status",
        "model_evaluation_jobs",
        ["tenant_id", "status", "created_at"],
    )

    op.create_table(
        "evaluation_artifacts",
        sa.Column("artifact_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "evaluation_id",
            sa.String(length=128),
            sa.ForeignKey("model_evaluation_runs.evaluation_id"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("object_key", sa.String(length=1024), nullable=False),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id", "evaluation_id", "object_key", name="uq_evaluation_artifact_key"
        ),
    )
    op.create_index("ix_evaluation_artifacts_tenant_id", "evaluation_artifacts", ["tenant_id"])

    _install_rls("model_evaluation_jobs")
    _install_rls("evaluation_artifacts")


def downgrade() -> None:
    op.drop_index("ix_evaluation_artifacts_tenant_id", table_name="evaluation_artifacts")
    op.drop_table("evaluation_artifacts")
    op.drop_index("ix_model_evaluation_jobs_status", table_name="model_evaluation_jobs")
    op.drop_index("ix_model_evaluation_jobs_tenant_id", table_name="model_evaluation_jobs")
    op.drop_table("model_evaluation_jobs")
