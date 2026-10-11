"""Add governed AI release manifests, transitions and approval decisions."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_m5_model_release_governance"
down_revision: str | None = "0005_m4_evaluation_runner"
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
        "model_releases",
        sa.Column("release_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column(
            "evaluation_id",
            sa.String(length=128),
            sa.ForeignKey("model_evaluation_runs.evaluation_id"),
            nullable=False,
        ),
        sa.Column(
            "candidate_experiment_id",
            sa.String(length=128),
            sa.ForeignKey("training_experiments.experiment_id"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("target_environment", sa.String(length=32), nullable=False),
        sa.Column("manifest_json", sa.JSON(), nullable=False),
        sa.Column("manifest_hash", sa.String(length=128), nullable=False),
        sa.Column(
            "rollback_release_id",
            sa.String(length=128),
            sa.ForeignKey("model_releases.release_id"),
            nullable=True,
        ),
        sa.Column("traffic_percent", sa.Float(), nullable=False),
        sa.Column("failure_reason", sa.String(length=255), nullable=True),
        sa.Column("created_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id", "idempotency_key", name="uq_model_release_idempotency"
        ),
        sa.UniqueConstraint("tenant_id", "manifest_hash", name="uq_model_release_manifest"),
    )
    op.create_index("ix_model_releases_tenant_id", "model_releases", ["tenant_id"])
    op.create_index(
        "ix_model_releases_status",
        "model_releases",
        ["tenant_id", "status", "created_at"],
    )

    op.create_table(
        "model_release_transitions",
        sa.Column("transition_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "release_id",
            sa.String(length=128),
            sa.ForeignKey("model_releases.release_id"),
            nullable=False,
        ),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("from_status", sa.String(length=32), nullable=False),
        sa.Column("to_status", sa.String(length=32), nullable=False),
        sa.Column("actor_subject_id", sa.String(length=128), nullable=False),
        sa.Column("reason_code", sa.String(length=128), nullable=False),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
        sa.Column("evidence_hash", sa.String(length=128), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "release_id",
            "sequence",
            name="uq_model_release_transition_sequence",
        ),
    )
    op.create_index(
        "ix_model_release_transitions_tenant_id",
        "model_release_transitions",
        ["tenant_id"],
    )

    op.create_table(
        "model_release_approvals",
        sa.Column("approval_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "release_id",
            sa.String(length=128),
            sa.ForeignKey("model_releases.release_id"),
            nullable=False,
        ),
        sa.Column("manifest_hash", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("requested_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("decided_by_subject_id", sa.String(length=128), nullable=True),
        sa.Column("decision_reason", sa.String(length=1024), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "release_id", name="uq_model_release_approval"),
    )
    op.create_index(
        "ix_model_release_approvals_tenant_id",
        "model_release_approvals",
        ["tenant_id"],
    )

    for table in (
        "model_releases",
        "model_release_transitions",
        "model_release_approvals",
    ):
        _install_rls(table)


def downgrade() -> None:
    op.drop_index("ix_model_release_approvals_tenant_id", table_name="model_release_approvals")
    op.drop_table("model_release_approvals")
    op.drop_index(
        "ix_model_release_transitions_tenant_id",
        table_name="model_release_transitions",
    )
    op.drop_table("model_release_transitions")
    op.drop_index("ix_model_releases_status", table_name="model_releases")
    op.drop_index("ix_model_releases_tenant_id", table_name="model_releases")
    op.drop_table("model_releases")
