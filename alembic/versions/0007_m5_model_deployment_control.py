"""Add KServe deployment desired state and immutable online observations."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_m5_model_deployment_control"
down_revision: str | None = "0006_m5_model_release_governance"
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
        "model_deployments",
        sa.Column("deployment_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "release_id",
            sa.String(length=128),
            sa.ForeignKey("model_releases.release_id"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("namespace", sa.String(length=128), nullable=False),
        sa.Column("service_name", sa.String(length=128), nullable=False),
        sa.Column("route_name", sa.String(length=128), nullable=False),
        sa.Column("stable_service_name", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("desired_stage", sa.String(length=32), nullable=False),
        sa.Column("current_stage", sa.String(length=32), nullable=False),
        sa.Column("desired_traffic_percent", sa.Float(), nullable=False),
        sa.Column("observed_traffic_percent", sa.Float(), nullable=False),
        sa.Column("desired_spec_json", sa.JSON(), nullable=False),
        sa.Column("desired_spec_hash", sa.String(length=128), nullable=False),
        sa.Column("applied_spec_hash", sa.String(length=128), nullable=True),
        sa.Column("provider_revision", sa.String(length=255), nullable=True),
        sa.Column("endpoint_url", sa.String(length=1024), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("failure_reason", sa.String(length=255), nullable=True),
        sa.Column("requested_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("reconciled_by_subject_id", sa.String(length=128), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "release_id", name="uq_model_deployment_release"),
    )
    op.create_index("ix_model_deployments_tenant_id", "model_deployments", ["tenant_id"])
    op.create_index(
        "ix_model_deployments_pending",
        "model_deployments",
        ["tenant_id", "status", "updated_at"],
    )

    op.create_table(
        "model_release_observations",
        sa.Column("observation_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "release_id",
            sa.String(length=128),
            sa.ForeignKey("model_releases.release_id"),
            nullable=False,
        ),
        sa.Column(
            "deployment_id",
            sa.String(length=128),
            sa.ForeignKey("model_deployments.deployment_id"),
            nullable=False,
        ),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("request_count", sa.BigInteger(), nullable=False),
        sa.Column("critical_case_count", sa.BigInteger(), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column("thresholds_json", sa.JSON(), nullable=False),
        sa.Column("source_refs_json", sa.JSON(), nullable=False),
        sa.Column("evidence_hash", sa.String(length=128), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("failure_reasons", sa.JSON(), nullable=False),
        sa.Column("collected_by_subject_id", sa.String(length=128), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "release_id",
            "stage",
            "sequence",
            name="uq_model_release_observation_sequence",
        ),
    )
    op.create_index(
        "ix_model_release_observations_tenant_id",
        "model_release_observations",
        ["tenant_id"],
    )

    for table in ("model_deployments", "model_release_observations"):
        _install_rls(table)


def downgrade() -> None:
    op.drop_index(
        "ix_model_release_observations_tenant_id",
        table_name="model_release_observations",
    )
    op.drop_table("model_release_observations")
    op.drop_index("ix_model_deployments_pending", table_name="model_deployments")
    op.drop_index("ix_model_deployments_tenant_id", table_name="model_deployments")
    op.drop_table("model_deployments")
