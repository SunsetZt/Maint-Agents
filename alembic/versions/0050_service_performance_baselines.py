"""Add governed service-performance baseline versions."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0050_service_performance_baselines"
down_revision: str | None = "0049_incident_resolution_lifecycle"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "service_performance_baselines",
        sa.Column("baseline_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("scope_type", sa.String(16), nullable=False),
        sa.Column("reference_window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reference_window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metric_contract_version", sa.String(64), nullable=False),
        sa.Column("mean_mttr_minutes", sa.Float(), nullable=False),
        sa.Column("first_time_fix_rate", sa.Float(), nullable=False),
        sa.Column("remote_resolution_rate", sa.Float(), nullable=False),
        sa.Column("closed_work_order_count", sa.Integer(), nullable=False),
        sa.Column("first_time_fix_sample_count", sa.Integer(), nullable=False),
        sa.Column("mttr_sample_count", sa.Integer(), nullable=False),
        sa.Column("resolution_sample_count", sa.Integer(), nullable=False),
        sa.Column("minimum_work_order_samples", sa.Integer(), nullable=False),
        sa.Column("minimum_resolution_samples", sa.Integer(), nullable=False),
        sa.Column("target_mttr_reduction_rate", sa.Float(), nullable=False),
        sa.Column("target_first_time_fix_lift", sa.Float(), nullable=False),
        sa.Column("target_remote_resolution_lift", sa.Float(), nullable=False),
        sa.Column("data_quality_status", sa.String(16), nullable=False),
        sa.Column("data_quality_json", sa.JSON(), nullable=False),
        sa.Column("content_digest", sa.String(71), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_by_subject_id", sa.String(128), nullable=False),
        sa.Column("activated_by_subject_id", sa.String(128), nullable=True),
        sa.Column("activation_reason", sa.Text(), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.CheckConstraint(
            "scope_type = 'TENANT'",
            name="ck_service_performance_baseline_scope",
        ),
        sa.CheckConstraint(
            "status IN ('DRAFT', 'ACTIVE', 'RETIRED')",
            name="ck_service_performance_baseline_status",
        ),
        sa.CheckConstraint(
            "data_quality_status = 'COMPLETE'",
            name="ck_service_performance_baseline_quality",
        ),
        sa.CheckConstraint(
            "minimum_work_order_samples > 0 AND minimum_resolution_samples > 0",
            name="ck_service_performance_baseline_minimum_samples",
        ),
        sa.CheckConstraint(
            "target_mttr_reduction_rate >= 0 AND target_mttr_reduction_rate <= 1 "
            "AND target_first_time_fix_lift >= 0 AND target_first_time_fix_lift <= 1 "
            "AND target_remote_resolution_lift >= 0 AND target_remote_resolution_lift <= 1",
            name="ck_service_performance_baseline_targets",
        ),
    )
    op.create_index(
        "ix_service_performance_baselines_tenant_id",
        "service_performance_baselines",
        ["tenant_id"],
    )
    op.create_index(
        "ix_service_performance_baselines_tenant_created",
        "service_performance_baselines",
        ["tenant_id", "created_at"],
    )
    op.create_index(
        "uq_service_performance_baseline_active",
        "service_performance_baselines",
        ["tenant_id"],
        unique=True,
        postgresql_where=sa.text("status = 'ACTIVE'"),
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE service_performance_baselines ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE service_performance_baselines FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY service_performance_baselines_tenant_isolation "
        "ON service_performance_baselines "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index(
        "uq_service_performance_baseline_active",
        table_name="service_performance_baselines",
    )
    op.drop_index(
        "ix_service_performance_baselines_tenant_created",
        table_name="service_performance_baselines",
    )
    op.drop_index(
        "ix_service_performance_baselines_tenant_id",
        table_name="service_performance_baselines",
    )
    op.drop_table("service_performance_baselines")
