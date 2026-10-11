"""Add governed remaining-useful-life forecast candidates."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0073_predictive_rul_forecasts"
down_revision: str | None = "0072_governed_repair_work_order_authorization"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "rul_forecast_candidates",
        sa.Column("rul_forecast_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "alert_candidate_id",
            sa.String(length=128),
            sa.ForeignKey("alert_candidates.alert_candidate_id"),
            nullable=False,
        ),
        sa.Column(
            "asset_id",
            sa.String(length=128),
            sa.ForeignKey("assets.asset_id"),
            nullable=False,
        ),
        sa.Column(
            "window_id",
            sa.String(length=128),
            sa.ForeignKey("telemetry_windows.window_id"),
            nullable=False,
        ),
        sa.Column("feature_snapshot_id", sa.String(length=128), nullable=False),
        sa.Column("source_model_release_id", sa.String(length=128), nullable=False),
        sa.Column("forecast_model_version", sa.String(length=128), nullable=False),
        sa.Column("estimate_minutes", sa.Float(), nullable=False),
        sa.Column("lower_bound_minutes", sa.Float(), nullable=False),
        sa.Column("upper_bound_minutes", sa.Float(), nullable=False),
        sa.Column("recommended_inspection_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("historical_failure_count", sa.Integer(), nullable=False),
        sa.Column("source_outcome_ids_json", sa.JSON(), nullable=False),
        sa.Column("cohort_digest", sa.String(length=128), nullable=False),
        sa.Column("history_cutoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_alert_version", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("reviewed_by_subject_id", sa.String(length=128), nullable=True),
        sa.Column("review_reason", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
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
            "alert_candidate_id",
            name="uq_rul_forecast_alert_candidate",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "created_by_subject_id",
            "idempotency_key",
            name="uq_rul_forecast_idempotency",
        ),
        sa.CheckConstraint(
            "lower_bound_minutes >= 0 "
            "AND estimate_minutes >= lower_bound_minutes "
            "AND upper_bound_minutes >= estimate_minutes",
            name="ck_rul_forecast_bounds",
        ),
        sa.CheckConstraint(
            "historical_failure_count >= 30",
            name="ck_rul_forecast_history_count",
        ),
        sa.CheckConstraint(
            "source_alert_version >= 1 AND version >= 1",
            name="ck_rul_forecast_versions",
        ),
        sa.CheckConstraint(
            "status IN ('PENDING_REVIEW', 'ACCEPTED', 'REJECTED')",
            name="ck_rul_forecast_status",
        ),
    )
    op.create_index(
        "ix_rul_forecast_candidates_tenant_id",
        "rul_forecast_candidates",
        ["tenant_id"],
    )
    op.create_index(
        "ix_rul_forecast_status",
        "rul_forecast_candidates",
        ["tenant_id", "status", "created_at"],
    )

    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE rul_forecast_candidates ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE rul_forecast_candidates FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY rul_forecast_candidates_tenant_isolation "
        "ON rul_forecast_candidates "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index("ix_rul_forecast_status", table_name="rul_forecast_candidates")
    op.drop_index(
        "ix_rul_forecast_candidates_tenant_id",
        table_name="rul_forecast_candidates",
    )
    op.drop_table("rul_forecast_candidates")
