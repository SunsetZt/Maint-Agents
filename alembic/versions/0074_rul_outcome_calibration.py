"""Persist RUL outcome calibration and retraining recommendation evidence."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0074_rul_outcome_calibration"
down_revision: str | None = "0073_predictive_rul_forecasts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "rul_forecast_calibrations",
        sa.Column("calibration_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "rul_forecast_id",
            sa.String(length=128),
            sa.ForeignKey("rul_forecast_candidates.rul_forecast_id"),
            nullable=False,
        ),
        sa.Column(
            "outcome_id",
            sa.String(length=128),
            sa.ForeignKey("predictive_outcomes.outcome_id"),
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
        sa.Column("source_model_release_id", sa.String(length=128), nullable=False),
        sa.Column("forecast_model_version", sa.String(length=128), nullable=False),
        sa.Column("forecast_review_status", sa.String(length=32), nullable=False),
        sa.Column("actual_minutes", sa.Float(), nullable=False),
        sa.Column("absolute_error_minutes", sa.Float(), nullable=False),
        sa.Column("interval_covered", sa.Boolean(), nullable=False),
        sa.Column("calibration_status", sa.String(length=32), nullable=False),
        sa.Column("retraining_recommended", sa.Boolean(), nullable=False),
        sa.Column("decision_reasons_json", sa.JSON(), nullable=False),
        sa.Column("evidence_digest", sa.String(length=128), nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
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
            "rul_forecast_id",
            name="uq_rul_calibration_forecast",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "outcome_id",
            name="uq_rul_calibration_outcome",
        ),
        sa.CheckConstraint(
            "actual_minutes >= 0 AND absolute_error_minutes >= 0",
            name="ck_rul_calibration_nonnegative",
        ),
        sa.CheckConstraint(
            "forecast_review_status IN ('PENDING_REVIEW', 'ACCEPTED', 'REJECTED')",
            name="ck_rul_calibration_forecast_review_status",
        ),
        sa.CheckConstraint(
            "calibration_status IN ('INSUFFICIENT_EVIDENCE', 'STABLE', 'RETRAINING_RECOMMENDED')",
            name="ck_rul_calibration_status",
        ),
    )
    op.create_index(
        "ix_rul_forecast_calibrations_tenant_id",
        "rul_forecast_calibrations",
        ["tenant_id"],
    )
    op.create_index(
        "ix_rul_calibration_release_time",
        "rul_forecast_calibrations",
        [
            "tenant_id",
            "source_model_release_id",
            "forecast_model_version",
            "evaluated_at",
        ],
    )

    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE rul_forecast_calibrations ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE rul_forecast_calibrations FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY rul_forecast_calibrations_tenant_isolation "
        "ON rul_forecast_calibrations "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index(
        "ix_rul_calibration_release_time",
        table_name="rul_forecast_calibrations",
    )
    op.drop_index(
        "ix_rul_forecast_calibrations_tenant_id",
        table_name="rul_forecast_calibrations",
    )
    op.drop_table("rul_forecast_calibrations")
