"""Add governed telemetry windows and predictive-maintenance alert candidates."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0018_m7_predictive_maintenance"
down_revision: str | None = "0017_m6_security_acceptance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> tuple[sa.Column[object], sa.Column[object]]:
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


def _tenant() -> sa.Column[object]:
    return sa.Column(
        "tenant_id",
        sa.String(length=64),
        sa.ForeignKey("tenants.id"),
        nullable=False,
    )


def upgrade() -> None:
    op.create_table(
        "telemetry_events",
        sa.Column("event_id", sa.String(length=128), primary_key=True),
        _tenant(),
        sa.Column(
            "asset_id", sa.String(length=128), sa.ForeignKey("assets.asset_id"), nullable=False
        ),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("sequence", sa.BigInteger(), nullable=False),
        sa.Column("source_system", sa.String(length=128), nullable=False),
        sa.Column("source_ref", sa.String(length=255), nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("measurements_json", sa.JSON(), nullable=False),
        sa.Column("quality_flags_json", sa.JSON(), nullable=False),
        sa.Column("payload_digest", sa.String(length=128), nullable=False),
        sa.Column("ingestion_status", sa.String(length=32), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "asset_id",
            "schema_version",
            "sequence",
            name="uq_telemetry_stream_sequence",
        ),
    )
    op.create_index("ix_telemetry_events_tenant_id", "telemetry_events", ["tenant_id"])
    op.create_index(
        "ix_telemetry_events_asset_time",
        "telemetry_events",
        ["tenant_id", "asset_id", "event_time"],
    )

    op.create_table(
        "telemetry_stream_states",
        sa.Column("stream_id", sa.String(length=128), primary_key=True),
        _tenant(),
        sa.Column(
            "asset_id", sa.String(length=128), sa.ForeignKey("assets.asset_id"), nullable=False
        ),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("max_event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("max_sequence", sa.BigInteger(), nullable=False),
        sa.Column("watermark_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "asset_id",
            "schema_version",
            name="uq_telemetry_stream_state",
        ),
    )
    op.create_index(
        "ix_telemetry_stream_states_tenant_id",
        "telemetry_stream_states",
        ["tenant_id"],
    )

    op.create_table(
        "telemetry_windows",
        sa.Column("window_id", sa.String(length=128), primary_key=True),
        _tenant(),
        sa.Column(
            "asset_id", sa.String(length=128), sa.ForeignKey("assets.asset_id"), nullable=False
        ),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_snapshot_id", sa.String(length=128), nullable=False),
        sa.Column("source_digest", sa.String(length=128), nullable=False),
        sa.Column("source_event_ids_json", sa.JSON(), nullable=False),
        sa.Column("features_json", sa.JSON(), nullable=False),
        sa.Column("missing_signals_json", sa.JSON(), nullable=False),
        sa.Column("supporting_signal_refs_json", sa.JSON(), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("quality_status", sa.String(length=32), nullable=False),
        sa.Column("includes_too_late", sa.Boolean(), nullable=False),
        sa.Column(
            "supersedes_window_id",
            sa.String(length=128),
            sa.ForeignKey("telemetry_windows.window_id"),
            nullable=True,
        ),
        sa.Column("created_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "feature_snapshot_id",
            name="uq_telemetry_feature_snapshot",
        ),
    )
    op.create_index("ix_telemetry_windows_tenant_id", "telemetry_windows", ["tenant_id"])
    op.create_index(
        "ix_telemetry_windows_asset_time",
        "telemetry_windows",
        ["tenant_id", "asset_id", "window_start", "window_end"],
    )

    op.create_table(
        "alert_candidates",
        sa.Column("alert_candidate_id", sa.String(length=128), primary_key=True),
        _tenant(),
        sa.Column(
            "asset_id", sa.String(length=128), sa.ForeignKey("assets.asset_id"), nullable=False
        ),
        sa.Column(
            "window_id",
            sa.String(length=128),
            sa.ForeignKey("telemetry_windows.window_id"),
            nullable=False,
        ),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_snapshot_id", sa.String(length=128), nullable=False),
        sa.Column("model_release_id", sa.String(length=128), nullable=False),
        sa.Column("detector_kind", sa.String(length=32), nullable=False),
        sa.Column("anomaly_score", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("threshold_policy_id", sa.String(length=128), nullable=False),
        sa.Column("supporting_signal_refs_json", sa.JSON(), nullable=False),
        sa.Column("explanation_codes_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reviewed_by_subject_id", sa.String(length=128), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "window_id",
            "threshold_policy_id",
            "model_release_id",
            name="uq_alert_candidate_detector",
        ),
    )
    op.create_index("ix_alert_candidates_tenant_id", "alert_candidates", ["tenant_id"])
    op.create_index(
        "ix_alert_candidates_status",
        "alert_candidates",
        ["tenant_id", "status", "detected_at"],
    )

    op.create_table(
        "alert_candidate_decisions",
        sa.Column("decision_id", sa.String(length=128), primary_key=True),
        _tenant(),
        sa.Column(
            "alert_candidate_id",
            sa.String(length=128),
            sa.ForeignKey("alert_candidates.alert_candidate_id"),
            nullable=False,
        ),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("reason_code", sa.String(length=128), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("decided_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("candidate_version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "alert_candidate_id",
            name="uq_alert_candidate_decision",
        ),
    )
    op.create_index(
        "ix_alert_candidate_decisions_tenant_id",
        "alert_candidate_decisions",
        ["tenant_id"],
    )

    op.create_table(
        "predictive_maintenance_referrals",
        sa.Column("referral_id", sa.String(length=128), primary_key=True),
        _tenant(),
        sa.Column(
            "alert_candidate_id",
            sa.String(length=128),
            sa.ForeignKey("alert_candidates.alert_candidate_id"),
            nullable=False,
        ),
        sa.Column(
            "incident_draft_id",
            sa.String(length=128),
            sa.ForeignKey("incident_drafts.draft_id"),
            nullable=False,
        ),
        sa.Column("linked_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("linked_at", sa.DateTime(timezone=True), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "alert_candidate_id",
            name="uq_predictive_referral_alert",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "incident_draft_id",
            name="uq_predictive_referral_draft",
        ),
    )
    op.create_index(
        "ix_predictive_maintenance_referrals_tenant_id",
        "predictive_maintenance_referrals",
        ["tenant_id"],
    )

    op.create_table(
        "predictive_outcomes",
        sa.Column("outcome_id", sa.String(length=128), primary_key=True),
        _tenant(),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("request_hash", sa.String(length=128), nullable=False),
        sa.Column(
            "asset_id", sa.String(length=128), sa.ForeignKey("assets.asset_id"), nullable=False
        ),
        sa.Column(
            "alert_candidate_id",
            sa.String(length=128),
            sa.ForeignKey("alert_candidates.alert_candidate_id"),
            nullable=True,
        ),
        sa.Column("outcome_type", sa.String(length=32), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("failure_observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "work_order_id",
            sa.String(length=128),
            sa.ForeignKey("work_orders.work_order_id"),
            nullable=True,
        ),
        sa.Column("avoided_downtime_minutes", sa.Integer(), nullable=False),
        sa.Column("evidence_ref", sa.String(length=255), nullable=False),
        sa.Column("evidence_digest", sa.String(length=128), nullable=False),
        sa.Column("recorded_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_predictive_outcome_idempotency",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "alert_candidate_id",
            name="uq_predictive_outcome_alert",
        ),
    )
    op.create_index("ix_predictive_outcomes_tenant_id", "predictive_outcomes", ["tenant_id"])
    op.create_index(
        "ix_predictive_outcomes_asset_time",
        "predictive_outcomes",
        ["tenant_id", "asset_id", "observed_at"],
    )

    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table in (
        "telemetry_events",
        "telemetry_stream_states",
        "telemetry_windows",
        "alert_candidates",
        "alert_candidate_decisions",
        "predictive_maintenance_referrals",
        "predictive_outcomes",
    ):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    op.drop_index("ix_predictive_outcomes_asset_time", table_name="predictive_outcomes")
    op.drop_index("ix_predictive_outcomes_tenant_id", table_name="predictive_outcomes")
    op.drop_table("predictive_outcomes")
    op.drop_index(
        "ix_predictive_maintenance_referrals_tenant_id",
        table_name="predictive_maintenance_referrals",
    )
    op.drop_table("predictive_maintenance_referrals")
    op.drop_index(
        "ix_alert_candidate_decisions_tenant_id",
        table_name="alert_candidate_decisions",
    )
    op.drop_table("alert_candidate_decisions")
    op.drop_index("ix_alert_candidates_status", table_name="alert_candidates")
    op.drop_index("ix_alert_candidates_tenant_id", table_name="alert_candidates")
    op.drop_table("alert_candidates")
    op.drop_index("ix_telemetry_windows_asset_time", table_name="telemetry_windows")
    op.drop_index("ix_telemetry_windows_tenant_id", table_name="telemetry_windows")
    op.drop_table("telemetry_windows")
    op.drop_index(
        "ix_telemetry_stream_states_tenant_id",
        table_name="telemetry_stream_states",
    )
    op.drop_table("telemetry_stream_states")
    op.drop_index("ix_telemetry_events_asset_time", table_name="telemetry_events")
    op.drop_index("ix_telemetry_events_tenant_id", table_name="telemetry_events")
    op.drop_table("telemetry_events")
