"""Add tenant-scoped WebRTC media session leases."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009_m3_realtime_media"
down_revision: str | None = "0008_m5_model_gateway"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "realtime_media_sessions",
        sa.Column("session_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "incident_id",
            sa.String(length=128),
            sa.ForeignKey("incidents.incident_id"),
            nullable=False,
        ),
        sa.Column(
            "diagnosis_run_id",
            sa.String(length=128),
            sa.ForeignKey("diagnosis_runs.diagnosis_run_id"),
            nullable=True,
        ),
        sa.Column("subject_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("media_kind", sa.String(length=32), nullable=False),
        sa.Column("token_digest", sa.String(length=128), nullable=False),
        sa.Column("token_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("token_consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("media_token_digest", sa.String(length=128), nullable=True),
        sa.Column("media_token_issued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("max_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reconnect_count", sa.Integer(), nullable=False),
        sa.Column("ended_reason", sa.String(length=128), nullable=True),
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
            "tenant_id", "token_digest", name="uq_realtime_media_token"
        ),
    )
    op.create_index(
        "ix_realtime_media_tenant_subject",
        "realtime_media_sessions",
        ["tenant_id", "subject_id", "status"],
    )
    op.create_index(
        "ix_realtime_media_tenant_incident",
        "realtime_media_sessions",
        ["tenant_id", "incident_id"],
    )
    op.create_table(
        "realtime_transcript_segments",
        sa.Column("segment_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "session_id",
            sa.String(length=128),
            sa.ForeignKey("realtime_media_sessions.session_id"),
            nullable=False,
        ),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("start_ms", sa.BigInteger(), nullable=False),
        sa.Column("end_ms", sa.BigInteger(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("confirmed_text", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("language", sa.String(length=32), nullable=False),
        sa.Column("entities_json", sa.JSON(), nullable=False),
        sa.Column("audio_sha256", sa.String(length=128), nullable=False),
        sa.Column("resolved_release_id", sa.String(length=128), nullable=False),
        sa.Column("manifest_hash", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("reviewer_subject_id", sa.String(length=128), nullable=True),
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
            "session_id",
            "sequence",
            name="uq_realtime_transcript_sequence",
        ),
    )
    op.create_index(
        "ix_realtime_transcript_tenant_session",
        "realtime_transcript_segments",
        ["tenant_id", "session_id", "sequence"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table_name in ("realtime_media_sessions", "realtime_transcript_segments"):
        op.execute(f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table_name} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table_name}_tenant_isolation ON {table_name} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    op.drop_index(
        "ix_realtime_transcript_tenant_session",
        table_name="realtime_transcript_segments",
    )
    op.drop_table("realtime_transcript_segments")
    op.drop_index(
        "ix_realtime_media_tenant_incident", table_name="realtime_media_sessions"
    )
    op.drop_index(
        "ix_realtime_media_tenant_subject", table_name="realtime_media_sessions"
    )
    op.drop_table("realtime_media_sessions")
