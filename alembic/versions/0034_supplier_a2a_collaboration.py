"""Add tenant-isolated supplier A2A collaboration tasks and event evidence."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0034_supplier_a2a_collaboration"
down_revision: str | None = "0033_media_content_credentials"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "supplier_collaborations",
        sa.Column("collaboration_id", sa.String(length=128), primary_key=True),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("request_fingerprint", sa.String(length=128), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("incident_id", sa.String(length=128), nullable=False),
        sa.Column("diagnosis_run_id", sa.String(length=128), nullable=False),
        sa.Column("asset_id", sa.String(length=128), nullable=False),
        sa.Column("agent_name", sa.String(length=128), nullable=False),
        sa.Column("agent_version", sa.String(length=64), nullable=False),
        sa.Column("skill_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("sharing_reason", sa.Text(), nullable=False),
        sa.Column("request_payload", sa.JSON(), nullable=False),
        sa.Column("request_payload_digest", sa.String(length=128), nullable=False),
        sa.Column("dlp_policy_version", sa.String(length=128), nullable=False),
        sa.Column("dlp_finding_count", sa.Integer(), nullable=False),
        sa.Column("remote_task_id", sa.String(length=255), nullable=True),
        sa.Column("remote_context_id", sa.String(length=255), nullable=True),
        sa.Column("response_payload", sa.JSON(), nullable=True),
        sa.Column("response_digest", sa.String(length=128), nullable=True),
        sa.Column("guardrail_policy_version", sa.String(length=128), nullable=True),
        sa.Column("requested_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("reviewed_by_subject_id", sa.String(length=128), nullable=True),
        sa.Column("review_decision", sa.String(length=32), nullable=True),
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
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.incident_id"]),
        sa.ForeignKeyConstraint(["diagnosis_run_id"], ["diagnosis_runs.diagnosis_run_id"]),
        sa.ForeignKeyConstraint(["asset_id"], ["assets.asset_id"]),
        sa.UniqueConstraint(
            "tenant_id",
            "requested_by_subject_id",
            "idempotency_key",
            name="uq_supplier_collaboration_idempotency",
        ),
    )
    op.create_index(
        "ix_supplier_collaboration_incident",
        "supplier_collaborations",
        ["tenant_id", "incident_id", "created_at"],
    )
    op.create_index(
        "ix_supplier_collaboration_status",
        "supplier_collaborations",
        ["tenant_id", "status", "updated_at"],
    )
    op.create_table(
        "supplier_collaboration_events",
        sa.Column("event_id", sa.String(length=128), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("collaboration_id", sa.String(length=128), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("actor_subject_id", sa.String(length=128), nullable=False),
        sa.Column("reason_code", sa.String(length=128), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
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
        sa.ForeignKeyConstraint(
            ["collaboration_id"],
            ["supplier_collaborations.collaboration_id"],
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "collaboration_id",
            "sequence",
            name="uq_supplier_collaboration_event_sequence",
        ),
    )
    op.create_index(
        "ix_supplier_collaboration_event_time",
        "supplier_collaboration_events",
        ["tenant_id", "collaboration_id", "occurred_at"],
    )
    for table in ("supplier_collaborations", "supplier_collaboration_events"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            "USING (tenant_id = current_setting('app.tenant_id', true)) "
            "WITH CHECK (tenant_id = current_setting('app.tenant_id', true))"
        )


def downgrade() -> None:
    op.drop_table("supplier_collaboration_events")
    op.drop_table("supplier_collaborations")
