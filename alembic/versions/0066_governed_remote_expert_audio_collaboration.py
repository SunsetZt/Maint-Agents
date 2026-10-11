"""Add WorkOrder-bound remote expert audio collaboration."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0066_governed_remote_expert_audio_collaboration"
down_revision: str | None = "0065_governed_field_edge_diagnosis"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "work_order_expert_collaborations",
        sa.Column("collaboration_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("work_order_id", sa.String(128), nullable=False),
        sa.Column("work_order_version", sa.Integer(), nullable=False),
        sa.Column("incident_id", sa.String(128), nullable=False),
        sa.Column("asset_id", sa.String(128), nullable=False),
        sa.Column("requested_by_subject_id", sa.String(128), nullable=False),
        sa.Column("invited_subject_id", sa.String(128), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("request_digest", sa.String(128), nullable=False),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_by_subject_id", sa.String(128), nullable=True),
        sa.Column("ended_reason", sa.Text(), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
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
        sa.CheckConstraint(
            "status IN ('WAITING_EXPERT', 'ACTIVE', 'ENDED', 'EXPIRED')",
            name="ck_expert_collaboration_status",
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(["work_order_id"], ["work_orders.work_order_id"]),
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.incident_id"]),
        sa.ForeignKeyConstraint(["asset_id"], ["assets.asset_id"]),
        sa.UniqueConstraint(
            "tenant_id",
            "requested_by_subject_id",
            "idempotency_key",
            name="uq_expert_collaboration_idempotency",
        ),
    )
    op.create_index(
        "ix_work_order_expert_collaborations_tenant_id",
        "work_order_expert_collaborations",
        ["tenant_id"],
    )
    op.create_index(
        "ix_expert_collaboration_work_order_status",
        "work_order_expert_collaborations",
        ["tenant_id", "work_order_id", "status"],
    )
    op.create_index(
        "uq_expert_collaboration_open_work_order",
        "work_order_expert_collaborations",
        ["tenant_id", "work_order_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('WAITING_EXPERT', 'ACTIVE')"),
        sqlite_where=sa.text("status IN ('WAITING_EXPERT', 'ACTIVE')"),
    )
    op.create_index(
        "ix_expert_collaboration_invited_status",
        "work_order_expert_collaborations",
        ["tenant_id", "invited_subject_id", "status"],
    )
    op.create_table(
        "work_order_expert_collaboration_events",
        sa.Column("event_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("collaboration_id", sa.String(128), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("actor_subject_id", sa.String(128), nullable=False),
        sa.Column("from_status", sa.String(32), nullable=True),
        sa.Column("to_status", sa.String(32), nullable=False),
        sa.Column("reason_code", sa.String(64), nullable=False),
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
            ["work_order_expert_collaborations.collaboration_id"],
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "collaboration_id",
            "sequence",
            name="uq_expert_collaboration_event_sequence",
        ),
    )
    op.create_index(
        "ix_work_order_expert_collaboration_events_tenant_id",
        "work_order_expert_collaboration_events",
        ["tenant_id"],
    )
    op.drop_constraint(
        "ck_realtime_media_sessions_work_order_binding",
        "realtime_media_sessions",
        type_="check",
    )
    op.add_column(
        "realtime_media_sessions",
        sa.Column("expert_collaboration_id", sa.String(128), nullable=True),
    )
    op.create_foreign_key(
        "fk_realtime_media_sessions_expert_collaboration",
        "realtime_media_sessions",
        "work_order_expert_collaborations",
        ["expert_collaboration_id"],
        ["collaboration_id"],
    )
    op.create_check_constraint(
        "ck_realtime_media_sessions_work_order_binding",
        "realtime_media_sessions",
        "(expert_collaboration_id IS NULL AND "
        "((work_order_id IS NULL AND work_order_version IS NULL) OR "
        "(work_order_id IS NOT NULL AND work_order_version IS NOT NULL "
        "AND diagnosis_run_id IS NOT NULL))) OR "
        "(expert_collaboration_id IS NOT NULL AND work_order_id IS NOT NULL "
        "AND work_order_version IS NOT NULL AND diagnosis_run_id IS NULL)",
    )
    op.create_index(
        "ix_realtime_media_sessions_expert_participant",
        "realtime_media_sessions",
        ["tenant_id", "expert_collaboration_id", "subject_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_realtime_media_sessions_expert_participant",
        table_name="realtime_media_sessions",
    )
    op.drop_constraint(
        "ck_realtime_media_sessions_work_order_binding",
        "realtime_media_sessions",
        type_="check",
    )
    op.drop_constraint(
        "fk_realtime_media_sessions_expert_collaboration",
        "realtime_media_sessions",
        type_="foreignkey",
    )
    op.drop_column("realtime_media_sessions", "expert_collaboration_id")
    op.create_check_constraint(
        "ck_realtime_media_sessions_work_order_binding",
        "realtime_media_sessions",
        "(work_order_id IS NULL AND work_order_version IS NULL) OR "
        "(work_order_id IS NOT NULL AND work_order_version IS NOT NULL "
        "AND diagnosis_run_id IS NOT NULL)",
    )
    op.drop_index(
        "ix_work_order_expert_collaboration_events_tenant_id",
        table_name="work_order_expert_collaboration_events",
    )
    op.drop_table("work_order_expert_collaboration_events")
    op.drop_index(
        "ix_expert_collaboration_invited_status",
        table_name="work_order_expert_collaborations",
    )
    op.drop_index(
        "uq_expert_collaboration_open_work_order",
        table_name="work_order_expert_collaborations",
    )
    op.drop_index(
        "ix_expert_collaboration_work_order_status",
        table_name="work_order_expert_collaborations",
    )
    op.drop_index(
        "ix_work_order_expert_collaborations_tenant_id",
        table_name="work_order_expert_collaborations",
    )
    op.drop_table("work_order_expert_collaborations")
