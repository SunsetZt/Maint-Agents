"""Add governed remote expert recommendation handoff."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0067_governed_remote_expert_outcome_handoff"
down_revision: str | None = "0066_governed_remote_expert_audio_collaboration"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "work_order_expert_collaboration_recommendations",
        sa.Column("recommendation_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("collaboration_id", sa.String(128), nullable=False),
        sa.Column("work_order_id", sa.String(128), nullable=False),
        sa.Column("work_order_version", sa.Integer(), nullable=False),
        sa.Column("incident_id", sa.String(128), nullable=False),
        sa.Column("asset_id", sa.String(128), nullable=False),
        sa.Column("expert_subject_id", sa.String(128), nullable=False),
        sa.Column("recommendation_type", sa.String(32), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("basis", sa.Text(), nullable=False),
        sa.Column("recommended_checks_json", sa.JSON(), nullable=False),
        sa.Column("safety_notice", sa.Text(), nullable=True),
        sa.Column("evidence_entry_ids_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("request_digest", sa.String(128), nullable=False),
        sa.Column("reviewed_by_subject_id", sa.String(128), nullable=True),
        sa.Column("review_reason", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("field_entry_id", sa.String(128), nullable=True),
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
            "recommendation_type IN "
            "('ADVICE', 'REQUEST_MORE_EVIDENCE', 'STOP_AND_ESCALATE')",
            name="ck_expert_recommendation_type",
        ),
        sa.CheckConstraint(
            "status IN ('PENDING_FIELD_REVIEW', 'ACCEPTED', 'REJECTED')",
            name="ck_expert_recommendation_status",
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(
            ["collaboration_id"],
            ["work_order_expert_collaborations.collaboration_id"],
        ),
        sa.ForeignKeyConstraint(["work_order_id"], ["work_orders.work_order_id"]),
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.incident_id"]),
        sa.ForeignKeyConstraint(["asset_id"], ["assets.asset_id"]),
        sa.ForeignKeyConstraint(
            ["field_entry_id"], ["work_order_field_entries.entry_id"]
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "collaboration_id",
            name="uq_expert_recommendation_collaboration",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "expert_subject_id",
            "idempotency_key",
            name="uq_expert_recommendation_idempotency",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "field_entry_id",
            name="uq_expert_recommendation_field_entry",
        ),
    )
    op.create_index(
        "ix_work_order_expert_collaboration_recommendations_tenant_id",
        "work_order_expert_collaboration_recommendations",
        ["tenant_id"],
    )
    op.create_index(
        "ix_expert_recommendation_work_order_status",
        "work_order_expert_collaboration_recommendations",
        ["tenant_id", "work_order_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_expert_recommendation_work_order_status",
        table_name="work_order_expert_collaboration_recommendations",
    )
    op.drop_index(
        "ix_work_order_expert_collaboration_recommendations_tenant_id",
        table_name="work_order_expert_collaboration_recommendations",
    )
    op.drop_table("work_order_expert_collaboration_recommendations")
