"""Bind governed realtime voice sessions to field work orders."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0063_governed_field_realtime_voice_guidance"
down_revision: str | None = "0062_governed_field_evidence_recognition"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "realtime_media_sessions",
        sa.Column("work_order_id", sa.String(128), nullable=True),
    )
    op.add_column(
        "realtime_media_sessions",
        sa.Column("work_order_version", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_realtime_media_sessions_work_order",
        "realtime_media_sessions",
        "work_orders",
        ["work_order_id"],
        ["work_order_id"],
    )
    op.create_check_constraint(
        "ck_realtime_media_sessions_work_order_binding",
        "realtime_media_sessions",
        "(work_order_id IS NULL AND work_order_version IS NULL) OR "
        "(work_order_id IS NOT NULL AND work_order_version IS NOT NULL "
        "AND diagnosis_run_id IS NOT NULL)",
    )
    op.create_index(
        "ix_realtime_media_sessions_field_binding",
        "realtime_media_sessions",
        ["tenant_id", "work_order_id", "subject_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_realtime_media_sessions_field_binding",
        table_name="realtime_media_sessions",
    )
    op.drop_constraint(
        "ck_realtime_media_sessions_work_order_binding",
        "realtime_media_sessions",
        type_="check",
    )
    op.drop_constraint(
        "fk_realtime_media_sessions_work_order",
        "realtime_media_sessions",
        type_="foreignkey",
    )
    op.drop_column("realtime_media_sessions", "work_order_version")
    op.drop_column("realtime_media_sessions", "work_order_id")
