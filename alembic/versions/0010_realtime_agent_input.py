"""Link confirmed realtime transcripts to their Agent input events."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010_realtime_agent_input"
down_revision: str | None = "0009_m3_realtime_media"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "realtime_transcript_segments",
        sa.Column("agent_event_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "realtime_transcript_segments",
        sa.Column("agent_event_sequence", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_realtime_transcript_agent_event",
        "realtime_transcript_segments",
        "agent_events",
        ["agent_event_id"],
        ["event_id"],
    )
    op.create_unique_constraint(
        "uq_realtime_transcript_agent_event",
        "realtime_transcript_segments",
        ["agent_event_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_realtime_transcript_agent_event",
        "realtime_transcript_segments",
        type_="unique",
    )
    op.drop_constraint(
        "fk_realtime_transcript_agent_event",
        "realtime_transcript_segments",
        type_="foreignkey",
    )
    op.drop_column("realtime_transcript_segments", "agent_event_sequence")
    op.drop_column("realtime_transcript_segments", "agent_event_id")
