"""Add versioned W3C Trace Context transport to the transactional Outbox."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0041_event_trace_context"
down_revision: str | None = "0040_knowledge_index_rollback"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "event_outbox",
        sa.Column("trace_context_profile", sa.String(64), nullable=True),
    )
    op.add_column(
        "event_outbox",
        sa.Column("source_traceparent", sa.String(512), nullable=True),
    )
    op.add_column(
        "event_outbox",
        sa.Column("source_tracestate", sa.String(512), nullable=True),
    )
    op.add_column(
        "event_outbox",
        sa.Column("tracingspancontext", sa.Text(), nullable=True),
    )
    op.add_column(
        "event_inbox",
        sa.Column("trace_context_propagation", sa.String(32), nullable=True),
    )
    op.add_column(
        "event_inbox",
        sa.Column("parent_trace_id", sa.String(32), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("event_inbox", "parent_trace_id")
    op.drop_column("event_inbox", "trace_context_propagation")
    op.drop_column("event_outbox", "tracingspancontext")
    op.drop_column("event_outbox", "source_tracestate")
    op.drop_column("event_outbox", "source_traceparent")
    op.drop_column("event_outbox", "trace_context_profile")
