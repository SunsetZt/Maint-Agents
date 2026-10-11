"""Persist external side-effect reconciliation tasks and audit events."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0047_execution_reconciliations"
down_revision: str | None = "0046_refund_requests"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "execution_reconciliations",
        sa.Column("reconciliation_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("proposal_id", sa.String(128), nullable=False),
        sa.Column("approval_id", sa.String(128), nullable=False),
        sa.Column("incident_id", sa.String(128), nullable=False),
        sa.Column("initial_execution_attempt_id", sa.String(128), nullable=False),
        sa.Column("resolved_execution_attempt_id", sa.String(128), nullable=True),
        sa.Column("operation_id", sa.String(128), nullable=False),
        sa.Column("tool_id", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("outcome", sa.String(32), nullable=False),
        sa.Column("reason", sa.String(255), nullable=True),
        sa.Column("external_reference_id", sa.String(255), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(["proposal_id"], ["action_proposals.proposal_id"]),
        sa.ForeignKeyConstraint(["approval_id"], ["approval_requests.approval_id"]),
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.incident_id"]),
        sa.ForeignKeyConstraint(
            ["initial_execution_attempt_id"],
            ["execution_attempts.attempt_id"],
        ),
        sa.ForeignKeyConstraint(
            ["resolved_execution_attempt_id"],
            ["execution_attempts.attempt_id"],
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "operation_id",
            name="uq_execution_reconciliation_operation",
        ),
        sa.CheckConstraint("version > 0", name="ck_execution_reconciliation_version"),
        sa.CheckConstraint(
            "status IN ('PENDING', 'RESOLVED', 'ESCALATED')",
            name="ck_execution_reconciliation_status",
        ),
        sa.CheckConstraint(
            "outcome IN ('UNKNOWN', 'SUCCEEDED', 'FAILED')",
            name="ck_execution_reconciliation_outcome",
        ),
    )
    op.create_index(
        "ix_execution_reconciliations_tenant_id",
        "execution_reconciliations",
        ["tenant_id"],
    )
    op.create_index(
        "ix_execution_reconciliations_status",
        "execution_reconciliations",
        ["tenant_id", "status", "created_at"],
    )
    op.create_index(
        "ix_execution_reconciliations_incident",
        "execution_reconciliations",
        ["tenant_id", "incident_id", "created_at"],
    )

    op.create_table(
        "execution_reconciliation_events",
        sa.Column("event_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("reconciliation_id", sa.String(128), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("actor_subject_id", sa.String(128), nullable=False),
        sa.Column("reason", sa.String(255), nullable=True),
        sa.Column("outcome", sa.String(32), nullable=False),
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
            ["reconciliation_id"],
            ["execution_reconciliations.reconciliation_id"],
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "reconciliation_id",
            "sequence",
            name="uq_execution_reconciliation_event_sequence",
        ),
        sa.CheckConstraint(
            "sequence > 0",
            name="ck_execution_reconciliation_event_sequence",
        ),
    )
    op.create_index(
        "ix_execution_reconciliation_events_tenant_id",
        "execution_reconciliation_events",
        ["tenant_id"],
    )
    op.create_index(
        "ix_execution_reconciliation_events_task",
        "execution_reconciliation_events",
        ["tenant_id", "reconciliation_id", "sequence"],
    )

    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table in ("execution_reconciliations", "execution_reconciliation_events"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    op.drop_index(
        "ix_execution_reconciliation_events_task",
        table_name="execution_reconciliation_events",
    )
    op.drop_index(
        "ix_execution_reconciliation_events_tenant_id",
        table_name="execution_reconciliation_events",
    )
    op.drop_table("execution_reconciliation_events")
    op.drop_index(
        "ix_execution_reconciliations_incident",
        table_name="execution_reconciliations",
    )
    op.drop_index(
        "ix_execution_reconciliations_status",
        table_name="execution_reconciliations",
    )
    op.drop_index(
        "ix_execution_reconciliations_tenant_id",
        table_name="execution_reconciliations",
    )
    op.drop_table("execution_reconciliations")
