"""Add enterprise FSM assignment handoff, pending gate and signed receipts."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0055_enterprise_fsm_work_order_assignment"
down_revision: str | None = "0054_enterprise_finance_refund_delivery"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "work_orders",
        sa.Column("pending_assignment_operation_id", sa.String(128), nullable=True),
    )
    op.create_unique_constraint(
        "uq_work_order_pending_assignment_operation",
        "work_orders",
        ["tenant_id", "pending_assignment_operation_id"],
    )
    op.create_table(
        "fsm_assignment_deliveries",
        sa.Column("delivery_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("operation_id", sa.String(128), nullable=False),
        sa.Column("proposal_id", sa.String(128), nullable=False),
        sa.Column("approval_id", sa.String(128), nullable=False),
        sa.Column("work_order_id", sa.String(128), nullable=False),
        sa.Column("incident_id", sa.String(128), nullable=False),
        sa.Column("provider", sa.String(128), nullable=False),
        sa.Column("candidate_profile_id", sa.String(128), nullable=False),
        sa.Column("profile_version", sa.String(128), nullable=False),
        sa.Column("assignee_subject_id", sa.String(128), nullable=False),
        sa.Column("candidate_binding", sa.JSON(), nullable=False),
        sa.Column("external_assignment_id", sa.String(255), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("reason", sa.String(255), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("terminal_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_receipt_at", sa.DateTime(timezone=True), nullable=True),
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
            "status IN ('SUBMITTING', 'RECONCILING', 'ACCEPTED', "
            "'REJECTED', 'CANCELLED')",
            name="ck_fsm_assignment_delivery_status",
        ),
        sa.CheckConstraint("version > 0", name="ck_fsm_assignment_delivery_version"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(["proposal_id"], ["action_proposals.proposal_id"]),
        sa.ForeignKeyConstraint(["approval_id"], ["approval_requests.approval_id"]),
        sa.ForeignKeyConstraint(["work_order_id"], ["work_orders.work_order_id"]),
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.incident_id"]),
        sa.UniqueConstraint(
            "tenant_id", "operation_id", name="uq_fsm_assignment_delivery_operation"
        ),
    )
    op.create_index(
        "ix_fsm_assignment_deliveries_tenant_id",
        "fsm_assignment_deliveries",
        ["tenant_id"],
    )
    op.create_index(
        "ix_fsm_assignment_deliveries_work_order",
        "fsm_assignment_deliveries",
        ["tenant_id", "work_order_id", "created_at"],
    )
    op.create_table(
        "fsm_assignment_receipts",
        sa.Column("receipt_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("delivery_id", sa.String(128), nullable=False),
        sa.Column("provider", sa.String(128), nullable=False),
        sa.Column("provider_event_id", sa.String(255), nullable=False),
        sa.Column("operation_id", sa.String(128), nullable=False),
        sa.Column("candidate_profile_id", sa.String(128), nullable=False),
        sa.Column("assignee_subject_id", sa.String(128), nullable=False),
        sa.Column("external_assignment_id", sa.String(255), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("reason", sa.String(255), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("signature_digest", sa.String(128), nullable=False),
        sa.Column("processing_result", sa.String(64), nullable=False),
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
            ["delivery_id"], ["fsm_assignment_deliveries.delivery_id"]
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "provider",
            "provider_event_id",
            name="uq_fsm_assignment_receipt_event",
        ),
    )
    op.create_index(
        "ix_fsm_assignment_receipts_tenant_id",
        "fsm_assignment_receipts",
        ["tenant_id"],
    )
    op.create_index(
        "ix_fsm_assignment_receipts_delivery",
        "fsm_assignment_receipts",
        ["tenant_id", "delivery_id", "occurred_at"],
    )
    op.add_column(
        "work_order_assignments",
        sa.Column("fsm_assignment_delivery_id", sa.String(128), nullable=True),
    )
    op.create_foreign_key(
        "fk_work_order_assignment_fsm_delivery",
        "work_order_assignments",
        "fsm_assignment_deliveries",
        ["fsm_assignment_delivery_id"],
        ["delivery_id"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table in ("fsm_assignment_deliveries", "fsm_assignment_receipts"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    op.drop_constraint(
        "fk_work_order_assignment_fsm_delivery",
        "work_order_assignments",
        type_="foreignkey",
    )
    op.drop_column("work_order_assignments", "fsm_assignment_delivery_id")
    op.drop_index(
        "ix_fsm_assignment_receipts_delivery", table_name="fsm_assignment_receipts"
    )
    op.drop_index(
        "ix_fsm_assignment_receipts_tenant_id", table_name="fsm_assignment_receipts"
    )
    op.drop_table("fsm_assignment_receipts")
    op.drop_index(
        "ix_fsm_assignment_deliveries_work_order",
        table_name="fsm_assignment_deliveries",
    )
    op.drop_index(
        "ix_fsm_assignment_deliveries_tenant_id",
        table_name="fsm_assignment_deliveries",
    )
    op.drop_table("fsm_assignment_deliveries")
    op.drop_constraint(
        "uq_work_order_pending_assignment_operation", "work_orders", type_="unique"
    )
    op.drop_column("work_orders", "pending_assignment_operation_id")
