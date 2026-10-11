"""Add authoritative WMS part issue lifecycle and WorkOrder compatibility gate."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0057_authoritative_part_issue_handoff"
down_revision: str | None = "0056_enterprise_procurement_fulfillment"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "work_orders",
        sa.Column(
            "part_issue_required",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_table(
        "part_issues",
        sa.Column("part_issue_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("operation_id", sa.String(128), nullable=False),
        sa.Column("proposal_id", sa.String(128), nullable=False),
        sa.Column("approval_id", sa.String(128), nullable=False),
        sa.Column("work_order_id", sa.String(128), nullable=False),
        sa.Column("incident_id", sa.String(128), nullable=False),
        sa.Column("reservation_id", sa.String(128), nullable=False),
        sa.Column("part_number", sa.String(128), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("external_issue_id", sa.String(255), nullable=True),
        sa.Column("source", sa.String(128), nullable=True),
        sa.Column("source_record_id", sa.String(255), nullable=True),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reconciliation_id", sa.String(128), nullable=True),
        sa.Column("reason", sa.String(255), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.CheckConstraint("quantity > 0", name="ck_part_issue_quantity"),
        sa.CheckConstraint(
            "status IN ('PENDING_APPROVAL', 'APPROVED', 'REJECTED', "
            "'RECONCILING', 'ISSUED', 'FAILED')",
            name="ck_part_issue_status",
        ),
        sa.CheckConstraint("version > 0", name="ck_part_issue_version"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(["proposal_id"], ["action_proposals.proposal_id"]),
        sa.ForeignKeyConstraint(["approval_id"], ["approval_requests.approval_id"]),
        sa.ForeignKeyConstraint(["work_order_id"], ["work_orders.work_order_id"]),
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.incident_id"]),
        sa.ForeignKeyConstraint(["reservation_id"], ["part_reservations.reservation_id"]),
        sa.ForeignKeyConstraint(
            ["reconciliation_id"], ["execution_reconciliations.reconciliation_id"]
        ),
        sa.UniqueConstraint(
            "tenant_id", "operation_id", name="uq_part_issue_operation"
        ),
        sa.UniqueConstraint(
            "tenant_id", "work_order_id", name="uq_part_issue_work_order"
        ),
        sa.UniqueConstraint(
            "tenant_id", "reservation_id", name="uq_part_issue_reservation"
        ),
    )
    op.create_index("ix_part_issues_tenant_id", "part_issues", ["tenant_id"])
    op.create_index(
        "ix_part_issues_work_order",
        "part_issues",
        ["tenant_id", "work_order_id", "created_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE part_issues ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE part_issues FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY part_issues_tenant_isolation ON part_issues "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index("ix_part_issues_work_order", table_name="part_issues")
    op.drop_index("ix_part_issues_tenant_id", table_name="part_issues")
    op.drop_table("part_issues")
    op.drop_column("work_orders", "part_issue_required")
