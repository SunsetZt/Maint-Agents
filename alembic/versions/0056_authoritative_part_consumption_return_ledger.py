"""Add authoritative consumption/return ledger and WorkOrder compatibility gate."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0058_authoritative_part_consumption_return_ledger"
down_revision: str | None = "0057_authoritative_part_issue_handoff"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "work_orders",
        sa.Column(
            "part_accounting_required",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_table(
        "part_material_movements",
        sa.Column("movement_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("operation_id", sa.String(128), nullable=False),
        sa.Column("proposal_id", sa.String(128), nullable=False),
        sa.Column("approval_id", sa.String(128), nullable=False),
        sa.Column("work_order_id", sa.String(128), nullable=False),
        sa.Column("incident_id", sa.String(128), nullable=False),
        sa.Column("part_issue_id", sa.String(128), nullable=False),
        sa.Column("reservation_id", sa.String(128), nullable=False),
        sa.Column("field_entry_id", sa.String(128), nullable=True),
        sa.Column("movement_kind", sa.String(16), nullable=False),
        sa.Column("part_number", sa.String(128), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("external_movement_id", sa.String(255), nullable=True),
        sa.Column("source", sa.String(128), nullable=True),
        sa.Column("source_record_id", sa.String(255), nullable=True),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reconciliation_id", sa.String(128), nullable=True),
        sa.Column("reason", sa.String(255), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.CheckConstraint("quantity > 0", name="ck_part_material_movement_quantity"),
        sa.CheckConstraint(
            "movement_kind IN ('CONSUME', 'RETURN')",
            name="ck_part_material_movement_kind",
        ),
        sa.CheckConstraint(
            "status IN ('PENDING_APPROVAL', 'APPROVED', 'REJECTED', "
            "'RECONCILING', 'APPLIED', 'FAILED')",
            name="ck_part_material_movement_status",
        ),
        sa.CheckConstraint("version > 0", name="ck_part_material_movement_version"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(["proposal_id"], ["action_proposals.proposal_id"]),
        sa.ForeignKeyConstraint(["approval_id"], ["approval_requests.approval_id"]),
        sa.ForeignKeyConstraint(["work_order_id"], ["work_orders.work_order_id"]),
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.incident_id"]),
        sa.ForeignKeyConstraint(["part_issue_id"], ["part_issues.part_issue_id"]),
        sa.ForeignKeyConstraint(["reservation_id"], ["part_reservations.reservation_id"]),
        sa.ForeignKeyConstraint(["field_entry_id"], ["work_order_field_entries.entry_id"]),
        sa.ForeignKeyConstraint(
            ["reconciliation_id"], ["execution_reconciliations.reconciliation_id"]
        ),
        sa.UniqueConstraint(
            "tenant_id", "operation_id", name="uq_part_material_movement_operation"
        ),
    )
    op.create_index(
        "ix_part_material_movements_tenant_id",
        "part_material_movements",
        ["tenant_id"],
    )
    op.create_index(
        "ix_part_material_movements_work_order",
        "part_material_movements",
        ["tenant_id", "work_order_id", "created_at"],
    )
    live_consumption = sa.text(
        "field_entry_id IS NOT NULL AND movement_kind = 'CONSUME' AND "
        "status IN ('PENDING_APPROVAL', 'APPROVED', 'RECONCILING', 'APPLIED')"
    )
    op.create_index(
        "uq_part_material_movement_live_field_entry",
        "part_material_movements",
        ["tenant_id", "field_entry_id"],
        unique=True,
        postgresql_where=live_consumption,
        sqlite_where=live_consumption,
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE part_material_movements ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE part_material_movements FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY part_material_movements_tenant_isolation "
        "ON part_material_movements "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index(
        "uq_part_material_movement_live_field_entry",
        table_name="part_material_movements",
    )
    op.drop_index(
        "ix_part_material_movements_work_order",
        table_name="part_material_movements",
    )
    op.drop_index(
        "ix_part_material_movements_tenant_id",
        table_name="part_material_movements",
    )
    op.drop_table("part_material_movements")
    op.drop_column("work_orders", "part_accounting_required")
