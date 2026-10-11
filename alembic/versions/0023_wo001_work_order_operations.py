"""Add WO-001 SLA planning and append-only work-order controls."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0023_wo001_work_order_operations"
down_revision: str | None = "0022_ast001_asset_master_data"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "work_orders",
        sa.Column(
            "priority",
            sa.String(length=16),
            nullable=False,
            server_default="NORMAL",
        ),
    )
    op.add_column(
        "work_orders",
        sa.Column("sla_due_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "work_orders",
        sa.Column("service_window_start", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "work_orders",
        sa.Column("service_window_end", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "work_order_controls",
        sa.Column("control_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "work_order_id",
            sa.String(length=128),
            sa.ForeignKey("work_orders.work_order_id"),
            nullable=False,
        ),
        sa.Column("work_order_version", sa.Integer(), nullable=False),
        sa.Column("command_type", sa.String(length=32), nullable=False),
        sa.Column("previous_status", sa.String(length=32), nullable=False),
        sa.Column("target_status", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("recovery_condition", sa.Text(), nullable=True),
        sa.Column("responsible_subject_id", sa.String(length=128), nullable=True),
        sa.Column("service_window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("service_window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sla_due_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.UniqueConstraint(
            "tenant_id",
            "work_order_id",
            "work_order_version",
            name="uq_work_order_control_version",
        ),
    )
    op.create_index(
        "ix_work_order_controls_tenant_id",
        "work_order_controls",
        ["tenant_id"],
    )
    op.create_index(
        "ix_work_order_controls_work_order",
        "work_order_controls",
        ["tenant_id", "work_order_id", "occurred_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE work_order_controls ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE work_order_controls FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY work_order_controls_tenant_isolation ON work_order_controls "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index("ix_work_order_controls_work_order", table_name="work_order_controls")
    op.drop_index("ix_work_order_controls_tenant_id", table_name="work_order_controls")
    op.drop_table("work_order_controls")
    op.drop_column("work_orders", "service_window_end")
    op.drop_column("work_orders", "service_window_start")
    op.drop_column("work_orders", "sla_due_at")
    op.drop_column("work_orders", "priority")
