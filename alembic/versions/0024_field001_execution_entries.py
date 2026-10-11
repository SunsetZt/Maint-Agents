"""Add FIELD-001 append-only execution entries."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0024_field001_execution_entries"
down_revision: str | None = "0023_wo001_work_order_operations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "work_order_field_entries",
        sa.Column("entry_id", sa.String(length=128), primary_key=True),
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
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("client_operation_id", sa.String(length=128), nullable=False),
        sa.Column("entry_type", sa.String(length=32), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("actor_subject_id", sa.String(length=128), nullable=False),
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
            "sequence",
            name="uq_work_order_field_entry_sequence",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "work_order_id",
            "client_operation_id",
            name="uq_work_order_field_entry_operation",
        ),
    )
    op.create_index(
        "ix_work_order_field_entries_tenant_id",
        "work_order_field_entries",
        ["tenant_id"],
    )
    op.create_index(
        "ix_work_order_field_entries_work_order",
        "work_order_field_entries",
        ["tenant_id", "work_order_id", "occurred_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE work_order_field_entries ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE work_order_field_entries FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY work_order_field_entries_tenant_isolation "
        f"ON work_order_field_entries USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index(
        "ix_work_order_field_entries_work_order",
        table_name="work_order_field_entries",
    )
    op.drop_index(
        "ix_work_order_field_entries_tenant_id",
        table_name="work_order_field_entries",
    )
    op.drop_table("work_order_field_entries")
