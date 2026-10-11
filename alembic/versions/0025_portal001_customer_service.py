"""Add PORTAL-001 customer service updates and confirmations."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0025_portal001_customer_service"
down_revision: str | None = "0024_field001_execution_entries"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "customer_service_updates",
        sa.Column("update_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "incident_id",
            sa.String(length=128),
            sa.ForeignKey("incidents.incident_id"),
            nullable=False,
        ),
        sa.Column(
            "work_order_id",
            sa.String(length=128),
            sa.ForeignKey("work_orders.work_order_id"),
            nullable=True,
        ),
        sa.Column("client_operation_id", sa.String(length=128), nullable=False),
        sa.Column("update_type", sa.String(length=32), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("evidence_ids", sa.JSON(), nullable=False),
        sa.Column("result_accepted", sa.Boolean(), nullable=True),
        sa.Column("satisfaction_rating", sa.Integer(), nullable=True),
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
            "incident_id",
            "client_operation_id",
            name="uq_customer_service_update_operation",
        ),
    )
    op.create_index(
        "ix_customer_service_updates_tenant_id",
        "customer_service_updates",
        ["tenant_id"],
    )
    op.create_index(
        "ix_customer_service_updates_incident",
        "customer_service_updates",
        ["tenant_id", "incident_id", "occurred_at"],
    )
    op.create_index(
        "ix_customer_service_updates_work_order",
        "customer_service_updates",
        ["tenant_id", "work_order_id", "occurred_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE customer_service_updates ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE customer_service_updates FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY customer_service_updates_tenant_isolation "
        f"ON customer_service_updates USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index(
        "ix_customer_service_updates_work_order",
        table_name="customer_service_updates",
    )
    op.drop_index(
        "ix_customer_service_updates_incident",
        table_name="customer_service_updates",
    )
    op.drop_index(
        "ix_customer_service_updates_tenant_id",
        table_name="customer_service_updates",
    )
    op.drop_table("customer_service_updates")
