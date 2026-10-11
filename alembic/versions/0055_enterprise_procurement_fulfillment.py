"""Add enterprise procurement fulfillment and normalized receipt facts."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0056_enterprise_procurement_fulfillment"
down_revision: str | None = "0055_enterprise_fsm_work_order_assignment"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "purchase_fulfillments",
        sa.Column("fulfillment_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("purchase_request_id", sa.String(128), nullable=False),
        sa.Column("provider", sa.String(128), nullable=False),
        sa.Column("operation_id", sa.String(128), nullable=False),
        sa.Column("external_request_id", sa.String(255), nullable=False),
        sa.Column("external_order_id", sa.String(255), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("requested_quantity", sa.Integer(), nullable=False),
        sa.Column("received_quantity", sa.Integer(), nullable=False),
        sa.Column("expected_delivery_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_event_occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_receipt_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reason", sa.String(255), nullable=True),
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
            "status IN ('APPROVED', 'ORDERED', 'IN_TRANSIT', "
            "'PARTIALLY_RECEIVED', 'RECEIVED', 'REJECTED', 'CANCELLED')",
            name="ck_purchase_fulfillment_status",
        ),
        sa.CheckConstraint(
            "requested_quantity > 0",
            name="ck_purchase_fulfillment_requested_quantity",
        ),
        sa.CheckConstraint(
            "received_quantity >= 0 AND received_quantity <= requested_quantity",
            name="ck_purchase_fulfillment_received_quantity",
        ),
        sa.CheckConstraint("version > 0", name="ck_purchase_fulfillment_version"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(
            ["purchase_request_id"], ["purchase_requests.purchase_request_id"]
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "purchase_request_id",
            name="uq_purchase_fulfillment_request",
        ),
    )
    op.create_index(
        "ix_purchase_fulfillments_tenant_id",
        "purchase_fulfillments",
        ["tenant_id"],
    )
    op.create_index(
        "ix_purchase_fulfillments_request",
        "purchase_fulfillments",
        ["tenant_id", "purchase_request_id"],
    )
    op.create_table(
        "purchase_fulfillment_receipts",
        sa.Column("receipt_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("fulfillment_id", sa.String(128), nullable=False),
        sa.Column("purchase_request_id", sa.String(128), nullable=False),
        sa.Column("provider", sa.String(128), nullable=False),
        sa.Column("provider_event_id", sa.String(255), nullable=False),
        sa.Column("operation_id", sa.String(128), nullable=False),
        sa.Column("external_request_id", sa.String(255), nullable=False),
        sa.Column("external_order_id", sa.String(255), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("received_quantity", sa.Integer(), nullable=False),
        sa.Column("expected_delivery_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reason", sa.String(255), nullable=True),
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
            ["fulfillment_id"], ["purchase_fulfillments.fulfillment_id"]
        ),
        sa.ForeignKeyConstraint(
            ["purchase_request_id"], ["purchase_requests.purchase_request_id"]
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "provider",
            "provider_event_id",
            name="uq_purchase_fulfillment_receipt_event",
        ),
    )
    op.create_index(
        "ix_purchase_fulfillment_receipts_tenant_id",
        "purchase_fulfillment_receipts",
        ["tenant_id"],
    )
    op.create_index(
        "ix_purchase_fulfillment_receipts_fulfillment",
        "purchase_fulfillment_receipts",
        ["tenant_id", "fulfillment_id", "occurred_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table in ("purchase_fulfillments", "purchase_fulfillment_receipts"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    op.drop_index(
        "ix_purchase_fulfillment_receipts_fulfillment",
        table_name="purchase_fulfillment_receipts",
    )
    op.drop_index(
        "ix_purchase_fulfillment_receipts_tenant_id",
        table_name="purchase_fulfillment_receipts",
    )
    op.drop_table("purchase_fulfillment_receipts")
    op.drop_index(
        "ix_purchase_fulfillments_request", table_name="purchase_fulfillments"
    )
    op.drop_index(
        "ix_purchase_fulfillments_tenant_id", table_name="purchase_fulfillments"
    )
    op.drop_table("purchase_fulfillments")
