"""Add customer notification delivery and normalized receipt ledgers."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0051_customer_notification_deliveries"
down_revision: str | None = "0050_service_performance_baselines"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "customer_notification_deliveries",
        sa.Column("delivery_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("proposal_id", sa.String(128), nullable=False),
        sa.Column("approval_id", sa.String(128), nullable=False),
        sa.Column("incident_id", sa.String(128), nullable=False),
        sa.Column("portal_update_id", sa.String(128), nullable=False),
        sa.Column("operation_id", sa.String(128), nullable=False),
        sa.Column("contact_point_id", sa.String(128), nullable=False),
        sa.Column("contact_channel", sa.String(16), nullable=False),
        sa.Column("source_system", sa.String(128), nullable=False),
        sa.Column("source_record_id", sa.String(128), nullable=False),
        sa.Column("source_version", sa.String(128), nullable=False),
        sa.Column("source_as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("masked_destination", sa.String(320), nullable=False),
        sa.Column("channel", sa.String(32), nullable=False),
        sa.Column("provider", sa.String(128), nullable=False),
        sa.Column("provider_message_id", sa.String(255), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("reason", sa.String(255), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("terminal_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_receipt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
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
        sa.ForeignKeyConstraint(["portal_update_id"], ["customer_service_updates.update_id"]),
        sa.UniqueConstraint(
            "tenant_id",
            "operation_id",
            name="uq_customer_notification_delivery_operation",
        ),
        sa.CheckConstraint("version > 0", name="ck_customer_notification_delivery_version"),
        sa.CheckConstraint(
            "status IN ('PENDING', 'SUBMITTED', 'RECONCILING', 'DELIVERED', "
            "'FAILED', 'UNSUBSCRIBED')",
            name="ck_customer_notification_delivery_status",
        ),
    )
    op.create_index(
        "ix_customer_notification_deliveries_tenant_id",
        "customer_notification_deliveries",
        ["tenant_id"],
    )
    op.create_index(
        "ix_customer_notification_deliveries_incident",
        "customer_notification_deliveries",
        ["tenant_id", "incident_id", "created_at"],
    )
    op.create_table(
        "customer_notification_receipts",
        sa.Column("receipt_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("delivery_id", sa.String(128), nullable=False),
        sa.Column("provider", sa.String(128), nullable=False),
        sa.Column("provider_event_id", sa.String(255), nullable=False),
        sa.Column("operation_id", sa.String(128), nullable=False),
        sa.Column("provider_message_id", sa.String(255), nullable=False),
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
            ["delivery_id"],
            ["customer_notification_deliveries.delivery_id"],
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "provider",
            "provider_event_id",
            name="uq_customer_notification_receipt_event",
        ),
    )
    op.create_index(
        "ix_customer_notification_receipts_tenant_id",
        "customer_notification_receipts",
        ["tenant_id"],
    )
    op.create_index(
        "ix_customer_notification_receipts_delivery",
        "customer_notification_receipts",
        ["tenant_id", "delivery_id", "occurred_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table in ("customer_notification_deliveries", "customer_notification_receipts"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    op.drop_index(
        "ix_customer_notification_receipts_delivery",
        table_name="customer_notification_receipts",
    )
    op.drop_index(
        "ix_customer_notification_receipts_tenant_id",
        table_name="customer_notification_receipts",
    )
    op.drop_table("customer_notification_receipts")
    op.drop_index(
        "ix_customer_notification_deliveries_incident",
        table_name="customer_notification_deliveries",
    )
    op.drop_index(
        "ix_customer_notification_deliveries_tenant_id",
        table_name="customer_notification_deliveries",
    )
    op.drop_table("customer_notification_deliveries")
