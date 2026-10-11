"""Add enterprise purchase requisition delivery and normalized receipt facts."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0052_enterprise_purchase_requisition_delivery"
down_revision: str | None = "0051_customer_notification_deliveries"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "purchase_requests",
        sa.Column(
            "delivery_target",
            sa.String(32),
            nullable=False,
            server_default=sa.text("'INTERNAL'"),
        ),
    )
    op.add_column("purchase_requests", sa.Column("provider", sa.String(128), nullable=True))
    op.add_column("purchase_requests", sa.Column("profile_id", sa.String(128), nullable=True))
    op.add_column(
        "purchase_requests", sa.Column("profile_version", sa.String(128), nullable=True)
    )
    op.add_column(
        "purchase_requests",
        sa.Column("profile_display_name", sa.String(128), nullable=True),
    )
    op.add_column(
        "purchase_requests",
        sa.Column("profile_as_of", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "purchase_requests",
        sa.Column("profile_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "purchase_requests",
        sa.Column("external_request_id", sa.String(255), nullable=True),
    )
    op.add_column("purchase_requests", sa.Column("reason", sa.String(255), nullable=True))
    op.add_column(
        "purchase_requests",
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "purchase_requests",
        sa.Column("terminal_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "purchase_requests",
        sa.Column("last_receipt_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "purchase_requests",
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
    )
    op.create_check_constraint(
        "ck_purchase_request_delivery_target",
        "purchase_requests",
        "delivery_target IN ('INTERNAL', 'ENTERPRISE_ERP')",
    )
    op.create_check_constraint(
        "ck_purchase_request_status",
        "purchase_requests",
        "status IN ('SUBMITTED', 'SUBMITTING', 'RECONCILING', "
        "'ACCEPTED', 'REJECTED', 'CANCELLED')",
    )
    op.create_check_constraint(
        "ck_purchase_request_version",
        "purchase_requests",
        "version > 0",
    )
    op.alter_column("purchase_requests", "delivery_target", server_default=None)
    op.alter_column("purchase_requests", "version", server_default=None)

    op.create_table(
        "purchase_request_receipts",
        sa.Column("receipt_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("purchase_request_id", sa.String(128), nullable=False),
        sa.Column("provider", sa.String(128), nullable=False),
        sa.Column("provider_event_id", sa.String(255), nullable=False),
        sa.Column("operation_id", sa.String(128), nullable=False),
        sa.Column("external_request_id", sa.String(255), nullable=False),
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
            ["purchase_request_id"],
            ["purchase_requests.purchase_request_id"],
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "provider",
            "provider_event_id",
            name="uq_purchase_request_receipt_event",
        ),
    )
    op.create_index(
        "ix_purchase_request_receipts_tenant_id",
        "purchase_request_receipts",
        ["tenant_id"],
    )
    op.create_index(
        "ix_purchase_request_receipts_request",
        "purchase_request_receipts",
        ["tenant_id", "purchase_request_id", "occurred_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE purchase_request_receipts ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE purchase_request_receipts FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY purchase_request_receipts_tenant_isolation "
        "ON purchase_request_receipts "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index(
        "ix_purchase_request_receipts_request",
        table_name="purchase_request_receipts",
    )
    op.drop_index(
        "ix_purchase_request_receipts_tenant_id",
        table_name="purchase_request_receipts",
    )
    op.drop_table("purchase_request_receipts")
    op.drop_constraint(
        "ck_purchase_request_version", "purchase_requests", type_="check"
    )
    op.drop_constraint(
        "ck_purchase_request_status", "purchase_requests", type_="check"
    )
    op.drop_constraint(
        "ck_purchase_request_delivery_target", "purchase_requests", type_="check"
    )
    for column in (
        "version",
        "last_receipt_at",
        "terminal_at",
        "submitted_at",
        "reason",
        "external_request_id",
        "profile_expires_at",
        "profile_as_of",
        "profile_display_name",
        "profile_version",
        "profile_id",
        "provider",
        "delivery_target",
    ):
        op.drop_column("purchase_requests", column)
