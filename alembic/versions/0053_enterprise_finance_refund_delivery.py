"""Add enterprise finance refund handoff, payment state and receipt facts."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0054_enterprise_finance_refund_delivery"
down_revision: str | None = "0053_diagnosis_quality_feedback"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    columns = (
        sa.Column("reason_code", sa.String(64), nullable=True),
        sa.Column(
            "delivery_target",
            sa.String(32),
            nullable=False,
            server_default=sa.text("'INTERNAL'"),
        ),
        sa.Column("provider", sa.String(128), nullable=True),
        sa.Column("profile_id", sa.String(128), nullable=True),
        sa.Column("profile_version", sa.String(128), nullable=True),
        sa.Column("profile_display_name", sa.String(128), nullable=True),
        sa.Column("profile_as_of", sa.DateTime(timezone=True), nullable=True),
        sa.Column("profile_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("external_request_id", sa.String(255), nullable=True),
        sa.Column(
            "payment_status",
            sa.String(32),
            nullable=False,
            server_default=sa.text("'NOT_APPLICABLE'"),
        ),
        sa.Column("provider_reason", sa.String(255), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("terminal_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_receipt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
    )
    for column in columns:
        op.add_column("refund_requests", column)
    op.create_check_constraint(
        "ck_refund_request_delivery_target",
        "refund_requests",
        "delivery_target IN ('INTERNAL', 'ENTERPRISE_FINANCE')",
    )
    op.drop_constraint(
        "ck_refund_request_status",
        "refund_requests",
        type_="check",
    )
    op.create_check_constraint(
        "ck_refund_request_status",
        "refund_requests",
        "status IN ('SUBMITTED', 'SUBMITTING', 'RECONCILING', "
        "'ACCEPTED', 'REJECTED', 'CANCELLED')",
    )
    op.create_check_constraint(
        "ck_refund_request_payment_status",
        "refund_requests",
        "payment_status IN ('NOT_APPLICABLE', 'NOT_STARTED', 'PROCESSING', "
        "'PAID', 'FAILED')",
    )
    op.create_check_constraint(
        "ck_refund_request_version", "refund_requests", "version > 0"
    )
    op.alter_column("refund_requests", "delivery_target", server_default=None)
    op.alter_column("refund_requests", "payment_status", server_default=None)
    op.alter_column("refund_requests", "version", server_default=None)

    op.create_table(
        "refund_request_receipts",
        sa.Column("receipt_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("refund_request_id", sa.String(128), nullable=False),
        sa.Column("provider", sa.String(128), nullable=False),
        sa.Column("provider_event_id", sa.String(255), nullable=False),
        sa.Column("operation_id", sa.String(128), nullable=False),
        sa.Column("external_request_id", sa.String(255), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("payment_status", sa.String(32), nullable=False),
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
            ["refund_request_id"], ["refund_requests.refund_request_id"]
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "provider",
            "provider_event_id",
            name="uq_refund_request_receipt_event",
        ),
    )
    op.create_index(
        "ix_refund_request_receipts_tenant_id",
        "refund_request_receipts",
        ["tenant_id"],
    )
    op.create_index(
        "ix_refund_request_receipts_request",
        "refund_request_receipts",
        ["tenant_id", "refund_request_id", "occurred_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE refund_request_receipts ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE refund_request_receipts FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY refund_request_receipts_tenant_isolation "
        "ON refund_request_receipts "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index(
        "ix_refund_request_receipts_request", table_name="refund_request_receipts"
    )
    op.drop_index(
        "ix_refund_request_receipts_tenant_id", table_name="refund_request_receipts"
    )
    op.drop_table("refund_request_receipts")
    for constraint in (
        "ck_refund_request_version",
        "ck_refund_request_payment_status",
        "ck_refund_request_status",
        "ck_refund_request_delivery_target",
    ):
        op.drop_constraint(constraint, "refund_requests", type_="check")
    op.create_check_constraint(
        "ck_refund_request_status",
        "refund_requests",
        "status = 'SUBMITTED'",
    )
    for column in (
        "version",
        "last_receipt_at",
        "terminal_at",
        "submitted_at",
        "provider_reason",
        "payment_status",
        "external_request_id",
        "profile_expires_at",
        "profile_as_of",
        "profile_display_name",
        "profile_version",
        "profile_id",
        "provider",
        "delivery_target",
        "reason_code",
    ):
        op.drop_column("refund_requests", column)
