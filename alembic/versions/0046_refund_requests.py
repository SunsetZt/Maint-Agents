"""Add internal refund requests created by approved T2 proposals."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0046_refund_requests"
down_revision: str | None = "0045_equipment_control_handoffs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "refund_requests",
        sa.Column("refund_request_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("operation_id", sa.String(128), nullable=False),
        sa.Column("proposal_id", sa.String(128), nullable=False),
        sa.Column("incident_id", sa.String(128), nullable=False),
        sa.Column("work_order_id", sa.String(128), nullable=False),
        sa.Column("asset_id", sa.String(128), nullable=False),
        sa.Column("customer_subject_id", sa.String(128), nullable=False),
        sa.Column("customer_confirmation_update_id", sa.String(128), nullable=False),
        sa.Column("amount_minor", sa.Integer(), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("cost_center", sa.String(64), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("compensation_policy_id", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("requested_by_subject_id", sa.String(128), nullable=False),
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
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.incident_id"]),
        sa.ForeignKeyConstraint(["work_order_id"], ["work_orders.work_order_id"]),
        sa.ForeignKeyConstraint(["asset_id"], ["assets.asset_id"]),
        sa.ForeignKeyConstraint(
            ["customer_confirmation_update_id"],
            ["customer_service_updates.update_id"],
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "operation_id",
            name="uq_refund_request_operation",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "work_order_id",
            name="uq_refund_request_work_order",
        ),
        sa.CheckConstraint("amount_minor > 0", name="ck_refund_request_amount"),
        sa.CheckConstraint(
            "status = 'SUBMITTED'",
            name="ck_refund_request_status",
        ),
    )
    op.create_index(
        "ix_refund_requests_tenant_id",
        "refund_requests",
        ["tenant_id"],
    )
    op.create_index(
        "ix_refund_requests_incident",
        "refund_requests",
        ["tenant_id", "incident_id", "created_at"],
    )
    op.create_index(
        "ix_refund_requests_status",
        "refund_requests",
        ["tenant_id", "status", "created_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE refund_requests ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE refund_requests FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY refund_requests_tenant_isolation ON refund_requests "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index("ix_refund_requests_status", table_name="refund_requests")
    op.drop_index("ix_refund_requests_incident", table_name="refund_requests")
    op.drop_index("ix_refund_requests_tenant_id", table_name="refund_requests")
    op.drop_table("refund_requests")
