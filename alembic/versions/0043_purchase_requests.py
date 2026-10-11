"""Add tenant-scoped purchase requests created by approved T2 proposals."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0043_purchase_requests"
down_revision: str | None = "0042_emergency_access"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "purchase_requests",
        sa.Column("purchase_request_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("operation_id", sa.String(128), nullable=False),
        sa.Column("proposal_id", sa.String(128), nullable=False),
        sa.Column("incident_id", sa.String(128), nullable=False),
        sa.Column("asset_id", sa.String(128), nullable=False),
        sa.Column("part_number", sa.String(128), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("available_quantity_at_submission", sa.Integer(), nullable=False),
        sa.Column("estimated_unit_cost_minor", sa.Integer(), nullable=False),
        sa.Column("estimated_total_cost_minor", sa.Integer(), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("cost_center", sa.String(64), nullable=False),
        sa.Column("justification", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("requested_by_subject_id", sa.String(128), nullable=False),
        sa.Column("inventory_source", sa.String(128), nullable=False),
        sa.Column("inventory_source_record_id", sa.String(255), nullable=False),
        sa.Column("inventory_as_of", sa.DateTime(timezone=True), nullable=False),
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
        sa.ForeignKeyConstraint(["asset_id"], ["assets.asset_id"]),
        sa.UniqueConstraint(
            "tenant_id",
            "operation_id",
            name="uq_purchase_request_operation",
        ),
        sa.CheckConstraint("quantity > 0", name="ck_purchase_request_quantity"),
        sa.CheckConstraint(
            "available_quantity_at_submission >= 0",
            name="ck_purchase_request_available_quantity",
        ),
        sa.CheckConstraint(
            "estimated_unit_cost_minor >= 0",
            name="ck_purchase_request_unit_cost",
        ),
        sa.CheckConstraint(
            "estimated_total_cost_minor >= 0",
            name="ck_purchase_request_total_cost",
        ),
    )
    op.create_index(
        "ix_purchase_requests_tenant_id",
        "purchase_requests",
        ["tenant_id"],
    )
    op.create_index(
        "ix_purchase_requests_incident",
        "purchase_requests",
        ["tenant_id", "incident_id", "created_at"],
    )
    op.create_index(
        "ix_purchase_requests_status",
        "purchase_requests",
        ["tenant_id", "status", "created_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE purchase_requests ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE purchase_requests FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY purchase_requests_tenant_isolation ON purchase_requests "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index("ix_purchase_requests_status", table_name="purchase_requests")
    op.drop_index("ix_purchase_requests_incident", table_name="purchase_requests")
    op.drop_index("ix_purchase_requests_tenant_id", table_name="purchase_requests")
    op.drop_table("purchase_requests")
