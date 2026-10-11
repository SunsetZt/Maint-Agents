"""Add governed service-authorized WorkOrder creation facts."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0072_governed_repair_work_order_authorization"
down_revision: str | None = "0071_governed_service_quotation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "work_orders",
        sa.Column(
            "creation_mode",
            sa.String(length=32),
            nullable=False,
            server_default="PARTS_RESERVATION",
        ),
    )
    op.add_column(
        "work_orders",
        sa.Column("authorization_type", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "work_orders",
        sa.Column("diagnosis_run_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "work_orders",
        sa.Column("diagnosis_version", sa.Integer(), nullable=True),
    )
    op.add_column(
        "work_orders",
        sa.Column("service_quotation_id", sa.String(length=128), nullable=True),
    )
    op.alter_column(
        "work_orders",
        "reservation_id",
        existing_type=sa.String(length=128),
        nullable=True,
    )
    op.create_foreign_key(
        "fk_work_orders_diagnosis_run",
        "work_orders",
        "diagnosis_runs",
        ["diagnosis_run_id"],
        ["diagnosis_run_id"],
    )
    op.create_foreign_key(
        "fk_work_orders_service_quotation",
        "work_orders",
        "service_quotations",
        ["service_quotation_id"],
        ["quotation_id"],
    )
    op.create_unique_constraint(
        "uq_work_order_proposal",
        "work_orders",
        ["tenant_id", "proposal_id"],
    )
    op.create_check_constraint(
        "ck_work_order_creation_mode",
        "work_orders",
        "creation_mode IN ('PARTS_RESERVATION', 'SERVICE_AUTHORIZATION')",
    )
    op.create_check_constraint(
        "ck_work_order_creation_binding",
        "work_orders",
        "(" 
        "creation_mode = 'PARTS_RESERVATION' "
        "AND reservation_id IS NOT NULL "
        "AND authorization_type IS NULL "
        "AND service_quotation_id IS NULL"
        ") OR ("
        "creation_mode = 'SERVICE_AUTHORIZATION' "
        "AND reservation_id IS NULL "
        "AND diagnosis_run_id IS NOT NULL "
        "AND diagnosis_version IS NOT NULL "
        "AND ((authorization_type = 'COVERED_SERVICE' "
        "AND service_quotation_id IS NULL) "
        "OR (authorization_type = 'ACCEPTED_QUOTATION' "
        "AND service_quotation_id IS NOT NULL))"
        ")",
    )


def downgrade() -> None:
    service_authorized_count = int(
        op.get_bind().execute(
            sa.text(
                "SELECT count(*) FROM work_orders "
                "WHERE creation_mode = 'SERVICE_AUTHORIZATION'"
            )
        ).scalar()
        or 0
    )
    if service_authorized_count:
        raise RuntimeError(
            "cannot downgrade while service-authorized WorkOrders exist"
        )
    op.drop_constraint(
        "ck_work_order_creation_binding",
        "work_orders",
        type_="check",
    )
    op.drop_constraint(
        "ck_work_order_creation_mode",
        "work_orders",
        type_="check",
    )
    op.drop_constraint("uq_work_order_proposal", "work_orders", type_="unique")
    op.drop_constraint(
        "fk_work_orders_service_quotation",
        "work_orders",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_work_orders_diagnosis_run",
        "work_orders",
        type_="foreignkey",
    )
    op.alter_column(
        "work_orders",
        "reservation_id",
        existing_type=sa.String(length=128),
        nullable=False,
    )
    op.drop_column("work_orders", "service_quotation_id")
    op.drop_column("work_orders", "diagnosis_version")
    op.drop_column("work_orders", "diagnosis_run_id")
    op.drop_column("work_orders", "authorization_type")
    op.drop_column("work_orders", "creation_mode")
