"""Bind approved Agent assignment executions to assignment history."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0044_agent_assignment_execution"
down_revision: str | None = "0043_purchase_requests"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "work_order_assignments",
        sa.Column("operation_id", sa.String(128), nullable=True),
    )
    op.add_column(
        "work_order_assignments",
        sa.Column("proposal_id", sa.String(128), nullable=True),
    )
    op.create_foreign_key(
        "fk_work_order_assignments_proposal",
        "work_order_assignments",
        "action_proposals",
        ["proposal_id"],
        ["proposal_id"],
    )
    op.create_unique_constraint(
        "uq_work_order_assignment_operation",
        "work_order_assignments",
        ["tenant_id", "operation_id"],
    )
    op.create_index(
        "ix_work_order_assignments_proposal",
        "work_order_assignments",
        ["tenant_id", "proposal_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_work_order_assignments_proposal",
        table_name="work_order_assignments",
    )
    op.drop_constraint(
        "uq_work_order_assignment_operation",
        "work_order_assignments",
        type_="unique",
    )
    op.drop_constraint(
        "fk_work_order_assignments_proposal",
        "work_order_assignments",
        type_="foreignkey",
    )
    op.drop_column("work_order_assignments", "proposal_id")
    op.drop_column("work_order_assignments", "operation_id")
