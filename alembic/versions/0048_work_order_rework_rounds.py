"""Add explicit work-order repair rounds and rework evidence boundaries."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0048_work_order_rework_rounds"
down_revision: str | None = "0047_execution_reconciliations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "work_order_completions",
        sa.Column("round_number", sa.Integer(), nullable=True),
    )
    op.add_column(
        "work_order_completions",
        sa.Column("field_entry_sequence_start", sa.Integer(), nullable=True),
    )
    op.add_column(
        "work_order_completions",
        sa.Column("field_entry_sequence_end", sa.Integer(), nullable=True),
    )
    op.execute(
        """
        WITH ranked AS (
          SELECT completion_id,
                 row_number() OVER (
                   PARTITION BY tenant_id, work_order_id
                   ORDER BY completed_at, completion_id
                 ) AS repair_round
          FROM work_order_completions
        )
        UPDATE work_order_completions AS completion
        SET round_number = ranked.repair_round
        FROM ranked
        WHERE completion.completion_id = ranked.completion_id
        """
    )
    op.alter_column("work_order_completions", "round_number", nullable=False)
    op.create_unique_constraint(
        "uq_work_order_completion_round",
        "work_order_completions",
        ["tenant_id", "work_order_id", "round_number"],
    )
    op.create_check_constraint(
        "ck_work_order_completion_round",
        "work_order_completions",
        "round_number > 0",
    )

    op.add_column(
        "work_order_verifications",
        sa.Column("completion_id", sa.String(128), nullable=True),
    )
    op.add_column(
        "work_order_verifications",
        sa.Column("round_number", sa.Integer(), nullable=True),
    )
    op.execute(
        """
        UPDATE work_order_verifications AS verification
        SET completion_id = (
          SELECT completion.completion_id
          FROM work_order_completions AS completion
          WHERE completion.tenant_id = verification.tenant_id
            AND completion.work_order_id = verification.work_order_id
            AND completion.completed_at <= verification.verified_at
          ORDER BY completion.completed_at DESC, completion.completion_id DESC
          LIMIT 1
        ),
            round_number = (
          SELECT completion.round_number
          FROM work_order_completions AS completion
          WHERE completion.tenant_id = verification.tenant_id
            AND completion.work_order_id = verification.work_order_id
            AND completion.completed_at <= verification.verified_at
          ORDER BY completion.completed_at DESC, completion.completion_id DESC
          LIMIT 1
        )
        """
    )
    op.alter_column("work_order_verifications", "completion_id", nullable=False)
    op.alter_column("work_order_verifications", "round_number", nullable=False)
    op.create_foreign_key(
        "fk_work_order_verification_completion",
        "work_order_verifications",
        "work_order_completions",
        ["completion_id"],
        ["completion_id"],
    )
    op.create_unique_constraint(
        "uq_work_order_verification_completion",
        "work_order_verifications",
        ["tenant_id", "completion_id"],
    )
    op.create_check_constraint(
        "ck_work_order_verification_round",
        "work_order_verifications",
        "round_number > 0",
    )

    op.create_table(
        "work_order_reworks",
        sa.Column("rework_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("work_order_id", sa.String(128), nullable=False),
        sa.Column("source_completion_id", sa.String(128), nullable=False),
        sa.Column("failed_verification_id", sa.String(128), nullable=False),
        sa.Column("resolved_completion_id", sa.String(128), nullable=True),
        sa.Column("round_number", sa.Integer(), nullable=False),
        sa.Column("entry_sequence_checkpoint", sa.Integer(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(["work_order_id"], ["work_orders.work_order_id"]),
        sa.ForeignKeyConstraint(
            ["source_completion_id"],
            ["work_order_completions.completion_id"],
        ),
        sa.ForeignKeyConstraint(
            ["failed_verification_id"],
            ["work_order_verifications.verification_id"],
        ),
        sa.ForeignKeyConstraint(
            ["resolved_completion_id"],
            ["work_order_completions.completion_id"],
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "failed_verification_id",
            name="uq_work_order_rework_verification",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "work_order_id",
            "round_number",
            name="uq_work_order_rework_round",
        ),
        sa.CheckConstraint("round_number > 1", name="ck_work_order_rework_round"),
        sa.CheckConstraint(
            "status IN ('OPEN', 'RESUBMITTED')",
            name="ck_work_order_rework_status",
        ),
    )
    op.create_index(
        "ix_work_order_reworks_tenant_id",
        "work_order_reworks",
        ["tenant_id"],
    )
    op.create_index(
        "ix_work_order_reworks_work_order",
        "work_order_reworks",
        ["tenant_id", "work_order_id", "opened_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE work_order_reworks ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE work_order_reworks FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY work_order_reworks_tenant_isolation ON work_order_reworks "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index("ix_work_order_reworks_work_order", table_name="work_order_reworks")
    op.drop_index("ix_work_order_reworks_tenant_id", table_name="work_order_reworks")
    op.drop_table("work_order_reworks")
    op.drop_constraint(
        "ck_work_order_verification_round",
        "work_order_verifications",
        type_="check",
    )
    op.drop_constraint(
        "uq_work_order_verification_completion",
        "work_order_verifications",
        type_="unique",
    )
    op.drop_constraint(
        "fk_work_order_verification_completion",
        "work_order_verifications",
        type_="foreignkey",
    )
    op.drop_column("work_order_verifications", "round_number")
    op.drop_column("work_order_verifications", "completion_id")
    op.drop_constraint(
        "ck_work_order_completion_round",
        "work_order_completions",
        type_="check",
    )
    op.drop_constraint(
        "uq_work_order_completion_round",
        "work_order_completions",
        type_="unique",
    )
    op.drop_column("work_order_completions", "field_entry_sequence_end")
    op.drop_column("work_order_completions", "field_entry_sequence_start")
    op.drop_column("work_order_completions", "round_number")
