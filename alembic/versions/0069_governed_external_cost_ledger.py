"""Add immutable tenant cost ledger batches and entries."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0069_governed_external_cost_ledger"
down_revision: str | None = "0068_restore_tenant_rls_coverage"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "cost_ledger_batches",
        sa.Column("batch_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("source_system", sa.String(length=128), nullable=False),
        sa.Column("external_batch_id", sa.String(length=128), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("coverage_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("coverage_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("covered_categories_json", sa.JSON(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("evidence_uri", sa.Text(), nullable=False),
        sa.Column("evidence_sha256", sa.String(length=71), nullable=False),
        sa.Column("batch_digest", sa.String(length=71), nullable=False),
        sa.Column("entry_count", sa.Integer(), nullable=False),
        sa.Column("total_amount_usd", sa.Float(), nullable=False),
        sa.Column("imported_by_subject_id", sa.String(length=128), nullable=False),
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
            "imported_by_subject_id",
            "idempotency_key",
            name="uq_cost_ledger_batch_idempotency",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "source_system",
            "external_batch_id",
            name="uq_cost_ledger_batch_external",
        ),
        sa.CheckConstraint("currency = 'USD'", name="ck_cost_ledger_batch_currency"),
        sa.CheckConstraint(
            "coverage_end > coverage_start",
            name="ck_cost_ledger_batch_coverage",
        ),
        sa.CheckConstraint("entry_count > 0", name="ck_cost_ledger_batch_entry_count"),
        sa.CheckConstraint(
            "total_amount_usd >= 0",
            name="ck_cost_ledger_batch_total_amount",
        ),
    )
    op.create_index(
        "ix_cost_ledger_batches_tenant_id",
        "cost_ledger_batches",
        ["tenant_id"],
    )
    op.create_index(
        "ix_cost_ledger_batch_coverage",
        "cost_ledger_batches",
        ["tenant_id", "coverage_start", "coverage_end"],
    )

    op.create_table(
        "cost_ledger_entries",
        sa.Column("entry_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "batch_id",
            sa.String(length=128),
            sa.ForeignKey("cost_ledger_batches.batch_id"),
            nullable=False,
        ),
        sa.Column("external_entry_id", sa.String(length=128), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("amount_usd", sa.Float(), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=True),
        sa.Column("unit", sa.String(length=32), nullable=True),
        sa.Column("release_id", sa.String(length=128), nullable=True),
        sa.Column("scenario", sa.String(length=64), nullable=True),
        sa.Column("diagnosis_run_id", sa.String(length=128), nullable=True),
        sa.Column("work_order_id", sa.String(length=128), nullable=True),
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
            "batch_id",
            "external_entry_id",
            name="uq_cost_ledger_entry_external",
        ),
        sa.CheckConstraint("amount_usd >= 0", name="ck_cost_ledger_entry_amount"),
        sa.CheckConstraint(
            "quantity IS NULL OR quantity >= 0",
            name="ck_cost_ledger_entry_quantity",
        ),
    )
    op.create_index(
        "ix_cost_ledger_entries_tenant_id",
        "cost_ledger_entries",
        ["tenant_id"],
    )
    op.create_index(
        "ix_cost_ledger_entries_batch_id",
        "cost_ledger_entries",
        ["batch_id"],
    )
    op.create_index(
        "ix_cost_ledger_entry_window",
        "cost_ledger_entries",
        ["tenant_id", "occurred_at", "category"],
    )

    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table in ("cost_ledger_batches", "cost_ledger_entries"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    op.drop_index("ix_cost_ledger_entry_window", table_name="cost_ledger_entries")
    op.drop_index("ix_cost_ledger_entries_batch_id", table_name="cost_ledger_entries")
    op.drop_index("ix_cost_ledger_entries_tenant_id", table_name="cost_ledger_entries")
    op.drop_table("cost_ledger_entries")
    op.drop_index("ix_cost_ledger_batch_coverage", table_name="cost_ledger_batches")
    op.drop_index("ix_cost_ledger_batches_tenant_id", table_name="cost_ledger_batches")
    op.drop_table("cost_ledger_batches")
