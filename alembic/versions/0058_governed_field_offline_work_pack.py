"""Add governed, expiring field offline work packs."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0060_governed_field_offline_work_pack"
down_revision: str | None = "0059_reviewed_graphrag_extraction"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "work_order_offline_packs",
        sa.Column("pack_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("subject_id", sa.String(128), nullable=False),
        sa.Column("work_order_id", sa.String(128), nullable=False),
        sa.Column("work_order_version", sa.Integer(), nullable=False),
        sa.Column("asset_id", sa.String(128), nullable=False),
        sa.Column("asset_version", sa.Integer(), nullable=False),
        sa.Column("assignment_id", sa.String(128), nullable=False),
        sa.Column("assignment_assigned_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("schema_version", sa.String(64), nullable=False),
        sa.Column("snapshot_json", sa.JSON(), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("request_digest", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_by_subject_id", sa.String(128), nullable=True),
        sa.Column("revocation_reason", sa.Text(), nullable=True),
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
            "status IN ('ACTIVE', 'SUPERSEDED', 'REVOKED')",
            name="ck_work_order_offline_pack_status",
        ),
        sa.CheckConstraint("version > 0", name="ck_work_order_offline_pack_version"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(["work_order_id"], ["work_orders.work_order_id"]),
        sa.ForeignKeyConstraint(["asset_id"], ["assets.asset_id"]),
        sa.ForeignKeyConstraint(
            ["assignment_id"], ["work_order_assignments.assignment_id"]
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "subject_id",
            "idempotency_key",
            name="uq_work_order_offline_pack_idempotency",
        ),
    )
    op.create_index(
        "ix_work_order_offline_packs_tenant_id",
        "work_order_offline_packs",
        ["tenant_id"],
    )
    op.create_index(
        "ix_work_order_offline_pack_current",
        "work_order_offline_packs",
        ["tenant_id", "subject_id", "work_order_id", "status", "issued_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE work_order_offline_packs ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE work_order_offline_packs FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY work_order_offline_packs_tenant_isolation "
        "ON work_order_offline_packs "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.execute(
        "DROP POLICY IF EXISTS work_order_offline_packs_tenant_isolation "
        "ON work_order_offline_packs"
    )
    op.drop_index(
        "ix_work_order_offline_pack_current",
        table_name="work_order_offline_packs",
    )
    op.drop_index(
        "ix_work_order_offline_packs_tenant_id",
        table_name="work_order_offline_packs",
    )
    op.drop_table("work_order_offline_packs")
