"""Add human-confirmed and revocable diagnosis memory governance."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0031_governed_memory"
down_revision: str | None = "0030_context_manifest"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "governed_memories",
        sa.Column("memory_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("memory_type", sa.String(length=32), nullable=False),
        sa.Column("purpose", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("owner_subject_id", sa.String(length=128), nullable=False),
        sa.Column(
            "incident_id",
            sa.String(length=128),
            sa.ForeignKey("incidents.incident_id"),
            nullable=True,
        ),
        sa.Column(
            "asset_id",
            sa.String(length=128),
            sa.ForeignKey("assets.asset_id"),
            nullable=True,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("source_reference_type", sa.String(length=32), nullable=False),
        sa.Column("source_reference_id", sa.String(length=255), nullable=False),
        sa.Column("confirmed_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("revoked_by_subject_id", sa.String(length=128), nullable=True),
        sa.Column("revocation_reason", sa.Text(), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_governed_memory_idempotency",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "memory_type",
            "source_reference_id",
            name="uq_governed_memory_source_reference",
        ),
    )
    op.create_index("ix_governed_memories_tenant_id", "governed_memories", ["tenant_id"])
    op.create_index(
        "ix_governed_memory_incident_status",
        "governed_memories",
        ["tenant_id", "incident_id", "status"],
    )
    op.create_index(
        "ix_governed_memory_owner_status",
        "governed_memories",
        ["tenant_id", "owner_subject_id", "status"],
    )
    op.create_table(
        "governed_memory_transitions",
        sa.Column("transition_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "memory_id",
            sa.String(length=128),
            sa.ForeignKey("governed_memories.memory_id"),
            nullable=False,
        ),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("from_status", sa.String(length=32), nullable=False),
        sa.Column("to_status", sa.String(length=32), nullable=False),
        sa.Column("actor_subject_id", sa.String(length=128), nullable=False),
        sa.Column("reason_code", sa.String(length=128), nullable=False),
        sa.Column("evidence_hash", sa.String(length=128), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
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
            "memory_id",
            "sequence",
            name="uq_governed_memory_transition_sequence",
        ),
    )
    op.create_index(
        "ix_governed_memory_transitions_tenant_id",
        "governed_memory_transitions",
        ["tenant_id"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table in ("governed_memories", "governed_memory_transitions"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    op.drop_index(
        "ix_governed_memory_transitions_tenant_id",
        table_name="governed_memory_transitions",
    )
    op.drop_table("governed_memory_transitions")
    op.drop_index("ix_governed_memory_owner_status", table_name="governed_memories")
    op.drop_index("ix_governed_memory_incident_status", table_name="governed_memories")
    op.drop_index("ix_governed_memories_tenant_id", table_name="governed_memories")
    op.drop_table("governed_memories")
