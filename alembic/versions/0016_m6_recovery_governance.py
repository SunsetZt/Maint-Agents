"""Add tenant-scoped backup evidence and recovery drill governance."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0016_m6_recovery_governance"
down_revision: str | None = "0015_m6_supply_chain_evidence"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "recovery_evidence",
        sa.Column("evidence_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("component", sa.String(length=64), nullable=False),
        sa.Column("backup_id", sa.String(length=255), nullable=False),
        sa.Column("backup_type", sa.String(length=64), nullable=False),
        sa.Column("recovery_point_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("verification_status", sa.String(length=32), nullable=False),
        sa.Column("verification_method", sa.String(length=64), nullable=False),
        sa.Column("evidence_ref", sa.String(length=255), nullable=False),
        sa.Column("artifact_digest", sa.String(length=128), nullable=False),
        sa.Column("immutable", sa.Boolean(), nullable=False),
        sa.Column("encrypted", sa.Boolean(), nullable=False),
        sa.Column("details_json", sa.JSON(), nullable=False),
        sa.Column("verified_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=False),
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
            "component",
            "backup_id",
            name="uq_recovery_evidence_backup",
        ),
    )
    op.create_index(
        "ix_recovery_evidence_tenant_id",
        "recovery_evidence",
        ["tenant_id"],
    )
    op.create_index(
        "ix_recovery_evidence_latest",
        "recovery_evidence",
        ["tenant_id", "component", "verified_at"],
    )

    op.create_table(
        "recovery_drills",
        sa.Column("drill_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("request_hash", sa.String(length=128), nullable=False),
        sa.Column("scenario", sa.String(length=128), nullable=False),
        sa.Column("scope_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("outage_detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("results_json", sa.JSON(), nullable=False),
        sa.Column("blocker_codes_json", sa.JSON(), nullable=False),
        sa.Column("requested_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("completed_by_subject_id", sa.String(length=128), nullable=True),
        sa.Column("objective_met", sa.Boolean(), nullable=False),
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
            name="uq_recovery_drill_idempotency",
        ),
    )
    op.create_index(
        "ix_recovery_drills_tenant_id",
        "recovery_drills",
        ["tenant_id"],
    )
    op.create_index(
        "ix_recovery_drill_status",
        "recovery_drills",
        ["tenant_id", "status", "completed_at"],
    )

    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table in ("recovery_evidence", "recovery_drills"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    op.drop_index("ix_recovery_drill_status", table_name="recovery_drills")
    op.drop_index("ix_recovery_drills_tenant_id", table_name="recovery_drills")
    op.drop_table("recovery_drills")
    op.drop_index("ix_recovery_evidence_latest", table_name="recovery_evidence")
    op.drop_index("ix_recovery_evidence_tenant_id", table_name="recovery_evidence")
    op.drop_table("recovery_evidence")
