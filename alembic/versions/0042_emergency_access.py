"""Add tenant-bound break-glass grants, decisions, and usage evidence."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0042_emergency_access"
down_revision: str | None = "0041_event_trace_context"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "emergency_access_grants",
        sa.Column("grant_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("incident_number", sa.String(128), nullable=False),
        sa.Column("requester_subject_id", sa.String(128), nullable=False),
        sa.Column("action", sa.String(128), nullable=False),
        sa.Column("resource_id", sa.String(128), nullable=False),
        sa.Column("justification", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("requested_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("approver_subject_id", sa.String(128), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_by_subject_id", sa.String(128), nullable=True),
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
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
    )
    op.create_index(
        "ix_emergency_access_grants_tenant_id",
        "emergency_access_grants",
        ["tenant_id"],
    )
    op.create_index(
        "ix_emergency_access_grants_status_expiry",
        "emergency_access_grants",
        ["tenant_id", "status", "requested_expires_at"],
    )
    op.create_table(
        "emergency_access_decisions",
        sa.Column("decision_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("grant_id", sa.String(128), nullable=False),
        sa.Column("decision", sa.String(32), nullable=False),
        sa.Column("actor_subject_id", sa.String(128), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
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
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(["grant_id"], ["emergency_access_grants.grant_id"]),
    )
    op.create_index(
        "ix_emergency_access_decisions_tenant_id",
        "emergency_access_decisions",
        ["tenant_id"],
    )
    op.create_index(
        "ix_emergency_access_decisions_grant_id",
        "emergency_access_decisions",
        ["grant_id"],
    )
    op.create_table(
        "emergency_access_usage",
        sa.Column("usage_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("grant_id", sa.String(128), nullable=False),
        sa.Column("subject_id", sa.String(128), nullable=False),
        sa.Column("action", sa.String(128), nullable=False),
        sa.Column("resource_id", sa.String(128), nullable=False),
        sa.Column("request_id", sa.String(128), nullable=False),
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
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(["grant_id"], ["emergency_access_grants.grant_id"]),
        sa.UniqueConstraint(
            "tenant_id",
            "grant_id",
            "request_id",
            "action",
            "resource_id",
            name="uq_emergency_access_usage_request",
        ),
    )
    op.create_index(
        "ix_emergency_access_usage_tenant_id",
        "emergency_access_usage",
        ["tenant_id"],
    )
    op.create_index(
        "ix_emergency_access_usage_grant_id",
        "emergency_access_usage",
        ["grant_id"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table_name in (
        "emergency_access_grants",
        "emergency_access_decisions",
        "emergency_access_usage",
    ):
        op.execute(f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table_name} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table_name}_tenant_isolation ON {table_name} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    op.drop_index("ix_emergency_access_usage_grant_id", table_name="emergency_access_usage")
    op.drop_index("ix_emergency_access_usage_tenant_id", table_name="emergency_access_usage")
    op.drop_table("emergency_access_usage")
    op.drop_index(
        "ix_emergency_access_decisions_grant_id",
        table_name="emergency_access_decisions",
    )
    op.drop_index(
        "ix_emergency_access_decisions_tenant_id",
        table_name="emergency_access_decisions",
    )
    op.drop_table("emergency_access_decisions")
    op.drop_index(
        "ix_emergency_access_grants_status_expiry",
        table_name="emergency_access_grants",
    )
    op.drop_index(
        "ix_emergency_access_grants_tenant_id",
        table_name="emergency_access_grants",
    )
    op.drop_table("emergency_access_grants")
