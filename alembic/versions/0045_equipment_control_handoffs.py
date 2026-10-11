"""Persist evidence-only T3 equipment control handoffs."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0045_equipment_control_handoffs"
down_revision: str | None = "0044_agent_assignment_execution"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "equipment_control_handoffs",
        sa.Column("handoff_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("client_operation_id", sa.String(128), nullable=False),
        sa.Column("request_digest", sa.String(128), nullable=False),
        sa.Column("incident_id", sa.String(128), nullable=False),
        sa.Column("asset_id", sa.String(128), nullable=False),
        sa.Column("tool_call_id", sa.String(128), nullable=False),
        sa.Column("requested_tool_id", sa.String(128), nullable=False),
        sa.Column("requested_operation", sa.String(64), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("evidence_ids", sa.JSON(), nullable=False),
        sa.Column("target_system", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("requested_by_subject_id", sa.String(128), nullable=False),
        sa.Column("incident_version", sa.Integer(), nullable=False),
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
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.incident_id"]),
        sa.ForeignKeyConstraint(["asset_id"], ["assets.asset_id"]),
        sa.ForeignKeyConstraint(["tool_call_id"], ["tool_calls.tool_call_id"]),
        sa.UniqueConstraint(
            "tenant_id",
            "client_operation_id",
            name="uq_equipment_control_handoff_operation",
        ),
        sa.CheckConstraint("incident_version > 0", name="ck_equipment_handoff_version"),
        sa.CheckConstraint(
            "status = 'EXTERNAL_REVIEW_REQUIRED'",
            name="ck_equipment_handoff_status",
        ),
    )
    op.create_index(
        "ix_equipment_control_handoffs_tenant_id",
        "equipment_control_handoffs",
        ["tenant_id"],
    )
    op.create_index(
        "ix_equipment_control_handoffs_incident",
        "equipment_control_handoffs",
        ["tenant_id", "incident_id", "created_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE equipment_control_handoffs ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE equipment_control_handoffs FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY equipment_control_handoffs_tenant_isolation "
        "ON equipment_control_handoffs "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index(
        "ix_equipment_control_handoffs_incident",
        table_name="equipment_control_handoffs",
    )
    op.drop_index(
        "ix_equipment_control_handoffs_tenant_id",
        table_name="equipment_control_handoffs",
    )
    op.drop_table("equipment_control_handoffs")
