"""Add immutable incident resolution facts."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0049_incident_resolution_lifecycle"
down_revision: str | None = "0048_work_order_rework_rounds"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "incident_resolutions",
        sa.Column("resolution_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("incident_id", sa.String(128), nullable=False),
        sa.Column("mode", sa.String(16), nullable=False),
        sa.Column("diagnosis_run_id", sa.String(128), nullable=True),
        sa.Column("work_order_id", sa.String(128), nullable=True),
        sa.Column("completion_id", sa.String(128), nullable=True),
        sa.Column("verification_id", sa.String(128), nullable=True),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("evidence_ids", sa.JSON(), nullable=False),
        sa.Column("resolved_by_subject_id", sa.String(128), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=False),
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
        sa.ForeignKeyConstraint(
            ["diagnosis_run_id"], ["diagnosis_runs.diagnosis_run_id"]
        ),
        sa.ForeignKeyConstraint(["work_order_id"], ["work_orders.work_order_id"]),
        sa.ForeignKeyConstraint(
            ["completion_id"], ["work_order_completions.completion_id"]
        ),
        sa.ForeignKeyConstraint(
            ["verification_id"], ["work_order_verifications.verification_id"]
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "incident_id",
            name="uq_incident_resolution_incident",
        ),
        sa.CheckConstraint(
            "(mode = 'REMOTE' AND diagnosis_run_id IS NOT NULL "
            "AND work_order_id IS NULL AND completion_id IS NULL "
            "AND verification_id IS NULL) OR "
            "(mode = 'WORK_ORDER' AND work_order_id IS NOT NULL "
            "AND completion_id IS NOT NULL AND verification_id IS NOT NULL)",
            name="ck_incident_resolution_source",
        ),
    )
    op.create_index(
        "ix_incident_resolutions_tenant_id",
        "incident_resolutions",
        ["tenant_id"],
    )
    op.create_index(
        "ix_incident_resolutions_incident",
        "incident_resolutions",
        ["tenant_id", "incident_id", "resolved_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE incident_resolutions ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE incident_resolutions FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY incident_resolutions_tenant_isolation ON incident_resolutions "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index(
        "ix_incident_resolutions_incident",
        table_name="incident_resolutions",
    )
    op.drop_index(
        "ix_incident_resolutions_tenant_id",
        table_name="incident_resolutions",
    )
    op.drop_table("incident_resolutions")
