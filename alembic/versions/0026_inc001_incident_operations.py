"""Add INC-001 incident intake operations and append-only decision evidence."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0026_inc001_incident_operations"
down_revision: str | None = "0025_portal001_customer_service"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("incidents", sa.Column("severity", sa.String(length=16), nullable=True))
    op.add_column("incidents", sa.Column("category", sa.String(length=64), nullable=True))
    op.add_column("incidents", sa.Column("responsible_queue", sa.String(length=128), nullable=True))
    op.add_column(
        "incidents",
        sa.Column("information_due_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "incidents",
        sa.Column(
            "escalation_responsible_subject_id",
            sa.String(length=128),
            nullable=True,
        ),
    )

    op.create_table(
        "incident_controls",
        sa.Column("control_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "incident_id",
            sa.String(length=128),
            sa.ForeignKey("incidents.incident_id"),
            nullable=False,
        ),
        sa.Column("incident_version", sa.Integer(), nullable=False),
        sa.Column("command_type", sa.String(length=32), nullable=False),
        sa.Column("previous_status", sa.String(length=32), nullable=False),
        sa.Column("target_status", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("actor_subject_id", sa.String(length=128), nullable=False),
        sa.Column("responsible_subject_id", sa.String(length=128), nullable=True),
        sa.Column("recovery_condition", sa.Text(), nullable=True),
        sa.Column("related_incident_id", sa.String(length=128), nullable=True),
        sa.Column("details_json", sa.JSON(), nullable=False),
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
            "incident_id",
            "incident_version",
            name="uq_incident_control_version",
        ),
    )
    op.create_index("ix_incident_controls_tenant_id", "incident_controls", ["tenant_id"])
    op.create_index(
        "ix_incident_controls_incident",
        "incident_controls",
        ["tenant_id", "incident_id", "occurred_at"],
    )

    op.create_table(
        "incident_relations",
        sa.Column("relation_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "source_incident_id",
            sa.String(length=128),
            sa.ForeignKey("incidents.incident_id"),
            nullable=False,
        ),
        sa.Column(
            "target_incident_id",
            sa.String(length=128),
            sa.ForeignKey("incidents.incident_id"),
            nullable=False,
        ),
        sa.Column("relation_type", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("created_by_subject_id", sa.String(length=128), nullable=False),
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
            "source_incident_id",
            "relation_type",
            name="uq_incident_relation_source_type",
        ),
    )
    op.create_index("ix_incident_relations_tenant_id", "incident_relations", ["tenant_id"])
    op.create_index(
        "ix_incident_relations_target",
        "incident_relations",
        ["tenant_id", "target_incident_id"],
    )

    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table in ("incident_controls", "incident_relations"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    op.drop_index("ix_incident_relations_target", table_name="incident_relations")
    op.drop_index("ix_incident_relations_tenant_id", table_name="incident_relations")
    op.drop_table("incident_relations")
    op.drop_index("ix_incident_controls_incident", table_name="incident_controls")
    op.drop_index("ix_incident_controls_tenant_id", table_name="incident_controls")
    op.drop_table("incident_controls")
    op.drop_column("incidents", "escalation_responsible_subject_id")
    op.drop_column("incidents", "information_due_at")
    op.drop_column("incidents", "responsible_queue")
    op.drop_column("incidents", "category")
    op.drop_column("incidents", "severity")
