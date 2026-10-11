"""Add IAM-005 tenant membership, access and retention administration."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0027_iam005_tenant_administration"
down_revision: str | None = "0026_inc001_incident_operations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("tenants", sa.Column("display_name", sa.String(length=255), nullable=True))
    op.add_column(
        "tenants",
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column("subjects", sa.Column("display_name", sa.String(length=255), nullable=True))
    op.add_column("subjects", sa.Column("email", sa.String(length=320), nullable=True))
    op.add_column(
        "subjects",
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
    )

    op.create_table(
        "tenant_retention_policies",
        sa.Column("policy_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("incident_days", sa.Integer(), nullable=False),
        sa.Column("media_days", sa.Integer(), nullable=False),
        sa.Column("knowledge_days", sa.Integer(), nullable=False),
        sa.Column("training_data_days", sa.Integer(), nullable=False),
        sa.Column("audit_days", sa.Integer(), nullable=False),
        sa.Column("legal_hold", sa.Boolean(), nullable=False),
        sa.Column("updated_by_subject_id", sa.String(length=128), nullable=False),
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
        sa.UniqueConstraint("tenant_id", name="uq_tenant_retention_policy"),
    )
    op.create_index(
        "ix_tenant_retention_policies_tenant_id",
        "tenant_retention_policies",
        ["tenant_id"],
    )

    op.create_table(
        "tenant_administration_records",
        sa.Column("administration_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("client_operation_id", sa.String(length=128), nullable=False),
        sa.Column("command_type", sa.String(length=32), nullable=False),
        sa.Column("target_type", sa.String(length=32), nullable=False),
        sa.Column("target_id", sa.String(length=128), nullable=False),
        sa.Column("actor_subject_id", sa.String(length=128), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("previous_version", sa.Integer(), nullable=False),
        sa.Column("target_version", sa.Integer(), nullable=False),
        sa.Column("before_json", sa.JSON(), nullable=False),
        sa.Column("after_json", sa.JSON(), nullable=False),
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
            "actor_subject_id",
            "client_operation_id",
            name="uq_tenant_administration_operation",
        ),
    )
    op.create_index(
        "ix_tenant_administration_records_tenant_id",
        "tenant_administration_records",
        ["tenant_id"],
    )
    op.create_index(
        "ix_tenant_administration_occurred",
        "tenant_administration_records",
        ["tenant_id", "occurred_at"],
    )

    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table in ("tenant_retention_policies", "tenant_administration_records"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    op.drop_index(
        "ix_tenant_administration_occurred",
        table_name="tenant_administration_records",
    )
    op.drop_index(
        "ix_tenant_administration_records_tenant_id",
        table_name="tenant_administration_records",
    )
    op.drop_table("tenant_administration_records")
    op.drop_index(
        "ix_tenant_retention_policies_tenant_id",
        table_name="tenant_retention_policies",
    )
    op.drop_table("tenant_retention_policies")
    op.drop_column("subjects", "version")
    op.drop_column("subjects", "email")
    op.drop_column("subjects", "display_name")
    op.drop_column("tenants", "version")
    op.drop_column("tenants", "display_name")
