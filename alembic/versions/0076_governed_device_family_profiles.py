"""Add governed device-family onboarding profiles."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0076_governed_device_family_profiles"
down_revision: str | None = "0075_multi_agent_maintenance_planning"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "device_family_profiles",
        sa.Column("profile_id", sa.String(length=128), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("family_code", sa.String(length=64), nullable=False),
        sa.Column("family_version", sa.Integer(), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("risk_class", sa.String(length=32), nullable=False),
        sa.Column("model_codes_json", sa.JSON(), nullable=False),
        sa.Column("source_system", sa.String(length=128), nullable=False),
        sa.Column("source_record_id", sa.String(length=255), nullable=False),
        sa.Column("source_as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "knowledge_release_id",
            sa.String(length=128),
            sa.ForeignKey("index_releases.release_id"),
            nullable=False,
        ),
        sa.Column(
            "evaluation_suite_id",
            sa.String(length=128),
            sa.ForeignKey("evaluation_suites.suite_id"),
            nullable=False,
        ),
        sa.Column(
            "evaluation_id",
            sa.String(length=128),
            sa.ForeignKey("model_evaluation_runs.evaluation_id"),
            nullable=False,
        ),
        sa.Column("minimum_gold_samples", sa.Integer(), nullable=False),
        sa.Column("readiness_json", sa.JSON(), nullable=False),
        sa.Column("evidence_digest", sa.String(length=128), nullable=False),
        sa.Column("configuration_digest", sa.String(length=128), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("request_hash", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("requested_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("reviewed_by_subject_id", sa.String(length=128), nullable=True),
        sa.Column("review_decision", sa.String(length=32), nullable=True),
        sa.Column("review_reason", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "family_code",
            "family_version",
            name="uq_device_family_profile_version",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "requested_by_subject_id",
            "idempotency_key",
            name="uq_device_family_profile_idempotency",
        ),
        sa.CheckConstraint(
            "status IN ('BLOCKED', 'READY', 'ACTIVE', 'REJECTED', 'RETIRED')",
            name="ck_device_family_profile_status",
        ),
        sa.CheckConstraint(
            "risk_class IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')",
            name="ck_device_family_profile_risk_class",
        ),
        sa.CheckConstraint(
            "family_version >= 1 AND version >= 1 AND minimum_gold_samples >= 1",
            name="ck_device_family_profile_versions",
        ),
    )
    op.create_index(
        "ix_device_family_profiles_tenant_id",
        "device_family_profiles",
        ["tenant_id"],
    )
    op.create_index(
        "ix_device_family_profiles_status",
        "device_family_profiles",
        ["tenant_id", "status", "family_code"],
    )
    op.create_index(
        "uq_active_device_family_code",
        "device_family_profiles",
        ["tenant_id", "family_code"],
        unique=True,
        postgresql_where=sa.text("status = 'ACTIVE'"),
        sqlite_where=sa.text("status = 'ACTIVE'"),
    )

    op.create_table(
        "device_family_model_bindings",
        sa.Column("binding_id", sa.String(length=128), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column(
            "profile_id",
            sa.String(length=128),
            sa.ForeignKey("device_family_profiles.profile_id"),
            nullable=False,
        ),
        sa.Column("family_code", sa.String(length=64), nullable=False),
        sa.Column("profile_version", sa.Integer(), nullable=False),
        sa.Column("model_code", sa.String(length=128), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "profile_id",
            "model_code",
            name="uq_device_family_profile_model",
        ),
        sa.CheckConstraint(
            "profile_version >= 1",
            name="ck_device_family_model_profile_version",
        ),
    )
    op.create_index(
        "ix_device_family_model_bindings_tenant_id",
        "device_family_model_bindings",
        ["tenant_id"],
    )
    op.create_index(
        "ix_device_family_model_profile",
        "device_family_model_bindings",
        ["tenant_id", "profile_id", "is_active"],
    )
    op.create_index(
        "uq_active_device_family_model",
        "device_family_model_bindings",
        ["tenant_id", "model_code"],
        unique=True,
        postgresql_where=sa.text("is_active"),
        sqlite_where=sa.text("is_active = 1"),
    )

    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table in ("device_family_profiles", "device_family_model_bindings"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    op.drop_index("uq_active_device_family_model", table_name="device_family_model_bindings")
    op.drop_index("ix_device_family_model_profile", table_name="device_family_model_bindings")
    op.drop_index(
        "ix_device_family_model_bindings_tenant_id",
        table_name="device_family_model_bindings",
    )
    op.drop_table("device_family_model_bindings")
    op.drop_index("uq_active_device_family_code", table_name="device_family_profiles")
    op.drop_index("ix_device_family_profiles_status", table_name="device_family_profiles")
    op.drop_index("ix_device_family_profiles_tenant_id", table_name="device_family_profiles")
    op.drop_table("device_family_profiles")
