"""Add immutable tenant imports for project-authoritative model candidates."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0078_enterprise_model_asset_imports"
down_revision: str | None = "0077_enterprise_model_release_onboarding"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "enterprise_model_asset_imports",
        sa.Column("import_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("component", sa.String(length=16), nullable=False),
        sa.Column("source_candidate_experiment_id", sa.String(length=128), nullable=False),
        sa.Column(
            "imported_candidate_experiment_id",
            sa.String(length=128),
            sa.ForeignKey("training_experiments.experiment_id"),
            nullable=False,
        ),
        sa.Column(
            "imported_evaluation_id",
            sa.String(length=128),
            sa.ForeignKey("model_evaluation_runs.evaluation_id"),
            nullable=False,
        ),
        sa.Column(
            "imported_suite_id",
            sa.String(length=128),
            sa.ForeignKey("evaluation_suites.suite_id"),
            nullable=False,
        ),
        sa.Column("source_paths_json", sa.JSON(), nullable=False),
        sa.Column("source_file_hashes_json", sa.JSON(), nullable=False),
        sa.Column("source_evidence_chain_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_classification", sa.String(length=64), nullable=False),
        sa.Column("source_decision", sa.String(length=128), nullable=False),
        sa.Column("operational_classification", sa.String(length=64), nullable=False),
        sa.Column("actual_sample_count", sa.Integer(), nullable=False),
        sa.Column("source_artifact_size_recorded", sa.Boolean(), nullable=False),
        sa.Column("release_scope", sa.String(length=32), nullable=False),
        sa.Column("imported_records_json", sa.JSON(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("import_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("requested_by_subject_id", sa.String(length=128), nullable=False),
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
            "source_candidate_experiment_id",
            name="uq_enterprise_model_asset_import_source",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "requested_by_subject_id",
            "idempotency_key",
            name="uq_enterprise_model_asset_import_idempotency",
        ),
        sa.CheckConstraint(
            "component IN ('LLM', 'VLM', 'RUL')",
            name="ck_enterprise_model_asset_import_component",
        ),
        sa.CheckConstraint(
            "status IN ('ACTIVE', 'REVOKED')",
            name="ck_enterprise_model_asset_import_status",
        ),
        sa.CheckConstraint(
            "release_scope = 'STAGING_ONLY'",
            name="ck_enterprise_model_asset_import_scope",
        ),
        sa.CheckConstraint(
            "actual_sample_count >= 1 AND version >= 1",
            name="ck_enterprise_model_asset_import_counts",
        ),
    )
    op.create_index(
        "ix_enterprise_model_asset_imports_tenant_id",
        "enterprise_model_asset_imports",
        ["tenant_id"],
    )
    op.create_index(
        "ix_enterprise_model_asset_import_status",
        "enterprise_model_asset_imports",
        ["tenant_id", "status", "component"],
    )
    table = "enterprise_model_asset_imports"
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    policy_name = f"{table}_tenant_isolation"[:63]
    op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY {policy_name} ON {table} USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index(
        "ix_enterprise_model_asset_import_status",
        table_name="enterprise_model_asset_imports",
    )
    op.drop_index(
        "ix_enterprise_model_asset_imports_tenant_id",
        table_name="enterprise_model_asset_imports",
    )
    op.drop_table("enterprise_model_asset_imports")
