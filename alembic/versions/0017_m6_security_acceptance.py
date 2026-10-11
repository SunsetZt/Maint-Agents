"""Add governed security exercises and production acceptance dossiers."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0017_m6_security_acceptance"
down_revision: str | None = "0016_m6_recovery_governance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> tuple[sa.Column[object], sa.Column[object]]:
    return (
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
    )


def upgrade() -> None:
    op.create_table(
        "security_exercises",
        sa.Column("exercise_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("request_hash", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("target_environment", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("scenario_ids_json", sa.JSON(), nullable=False),
        sa.Column("results_json", sa.JSON(), nullable=False),
        sa.Column("blocker_codes_json", sa.JSON(), nullable=False),
        sa.Column("report_digest", sa.String(length=128), nullable=True),
        sa.Column("verifier_version", sa.String(length=64), nullable=True),
        sa.Column("requested_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("completed_by_subject_id", sa.String(length=128), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_security_exercise_idempotency",
        ),
    )
    op.create_index("ix_security_exercises_tenant_id", "security_exercises", ["tenant_id"])
    op.create_index(
        "ix_security_exercise_status",
        "security_exercises",
        ["tenant_id", "status", "completed_at"],
    )

    op.create_table(
        "production_defects",
        sa.Column("defect_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("defect_key", sa.String(length=255), nullable=False),
        sa.Column(
            "exercise_id",
            sa.String(length=128),
            sa.ForeignKey("security_exercises.exercise_id"),
            nullable=False,
        ),
        sa.Column("scenario_id", sa.String(length=128), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("release_blocking", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("description_code", sa.String(length=128), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("closed_by_subject_id", sa.String(length=128), nullable=True),
        sa.Column("closure_evidence_ref", sa.String(length=255), nullable=True),
        sa.Column("closure_evidence_digest", sa.String(length=128), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "defect_key", name="uq_production_defect_key"),
    )
    op.create_index("ix_production_defects_tenant_id", "production_defects", ["tenant_id"])
    op.create_index(
        "ix_production_defect_status",
        "production_defects",
        ["tenant_id", "status", "severity"],
    )

    op.create_table(
        "production_acceptances",
        sa.Column("acceptance_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("request_hash", sa.String(length=128), nullable=False),
        sa.Column("release_scope", sa.String(length=128), nullable=False),
        sa.Column("target_environment", sa.String(length=32), nullable=False),
        sa.Column(
            "security_exercise_id",
            sa.String(length=128),
            sa.ForeignKey("security_exercises.exercise_id"),
            nullable=False,
        ),
        sa.Column(
            "model_release_id",
            sa.String(length=128),
            sa.ForeignKey("model_releases.release_id"),
            nullable=False,
        ),
        sa.Column("business_evidence_ref", sa.String(length=255), nullable=False),
        sa.Column("business_evidence_digest", sa.String(length=128), nullable=False),
        sa.Column("rollback_evidence_ref", sa.String(length=255), nullable=False),
        sa.Column("rollback_evidence_digest", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("blocker_codes_json", sa.JSON(), nullable=False),
        sa.Column("readiness_snapshot_json", sa.JSON(), nullable=False),
        sa.Column("created_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("signed_off_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_production_acceptance_idempotency",
        ),
    )
    op.create_index(
        "ix_production_acceptances_tenant_id",
        "production_acceptances",
        ["tenant_id"],
    )
    op.create_index(
        "ix_production_acceptance_status",
        "production_acceptances",
        ["tenant_id", "status", "evaluated_at"],
    )

    op.create_table(
        "production_acceptance_signoffs",
        sa.Column("signoff_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "acceptance_id",
            sa.String(length=128),
            sa.ForeignKey("production_acceptances.acceptance_id"),
            nullable=False,
        ),
        sa.Column("signoff_role", sa.String(length=32), nullable=False),
        sa.Column("readiness_digest", sa.String(length=128), nullable=False),
        sa.Column("evidence_ref", sa.String(length=255), nullable=False),
        sa.Column("evidence_digest", sa.String(length=128), nullable=False),
        sa.Column("signed_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("signed_at", sa.DateTime(timezone=True), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "acceptance_id",
            "signoff_role",
            name="uq_production_acceptance_signoff_role",
        ),
    )
    op.create_index(
        "ix_production_acceptance_signoffs_tenant_id",
        "production_acceptance_signoffs",
        ["tenant_id"],
    )

    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table in (
        "security_exercises",
        "production_defects",
        "production_acceptances",
        "production_acceptance_signoffs",
    ):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    op.drop_index(
        "ix_production_acceptance_signoffs_tenant_id",
        table_name="production_acceptance_signoffs",
    )
    op.drop_table("production_acceptance_signoffs")
    op.drop_index("ix_production_acceptance_status", table_name="production_acceptances")
    op.drop_index("ix_production_acceptances_tenant_id", table_name="production_acceptances")
    op.drop_table("production_acceptances")
    op.drop_index("ix_production_defect_status", table_name="production_defects")
    op.drop_index("ix_production_defects_tenant_id", table_name="production_defects")
    op.drop_table("production_defects")
    op.drop_index("ix_security_exercise_status", table_name="security_exercises")
    op.drop_index("ix_security_exercises_tenant_id", table_name="security_exercises")
    op.drop_table("security_exercises")
