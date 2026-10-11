"""Add append-only DIA-004 expert diagnosis intervention records."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0021_dia004_expert_intervention"
down_revision: str | None = "0020_kb002_index_evaluation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "diagnosis_expert_interventions",
        sa.Column("intervention_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "diagnosis_run_id",
            sa.String(length=128),
            sa.ForeignKey("diagnosis_runs.diagnosis_run_id"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("source_run_version", sa.Integer(), nullable=False),
        sa.Column("ai_report_snapshot", sa.JSON(), nullable=True),
        sa.Column("ai_report_digest", sa.String(length=128), nullable=True),
        sa.Column("takeover_reason", sa.String(length=1024), nullable=False),
        sa.Column("requested_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
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
    )
    op.create_index(
        "ix_diagnosis_expert_interventions_tenant_id",
        "diagnosis_expert_interventions",
        ["tenant_id"],
    )
    op.create_index(
        "ix_diagnosis_expert_interventions_run",
        "diagnosis_expert_interventions",
        ["tenant_id", "diagnosis_run_id", "created_at"],
    )
    op.create_index(
        "uq_open_diagnosis_expert_intervention",
        "diagnosis_expert_interventions",
        ["tenant_id", "diagnosis_run_id"],
        unique=True,
        postgresql_where=sa.text("status = 'OPEN'"),
        sqlite_where=sa.text("status = 'OPEN'"),
    )
    op.create_table(
        "diagnosis_expert_revisions",
        sa.Column("revision_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "intervention_id",
            sa.String(length=128),
            sa.ForeignKey("diagnosis_expert_interventions.intervention_id"),
            nullable=False,
        ),
        sa.Column(
            "diagnosis_run_id",
            sa.String(length=128),
            sa.ForeignKey("diagnosis_runs.diagnosis_run_id"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("outcome", sa.String(length=32), nullable=False),
        sa.Column("report_json", sa.JSON(), nullable=False),
        sa.Column("report_digest", sa.String(length=128), nullable=False),
        sa.Column("source_ai_report_digest", sa.String(length=128), nullable=False),
        sa.Column("revision_reason", sa.String(length=1024), nullable=False),
        sa.Column("authored_by_subject_id", sa.String(length=128), nullable=False),
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
            "intervention_id",
            "ordinal",
            name="uq_diagnosis_expert_revision_ordinal",
        ),
    )
    op.create_index(
        "ix_diagnosis_expert_revisions_tenant_id",
        "diagnosis_expert_revisions",
        ["tenant_id"],
    )
    op.create_index(
        "ix_diagnosis_expert_revisions_run",
        "diagnosis_expert_revisions",
        ["tenant_id", "diagnosis_run_id", "created_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table_name in (
        "diagnosis_expert_interventions",
        "diagnosis_expert_revisions",
    ):
        op.execute(f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table_name} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table_name}_tenant_isolation ON {table_name} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    op.drop_index(
        "ix_diagnosis_expert_revisions_run",
        table_name="diagnosis_expert_revisions",
    )
    op.drop_index(
        "ix_diagnosis_expert_revisions_tenant_id",
        table_name="diagnosis_expert_revisions",
    )
    op.drop_table("diagnosis_expert_revisions")
    op.drop_index(
        "uq_open_diagnosis_expert_intervention",
        table_name="diagnosis_expert_interventions",
    )
    op.drop_index(
        "ix_diagnosis_expert_interventions_run",
        table_name="diagnosis_expert_interventions",
    )
    op.drop_index(
        "ix_diagnosis_expert_interventions_tenant_id",
        table_name="diagnosis_expert_interventions",
    )
    op.drop_table("diagnosis_expert_interventions")
