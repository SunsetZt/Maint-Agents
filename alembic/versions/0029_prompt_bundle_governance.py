"""Add GitOps Prompt Bundle governance metadata and lifecycle evidence."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0029_prompt_bundle_governance"
down_revision: str | None = "0028_ops_cost_attribution"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "prompt_bundles",
        sa.Column("prompt_bundle_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("bundle_version", sa.String(length=64), nullable=False),
        sa.Column("task_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("source_commit", sa.String(length=128), nullable=False),
        sa.Column("source_path", sa.String(length=512), nullable=False),
        sa.Column("section_names_json", sa.JSON(), nullable=False),
        sa.Column("required_variables_json", sa.JSON(), nullable=False),
        sa.Column("output_schema_name", sa.String(length=128), nullable=False),
        sa.Column("compatible_model_aliases_json", sa.JSON(), nullable=False),
        sa.Column("change_summary", sa.Text(), nullable=False),
        sa.Column(
            "evaluation_id",
            sa.String(length=128),
            sa.ForeignKey("model_evaluation_runs.evaluation_id"),
            nullable=True,
        ),
        sa.Column("evaluation_report_hash", sa.String(length=128), nullable=True),
        sa.Column("created_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("reviewed_by_subject_id", sa.String(length=128), nullable=True),
        sa.Column("review_reason", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
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
            "name",
            "bundle_version",
            name="uq_prompt_bundle_name_version",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_prompt_bundle_idempotency",
        ),
    )
    op.create_index("ix_prompt_bundles_tenant_id", "prompt_bundles", ["tenant_id"])
    op.create_index(
        "ix_prompt_bundle_status",
        "prompt_bundles",
        ["tenant_id", "status"],
    )
    op.create_table(
        "prompt_bundle_transitions",
        sa.Column("transition_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "prompt_bundle_id",
            sa.String(length=128),
            sa.ForeignKey("prompt_bundles.prompt_bundle_id"),
            nullable=False,
        ),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("from_status", sa.String(length=32), nullable=False),
        sa.Column("to_status", sa.String(length=32), nullable=False),
        sa.Column("actor_subject_id", sa.String(length=128), nullable=False),
        sa.Column("reason_code", sa.String(length=128), nullable=False),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
        sa.Column("evidence_hash", sa.String(length=128), nullable=False),
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
            "prompt_bundle_id",
            "sequence",
            name="uq_prompt_bundle_transition_sequence",
        ),
    )
    op.create_index(
        "ix_prompt_bundle_transitions_tenant_id",
        "prompt_bundle_transitions",
        ["tenant_id"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table in ("prompt_bundles", "prompt_bundle_transitions"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    op.drop_index(
        "ix_prompt_bundle_transitions_tenant_id",
        table_name="prompt_bundle_transitions",
    )
    op.drop_table("prompt_bundle_transitions")
    op.drop_index("ix_prompt_bundle_status", table_name="prompt_bundles")
    op.drop_index("ix_prompt_bundles_tenant_id", table_name="prompt_bundles")
    op.drop_table("prompt_bundles")
