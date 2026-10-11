"""Add governed GraphRAG extraction and independent review jobs."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0059_reviewed_graphrag_extraction"
down_revision: str | None = "0058_authoritative_part_consumption_return_ledger"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge_graph_extraction_jobs",
        sa.Column("extraction_job_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("graph_release_id", sa.String(128), nullable=False),
        sa.Column("source_index_release_id", sa.String(128), nullable=False),
        sa.Column("expected_graph_version", sa.Integer(), nullable=False),
        sa.Column("citation_ids_json", sa.JSON(), nullable=False),
        sa.Column("citation_bindings_json", sa.JSON(), nullable=False),
        sa.Column("source_binding_hash", sa.String(128), nullable=False),
        sa.Column("extraction_profile", sa.String(128), nullable=False),
        sa.Column("workflow_id", sa.String(255), nullable=False),
        sa.Column("inference_request_id", sa.String(128), nullable=False),
        sa.Column("model_alias", sa.String(128), nullable=False),
        sa.Column("model_release_id", sa.String(128), nullable=False),
        sa.Column("model_manifest_hash", sa.String(128), nullable=False),
        sa.Column("prompt_bundle_id", sa.String(128), nullable=False),
        sa.Column("prompt_bundle_hash", sa.String(128), nullable=False),
        sa.Column("response_schema_version", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("stage", sa.String(64), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("candidate_bundle_json", sa.JSON(), nullable=True),
        sa.Column("result_hash", sa.String(128), nullable=True),
        sa.Column("context_hash", sa.String(128), nullable=True),
        sa.Column("failure_code", sa.String(128), nullable=True),
        sa.Column("requested_by_subject_id", sa.String(128), nullable=False),
        sa.Column("reviewed_by_subject_id", sa.String(128), nullable=True),
        sa.Column("review_decision", sa.String(16), nullable=True),
        sa.Column("review_reason", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
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
        sa.CheckConstraint(
            "status IN ('QUEUED', 'RUNNING', 'REVIEW_PENDING', 'ACCEPTED', "
            "'REJECTED', 'FAILED')",
            name="ck_knowledge_graph_extraction_status",
        ),
        sa.CheckConstraint(
            "attempt_count >= 0", name="ck_knowledge_graph_extraction_attempt"
        ),
        sa.CheckConstraint("version > 0", name="ck_knowledge_graph_extraction_version"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(
            ["graph_release_id"], ["knowledge_graph_releases.graph_release_id"]
        ),
        sa.ForeignKeyConstraint(["source_index_release_id"], ["index_releases.release_id"]),
        sa.UniqueConstraint(
            "tenant_id", "workflow_id", name="uq_knowledge_graph_extraction_workflow"
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "inference_request_id",
            name="uq_knowledge_graph_extraction_inference",
        ),
    )
    op.create_index(
        "ix_knowledge_graph_extraction_release_status",
        "knowledge_graph_extraction_jobs",
        ["tenant_id", "graph_release_id", "status"],
    )
    op.create_index(
        "ix_knowledge_graph_extraction_jobs_tenant_id",
        "knowledge_graph_extraction_jobs",
        ["tenant_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_knowledge_graph_extraction_jobs_tenant_id",
        table_name="knowledge_graph_extraction_jobs",
    )
    op.drop_index(
        "ix_knowledge_graph_extraction_release_status",
        table_name="knowledge_graph_extraction_jobs",
    )
    op.drop_table("knowledge_graph_extraction_jobs")
