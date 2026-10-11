"""Add durable file-based enterprise knowledge ingestion."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012_knowledge_file_ingestion"
down_revision: str | None = "0011_knowledge_ingestion"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "knowledge_document_versions",
        sa.Column("extraction_metadata", sa.JSON(), nullable=True),
    )
    op.create_table(
        "knowledge_ingestion_jobs",
        sa.Column("ingestion_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("workflow_id", sa.String(length=255), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("source_filename", sa.String(length=255), nullable=False),
        sa.Column("declared_mime", sa.String(length=128), nullable=False),
        sa.Column("detected_mime", sa.String(length=128), nullable=True),
        sa.Column("classification", sa.String(length=32), nullable=False),
        sa.Column("acl_subject_ids", sa.JSON(), nullable=False),
        sa.Column("acl_roles", sa.JSON(), nullable=False),
        sa.Column("device_families", sa.JSON(), nullable=False),
        sa.Column("device_models", sa.JSON(), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("quarantine_key", sa.String(length=1024), nullable=False),
        sa.Column("clean_key", sa.String(length=1024), nullable=True),
        sa.Column("source_uri", sa.String(length=1024), nullable=True),
        sa.Column("source_checksum", sa.String(length=128), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("failure_reason", sa.String(length=255), nullable=True),
        sa.Column("parser_version", sa.String(length=128), nullable=True),
        sa.Column(
            "document_id",
            sa.String(length=128),
            sa.ForeignKey("knowledge_documents.document_id"),
            nullable=True,
        ),
        sa.Column(
            "document_version_id",
            sa.String(length=128),
            sa.ForeignKey("knowledge_document_versions.document_version_id"),
            nullable=True,
        ),
        sa.Column("created_by_subject_id", sa.String(length=128), nullable=False),
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
            "workflow_id",
            name="uq_knowledge_ingestion_workflow",
        ),
    )
    op.create_index(
        "ix_knowledge_ingestion_jobs_tenant_id",
        "knowledge_ingestion_jobs",
        ["tenant_id"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE knowledge_ingestion_jobs ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE knowledge_ingestion_jobs FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY knowledge_ingestion_jobs_tenant_isolation "
        f"ON knowledge_ingestion_jobs USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_table("knowledge_ingestion_jobs")
    op.drop_column("knowledge_document_versions", "extraction_metadata")
