"""Persist governed extracted knowledge text and review facts."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011_knowledge_ingestion"
down_revision: str | None = "0010_realtime_agent_input"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "knowledge_document_versions",
        sa.Column("extracted_text", sa.Text(), nullable=True),
    )
    op.add_column(
        "knowledge_document_versions",
        sa.Column("extraction_checksum", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "knowledge_document_versions",
        sa.Column("created_by_subject_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "knowledge_document_versions",
        sa.Column("reviewed_by_subject_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "knowledge_document_versions",
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "knowledge_document_versions",
        sa.Column("state_version", sa.Integer(), nullable=False, server_default="1"),
    )
    op.create_index(
        "uq_active_index_release",
        "index_releases",
        ["tenant_id", "name"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )


def downgrade() -> None:
    op.drop_index("uq_active_index_release", table_name="index_releases")
    op.drop_column("knowledge_document_versions", "state_version")
    op.drop_column("knowledge_document_versions", "reviewed_at")
    op.drop_column("knowledge_document_versions", "reviewed_by_subject_id")
    op.drop_column("knowledge_document_versions", "created_by_subject_id")
    op.drop_column("knowledge_document_versions", "extraction_checksum")
    op.drop_column("knowledge_document_versions", "extracted_text")
