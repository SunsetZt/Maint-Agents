"""Bind OpenSearch profiles to immutable production Embedding artifacts."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0037_search_profile_embedding_binding"
down_revision: str | None = "0036_opensearch_search_profiles"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for column in (
        sa.Column("embedding_model_release_id", sa.String(128), nullable=True),
        sa.Column("embedding_manifest_hash", sa.String(128), nullable=True),
        sa.Column("embedding_component_model_id", sa.String(128), nullable=True),
        sa.Column("embedding_artifact_content_hash", sa.String(128), nullable=True),
        sa.Column("embedding_dimension", sa.Integer(), nullable=True),
    ):
        op.add_column("knowledge_search_profiles", column)
    op.create_check_constraint(
        "ck_knowledge_search_embedding_dimension",
        "knowledge_search_profiles",
        "embedding_dimension IS NULL OR (embedding_dimension >= 8 AND embedding_dimension <= 8192)",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_knowledge_search_embedding_dimension",
        "knowledge_search_profiles",
        type_="check",
    )
    for name in (
        "embedding_dimension",
        "embedding_artifact_content_hash",
        "embedding_component_model_id",
        "embedding_manifest_hash",
        "embedding_model_release_id",
    ):
        op.drop_column("knowledge_search_profiles", name)
