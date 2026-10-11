"""Add optimistic IndexRelease activation state and append-only rollback evidence."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0040_knowledge_index_rollback"
down_revision: str | None = "0039_controlled_external_search"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "index_releases",
        sa.Column(
            "activation_version",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),
    )
    op.create_table(
        "knowledge_index_activations",
        sa.Column("activation_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("from_release_id", sa.String(128), nullable=True),
        sa.Column("to_release_id", sa.String(128), nullable=False),
        sa.Column("from_content_checksum", sa.String(128), nullable=True),
        sa.Column("to_content_checksum", sa.String(128), nullable=False),
        sa.Column("actor_subject_id", sa.String(128), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("retired_search_profile_ids", sa.JSON(), nullable=False),
        sa.Column("revoked_graph_release_ids", sa.JSON(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(["from_release_id"], ["index_releases.release_id"]),
        sa.ForeignKeyConstraint(["to_release_id"], ["index_releases.release_id"]),
    )
    op.create_index(
        "ix_knowledge_index_activations_tenant_id",
        "knowledge_index_activations",
        ["tenant_id"],
    )
    op.create_index(
        "ix_knowledge_index_activations_name",
        "knowledge_index_activations",
        ["tenant_id", "name", "occurred_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE knowledge_index_activations ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE knowledge_index_activations FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY knowledge_index_activations_tenant_isolation "
        f"ON knowledge_index_activations USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index(
        "ix_knowledge_index_activations_name",
        table_name="knowledge_index_activations",
    )
    op.drop_index(
        "ix_knowledge_index_activations_tenant_id",
        table_name="knowledge_index_activations",
    )
    op.drop_table("knowledge_index_activations")
    op.drop_column("index_releases", "activation_version")
