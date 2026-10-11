"""Add governed tenant-isolated GraphRAG causal evidence projections."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0035_graphrag_causal_evidence"
down_revision: str | None = "0034_supplier_a2a_collaboration"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge_graph_releases",
        sa.Column("graph_release_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("source_index_release_id", sa.String(128), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("schema_version", sa.String(64), nullable=False),
        sa.Column("manifest_hash", sa.String(128), nullable=True),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column("gate_results_json", sa.JSON(), nullable=False),
        sa.Column("failure_codes", sa.JSON(), nullable=False),
        sa.Column("created_by_subject_id", sa.String(128), nullable=False),
        sa.Column("evaluated_by_subject_id", sa.String(128), nullable=True),
        sa.Column("activated_by_subject_id", sa.String(128), nullable=True),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("state_version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(["source_index_release_id"], ["index_releases.release_id"]),
        sa.UniqueConstraint(
            "tenant_id", "name", "version", name="uq_knowledge_graph_release_version"
        ),
    )
    op.create_index(
        "uq_active_knowledge_graph_release",
        "knowledge_graph_releases",
        ["tenant_id", "name"],
        unique=True,
        postgresql_where=sa.text("is_active"),
        sqlite_where=sa.text("is_active = 1"),
    )
    op.create_table(
        "knowledge_graph_nodes",
        sa.Column("graph_node_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("graph_release_id", sa.String(128), nullable=False),
        sa.Column("node_key", sa.String(128), nullable=False),
        sa.Column("node_type", sa.String(32), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("citation_id", sa.String(128), nullable=False),
        sa.Column("document_version_id", sa.String(128), nullable=False),
        sa.Column("classification", sa.String(32), nullable=False),
        sa.Column("device_families", sa.JSON(), nullable=False),
        sa.Column("device_models", sa.JSON(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(
            ["graph_release_id"], ["knowledge_graph_releases.graph_release_id"]
        ),
        sa.ForeignKeyConstraint(["citation_id"], ["citation_anchors.citation_id"]),
        sa.ForeignKeyConstraint(
            ["document_version_id"], ["knowledge_document_versions.document_version_id"]
        ),
        sa.UniqueConstraint(
            "tenant_id", "graph_release_id", "node_key", name="uq_knowledge_graph_node_key"
        ),
    )
    op.create_index(
        "ix_knowledge_graph_node_source",
        "knowledge_graph_nodes",
        ["tenant_id", "document_version_id"],
    )
    op.create_table(
        "knowledge_graph_edges",
        sa.Column("graph_edge_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("graph_release_id", sa.String(128), nullable=False),
        sa.Column("source_node_id", sa.String(128), nullable=False),
        sa.Column("target_node_id", sa.String(128), nullable=False),
        sa.Column("relation_type", sa.String(32), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("citation_id", sa.String(128), nullable=False),
        sa.Column("document_version_id", sa.String(128), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_by_subject_id", sa.String(128), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(
            ["graph_release_id"], ["knowledge_graph_releases.graph_release_id"]
        ),
        sa.ForeignKeyConstraint(["source_node_id"], ["knowledge_graph_nodes.graph_node_id"]),
        sa.ForeignKeyConstraint(["target_node_id"], ["knowledge_graph_nodes.graph_node_id"]),
        sa.ForeignKeyConstraint(["citation_id"], ["citation_anchors.citation_id"]),
        sa.ForeignKeyConstraint(
            ["document_version_id"], ["knowledge_document_versions.document_version_id"]
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "graph_release_id",
            "source_node_id",
            "target_node_id",
            "relation_type",
            name="uq_knowledge_graph_edge_fact",
        ),
    )
    op.create_index(
        "ix_knowledge_graph_edge_source",
        "knowledge_graph_edges",
        ["tenant_id", "document_version_id"],
    )
    op.create_table(
        "knowledge_graph_evaluations",
        sa.Column("evaluation_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("graph_release_id", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("policy_version", sa.String(128), nullable=False),
        sa.Column("cases_json", sa.JSON(), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column("gate_results_json", sa.JSON(), nullable=False),
        sa.Column("failure_codes", sa.JSON(), nullable=False),
        sa.Column("requested_by_subject_id", sa.String(128), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(
            ["graph_release_id"], ["knowledge_graph_releases.graph_release_id"]
        ),
    )
    for table in (
        "knowledge_graph_releases",
        "knowledge_graph_nodes",
        "knowledge_graph_edges",
        "knowledge_graph_evaluations",
    ):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            "USING (tenant_id = current_setting('app.tenant_id', true)) "
            "WITH CHECK (tenant_id = current_setting('app.tenant_id', true))"
        )


def downgrade() -> None:
    op.drop_table("knowledge_graph_evaluations")
    op.drop_table("knowledge_graph_edges")
    op.drop_table("knowledge_graph_nodes")
    op.drop_table("knowledge_graph_releases")
