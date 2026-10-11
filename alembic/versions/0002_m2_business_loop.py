"""Add the M2 incident and multimodal evidence persistence boundary."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0002_m2_business_loop"
down_revision: str | None = "0001_m1_foundation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TENANT_TABLES = (
    "recognition_runs",
    "evidence_bundles",
    "recognition_results",
    "recognition_corrections",
    "incidents",
    "knowledge_documents",
    "knowledge_document_versions",
    "index_releases",
    "knowledge_chunks",
    "citation_anchors",
    "diagnosis_runs",
    "agent_runs",
    "agent_events",
    "agent_checkpoints",
    "tool_calls",
    "action_proposals",
    "approval_requests",
    "approval_decisions",
    "execution_attempts",
    "part_reservations",
    "work_orders",
    "work_order_events",
    "work_order_assignments",
    "work_order_completions",
    "work_order_verifications",
)


def _timestamps() -> tuple[sa.Column[sa.DateTime], sa.Column[sa.DateTime]]:
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


def _tenant_id() -> sa.Column[str]:
    return sa.Column(
        "tenant_id",
        sa.String(length=64),
        sa.ForeignKey("tenants.id"),
        nullable=False,
    )


def _install_rls(table_name: str) -> None:
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute(f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {table_name} FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY {table_name}_tenant_isolation ON {table_name} "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def upgrade() -> None:
    op.add_column("assets", sa.Column("model_code", sa.String(length=128), nullable=True))

    op.create_table(
        "recognition_runs",
        sa.Column("recognition_run_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "draft_id",
            sa.String(length=128),
            sa.ForeignKey("incident_drafts.draft_id"),
            nullable=False,
        ),
        sa.Column(
            "media_id",
            sa.String(length=128),
            sa.ForeignKey("media_objects.media_id"),
            nullable=False,
        ),
        sa.Column("workflow_id", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("processor_profile", sa.String(length=128), nullable=False),
        sa.Column("evidence_bundle_id", sa.String(length=128), nullable=True),
        sa.Column("failure_reason", sa.String(length=255), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "workflow_id", name="uq_recognition_workflow"),
    )
    op.create_table(
        "evidence_bundles",
        sa.Column("bundle_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "draft_id",
            sa.String(length=128),
            sa.ForeignKey("incident_drafts.draft_id"),
            nullable=False,
        ),
        sa.Column(
            "asset_id",
            sa.String(length=128),
            sa.ForeignKey("assets.asset_id"),
            nullable=False,
        ),
        sa.Column(
            "media_id",
            sa.String(length=128),
            sa.ForeignKey("media_objects.media_id"),
            nullable=False,
        ),
        sa.Column("source_sha256", sa.String(length=128), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("processor_versions", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "draft_id", "media_id", name="uq_evidence_draft_media"),
    )
    op.create_table(
        "recognition_results",
        sa.Column("result_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "bundle_id",
            sa.String(length=128),
            sa.ForeignKey("evidence_bundles.bundle_id"),
            nullable=False,
        ),
        sa.Column("result_type", sa.String(length=32), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "bundle_id",
            "result_type",
            "ordinal",
            name="uq_recognition_result_ordinal",
        ),
    )
    op.create_table(
        "recognition_corrections",
        sa.Column("correction_id", sa.String(length=255), primary_key=True),
        _tenant_id(),
        sa.Column(
            "bundle_id",
            sa.String(length=128),
            sa.ForeignKey("evidence_bundles.bundle_id"),
            nullable=False,
        ),
        sa.Column("target_type", sa.String(length=32), nullable=False),
        sa.Column("target_id", sa.String(length=128), nullable=False),
        sa.Column("original_value", sa.Text(), nullable=True),
        sa.Column("corrected_value", sa.Text(), nullable=True),
        sa.Column("disposition", sa.String(length=32), nullable=True),
        sa.Column("reviewer_subject_id", sa.String(length=128), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("evidence_version", sa.Integer(), nullable=False),
        *_timestamps(),
    )
    op.create_table(
        "incidents",
        sa.Column("incident_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "asset_id",
            sa.String(length=128),
            sa.ForeignKey("assets.asset_id"),
            nullable=False,
        ),
        sa.Column(
            "source_draft_id",
            sa.String(length=128),
            sa.ForeignKey("incident_drafts.draft_id"),
            nullable=False,
        ),
        sa.Column(
            "evidence_bundle_id",
            sa.String(length=128),
            sa.ForeignKey("evidence_bundles.bundle_id"),
            nullable=False,
        ),
        sa.Column("reporter_subject_id", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "source_draft_id", name="uq_incident_source_draft"),
    )
    op.create_table(
        "knowledge_documents",
        sa.Column("document_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("source_uri", sa.String(length=1024), nullable=False),
        sa.Column("classification", sa.String(length=32), nullable=False),
        sa.Column("source_checksum", sa.String(length=128), nullable=False),
        sa.Column("current_version", sa.Integer(), nullable=False),
        *_timestamps(),
    )
    op.create_table(
        "knowledge_document_versions",
        sa.Column("document_version_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "document_id",
            sa.String(length=128),
            sa.ForeignKey("knowledge_documents.document_id"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("parser_version", sa.String(length=128), nullable=False),
        sa.Column("chunker_version", sa.String(length=128), nullable=False),
        sa.Column("acl_subject_ids", sa.JSON(), nullable=False),
        sa.Column("acl_roles", sa.JSON(), nullable=False),
        sa.Column("device_families", sa.JSON(), nullable=False),
        sa.Column("device_models", sa.JSON(), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_checksum", sa.String(length=128), nullable=False),
        sa.Column("content_checksum", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "document_id",
            "version",
            name="uq_knowledge_document_version",
        ),
    )
    op.create_table(
        "index_releases",
        sa.Column("release_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("content_checksum", sa.String(length=128), nullable=False),
        sa.Column("published_by", sa.String(length=128), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "name", "version", name="uq_index_release_version"),
    )
    op.create_table(
        "knowledge_chunks",
        sa.Column("chunk_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "document_version_id",
            sa.String(length=128),
            sa.ForeignKey("knowledge_document_versions.document_version_id"),
            nullable=False,
        ),
        sa.Column(
            "release_id",
            sa.String(length=128),
            sa.ForeignKey("index_releases.release_id"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_checksum", sa.String(length=128), nullable=False),
        sa.Column("embedding", Vector(16), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "release_id",
            "document_version_id",
            "ordinal",
            name="uq_knowledge_chunk_ordinal",
        ),
    )
    op.create_table(
        "citation_anchors",
        sa.Column("citation_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "chunk_id",
            sa.String(length=128),
            sa.ForeignKey("knowledge_chunks.chunk_id"),
            nullable=False,
        ),
        sa.Column(
            "document_version_id",
            sa.String(length=128),
            sa.ForeignKey("knowledge_document_versions.document_version_id"),
            nullable=False,
        ),
        sa.Column("anchor_kind", sa.String(length=32), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("bounding_box", sa.JSON(), nullable=True),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("excerpt", sa.Text(), nullable=False),
        sa.Column("excerpt_checksum", sa.String(length=128), nullable=False),
        *_timestamps(),
    )
    op.create_table(
        "diagnosis_runs",
        sa.Column("diagnosis_run_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "incident_id",
            sa.String(length=128),
            sa.ForeignKey("incidents.incident_id"),
            nullable=False,
        ),
        sa.Column(
            "evidence_bundle_id",
            sa.String(length=128),
            sa.ForeignKey("evidence_bundles.bundle_id"),
            nullable=False,
        ),
        sa.Column("agent_run_id", sa.String(length=128), nullable=False),
        sa.Column("workflow_id", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("manifest", sa.JSON(), nullable=False),
        sa.Column("report", sa.JSON(), nullable=True),
        sa.Column("stop_reason", sa.String(length=128), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "workflow_id", name="uq_diagnosis_workflow"),
        sa.UniqueConstraint("tenant_id", "agent_run_id", name="uq_diagnosis_agent_run"),
    )
    op.create_table(
        "agent_runs",
        sa.Column("agent_run_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "diagnosis_run_id",
            sa.String(length=128),
            sa.ForeignKey("diagnosis_runs.diagnosis_run_id"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("budget", sa.JSON(), nullable=False),
        sa.Column("last_sequence", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
    )
    op.create_table(
        "agent_events",
        sa.Column("event_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "agent_run_id",
            sa.String(length=128),
            sa.ForeignKey("agent_runs.agent_run_id"),
            nullable=False,
        ),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("visibility", sa.String(length=32), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id", "agent_run_id", "sequence", name="uq_agent_event_sequence"
        ),
    )
    op.create_table(
        "agent_checkpoints",
        sa.Column("checkpoint_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "agent_run_id",
            sa.String(length=128),
            sa.ForeignKey("agent_runs.agent_run_id"),
            nullable=False,
        ),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("graph_version", sa.String(length=128), nullable=False),
        sa.Column("state", sa.JSON(), nullable=False),
        sa.Column("state_checksum", sa.String(length=128), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "agent_run_id",
            "sequence",
            name="uq_agent_checkpoint_sequence",
        ),
    )
    op.create_table(
        "tool_calls",
        sa.Column("tool_call_id", sa.String(128), primary_key=True),
        _tenant_id(),
        sa.Column("agent_run_id", sa.String(128), nullable=True),
        sa.Column("tool_id", sa.String(128), nullable=False),
        sa.Column("tool_version", sa.String(64), nullable=False),
        sa.Column("risk_tier", sa.String(8), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("input_digest", sa.String(128), nullable=False),
        sa.Column("output_metadata", sa.JSON(), nullable=True),
        *_timestamps(),
    )
    op.create_table(
        "action_proposals",
        sa.Column("proposal_id", sa.String(128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "incident_id", sa.String(128), sa.ForeignKey("incidents.incident_id"), nullable=False
        ),
        sa.Column("initiator_subject_id", sa.String(128), nullable=False),
        sa.Column("tool_id", sa.String(128), nullable=False),
        sa.Column("tool_version", sa.String(64), nullable=False),
        sa.Column("risk_tier", sa.String(8), nullable=False),
        sa.Column("parameters", sa.JSON(), nullable=False),
        sa.Column("parameters_hash", sa.String(128), nullable=False),
        sa.Column("resource_version", sa.Integer(), nullable=False),
        sa.Column("operation_id", sa.String(128), nullable=False, unique=True),
        sa.Column("status", sa.String(32), nullable=False),
        *_timestamps(),
    )
    op.create_table(
        "approval_requests",
        sa.Column("approval_id", sa.String(128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "proposal_id",
            sa.String(128),
            sa.ForeignKey("action_proposals.proposal_id"),
            nullable=False,
            unique=True,
        ),
        sa.Column("policy_version", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
    )
    op.create_table(
        "approval_decisions",
        sa.Column("decision_id", sa.String(128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "approval_id",
            sa.String(128),
            sa.ForeignKey("approval_requests.approval_id"),
            nullable=False,
        ),
        sa.Column("decision", sa.String(32), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("decider_subject_id", sa.String(128), nullable=False),
        sa.Column("parameters_hash", sa.String(128), nullable=False),
        sa.Column("resource_version", sa.Integer(), nullable=False),
        sa.Column("tool_version", sa.String(64), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        *_timestamps(),
    )
    op.create_table(
        "execution_attempts",
        sa.Column("attempt_id", sa.String(128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "proposal_id",
            sa.String(128),
            sa.ForeignKey("action_proposals.proposal_id"),
            nullable=False,
        ),
        sa.Column(
            "approval_id",
            sa.String(128),
            sa.ForeignKey("approval_requests.approval_id"),
            nullable=False,
        ),
        sa.Column("operation_id", sa.String(128), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("reason", sa.String(255), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
    )
    op.create_table(
        "part_reservations",
        sa.Column("reservation_id", sa.String(128), primary_key=True),
        _tenant_id(),
        sa.Column("operation_id", sa.String(128), nullable=False),
        sa.Column(
            "incident_id", sa.String(128), sa.ForeignKey("incidents.incident_id"), nullable=False
        ),
        sa.Column("part_number", sa.String(128), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("source", sa.String(128), nullable=False),
        sa.Column("source_record_id", sa.String(128), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "operation_id", name="uq_part_reservation_operation"),
    )
    op.create_table(
        "work_orders",
        sa.Column("work_order_id", sa.String(128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "incident_id", sa.String(128), sa.ForeignKey("incidents.incident_id"), nullable=False
        ),
        sa.Column(
            "proposal_id",
            sa.String(128),
            sa.ForeignKey("action_proposals.proposal_id"),
            nullable=False,
        ),
        sa.Column(
            "reservation_id",
            sa.String(128),
            sa.ForeignKey("part_reservations.reservation_id"),
            nullable=False,
        ),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("assigned_subject_id", sa.String(128), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
    )
    for table_name, id_name, extra_columns in (
        (
            "work_order_events",
            "event_id",
            [
                sa.Column("sequence", sa.Integer(), nullable=False),
                sa.Column("event_type", sa.String(64), nullable=False),
                sa.Column("actor_subject_id", sa.String(128), nullable=False),
                sa.Column("payload", sa.JSON(), nullable=False),
                sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
            ],
        ),
        (
            "work_order_assignments",
            "assignment_id",
            [
                sa.Column("assignee_subject_id", sa.String(128), nullable=False),
                sa.Column("assigned_by_subject_id", sa.String(128), nullable=False),
                sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False),
            ],
        ),
        (
            "work_order_completions",
            "completion_id",
            [
                sa.Column("completed_by_subject_id", sa.String(128), nullable=False),
                sa.Column("root_cause", sa.Text(), nullable=False),
                sa.Column("actions", sa.JSON(), nullable=False),
                sa.Column("evidence_ids", sa.JSON(), nullable=False),
                sa.Column("part_reservation_ids", sa.JSON(), nullable=False),
                sa.Column("cost_amount", sa.String(64), nullable=True),
                sa.Column("customer_confirmation", sa.String(255), nullable=True),
                sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
            ],
        ),
        (
            "work_order_verifications",
            "verification_id",
            [
                sa.Column("verifier_subject_id", sa.String(128), nullable=False),
                sa.Column("passed", sa.Boolean(), nullable=False),
                sa.Column("reason", sa.Text(), nullable=False),
                sa.Column("verified_at", sa.DateTime(timezone=True), nullable=False),
            ],
        ),
    ):
        constraints = []
        if table_name == "work_order_events":
            constraints.append(
                sa.UniqueConstraint(
                    "tenant_id", "work_order_id", "sequence", name="uq_work_order_event"
                )
            )
        op.create_table(
            table_name,
            sa.Column(id_name, sa.String(128), primary_key=True),
            _tenant_id(),
            sa.Column(
                "work_order_id",
                sa.String(128),
                sa.ForeignKey("work_orders.work_order_id"),
                nullable=False,
            ),
            *extra_columns,
            *_timestamps(),
            *constraints,
        )

    op.execute(
        "CREATE INDEX ix_knowledge_chunks_fts ON knowledge_chunks "
        "USING gin (to_tsvector('simple', content))"
    )
    op.create_index(
        "ix_knowledge_chunks_embedding_hnsw",
        "knowledge_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    for table_name in TENANT_TABLES:
        op.create_index(f"ix_{table_name}_tenant_id", table_name, ["tenant_id"])
        _install_rls(table_name)


def downgrade() -> None:
    for table_name in reversed(TENANT_TABLES):
        op.drop_table(table_name)
    op.drop_column("assets", "model_code")
