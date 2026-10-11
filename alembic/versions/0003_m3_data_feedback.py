"""Add the M3 transactional event Outbox boundary."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_m3_data_feedback"
down_revision: str | None = "0002_m2_business_loop"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


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
    op.create_table(
        "event_outbox",
        sa.Column("event_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column("event_type", sa.String(length=128), nullable=False),
        sa.Column("event_version", sa.Integer(), nullable=False),
        sa.Column("aggregate_type", sa.String(length=64), nullable=False),
        sa.Column("aggregate_id", sa.String(length=128), nullable=False),
        sa.Column("aggregate_version", sa.Integer(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("producer", sa.String(length=128), nullable=False),
        sa.Column("partition_key", sa.String(length=255), nullable=False),
        sa.Column("trace_id", sa.String(length=128), nullable=False),
        sa.Column("correlation_id", sa.String(length=128), nullable=False),
        sa.Column("causation_id", sa.String(length=128), nullable=False),
        sa.Column("data_classification", sa.String(length=32), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "aggregate_type",
            "aggregate_id",
            "sequence",
            name="uq_event_outbox_aggregate_sequence",
        ),
    )
    op.create_index(
        "ix_event_outbox_tenant_id",
        "event_outbox",
        ["tenant_id"],
    )
    op.create_index(
        "ix_event_outbox_occurred_event",
        "event_outbox",
        ["occurred_at", "event_id"],
    )
    op.create_table(
        "event_inbox",
        sa.Column("inbox_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column("consumer_name", sa.String(length=128), nullable=False),
        sa.Column("event_id", sa.String(length=128), nullable=False),
        sa.Column("event_digest", sa.String(length=128), nullable=False),
        sa.Column("event_type", sa.String(length=128), nullable=False),
        sa.Column("aggregate_type", sa.String(length=64), nullable=False),
        sa.Column("aggregate_id", sa.String(length=128), nullable=False),
        sa.Column("aggregate_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("result_ref", sa.String(length=128), nullable=True),
        sa.Column("failure_reason", sa.String(length=255), nullable=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.UniqueConstraint(
            "consumer_name",
            "event_id",
            name="uq_event_inbox_consumer_event",
        ),
    )
    op.create_index("ix_event_inbox_tenant_id", "event_inbox", ["tenant_id"])
    op.create_table(
        "event_aggregate_projections",
        sa.Column("projection_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column("consumer_name", sa.String(length=128), nullable=False),
        sa.Column("aggregate_type", sa.String(length=64), nullable=False),
        sa.Column("aggregate_id", sa.String(length=128), nullable=False),
        sa.Column("last_aggregate_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("gap_expected_version", sa.Integer(), nullable=True),
        sa.Column("gap_received_version", sa.Integer(), nullable=True),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "consumer_name",
            "aggregate_type",
            "aggregate_id",
            name="uq_event_projection_consumer_aggregate",
        ),
    )
    op.create_index(
        "ix_event_aggregate_projections_tenant_id",
        "event_aggregate_projections",
        ["tenant_id"],
    )
    op.create_table(
        "feedback_candidates",
        sa.Column("candidate_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column("source_event_id", sa.String(length=128), nullable=False),
        sa.Column(
            "work_order_id",
            sa.String(length=128),
            sa.ForeignKey("work_orders.work_order_id"),
            nullable=False,
        ),
        sa.Column(
            "incident_id",
            sa.String(length=128),
            sa.ForeignKey("incidents.incident_id"),
            nullable=False,
        ),
        sa.Column(
            "completion_id",
            sa.String(length=128),
            sa.ForeignKey("work_order_completions.completion_id"),
            nullable=False,
        ),
        sa.Column(
            "verification_id",
            sa.String(length=128),
            sa.ForeignKey("work_order_verifications.verification_id"),
            nullable=False,
        ),
        sa.Column("source_content_hash", sa.String(length=128), nullable=False),
        sa.Column("data_classification", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("eligibility_status", sa.String(length=32), nullable=False),
        sa.Column("allow_training", sa.Boolean(), nullable=True),
        sa.Column("consent_basis", sa.String(length=255), nullable=True),
        sa.Column("license_status", sa.String(length=32), nullable=False),
        sa.Column("retention_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lineage_origin", sa.JSON(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "source_event_id",
            name="uq_feedback_candidate_source_event",
        ),
    )
    op.create_index(
        "ix_feedback_candidates_tenant_id",
        "feedback_candidates",
        ["tenant_id"],
    )
    op.create_table(
        "data_eligibility_decisions",
        sa.Column("decision_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "candidate_id",
            sa.String(length=128),
            sa.ForeignKey("feedback_candidates.candidate_id"),
            nullable=False,
        ),
        sa.Column("purpose", sa.String(length=128), nullable=False),
        sa.Column("allow_training", sa.Boolean(), nullable=False),
        sa.Column("consent_basis", sa.String(length=255), nullable=True),
        sa.Column("license_status", sa.String(length=32), nullable=False),
        sa.Column("retention_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decided_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("policy_version", sa.String(length=128), nullable=False),
        sa.Column("candidate_version", sa.Integer(), nullable=False),
        *_timestamps(),
    )
    op.create_index(
        "ix_data_eligibility_decisions_tenant_id",
        "data_eligibility_decisions",
        ["tenant_id"],
    )
    op.create_table(
        "dlp_processing_results",
        sa.Column("dlp_result_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "candidate_id",
            sa.String(length=128),
            sa.ForeignKey("feedback_candidates.candidate_id"),
            nullable=False,
        ),
        sa.Column("source_content_hash", sa.String(length=128), nullable=False),
        sa.Column("policy_version", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("redacted_content", sa.JSON(), nullable=False),
        sa.Column("findings", sa.JSON(), nullable=False),
        sa.Column("residual_entity_types", sa.JSON(), nullable=False),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("processed_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("candidate_version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "candidate_id",
            "source_content_hash",
            "policy_version",
            name="uq_dlp_candidate_source_policy",
        ),
    )
    op.create_index(
        "ix_dlp_processing_results_tenant_id",
        "dlp_processing_results",
        ["tenant_id"],
    )
    op.create_table(
        "annotation_tasks",
        sa.Column("task_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "candidate_id",
            sa.String(length=128),
            sa.ForeignKey("feedback_candidates.candidate_id"),
            nullable=False,
        ),
        sa.Column(
            "dlp_result_id",
            sa.String(length=128),
            sa.ForeignKey("dlp_processing_results.dlp_result_id"),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("project_id", sa.String(length=128), nullable=False),
        sa.Column("external_task_id", sa.String(length=128), nullable=False),
        sa.Column("external_version", sa.String(length=128), nullable=False),
        sa.Column("payload_hash", sa.String(length=128), nullable=False),
        sa.Column("schema_version", sa.String(length=128), nullable=False),
        sa.Column("risk_level", sa.String(length=32), nullable=False),
        sa.Column("required_reviews", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("consistency_status", sa.String(length=32), nullable=False),
        sa.Column("created_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_annotation_task_idempotency",
        ),
    )
    op.create_index("ix_annotation_tasks_tenant_id", "annotation_tasks", ["tenant_id"])
    op.create_table(
        "annotation_revisions",
        sa.Column("revision_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "task_id",
            sa.String(length=128),
            sa.ForeignKey("annotation_tasks.task_id"),
            nullable=False,
        ),
        sa.Column("external_annotation_id", sa.String(length=128), nullable=False),
        sa.Column("external_version", sa.String(length=128), nullable=False),
        sa.Column("reviewer_subject_id", sa.String(length=128), nullable=False),
        sa.Column("labels", sa.JSON(), nullable=False),
        sa.Column("labels_digest", sa.String(length=128), nullable=False),
        sa.Column("task_payload_hash", sa.String(length=128), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "task_id",
            "external_annotation_id",
            "external_version",
            name="uq_annotation_revision_external_version",
        ),
    )
    op.create_index(
        "ix_annotation_revisions_tenant_id",
        "annotation_revisions",
        ["tenant_id"],
    )
    op.create_table(
        "annotation_reviews",
        sa.Column("review_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "task_id",
            sa.String(length=128),
            sa.ForeignKey("annotation_tasks.task_id"),
            nullable=False,
        ),
        sa.Column(
            "revision_id",
            sa.String(length=128),
            sa.ForeignKey("annotation_revisions.revision_id"),
            nullable=False,
        ),
        sa.Column("reviewer_subject_id", sa.String(length=128), nullable=False),
        sa.Column("outcome", sa.String(length=32), nullable=False),
        sa.Column("consistency_key", sa.String(length=128), nullable=False),
        sa.Column("task_payload_hash", sa.String(length=128), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "task_id",
            "revision_id",
            name="uq_annotation_review_revision",
        ),
    )
    op.create_index(
        "ix_annotation_reviews_tenant_id",
        "annotation_reviews",
        ["tenant_id"],
    )
    op.create_table(
        "curation_runs",
        sa.Column("run_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("engine", sa.String(length=32), nullable=False),
        sa.Column("contract_version", sa.String(length=128), nullable=False),
        sa.Column("code_version", sa.String(length=128), nullable=False),
        sa.Column("config_hash", sa.String(length=128), nullable=False),
        sa.Column("input_manifest_hash", sa.String(length=128), nullable=False),
        sa.Column("input_count", sa.Integer(), nullable=False),
        sa.Column("exclusion_report", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("failure_reason", sa.String(length=255), nullable=True),
        *_timestamps(),
    )
    op.create_index("ix_curation_runs_tenant_id", "curation_runs", ["tenant_id"])
    op.create_table(
        "dataset_snapshots",
        sa.Column("snapshot_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "run_id",
            sa.String(length=128),
            sa.ForeignKey("curation_runs.run_id"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("contract_version", sa.String(length=128), nullable=False),
        sa.Column("input_manifest_hash", sa.String(length=128), nullable=False),
        sa.Column("manifest_key", sa.String(length=1024), nullable=True),
        sa.Column("manifest_hash", sa.String(length=128), nullable=True),
        sa.Column("quality_report_key", sa.String(length=1024), nullable=True),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("split_counts", sa.JSON(), nullable=False),
        sa.Column("source_work_order_ids", sa.JSON(), nullable=False),
        sa.Column("lineage_status", sa.String(length=32), nullable=False),
        sa.Column("training_eligible", sa.Boolean(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "run_id", name="uq_dataset_snapshot_run"),
    )
    op.create_index(
        "ix_dataset_snapshots_tenant_id", "dataset_snapshots", ["tenant_id"]
    )
    op.create_table(
        "dataset_artifacts",
        sa.Column("artifact_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "snapshot_id",
            sa.String(length=128),
            sa.ForeignKey("dataset_snapshots.snapshot_id"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("split", sa.String(length=32), nullable=True),
        sa.Column("object_key", sa.String(length=1024), nullable=False),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("schema_hash", sa.String(length=128), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id", "snapshot_id", "object_key", name="uq_dataset_artifact_key"
        ),
    )
    op.create_index(
        "ix_dataset_artifacts_tenant_id", "dataset_artifacts", ["tenant_id"]
    )
    op.create_table(
        "data_lineage_runs",
        sa.Column("lineage_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column(
            "run_id",
            sa.String(length=128),
            sa.ForeignKey("curation_runs.run_id"),
            nullable=False,
        ),
        sa.Column(
            "snapshot_id",
            sa.String(length=128),
            sa.ForeignKey("dataset_snapshots.snapshot_id"),
            nullable=False,
        ),
        sa.Column("openlineage_run_id", sa.String(length=128), nullable=False),
        sa.Column("job_namespace", sa.String(length=255), nullable=False),
        sa.Column("job_name", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("source_work_order_ids", sa.JSON(), nullable=False),
        sa.Column("output_dataset", sa.String(length=1024), nullable=False),
        sa.Column("failure_reason", sa.String(length=255), nullable=True),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "run_id", name="uq_data_lineage_run"),
    )
    op.create_index(
        "ix_data_lineage_runs_tenant_id", "data_lineage_runs", ["tenant_id"]
    )
    for table_name in (
        "event_outbox",
        "event_inbox",
        "event_aggregate_projections",
        "feedback_candidates",
        "data_eligibility_decisions",
        "dlp_processing_results",
        "annotation_tasks",
        "annotation_revisions",
        "annotation_reviews",
        "curation_runs",
        "dataset_snapshots",
        "dataset_artifacts",
        "data_lineage_runs",
    ):
        _install_rls(table_name)


def downgrade() -> None:
    op.drop_index("ix_data_lineage_runs_tenant_id", table_name="data_lineage_runs")
    op.drop_table("data_lineage_runs")
    op.drop_index("ix_dataset_artifacts_tenant_id", table_name="dataset_artifacts")
    op.drop_table("dataset_artifacts")
    op.drop_index("ix_dataset_snapshots_tenant_id", table_name="dataset_snapshots")
    op.drop_table("dataset_snapshots")
    op.drop_index("ix_curation_runs_tenant_id", table_name="curation_runs")
    op.drop_table("curation_runs")
    op.drop_index("ix_annotation_reviews_tenant_id", table_name="annotation_reviews")
    op.drop_table("annotation_reviews")
    op.drop_index("ix_annotation_revisions_tenant_id", table_name="annotation_revisions")
    op.drop_table("annotation_revisions")
    op.drop_index("ix_annotation_tasks_tenant_id", table_name="annotation_tasks")
    op.drop_table("annotation_tasks")
    op.drop_index(
        "ix_dlp_processing_results_tenant_id",
        table_name="dlp_processing_results",
    )
    op.drop_table("dlp_processing_results")
    op.drop_index(
        "ix_data_eligibility_decisions_tenant_id",
        table_name="data_eligibility_decisions",
    )
    op.drop_table("data_eligibility_decisions")
    op.drop_index("ix_feedback_candidates_tenant_id", table_name="feedback_candidates")
    op.drop_table("feedback_candidates")
    op.drop_index(
        "ix_event_aggregate_projections_tenant_id",
        table_name="event_aggregate_projections",
    )
    op.drop_table("event_aggregate_projections")
    op.drop_index("ix_event_inbox_tenant_id", table_name="event_inbox")
    op.drop_table("event_inbox")
    op.drop_index("ix_event_outbox_occurred_event", table_name="event_outbox")
    op.drop_index("ix_event_outbox_tenant_id", table_name="event_outbox")
    op.drop_table("event_outbox")
