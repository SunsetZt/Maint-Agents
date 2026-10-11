"""Persist maintenance review exposure without certifying legacy source history."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

revision: str = "0086_maintenance_review_exposure"
down_revision: str | None = "0085_diagnosis_incident_state_repair"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> list[sa.Column[Any]]:
    return [
        sa.Column(name, sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
        for name in ("created_at", "updated_at")
    ]


def upgrade() -> None:
    op.create_table(
        "maintenance_review_exposures",
        sa.Column("exposure_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column(
            "subject_id", sa.String(128), sa.ForeignKey("subjects.subject_id"), nullable=False
        ),
        sa.Column("source_family_key", sa.String(128), nullable=False),
        sa.Column("source_version_digest", sa.String(128), nullable=False),
        sa.Column("reason", sa.String(64), nullable=False),
        sa.Column("resource_kind", sa.String(64), nullable=False),
        sa.Column("resource_id", sa.String(128), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("request_id", sa.String(128), nullable=False),
        sa.Column("dedup_key", sa.String(128), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id", "subject_id", "dedup_key", name="uq_review_exposure_dedup"
        ),
    )
    op.create_index(
        "ix_review_exposure_family",
        "maintenance_review_exposures",
        ["tenant_id", "subject_id", "source_family_key"],
    )
    op.create_table(
        "maintenance_review_assignments",
        sa.Column("claim_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column(
            "subject_id", sa.String(128), sa.ForeignKey("subjects.subject_id"), nullable=False
        ),
        sa.Column(
            "evaluation_id",
            sa.String(128),
            sa.ForeignKey("maintenance_planning_evaluation_runs.evaluation_id"),
            nullable=False,
        ),
        sa.Column("case_id", sa.String(128), nullable=False),
        sa.Column("source_family_keys", sa.JSON(), nullable=False),
        sa.Column("source_binding_digest", sa.String(128), nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("request_digest", sa.String(128), nullable=False),
        sa.Column("policy_version", sa.String(128), nullable=False),
        sa.Column("policy_epoch_version", sa.Integer(), nullable=False),
        sa.Column("reader_coverage_digest", sa.String(128), nullable=False),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("withdrawn_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("submission_digest", sa.String(128), nullable=True),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id", "subject_id", "idempotency_key", name="uq_review_claim_key"
        ),
        sa.UniqueConstraint(
            "tenant_id", "subject_id", "evaluation_id", "case_id", name="uq_review_claim_case"
        ),
        sa.CheckConstraint(
            "state IN ('ACTIVE', 'SUBMITTED', 'WITHDRAWN')",
            name="ck_maintenance_review_assignment_state",
        ),
        sa.CheckConstraint(
            "version > 0 AND policy_epoch_version > 0",
            name="ck_maintenance_review_assignment_version",
        ),
        sa.CheckConstraint(
            "(state = 'ACTIVE' AND submitted_at IS NULL AND withdrawn_at IS NULL "
            "AND submission_digest IS NULL) OR "
            "(state = 'SUBMITTED' AND submitted_at IS NOT NULL AND withdrawn_at IS NULL "
            "AND submission_digest IS NOT NULL) OR "
            "(state = 'WITHDRAWN' AND submitted_at IS NULL AND withdrawn_at IS NOT NULL "
            "AND submission_digest IS NULL)",
            name="ck_review_claim_timestamps",
        ),
    )
    op.create_index(
        "ix_review_claim_subject_state",
        "maintenance_review_assignments",
        ["tenant_id", "subject_id", "state"],
    )
    op.create_table(
        "maintenance_review_rollouts",
        sa.Column("receipt_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("receipt_digest", sa.String(128), nullable=False),
        sa.Column("receipt_text", sa.Text(), nullable=False),
        sa.Column("signature_bundle", sa.Text(), nullable=False),
        sa.Column("proof_json", sa.JSON(), nullable=False),
        sa.Column("verification_digest", sa.String(128), nullable=False),
        sa.Column(
            "registered_by", sa.String(128), sa.ForeignKey("subjects.subject_id"), nullable=False
        ),
        sa.Column("registered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("request_id", sa.String(128), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "receipt_digest", name="uq_review_rollout_digest"),
    )
    op.create_table(
        "maintenance_review_policy_commands",
        sa.Column("command_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column(
            "subject_id", sa.String(128), sa.ForeignKey("subjects.subject_id"), nullable=False
        ),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("request_digest", sa.String(128), nullable=False),
        sa.Column("result_json", sa.JSON(), nullable=False),
        sa.Column("request_id", sa.String(128), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id", "subject_id", "idempotency_key", name="uq_review_policy_command"
        ),
    )
    op.create_table(
        "maintenance_review_policies",
        sa.Column("tenant_id", sa.String(64), sa.ForeignKey("tenants.id"), primary_key=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "receipt_id",
            sa.String(128),
            sa.ForeignKey("maintenance_review_rollouts.receipt_id"),
            nullable=True,
        ),
        sa.Column("receipt_digest", sa.String(128), nullable=True),
        sa.Column("policy_version", sa.String(128), nullable=False),
        sa.Column("reader_coverage_digest", sa.String(128), nullable=False),
        sa.Column("cutover_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "activated_by", sa.String(128), sa.ForeignKey("subjects.subject_id"), nullable=False
        ),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("request_digest", sa.String(128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.CheckConstraint("version > 0", name="ck_review_policy_version"),
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table in (
        "maintenance_review_exposures",
        "maintenance_review_assignments",
        "maintenance_review_policies",
        "maintenance_review_rollouts",
        "maintenance_review_policy_commands",
    ):
        if table != "maintenance_review_policies":
            op.create_index(f"ix_{table}_tenant_id", table, ["tenant_id"])
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    raise RuntimeError("retain maintenance review history; roll back application code only")
