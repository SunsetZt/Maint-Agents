"""Add context-bound recognition for governed field evidence."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0062_governed_field_evidence_recognition"
down_revision: str | None = "0061_governed_field_evidence_capture"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "recognition_runs",
        sa.Column(
            "context_kind",
            sa.String(32),
            nullable=False,
            server_default="INCIDENT_DRAFT",
        ),
    )
    op.add_column(
        "recognition_runs",
        sa.Column("work_order_id", sa.String(128), nullable=True),
    )
    op.add_column(
        "recognition_runs",
        sa.Column("field_evidence_upload_id", sa.String(128), nullable=True),
    )
    op.add_column(
        "recognition_runs",
        sa.Column("work_order_version", sa.Integer(), nullable=True),
    )
    op.add_column(
        "recognition_runs",
        sa.Column("requested_by_subject_id", sa.String(128), nullable=True),
    )
    op.create_foreign_key(
        "fk_recognition_runs_work_order",
        "recognition_runs",
        "work_orders",
        ["work_order_id"],
        ["work_order_id"],
    )
    op.create_foreign_key(
        "fk_recognition_runs_field_evidence_upload",
        "recognition_runs",
        "work_order_field_evidence_uploads",
        ["field_evidence_upload_id"],
        ["evidence_upload_id"],
    )
    op.create_unique_constraint(
        "uq_recognition_field_evidence_binding",
        "recognition_runs",
        [
            "tenant_id",
            "context_kind",
            "field_evidence_upload_id",
            "requested_by_subject_id",
            "work_order_version",
        ],
    )
    op.create_check_constraint(
        "ck_recognition_run_context_kind",
        "recognition_runs",
        "context_kind IN ('INCIDENT_DRAFT', 'FIELD_EVIDENCE')",
    )
    op.create_check_constraint(
        "ck_recognition_run_context_binding",
        "recognition_runs",
        "(context_kind = 'INCIDENT_DRAFT' AND work_order_id IS NULL "
        "AND field_evidence_upload_id IS NULL AND work_order_version IS NULL) OR "
        "(context_kind = 'FIELD_EVIDENCE' AND work_order_id IS NOT NULL "
        "AND field_evidence_upload_id IS NOT NULL AND work_order_version IS NOT NULL "
        "AND requested_by_subject_id IS NOT NULL)",
    )
    op.create_index(
        "ix_recognition_runs_field_context",
        "recognition_runs",
        ["tenant_id", "work_order_id", "field_evidence_upload_id", "context_kind"],
    )

    op.add_column(
        "evidence_bundles",
        sa.Column(
            "context_kind",
            sa.String(32),
            nullable=False,
            server_default="INCIDENT_DRAFT",
        ),
    )
    op.add_column(
        "evidence_bundles",
        sa.Column("work_order_id", sa.String(128), nullable=True),
    )
    op.add_column(
        "evidence_bundles",
        sa.Column("field_evidence_upload_id", sa.String(128), nullable=True),
    )
    op.create_foreign_key(
        "fk_evidence_bundles_work_order",
        "evidence_bundles",
        "work_orders",
        ["work_order_id"],
        ["work_order_id"],
    )
    op.create_foreign_key(
        "fk_evidence_bundles_field_evidence_upload",
        "evidence_bundles",
        "work_order_field_evidence_uploads",
        ["field_evidence_upload_id"],
        ["evidence_upload_id"],
    )
    op.drop_constraint("uq_evidence_draft_media", "evidence_bundles", type_="unique")
    op.create_index(
        "uq_evidence_incident_draft_media",
        "evidence_bundles",
        ["tenant_id", "draft_id", "media_id"],
        unique=True,
        postgresql_where=sa.text("context_kind = 'INCIDENT_DRAFT'"),
    )
    op.create_check_constraint(
        "ck_evidence_bundle_context_kind",
        "evidence_bundles",
        "context_kind IN ('INCIDENT_DRAFT', 'FIELD_EVIDENCE')",
    )
    op.create_check_constraint(
        "ck_evidence_bundle_context_binding",
        "evidence_bundles",
        "(context_kind = 'INCIDENT_DRAFT' AND work_order_id IS NULL "
        "AND field_evidence_upload_id IS NULL) OR "
        "(context_kind = 'FIELD_EVIDENCE' AND work_order_id IS NOT NULL "
        "AND field_evidence_upload_id IS NOT NULL)",
    )
    op.create_index(
        "ix_evidence_bundles_field_context",
        "evidence_bundles",
        ["tenant_id", "work_order_id", "field_evidence_upload_id", "context_kind"],
    )


def downgrade() -> None:
    op.drop_index("ix_evidence_bundles_field_context", table_name="evidence_bundles")
    op.drop_constraint(
        "ck_evidence_bundle_context_binding", "evidence_bundles", type_="check"
    )
    op.drop_constraint(
        "ck_evidence_bundle_context_kind", "evidence_bundles", type_="check"
    )
    op.drop_index("uq_evidence_incident_draft_media", table_name="evidence_bundles")
    op.create_unique_constraint(
        "uq_evidence_draft_media",
        "evidence_bundles",
        ["tenant_id", "draft_id", "media_id"],
    )
    op.drop_constraint(
        "fk_evidence_bundles_field_evidence_upload",
        "evidence_bundles",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_evidence_bundles_work_order", "evidence_bundles", type_="foreignkey"
    )
    op.drop_column("evidence_bundles", "field_evidence_upload_id")
    op.drop_column("evidence_bundles", "work_order_id")
    op.drop_column("evidence_bundles", "context_kind")

    op.drop_index("ix_recognition_runs_field_context", table_name="recognition_runs")
    op.drop_constraint(
        "ck_recognition_run_context_binding", "recognition_runs", type_="check"
    )
    op.drop_constraint(
        "ck_recognition_run_context_kind", "recognition_runs", type_="check"
    )
    op.drop_constraint(
        "uq_recognition_field_evidence_binding", "recognition_runs", type_="unique"
    )
    op.drop_constraint(
        "fk_recognition_runs_field_evidence_upload",
        "recognition_runs",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_recognition_runs_work_order", "recognition_runs", type_="foreignkey"
    )
    op.drop_column("recognition_runs", "requested_by_subject_id")
    op.drop_column("recognition_runs", "work_order_version")
    op.drop_column("recognition_runs", "field_evidence_upload_id")
    op.drop_column("recognition_runs", "work_order_id")
    op.drop_column("recognition_runs", "context_kind")
