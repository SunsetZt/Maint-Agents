"""Add immutable governed field-evidence upload bindings."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0061_governed_field_evidence_capture"
down_revision: str | None = "0060_governed_field_offline_work_pack"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "work_order_field_evidence_uploads",
        sa.Column("evidence_upload_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("work_order_id", sa.String(128), nullable=False),
        sa.Column("incident_id", sa.String(128), nullable=False),
        sa.Column("source_draft_id", sa.String(128), nullable=False),
        sa.Column("media_id", sa.String(128), nullable=False),
        sa.Column("uploaded_by_subject_id", sa.String(128), nullable=False),
        sa.Column("client_operation_id", sa.String(128), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        sa.Column("declared_mime", sa.String(255), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("work_order_version", sa.Integer(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
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
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(["work_order_id"], ["work_orders.work_order_id"]),
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.incident_id"]),
        sa.ForeignKeyConstraint(["source_draft_id"], ["incident_drafts.draft_id"]),
        sa.ForeignKeyConstraint(["media_id"], ["media_objects.media_id"]),
        sa.UniqueConstraint(
            "tenant_id",
            "work_order_id",
            "uploaded_by_subject_id",
            "client_operation_id",
            name="uq_work_order_field_evidence_operation",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "media_id",
            name="uq_work_order_field_evidence_media",
        ),
    )
    op.create_index(
        "ix_work_order_field_evidence_uploads_tenant_id",
        "work_order_field_evidence_uploads",
        ["tenant_id"],
    )
    op.create_index(
        "ix_work_order_field_evidence_uploads_work_order",
        "work_order_field_evidence_uploads",
        ["tenant_id", "work_order_id", "occurred_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE work_order_field_evidence_uploads ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE work_order_field_evidence_uploads FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY work_order_field_evidence_uploads_tenant_isolation "
        "ON work_order_field_evidence_uploads "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.execute(
        "DROP POLICY IF EXISTS work_order_field_evidence_uploads_tenant_isolation "
        "ON work_order_field_evidence_uploads"
    )
    op.drop_index(
        "ix_work_order_field_evidence_uploads_work_order",
        table_name="work_order_field_evidence_uploads",
    )
    op.drop_index(
        "ix_work_order_field_evidence_uploads_tenant_id",
        table_name="work_order_field_evidence_uploads",
    )
    op.drop_table("work_order_field_evidence_uploads")
