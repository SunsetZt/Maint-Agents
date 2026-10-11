"""Add signed FIELD edge diagnosis packs and review-only candidates."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0065_governed_field_edge_diagnosis"
down_revision: str | None = "0064_governed_multimodal_injection_gate"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "work_order_edge_diagnosis_packs",
        sa.Column("pack_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("subject_id", sa.String(128), nullable=False),
        sa.Column("work_order_id", sa.String(128), nullable=False),
        sa.Column("work_order_version", sa.Integer(), nullable=False),
        sa.Column("asset_id", sa.String(128), nullable=False),
        sa.Column("asset_version", sa.Integer(), nullable=False),
        sa.Column("assignment_id", sa.String(128), nullable=False),
        sa.Column("assignment_assigned_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("offline_pack_id", sa.String(128), nullable=False),
        sa.Column("offline_pack_content_hash", sa.String(128), nullable=False),
        sa.Column("release_id", sa.String(128), nullable=False),
        sa.Column("manifest_hash", sa.String(128), nullable=False),
        sa.Column("model_file", sa.String(255), nullable=False),
        sa.Column("model_content_hash", sa.String(128), nullable=False),
        sa.Column("prompt_bundle_id", sa.String(128), nullable=False),
        sa.Column("prompt_bundle_hash", sa.String(128), nullable=False),
        sa.Column("index_release_id", sa.String(128), nullable=False),
        sa.Column("index_content_checksum", sa.String(128), nullable=False),
        sa.Column("key_id", sa.String(128), nullable=False),
        sa.Column("claims_json", sa.JSON(), nullable=False),
        sa.Column("signed_token", sa.Text(), nullable=False),
        sa.Column("pack_digest", sa.String(128), nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("request_digest", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
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
        sa.CheckConstraint(
            "status IN ('ACTIVE', 'SUPERSEDED')",
            name="ck_work_order_edge_pack_status",
        ),
        sa.CheckConstraint("version > 0", name="ck_work_order_edge_pack_version"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(["work_order_id"], ["work_orders.work_order_id"]),
        sa.ForeignKeyConstraint(["asset_id"], ["assets.asset_id"]),
        sa.ForeignKeyConstraint(
            ["assignment_id"], ["work_order_assignments.assignment_id"]
        ),
        sa.ForeignKeyConstraint(
            ["offline_pack_id"], ["work_order_offline_packs.pack_id"]
        ),
        sa.ForeignKeyConstraint(["release_id"], ["model_releases.release_id"]),
        sa.ForeignKeyConstraint(["index_release_id"], ["index_releases.release_id"]),
        sa.UniqueConstraint(
            "tenant_id",
            "subject_id",
            "idempotency_key",
            name="uq_work_order_edge_pack_idempotency",
        ),
    )
    op.create_index(
        "ix_work_order_edge_diagnosis_packs_tenant_id",
        "work_order_edge_diagnosis_packs",
        ["tenant_id"],
    )
    op.create_index(
        "ix_work_order_edge_pack_current",
        "work_order_edge_diagnosis_packs",
        ["tenant_id", "subject_id", "work_order_id", "status", "issued_at"],
    )
    op.create_table(
        "work_order_edge_diagnosis_candidates",
        sa.Column("candidate_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("subject_id", sa.String(128), nullable=False),
        sa.Column("work_order_id", sa.String(128), nullable=False),
        sa.Column("pack_id", sa.String(128), nullable=False),
        sa.Column("client_operation_id", sa.String(128), nullable=False),
        sa.Column("request_digest", sa.String(128), nullable=False),
        sa.Column("result_content_hash", sa.String(128), nullable=False),
        sa.Column("candidate_json", sa.JSON(), nullable=False),
        sa.Column("runtime_evidence_json", sa.JSON(), nullable=False),
        sa.Column("security_policy_version", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("reviewed_by_subject_id", sa.String(128), nullable=True),
        sa.Column("review_reason", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
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
        sa.CheckConstraint(
            "status IN ('PENDING_REVIEW', 'ACCEPTED', 'REJECTED')",
            name="ck_work_order_edge_candidate_status",
        ),
        sa.CheckConstraint("version > 0", name="ck_work_order_edge_candidate_version"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(["work_order_id"], ["work_orders.work_order_id"]),
        sa.ForeignKeyConstraint(
            ["pack_id"], ["work_order_edge_diagnosis_packs.pack_id"]
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "subject_id",
            "client_operation_id",
            name="uq_work_order_edge_candidate_operation",
        ),
    )
    op.create_index(
        "ix_work_order_edge_diagnosis_candidates_tenant_id",
        "work_order_edge_diagnosis_candidates",
        ["tenant_id"],
    )
    op.create_index(
        "ix_work_order_edge_candidate_work_order",
        "work_order_edge_diagnosis_candidates",
        ["tenant_id", "work_order_id", "created_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table in (
        "work_order_edge_diagnosis_packs",
        "work_order_edge_diagnosis_candidates",
    ):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    for table in (
        "work_order_edge_diagnosis_candidates",
        "work_order_edge_diagnosis_packs",
    ):
        op.execute(f"DROP POLICY IF EXISTS {table}_tenant_isolation ON {table}")
    op.drop_index(
        "ix_work_order_edge_candidate_work_order",
        table_name="work_order_edge_diagnosis_candidates",
    )
    op.drop_index(
        "ix_work_order_edge_diagnosis_candidates_tenant_id",
        table_name="work_order_edge_diagnosis_candidates",
    )
    op.drop_table("work_order_edge_diagnosis_candidates")
    op.drop_index(
        "ix_work_order_edge_pack_current",
        table_name="work_order_edge_diagnosis_packs",
    )
    op.drop_index(
        "ix_work_order_edge_diagnosis_packs_tenant_id",
        table_name="work_order_edge_diagnosis_packs",
    )
    op.drop_table("work_order_edge_diagnosis_packs")
