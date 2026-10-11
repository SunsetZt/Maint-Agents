"""Create the M1 tenant data foundation and fail-closed RLS policies."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_m1_foundation"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TENANT_TABLES = (
    "subjects",
    "subject_roles",
    "assets",
    "asset_scopes",
    "incident_drafts",
    "idempotency_records",
    "draft_timeline_events",
    "media_objects",
    "security_audit_events",
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


def _install_rls(table_name: str) -> None:
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute(f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {table_name} FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY {table_name}_tenant_isolation ON {table_name} "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "tenants",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("status", sa.String(length=24), nullable=False),
        *_timestamps(),
    )
    op.create_table(
        "subjects",
        sa.Column("subject_id", sa.String(length=128), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("oidc_issuer", sa.String(length=512), nullable=False),
        sa.Column("oidc_subject", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "oidc_issuer", "oidc_subject", name="uq_subject_oidc"),
    )
    op.create_table(
        "subject_roles",
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            primary_key=True,
        ),
        sa.Column(
            "subject_id",
            sa.String(length=128),
            sa.ForeignKey("subjects.subject_id"),
            primary_key=True,
        ),
        sa.Column("role", sa.String(length=64), primary_key=True),
        *_timestamps(),
    )
    op.create_table(
        "assets",
        sa.Column("asset_id", sa.String(length=128), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("source_system", sa.String(length=128), nullable=False),
        sa.Column("source_record_id", sa.String(length=255), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id", "source_system", "source_record_id", name="uq_asset_source"
        ),
    )
    op.create_index("ix_assets_tenant_id", "assets", ["tenant_id"])
    op.create_table(
        "asset_scopes",
        sa.Column("scope_id", sa.String(length=128), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("subject_id", sa.String(length=128), nullable=True),
        sa.Column("role", sa.String(length=64), nullable=True),
        sa.Column("asset_id", sa.String(length=128), nullable=True),
        sa.Column("site_id", sa.String(length=128), nullable=True),
        *_timestamps(),
    )
    op.create_table(
        "incident_drafts",
        sa.Column("draft_id", sa.String(length=128), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column(
            "asset_id",
            sa.String(length=128),
            sa.ForeignKey("assets.asset_id"),
            nullable=False,
        ),
        sa.Column("author_subject_id", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        *_timestamps(),
    )
    op.create_table(
        "idempotency_records",
        sa.Column("record_id", sa.String(length=128), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("subject_id", sa.String(length=128), nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("request_hash", sa.String(length=128), nullable=False),
        sa.Column("result_ref", sa.String(length=255), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "subject_id", "key", name="uq_idempotency_key"),
    )
    op.create_table(
        "draft_timeline_events",
        sa.Column("event_id", sa.String(length=128), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column(
            "draft_id",
            sa.String(length=128),
            sa.ForeignKey("incident_drafts.draft_id"),
            nullable=False,
        ),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload_metadata", sa.JSON(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "draft_id", "sequence", name="uq_draft_timeline_sequence"),
    )
    op.create_table(
        "media_objects",
        sa.Column("media_id", sa.String(length=128), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column(
            "draft_id",
            sa.String(length=128),
            sa.ForeignKey("incident_drafts.draft_id"),
            nullable=False,
        ),
        sa.Column("quarantine_key", sa.String(length=1024), nullable=False),
        sa.Column("clean_key", sa.String(length=1024), nullable=True),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("declared_mime", sa.String(length=255), nullable=False),
        sa.Column("detected_mime", sa.String(length=255), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("scan_state", sa.String(length=32), nullable=False),
        sa.Column("scan_version", sa.Integer(), nullable=False),
        *_timestamps(),
    )
    op.create_table(
        "security_audit_events",
        sa.Column("event_id", sa.String(length=128), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), sa.ForeignKey("tenants.id"), nullable=True),
        sa.Column("subject_hash", sa.String(length=128), nullable=False),
        sa.Column("action", sa.String(length=128), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.String(length=255), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=False),
        *_timestamps(),
    )
    for table_name in TENANT_TABLES:
        _install_rls(table_name)


def downgrade() -> None:
    for table_name in reversed(TENANT_TABLES):
        op.drop_table(table_name)
    op.drop_table("tenants")
