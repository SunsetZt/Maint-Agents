"""Persist AI guardrail evidence and complete tenant audit fingerprints."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0014_m6_security_guardrails"
down_revision: str | None = "0013_knowledge_ingestion_attempts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "model_inference_requests",
        sa.Column(
            "guardrail_policy_version",
            sa.String(length=128),
            nullable=False,
            server_default="not-evaluated",
        ),
    )
    op.add_column(
        "model_inference_requests",
        sa.Column(
            "guardrail_findings_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )
    op.add_column(
        "security_audit_events",
        sa.Column(
            "tenant_hash",
            sa.String(length=128),
            nullable=False,
            server_default="legacy",
        ),
    )
    op.add_column(
        "security_audit_events",
        sa.Column("resource_hash", sa.String(length=128), nullable=True),
    )
    op.create_index(
        "ix_security_audit_tenant_created",
        "security_audit_events",
        ["tenant_id", "created_at", "event_id"],
    )
    op.create_index(
        "ix_security_audit_tenant_decision",
        "security_audit_events",
        ["tenant_id", "decision", "created_at"],
    )
    op.alter_column(
        "model_inference_requests", "guardrail_policy_version", server_default=None
    )
    op.alter_column(
        "model_inference_requests", "guardrail_findings_json", server_default=None
    )
    op.alter_column("security_audit_events", "tenant_hash", server_default=None)


def downgrade() -> None:
    op.drop_index(
        "ix_security_audit_tenant_decision", table_name="security_audit_events"
    )
    op.drop_index(
        "ix_security_audit_tenant_created", table_name="security_audit_events"
    )
    op.drop_column("security_audit_events", "resource_hash")
    op.drop_column("security_audit_events", "tenant_hash")
    op.drop_column("model_inference_requests", "guardrail_findings_json")
    op.drop_column("model_inference_requests", "guardrail_policy_version")
