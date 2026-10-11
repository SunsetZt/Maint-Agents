"""Add production aliases, tenant quotas and prompt-minimized inference audit."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_m5_model_gateway"
down_revision: str | None = "0007_m5_model_deployment_control"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> tuple[sa.Column[sa.DateTime], sa.Column[sa.DateTime]]:
    return (
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )


def _tenant_id() -> sa.Column[str]:
    return sa.Column(
        "tenant_id", sa.String(length=64), sa.ForeignKey("tenants.id"), nullable=False
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
        "model_aliases",
        sa.Column("alias_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column("alias", sa.String(length=128), nullable=False),
        sa.Column(
            "active_release_id",
            sa.String(length=128),
            sa.ForeignKey("model_releases.release_id"),
            nullable=False,
        ),
        sa.Column(
            "deployment_id",
            sa.String(length=128),
            sa.ForeignKey("model_deployments.deployment_id"),
            nullable=False,
        ),
        sa.Column("manifest_hash", sa.String(length=128), nullable=False),
        sa.Column("endpoint_url", sa.String(length=1024), nullable=False),
        sa.Column("runtime_profile", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("updated_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "alias", name="uq_model_alias_tenant_name"),
    )
    op.create_index("ix_model_aliases_tenant_id", "model_aliases", ["tenant_id"])

    op.create_table(
        "model_gateway_quotas",
        sa.Column("quota_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column("model_alias", sa.String(length=128), nullable=False),
        sa.Column("requests_per_minute", sa.Integer(), nullable=False),
        sa.Column("tokens_per_day", sa.BigInteger(), nullable=False),
        sa.Column("max_output_tokens", sa.Integer(), nullable=False),
        sa.Column("allowed_request_classes", sa.JSON(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "model_alias", name="uq_model_gateway_quota_alias"),
    )
    op.create_index(
        "ix_model_gateway_quotas_tenant_id", "model_gateway_quotas", ["tenant_id"]
    )

    op.create_table(
        "model_inference_requests",
        sa.Column("inference_request_id", sa.String(length=128), primary_key=True),
        _tenant_id(),
        sa.Column("model_alias", sa.String(length=128), nullable=False),
        sa.Column(
            "resolved_release_id",
            sa.String(length=128),
            sa.ForeignKey("model_releases.release_id"),
            nullable=False,
        ),
        sa.Column("manifest_hash", sa.String(length=128), nullable=False),
        sa.Column("subject_id", sa.String(length=128), nullable=False),
        sa.Column("trace_id", sa.String(length=128), nullable=False),
        sa.Column("request_class", sa.String(length=64), nullable=False),
        sa.Column("data_classification", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("deadline", sa.DateTime(timezone=True), nullable=False),
        sa.Column("max_output_tokens", sa.Integer(), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=False),
        sa.Column("prompt_hash", sa.String(length=128), nullable=False),
        sa.Column("response_json", sa.JSON(), nullable=True),
        sa.Column("response_hash", sa.String(length=128), nullable=True),
        sa.Column("usage_prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("usage_completion_tokens", sa.Integer(), nullable=False),
        sa.Column("finish_reason", sa.String(length=64), nullable=True),
        sa.Column("latency_breakdown_json", sa.JSON(), nullable=False),
        sa.Column("safety_decision", sa.String(length=64), nullable=False),
        sa.Column("degraded_from", sa.String(length=128), nullable=True),
        sa.Column("failure_reason", sa.String(length=255), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
    )
    op.create_index(
        "ix_model_inference_tenant_created",
        "model_inference_requests",
        ["tenant_id", "created_at"],
    )
    op.create_index(
        "ix_model_inference_tenant_release",
        "model_inference_requests",
        ["tenant_id", "resolved_release_id"],
    )

    for table in ("model_aliases", "model_gateway_quotas", "model_inference_requests"):
        _install_rls(table)


def downgrade() -> None:
    op.drop_index(
        "ix_model_inference_tenant_release", table_name="model_inference_requests"
    )
    op.drop_index(
        "ix_model_inference_tenant_created", table_name="model_inference_requests"
    )
    op.drop_table("model_inference_requests")
    op.drop_index("ix_model_gateway_quotas_tenant_id", table_name="model_gateway_quotas")
    op.drop_table("model_gateway_quotas")
    op.drop_index("ix_model_aliases_tenant_id", table_name="model_aliases")
    op.drop_table("model_aliases")
