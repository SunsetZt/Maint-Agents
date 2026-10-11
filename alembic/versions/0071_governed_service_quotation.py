"""Add governed service quotation publication and customer decision facts."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0071_governed_service_quotation"
down_revision: str | None = "0070_signed_staging_acceptance_evidence"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "service_quotations",
        sa.Column("quotation_id", sa.String(length=128), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("incident_id", sa.String(length=128), sa.ForeignKey("incidents.incident_id"), nullable=False),
        sa.Column("asset_id", sa.String(length=128), sa.ForeignKey("assets.asset_id"), nullable=False),
        sa.Column("diagnosis_run_id", sa.String(length=128), sa.ForeignKey("diagnosis_runs.diagnosis_run_id"), nullable=False),
        sa.Column("diagnosis_version", sa.Integer(), nullable=False),
        sa.Column("quotation_version", sa.Integer(), nullable=False),
        sa.Column("supersedes_quotation_id", sa.String(length=128), sa.ForeignKey("service_quotations.quotation_id"), nullable=True),
        sa.Column("proposal_id", sa.String(length=128), sa.ForeignKey("action_proposals.proposal_id"), nullable=False),
        sa.Column("approval_id", sa.String(length=128), sa.ForeignKey("approval_requests.approval_id"), nullable=False),
        sa.Column("operation_id", sa.String(length=128), nullable=False),
        sa.Column("provider", sa.String(length=128), nullable=False),
        sa.Column("candidate_id", sa.String(length=128), nullable=False),
        sa.Column("source_version", sa.String(length=128), nullable=False),
        sa.Column("source_as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("line_items_json", sa.JSON(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("subtotal", sa.String(length=32), nullable=False),
        sa.Column("discount", sa.String(length=32), nullable=False),
        sa.Column("tax", sa.String(length=32), nullable=False),
        sa.Column("total", sa.String(length=32), nullable=False),
        sa.Column("entitlement_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("state_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "operation_id", name="uq_service_quotation_operation"),
        sa.UniqueConstraint("tenant_id", "incident_id", "quotation_version", name="uq_service_quotation_incident_version"),
        sa.CheckConstraint("status IN ('PUBLISHED', 'ACCEPTED', 'REJECTED', 'EXPIRED', 'SUPERSEDED')", name="ck_service_quotation_status"),
        sa.CheckConstraint("quotation_version > 0", name="ck_service_quotation_version"),
        sa.CheckConstraint("state_version > 0", name="ck_service_quotation_state_version"),
    )
    op.create_index("ix_service_quotations_tenant_id", "service_quotations", ["tenant_id"])
    op.create_index("ix_service_quotations_incident", "service_quotations", ["tenant_id", "incident_id", "quotation_version"])

    op.create_table(
        "service_quotation_decisions",
        sa.Column("decision_id", sa.String(length=128), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("quotation_id", sa.String(length=128), sa.ForeignKey("service_quotations.quotation_id"), nullable=False),
        sa.Column("quotation_version", sa.Integer(), nullable=False),
        sa.Column("incident_id", sa.String(length=128), sa.ForeignKey("incidents.incident_id"), nullable=False),
        sa.Column("reporter_subject_id", sa.String(length=128), nullable=False),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column("client_operation_id", sa.String(length=255), nullable=False),
        sa.Column("request_digest", sa.String(length=128), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "quotation_id", name="uq_service_quotation_decision_quotation"),
        sa.UniqueConstraint("tenant_id", "reporter_subject_id", "client_operation_id", name="uq_service_quotation_decision_operation"),
        sa.CheckConstraint("decision IN ('ACCEPTED', 'REJECTED')", name="ck_service_quotation_decision_value"),
    )
    op.create_index("ix_service_quotation_decisions_tenant_id", "service_quotation_decisions", ["tenant_id"])
    op.create_index("ix_service_quotation_decisions_incident", "service_quotation_decisions", ["tenant_id", "incident_id", "decided_at"])

    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table in ("service_quotations", "service_quotation_decisions"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    op.drop_index("ix_service_quotation_decisions_incident", table_name="service_quotation_decisions")
    op.drop_index("ix_service_quotation_decisions_tenant_id", table_name="service_quotation_decisions")
    op.drop_table("service_quotation_decisions")
    op.drop_index("ix_service_quotations_incident", table_name="service_quotations")
    op.drop_index("ix_service_quotations_tenant_id", table_name="service_quotations")
    op.drop_table("service_quotations")
