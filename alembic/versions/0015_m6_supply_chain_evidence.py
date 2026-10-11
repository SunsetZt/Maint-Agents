"""Add immutable software and model supply-chain verification evidence."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0015_m6_supply_chain_evidence"
down_revision: str | None = "0014_m6_security_guardrails"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "supply_chain_evidence",
        sa.Column("evidence_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("image_repository", sa.String(length=512), nullable=False),
        sa.Column("image_digest", sa.String(length=128), nullable=False),
        sa.Column("model_artifact_digest", sa.String(length=128), nullable=False),
        sa.Column("source_repository", sa.String(length=1024), nullable=False),
        sa.Column("source_revision", sa.String(length=64), nullable=False),
        sa.Column("sbom_digest", sa.String(length=128), nullable=False),
        sa.Column("sbom_format", sa.String(length=32), nullable=False),
        sa.Column("vulnerability_report_digest", sa.String(length=128), nullable=False),
        sa.Column("vulnerability_scan_status", sa.String(length=32), nullable=False),
        sa.Column(
            "maximum_vulnerability_severity", sa.String(length=32), nullable=False
        ),
        sa.Column("vulnerability_scanner", sa.String(length=128), nullable=False),
        sa.Column("license_report_digest", sa.String(length=128), nullable=False),
        sa.Column("license_status", sa.String(length=32), nullable=False),
        sa.Column("provenance_digest", sa.String(length=128), nullable=False),
        sa.Column("image_signature_digest", sa.String(length=128), nullable=False),
        sa.Column("model_signature_digest", sa.String(length=128), nullable=False),
        sa.Column("certificate_identity", sa.String(length=1024), nullable=False),
        sa.Column("certificate_oidc_issuer", sa.String(length=1024), nullable=False),
        sa.Column("verifier_version", sa.String(length=64), nullable=False),
        sa.Column("verification_status", sa.String(length=32), nullable=False),
        sa.Column("verification_hash", sa.String(length=128), nullable=False),
        sa.Column("verified_by_subject_id", sa.String(length=128), nullable=False),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=False),
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
        sa.UniqueConstraint(
            "tenant_id", "verification_hash", name="uq_supply_chain_evidence_verification"
        ),
    )
    op.create_index(
        "ix_supply_chain_evidence_tenant_id", "supply_chain_evidence", ["tenant_id"]
    )
    op.create_index(
        "ix_supply_chain_evidence_image",
        "supply_chain_evidence",
        ["tenant_id", "image_repository", "image_digest"],
    )
    op.create_index(
        "ix_supply_chain_evidence_status",
        "supply_chain_evidence",
        ["tenant_id", "verification_status", "verified_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE supply_chain_evidence ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE supply_chain_evidence FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY supply_chain_evidence_tenant_isolation "
        "ON supply_chain_evidence "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index(
        "ix_supply_chain_evidence_status", table_name="supply_chain_evidence"
    )
    op.drop_index("ix_supply_chain_evidence_image", table_name="supply_chain_evidence")
    op.drop_index(
        "ix_supply_chain_evidence_tenant_id", table_name="supply_chain_evidence"
    )
    op.drop_table("supply_chain_evidence")
