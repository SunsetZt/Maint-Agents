"""Add verifier-owned signed Staging acceptance evidence."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0070_signed_staging_acceptance_evidence"
down_revision: str | None = "0069_governed_external_cost_ledger"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "staging_acceptance_attestations",
        sa.Column("evidence_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("manifest_id", sa.String(length=128), nullable=False),
        sa.Column("manifest_digest", sa.String(length=128), nullable=False),
        sa.Column("environment_id", sa.String(length=128), nullable=False),
        sa.Column("git_commit", sa.String(length=64), nullable=False),
        sa.Column("signature_bundle_digest", sa.String(length=128), nullable=False),
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
            "tenant_id",
            "verification_hash",
            name="uq_staging_acceptance_attestation_verification",
        ),
    )
    op.create_index(
        "ix_staging_acceptance_attestations_tenant_id",
        "staging_acceptance_attestations",
        ["tenant_id"],
    )
    op.create_index(
        "ix_staging_acceptance_attestation_manifest",
        "staging_acceptance_attestations",
        ["tenant_id", "manifest_id", "verified_at"],
    )
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    op.execute("ALTER TABLE staging_acceptance_attestations ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE staging_acceptance_attestations FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY staging_acceptance_attestations_tenant_isolation "
        "ON staging_acceptance_attestations "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_index(
        "ix_staging_acceptance_attestation_manifest",
        table_name="staging_acceptance_attestations",
    )
    op.drop_index(
        "ix_staging_acceptance_attestations_tenant_id",
        table_name="staging_acceptance_attestations",
    )
    op.drop_table("staging_acceptance_attestations")

