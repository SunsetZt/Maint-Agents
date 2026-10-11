"""Add prompt-free Context Manifest evidence to model inference audits."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0030_context_manifest"
down_revision: str | None = "0029_prompt_bundle_governance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "model_inference_requests",
        sa.Column(
            "context_manifest_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::json"),
        ),
    )
    op.add_column(
        "model_inference_requests",
        sa.Column("context_hash", sa.String(length=128), nullable=True),
    )
    op.create_index(
        "ix_model_inference_context_hash",
        "model_inference_requests",
        ["tenant_id", "context_hash"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_model_inference_context_hash",
        table_name="model_inference_requests",
    )
    op.drop_column("model_inference_requests", "context_hash")
    op.drop_column("model_inference_requests", "context_manifest_json")
