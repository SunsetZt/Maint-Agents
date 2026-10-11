"""Persist bounded C2PA inspection evidence for uploaded media."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0033_media_content_credentials"
down_revision: str | None = "0032_synthetic_vlm_ablation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "media_objects",
        sa.Column(
            "content_credential_status",
            sa.String(length=32),
            nullable=False,
            server_default="PENDING_INSPECTION",
        ),
    )
    op.add_column(
        "media_objects",
        sa.Column("content_credential_report", sa.JSON(), nullable=True),
    )
    op.add_column(
        "media_objects",
        sa.Column("content_credential_digest", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "media_objects",
        sa.Column(
            "content_credential_inspected_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.execute(
        "UPDATE media_objects SET content_credential_status = "
        "CASE WHEN scan_state = 'PENDING' THEN 'PENDING_INSPECTION' "
        "ELSE 'NOT_INSPECTED' END"
    )
    op.alter_column("media_objects", "content_credential_status", server_default=None)


def downgrade() -> None:
    op.drop_column("media_objects", "content_credential_inspected_at")
    op.drop_column("media_objects", "content_credential_digest")
    op.drop_column("media_objects", "content_credential_report")
    op.drop_column("media_objects", "content_credential_status")
