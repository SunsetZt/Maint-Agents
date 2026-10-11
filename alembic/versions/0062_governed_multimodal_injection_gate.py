"""Persist governed multimodal prompt-injection findings and automation gates."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0064_governed_multimodal_injection_gate"
down_revision: str | None = "0063_governed_field_realtime_voice_guidance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "evidence_bundles",
        sa.Column("security_policy_version", sa.String(128), nullable=True),
    )
    op.add_column(
        "evidence_bundles",
        sa.Column(
            "security_findings_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )
    op.add_column(
        "evidence_bundles",
        sa.Column(
            "automation_eligible",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "evidence_bundles",
        sa.Column(
            "automation_blockers_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )
    op.execute(
        sa.text(
            "UPDATE evidence_bundles "
            "SET automation_eligible = true, automation_blockers_json = '[]' "
            "WHERE status = 'CONFIRMED'"
        )
    )
    op.add_column(
        "realtime_transcript_segments",
        sa.Column("security_policy_version", sa.String(128), nullable=True),
    )
    op.add_column(
        "realtime_transcript_segments",
        sa.Column(
            "security_findings_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )


def downgrade() -> None:
    op.drop_column("realtime_transcript_segments", "security_findings_json")
    op.drop_column("realtime_transcript_segments", "security_policy_version")
    op.drop_column("evidence_bundles", "automation_blockers_json")
    op.drop_column("evidence_bundles", "automation_eligible")
    op.drop_column("evidence_bundles", "security_findings_json")
    op.drop_column("evidence_bundles", "security_policy_version")
