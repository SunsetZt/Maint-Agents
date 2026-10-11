"""Bind synthetic snapshot derivation and VLM ablation evaluation evidence."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0032_synthetic_vlm_ablation"
down_revision: str | None = "0031_governed_memory"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "dataset_snapshots",
        sa.Column("base_snapshot_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "dataset_snapshots",
        sa.Column("augmentation_contract_version", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "dataset_snapshots",
        sa.Column("sample_origin_counts", sa.JSON(), nullable=True),
    )
    op.add_column(
        "dataset_snapshots",
        sa.Column("synthetic_split_counts", sa.JSON(), nullable=True),
    )
    op.create_foreign_key(
        "fk_dataset_snapshots_base_snapshot_id",
        "dataset_snapshots",
        "dataset_snapshots",
        ["base_snapshot_id"],
        ["snapshot_id"],
    )
    op.create_index(
        "ix_dataset_snapshots_base_snapshot",
        "dataset_snapshots",
        ["tenant_id", "base_snapshot_id"],
    )

    op.add_column(
        "model_evaluation_runs",
        sa.Column(
            "comparison_kind",
            sa.String(length=64),
            nullable=False,
            server_default="MODEL_METHOD",
        ),
    )
    op.add_column(
        "model_evaluation_runs",
        sa.Column(
            "comparison_context",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::json"),
        ),
    )
    op.alter_column("model_evaluation_runs", "comparison_kind", server_default=None)
    op.alter_column("model_evaluation_runs", "comparison_context", server_default=None)


def downgrade() -> None:
    op.drop_column("model_evaluation_runs", "comparison_context")
    op.drop_column("model_evaluation_runs", "comparison_kind")
    op.drop_index("ix_dataset_snapshots_base_snapshot", table_name="dataset_snapshots")
    op.drop_constraint(
        "fk_dataset_snapshots_base_snapshot_id",
        "dataset_snapshots",
        type_="foreignkey",
    )
    op.drop_column("dataset_snapshots", "synthetic_split_counts")
    op.drop_column("dataset_snapshots", "sample_origin_counts")
    op.drop_column("dataset_snapshots", "augmentation_contract_version")
    op.drop_column("dataset_snapshots", "base_snapshot_id")
