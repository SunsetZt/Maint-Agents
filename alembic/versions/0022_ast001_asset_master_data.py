"""Add AST-001 customer, site, warranty and installed-component asset facts."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0022_ast001_asset_master_data"
down_revision: str | None = "0021_dia004_expert_intervention"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _tenant() -> sa.Column[object]:
    return sa.Column(
        "tenant_id",
        sa.String(length=64),
        sa.ForeignKey("tenants.id"),
        nullable=False,
    )


def _timestamps() -> tuple[sa.Column[object], sa.Column[object]]:
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


def _source_columns() -> tuple[sa.Column[object], ...]:
    return (
        sa.Column("source_system", sa.String(length=128), nullable=False),
        sa.Column("source_record_id", sa.String(length=255), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
    )


def upgrade() -> None:
    op.add_column("assets", sa.Column("display_name", sa.String(length=255), nullable=True))
    op.add_column("assets", sa.Column("serial_number", sa.String(length=255), nullable=True))
    op.add_column("assets", sa.Column("lifecycle_status", sa.String(length=32), nullable=True))

    op.create_table(
        "asset_customer_links",
        sa.Column("customer_link_id", sa.String(length=128), primary_key=True),
        _tenant(),
        sa.Column(
            "asset_id",
            sa.String(length=128),
            sa.ForeignKey("assets.asset_id"),
            nullable=False,
        ),
        sa.Column("customer_id", sa.String(length=128), nullable=False),
        sa.Column("customer_name", sa.String(length=255), nullable=False),
        *_source_columns(),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "asset_id", name="uq_asset_customer_link"),
    )
    op.create_index("ix_asset_customer_links_tenant_id", "asset_customer_links", ["tenant_id"])
    op.create_index(
        "ix_asset_customer_links_customer",
        "asset_customer_links",
        ["tenant_id", "customer_id"],
    )

    op.create_table(
        "asset_site_links",
        sa.Column("site_link_id", sa.String(length=128), primary_key=True),
        _tenant(),
        sa.Column(
            "asset_id",
            sa.String(length=128),
            sa.ForeignKey("assets.asset_id"),
            nullable=False,
        ),
        sa.Column("site_id", sa.String(length=128), nullable=False),
        sa.Column("site_name", sa.String(length=255), nullable=False),
        sa.Column("address", sa.String(length=1024), nullable=True),
        *_source_columns(),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "asset_id", name="uq_asset_site_link"),
    )
    op.create_index("ix_asset_site_links_tenant_id", "asset_site_links", ["tenant_id"])
    op.create_index("ix_asset_site_links_site", "asset_site_links", ["tenant_id", "site_id"])

    op.create_table(
        "asset_warranties",
        sa.Column("warranty_id", sa.String(length=128), primary_key=True),
        _tenant(),
        sa.Column(
            "asset_id",
            sa.String(length=128),
            sa.ForeignKey("assets.asset_id"),
            nullable=False,
        ),
        sa.Column("contract_number", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("coverage_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("coverage_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("service_level", sa.String(length=128), nullable=True),
        *_source_columns(),
        *_timestamps(),
        sa.UniqueConstraint("tenant_id", "asset_id", name="uq_asset_warranty"),
    )
    op.create_index("ix_asset_warranties_tenant_id", "asset_warranties", ["tenant_id"])
    op.create_index(
        "ix_asset_warranties_contract",
        "asset_warranties",
        ["tenant_id", "contract_number"],
    )

    op.create_table(
        "asset_components",
        sa.Column("component_id", sa.String(length=128), primary_key=True),
        _tenant(),
        sa.Column(
            "asset_id",
            sa.String(length=128),
            sa.ForeignKey("assets.asset_id"),
            nullable=False,
        ),
        sa.Column("part_number", sa.String(length=128), nullable=False),
        sa.Column("part_name", sa.String(length=255), nullable=False),
        sa.Column("serial_number", sa.String(length=255), nullable=True),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("installed_at", sa.DateTime(timezone=True), nullable=True),
        *_source_columns(),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "asset_id",
            "source_system",
            "source_record_id",
            name="uq_asset_component_source",
        ),
    )
    op.create_index("ix_asset_components_tenant_id", "asset_components", ["tenant_id"])
    op.create_index(
        "ix_asset_components_asset",
        "asset_components",
        ["tenant_id", "asset_id", "status"],
    )

    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table_name in (
        "asset_customer_links",
        "asset_site_links",
        "asset_warranties",
        "asset_components",
    ):
        op.execute(f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table_name} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table_name}_tenant_isolation ON {table_name} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    op.drop_index("ix_asset_components_asset", table_name="asset_components")
    op.drop_index("ix_asset_components_tenant_id", table_name="asset_components")
    op.drop_table("asset_components")
    op.drop_index("ix_asset_warranties_contract", table_name="asset_warranties")
    op.drop_index("ix_asset_warranties_tenant_id", table_name="asset_warranties")
    op.drop_table("asset_warranties")
    op.drop_index("ix_asset_site_links_site", table_name="asset_site_links")
    op.drop_index("ix_asset_site_links_tenant_id", table_name="asset_site_links")
    op.drop_table("asset_site_links")
    op.drop_index("ix_asset_customer_links_customer", table_name="asset_customer_links")
    op.drop_index("ix_asset_customer_links_tenant_id", table_name="asset_customer_links")
    op.drop_table("asset_customer_links")
    op.drop_column("assets", "lifecycle_status")
    op.drop_column("assets", "serial_number")
    op.drop_column("assets", "display_name")
