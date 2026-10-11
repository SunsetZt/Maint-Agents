"""Restore PostgreSQL RLS coverage for tenant-scoped tables."""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0068_restore_tenant_rls_coverage"
down_revision: str | None = "0067_governed_remote_expert_outcome_handoff"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TENANT_TABLES = (
    "knowledge_graph_extraction_jobs",
    "work_order_expert_collaborations",
    "work_order_expert_collaboration_events",
    "work_order_expert_collaboration_recommendations",
)


def _policy_name(table_name: str) -> str:
    return f"{table_name}_tenant_isolation"[:63]


def upgrade() -> None:
    predicate = "tenant_id = current_setting('app.tenant_id', true)"
    for table_name in _TENANT_TABLES:
        policy_name = _policy_name(table_name)
        op.execute(f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table_name} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {policy_name} ON {table_name} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


def downgrade() -> None:
    for table_name in reversed(_TENANT_TABLES):
        policy_name = _policy_name(table_name)
        op.execute(f"DROP POLICY IF EXISTS {policy_name} ON {table_name}")
        op.execute(f"ALTER TABLE {table_name} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table_name} DISABLE ROW LEVEL SECURITY")
