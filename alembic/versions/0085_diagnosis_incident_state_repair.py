"""Repair Incident state after a successfully completed latest diagnosis."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0085_diagnosis_incident_state_repair"
down_revision: str | None = "0084_enterprise_embedding_model_asset_governance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            UPDATE incidents
            SET status = 'DIAGNOSED',
                version = version + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE status IN ('TRIAGED', 'DIAGNOSING')
              AND EXISTS (
                  SELECT 1
                  FROM diagnosis_runs AS completed
                  WHERE completed.tenant_id = incidents.tenant_id
                    AND completed.incident_id = incidents.incident_id
                    AND completed.status = 'COMPLETED'
                    AND NOT EXISTS (
                        SELECT 1
                        FROM diagnosis_runs AS newer
                        WHERE newer.tenant_id = completed.tenant_id
                          AND newer.incident_id = completed.incident_id
                          AND (
                              newer.created_at > completed.created_at
                              OR (
                                  newer.created_at = completed.created_at
                                  AND newer.diagnosis_run_id > completed.diagnosis_run_id
                              )
                          )
                    )
              )
            """
        )
    )


def downgrade() -> None:
    # A repaired Incident may already have progressed to a WorkOrder or resolution.
    # Reverting business state during a schema downgrade would corrupt that history.
    pass
