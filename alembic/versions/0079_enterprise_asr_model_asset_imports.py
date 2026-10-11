"""Allow governed ASR candidates in enterprise imports and release onboarding."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0079_enterprise_asr_model_asset_imports"
down_revision: str | None = "0078_enterprise_model_asset_imports"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_IMPORT_TABLE = "enterprise_model_asset_imports"
_IMPORT_CONSTRAINT = "ck_enterprise_model_asset_import_component"
_ONBOARDING_TABLE = "enterprise_model_release_onboardings"
_ONBOARDING_CONSTRAINT = "ck_enterprise_release_onboarding_component"


def upgrade() -> None:
    with op.batch_alter_table(_IMPORT_TABLE) as batch:
        batch.drop_constraint(_IMPORT_CONSTRAINT, type_="check")
        batch.create_check_constraint(
            _IMPORT_CONSTRAINT,
            "component IN ('LLM', 'VLM', 'ASR', 'RUL')",
        )
    with op.batch_alter_table(_ONBOARDING_TABLE) as batch:
        batch.drop_constraint(_ONBOARDING_CONSTRAINT, type_="check")
        batch.create_check_constraint(
            _ONBOARDING_CONSTRAINT,
            "component IN ('LLM', 'VLM', 'ASR', 'RUL')",
        )


def downgrade() -> None:
    connection = op.get_bind()
    asr_count = connection.scalar(
        sa.text("SELECT COUNT(*) FROM enterprise_model_asset_imports WHERE component = 'ASR'")
    )
    asr_onboarding_count = connection.scalar(
        sa.text(
            "SELECT COUNT(*) FROM enterprise_model_release_onboardings "
            "WHERE component = 'ASR'"
        )
    )
    if int(asr_count or 0) > 0 or int(asr_onboarding_count or 0) > 0:
        raise RuntimeError(
            "cannot downgrade while governed ASR imports or release onboardings exist"
        )
    with op.batch_alter_table(_ONBOARDING_TABLE) as batch:
        batch.drop_constraint(_ONBOARDING_CONSTRAINT, type_="check")
        batch.create_check_constraint(
            _ONBOARDING_CONSTRAINT,
            "component IN ('LLM', 'VLM', 'RUL')",
        )
    with op.batch_alter_table(_IMPORT_TABLE) as batch:
        batch.drop_constraint(_IMPORT_CONSTRAINT, type_="check")
        batch.create_check_constraint(
            _IMPORT_CONSTRAINT,
            "component IN ('LLM', 'VLM', 'RUL')",
        )
