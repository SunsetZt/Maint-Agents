"""Allow governed Embedding candidates in enterprise imports and release onboarding."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0084_enterprise_embedding_model_asset_governance"
down_revision: str | None = "0083_enterprise_reranker_model_asset_governance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_IMPORT_TABLE = "enterprise_model_asset_imports"
_IMPORT_CONSTRAINT = "ck_enterprise_model_asset_import_component"
_ONBOARDING_TABLE = "enterprise_model_release_onboardings"
_ONBOARDING_CONSTRAINT = "ck_enterprise_release_onboarding_component"
_EXPANDED_COMPONENTS = (
    "component IN ('LLM', 'VLM', 'ASR', 'TTS', 'RUL', 'EMBEDDING', 'RERANKER')"
)
_PREVIOUS_COMPONENTS = (
    "component IN ('LLM', 'VLM', 'ASR', 'TTS', 'RUL', 'RERANKER')"
)


def upgrade() -> None:
    with op.batch_alter_table(_IMPORT_TABLE) as batch:
        batch.drop_constraint(_IMPORT_CONSTRAINT, type_="check")
        batch.create_check_constraint(_IMPORT_CONSTRAINT, _EXPANDED_COMPONENTS)
    with op.batch_alter_table(_ONBOARDING_TABLE) as batch:
        batch.drop_constraint(_ONBOARDING_CONSTRAINT, type_="check")
        batch.create_check_constraint(_ONBOARDING_CONSTRAINT, _EXPANDED_COMPONENTS)


def downgrade() -> None:
    connection = op.get_bind()
    embedding_import_count = connection.scalar(
        sa.text(
            "SELECT COUNT(*) FROM enterprise_model_asset_imports "
            "WHERE component = 'EMBEDDING'"
        )
    )
    embedding_onboarding_count = connection.scalar(
        sa.text(
            "SELECT COUNT(*) FROM enterprise_model_release_onboardings "
            "WHERE component = 'EMBEDDING'"
        )
    )
    if int(embedding_import_count or 0) > 0 or int(embedding_onboarding_count or 0) > 0:
        raise RuntimeError(
            "cannot downgrade while governed Embedding imports or release onboardings exist"
        )
    with op.batch_alter_table(_ONBOARDING_TABLE) as batch:
        batch.drop_constraint(_ONBOARDING_CONSTRAINT, type_="check")
        batch.create_check_constraint(_ONBOARDING_CONSTRAINT, _PREVIOUS_COMPONENTS)
    with op.batch_alter_table(_IMPORT_TABLE) as batch:
        batch.drop_constraint(_IMPORT_CONSTRAINT, type_="check")
        batch.create_check_constraint(_IMPORT_CONSTRAINT, _PREVIOUS_COMPONENTS)
