"""Shared assignee eligibility rule for manual and Agent-proposed dispatch."""

from __future__ import annotations

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import Role
from industrial_ops_agent.persistence.models import (
    AssetScopeRecord,
    SubjectRecord,
    SubjectRoleRecord,
)


def is_eligible_field_engineer(
    session: Session,
    *,
    tenant_id: str,
    subject_id: str,
    asset_id: str,
    site_id: str | None,
) -> bool:
    """Require an active field engineer with an explicit subject/role resource scope."""

    resource_scope = AssetScopeRecord.asset_id == asset_id
    if site_id is not None:
        resource_scope = or_(resource_scope, AssetScopeRecord.site_id == site_id)
    scope_owner = or_(
        AssetScopeRecord.subject_id == subject_id,
        and_(
            AssetScopeRecord.subject_id.is_(None),
            AssetScopeRecord.role == Role.FIELD_ENGINEER.value,
        ),
    )
    eligible = session.scalar(
        select(SubjectRecord.subject_id)
        .join(
            SubjectRoleRecord,
            and_(
                SubjectRoleRecord.tenant_id == SubjectRecord.tenant_id,
                SubjectRoleRecord.subject_id == SubjectRecord.subject_id,
            ),
        )
        .join(
            AssetScopeRecord,
            and_(
                AssetScopeRecord.tenant_id == SubjectRecord.tenant_id,
                scope_owner,
                resource_scope,
            ),
        )
        .where(
            SubjectRecord.tenant_id == tenant_id,
            SubjectRecord.subject_id == subject_id,
            func.lower(SubjectRecord.status) == "active",
            SubjectRoleRecord.role == Role.FIELD_ENGINEER.value,
        )
        .limit(1)
    )
    return eligible is not None
