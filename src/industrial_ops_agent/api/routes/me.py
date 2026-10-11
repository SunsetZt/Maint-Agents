"""Safe projection of the verified API identity context."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from industrial_ops_agent.api.dependencies import get_identity
from industrial_ops_agent.auth.identity import IdentityContext

router = APIRouter(tags=["identity"])


class IdentityResponse(BaseModel):
    subject_id: str
    tenant_id: str
    roles: list[str]
    asset_ids: list[str]
    site_ids: list[str]
    expires_at: datetime


@router.get("/me", response_model=IdentityResponse)
async def me(
    identity: Annotated[IdentityContext, Depends(get_identity)],
) -> IdentityResponse:
    return IdentityResponse(
        subject_id=identity.subject_id,
        tenant_id=identity.tenant_id,
        roles=sorted(role.value for role in identity.roles),
        asset_ids=sorted(identity.asset_ids),
        site_ids=sorted(identity.site_ids),
        expires_at=identity.expires_at,
    )
