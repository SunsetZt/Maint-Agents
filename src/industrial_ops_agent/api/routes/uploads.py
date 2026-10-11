"""Quarantine upload and media scan-status API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Header, Request
from pydantic import BaseModel

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_identity,
    get_object_store,
)
from industrial_ops_agent.api.errors import STANDARD_ERROR_RESPONSES, AppError
from industrial_ops_agent.application.assets import ResourceNotVisible
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.media.models import MediaObject
from industrial_ops_agent.media.service import (
    MediaService,
    MediaUploadRejected,
    TenantObjectStore,
)
from industrial_ops_agent.persistence.database import Database

router = APIRouter(tags=["media"])


class MediaStatusResponse(BaseModel):
    media_id: str
    draft_id: str
    declared_mime: str
    detected_mime: str | None
    size_bytes: int
    scan_state: str
    scan_version: int
    content_credential_status: str
    content_credential_report: dict[str, Any] | None
    content_credential_digest: str | None
    content_credential_inspected_at: datetime | None
    created_at: datetime
    updated_at: datetime


class MediaEnvelope(BaseModel):
    data: MediaStatusResponse
    meta: dict[str, str]


@router.post(
    "/incident-drafts/{draft_id}/media",
    response_model=MediaEnvelope,
    status_code=201,
    responses=STANDARD_ERROR_RESPONSES,
)
async def upload_media(
    draft_id: str,
    request: Request,
    content: Annotated[bytes, Body(media_type="application/octet-stream")],
    declared_mime: Annotated[str, Header(alias="Content-Type")],
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    object_store: Annotated[TenantObjectStore, Depends(get_object_store)],
) -> MediaEnvelope:
    service = MediaService(database, authorizer, object_store)
    try:
        media = await service.upload(
            identity,
            draft_id,
            declared_mime=declared_mime,
            content=content,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    except MediaUploadRejected as exc:
        status_code = 415 if exc.reason_code == "mime_type_mismatch" else 422
        raise AppError(
            status_code=status_code,
            code=exc.reason_code,
            category="validation",
            message="Media upload failed validation",
            details={
                "media_id": exc.media.media_id,
                "scan_state": exc.media.scan_state.value,
            },
        ) from exc
    return _envelope(media, request.state.request_id)


@router.get(
    "/media/{media_id}/status",
    response_model=MediaEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def media_status(
    media_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    object_store: Annotated[TenantObjectStore, Depends(get_object_store)],
) -> MediaEnvelope:
    service = MediaService(database, authorizer, object_store)
    try:
        media = service.get_status(
            identity,
            media_id,
            request_id=request.state.request_id,
        )
    except ResourceNotVisible as exc:
        raise _not_visible() from exc
    return _envelope(media, request.state.request_id)


def _envelope(media: MediaObject, request_id: str) -> MediaEnvelope:
    return MediaEnvelope(
        data=MediaStatusResponse(
            media_id=media.media_id,
            draft_id=media.draft_id,
            declared_mime=media.declared_mime,
            detected_mime=media.detected_mime,
            size_bytes=media.size_bytes,
            scan_state=media.scan_state.value,
            scan_version=media.scan_version,
            content_credential_status=media.content_credential_status.value,
            content_credential_report=media.content_credential_report,
            content_credential_digest=media.content_credential_digest,
            content_credential_inspected_at=media.content_credential_inspected_at,
            created_at=media.created_at,
            updated_at=media.updated_at,
        ),
        meta={"request_id": request_id},
    )


def _not_visible() -> AppError:
    return AppError(
        status_code=404,
        code="resource_not_found_or_not_visible",
        category="not_found",
        message="Resource not found or not visible",
    )
