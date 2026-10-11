"""Human-confirmed and revocable memory governance API."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel, Field

from industrial_ops_agent.api.dependencies import get_authorizer, get_database, get_identity
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.memory_governance.service import (
    GovernedMemoryService,
    MemoryConflict,
    MemoryNotVisible,
    MemoryTransition,
    MemoryView,
)
from industrial_ops_agent.persistence.database import Database

router = APIRouter(tags=["memory-governance"])


class CreateMemoryBody(BaseModel):
    memory_type: Literal["SESSION_NOTE", "USER_PREFERENCE"]
    content: str = Field(min_length=1, max_length=2_000)
    source_reference_id: str = Field(
        min_length=3,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]+$",
    )
    incident_id: str | None = Field(default=None, min_length=3, max_length=128)
    confirmed: Literal[True]


class RevokeMemoryBody(BaseModel):
    reason: str = Field(min_length=8, max_length=1_000)


class MemoryTransitionResponse(BaseModel):
    transition_id: str
    sequence: int
    from_status: str
    to_status: str
    actor_subject_id: str
    reason_code: str
    evidence_hash: str
    occurred_at: datetime


class GovernedMemoryResponse(BaseModel):
    memory_id: str
    memory_type: Literal["SESSION_NOTE", "USER_PREFERENCE"]
    purpose: Literal["DIAGNOSIS"]
    status: Literal["ACTIVE", "REVOKED"]
    owner_subject_id: str
    incident_id: str | None
    asset_id: str | None
    content: str
    content_hash: str
    source_reference_type: Literal["AGENT_EVENT", "USER_CONFIRMATION"]
    source_reference_id: str
    confirmed_by_subject_id: str
    revoked_by_subject_id: str | None
    revocation_reason: str | None
    revoked_at: datetime | None
    version: int
    created_at: datetime
    updated_at: datetime
    legal_actions: list[Literal["REVOKE"]]
    transitions: list[MemoryTransitionResponse]


class GovernedMemoryEnvelope(BaseModel):
    data: GovernedMemoryResponse
    meta: dict[str, str]


class GovernedMemoryListEnvelope(BaseModel):
    data: list[GovernedMemoryResponse]
    legal_actions: list[Literal["CREATE"]]
    meta: dict[str, str | int]


@router.get("/memories", response_model=GovernedMemoryListEnvelope)
async def list_memories(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    memory_type: Annotated[
        Literal["SESSION_NOTE", "USER_PREFERENCE"] | None,
        Query(),
    ] = None,
    status: Annotated[Literal["ACTIVE", "REVOKED"] | None, Query()] = None,
    incident_id: Annotated[str | None, Query(min_length=3, max_length=128)] = None,
) -> GovernedMemoryListEnvelope:
    service = GovernedMemoryService(database, authorizer)
    try:
        items = await asyncio.to_thread(
            service.list,
            identity,
            memory_type=memory_type,
            status=status,
            incident_id=incident_id,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "memory_forbidden", "authorization") from exc
    can_create = authorizer.decide(
        identity,
        Action.WRITE_MEMORY,
        ResourceContext(identity.tenant_id, "governed-memories"),
    ).allowed
    return GovernedMemoryListEnvelope(
        data=[_memory(item) for item in items],
        legal_actions=["CREATE"] if can_create else [],
        meta={
            "request_id": request.state.request_id,
            "policy_version": service.POLICY_VERSION,
            "total": len(items),
        },
    )


@router.post("/memories", response_model=GovernedMemoryEnvelope, status_code=201)
async def create_memory(
    body: CreateMemoryBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=8, max_length=255),
    ],
) -> GovernedMemoryEnvelope:
    try:
        item = await asyncio.to_thread(
            GovernedMemoryService(database, authorizer).create,
            identity,
            **body.model_dump(),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "memory_forbidden", "authorization") from exc
    except MemoryNotVisible as exc:
        raise _error(404, "memory_source_not_found", "not_found") from exc
    except MemoryConflict as exc:
        raise _conflict(exc) from exc
    except ValueError as exc:
        raise _error(422, str(exc), "validation") from exc
    return _envelope(item, request.state.request_id)


@router.post(
    "/memories/{memory_id}/revoke",
    response_model=GovernedMemoryEnvelope,
)
async def revoke_memory(
    memory_id: str,
    body: RevokeMemoryBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> GovernedMemoryEnvelope:
    try:
        item = await asyncio.to_thread(
            GovernedMemoryService(database, authorizer).revoke,
            identity,
            memory_id,
            expected_version=_version(if_match),
            reason=body.reason,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "memory_forbidden", "authorization") from exc
    except MemoryNotVisible as exc:
        raise _error(404, "memory_not_found", "not_found") from exc
    except MemoryConflict as exc:
        raise _conflict(exc) from exc
    except ValueError as exc:
        raise _error(422, str(exc), "validation") from exc
    return _envelope(item, request.state.request_id)


def _memory(item: MemoryView) -> GovernedMemoryResponse:
    payload: dict[str, Any] = {
        **{
            field: getattr(item, field)
            for field in GovernedMemoryResponse.model_fields
            if field not in {"legal_actions", "transitions"}
        },
        "legal_actions": list(item.legal_actions),
        "transitions": [_transition(value) for value in item.transitions],
    }
    return GovernedMemoryResponse.model_validate(payload)


def _transition(item: MemoryTransition) -> MemoryTransitionResponse:
    return MemoryTransitionResponse.model_validate(
        {field: getattr(item, field) for field in MemoryTransitionResponse.model_fields}
    )


def _envelope(item: MemoryView, request_id: str) -> GovernedMemoryEnvelope:
    return GovernedMemoryEnvelope(data=_memory(item), meta={"request_id": request_id})


def _version(value: str) -> int:
    try:
        parsed = int(value.strip().strip('"'))
    except ValueError as exc:
        raise _error(400, "invalid_version_precondition", "validation") from exc
    if parsed <= 0:
        raise _error(400, "invalid_version_precondition", "validation")
    return parsed


def _conflict(exc: MemoryConflict) -> AppError:
    details = {"current_version": exc.current_version} if exc.current_version else None
    return AppError(409, exc.reason, "conflict", "Memory state conflict", details=details)


def _error(status_code: int, code: str, category: str) -> AppError:
    return AppError(status_code, code, category, "Memory governance request failed")
