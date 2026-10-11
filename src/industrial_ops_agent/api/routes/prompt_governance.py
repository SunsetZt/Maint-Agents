"""GitOps Prompt Bundle registration and review API."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, Path, Request
from pydantic import BaseModel, Field

from industrial_ops_agent.api.dependencies import get_authorizer, get_database, get_identity
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.prompt_governance.service import (
    PromptBundleConflict,
    PromptBundleGateDenied,
    PromptBundleNotVisible,
    PromptBundleView,
    PromptGovernanceService,
    PromptTransition,
)

router = APIRouter(tags=["prompt-governance"])


class CreatePromptBundleBody(BaseModel):
    prompt_bundle_id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[a-z0-9][a-z0-9._-]+$",
    )
    name: str = Field(min_length=3, max_length=128, pattern=r"^[a-z0-9][a-z0-9._-]+$")
    bundle_version: str = Field(
        min_length=5,
        max_length=64,
        pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[a-z0-9.-]+)?$",
    )
    task_type: Literal["DIAGNOSIS", "VLM", "ASR", "TTS", "RERANKING"]
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    source_path: str = Field(min_length=1, max_length=512)
    section_names: list[str] = Field(min_length=2, max_length=16)
    required_variables: list[str] = Field(min_length=1, max_length=32)
    output_schema_name: str = Field(min_length=1, max_length=128)
    compatible_model_aliases: list[str] = Field(min_length=1, max_length=16)
    change_summary: str = Field(min_length=8, max_length=1_000)


class SubmitPromptBundleBody(BaseModel):
    evaluation_id: str = Field(min_length=1, max_length=128)


class ReviewPromptBundleBody(BaseModel):
    decision: Literal["APPROVED", "REJECTED"]
    reason: str = Field(min_length=8, max_length=1_000)


class RetirePromptBundleBody(BaseModel):
    reason: str = Field(min_length=8, max_length=1_000)


class PromptTransitionResponse(BaseModel):
    transition_id: str
    sequence: int
    from_status: str
    to_status: str
    actor_subject_id: str
    reason_code: str
    evidence: dict[str, Any]
    evidence_hash: str
    occurred_at: datetime


class PromptBundleResponse(BaseModel):
    prompt_bundle_id: str
    origin: Literal["BUILTIN", "REGISTERED"]
    name: str
    bundle_version: str
    task_type: str
    status: str
    content_hash: str
    source_commit: str
    source_path: str
    section_names: list[str]
    required_variables: list[str]
    output_schema_name: str
    compatible_model_aliases: list[str]
    change_summary: str
    evaluation_id: str | None
    evaluation_report_hash: str | None
    created_by_subject_id: str
    reviewed_by_subject_id: str | None
    review_reason: str | None
    reviewed_at: datetime | None
    runtime_available: bool
    runtime_hash_matches: bool
    release_reference_count: int
    active_production_reference: bool
    version: int
    created_at: datetime | None
    updated_at: datetime | None
    legal_actions: list[Literal["SUBMIT", "APPROVE", "REJECT", "RETIRE"]]
    transitions: list[PromptTransitionResponse]


class PromptBundleEnvelope(BaseModel):
    data: PromptBundleResponse
    meta: dict[str, str]


class PromptBundleListEnvelope(BaseModel):
    data: list[PromptBundleResponse]
    legal_actions: list[Literal["CREATE"]]
    meta: dict[str, str | int]


PromptId = Annotated[
    str,
    Path(min_length=3, max_length=128, pattern=r"^[a-z0-9][a-z0-9._-]+$"),
]


@router.get("/prompt-bundles", response_model=PromptBundleListEnvelope)
async def list_prompt_bundles(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> PromptBundleListEnvelope:
    service = PromptGovernanceService(database, authorizer)
    try:
        items = await asyncio.to_thread(
            service.list,
            identity,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "prompt_bundle_forbidden", "authorization") from exc
    can_create = authorizer.decide(
        identity,
        Action.MANAGE_PROMPT_BUNDLE,
        ResourceContext(identity.tenant_id, "prompt-bundles"),
    ).allowed
    return PromptBundleListEnvelope(
        data=[_bundle(item) for item in items],
        legal_actions=["CREATE"] if can_create else [],
        meta={
            "request_id": request.state.request_id,
            "policy_version": service.POLICY_VERSION,
            "total": len(items),
        },
    )


@router.get("/prompt-bundles/{prompt_bundle_id}", response_model=PromptBundleEnvelope)
async def get_prompt_bundle(
    prompt_bundle_id: PromptId,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> PromptBundleEnvelope:
    try:
        item = await asyncio.to_thread(
            PromptGovernanceService(database, authorizer).get,
            identity,
            prompt_bundle_id,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "prompt_bundle_forbidden", "authorization") from exc
    except PromptBundleNotVisible as exc:
        raise _error(404, "prompt_bundle_not_found", "not_found") from exc
    return _envelope(item, request.state.request_id)


@router.post("/prompt-bundles", response_model=PromptBundleEnvelope, status_code=201)
async def create_prompt_bundle(
    body: CreatePromptBundleBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=8, max_length=255),
    ],
) -> PromptBundleEnvelope:
    try:
        item = await asyncio.to_thread(
            PromptGovernanceService(database, authorizer).create,
            identity,
            **body.model_dump(),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "prompt_bundle_forbidden", "authorization") from exc
    except PromptBundleConflict as exc:
        raise _conflict(exc) from exc
    except ValueError as exc:
        raise _error(422, str(exc), "validation") from exc
    return _envelope(item, request.state.request_id)


@router.post(
    "/prompt-bundles/{prompt_bundle_id}/submit",
    response_model=PromptBundleEnvelope,
)
async def submit_prompt_bundle(
    prompt_bundle_id: PromptId,
    body: SubmitPromptBundleBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> PromptBundleEnvelope:
    try:
        item = await asyncio.to_thread(
            PromptGovernanceService(database, authorizer).submit,
            identity,
            prompt_bundle_id,
            evaluation_id=body.evaluation_id,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "prompt_bundle_forbidden", "authorization") from exc
    except PromptBundleNotVisible as exc:
        raise _error(404, "prompt_bundle_not_found", "not_found") from exc
    except PromptBundleConflict as exc:
        raise _conflict(exc) from exc
    except PromptBundleGateDenied as exc:
        raise _error(409, exc.reason, "governance") from exc
    return _envelope(item, request.state.request_id)


@router.post(
    "/prompt-bundles/{prompt_bundle_id}/review",
    response_model=PromptBundleEnvelope,
)
async def review_prompt_bundle(
    prompt_bundle_id: PromptId,
    body: ReviewPromptBundleBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> PromptBundleEnvelope:
    try:
        item = await asyncio.to_thread(
            PromptGovernanceService(database, authorizer).review,
            identity,
            prompt_bundle_id,
            decision=body.decision,
            reason=body.reason,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "prompt_bundle_forbidden", "authorization") from exc
    except PromptBundleNotVisible as exc:
        raise _error(404, "prompt_bundle_not_found", "not_found") from exc
    except PromptBundleConflict as exc:
        raise _conflict(exc) from exc
    except PromptBundleGateDenied as exc:
        raise _error(409, exc.reason, "governance") from exc
    except ValueError as exc:
        raise _error(422, str(exc), "validation") from exc
    return _envelope(item, request.state.request_id)


@router.post(
    "/prompt-bundles/{prompt_bundle_id}/retire",
    response_model=PromptBundleEnvelope,
)
async def retire_prompt_bundle(
    prompt_bundle_id: PromptId,
    body: RetirePromptBundleBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> PromptBundleEnvelope:
    try:
        item = await asyncio.to_thread(
            PromptGovernanceService(database, authorizer).retire,
            identity,
            prompt_bundle_id,
            reason=body.reason,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _error(403, "prompt_bundle_forbidden", "authorization") from exc
    except PromptBundleNotVisible as exc:
        raise _error(404, "prompt_bundle_not_found", "not_found") from exc
    except PromptBundleConflict as exc:
        raise _conflict(exc) from exc
    except PromptBundleGateDenied as exc:
        raise _error(409, exc.reason, "governance") from exc
    except ValueError as exc:
        raise _error(422, str(exc), "validation") from exc
    return _envelope(item, request.state.request_id)


def _bundle(item: PromptBundleView) -> PromptBundleResponse:
    return PromptBundleResponse(
        prompt_bundle_id=item.prompt_bundle_id,
        origin=item.origin,
        name=item.name,
        bundle_version=item.bundle_version,
        task_type=item.task_type,
        status=item.status,
        content_hash=item.content_hash,
        source_commit=item.source_commit,
        source_path=item.source_path,
        section_names=list(item.section_names),
        required_variables=list(item.required_variables),
        output_schema_name=item.output_schema_name,
        compatible_model_aliases=list(item.compatible_model_aliases),
        change_summary=item.change_summary,
        evaluation_id=item.evaluation_id,
        evaluation_report_hash=item.evaluation_report_hash,
        created_by_subject_id=item.created_by_subject_id,
        reviewed_by_subject_id=item.reviewed_by_subject_id,
        review_reason=item.review_reason,
        reviewed_at=item.reviewed_at,
        runtime_available=item.runtime_available,
        runtime_hash_matches=item.runtime_hash_matches,
        release_reference_count=item.release_reference_count,
        active_production_reference=item.active_production_reference,
        version=item.version,
        created_at=item.created_at,
        updated_at=item.updated_at,
        legal_actions=item.legal_actions,
        transitions=[_transition(value) for value in item.transitions],
    )


def _transition(item: PromptTransition) -> PromptTransitionResponse:
    return PromptTransitionResponse(
        transition_id=item.transition_id,
        sequence=item.sequence,
        from_status=item.from_status,
        to_status=item.to_status,
        actor_subject_id=item.actor_subject_id,
        reason_code=item.reason_code,
        evidence=item.evidence,
        evidence_hash=item.evidence_hash,
        occurred_at=item.occurred_at,
    )


def _envelope(item: PromptBundleView, request_id: str) -> PromptBundleEnvelope:
    return PromptBundleEnvelope(
        data=_bundle(item),
        meta={
            "request_id": request_id,
            "policy_version": PromptGovernanceService.POLICY_VERSION,
        },
    )


def _version(value: str) -> int:
    try:
        version = int(value.strip().strip('"'))
    except ValueError as exc:
        raise _error(422, "prompt_bundle_version_invalid", "validation") from exc
    if version < 1:
        raise _error(422, "prompt_bundle_version_invalid", "validation")
    return version


def _conflict(exc: PromptBundleConflict) -> AppError:
    details = {"current_version": exc.current_version} if exc.current_version is not None else None
    return AppError(
        status_code=409,
        code=exc.reason,
        category="conflict",
        message="Prompt Bundle operation conflicted with current state",
        details=details,
    )


def _error(status_code: int, code: str, category: str) -> AppError:
    return AppError(
        status_code=status_code,
        code=code,
        category=category,
        message="Prompt Bundle request was rejected",
    )
