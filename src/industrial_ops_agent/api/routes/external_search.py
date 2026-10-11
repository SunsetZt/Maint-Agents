"""Governed public-reference search API; results stay separate from enterprise knowledge."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Any, Literal, cast

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel, Field

from industrial_ops_agent.api.dependencies import get_authorizer, get_database, get_identity
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.external_search.provider import ExternalSearchProvider
from industrial_ops_agent.external_search.service import (
    ExternalSearchError,
    ExternalSearchPolicyView,
    ExternalSearchQueryView,
    ExternalSearchService,
)
from industrial_ops_agent.persistence.database import Database

router = APIRouter(prefix="/external-search", tags=["external-search"])


class ExternalSearchPolicyBody(BaseModel):
    enabled: bool
    allowed_domains: list[str] = Field(max_length=50)
    official_domains: list[str] = Field(max_length=50)
    max_results: int = Field(ge=1, le=10)


class ExternalSearchPolicyResponse(BaseModel):
    enabled: bool
    allowed_domains: list[str]
    official_domains: list[str]
    max_results: int
    updated_by_subject_id: str | None
    version: int
    created_at: datetime | None
    updated_at: datetime | None
    runtime_available: bool
    legal_actions: list[Literal["RUN_SEARCH", "UPDATE_POLICY"]]


class ExternalSearchPolicyEnvelope(BaseModel):
    data: ExternalSearchPolicyResponse
    meta: dict[str, str]


class ExternalSearchBody(BaseModel):
    query_text: str = Field(min_length=3, max_length=500)
    use_case: Literal[
        "GENERAL_REFERENCE",
        "REPAIR_REFERENCE",
        "SAFETY_REFERENCE",
        "WARRANTY_REFERENCE",
    ]


class ExternalSearchConclusionBody(BaseModel):
    conclusion: Literal["NOT_USED", "REFERENCE_ONLY", "ESCALATED_TO_KNOWLEDGE_REVIEW"]
    reason: str = Field(min_length=3, max_length=1000)


class ExternalReferenceResponse(BaseModel):
    external_reference_id: str
    rank: int
    title: str
    url: str
    domain: str
    summary: str
    content_hash: str
    relevance_score: float
    trust_level: Literal["OFFICIAL", "ALLOWLISTED"]
    published_at: datetime | None
    guardrail_policy_version: str
    fetched_at: datetime


class ExternalSearchQueryResponse(BaseModel):
    external_search_id: str
    query_text: str
    query_hash: str
    use_case: str
    status: Literal["RUNNING", "SUCCEEDED", "NO_RESULTS", "BLOCKED", "FAILED"]
    provider: str
    provider_request_id: str | None
    requested_by_subject_id: str
    result_count: int
    blocked_result_count: int
    usage_credits: int | None
    failure_code: str | None
    usage_conclusion: Literal[
        "PENDING_REVIEW",
        "NOT_USED",
        "REFERENCE_ONLY",
        "ESCALATED_TO_KNOWLEDGE_REVIEW",
    ]
    conclusion_reason: str | None
    concluded_by_subject_id: str | None
    started_at: datetime
    completed_at: datetime | None
    version: int
    created_at: datetime
    updated_at: datetime
    references: list[ExternalReferenceResponse]
    legal_actions: list[Literal["CONCLUDE"]]


class ExternalSearchQueryEnvelope(BaseModel):
    data: ExternalSearchQueryResponse
    meta: dict[str, str]


class ExternalSearchQueryListEnvelope(BaseModel):
    data: list[ExternalSearchQueryResponse]
    meta: dict[str, str | int]


@router.get("/policy", response_model=ExternalSearchPolicyEnvelope)
async def get_external_search_policy(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> ExternalSearchPolicyEnvelope:
    try:
        item = await asyncio.to_thread(
            _service(request, database, authorizer).get_policy,
            identity,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, ExternalSearchError) as exc:
        raise _translate(exc) from exc
    return _policy_envelope(request, identity, authorizer, item)


@router.put("/policy", response_model=ExternalSearchPolicyEnvelope)
async def update_external_search_policy(
    body: ExternalSearchPolicyBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> ExternalSearchPolicyEnvelope:
    try:
        item = await asyncio.to_thread(
            _service(request, database, authorizer).update_policy,
            identity,
            enabled=body.enabled,
            allowed_domains=tuple(body.allowed_domains),
            official_domains=tuple(body.official_domains),
            max_results=body.max_results,
            expected_version=_version(if_match, allow_zero=True),
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, ExternalSearchError) as exc:
        raise _translate(exc) from exc
    return _policy_envelope(request, identity, authorizer, item)


@router.post("/queries", response_model=ExternalSearchQueryEnvelope, status_code=201)
async def run_external_search(
    body: ExternalSearchBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=8, max_length=255)],
) -> ExternalSearchQueryEnvelope:
    try:
        item = await asyncio.to_thread(
            _service(request, database, authorizer).search,
            identity,
            query_text=body.query_text,
            use_case=body.use_case,
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, ExternalSearchError) as exc:
        raise _translate(exc) from exc
    return _query_envelope(item, request.state.request_id, identity, authorizer)


@router.get("/queries", response_model=ExternalSearchQueryListEnvelope)
async def list_external_searches(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> ExternalSearchQueryListEnvelope:
    try:
        items = await asyncio.to_thread(
            _service(request, database, authorizer).list,
            identity,
            limit=limit,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, ExternalSearchError) as exc:
        raise _translate(exc) from exc
    return ExternalSearchQueryListEnvelope(
        data=[_query_response(item, identity, authorizer) for item in items],
        meta={"request_id": request.state.request_id, "total": len(items)},
    )


@router.get("/queries/{external_search_id}", response_model=ExternalSearchQueryEnvelope)
async def get_external_search(
    external_search_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> ExternalSearchQueryEnvelope:
    try:
        item = await asyncio.to_thread(
            _service(request, database, authorizer).get,
            identity,
            external_search_id,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, ExternalSearchError) as exc:
        raise _translate(exc) from exc
    return _query_envelope(item, request.state.request_id, identity, authorizer)


@router.post(
    "/queries/{external_search_id}/conclusion",
    response_model=ExternalSearchQueryEnvelope,
)
async def conclude_external_search(
    external_search_id: str,
    body: ExternalSearchConclusionBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> ExternalSearchQueryEnvelope:
    try:
        item = await asyncio.to_thread(
            _service(request, database, authorizer).conclude,
            identity,
            external_search_id,
            conclusion=body.conclusion,
            reason=body.reason,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, ExternalSearchError) as exc:
        raise _translate(exc) from exc
    return _query_envelope(item, request.state.request_id, identity, authorizer)


def _service(
    request: Request,
    database: Database,
    authorizer: Authorizer,
) -> ExternalSearchService:
    provider: ExternalSearchProvider | None = getattr(
        request.app.state, "external_search_provider", None
    )
    return ExternalSearchService(database, authorizer, provider)


def _policy_envelope(
    request: Request,
    identity: IdentityContext,
    authorizer: Authorizer,
    item: ExternalSearchPolicyView,
) -> ExternalSearchPolicyEnvelope:
    can_search = authorizer.decide(
        identity,
        Action.RUN_EXTERNAL_SEARCH,
        ResourceContext(identity.tenant_id, "external-search"),
    ).allowed
    can_update = authorizer.decide(
        identity,
        Action.MANAGE_EXTERNAL_SEARCH_POLICY,
        ResourceContext(identity.tenant_id, "external-search-policy"),
    ).allowed
    legal_actions: list[Literal["RUN_SEARCH", "UPDATE_POLICY"]] = []
    if can_search:
        legal_actions.append("RUN_SEARCH")
    if can_update:
        legal_actions.append("UPDATE_POLICY")
    provider = getattr(request.app.state, "external_search_provider", None)
    return ExternalSearchPolicyEnvelope(
        data=ExternalSearchPolicyResponse(
            **asdict(item),
            runtime_available=provider is not None,
            legal_actions=legal_actions,
        ),
        meta={"request_id": request.state.request_id},
    )


def _query_response(
    item: ExternalSearchQueryView,
    identity: IdentityContext,
    authorizer: Authorizer,
) -> ExternalSearchQueryResponse:
    can_conclude = (
        item.status != "RUNNING"
        and authorizer.decide(
            identity,
            Action.RUN_EXTERNAL_SEARCH,
            ResourceContext(identity.tenant_id, item.external_search_id),
        ).allowed
        and (
            identity.subject_id == item.requested_by_subject_id
            or "domain_expert" in {role.value for role in identity.roles}
        )
    )
    payload: dict[str, Any] = asdict(item)
    payload["status"] = cast(
        Literal["RUNNING", "SUCCEEDED", "NO_RESULTS", "BLOCKED", "FAILED"],
        item.status,
    )
    payload["usage_conclusion"] = cast(
        Literal[
            "PENDING_REVIEW",
            "NOT_USED",
            "REFERENCE_ONLY",
            "ESCALATED_TO_KNOWLEDGE_REVIEW",
        ],
        item.usage_conclusion,
    )
    payload["references"] = [
        ExternalReferenceResponse(
            **{
                **asdict(reference),
                "trust_level": cast(
                    Literal["OFFICIAL", "ALLOWLISTED"], reference.trust_level
                ),
            }
        )
        for reference in item.references
    ]
    return ExternalSearchQueryResponse(
        **payload,
        legal_actions=["CONCLUDE"] if can_conclude else [],
    )


def _query_envelope(
    item: ExternalSearchQueryView,
    request_id: str,
    identity: IdentityContext,
    authorizer: Authorizer,
) -> ExternalSearchQueryEnvelope:
    return ExternalSearchQueryEnvelope(
        data=_query_response(item, identity, authorizer),
        meta={"request_id": request_id},
    )


def _version(value: str, *, allow_zero: bool = False) -> int:
    normalized = value.strip()
    if normalized.startswith("W/"):
        normalized = normalized[2:].strip()
    normalized = normalized.strip('"')
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise _error(400, "invalid_if_match", "validation") from exc
    if parsed < (0 if allow_zero else 1):
        raise _error(400, "invalid_if_match", "validation")
    return parsed


def _translate(exc: Exception) -> AppError:
    if isinstance(exc, AuthorizationDenied):
        return _error(403, "external_search_forbidden", "authorization")
    if isinstance(exc, ExternalSearchError):
        if exc.reason == "external_search_not_visible":
            return _error(404, exc.reason, "not_found")
        conflicts = {
            "external_search_disabled_by_tenant_policy",
            "external_search_official_domain_required",
            "external_search_idempotency_conflict",
            "external_search_policy_state_changed",
            "external_search_state_changed",
            "external_search_conclusion_forbidden",
        }
        return _error(
            409 if exc.reason in conflicts else 400,
            exc.reason,
            "conflict" if exc.reason in conflicts else "validation",
            details=(
                {"current_version": exc.current_version}
                if exc.current_version is not None
                else None
            ),
        )
    return _error(500, "external_search_failed", "internal")


def _error(
    status: int,
    code: str,
    category: str,
    *,
    details: dict[str, Any] | None = None,
) -> AppError:
    return AppError(
        status_code=status,
        code=code,
        category=category,
        message=code.replace("_", " ").capitalize(),
        details=details,
    )
