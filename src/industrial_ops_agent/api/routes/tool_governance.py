"""Read-only enterprise Tool Registry and invocation governance API."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_identity,
    get_tool_gateway,
)
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.tool_governance.service import (
    AuthorityDeclaration,
    GovernedTool,
    RecentToolCall,
    ToolGovernanceService,
    ToolGovernanceSnapshot,
    ToolUsage,
)
from industrial_ops_agent.tools.gateway import ToolGateway

router = APIRouter(tags=["tool-governance"])


class AuthorityDeclarationResponse(BaseModel):
    contract_version: str
    domain: str
    owner: str
    fields: list[str]
    enterprise_source: str
    synthetic_source: str


class ToolUsageResponse(BaseModel):
    call_count: int
    succeeded_count: int
    failed_count: int
    approval_required_count: int
    external_handoff_count: int
    latest_call_at: datetime | None


class GovernedToolResponse(BaseModel):
    tool_id: str
    version: str
    risk_tier: Literal["T0", "T1", "T2", "T3"]
    required_action: str
    approval_required: bool
    idempotent: bool
    timeout_seconds: int
    max_attempts: int
    rate_limit_per_minute: int
    compensation: str
    execution_policy: Literal[
        "AUTO_READ_ONLY",
        "CONTROLLED_REVERSIBLE",
        "HUMAN_APPROVAL_REQUIRED",
        "EXTERNAL_HANDOFF_ONLY",
        "DENY_UNKNOWN_RISK",
    ]
    input_schema: dict[str, Any]
    authority_coverage: Literal["DECLARED", "NOT_APPLICABLE", "MISSING"]
    authority: AuthorityDeclarationResponse | None
    transport: str | None
    remote_tool_name: str | None
    server_id: str | None
    server_version: str | None
    required_scopes: list[str]
    input_schema_digest: str | None
    usage: ToolUsageResponse


class RecentToolCallResponse(BaseModel):
    tool_call_id: str
    agent_run_id: str | None
    tool_id: str
    tool_version: str
    risk_tier: str
    status: str
    occurred_at: datetime
    source: str | None
    source_as_of: datetime | None
    authority_evidence: Literal[
        "VERIFIED",
        "NOT_APPLICABLE",
        "NOT_RECORDED",
        "MISMATCH",
    ]
    reason_code: str | None
    transport: str | None
    server_id: str | None
    server_version: str | None


class ToolGovernanceTotalsResponse(BaseModel):
    registered_tool_count: int
    authority_declared_count: int
    call_count: int
    succeeded_count: int
    failed_count: int
    approval_required_count: int
    external_handoff_count: int
    active_agent_run_count: int
    unregistered_audit_count: int
    mcp_tool_count: int
    mcp_server_count: int
    risk_counts: dict[str, int]


class ToolGovernanceResponse(BaseModel):
    policy_version: str
    generated_at: datetime
    window_start: datetime
    window_end: datetime
    totals: ToolGovernanceTotalsResponse
    tools: list[GovernedToolResponse]
    recent_calls: list[RecentToolCallResponse]
    anomalies: list[str]


class ToolGovernanceEnvelope(BaseModel):
    data: ToolGovernanceResponse
    meta: dict[str, str]


@router.get("/operations/tools/governance", response_model=ToolGovernanceEnvelope)
async def get_tool_governance(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    window_hours: Annotated[int, Query(ge=1, le=24 * 30)] = 24,
    recent_limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> ToolGovernanceEnvelope:
    try:
        snapshot = await asyncio.to_thread(
            ToolGovernanceService(
                database,
                authorizer,
                gateway.definitions(),
                gateway.transport_bindings(),
            ).snapshot,
            identity,
            request_id=request.state.request_id,
            window_hours=window_hours,
            recent_limit=recent_limit,
        )
    except AuthorizationDenied as exc:
        raise AppError(
            status_code=403,
            code="tool_governance_forbidden",
            category="authorization",
            message="Tool governance request was rejected",
        ) from exc
    return ToolGovernanceEnvelope(
        data=_snapshot(snapshot),
        meta={"request_id": request.state.request_id},
    )


def _snapshot(snapshot: ToolGovernanceSnapshot) -> ToolGovernanceResponse:
    return ToolGovernanceResponse(
        policy_version=snapshot.policy_version,
        generated_at=snapshot.generated_at,
        window_start=snapshot.window_start,
        window_end=snapshot.window_end,
        totals=ToolGovernanceTotalsResponse.model_validate(
            snapshot.totals,
            from_attributes=True,
        ),
        tools=[_tool(item) for item in snapshot.tools],
        recent_calls=[_call(item) for item in snapshot.recent_calls],
        anomalies=list(snapshot.anomalies),
    )


def _tool(item: GovernedTool) -> GovernedToolResponse:
    return GovernedToolResponse(
        tool_id=item.tool_id,
        version=item.version,
        risk_tier=item.risk_tier,
        required_action=item.required_action,
        approval_required=item.approval_required,
        idempotent=item.idempotent,
        timeout_seconds=item.timeout_seconds,
        max_attempts=item.max_attempts,
        rate_limit_per_minute=item.rate_limit_per_minute,
        compensation=item.compensation,
        execution_policy=item.execution_policy,
        input_schema=item.input_schema,
        authority_coverage=item.authority_coverage,
        authority=_authority(item.authority) if item.authority is not None else None,
        transport=item.transport,
        remote_tool_name=item.remote_tool_name,
        server_id=item.server_id,
        server_version=item.server_version,
        required_scopes=list(item.required_scopes),
        input_schema_digest=item.input_schema_digest,
        usage=_usage(item.usage),
    )


def _authority(item: AuthorityDeclaration) -> AuthorityDeclarationResponse:
    return AuthorityDeclarationResponse(
        contract_version=item.contract_version,
        domain=item.domain,
        owner=item.owner,
        fields=list(item.fields),
        enterprise_source=item.enterprise_source,
        synthetic_source=item.synthetic_source,
    )


def _usage(item: ToolUsage) -> ToolUsageResponse:
    return ToolUsageResponse.model_validate(item, from_attributes=True)


def _call(item: RecentToolCall) -> RecentToolCallResponse:
    return RecentToolCallResponse(
        tool_call_id=item.tool_call_id,
        agent_run_id=item.agent_run_id,
        tool_id=item.tool_id,
        tool_version=item.tool_version,
        risk_tier=item.risk_tier,
        status=item.status,
        occurred_at=item.occurred_at,
        source=item.source,
        source_as_of=item.source_as_of,
        authority_evidence=item.authority_evidence,
        reason_code=item.reason_code,
        transport=item.transport,
        server_id=item.server_id,
        server_version=item.server_version,
    )
