"""Registry, authority, safety posture, and tenant-scoped tool-call audit views."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Literal, cast

from sqlalchemy import distinct, func, select

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _utc
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import ToolCallRecord
from industrial_ops_agent.tools.authority import (
    AUTHORITY_CONTRACT_VERSION,
    TOOL_FACT_AUTHORITIES,
    ToolFactAuthority,
)
from industrial_ops_agent.tools.contracts import ToolTransportBinding
from industrial_ops_agent.tools.registry import ToolDefinition, ToolRegistry

_REASON_CODE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")

RiskTier = Literal["T0", "T1", "T2", "T3"]
ExecutionPolicy = Literal[
    "AUTO_READ_ONLY",
    "CONTROLLED_REVERSIBLE",
    "HUMAN_APPROVAL_REQUIRED",
    "EXTERNAL_HANDOFF_ONLY",
    "DENY_UNKNOWN_RISK",
]


class AuthorityCoverage(StrEnum):
    DECLARED = "DECLARED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    MISSING = "MISSING"


class AuthorityEvidence(StrEnum):
    VERIFIED = "VERIFIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_RECORDED = "NOT_RECORDED"
    MISMATCH = "MISMATCH"


@dataclass(frozen=True, slots=True)
class AuthorityDeclaration:
    contract_version: str
    domain: str
    owner: str
    fields: tuple[str, ...]
    enterprise_source: str
    synthetic_source: str


@dataclass(frozen=True, slots=True)
class ToolUsage:
    call_count: int
    succeeded_count: int
    failed_count: int
    approval_required_count: int
    external_handoff_count: int
    latest_call_at: datetime | None


@dataclass(frozen=True, slots=True)
class GovernedTool:
    tool_id: str
    version: str
    risk_tier: RiskTier
    required_action: str
    approval_required: bool
    idempotent: bool
    timeout_seconds: int
    max_attempts: int
    rate_limit_per_minute: int
    compensation: str
    execution_policy: ExecutionPolicy
    input_schema: dict[str, Any]
    authority_coverage: AuthorityCoverage
    authority: AuthorityDeclaration | None
    transport: str | None
    remote_tool_name: str | None
    server_id: str | None
    server_version: str | None
    required_scopes: tuple[str, ...]
    input_schema_digest: str | None
    usage: ToolUsage


@dataclass(frozen=True, slots=True)
class RecentToolCall:
    tool_call_id: str
    agent_run_id: str | None
    tool_id: str
    tool_version: str
    risk_tier: str
    status: str
    occurred_at: datetime
    source: str | None
    source_as_of: datetime | None
    authority_evidence: AuthorityEvidence
    reason_code: str | None
    transport: str | None
    server_id: str | None
    server_version: str | None


@dataclass(frozen=True, slots=True)
class ToolGovernanceTotals:
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


@dataclass(frozen=True, slots=True)
class ToolGovernanceSnapshot:
    policy_version: str
    generated_at: datetime
    window_start: datetime
    window_end: datetime
    totals: ToolGovernanceTotals
    tools: tuple[GovernedTool, ...]
    recent_calls: tuple[RecentToolCall, ...]
    anomalies: tuple[str, ...]


@dataclass(slots=True)
class _UsageAccumulator:
    call_count: int = 0
    succeeded_count: int = 0
    failed_count: int = 0
    approval_required_count: int = 0
    external_handoff_count: int = 0
    latest_call_at: datetime | None = None

    def add(self, status: str, count: int, latest: datetime) -> None:
        self.call_count += count
        self.succeeded_count += count if status == "SUCCEEDED" else 0
        self.failed_count += count if status == "FAILED" else 0
        self.approval_required_count += count if status == "APPROVAL_REQUIRED" else 0
        self.external_handoff_count += count if status == "EXTERNAL_HANDOFF" else 0
        if self.latest_call_at is None or _utc(latest) > self.latest_call_at:
            self.latest_call_at = _utc(latest)

    def freeze(self) -> ToolUsage:
        return ToolUsage(
            call_count=self.call_count,
            succeeded_count=self.succeeded_count,
            failed_count=self.failed_count,
            approval_required_count=self.approval_required_count,
            external_handoff_count=self.external_handoff_count,
            latest_call_at=self.latest_call_at,
        )


class ToolGovernanceService:
    POLICY_VERSION = "agt005-mcp-tool-governance/v2"

    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        definitions: tuple[ToolDefinition, ...],
        transport_bindings: tuple[ToolTransportBinding, ...] = (),
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._registry = ToolRegistry(list(definitions))
        self._definitions = definitions
        self._transport_bindings = {
            item.local_tool_id: item for item in transport_bindings
        }

    def snapshot(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
        window_hours: int = 24,
        recent_limit: int = 50,
        now: datetime | None = None,
    ) -> ToolGovernanceSnapshot:
        self._authorizer.require(
            identity,
            Action.READ_TOOL_GOVERNANCE,
            ResourceContext(tenant_id=identity.tenant_id, resource_id="tool-governance"),
            request_id=request_id,
        )
        generated_at = _utc(now or datetime.now(UTC))
        window_start = generated_at - timedelta(hours=window_hours)
        with self._database.transaction(identity.tenant_context) as session:
            aggregate_rows = tuple(
                session.execute(
                    select(
                        ToolCallRecord.tool_id,
                        ToolCallRecord.tool_version,
                        ToolCallRecord.risk_tier,
                        ToolCallRecord.status,
                        func.count(ToolCallRecord.tool_call_id),
                        func.max(ToolCallRecord.created_at),
                    )
                    .where(
                        ToolCallRecord.tenant_id == identity.tenant_id,
                        ToolCallRecord.created_at >= window_start,
                        ToolCallRecord.created_at <= generated_at,
                    )
                    .group_by(
                        ToolCallRecord.tool_id,
                        ToolCallRecord.tool_version,
                        ToolCallRecord.risk_tier,
                        ToolCallRecord.status,
                    )
                )
            )
            recent_records = tuple(
                session.scalars(
                    select(ToolCallRecord)
                    .where(
                        ToolCallRecord.tenant_id == identity.tenant_id,
                        ToolCallRecord.created_at >= window_start,
                        ToolCallRecord.created_at <= generated_at,
                    )
                    .order_by(ToolCallRecord.created_at.desc(), ToolCallRecord.tool_call_id)
                    .limit(recent_limit)
                )
            )
            active_agent_runs = int(
                session.scalar(
                    select(func.count(distinct(ToolCallRecord.agent_run_id))).where(
                        ToolCallRecord.tenant_id == identity.tenant_id,
                        ToolCallRecord.created_at >= window_start,
                        ToolCallRecord.created_at <= generated_at,
                        ToolCallRecord.agent_run_id.is_not(None),
                    )
                )
                or 0
            )

        usage_by_key: dict[tuple[str, str], _UsageAccumulator] = {
            (item.tool_id, item.version): _UsageAccumulator() for item in self._definitions
        }
        unregistered_count = 0
        t2_execution_violation = 0
        t3_execution_violation = 0
        totals = _UsageAccumulator()
        for tool_id, version, risk_tier, status, count_value, latest_value in aggregate_rows:
            count = int(count_value)
            latest = _utc(latest_value)
            totals.add(status, count, latest)
            try:
                definition = self._registry.get(tool_id, version)
            except ValueError:
                unregistered_count += count
            else:
                usage_by_key[(definition.tool_id, definition.version)].add(
                    status,
                    count,
                    latest,
                )
            if risk_tier == "T2" and status == "SUCCEEDED":
                t2_execution_violation += count
            if risk_tier == "T3" and status != "EXTERNAL_HANDOFF":
                t3_execution_violation += count

        recent_calls = tuple(_recent_call(item) for item in recent_records)
        tools = tuple(
            _governed_tool(
                item,
                usage_by_key[(item.tool_id, item.version)].freeze(),
                self._transport_bindings.get(item.tool_id),
            )
            for item in self._definitions
        )
        anomalies: list[str] = []
        if unregistered_count:
            anomalies.append("unregistered_tool_audit_present")
        if t2_execution_violation:
            anomalies.append("t2_execution_without_approval_path")
        if t3_execution_violation:
            anomalies.append("t3_platform_execution_violation")
        if any(item.authority_evidence == AuthorityEvidence.MISMATCH for item in recent_calls):
            anomalies.append("authority_evidence_mismatch")
        if totals.call_count and totals.failed_count / totals.call_count >= 0.2:
            anomalies.append("tool_failure_ratio_high")
        if any(
            item.transport == "MCP_STREAMABLE_HTTP" and item.input_schema_digest is None
            for item in tools
        ):
            anomalies.append("mcp_schema_binding_missing")

        risk_counts: dict[str, int] = {}
        for item in self._definitions:
            risk_counts[item.risk_tier] = risk_counts.get(item.risk_tier, 0) + 1
        return ToolGovernanceSnapshot(
            policy_version=self.POLICY_VERSION,
            generated_at=generated_at,
            window_start=window_start,
            window_end=generated_at,
            totals=ToolGovernanceTotals(
                registered_tool_count=len(self._definitions),
                authority_declared_count=sum(item.authority is not None for item in tools),
                call_count=totals.call_count,
                succeeded_count=totals.succeeded_count,
                failed_count=totals.failed_count,
                approval_required_count=totals.approval_required_count,
                external_handoff_count=totals.external_handoff_count,
                active_agent_run_count=active_agent_runs,
                unregistered_audit_count=unregistered_count,
                mcp_tool_count=sum(
                    item.transport == "MCP_STREAMABLE_HTTP" for item in tools
                ),
                mcp_server_count=len(
                    {
                        item.server_id
                        for item in tools
                        if item.transport == "MCP_STREAMABLE_HTTP"
                        and item.server_id is not None
                    }
                ),
                risk_counts=dict(sorted(risk_counts.items())),
            ),
            tools=tools,
            recent_calls=recent_calls,
            anomalies=tuple(anomalies),
        )


def _governed_tool(
    definition: ToolDefinition,
    usage: ToolUsage,
    binding: ToolTransportBinding | None,
) -> GovernedTool:
    authority = TOOL_FACT_AUTHORITIES.get(definition.tool_id)
    authority_required = definition.risk_tier in {"T0", "T1"}
    coverage = (
        AuthorityCoverage.DECLARED
        if authority is not None
        else AuthorityCoverage.MISSING
        if authority_required
        else AuthorityCoverage.NOT_APPLICABLE
    )
    return GovernedTool(
        tool_id=definition.tool_id,
        version=definition.version,
        risk_tier=_risk_tier(definition.risk_tier),
        required_action=definition.required_action,
        approval_required=definition.approval_required,
        idempotent=definition.idempotent,
        timeout_seconds=definition.timeout_seconds,
        max_attempts=definition.max_attempts,
        rate_limit_per_minute=definition.rate_limit_per_minute,
        compensation=definition.compensation,
        execution_policy=_execution_policy(definition.risk_tier),
        input_schema=definition.input_schema,
        authority_coverage=coverage,
        authority=_authority_declaration(authority) if authority is not None else None,
        transport=binding.transport if binding is not None else None,
        remote_tool_name=binding.remote_tool_name if binding is not None else None,
        server_id=binding.server_id if binding is not None else None,
        server_version=binding.server_version if binding is not None else None,
        required_scopes=binding.required_scopes if binding is not None else (),
        input_schema_digest=binding.input_schema_digest if binding is not None else None,
        usage=usage,
    )


def _recent_call(record: ToolCallRecord) -> RecentToolCall:
    metadata = record.output_metadata if isinstance(record.output_metadata, dict) else {}
    authority = TOOL_FACT_AUTHORITIES.get(record.tool_id)
    source = metadata.get("source") if isinstance(metadata.get("source"), str) else None
    source_as_of = _optional_datetime(metadata.get("as_of"))
    reason_value = metadata.get("reason")
    reason_code = (
        reason_value
        if isinstance(reason_value, str) and _REASON_CODE.fullmatch(reason_value)
        else None
    )
    return RecentToolCall(
        tool_call_id=record.tool_call_id,
        agent_run_id=record.agent_run_id,
        tool_id=record.tool_id,
        tool_version=record.tool_version,
        risk_tier=record.risk_tier,
        status=record.status,
        occurred_at=_utc(record.created_at),
        source=source,
        source_as_of=source_as_of,
        authority_evidence=_authority_evidence(authority, metadata),
        reason_code=reason_code,
        transport=_safe_metadata_string(metadata, "transport"),
        server_id=_safe_metadata_string(metadata, "server_id"),
        server_version=_safe_metadata_string(metadata, "server_version"),
    )


def _safe_metadata_string(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if not isinstance(value, str) or not value or len(value) > 128:
        return None
    return value


def _authority_evidence(
    authority: ToolFactAuthority | None,
    metadata: dict[str, Any],
) -> AuthorityEvidence:
    if authority is None:
        return AuthorityEvidence.NOT_APPLICABLE
    source = metadata.get("source")
    declaration = metadata.get("authority")
    if source is None and declaration is None:
        return AuthorityEvidence.NOT_RECORDED
    if source == authority.enterprise_source:
        expected = authority.declaration("enterprise")
    elif source == authority.synthetic_source:
        expected = authority.declaration("synthetic")
    else:
        return AuthorityEvidence.MISMATCH
    return AuthorityEvidence.VERIFIED if declaration == expected else AuthorityEvidence.MISMATCH


def _authority_declaration(authority: ToolFactAuthority) -> AuthorityDeclaration:
    return AuthorityDeclaration(
        contract_version=AUTHORITY_CONTRACT_VERSION,
        domain=authority.domain,
        owner=authority.owner,
        fields=authority.fields,
        enterprise_source=authority.enterprise_source,
        synthetic_source=authority.synthetic_source,
    )


def _execution_policy(risk_tier: str) -> ExecutionPolicy:
    return cast(
        ExecutionPolicy,
        {
            "T0": "AUTO_READ_ONLY",
            "T1": "CONTROLLED_REVERSIBLE",
            "T2": "HUMAN_APPROVAL_REQUIRED",
            "T3": "EXTERNAL_HANDOFF_ONLY",
        }.get(risk_tier, "DENY_UNKNOWN_RISK"),
    )


def _risk_tier(value: str) -> RiskTier:
    if value not in {"T0", "T1", "T2", "T3"}:
        raise ValueError("tool_risk_tier_invalid")
    return cast(RiskTier, value)


def _optional_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _utc(value)
    if not isinstance(value, str):
        return None
    try:
        return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None
