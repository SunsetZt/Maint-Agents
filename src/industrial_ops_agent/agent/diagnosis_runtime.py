"""I/O for the single diagnosis graph, using the existing gateways and journal."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import TYPE_CHECKING, Any, cast

from jsonschema import ValidationError

from industrial_ops_agent.agent.checkpoints import AgentCheckpointStore
from industrial_ops_agent.agent.contracts import (
    DIAGNOSIS_AGENT_PROMPT_ID,
    DIAGNOSIS_GRAPH_VERSION,
    DIAGNOSIS_READONLY_TOOLS,
    DIAGNOSIS_STEP_SCHEMA,
    DIAGNOSIS_TOOL_FIELDS,
    budget_report,
    step_schema,
    validate_step,
)
from industrial_ops_agent.agent.models import DiagnosisReport
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.knowledge.models import RetrievalQuery, RetrievedEvidence
from industrial_ops_agent.knowledge.retrieval import HybridRetriever
from industrial_ops_agent.memory_governance.service import MemoryContextReader
from industrial_ops_agent.model_gateway.context_manifest import (
    ContextReference,
    ContextTruncation,
    canonical_sha256,
)
from industrial_ops_agent.model_gateway.evidence import require_successful_model_execution
from industrial_ops_agent.model_gateway.service import (
    GatewayRequest,
    ModelGateway,
    ModelGatewayError,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.prompting.registry import default_prompt_registry
from industrial_ops_agent.telemetry import current_otel_trace_id
from industrial_ops_agent.tools.contracts import EnterpriseToolError
from industrial_ops_agent.tools.gateway import ToolGateway, ToolRateLimitExceeded

if TYPE_CHECKING:
    from industrial_ops_agent.application.diagnoses import DiagnosisWorkflowInput

_RETRYABLE_MODEL_FAILURES = {
    "model_endpoint_unavailable",
    "model_endpoint_failed",
    "model_transport_unexpected_failure",
    "model_request_deadline_expired",
}
_TOOL_DESCRIPTIONS = {
    "asset.get": "Read this equipment model and lifecycle status.",
    "warranty.get": "Read this equipment warranty and coverage.",
    "parts.availability": "Read spare-parts availability for this equipment.",
    "schedule.availability": "Read technician and service-window availability.",
    "work_orders.history": "Read this equipment's recent repairs and latest resolution.",
}


class DiagnosisRuntime:
    def __init__(
        self,
        database: Database,
        command: DiagnosisWorkflowInput,
        initial: dict[str, Any],
        *,
        model_gateway: ModelGateway,
        tool_gateway: ToolGateway | None,
        model_alias: str,
        timeout_seconds: float,
        memory_reader: MemoryContextReader,
    ) -> None:
        self.database, self.command, self.initial = database, command, initial
        self.context = TenantContext(command.tenant_id, command.subject_id)
        self.gateway, self.tools = model_gateway, tool_gateway
        self.alias, self.timeout = model_alias, timeout_seconds
        self.memories = memory_reader
        self.checkpoints = AgentCheckpointStore(database)
        self.retriever = HybridRetriever(database)
        self.identity = _delegated_identity(command)
        self.binding_hash = canonical_sha256(
            {
                "diagnosis_run_id": command.diagnosis_run_id,
                "tenant_id": command.tenant_id,
                "subject_id": command.subject_id,
                "manifest": initial["manifest"],
            }
        )

    async def load(self) -> dict[str, Any]:
        manifest = self.initial["manifest"]
        if (
            manifest.get("agent_graph_version") != DIAGNOSIS_GRAPH_VERSION
            or manifest.get("prompt_bundle") != DIAGNOSIS_AGENT_PROMPT_ID
        ):
            raise ModelGatewayError("diagnosis_graph_binding_mismatch")
        state = await asyncio.to_thread(
            self.checkpoints.load_graph,
            self.context,
            self.command.agent_run_id,
            graph_version=DIAGNOSIS_GRAPH_VERSION,
        )
        if state is not None:
            if state.get("binding_hash") != self.binding_hash:
                raise ModelGatewayError("diagnosis_checkpoint_binding_mismatch")
            return state
        budget = dict(manifest["budget"])
        limits = {"max_steps": 8, "max_tool_calls": 5, "max_tokens": 6000, "timeout_seconds": 180}
        for key, ceiling in limits.items():
            budget[key] = min(ceiling, int(budget[key]))
            if budget[key] < (0 if key == "max_tool_calls" else 1):
                raise ValueError("diagnosis_budget_invalid")
        state = {
            **deepcopy(self.initial),
            "schema": "diagnosis-state/v2",
            "graph_version": DIAGNOSIS_GRAPH_VERSION,
            "binding_hash": self.binding_hash,
            "trace_id": current_otel_trace_id() or self.command.request_id,
            "deadline": (
                datetime.now(UTC) + timedelta(seconds=budget["timeout_seconds"])
            ).isoformat(),
            "budget": budget,
            "checkpoint_sequence": 0,
            "node_index": 0,
            "next_node": "prepare",
            "pending_node": None,
            "pending_request": None,
            "attempt": 1,
            "model_steps": 0,
            "tool_calls": 0,
            "tokens_used": 0,
            "unknown_token_reservations": 0,
            "evidence": [],
            "tool_facts": [],
            "tool_failures": [],
            "memories": [],
            "memory_truncations": [],
            "inference_ids": [],
            "model_execution": None,
            "decision": None,
            "report": None,
            "retrieval_query": self.initial["query_text"],
            "published": False,
        }
        return await self._save(
            state,
            state,
            "diagnosis.graph_started",
            {
                "graph_version": DIAGNOSIS_GRAPH_VERSION,
            },
        )

    async def _save(
        self,
        before: dict[str, Any],
        state: dict[str, Any],
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self.checkpoints.advance,
            self.context,
            self.command.agent_run_id,
            expected_sequence=before["checkpoint_sequence"],
            graph_version=DIAGNOSIS_GRAPH_VERSION,
            state=state,
            event_type=event_type,
            payload=payload,
        )

    def _require_current(self, state: dict[str, Any]) -> None:
        with self.database.transaction(self.context) as session:
            self.checkpoints.require_current(
                session,
                self.context,
                self.command.agent_run_id,
                state["checkpoint_sequence"],
            )

    async def enter(self, state: dict[str, Any], node: str) -> dict[str, Any]:
        await asyncio.to_thread(self._require_current, state)
        if state["next_node"] != node:
            raise RuntimeError("diagnosis_node_order_invalid")
        if state["pending_node"] == node:
            return state
        pending = {**state, "pending_node": node}
        return await self._save(
            state,
            pending,
            "diagnosis.graph_node_started",
            {
                "node": node,
                "step": state["node_index"],
                "graph_version": DIAGNOSIS_GRAPH_VERSION,
            },
        )

    async def commit(
        self,
        before: dict[str, Any],
        state: dict[str, Any],
        node: str,
    ) -> dict[str, Any]:
        # decide() may persist the request before doing I/O. Its latest sequence
        # remains the CAS fence; all other node results commit against 'before'.
        expected = {**before, "checkpoint_sequence": state["checkpoint_sequence"]}
        state = {**state, "pending_node": None, "node_index": state["node_index"] + 1}
        payload = {
            "node": node,
            "step": before["node_index"],
            "next_node": state["next_node"],
            "graph_version": DIAGNOSIS_GRAPH_VERSION,
            "model_steps": state["model_steps"],
            "tool_calls": state["tool_calls"],
            "tokens_used": state["tokens_used"],
        }
        if node == "decide":
            payload["inference_request_id"] = (
                state["inference_ids"][-1] if state["inference_ids"] else None
            )
            payload["action"] = state["decision"]["action"] if state["decision"] else None
        if node == "tool":
            payload["tool_id"] = state["decision"]["tool_id"]
            payload["tool_call_id"] = next(
                (
                    item["tool_call_id"]
                    for item in state["tool_facts"]
                    if item["tool_id"] == state["decision"]["tool_id"]
                ),
                None,
            )
        if node == "retrieve":
            payload["evidence_count"] = len(state["evidence"])
            payload["retrieval_guard"] = state.get("retrieval_guard", {})
        return await self._save(expected, state, "diagnosis.graph_node_completed", payload)

    def _remaining(self, state: dict[str, Any]) -> float:
        return (datetime.fromisoformat(state["deadline"]) - datetime.now(UTC)).total_seconds()

    def _exhausted(self, state: dict[str, Any]) -> str | None:
        if self._remaining(state) <= 0:
            return "diagnosis_time_budget_exhausted"
        if state["model_steps"] >= state["budget"]["max_steps"]:
            return "step_budget_exhausted"
        if (
            state["tokens_used"] + state["unknown_token_reservations"]
            >= state["budget"]["max_tokens"]
        ):
            return "token_budget_exhausted"
        return None

    async def prepare(self, state: dict[str, Any]) -> dict[str, Any]:
        selection = await asyncio.to_thread(
            self.memories.select_for_diagnosis,
            self.context,
            subject_id=self.command.subject_id,
            incident_id=self.command.incident_id,
        )
        state["memories"] = [item.prompt_payload() for item in selection.items]
        state["memory_truncations"] = [asdict(item) for item in selection.truncations]
        return state

    async def retrieve(self, state: dict[str, Any]) -> dict[str, Any]:
        if self._remaining(state) <= 0:
            state["report"] = budget_report("diagnosis_time_budget_exhausted")
            return state
        result = await asyncio.to_thread(
            self.retriever.search_with_trace,
            RetrievalQuery(
                tenant_id=self.command.tenant_id,
                subject_id=self.command.subject_id,
                roles=frozenset(self.command.roles),
                release_id=state["manifest"]["index_release_id"],
                device_family=state["device_family"],
                device_model=state["device_model"],
                query=state["retrieval_query"],
                as_of=datetime.fromisoformat(state["manifest"]["created_at"]),
            ),
        )
        merged = {item.citation_id: asdict(item) for item in result.evidence}
        for item in state["evidence"]:
            merged.setdefault(item["citation_id"], item)
        state["evidence"] = list(merged.values())[:12]
        state["retrieval_guard"] = asdict(result.guard)
        return state

    def _available_tools(self, state: dict[str, Any]) -> list[str]:
        if (
            self.tools is None
            or self.identity is None
            or state["tool_calls"] >= state["budget"]["max_tool_calls"]
        ):
            return []
        used = {item["tool_id"] for item in [*state["tool_facts"], *state["tool_failures"]]}
        versions = state["manifest"]["tool_versions"]
        return [
            name for name in DIAGNOSIS_READONLY_TOOLS if versions.get(name) and name not in used
        ]

    async def tool(self, state: dict[str, Any]) -> dict[str, Any]:
        tool_id = state["decision"]["tool_id"]
        if (
            tool_id not in self._available_tools(state)
            or self.tools is None
            or self.identity is None
        ):
            raise ModelGatewayError("diagnosis_tool_not_authorized")
        if self._remaining(state) <= 0:
            state["report"] = budget_report("diagnosis_time_budget_exhausted")
            return state
        state["tool_calls"] += 1
        try:
            result = await asyncio.to_thread(
                self.tools.invoke,
                self.identity,
                tool_id=tool_id,
                version=state["manifest"]["tool_versions"][tool_id],
                parameters={"asset_id": state["asset_id"]},
                request_id=state["trace_id"],
                agent_run_id=self.command.agent_run_id,
            )
        except AuthorizationDenied:
            raise
        except (EnterpriseToolError, ToolRateLimitExceeded) as exc:
            state["tool_failures"].append({"tool_id": tool_id, "reason": _tool_failure_reason(exc)})
            return state
        if result.status != "SUCCEEDED" or result.data is None:
            state["tool_failures"].append({"tool_id": tool_id, "reason": "tool_result_not_usable"})
            return state
        state["tool_facts"].append(
            {
                "tool_id": tool_id,
                "tool_call_id": result.tool_call_id,
                "source": str(result.data["source"])[:128],
                "source_record_id": str(result.data["source_record_id"])[:255],
                "as_of": str(result.data["as_of"])[:64],
                "authority": _bounded_tool_authority(result.data),
                "values": _bounded_tool_values(tool_id, result.data),
            }
        )
        return state

    async def decide(self, state: dict[str, Any]) -> dict[str, Any]:
        if state["pending_request"] is None:
            reason = self._exhausted(state)
            if reason:
                state["report"] = budget_report(reason)
                return state
            request = self._build_request(state)
            if request is None:
                state["report"] = budget_report("token_budget_exhausted")
                return state
            pending = deepcopy(state)
            pending["pending_request"] = _serialize_request(request)
            pending["model_steps"] += 1
            state = await self._save(
                state,
                pending,
                "diagnosis.model_started",
                {
                    "inference_request_id": request.inference_request_id,
                    "node": "decide",
                    "step": state["node_index"],
                    "model_steps": pending["model_steps"],
                    "attempt": state["attempt"],
                    "graph_version": DIAGNOSIS_GRAPH_VERSION,
                },
            )
        request = _deserialize_request(state["pending_request"])
        status, response, reason = await asyncio.to_thread(
            self.gateway.inspect_request,
            self.context,
            request,
        )
        while status == "PENDING":
            # The previous owner may still be completing this exact request.
            # Poll only its audit, without starting another inference.
            await asyncio.to_thread(self._require_current, state)
            remaining = (request.deadline - datetime.now(UTC)).total_seconds()
            if remaining <= 0:
                await asyncio.to_thread(self.gateway.expire_pending_request, self.context, request)
                status, response, reason = await asyncio.to_thread(
                    self.gateway.inspect_request,
                    self.context,
                    request,
                )
                if status != "SUCCEEDED":
                    raise ModelGatewayError("diagnosis_inference_outcome_unknown")
                break
            await asyncio.sleep(min(0.5, remaining))
            status, response, reason = await asyncio.to_thread(
                self.gateway.inspect_request,
                self.context,
                request,
            )
        if status in {"FAILED", "REJECTED"}:
            if reason not in _RETRYABLE_MODEL_FAILURES:
                raise ModelGatewayError(reason or "diagnosis_model_attempt_failed")
            return await self._retry_later(state, request, reason)
        if response is None and request.deadline <= datetime.now(UTC):
            return await self._retry_later(state, request, "model_request_deadline_expired")
        if response is None:
            try:
                response = await self.gateway.complete(self.context, request)
            except ModelGatewayError as exc:
                if exc.reason == "inference_request_already_consumed":
                    # Admission lost a race to the same fixed request. Re-enter the
                    # audit-replay/pending path, not another model invocation.
                    return await self.decide(state)
                if exc.reason in _RETRYABLE_MODEL_FAILURES:
                    return await self._retry_later(state, request, exc.reason)
                raise
        execution = await asyncio.to_thread(
            require_successful_model_execution,
            self.database,
            self.context,
            inference_request_id=response.inference_request_id,
            component="diagnosis",
            expected_alias=self.alias,
            expected_release_id=response.resolved_release_id,
            expected_manifest_hash=response.manifest_hash,
            expected_subject_id=self.command.subject_id,
            expected_trace_id=state["trace_id"],
            expected_agent_run_id=self.command.agent_run_id,
            expected_diagnosis_run_id=self.command.diagnosis_run_id,
        )
        allowed_citations = [item.reference_id for item in request.context_evidence]
        tools = [item for item in request.response_schema["properties"]["tool_id"]["enum"] if item]
        try:
            validate_step(response.content, allowed_citations, tools)
        except (ValidationError, ValueError, KeyError, TypeError) as exc:
            raise ModelGatewayError("diagnosis_step_contract_invalid") from exc
        state["decision"] = response.content
        state["tokens_used"] += response.prompt_tokens + response.completion_tokens
        state["inference_ids"].append(response.inference_request_id)
        state["model_execution"] = execution.as_dict()
        state["pending_request"] = None
        state["attempt"] = 1
        if state["tokens_used"] > state["budget"]["max_tokens"]:
            state["report"] = budget_report("token_budget_exhausted")
        return state

    async def _retry_later(
        self,
        state: dict[str, Any],
        request: GatewayRequest,
        reason: str,
    ) -> dict[str, Any]:
        updated = deepcopy(state)
        # Unknown failed-call usage cannot become a free retry.
        if reason != "model_request_deadline_expired":
            updated["unknown_token_reservations"] += request.max_output_tokens + _input_upper_bound(
                request
            )
        updated["pending_request"] = None
        updated["attempt"] += 1
        updated = await self._save(
            state,
            updated,
            "diagnosis.model_attempt_failed",
            {
                "inference_request_id": request.inference_request_id,
                "reason": reason,
                "node": "decide",
                "step": state["node_index"],
            },
        )
        exhausted = self._exhausted(updated)
        if exhausted:
            updated["report"] = budget_report(exhausted)
            return updated
        raise ModelGatewayError("diagnosis_model_retry_required")

    def _build_request(self, state: dict[str, Any]) -> GatewayRequest | None:
        return build_step_request(
            self.command,
            state,
            model_alias=self.alias,
            timeout_seconds=self.timeout,
            tools=self._available_tools(state),
        )

    async def finish(self, state: dict[str, Any]) -> dict[str, Any]:
        content = state["decision"]["report"]
        evidence = [RetrievedEvidence(**item) for item in state["evidence"]]
        state["report"] = _report_from_model(content, evidence).as_dict()
        return state


def build_step_request(
    command: DiagnosisWorkflowInput,
    state: dict[str, Any],
    *,
    model_alias: str,
    timeout_seconds: float,
    tools: list[str],
) -> GatewayRequest | None:
    bundle = default_prompt_registry().get(state["manifest"]["prompt_bundle"])
    if (
        bundle.prompt_bundle_id != DIAGNOSIS_AGENT_PROMPT_ID
        or bundle.output_schema_name != DIAGNOSIS_STEP_SCHEMA
    ):
        raise ModelGatewayError("diagnosis_prompt_binding_mismatch")
    expected_hash = state["manifest"].get("prompt_bundle_hash")
    if not expected_hash or expected_hash != bundle.content_hash:
        raise ModelGatewayError("diagnosis_prompt_hash_mismatch")
    remaining = (
        state["budget"]["max_tokens"] - state["tokens_used"] - state["unknown_token_reservations"]
    )
    selected = state["evidence"][:4]
    memories = state["memories"]
    content_limit = 600
    while True:
        evidence = [
            {key: item[key] for key in ("citation_id", "title", "content_checksum")}
            | {"content": item["content"][:content_limit]}
            for item in selected
        ]
        payload = {
            "incident": state["query_text"][:1800],
            "authorized_evidence": evidence,
            "authorized_enterprise_facts": state["tool_facts"],
            "enterprise_tool_failures": state["tool_failures"],
            "untrusted_human_confirmed_memories": memories,
            "available_tools": {name: _TOOL_DESCRIPTIONS[name] for name in tools},
            "remaining_budget": {
                "model_calls": state["budget"]["max_steps"] - state["model_steps"],
                "tool_calls": state["budget"]["max_tool_calls"] - state["tool_calls"],
            },
        }
        identity = (
            f"{command.diagnosis_run_id}:{DIAGNOSIS_GRAPH_VERSION}:decide:"
            f"{state['node_index']}:{state['attempt']}"
        )
        request = GatewayRequest(
            inference_request_id="inference-" + sha256(identity.encode()).hexdigest(),
            model_alias=model_alias,
            required_release_id=state["manifest"]["model_release"],
            subject_id=command.subject_id,
            trace_id=state["trace_id"],
            request_class="DIAGNOSIS",
            data_classification="CONFIDENTIAL",
            messages=(
                {"role": "system", "content": bundle.render_system()},
                {
                    "role": "user",
                    "content": json.dumps(
                        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ),
                },
            ),
            response_schema_name=DIAGNOSIS_STEP_SCHEMA,
            response_schema=step_schema([item["citation_id"] for item in evidence], tools),
            deadline=min(
                datetime.fromisoformat(state["deadline"]),
                datetime.now(UTC) + timedelta(seconds=timeout_seconds),
            ),
            max_output_tokens=1,
            temperature=0.0,
            agent_run_id=command.agent_run_id,
            diagnosis_run_id=command.diagnosis_run_id,
            prompt_bundle_id=bundle.prompt_bundle_id,
            prompt_bundle_hash=bundle.content_hash,
            index_release_id=state["manifest"]["index_release_id"],
            context_evidence=tuple(
                ContextReference(item["citation_id"], canonical_sha256(item)) for item in evidence
            ),
            context_tool_results=tuple(
                ContextReference(item["tool_call_id"], canonical_sha256(item))
                for item in state["tool_facts"]
            ),
            context_memories=tuple(
                ContextReference(item["memory_id"], canonical_sha256(item)) for item in memories
            ),
            context_token_budget=remaining,
            context_truncated_items=_truncations(state, selected, content_limit, memories),
        )
        output_budget = remaining - _input_upper_bound(request)
        if output_budget >= 128:
            from dataclasses import replace

            return replace(request, max_output_tokens=min(1024, output_budget))
        if len(selected) > 1:
            selected = selected[:-1]
        elif content_limit > 150:
            content_limit = 150
        elif memories:
            memories = []
        else:
            return None


def _input_upper_bound(request: GatewayRequest) -> int:
    # UTF-8 bytes upper-bound ordinary byte-BPE text tokens. Include schema and
    # protocol overhead, rather than assuming Chinese text is chars/4 tokens.
    return (
        512
        + sum(len(str(m["content"]).encode()) for m in request.messages)
        + len(
            json.dumps(request.response_schema, ensure_ascii=False, separators=(",", ":")).encode()
        )
    )


def _serialize_request(request: GatewayRequest) -> dict[str, Any]:
    value = asdict(request)
    value["deadline"] = request.deadline.isoformat()
    return value


def _deserialize_request(value: dict[str, Any]) -> GatewayRequest:
    values = dict(value)
    values["deadline"] = datetime.fromisoformat(values["deadline"])
    values["messages"] = tuple(values["messages"])
    for key in ("context_evidence", "context_memories", "context_tool_results"):
        values[key] = tuple(ContextReference(**item) for item in values[key])
    values["context_truncated_items"] = tuple(
        ContextTruncation(**item) for item in values["context_truncated_items"]
    )
    return GatewayRequest(**values)


def _truncations(
    state: dict[str, Any],
    selected: list[dict[str, Any]],
    limit: int,
    memories: list[dict[str, Any]],
) -> tuple[ContextTruncation, ...]:
    result = [ContextTruncation(**item) for item in state["memory_truncations"]]
    if len(state["query_text"]) > 1800:
        result.append(
            ContextTruncation(
                "QUERY",
                state["diagnosis_run_id"],
                "CHARACTER_LIMIT",
                len(state["query_text"]),
                1800,
            )
        )
    selected_ids = {item["citation_id"] for item in selected}
    for item in state["evidence"]:
        length = len(item["content"])
        kept = min(limit, length) if item["citation_id"] in selected_ids else 0
        if kept < length:
            result.append(
                ContextTruncation("EVIDENCE", item["citation_id"], "TOKEN_BUDGET", length, kept)
            )
    if not memories:
        result.extend(
            ContextTruncation("MEMORY", item["memory_id"], "TOKEN_BUDGET", len(item["content"]), 0)
            for item in state["memories"]
            if item["content"]
        )
    return tuple(result)


def _delegated_identity(command: DiagnosisWorkflowInput) -> IdentityContext | None:
    values = (command.oidc_subject, command.issued_at, command.expires_at)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise RuntimeError("diagnosis delegated identity binding is incomplete")
    try:
        return IdentityContext(
            subject_id=command.subject_id,
            oidc_subject=cast(str, command.oidc_subject),
            tenant_id=command.tenant_id,
            roles=frozenset(Role(value) for value in command.roles),
            asset_ids=frozenset(command.asset_ids),
            site_ids=frozenset(command.site_ids),
            issued_at=datetime.fromisoformat(cast(str, command.issued_at)),
            expires_at=datetime.fromisoformat(cast(str, command.expires_at)),
        )
    except ValueError as exc:
        raise RuntimeError("diagnosis delegated identity binding is invalid") from exc


def _tool_failure_reason(
    exc: AuthorizationDenied | EnterpriseToolError | ToolRateLimitExceeded,
) -> str:
    if isinstance(exc, EnterpriseToolError):
        return exc.reason
    if isinstance(exc, ToolRateLimitExceeded):
        return "tool_rate_limit_exceeded"
    return "tool_authorization_denied"


def _bounded_tool_values(tool_id: str, data: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field in DIAGNOSIS_TOOL_FIELDS[tool_id]:
        value = data.get(field)
        if value is None or isinstance(value, bool | int | float):
            values[field] = value
        elif isinstance(value, str):
            values[field] = value[:500]
        else:
            raise RuntimeError("enterprise tool returned an unsupported fact value")
    return values


def _bounded_tool_authority(data: dict[str, Any]) -> dict[str, Any]:
    raw = data.get("authority")
    if not isinstance(raw, dict) or not isinstance(raw.get("field_sources"), dict):
        raise RuntimeError("enterprise tool authority declaration is invalid")
    field_sources = raw["field_sources"]
    return {
        "contract_version": str(raw.get("contract_version", ""))[:128],
        "domain": str(raw.get("domain", ""))[:128],
        "owner": str(raw.get("owner", ""))[:128],
        "field_sources": {
            str(field)[:128]: str(source)[:128] for field, source in field_sources.items()
        },
    }


def _report_from_model(
    content: dict[str, Any], evidence: list[RetrievedEvidence]
) -> DiagnosisReport:
    citations = {item.citation_id: item.citation_payload() for item in evidence}
    selected = tuple(citations[citation_id] for citation_id in content["citation_ids"])
    status = str(content["status"])
    return DiagnosisReport(
        status=status,
        conclusion=content["conclusion"],
        citations=selected,
        contradictions=tuple(str(item) for item in content["contradictions"]),
        missing_information=tuple(str(item) for item in content["missing_information"]),
        next_checks=tuple(str(item) for item in content["next_checks"]),
        confidence=float(content["confidence"]),
        stop_reason=(
            "model_evidence_sufficient"
            if status == "COMPLETED"
            else "model_requires_more_information"
        ),
    )
