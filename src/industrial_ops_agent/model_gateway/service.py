"""Production route resolution, quota admission and minimized inference audit."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from time import perf_counter
from typing import Any, Final, Protocol

import structlog
from jsonschema import SchemaError, ValidationError, validate
from opentelemetry.trace import SpanKind, Status, StatusCode
from sqlalchemy import func, select

from industrial_ops_agent.guardrails import GuardrailDecision, PromptInjectionGuard
from industrial_ops_agent.model_gateway.context_manifest import (
    ContextManifest,
    ContextReference,
    ContextTruncation,
    build_context_manifest,
    context_manifest_is_valid,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    ModelAliasRecord,
    ModelDeploymentRecord,
    ModelGatewayQuotaRecord,
    ModelInferenceRecord,
    ModelReleaseRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.runtime_metrics import InferenceMetrics
from industrial_ops_agent.security_audit import SecurityAuditor
from industrial_ops_agent.supply_chain.service import require_media_deployment_supply_chain
from industrial_ops_agent.telemetry import safe_tenant_hash, traced_operation

MAX_MULTIMODAL_IMAGES = 8
MAX_MULTIMODAL_IMAGE_BYTES = 10 * 1024 * 1024
MAX_MULTIMODAL_TOTAL_BYTES = 12 * 1024 * 1024
MAX_MULTIMODAL_TEXT_CHARACTERS = 16_000
MODEL_ALIAS_BY_ENVIRONMENT: Final[dict[str, str]] = {
    "STAGING": "industrial-diagnosis-staging",
    "PRODUCTION": "industrial-diagnosis",
}
REQUIRED_MULTIMODAL_COMPONENTS: Final[frozenset[str]] = frozenset({"ocr", "vlm"})
REQUIRED_GATEWAY_REQUEST_CLASSES: Final[frozenset[str]] = frozenset({"DIAGNOSIS", "VLM"})


def normalize_model_environment(value: object) -> str:
    raw = getattr(value, "value", value)
    normalized = str(raw).strip().upper()
    if normalized not in MODEL_ALIAS_BY_ENVIRONMENT:
        raise ValueError("required model environment must be STAGING or PRODUCTION")
    return normalized


class ModelGatewayError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ModelTransportError(ModelGatewayError):
    pass


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    alias: str
    release_id: str
    manifest_hash: str
    endpoint_url: str
    runtime_profile: str
    multimodal_model_ids: dict[str, str]
    served_model_id: str | None = None
    served_model_digest: str | None = None
    deployment_id: str | None = None
    target_environment: str = "PRODUCTION"
    diagnosis_binding: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class GatewayRequest:
    inference_request_id: str
    model_alias: str
    required_release_id: str
    subject_id: str
    trace_id: str
    request_class: str
    data_classification: str
    messages: tuple[dict[str, Any], ...]
    response_schema_name: str
    response_schema: dict[str, Any]
    deadline: datetime
    max_output_tokens: int
    temperature: float = 0.0
    agent_run_id: str | None = None
    diagnosis_run_id: str | None = None
    prompt_bundle_id: str | None = None
    index_release_id: str | None = None
    prompt_bundle_hash: str | None = None
    context_evidence: tuple[ContextReference, ...] = ()
    context_memories: tuple[ContextReference, ...] = ()
    context_tool_results: tuple[ContextReference, ...] = ()
    context_token_budget: int | None = None
    context_truncated_items: tuple[ContextTruncation, ...] = ()


@dataclass(frozen=True, slots=True)
class GatewayResponse:
    inference_request_id: str
    resolved_release_id: str
    manifest_hash: str
    content: dict[str, Any]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    latency_breakdown: dict[str, float]
    context_hash: str
    replayed: bool = False


@dataclass(frozen=True, slots=True)
class TransportResult:
    content: dict[str, Any]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    latency_breakdown: dict[str, float]


class ModelTransport(Protocol):
    async def complete(
        self,
        model: ResolvedModel,
        request: GatewayRequest,
        *,
        timeout_seconds: float,
    ) -> TransportResult: ...


class ModelResolver(Protocol):
    def resolve(self, context: TenantContext, alias_name: str) -> ResolvedModel: ...


class EnvironmentAwareModelResolver:
    """Resolve only a fully governed route in one configured release environment."""

    def __init__(self, database: Database, required_environment: object) -> None:
        self._database = database
        self._required_environment = normalize_model_environment(required_environment)
        self._required_alias = MODEL_ALIAS_BY_ENVIRONMENT[self._required_environment]

    @property
    def required_environment(self) -> str:
        return self._required_environment

    @property
    def required_alias(self) -> str:
        return self._required_alias

    def resolve(self, context: TenantContext, alias_name: str) -> ResolvedModel:
        if alias_name != self._required_alias:
            raise ModelGatewayError("model_alias_environment_mismatch")
        with self._database.transaction(context) as session:
            alias = session.scalar(
                select(ModelAliasRecord).where(
                    ModelAliasRecord.tenant_id == context.tenant_id,
                    ModelAliasRecord.alias == alias_name,
                    ModelAliasRecord.status == "ACTIVE",
                )
            )
            if alias is None:
                raise ModelGatewayError("model_alias_unavailable")
            release = session.scalar(
                select(ModelReleaseRecord).where(
                    ModelReleaseRecord.tenant_id == context.tenant_id,
                    ModelReleaseRecord.release_id == alias.active_release_id,
                )
            )
            if release is None or release.status != "PRODUCTION":
                raise ModelGatewayError("model_release_unavailable")
            if release.target_environment != self._required_environment:
                raise ModelGatewayError("model_release_environment_mismatch")
            if (
                _manifest_digest(release.manifest_json) != release.manifest_hash
                or alias.manifest_hash != release.manifest_hash
            ):
                raise ModelGatewayError("model_manifest_inconsistent")
            deployment = session.scalar(
                select(ModelDeploymentRecord).where(
                    ModelDeploymentRecord.tenant_id == context.tenant_id,
                    ModelDeploymentRecord.deployment_id == alias.deployment_id,
                )
            )
            if deployment is None or deployment.status != "READY":
                raise ModelGatewayError("model_deployment_unavailable")
            deployment_release = (
                session.scalar(
                    select(ModelReleaseRecord).where(
                        ModelReleaseRecord.tenant_id == context.tenant_id,
                        ModelReleaseRecord.release_id == deployment.release_id,
                    )
                )
                if deployment is not None
                else None
            )
            full_route = bool(
                deployment.current_stage == "PRODUCTION"
                and deployment.desired_stage == "PRODUCTION"
                and deployment.release_id == alias.active_release_id
                and deployment.observed_traffic_percent == 100.0
                and deployment.desired_traffic_percent == 100.0
            )
            rollback_route = bool(
                deployment.current_stage == "ROLLED_BACK"
                and deployment.desired_stage == "ROLLED_BACK"
                and deployment.release_id != alias.active_release_id
                and deployment_release is not None
                and deployment_release.rollback_release_id == alias.active_release_id
                and deployment.observed_traffic_percent == 0.0
                and deployment.desired_traffic_percent == 0.0
            )
            if (
                deployment_release is None
                or deployment_release.target_environment != self._required_environment
                or _manifest_digest(deployment_release.manifest_json)
                != deployment_release.manifest_hash
            ):
                raise ModelGatewayError("model_manifest_inconsistent")
            deployment_manifest_hash = deployment.desired_spec_json.get(
                "manifest_hash",
                deployment.desired_spec_json.get("release_manifest_hash"),
            )
            if deployment_manifest_hash != deployment_release.manifest_hash:
                raise ModelGatewayError("model_manifest_inconsistent")
            if (
                not (full_route or rollback_route)
                or deployment.applied_spec_hash != deployment.desired_spec_hash
                or _manifest_digest(deployment.desired_spec_json) != deployment.desired_spec_hash
            ):
                raise ModelGatewayError("model_deployment_route_inconsistent")
            if (
                not alias.endpoint_url.startswith(("http://", "https://"))
                or alias.endpoint_url != deployment.endpoint_url
            ):
                raise ModelGatewayError("model_endpoint_unavailable")

            runtime = release.manifest_json.get("runtime")
            multimodal_model_ids = release.manifest_json.get("multimodal_model_ids")
            if not isinstance(runtime, dict) or not isinstance(multimodal_model_ids, dict):
                raise ModelGatewayError("model_component_binding_incomplete")
            runtime_profile = runtime.get("profile_id")
            if (
                not isinstance(runtime_profile, str)
                or not runtime_profile
                or alias.runtime_profile != runtime_profile
                or any(
                    not isinstance(multimodal_model_ids.get(component), str)
                    or not multimodal_model_ids[component]
                    for component in REQUIRED_MULTIMODAL_COMPONENTS
                )
            ):
                raise ModelGatewayError("model_component_binding_incomplete")

            try:
                # Import after package initialization: releases and deployment expose eager facades.
                from industrial_ops_agent.releases.staging_smoke import (
                    require_publishable_staging_admission,
                )

                require_publishable_staging_admission(
                    session, release, deployment_provider=deployment.provider
                )
                require_media_deployment_supply_chain(
                    session, release, deployment, allow_exact_recovery=True
                )
            except ValueError as exc:
                raise ModelGatewayError(str(exc)) from exc

            quota = session.scalar(
                select(ModelGatewayQuotaRecord).where(
                    ModelGatewayQuotaRecord.tenant_id == context.tenant_id,
                    ModelGatewayQuotaRecord.model_alias == alias.alias,
                )
            )
            if (
                quota is None
                or not quota.enabled
                or quota.requests_per_minute <= 0
                or quota.tokens_per_day <= 0
                or quota.max_output_tokens <= 0
                or not REQUIRED_GATEWAY_REQUEST_CLASSES.issubset(set(quota.allowed_request_classes))
            ):
                raise ModelGatewayError("model_quota_unavailable")
            return ResolvedModel(
                alias=alias.alias,
                release_id=release.release_id,
                manifest_hash=release.manifest_hash,
                endpoint_url=alias.endpoint_url,
                runtime_profile=runtime_profile,
                multimodal_model_ids=dict(multimodal_model_ids),
                deployment_id=deployment.deployment_id,
                target_environment=self._required_environment,
                diagnosis_binding={
                    key: release.manifest_json.get(key)
                    for key in (
                        "agent_graph_id",
                        "prompt_bundle_id",
                        "prompt_bundle",
                        "tool_registry",
                        "retrieval",
                    )
                },
            )


class ProductionModelResolver(EnvironmentAwareModelResolver):
    """Compatibility entrypoint fixed to the governed production route."""

    def __init__(self, database: Database) -> None:
        super().__init__(database, "PRODUCTION")


class ModelGateway:
    def __init__(
        self,
        database: Database,
        resolver: ModelResolver,
        transport: ModelTransport,
        *,
        guardrail: PromptInjectionGuard | None = None,
        security_auditor: SecurityAuditor | None = None,
        metrics: InferenceMetrics | None = None,
    ) -> None:
        self._database = database
        self._resolver = resolver
        self._transport = transport
        self._guardrail = guardrail or PromptInjectionGuard()
        self._security_auditor = security_auditor
        self._metrics = metrics

    async def complete(self, context: TenantContext, request: GatewayRequest) -> GatewayResponse:
        attributes: dict[str, str | int | float | bool] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": request.model_alias,
            "gen_ai.request.max_tokens": request.max_output_tokens,
            "gen_ai.request.temperature": request.temperature,
            "langfuse.observation.type": "generation",
            "langfuse.trace.name": "industrial-diagnosis",
            "ioap.inference_request_id": request.inference_request_id,
            "ioap.tenant_hash": safe_tenant_hash(context.tenant_id),
            "ioap.request_class": request.request_class,
            "ioap.data_classification": request.data_classification,
        }
        for key, value in (
            ("ioap.agent_run_id", request.agent_run_id),
            ("ioap.diagnosis_run_id", request.diagnosis_run_id),
            ("ioap.prompt_bundle_id", request.prompt_bundle_id),
            ("ioap.index_release_id", request.index_release_id),
        ):
            if value is not None:
                attributes[key] = value
        if request.agent_run_id is not None:
            attributes["langfuse.session.id"] = request.agent_run_id
        started = perf_counter()
        with traced_operation(
            "gen_ai.inference",
            kind=SpanKind.CLIENT,
            attributes=attributes,
        ) as span:
            try:
                response = await self._complete(context, request)
            except ModelGatewayError as exc:
                elapsed = perf_counter() - started
                span.set_attribute("error.type", exc.reason)
                span.set_status(Status(StatusCode.ERROR, exc.reason))
                if self._metrics is not None:
                    self._metrics.observe(
                        model_alias=request.model_alias,
                        status="failure",
                        seconds=elapsed,
                    )
                structlog.get_logger().warning(
                    "model_inference_failed",
                    model_alias=request.model_alias,
                    inference_request_id=request.inference_request_id,
                    tenant_hash=safe_tenant_hash(context.tenant_id),
                    reason=exc.reason,
                    duration_ms=round(elapsed * 1000, 3),
                )
                raise
            except Exception:
                elapsed = perf_counter() - started
                reason = "model_gateway_unexpected_failure"
                span.set_attribute("error.type", reason)
                span.set_status(Status(StatusCode.ERROR, reason))
                if self._metrics is not None:
                    self._metrics.observe(
                        model_alias=request.model_alias,
                        status="failure",
                        seconds=elapsed,
                    )
                structlog.get_logger().warning(
                    "model_inference_failed",
                    model_alias=request.model_alias,
                    inference_request_id=request.inference_request_id,
                    tenant_hash=safe_tenant_hash(context.tenant_id),
                    reason=reason,
                    duration_ms=round(elapsed * 1000, 3),
                )
                raise
            elapsed = perf_counter() - started
            span.set_attribute("gen_ai.response.model", response.resolved_release_id)
            span.set_attribute("gen_ai.response.finish_reasons", response.finish_reason)
            span.set_attribute("gen_ai.usage.input_tokens", response.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", response.completion_tokens)
            span.set_attribute("ioap.response.sha256", _digest(response.content))
            span.set_attribute("ioap.context.sha256", response.context_hash)
            span.set_attribute("ioap.replayed", response.replayed)
            if self._metrics is not None:
                self._metrics.observe(
                    model_alias=request.model_alias,
                    status="success",
                    seconds=elapsed,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                )
            structlog.get_logger().info(
                "model_inference_completed",
                model_alias=request.model_alias,
                resolved_release_id=response.resolved_release_id,
                inference_request_id=request.inference_request_id,
                tenant_hash=safe_tenant_hash(context.tenant_id),
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                finish_reason=response.finish_reason,
                replayed=response.replayed,
                duration_ms=round(elapsed * 1000, 3),
            )
            return response

    def inspect_request(
        self,
        context: TenantContext,
        request: GatewayRequest,
    ) -> tuple[str | None, GatewayResponse | None, str | None]:
        """Read an exact previous attempt without reserving quota or invoking a model."""
        model = self._resolver.resolve(context, request.model_alias)
        if model.release_id != request.required_release_id:
            raise ModelGatewayError("diagnosis_model_release_changed")
        manifest = _context_manifest(request, model)
        prompt_hash = _digest(
            {"messages": request.messages, "response_schema": request.response_schema}
        )
        with self._database.transaction(context) as session:
            record = session.get(ModelInferenceRecord, request.inference_request_id)
            if record is None:
                return None, None, None
            _require_replay_binding(record, context, request, model, prompt_hash, manifest)
            response = (
                _response_from_record(record, replayed=True)
                if record.status == "SUCCEEDED"
                else None
            )
            return record.status, response, record.failure_reason

    def expire_pending_request(self, context: TenantContext, request: GatewayRequest) -> None:
        """Conclude an uncertain attempt after its fixed deadline; never retry it here."""
        if request.deadline > datetime.now(UTC):
            raise ModelGatewayError("inference_deadline_not_reached")
        status, _, _ = self.inspect_request(context, request)
        if status == "PENDING":
            self._finish_failure(
                context, request.inference_request_id, "diagnosis_inference_outcome_unknown"
            )

    async def _complete(
        self,
        context: TenantContext,
        request: GatewayRequest,
    ) -> GatewayResponse:
        now = datetime.now(UTC)
        _validate_request(request, now)
        model = await asyncio.to_thread(self._resolver.resolve, context, request.model_alias)
        if model.release_id != request.required_release_id:
            raise ModelGatewayError("diagnosis_model_release_changed")
        context_manifest = _context_manifest(request, model)
        prompt_hash = _digest(
            {
                "messages": request.messages,
                "response_schema": request.response_schema,
            }
        )
        replay = await asyncio.to_thread(
            self._admit, context, request, model, prompt_hash, context_manifest, now
        )
        if replay is not None:
            return replay
        input_decision = self._guardrail.inspect_messages(request.messages)
        if input_decision.decision == "BLOCKED":
            await asyncio.to_thread(
                self._reject_guardrail, context, request, model, input_decision, phase="input"
            )
            raise ModelGatewayError("model_input_guardrail_blocked")
        remaining = (request.deadline - datetime.now(UTC)).total_seconds()
        try:
            if remaining <= 0:
                raise ModelGatewayError("model_request_deadline_expired")
            result = await self._transport.complete(model, request, timeout_seconds=remaining)
            _validate_result(result, request)
            output_decision = self._guardrail.inspect_output(result.content)
            if output_decision.decision == "BLOCKED":
                await asyncio.to_thread(
                    self._reject_guardrail, context, request, model, output_decision, phase="output"
                )
                raise ModelGatewayError("model_output_guardrail_blocked")
        except ModelGatewayError as exc:
            await asyncio.to_thread(
                self._finish_failure, context, request.inference_request_id, exc.reason
            )
            raise
        except Exception as exc:
            await asyncio.to_thread(
                self._finish_failure,
                context,
                request.inference_request_id,
                "model_transport_unexpected_failure",
            )
            raise ModelGatewayError("model_transport_unexpected_failure") from exc
        return await asyncio.to_thread(self._finish_success, context, request, model, result)

    def _admit(
        self,
        context: TenantContext,
        request: GatewayRequest,
        model: ResolvedModel,
        prompt_hash: str,
        context_manifest: ContextManifest,
        now: datetime,
    ) -> GatewayResponse | None:
        with self._database.transaction(context) as session:
            existing = session.get(ModelInferenceRecord, request.inference_request_id)
            if existing is not None:
                _require_replay_binding(
                    existing, context, request, model, prompt_hash, context_manifest
                )
                if existing.status == "SUCCEEDED" and existing.response_json is not None:
                    return _response_from_record(existing, replayed=True)
                raise ModelGatewayError("inference_request_already_consumed")

            quota = session.scalar(
                select(ModelGatewayQuotaRecord)
                .where(
                    ModelGatewayQuotaRecord.tenant_id == context.tenant_id,
                    ModelGatewayQuotaRecord.model_alias == request.model_alias,
                )
                .with_for_update()
            )
            if quota is None or not quota.enabled:
                raise ModelGatewayError("model_quota_unavailable")
            # A competing attempt can insert this ID while we wait for the quota lock.
            existing = session.get(ModelInferenceRecord, request.inference_request_id)
            if existing is not None:
                _require_replay_binding(
                    existing, context, request, model, prompt_hash, context_manifest
                )
                if existing.status == "SUCCEEDED" and existing.response_json is not None:
                    return _response_from_record(existing, replayed=True)
                raise ModelGatewayError("inference_request_already_consumed")
            if request.request_class not in quota.allowed_request_classes:
                raise ModelGatewayError("model_request_class_forbidden")
            if request.max_output_tokens > quota.max_output_tokens:
                raise ModelGatewayError("model_output_limit_exceeded")
            minute_count = int(
                session.scalar(
                    select(func.count(ModelInferenceRecord.inference_request_id)).where(
                        ModelInferenceRecord.tenant_id == context.tenant_id,
                        ModelInferenceRecord.model_alias == request.model_alias,
                        ModelInferenceRecord.created_at >= now - timedelta(minutes=1),
                    )
                )
                or 0
            )
            if minute_count >= quota.requests_per_minute:
                raise ModelGatewayError("model_request_rate_exceeded")
            day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            token_usage = int(
                session.scalar(
                    select(
                        func.coalesce(
                            func.sum(
                                ModelInferenceRecord.usage_prompt_tokens
                                + ModelInferenceRecord.usage_completion_tokens
                            ),
                            0,
                        )
                    ).where(
                        ModelInferenceRecord.tenant_id == context.tenant_id,
                        ModelInferenceRecord.model_alias == request.model_alias,
                        ModelInferenceRecord.created_at >= day_start,
                    )
                )
                or 0
            )
            pending_reservation = int(
                session.scalar(
                    select(
                        func.coalesce(func.sum(ModelInferenceRecord.max_output_tokens), 0)
                    ).where(
                        ModelInferenceRecord.tenant_id == context.tenant_id,
                        ModelInferenceRecord.model_alias == request.model_alias,
                        ModelInferenceRecord.created_at >= day_start,
                        ModelInferenceRecord.status == "PENDING",
                    )
                )
                or 0
            )
            if token_usage + pending_reservation + request.max_output_tokens > quota.tokens_per_day:
                raise ModelGatewayError("model_daily_token_quota_exceeded")
            session.add(
                ModelInferenceRecord(
                    inference_request_id=request.inference_request_id,
                    tenant_id=context.tenant_id,
                    model_alias=request.model_alias,
                    resolved_release_id=model.release_id,
                    manifest_hash=model.manifest_hash,
                    subject_id=request.subject_id,
                    trace_id=request.trace_id,
                    agent_run_id=request.agent_run_id,
                    diagnosis_run_id=request.diagnosis_run_id,
                    prompt_bundle_id=request.prompt_bundle_id,
                    index_release_id=request.index_release_id,
                    context_manifest_json=context_manifest.as_dict(),
                    context_hash=context_manifest.context_hash,
                    request_class=request.request_class,
                    data_classification=request.data_classification,
                    status="PENDING",
                    deadline=request.deadline,
                    max_output_tokens=request.max_output_tokens,
                    temperature=request.temperature,
                    prompt_hash=prompt_hash,
                    response_json=None,
                    response_hash=None,
                    usage_prompt_tokens=0,
                    usage_completion_tokens=0,
                    finish_reason=None,
                    latency_breakdown_json={},
                    safety_decision="PENDING",
                    guardrail_policy_version=self._guardrail.policy_version,
                    guardrail_findings_json=[],
                    degraded_from=None,
                    failure_reason=None,
                    completed_at=None,
                    created_at=now,
                    updated_at=now,
                )
            )
        return None

    def _reject_guardrail(
        self,
        context: TenantContext,
        request: GatewayRequest,
        model: ResolvedModel,
        decision: GuardrailDecision,
        *,
        phase: str,
    ) -> None:
        reason = f"model_{phase}_guardrail_blocked"
        now = datetime.now(UTC)
        with self._database.transaction(context) as session:
            record = session.scalar(
                select(ModelInferenceRecord)
                .where(
                    ModelInferenceRecord.tenant_id == context.tenant_id,
                    ModelInferenceRecord.inference_request_id == request.inference_request_id,
                )
                .with_for_update()
            )
            if record is None or record.status != "PENDING":
                raise ModelGatewayError("inference_audit_state_changed")
            record.status = "FAILED"
            record.safety_decision = "REJECTED"
            record.guardrail_policy_version = decision.policy_version
            record.guardrail_findings_json = [finding.audit_dict() for finding in decision.findings]
            record.failure_reason = reason
            record.completed_at = now
        if self._security_auditor is not None:
            self._security_auditor.record_context(
                tenant_id=context.tenant_id,
                subject_id=request.subject_id,
                action="model_gateway.guardrail",
                decision="deny",
                reason_code=reason,
                request_id=request.trace_id,
                resource_id=model.release_id,
            )

    def _finish_failure(
        self, context: TenantContext, inference_request_id: str, reason: str
    ) -> None:
        now = datetime.now(UTC)
        with self._database.transaction(context) as session:
            record = session.scalar(
                select(ModelInferenceRecord)
                .where(
                    ModelInferenceRecord.tenant_id == context.tenant_id,
                    ModelInferenceRecord.inference_request_id == inference_request_id,
                )
                .with_for_update()
            )
            if record is not None and record.status == "PENDING":
                record.status = "FAILED"
                record.safety_decision = "REJECTED"
                record.failure_reason = reason[:255]
                record.completed_at = now

    def _finish_success(
        self,
        context: TenantContext,
        request: GatewayRequest,
        model: ResolvedModel,
        result: TransportResult,
    ) -> GatewayResponse:
        now = datetime.now(UTC)
        with self._database.transaction(context) as session:
            record = session.get(ModelInferenceRecord, request.inference_request_id)
            if record is None or record.status != "PENDING":
                raise ModelGatewayError("inference_audit_state_changed")
            record.status = "SUCCEEDED"
            record.response_json = result.content
            record.response_hash = _digest(result.content)
            record.usage_prompt_tokens = result.prompt_tokens
            record.usage_completion_tokens = result.completion_tokens
            record.finish_reason = result.finish_reason[:64]
            record.latency_breakdown_json = result.latency_breakdown
            record.safety_decision = "ALLOWED"
            record.guardrail_policy_version = self._guardrail.policy_version
            record.guardrail_findings_json = []
            record.completed_at = now
            response = _response_from_record(record, replayed=False)
        return response


def _validate_request(request: GatewayRequest, now: datetime) -> None:
    if request.subject_id == "" or request.trace_id == "":
        raise ModelGatewayError("model_request_identity_invalid")
    if request.deadline.tzinfo is None or request.deadline <= now:
        raise ModelGatewayError("model_request_deadline_expired")
    if not 1 <= request.max_output_tokens <= 32768:
        raise ModelGatewayError("model_output_limit_invalid")
    if not math.isfinite(request.temperature) or not 0.0 <= request.temperature <= 2.0:
        raise ModelGatewayError("model_temperature_invalid")
    _validate_messages(request.messages)


def _context_manifest(request: GatewayRequest, model: ResolvedModel) -> ContextManifest:
    try:
        return build_context_manifest(
            trace_id=request.trace_id,
            prompt_bundle_id=request.prompt_bundle_id,
            prompt_bundle_hash=request.prompt_bundle_hash,
            model_release_id=model.release_id,
            retrieval_index_id=request.index_release_id,
            evidence=request.context_evidence,
            memories=request.context_memories,
            tool_results=request.context_tool_results,
            token_budget=request.context_token_budget or request.max_output_tokens,
            truncated_items=request.context_truncated_items,
            messages=request.messages,
            response_schema=request.response_schema,
        )
    except ValueError as exc:
        raise ModelGatewayError(str(exc)) from exc


def _validate_messages(messages: tuple[dict[str, Any], ...]) -> None:
    if not messages:
        raise ModelGatewayError("model_messages_invalid")
    image_count = 0
    total_image_bytes = 0
    total_text_characters = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system",
            "user",
            "assistant",
        }:
            raise ModelGatewayError("model_messages_invalid")
        content = message.get("content")
        if isinstance(content, str):
            if not content:
                raise ModelGatewayError("model_messages_invalid")
            total_text_characters += len(content)
            continue
        if not isinstance(content, list) or not content:
            raise ModelGatewayError("model_messages_invalid")
        for item in content:
            if not isinstance(item, dict):
                raise ModelGatewayError("model_messages_invalid")
            if item.get("type") == "text":
                text = item.get("text")
                if not isinstance(text, str) or not text:
                    raise ModelGatewayError("model_messages_invalid")
                total_text_characters += len(text)
                continue
            if item.get("type") != "image_url":
                raise ModelGatewayError("model_messages_invalid")
            image_url = item.get("image_url")
            if not isinstance(image_url, dict) or set(image_url) != {"url"}:
                raise ModelGatewayError("model_messages_invalid")
            content_bytes = _decode_image_data_url(image_url.get("url"))
            image_count += 1
            total_image_bytes += len(content_bytes)
    if (
        total_text_characters > MAX_MULTIMODAL_TEXT_CHARACTERS
        or image_count > MAX_MULTIMODAL_IMAGES
        or total_image_bytes > MAX_MULTIMODAL_TOTAL_BYTES
    ):
        raise ModelGatewayError("model_messages_limit_exceeded")


def _decode_image_data_url(value: object) -> bytes:
    if not isinstance(value, str):
        raise ModelGatewayError("model_image_invalid")
    prefixes = {
        "data:image/jpeg;base64,": b"\xff\xd8\xff",
        "data:image/png;base64,": b"\x89PNG\r\n\x1a\n",
    }
    signature = next((item for prefix, item in prefixes.items() if value.startswith(prefix)), None)
    if signature is None:
        raise ModelGatewayError("model_image_invalid")
    encoded = value.split(",", 1)[1]
    if len(encoded) > ((MAX_MULTIMODAL_IMAGE_BYTES + 2) // 3) * 4:
        raise ModelGatewayError("model_image_limit_exceeded")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ModelGatewayError("model_image_invalid") from exc
    if (
        not content
        or len(content) > MAX_MULTIMODAL_IMAGE_BYTES
        or not content.startswith(signature)
    ):
        raise ModelGatewayError("model_image_invalid")
    return content


def _validate_result(result: TransportResult, request: GatewayRequest) -> None:
    try:
        validate(instance=result.content, schema=request.response_schema)
    except (ValidationError, SchemaError) as exc:
        raise ModelGatewayError("model_response_schema_rejected") from exc
    if (
        result.prompt_tokens < 0
        or result.completion_tokens < 0
        or result.completion_tokens > request.max_output_tokens
        or any(not math.isfinite(value) or value < 0 for value in result.latency_breakdown.values())
    ):
        raise ModelGatewayError("model_response_usage_invalid")


def _response_from_record(record: ModelInferenceRecord, *, replayed: bool) -> GatewayResponse:
    if record.response_json is None or record.finish_reason is None or record.context_hash is None:
        raise ModelGatewayError("inference_audit_result_incomplete")
    return GatewayResponse(
        inference_request_id=record.inference_request_id,
        resolved_release_id=record.resolved_release_id,
        manifest_hash=record.manifest_hash,
        content=record.response_json,
        finish_reason=record.finish_reason,
        prompt_tokens=record.usage_prompt_tokens,
        completion_tokens=record.usage_completion_tokens,
        latency_breakdown=record.latency_breakdown_json,
        context_hash=record.context_hash,
        replayed=replayed,
    )


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(encoded.encode()).hexdigest()


def _manifest_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return sha256(encoded.encode()).hexdigest()


def _require_replay_binding(
    record: ModelInferenceRecord,
    context: TenantContext,
    request: GatewayRequest,
    model: ResolvedModel,
    prompt_hash: str,
    manifest: ContextManifest,
) -> None:
    if (
        record.tenant_id != context.tenant_id
        or record.subject_id != request.subject_id
        or record.agent_run_id != request.agent_run_id
        or record.diagnosis_run_id != request.diagnosis_run_id
        or record.prompt_hash != prompt_hash
        or record.resolved_release_id != model.release_id
        or record.manifest_hash != model.manifest_hash
        or record.context_hash != manifest.context_hash
        or not context_manifest_is_valid(record.context_manifest_json)
        or record.context_manifest_json.get("trace_id") != record.trace_id
    ):
        raise ModelGatewayError("inference_idempotency_conflict")
