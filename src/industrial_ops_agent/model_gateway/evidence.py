"""Current, tenant-bound proof that a governed model request really executed."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from sqlalchemy import select

from industrial_ops_agent.model_gateway.service import (
    EnvironmentAwareModelResolver,
    ModelGatewayError,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import ModelInferenceRecord, ModelReleaseRecord
from industrial_ops_agent.persistence.tenant import TenantContext

ModelComponent = Literal["vlm", "diagnosis"]


@dataclass(frozen=True, slots=True)
class ModelExecutionEvidence:
    component: ModelComponent
    request_class: str
    target_environment: str
    model_alias: str
    release_id: str
    deployment_id: str
    inference_request_id: str
    processor_version: str
    manifest_hash: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    guardrail_decision: str
    guardrail_policy_version: str
    request_id: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_processor_versions(self, prefix: str) -> dict[str, str]:
        return {
            f"{prefix}.{key}": str(value)
            for key, value in self.as_dict().items()
            if key != "component"
        }

    @classmethod
    def from_processor_versions(
        cls,
        values: dict[str, str],
        *,
        prefix: str,
        component: ModelComponent,
    ) -> ModelExecutionEvidence | None:
        def field(name: str) -> str:
            value = values.get(f"{prefix}.{name}", "").strip()
            if not value:
                raise ValueError("stored model execution evidence is incomplete")
            return value

        if not values.get(f"{prefix}.inference_request_id"):
            return None
        try:
            return cls(
                component=component,
                request_class=field("request_class"),
                target_environment=field("target_environment"),
                model_alias=field("model_alias"),
                release_id=field("release_id"),
                deployment_id=field("deployment_id"),
                inference_request_id=field("inference_request_id"),
                processor_version=field("processor_version"),
                manifest_hash=field("manifest_hash"),
                latency_ms=float(field("latency_ms")),
                prompt_tokens=int(field("prompt_tokens")),
                completion_tokens=int(field("completion_tokens")),
                guardrail_decision=field("guardrail_decision"),
                guardrail_policy_version=field("guardrail_policy_version"),
                request_id=field("request_id"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("stored model execution evidence is invalid") from exc


def require_successful_model_execution(
    database: Database,
    context: TenantContext,
    *,
    inference_request_id: str,
    component: ModelComponent,
    expected_alias: str,
    expected_release_id: str,
    expected_manifest_hash: str,
    expected_subject_id: str,
    expected_trace_id: str,
    expected_agent_run_id: str | None = None,
    expected_diagnosis_run_id: str | None = None,
) -> ModelExecutionEvidence:
    reason = f"{component}_inference_audit_mismatch"
    with database.transaction(context) as session:
        record = session.scalar(
            select(ModelInferenceRecord).where(
                ModelInferenceRecord.tenant_id == context.tenant_id,
                ModelInferenceRecord.inference_request_id == inference_request_id,
            )
        )
        release = session.scalar(
            select(ModelReleaseRecord).where(
                ModelReleaseRecord.tenant_id == context.tenant_id,
                ModelReleaseRecord.release_id == expected_release_id,
            )
        )
        if record is None or release is None:
            raise ModelGatewayError(reason)
        target_environment = release.target_environment

    try:
        resolved = EnvironmentAwareModelResolver(database, target_environment).resolve(
            context, expected_alias
        )
    except (ModelGatewayError, ValueError) as exc:
        raise ModelGatewayError(reason) from exc
    expected_class = "VLM" if component == "vlm" else "DIAGNOSIS"
    checks = (
        record.status == "SUCCEEDED",
        record.response_json is not None,
        record.completed_at is not None,
        record.degraded_from is None,
        record.safety_decision == "ALLOWED",
        record.request_class == expected_class,
        record.model_alias == expected_alias,
        record.resolved_release_id == expected_release_id == resolved.release_id,
        record.manifest_hash == expected_manifest_hash == resolved.manifest_hash,
        record.subject_id == expected_subject_id == context.subject_id,
        record.trace_id == expected_trace_id,
        record.agent_run_id == expected_agent_run_id,
        record.diagnosis_run_id == expected_diagnosis_run_id,
        resolved.deployment_id is not None,
    )
    if not all(checks):
        raise ModelGatewayError(reason)
    processor_version = (
        resolved.multimodal_model_ids["vlm"]
        if component == "vlm"
        else resolved.runtime_profile
    )
    latency_ms = record.latency_breakdown_json.get("upstream_total_ms")
    if latency_ms is None:
        latency_ms = sum(record.latency_breakdown_json.values())
    return ModelExecutionEvidence(
        component=component,
        request_class=record.request_class,
        target_environment=resolved.target_environment,
        model_alias=record.model_alias,
        release_id=record.resolved_release_id,
        deployment_id=resolved.deployment_id,
        inference_request_id=record.inference_request_id,
        processor_version=processor_version,
        manifest_hash=record.manifest_hash,
        latency_ms=float(latency_ms),
        prompt_tokens=record.usage_prompt_tokens,
        completion_tokens=record.usage_completion_tokens,
        guardrail_decision=record.safety_decision,
        guardrail_policy_version=record.guardrail_policy_version,
        request_id=record.trace_id,
    )
