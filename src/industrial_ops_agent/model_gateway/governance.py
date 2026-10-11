"""Tenant-visible model route, quota and inference audit governance."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.model_gateway.service import (
    EnvironmentAwareModelResolver,
    ModelGatewayError,
    normalize_model_environment,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    ModelAliasRecord,
    ModelGatewayQuotaRecord,
    ModelInferenceRecord,
    ModelReleaseRecord,
)


class ModelGatewayGovernanceConflict(RuntimeError):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ModelRouteView:
    alias: ModelAliasRecord
    quota: ModelGatewayQuotaRecord | None
    target_environment: str


@dataclass(frozen=True, slots=True)
class ModelRuntimeComponentView:
    binding_status: str
    reason: str | None
    alias: str | None = None
    release_id: str | None = None
    deployment_id: str | None = None
    provider: str | None = None
    processor_version: str | None = None
    model_version: str | None = None
    accelerator: str | None = None
    endpoint_status: str | None = None


@dataclass(frozen=True, slots=True)
class ModelRuntimeStatusView:
    status: str
    target_environment: str
    gateway_status: str
    gpu_serving_required: bool
    components: dict[str, ModelRuntimeComponentView]


class ModelGatewayGovernanceService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def list_routes(
        self, identity: IdentityContext, *, request_id: str
    ) -> Sequence[ModelRouteView]:
        self._require(identity, Action.READ_MODEL_GATEWAY, "model-gateway", request_id)
        with self._database.transaction(identity.tenant_context) as session:
            aliases = list(
                session.scalars(
                    select(ModelAliasRecord)
                    .where(ModelAliasRecord.tenant_id == identity.tenant_id)
                    .order_by(ModelAliasRecord.alias)
                )
            )
            quotas = {
                item.model_alias: item
                for item in session.scalars(
                    select(ModelGatewayQuotaRecord).where(
                        ModelGatewayQuotaRecord.tenant_id == identity.tenant_id
                    )
                )
            }
            environments = {
                release.release_id: release.target_environment
                for release in session.scalars(
                    select(ModelReleaseRecord).where(
                        ModelReleaseRecord.tenant_id == identity.tenant_id
                    )
                )
            }
            return [
                ModelRouteView(
                    alias,
                    quotas.get(alias.alias),
                    environments.get(alias.active_release_id, "UNRESOLVED"),
                )
                for alias in aliases
            ]

    def runtime_status(
        self,
        identity: IdentityContext,
        resolver: EnvironmentAwareModelResolver | None,
        *,
        required_environment: object,
        alias_name: str,
        request_id: str,
    ) -> ModelRuntimeStatusView:
        self._require(identity, Action.READ_MODEL_GATEWAY, "model-runtime-status", request_id)
        target_environment = normalize_model_environment(required_environment)
        if resolver is None:
            return _unavailable_runtime(target_environment, "model_gateway_not_configured")
        if (
            resolver.required_environment != target_environment
            or resolver.required_alias != alias_name
        ):
            return _unavailable_runtime(target_environment, "model_release_environment_mismatch")
        try:
            model = resolver.resolve(identity.tenant_context, alias_name)
        except ModelGatewayError as exc:
            return _unavailable_runtime(target_environment, _runtime_reason(exc.reason))

        with self._database.transaction(identity.tenant_context) as session:
            release = session.scalar(
                select(ModelReleaseRecord).where(
                    ModelReleaseRecord.tenant_id == identity.tenant_id,
                    ModelReleaseRecord.release_id == model.release_id,
                )
            )
        if release is None or release.target_environment != target_environment:
            return _unavailable_runtime(target_environment, "model_release_environment_mismatch")
        hardware = release.manifest_json.get("hardware_profile")
        accelerator = hardware.get("accelerator") if isinstance(hardware, dict) else None
        gpu_count = hardware.get("gpu_count") if isinstance(hardware, dict) else None
        if (
            not isinstance(accelerator, str)
            or "GPU" not in accelerator.upper()
            or not isinstance(gpu_count, int)
            or gpu_count < 1
        ):
            return _unavailable_runtime(target_environment, "gpu_capacity_unavailable")

        common = {
            "binding_status": "READY",
            "reason": None,
            "alias": model.alias,
            "release_id": model.release_id,
            "deployment_id": model.deployment_id,
            "accelerator": accelerator,
            "endpoint_status": "READY",
        }
        return ModelRuntimeStatusView(
            status="READY",
            target_environment=target_environment,
            gateway_status="READY",
            gpu_serving_required=True,
            components={
                "ocr": ModelRuntimeComponentView(
                    **common,
                    provider="paddleocr",
                    processor_version=model.multimodal_model_ids["ocr"],
                ),
                "vlm": ModelRuntimeComponentView(
                    **common,
                    provider="model-gateway",
                    model_version=model.multimodal_model_ids["vlm"],
                ),
                "diagnosis": ModelRuntimeComponentView(
                    **common,
                    provider="model-gateway",
                    model_version=model.runtime_profile,
                ),
            },
        )

    def list_inferences(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
        limit: int = 50,
    ) -> Sequence[ModelInferenceRecord]:
        from industrial_ops_agent.maintenance_planning.review_isolation import (
            diagnosis_for_agent,
            record_disclosure,
        )
        from industrial_ops_agent.persistence.models import (
            MaintenancePlanningContributionRecord,
            MaintenancePlanningCouncilRecord,
        )

        self._require(identity, Action.READ_MODEL_GATEWAY, "model-inferences", request_id)
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        with self._database.transaction(identity.tenant_context) as session:
            records = list(
                session.scalars(
                    select(ModelInferenceRecord)
                    .where(ModelInferenceRecord.tenant_id == identity.tenant_id)
                    .order_by(ModelInferenceRecord.created_at.desc())
                    .limit(limit)
                )
            )
            diagnoses: set[str] = set()
            for record in records:
                if record.diagnosis_run_id is not None:
                    diagnoses.add(record.diagnosis_run_id)
                if record.agent_run_id is not None:
                    diagnoses.add(
                        diagnosis_for_agent(session, identity.tenant_id, record.agent_run_id)
                    )
            # Council inference records can precede the Agent-run linkage.
            diagnoses.update(
                session.scalars(
                    select(MaintenancePlanningCouncilRecord.diagnosis_run_id)
                    .join(
                        MaintenancePlanningContributionRecord,
                        (
                            MaintenancePlanningContributionRecord.tenant_id
                            == MaintenancePlanningCouncilRecord.tenant_id
                        )
                        & (
                            MaintenancePlanningContributionRecord.council_id
                            == MaintenancePlanningCouncilRecord.council_id
                        ),
                    )
                    .where(
                        MaintenancePlanningCouncilRecord.tenant_id == identity.tenant_id,
                        MaintenancePlanningContributionRecord.inference_request_id.in_(
                            [record.inference_request_id for record in records]
                        ),
                    )
                )
            )
            record_disclosure(
                session,
                identity,
                diagnoses,
                resource_kind="model-inference-list",
                resource_id="model-inferences",
                request_id=request_id,
            )
            return records

    def update_quota(
        self,
        identity: IdentityContext,
        alias_name: str,
        *,
        requests_per_minute: int,
        tokens_per_day: int,
        max_output_tokens: int,
        allowed_request_classes: list[str],
        enabled: bool,
        expected_version: int,
        request_id: str,
    ) -> ModelRouteView:
        self._require(identity, Action.MANAGE_MODEL_GATEWAY_QUOTA, alias_name, request_id)
        if (
            not 1 <= requests_per_minute <= 100_000
            or not 1 <= tokens_per_day <= 10_000_000_000
            or not 1 <= max_output_tokens <= 32_768
            or not allowed_request_classes
            or any(
                value
                not in {
                    "DIAGNOSIS",
                    "VLM",
                    "ASR",
                    "TTS",
                    "GRAPH_CAUSAL_EXTRACTION",
                    "MAINTENANCE_PLANNING",
                }
                for value in allowed_request_classes
            )
        ):
            raise ValueError("model quota policy is invalid")
        with self._database.transaction(identity.tenant_context) as session:
            alias = session.scalar(
                select(ModelAliasRecord).where(
                    ModelAliasRecord.tenant_id == identity.tenant_id,
                    ModelAliasRecord.alias == alias_name,
                )
            )
            quota = session.scalar(
                select(ModelGatewayQuotaRecord)
                .where(
                    ModelGatewayQuotaRecord.tenant_id == identity.tenant_id,
                    ModelGatewayQuotaRecord.model_alias == alias_name,
                )
                .with_for_update()
            )
            if alias is None or quota is None:
                raise ModelGatewayGovernanceConflict("model_route_not_found")
            if quota.version != expected_version:
                raise ModelGatewayGovernanceConflict("model_quota_version_mismatch", quota.version)
            quota.requests_per_minute = requests_per_minute
            quota.tokens_per_day = tokens_per_day
            quota.max_output_tokens = max_output_tokens
            quota.allowed_request_classes = sorted(set(allowed_request_classes))
            quota.enabled = enabled
            quota.created_by_subject_id = identity.subject_id
            quota.version += 1
            session.flush()
            release = session.scalar(
                select(ModelReleaseRecord).where(
                    ModelReleaseRecord.tenant_id == identity.tenant_id,
                    ModelReleaseRecord.release_id == alias.active_release_id,
                )
            )
            return ModelRouteView(
                alias,
                quota,
                release.target_environment if release is not None else "UNRESOLVED",
            )

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        resource_id: str,
        request_id: str,
    ) -> None:
        self._authorizer.require(
            identity,
            action,
            ResourceContext(tenant_id=identity.tenant_id, resource_id=resource_id),
            request_id=request_id,
        )


def _runtime_reason(reason: str) -> str:
    return {
        "model_deployment_unavailable": "model_deployment_not_ready",
        "model_component_binding_incomplete": "model_component_binding_missing",
    }.get(reason, reason)


def _unavailable_runtime(environment: str, reason: str) -> ModelRuntimeStatusView:
    component = ModelRuntimeComponentView(
        binding_status="MODEL_UNAVAILABLE",
        reason=reason,
    )
    return ModelRuntimeStatusView(
        status="MODEL_UNAVAILABLE",
        target_environment=environment,
        gateway_status="MODEL_UNAVAILABLE",
        gpu_serving_required=True,
        components={"ocr": component, "vlm": component, "diagnosis": component},
    )
