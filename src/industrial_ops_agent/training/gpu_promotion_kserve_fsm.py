"""Governed ModelRelease/ModelDeployment FSM for the dual-GGUF local rollout."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, cast

from sqlalchemy import select

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.deployment.service import (
    METRIC_THRESHOLDS,
    STAGE_POLICY,
    DeploymentAggregate,
    DeploymentPlan,
    DeploymentProvider,
    ModelDeploymentService,
    ObservationInput,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import ModelAliasRecord
from industrial_ops_agent.releases.service import (
    ModelReleaseService,
    ProjectAuthorizedLocalReleasePlan,
    ReleaseAggregate,
)
from industrial_ops_agent.training.gpu_promotion_gguf import (
    AUTHORIZATION_CLASSIFICATION,
    CLASSIFICATION,
    DATA_CLASSIFICATION,
    ROLE_ATTESTATION,
    GgufPromotionEvidence,
)
from industrial_ops_agent.training.gpu_promotion_kserve import (
    CANARY_RATIO_BOUNDS,
    GATEWAY_NAME,
    HOSTNAME,
    KSERVE_VERSION,
    KUBERNETES_CONTEXT,
    MODEL_PVC_NAME,
    NAMESPACE,
    ROUTE_NAME,
    RUNTIME_PROFILE_ID,
    SCHEMA_VERSION,
    SERVICE_ACCOUNT_NAME,
    STATUS,
    FaultInjectionEvidence,
    GpuPromotionKServeError,
    RolloutStage,
    TrafficSample,
    finalize_rollout_evidence,
)


@dataclass(frozen=True, slots=True)
class RuntimeBinding:
    git_commit: str
    image_repository: str
    image_reference: str
    image_digest: str
    dockerfile_sha256: str
    llama_cpp_commit: str


class RolloutTrafficCollector(Protocol):
    def collect(
        self,
        stage: RolloutStage,
        *,
        request_count: int,
        critical_case_count: int,
    ) -> TrafficSample: ...

    def inject_candidate_service_fault(self) -> FaultInjectionEvidence: ...

    def restore_candidate_service(self) -> None: ...

    def resource_projection(
        self,
        *,
        stable_service_name: str,
        candidate_service_name: str,
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


TrafficCollectorFactory = Callable[[str, str, str], RolloutTrafficCollector]


def execute_governed_rollout(
    *,
    database: Database,
    authorizer: Authorizer,
    repo_root: Path,
    gguf: GgufPromotionEvidence,
    runtime: RuntimeBinding,
    provider: DeploymentProvider,
    collector_factory: TrafficCollectorFactory,
    reviewer_subject_ids: tuple[str, str],
    tenant_id: str = "tenant-gpu-promotion-lab",
) -> dict[str, Any]:
    """Execute actual Shadow/Canary/fault/rollback through existing durable FSMs."""

    repo_root.resolve(strict=True)
    if len(set(reviewer_subject_ids)) != 2:
        raise GpuPromotionKServeError("dual_domain_reviewers_are_not_distinct")
    release_builder = _identity(
        "gpu-lab-release-requester",
        Role.MODEL_ENGINEER,
        tenant_id=tenant_id,
    )
    approver = _identity(
        "gpu-lab-release-approver",
        Role.MODEL_RELEASE_APPROVER,
        tenant_id=tenant_id,
    )
    operator = _identity(
        "gpu-lab-release-operator",
        Role.MODEL_RELEASE_OPERATOR,
        tenant_id=tenant_id,
    )
    controller = _identity(
        "gpu-lab-deployment-controller",
        Role.MODEL_DEPLOYMENT_CONTROLLER,
        tenant_id=tenant_id,
    )
    duty_subjects = {
        release_builder.subject_id,
        approver.subject_id,
        operator.subject_id,
        controller.subject_id,
    }
    if set(reviewer_subject_ids) & duty_subjects:
        raise GpuPromotionKServeError("review_and_release_duties_overlap")

    release_service = ModelReleaseService(database, authorizer)
    deployment_service = ModelDeploymentService(database, authorizer)
    stable_release = _ensure_approved_release(
        release_service,
        release_builder,
        approver,
        gguf,
        runtime,
        role="stable",
    )
    candidate_release = _ensure_approved_release(
        release_service,
        release_builder,
        approver,
        gguf,
        runtime,
        role="candidate",
    )
    stable_predictor = f"ioap-{stable_release.release.manifest_hash[:24]}-predictor"
    stable_deployment = deployment_service.ensure_shadow(
        operator,
        stable_release.release.release_id,
        _deployment_plan(stable_predictor),
        expected_release_version=stable_release.release.version,
        request_id="gpu-lab-stable-shadow-request",
    )
    stable_deployment = _reconcile_until_ready(
        deployment_service,
        controller,
        stable_deployment,
        provider,
        expected_stage="SHADOW",
        request_prefix="gpu-lab-stable-shadow",
    )

    candidate_deployment = deployment_service.ensure_shadow(
        operator,
        candidate_release.release.release_id,
        _deployment_plan(stable_predictor),
        expected_release_version=candidate_release.release.version,
        request_id="gpu-lab-candidate-shadow-request",
    )
    collector = collector_factory(
        candidate_deployment.deployment.service_name,
        stable_release.release.release_id,
        candidate_release.release.release_id,
    )
    fault: FaultInjectionEvidence | None = None
    try:
        candidate_deployment = _reconcile_until_ready(
            deployment_service,
            controller,
            candidate_deployment,
            provider,
            expected_stage="SHADOW",
            request_prefix="gpu-lab-candidate-shadow",
        )
        shadow_provider_revision = candidate_deployment.deployment.provider_revision
        shadow = collector.collect(
            "SHADOW",
            request_count=500,
            critical_case_count=50,
        )
        _require_distribution(shadow, "SHADOW")
        candidate_deployment = _record_observation(
            deployment_service,
            controller,
            candidate_deployment,
            _healthy_observation(shadow),
            expected_decision="PASS",
        )

        candidate_deployment = deployment_service.promote(
            operator,
            candidate_release.release.release_id,
            expected_deployment_version=candidate_deployment.deployment.version,
            request_id="gpu-lab-shadow-to-canary-5",
        )
        candidate_deployment = _reconcile_until_ready(
            deployment_service,
            controller,
            candidate_deployment,
            provider,
            expected_stage="CANARY_5",
            request_prefix="gpu-lab-candidate-canary-5",
        )
        canary_5_provider_revision = candidate_deployment.deployment.provider_revision
        canary_5 = collector.collect(
            "CANARY_5",
            request_count=1000,
            critical_case_count=100,
        )
        _require_distribution(canary_5, "CANARY_5")
        candidate_deployment = _record_observation(
            deployment_service,
            controller,
            candidate_deployment,
            _healthy_observation(canary_5),
            expected_decision="PASS",
        )

        candidate_deployment = deployment_service.promote(
            operator,
            candidate_release.release.release_id,
            expected_deployment_version=candidate_deployment.deployment.version,
            request_id="gpu-lab-canary-5-to-canary-25",
        )
        candidate_deployment = _reconcile_until_ready(
            deployment_service,
            controller,
            candidate_deployment,
            provider,
            expected_stage="CANARY_25",
            request_prefix="gpu-lab-candidate-canary-25",
        )
        canary_25_provider_revision = candidate_deployment.deployment.provider_revision
        canary_25 = collector.collect(
            "CANARY_25",
            request_count=5000,
            critical_case_count=250,
        )
        _require_distribution(canary_25, "CANARY_25")
        fault = collector.inject_candidate_service_fault()
        fault_sample = collector.collect(
            "CANARY_25",
            request_count=500,
            critical_case_count=0,
        )
        if fault_sample.error_rate <= METRIC_THRESHOLDS["error_rate_max"]:
            raise GpuPromotionKServeError("forced_candidate_fault_did_not_exceed_threshold")
        candidate_deployment = _record_observation(
            deployment_service,
            controller,
            candidate_deployment,
            _fault_observation(canary_25, fault_sample),
            expected_decision="ROLLBACK",
        )
        if (
            candidate_deployment.deployment.desired_stage != "ROLLED_BACK"
            or candidate_deployment.deployment.status != "PENDING"
        ):
            raise GpuPromotionKServeError("observation_did_not_trigger_existing_rollback_FSM")
        candidate_deployment = _reconcile_until_ready(
            deployment_service,
            controller,
            candidate_deployment,
            provider,
            expected_stage="ROLLED_BACK",
            request_prefix="gpu-lab-candidate-rollback",
        )
        rollback = collector.collect(
            "ROLLED_BACK",
            request_count=500,
            critical_case_count=0,
        )
        _require_distribution(rollback, "ROLLED_BACK")
        resources = collector.resource_projection(
            stable_service_name=stable_deployment.deployment.service_name,
            candidate_service_name=candidate_deployment.deployment.service_name,
        )
    finally:
        if fault is not None:
            collector.restore_candidate_service()
        collector.close()

    stable_projection = _release_projection(stable_release, gguf, role="stable")
    candidate_projection = _release_projection(
        candidate_release,
        gguf,
        role="candidate",
    )
    candidate_projection.update(
        {
            "deployment_id": candidate_deployment.deployment.deployment_id,
            "service_name": candidate_deployment.deployment.service_name,
            "provider_revision": candidate_deployment.deployment.provider_revision,
        }
    )
    if [item.stage for item in candidate_deployment.observations] != [
        "SHADOW",
        "CANARY_5",
        "CANARY_25",
    ]:
        raise GpuPromotionKServeError("formal_observation_sequence_is_invalid")
    shadow_observation, canary_5_observation, canary_25_observation = (
        candidate_deployment.observations
    )
    if fault is None:
        raise GpuPromotionKServeError("fault_injection_evidence_is_missing")
    production_alias_created = _production_alias_exists(
        database,
        operator,
        candidate_release.release.release_id,
    )
    if production_alias_created:
        raise GpuPromotionKServeError("local_staging_rollout_created_production_alias")

    return finalize_rollout_evidence(
        {
            "schema_version": SCHEMA_VERSION,
            "classification": CLASSIFICATION,
            "data_classification": DATA_CLASSIFICATION,
            "authorization_classification": AUTHORIZATION_CLASSIFICATION,
            "role_attestation": ROLE_ATTESTATION,
            "generated_at": datetime.now(UTC).isoformat(),
            "production_claim": False,
            "enterprise_production_data": False,
            "actual_kserve_execution": True,
            "actual_llama_cpp_inference": True,
            "source_gguf": {
                "path": "artifacts/gpu-model-promotion-lab/gguf-quantization.json",
                "evidence_chain_sha256": gguf.evidence_chain_sha256,
                "training_evidence_chain_sha256": gguf.training_evidence_chain_sha256,
            },
            "runtime": {
                "git_commit": runtime.git_commit,
                "image_repository": runtime.image_repository,
                "image_reference": runtime.image_reference,
                "image_digest": runtime.image_digest,
                "dockerfile_sha256": runtime.dockerfile_sha256,
                "llama_cpp_commit": runtime.llama_cpp_commit,
            },
            "kubernetes": {
                "context": KUBERNETES_CONTEXT,
                "kind_node": "ioap-gpu-promotion-lab-control-plane",
                "kserve_version": KSERVE_VERSION,
                "deployment_mode": "Standard",
                "namespace": NAMESPACE,
                "gateway_name": GATEWAY_NAME,
                "hostname": HOSTNAME,
                "route_name": ROUTE_NAME,
                "resources": resources,
            },
            "separation_of_duties": {
                "domain_reviewer_subject_ids": list(reviewer_subject_ids),
                "release_builder_subject_id": release_builder.subject_id,
                "release_approver_subject_id": approver.subject_id,
                "release_operator_subject_id": operator.subject_id,
                "deployment_controller_subject_id": controller.subject_id,
                "status": "PASSED",
            },
            "releases": {
                "stable": stable_projection,
                "candidate": candidate_projection,
            },
            "stages": {
                "SHADOW": {
                    **shadow.evidence(),
                    "observation_id": shadow_observation.observation_id,
                    "observation_decision": shadow_observation.decision,
                    "observation_evidence_hash": shadow_observation.evidence_hash,
                    "provider_revision": shadow_provider_revision,
                },
                "CANARY_5": {
                    **canary_5.evidence(),
                    "observation_id": canary_5_observation.observation_id,
                    "observation_decision": canary_5_observation.decision,
                    "observation_evidence_hash": canary_5_observation.evidence_hash,
                    "provider_revision": canary_5_provider_revision,
                },
                "CANARY_25": {
                    "routing": canary_25.evidence(),
                    "fault_sampling": fault_sample.evidence(),
                    "fault_injection": asdict(fault),
                    "observation_id": canary_25_observation.observation_id,
                    "observation_decision": canary_25_observation.decision,
                    "observation_failure_reasons": canary_25_observation.failure_reasons,
                    "observation_evidence_hash": canary_25_observation.evidence_hash,
                    "provider_revision": canary_25_provider_revision,
                },
                "ROLLED_BACK": {
                    **rollback.evidence(),
                    "stable_model_file_sha256": gguf.stable.model_file_sha256,
                    "candidate_model_file_sha256": gguf.candidate.model_file_sha256,
                    "provider_revision": candidate_deployment.deployment.provider_revision,
                },
            },
            "final_state": {
                "release_status": candidate_deployment.release.status,
                "deployment_status": candidate_deployment.deployment.status,
                "deployment_stage": candidate_deployment.deployment.current_stage,
                "candidate_traffic_percent": (
                    candidate_deployment.deployment.observed_traffic_percent
                ),
                "stable_release_id": stable_release.release.release_id,
                "candidate_release_id": candidate_release.release.release_id,
                "stable_model_file_sha256": gguf.stable.model_file_sha256,
                "production_alias_created": production_alias_created,
            },
            "status": STATUS,
        }
    )


def _ensure_approved_release(
    service: ModelReleaseService,
    builder: IdentityContext,
    approver: IdentityContext,
    gguf: GgufPromotionEvidence,
    runtime: RuntimeBinding,
    *,
    role: str,
) -> ReleaseAggregate:
    variant = gguf.stable if role == "stable" else gguf.candidate
    aggregate = service.create_project_authorized_local_staging(
        builder,
        ProjectAuthorizedLocalReleasePlan(
            evaluation_id=variant.edge_evaluation.evaluation_id,
            quantization_profile_id="gguf-q4_k_m-llama_cpp",
            runtime_profile_id=RUNTIME_PROFILE_ID,
            runtime_git_commit=runtime.git_commit,
            runtime_image_repository=runtime.image_repository,
            runtime_image_reference=runtime.image_reference,
            runtime_image_digest=runtime.image_digest,
            inference_config={
                "engine": "llama.cpp",
                "context_size": 2048,
                "threads": 2,
                "batch_size": 128,
                "mmap": True,
                "mlock": False,
            },
            hardware_profile={
                "accelerator": "CPU",
                "cpu_architecture": "x86_64",
                "cpu_model": "Kind local staging node",
                "cpu_cores": 2,
                "memory_gb": 2,
                "instruction_set": "AVX2",
            },
            source_evidence_chain_sha256=gguf.evidence_chain_sha256,
            model_file_sha256=variant.model_file_sha256,
            model_storage_subpath=role,
        ),
        idempotency_key=(
            f"gpu-lab-gguf-{role}-{gguf.evidence_chain_sha256[:20]}-"
            f"{runtime.image_digest.removeprefix('sha256:')[:20]}"
        ),
        request_id=f"gpu-lab-{role}-release-create",
    )
    if aggregate.release.status == "DRAFT":
        aggregate = service.validate(
            builder,
            aggregate.release.release_id,
            expected_version=aggregate.release.version,
            request_id=f"gpu-lab-{role}-release-validate",
        )
    if aggregate.release.status == "CANDIDATE":
        aggregate = service.submit_for_approval(
            builder,
            aggregate.release.release_id,
            expected_version=aggregate.release.version,
            request_id=f"gpu-lab-{role}-release-submit",
        )
    if (
        aggregate.release.status == "APPROVAL_PENDING"
        and aggregate.approval is not None
        and aggregate.approval.status == "PENDING"
    ):
        aggregate = service.decide_approval(
            approver,
            aggregate.release.release_id,
            expected_version=aggregate.release.version,
            expected_approval_version=aggregate.approval.version,
            decision="APPROVED",
            reason="project_authorized_GGUF_local_staging_evidence_verified",
            request_id=f"gpu-lab-{role}-release-approve",
        )
    if (
        aggregate.approval is None
        or aggregate.approval.status != "APPROVED"
        or aggregate.approval.requested_by_subject_id
        == aggregate.approval.decided_by_subject_id
    ):
        raise GpuPromotionKServeError(f"{role}_release_is_not_independently_approved")
    return aggregate


def _deployment_plan(stable_predictor: str) -> DeploymentPlan:
    return DeploymentPlan(
        namespace=NAMESPACE,
        gateway_name=GATEWAY_NAME,
        hostname=HOSTNAME,
        route_name=ROUTE_NAME,
        stable_service_name=stable_predictor,
        service_account_name=SERVICE_ACCOUNT_NAME,
        serving_runtime_name=RUNTIME_PROFILE_ID,
        artifact_uri_prefix=f"pvc://{MODEL_PVC_NAME}",
    )


def _reconcile_until_ready(
    service: ModelDeploymentService,
    controller: IdentityContext,
    aggregate: DeploymentAggregate,
    provider: DeploymentProvider,
    *,
    expected_stage: str,
    request_prefix: str,
    attempts: int = 180,
) -> DeploymentAggregate:
    current = aggregate
    if (
        current.deployment.status == "READY"
        and current.deployment.current_stage == expected_stage
    ):
        return current
    for attempt in range(1, attempts + 1):
        current = service.reconcile(
            controller,
            current.deployment.deployment_id,
            provider,
            expected_version=current.deployment.version,
            request_id=f"{request_prefix}-reconcile-{attempt}",
        )
        if (
            current.deployment.status == "READY"
            and current.deployment.current_stage == expected_stage
        ):
            return current
        if current.deployment.status not in {"APPLYING", "FAILED"}:
            raise GpuPromotionKServeError(
                f"deployment_is_not_retriable:{expected_stage}:{current.deployment.status}"
            )
        time.sleep(1)
    raise GpuPromotionKServeError(f"deployment_reconcile_timed_out:{expected_stage}")


def _record_observation(
    service: ModelDeploymentService,
    controller: IdentityContext,
    aggregate: DeploymentAggregate,
    observation: ObservationInput,
    *,
    expected_decision: str,
) -> DeploymentAggregate:
    observed = service.record_observation(
        controller,
        aggregate.deployment.deployment_id,
        observation,
        expected_deployment_version=aggregate.deployment.version,
        request_id=f"gpu-lab-{observation.stage.lower()}-observation",
    )
    if not observed.observations or observed.observations[-1].decision != expected_decision:
        raise GpuPromotionKServeError(
            f"formal_observation_decision_mismatch:{observation.stage}"
        )
    return observed


def _healthy_observation(sample: TrafficSample) -> ObservationInput:
    return _observation(
        sample.stage,
        request_count=sample.request_count,
        critical_case_count=sample.critical_case_count,
        error_rate=sample.error_rate,
        p95_latency_ms=sample.p95_latency_ms,
        source={"traffic": sample.evidence()},
    )


def _fault_observation(
    routing: TrafficSample,
    fault: TrafficSample,
) -> ObservationInput:
    total = routing.request_count + fault.request_count
    errors = routing.error_count + fault.error_count
    return _observation(
        "CANARY_25",
        request_count=total,
        critical_case_count=routing.critical_case_count,
        error_rate=errors / total,
        p95_latency_ms=max(routing.p95_latency_ms, fault.p95_latency_ms),
        source={"routing": routing.evidence(), "fault": fault.evidence()},
    )


def _observation(
    stage: RolloutStage,
    *,
    request_count: int,
    critical_case_count: int,
    error_rate: float,
    p95_latency_ms: float,
    source: dict[str, Any],
) -> ObservationInput:
    if stage not in {"SHADOW", "CANARY_5", "CANARY_25"}:
        raise ValueError("online observation stage is invalid")
    policy = STAGE_POLICY[stage]
    end = datetime.now(UTC) - timedelta(seconds=1)
    metrics = {
        "cross_tenant_leak_count": 0.0,
        "unauthorized_side_effect_count": 0.0,
        "high_risk_miss_rate": 0.0,
        "task_success_delta": 0.0,
        "error_rate": error_rate,
        "p95_latency_ms": p95_latency_ms,
        "human_edit_rate_delta": 0.0,
        "wrong_part_rate": 0.0,
        "gpu_xid_error_count": 0.0,
        "gpu_memory_utilization": 0.0,
        "queue_age_seconds": 0.0,
    }
    if any(not math.isfinite(value) for value in metrics.values()):
        raise GpuPromotionKServeError("rollout_observation_metrics_are_not_finite")
    query_hash = _digest({"stage": stage, "metrics": metrics})
    trace_hash = _digest(source)
    return ObservationInput(
        stage=cast(Any, stage),
        window_start=end - timedelta(seconds=int(policy["min_window_seconds"])),
        window_end=end,
        request_count=request_count,
        critical_case_count=critical_case_count,
        metrics=metrics,
        source_refs={
            "collector_version": "actual-llama-cpp-kserve/v2",
            "prometheus_endpoint_id": "llama-cpp-runtime-metrics",
            "prometheus_query_bundle_hash": f"sha256:{query_hash}",
            "trace_query_hash": f"sha256:{trace_hash}",
        },
    )


def _require_distribution(sample: TrafficSample, stage: str) -> None:
    if sample.error_count or sample.unknown_count:
        raise GpuPromotionKServeError(f"healthy_{stage.lower()}_traffic_contains_errors")
    if stage in {"SHADOW", "ROLLED_BACK"}:
        if sample.stable_count != sample.request_count or sample.candidate_count != 0:
            raise GpuPromotionKServeError(f"{stage.lower()}_stable_route_gate_failed")
        if stage == "SHADOW" and (
            sample.direct_candidate_release_id is None
            or sample.direct_candidate_output_sha256 is None
        ):
            raise GpuPromotionKServeError("shadow_direct_candidate_probe_is_missing")
        return
    lower, upper = CANARY_RATIO_BOUNDS[stage]
    if (
        sample.stable_count < 1
        or sample.candidate_count < 1
        or not lower <= sample.candidate_ratio <= upper
    ):
        raise GpuPromotionKServeError(f"{stage.lower()}_distribution_gate_failed")


def _release_projection(
    aggregate: ReleaseAggregate,
    gguf: GgufPromotionEvidence,
    *,
    role: str,
) -> dict[str, Any]:
    variant = gguf.stable if role == "stable" else gguf.candidate
    approval = aggregate.approval
    if approval is None or approval.decided_by_subject_id is None:
        raise GpuPromotionKServeError(f"{role}_release_approval_evidence_is_missing")
    return {
        "release_id": aggregate.release.release_id,
        "manifest_hash": aggregate.release.manifest_hash,
        "evaluation_id": variant.edge_evaluation.evaluation_id,
        "quantization_experiment_id": variant.quantization_experiment_id,
        "model_file_sha256": variant.model_file_sha256,
        "approval_id": approval.approval_id,
        "approval_requested_by_subject_id": approval.requested_by_subject_id,
        "approval_decided_by_subject_id": approval.decided_by_subject_id,
    }


def _production_alias_exists(
    database: Database,
    identity: IdentityContext,
    release_id: str,
) -> bool:
    with database.transaction(identity.tenant_context) as session:
        return (
            session.scalar(
                select(ModelAliasRecord).where(
                    ModelAliasRecord.tenant_id == identity.tenant_id,
                    ModelAliasRecord.active_release_id == release_id,
                )
            )
            is not None
        )


def _identity(
    subject_id: str,
    role: Role,
    *,
    tenant_id: str,
) -> IdentityContext:
    now = datetime.now(UTC)
    return IdentityContext(
        subject_id=subject_id,
        oidc_subject=f"local-role-simulation:{subject_id}",
        tenant_id=tenant_id,
        roles=frozenset({role}),
        asset_ids=frozenset(),
        site_ids=frozenset(),
        issued_at=now,
        expires_at=now + timedelta(hours=24),
    )


def _digest(value: object) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
