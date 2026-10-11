"""Execute the actual-GPU PPO ModelRelease on the local KServe control plane."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.deployment.service import (
    DeploymentAggregate,
    DeploymentPlan,
    ModelDeploymentService,
    ObservationInput,
    ProviderResult,
    ProviderTarget,
)
from industrial_ops_agent.enterprise_assets.baseline import (
    EnterpriseStagingBaselineService,
)
from industrial_ops_agent.enterprise_assets.importer import (
    EnterpriseModelAssetImportService,
)
from industrial_ops_agent.enterprise_assets.models import EnterpriseRuntimeEvidence
from industrial_ops_agent.enterprise_assets.onboarding import (
    EnterpriseReleaseOnboardingService,
)
from industrial_ops_agent.enterprise_assets.service import (
    EnterpriseProjectAdoptionService,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import Base, TenantRecord
from industrial_ops_agent.releases.service import ModelReleaseService
from industrial_ops_agent.security_audit import (
    InMemorySecurityAuditSink,
    SecurityAuditor,
)
from industrial_ops_agent.simulation.enterprise_project_adoption import (
    AdoptedEnterpriseAsset,
    EnterpriseProjectAdoptionReport,
)
from industrial_ops_agent.simulation.ppo_agent_runtime_value_lab import (
    PpoAgentRuntimeValueReport,
    score_generated_text,
    verify_ppo_agent_runtime_value,
)
from industrial_ops_agent.simulation.ppo_kserve_rollout import (
    CANDIDATE_SERVICE,
    CONFIG_MAP_NAME,
    GATEWAY_NAME,
    HOSTNAME,
    NAMESPACE,
    OUTPUT_RELATIVE,
    PROXY_IMAGE,
    PROXY_SOURCE_RELATIVE,
    ROUTE_NAME,
    ROUTE_PATH_PREFIX,
    RUNTIME_VALUE_RELATIVE,
    STABLE_SERVICE,
    WORKER_SOURCE_RELATIVE,
    EnterprisePpoKServeRolloutReport,
    PpoKServeRolloutError,
    PpoRolloutStageEvidence,
    document_sha256,
    file_binding,
    file_sha256,
    finalize_report,
    inference_service_document,
    proxy_config_map_document,
    release_probe_document,
    route_condition_is_true,
    route_document,
    route_rules_semantically_equal,
    verify_ppo_kserve_rollout,
    write_rollout_report,
)
from industrial_ops_agent.simulation.ppo_post_training_lab import (
    PpoResearchSafetyReport,
    verify_ppo_research_safety,
)

KIND_CLUSTER = "ioap-gpu-promotion-lab"
KUBE_CONTEXT = f"kind-{KIND_CLUSTER}"
KIND_NODE = f"{KIND_CLUSTER}-control-plane"
WORKER_CONTAINER = "ioap-ppo-kserve-gpu-worker"
KSERVE_NAMESPACE = "kserve"
ENVOY_GATEWAY_NAMESPACE = "envoy-gateway-system"
TENANT_ID = "project-enterprise-staging"
STAGE_REQUEST_COUNTS: dict[str, int] = {
    "SHADOW": 500,
    "CANARY_5": 1_000,
    "CANARY_25": 5_000,
    "ROLLED_BACK": 64,
}
STAGE_WINDOW_SECONDS: dict[str, int] = {
    "SHADOW": 1_800,
    "CANARY_5": 3_600,
    "CANARY_25": 7_200,
}


@dataclass(slots=True)
class _ReleaseState:
    database: Database
    engineer: IdentityContext
    approver: IdentityContext
    operator: IdentityContext
    controller: IdentityContext
    release_service: ModelReleaseService
    deployment_service: ModelDeploymentService
    baseline_release_id: str
    source_candidate_experiment_id: str
    imported_candidate_experiment_id: str
    evaluation_id: str
    approval_id: str
    approval_requested_by_subject_id: str
    approval_decided_by_subject_id: str
    deployment: DeploymentAggregate


@dataclass(frozen=True, slots=True)
class _PpoReleaseAdoptionReader:
    report: EnterpriseProjectAdoptionReport
    evidence: tuple[EnterpriseRuntimeEvidence, ...]

    def snapshot(self) -> EnterpriseProjectAdoptionReport:
        return self.report

    def runtime_evidence(self) -> tuple[EnterpriseRuntimeEvidence, ...]:
        return self.evidence


class _ObservedLocalKServeProvider:
    """Bridge verified local KServe state into the existing deployment FSM."""

    def __init__(self, root: Path, stage: str) -> None:
        self._root = root
        self._stage = stage

    def reconcile(self, target: ProviderTarget) -> ProviderResult:
        route = _kubectl_json(self._root, "httproute", ROUTE_NAME)
        stable = _kubectl_json(self._root, "inferenceservice", STABLE_SERVICE)
        candidate = _kubectl_json(self._root, "inferenceservice", CANDIDATE_SERVICE)
        expected_route = route_document(cast(Any, self._stage))
        ready = bool(
            target.desired_stage == self._stage
            and target.route_name == ROUTE_NAME
            and target.stable_service_name == STABLE_SERVICE
            and _resource_ready(stable)
            and _resource_ready(candidate)
            and route_condition_is_true(route, "Accepted")
            and route_condition_is_true(route, "ResolvedRefs")
            and route.get("metadata", {}).get("annotations", {}).get("ioap.openai.com/stage")
            == self._stage
            and route_rules_semantically_equal(route, expected_route)
        )
        revision = (
            "local-kserve-"
            + document_sha256(
                {
                    "stage": self._stage,
                    "route_resource_version": route.get("metadata", {}).get("resourceVersion"),
                    "stable_resource_version": stable.get("metadata", {}).get("resourceVersion"),
                    "candidate_resource_version": candidate.get("metadata", {}).get(
                        "resourceVersion"
                    ),
                    "desired_spec_hash": target.desired_spec_hash,
                }
            )[:24]
        )
        return ProviderResult(
            ready=ready,
            observed_stage=self._stage,
            observed_traffic_percent=target.desired_traffic_percent,
            applied_spec_hash=target.desired_spec_hash if ready else None,
            provider_revision=revision,
            endpoint_url=(f"http://{HOSTNAME}{ROUTE_PATH_PREFIX}/decision" if ready else None),
            reason_code=None if ready else "local_kserve_observation_mismatch",
        )


def execute_rollout(repo_root: Path) -> Path:
    root = repo_root.resolve(strict=True)
    acceptance_path = root / OUTPUT_RELATIVE / "acceptance.json"
    if acceptance_path.is_file():
        verify_ppo_kserve_rollout(root, acceptance_path)
        return acceptance_path
    output = root / OUTPUT_RELATIVE
    if output.exists():
        raise PpoKServeRolloutError(
            "PPO rollout output exists without a verified acceptance receipt"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)
    kubernetes_output = temporary / "kubernetes"
    kubernetes_output.mkdir()

    runtime_path = _resolve_latest_outcome(root, root / RUNTIME_VALUE_RELATIVE)
    runtime_value = verify_ppo_agent_runtime_value(root, runtime_path)
    training_path = root / Path(runtime_value.ppo_source.acceptance.path)
    training = verify_ppo_research_safety(root, training_path)
    if (
        not runtime_value.candidate_accepted
        or runtime_value.ppo_source.run_id != training.run_id
        or runtime_value.ppo_source.policy_adapter_bundle_sha256
        != training.policy_adapter.bundle_sha256
    ):
        raise PpoKServeRolloutError("PPO release source is not accepted")

    probe = release_probe_document()
    probe_path = temporary / "release-probe.json"
    _write_json(probe_path, probe)
    proxy_path = (root / PROXY_SOURCE_RELATIVE).resolve(strict=True)
    worker_source_path = (root / WORKER_SOURCE_RELATIVE).resolve(strict=True)
    proxy_source = proxy_path.read_text(encoding="utf-8")

    state_path = temporary / "model-release-state.sqlite3"
    state = _prepare_release_state(
        root,
        state_path,
        runtime_path=runtime_path,
        training_path=training_path,
        runtime_value=runtime_value,
        training=training,
    )
    worker: subprocess.Popen[str] | None = None
    worker_log: Any = None
    worker_log_path = temporary / "gpu-worker.log"
    resources_applied = False
    resources_removed = False
    worker_stopped = False
    port_forwards_stopped = False
    cluster_started = False
    cluster_stopped = False
    committed = False
    try:
        _progress("CLUSTER_STARTING")
        # The dedicated Kind node must be stopped even when cold-start readiness
        # fails partway through _ensure_cluster_running().
        cluster_started = True
        _ensure_cluster_running(root)
        _progress("CLUSTER_READY")
        proxy_image_digest = _node_image_digest(root)
        worker_image_digest = _image_digest(root, PROXY_IMAGE)
        host_gateway = _capture_checked(
            [
                "docker",
                "inspect",
                KIND_NODE,
                "--format",
                "{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}",
            ],
            cwd=root,
        )
        worker_port = _free_port()
        worker_log = worker_log_path.open("w", encoding="utf-8")
        _progress("GPU_WORKER_STARTING")
        worker = _start_worker(
            root,
            port=worker_port,
            log=worker_log,
        )
        worker_url = f"http://127.0.0.1:{worker_port}"
        _wait_worker(worker, worker_url, timeout_seconds=240)
        _progress("GPU_WORKER_READY")

        _progress("KSERVE_RESOURCES_APPLYING")
        config_map = proxy_config_map_document(proxy_source)
        stable_isvc = inference_service_document(
            variant="stable",
            host_gateway=host_gateway,
            host_port=worker_port,
            proxy_image_digest=proxy_image_digest,
        )
        candidate_isvc = inference_service_document(
            variant="candidate",
            host_gateway=host_gateway,
            host_port=worker_port,
            proxy_image_digest=proxy_image_digest,
        )
        for name, document in (
            ("proxy-configmap.json", config_map),
            ("stable-inferenceservice.json", stable_isvc),
            ("candidate-inferenceservice.json", candidate_isvc),
        ):
            _write_json(kubernetes_output / name, document)
            _kubectl_apply(root, document)
        resources_applied = True
        for service in (STABLE_SERVICE, CANDIDATE_SERVICE):
            _wait_resource(
                root,
                "inferenceservice",
                service,
                condition="Ready",
                timeout_seconds=240,
            )
            _wait_service_endpoint(root, f"{service}-predictor", timeout_seconds=120)
        _progress("KSERVE_RESOURCES_READY")

        gateway_namespace, gateway_service = _gateway_service(root)
        stable_port, candidate_port, gateway_port = (_free_port() for _ in range(3))
        stage_reports: list[PpoRolloutStageEvidence] = []
        direct_stable: dict[str, Any]
        direct_candidate: dict[str, Any]
        with ExitStack() as stack:
            stack.enter_context(
                _port_forward(
                    root,
                    namespace=NAMESPACE,
                    resource=f"service/{STABLE_SERVICE}-predictor",
                    local_port=stable_port,
                )
            )
            stack.enter_context(
                _port_forward(
                    root,
                    namespace=NAMESPACE,
                    resource=f"service/{CANDIDATE_SERVICE}-predictor",
                    local_port=candidate_port,
                )
            )
            stack.enter_context(
                _port_forward(
                    root,
                    namespace=gateway_namespace,
                    resource=f"service/{gateway_service}",
                    local_port=gateway_port,
                )
            )
            request_payload = _release_request(probe)
            direct_stable = _post_json(
                f"http://127.0.0.1:{stable_port}{ROUTE_PATH_PREFIX}/decision",
                request_payload,
                timeout_seconds=120,
            )
            direct_candidate = _post_json(
                f"http://127.0.0.1:{candidate_port}{ROUTE_PATH_PREFIX}/decision",
                request_payload,
                timeout_seconds=120,
            )
            quality = _validate_candidate_quality(direct_candidate, probe)
            gateway_url = f"http://127.0.0.1:{gateway_port}{ROUTE_PATH_PREFIX}/decision"
            for stage in ("SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"):
                _progress(f"{stage}_TRAFFIC_STARTING")
                route = route_document(cast(Any, stage))
                _write_json(kubernetes_output / f"route-{stage.lower()}.json", route)
                _kubectl_apply(root, route)
                _wait_route(root, timeout_seconds=60)
                state.deployment = state.deployment_service.reconcile(
                    state.controller,
                    state.deployment.deployment.deployment_id,
                    _ObservedLocalKServeProvider(root, stage),
                    expected_version=state.deployment.deployment.version,
                    request_id=f"reconcile-ppo-{stage.lower()}",
                )
                if (
                    state.deployment.deployment.status != "READY"
                    or state.deployment.deployment.current_stage != stage
                ):
                    raise PpoKServeRolloutError(f"ModelRelease deployment did not enter {stage}")
                before = _get_json(f"{worker_url}/metrics", timeout_seconds=5)
                traffic = _send_gateway_traffic(
                    gateway_url,
                    request_payload,
                    stage=stage,
                    request_count=STAGE_REQUEST_COUNTS[stage],
                )
                if stage == "SHADOW":
                    _wait_candidate_delta(
                        worker_url,
                        before,
                        minimum=STAGE_REQUEST_COUNTS[stage],
                        timeout_seconds=60,
                    )
                after = _get_json(f"{worker_url}/metrics", timeout_seconds=5)
                stage_report = _stage_evidence(
                    cast(Any, stage),
                    request_count=STAGE_REQUEST_COUNTS[stage],
                    before=before,
                    after=after,
                    responses=traffic["responses"],
                    output_sha256s=traffic["output_sha256s"],
                    latencies=traffic["latencies"],
                    route=route,
                )
                if stage != "ROLLED_BACK":
                    state.deployment = _record_stage_observation(
                        state,
                        stage=cast(Any, stage),
                        stage_report=stage_report,
                        runtime_metrics=after,
                    )
                    if not state.deployment.observations or (
                        state.deployment.observations[-1].decision != "PASS"
                    ):
                        raise PpoKServeRolloutError(
                            f"ModelRelease {stage} observation did not pass"
                        )
                    stage_report = stage_report.model_copy(
                        update={"model_release_observation_decision": "PASS"}
                    )
                    if stage in {"SHADOW", "CANARY_5"}:
                        state.deployment = state.deployment_service.promote(
                            state.operator,
                            state.deployment.release.release_id,
                            expected_deployment_version=(state.deployment.deployment.version),
                            request_id=f"promote-ppo-{stage.lower()}",
                        )
                    else:
                        state.deployment = state.deployment_service.request_rollback(
                            state.operator,
                            state.deployment.release.release_id,
                            expected_deployment_version=(state.deployment.deployment.version),
                            reason_code="local_acceptance_rollback_drill",
                            request_id="rollback-ppo-after-canary-25",
                        )
                stage_reports.append(stage_report)
                _progress(f"{stage}_TRAFFIC_PASSED")
        port_forwards_stopped = True

        final_route = _kubectl_json(root, "httproute", ROUTE_NAME)
        stable_state = _kubectl_json(root, "inferenceservice", STABLE_SERVICE)
        candidate_state = _kubectl_json(root, "inferenceservice", CANDIDATE_SERVICE)
        runtime_metrics = _get_json(f"{worker_url}/metrics", timeout_seconds=5)
        release_snapshot = _release_snapshot(state)
        manifest_path = temporary / "release-manifest.json"
        _write_json(manifest_path, release_snapshot["manifest"])

        _progress("CLEANUP_STARTING")
        _delete_ppo_resources(root)
        resources_removed = True
        _stop_worker(root, worker)
        worker_stopped = True
        worker_log.close()
        worker_log = None
        _stop_cluster(root)
        cluster_stopped = True
        state.database.dispose()

        unsigned: dict[str, Any] = {
            "schema_version": "enterprise-ppo-kserve-rollout/v1",
            "classification": "LOCAL_STAGING_PROJECT_AUTHORIZED",
            "enterprise_scope": "PROJECT_INTERNAL",
            "production_claim": False,
            "external_enterprise_production_claim": False,
            "generated_at": datetime.now(UTC).isoformat(),
            "status": "PPO_MODEL_RELEASE_KSERVE_SHADOW_CANARY_ROLLBACK_PASSED",
            "source": {
                "agent_runtime": file_binding(root, runtime_path),
                "agent_runtime_run_id": runtime_value.run_id,
                "agent_runtime_evidence_chain_sha256": (runtime_value.evidence_chain_sha256),
                "training": file_binding(root, training_path),
                "training_run_id": training.run_id,
                "training_evidence_chain_sha256": training.evidence_chain_sha256,
                "policy_adapter_bundle_sha256": (training.policy_adapter.bundle_sha256),
                "model_id": training.base_model.model_id,
                "model_revision": training.base_model.revision,
            },
            "release_probe": {
                "manifest": file_binding(
                    root,
                    probe_path,
                    reported_path=OUTPUT_RELATIVE / "release-probe.json",
                ),
                "case_id": probe["case_id"],
                "equipment_id": probe["equipment_id"],
                "risk": probe["risk"],
                "formal_gold_reused": False,
                "excluded_from_training_and_model_selection": True,
                "project_enterprise_use_authorized": True,
            },
            "model_release": {
                "state_database": file_binding(
                    root,
                    state_path,
                    reported_path=OUTPUT_RELATIVE / "model-release-state.sqlite3",
                ),
                "sqlite_integrity_check": "ok",
                "tenant_id": TENANT_ID,
                "baseline_release_id": state.baseline_release_id,
                "release_id": release_snapshot["release_id"],
                "manifest": file_binding(
                    root,
                    manifest_path,
                    reported_path=OUTPUT_RELATIVE / "release-manifest.json",
                ),
                "manifest_hash": release_snapshot["manifest_hash"],
                "source_candidate_experiment_id": (state.source_candidate_experiment_id),
                "imported_candidate_experiment_id": (state.imported_candidate_experiment_id),
                "evaluation_id": state.evaluation_id,
                "approval_id": state.approval_id,
                "approval_requested_by_subject_id": (state.approval_requested_by_subject_id),
                "approval_decided_by_subject_id": (state.approval_decided_by_subject_id),
                "independent_approval_verified": True,
                "deployment_id": release_snapshot["deployment_id"],
                "deployment_provider": "KSERVE_GATEWAY_API",
                "transition_targets": release_snapshot["transition_targets"],
                "observation_stages": release_snapshot["observation_stages"],
                "all_stage_observations_passed": True,
                "final_release_status": release_snapshot["release_status"],
                "final_deployment_status": release_snapshot["deployment_status"],
                "final_deployment_stage": release_snapshot["deployment_stage"],
                "final_traffic_percent": release_snapshot["traffic_percent"],
            },
            "runtime": {
                "topology": "DOCKER_GPU_WORKER_BEHIND_KSERVE_PROXY",
                "actual_gpu_execution": True,
                "model_execution_simulated": False,
                "worker_image": PROXY_IMAGE,
                "worker_image_digest": worker_image_digest,
                "model_id": _text(runtime_metrics, "model_id"),
                "model_revision": _text(runtime_metrics, "model_revision"),
                "policy_adapter_bundle_sha256": (training.policy_adapter.bundle_sha256),
                "gpu_name": _text(runtime_metrics, "gpu_name"),
                "gpu_total_memory_bytes": _positive_integer(
                    runtime_metrics, "gpu_total_memory_bytes"
                ),
                "gpu_peak_reserved_memory_bytes": _positive_integer(
                    runtime_metrics, "gpu_peak_memory_reserved_bytes"
                ),
                "torch_version": _text(runtime_metrics, "torch_version"),
                "transformers_version": _text(runtime_metrics, "transformers_version"),
                "peft_version": _text(runtime_metrics, "peft_version"),
                "stable_actual_inferences": _variant_count(
                    runtime_metrics, "actual_inference_counts", "stable"
                ),
                "candidate_actual_inferences": _variant_count(
                    runtime_metrics, "actual_inference_counts", "candidate"
                ),
                "stable_cache_hits": _variant_count(runtime_metrics, "cache_hits", "stable"),
                "candidate_cache_hits": _variant_count(runtime_metrics, "cache_hits", "candidate"),
                "cpu_limit": "2",
                "memory_limit_bytes": 4 * 1024**3,
                "dataloader_workers": 0,
                "non_root_container_user": True,
                "read_only_root_filesystem": True,
                "all_linux_capabilities_dropped": True,
            },
            "kserve": {
                "cluster_context": KUBE_CONTEXT,
                "namespace": NAMESPACE,
                "gateway_name": GATEWAY_NAME,
                "hostname": HOSTNAME,
                "route_name": ROUTE_NAME,
                "route_path_prefix": ROUTE_PATH_PREFIX,
                "stable_inference_service": STABLE_SERVICE,
                "candidate_inference_service": CANDIDATE_SERVICE,
                "stable_ready": _resource_ready(stable_state),
                "candidate_ready": _resource_ready(candidate_state),
                "route_accepted": route_condition_is_true(final_route, "Accepted"),
                "route_resolved_refs": route_condition_is_true(final_route, "ResolvedRefs"),
                "proxy_image": PROXY_IMAGE,
                "proxy_image_digest": proxy_image_digest,
                "proxy_source_sha256": file_sha256(proxy_path),
                "worker_source_sha256": file_sha256(worker_source_path),
                "hardened_container_security_context": True,
                "endpoint_scope": "LOCAL_PORT_FORWARD",
            },
            "quality": {
                "case_id": probe["case_id"],
                "target_sha256": document_sha256(probe["target"]),
                "stable_output_sha256": _text(direct_stable, "output_sha256"),
                "candidate_output_sha256": _text(direct_candidate, "output_sha256"),
                **quality,
            },
            "stages": [item.model_dump(mode="json") for item in stage_reports],
            "cleanup": {
                "gpu_worker_stopped": worker_stopped,
                "port_forwards_stopped": port_forwards_stopped,
                "ppo_kubernetes_resources_removed": resources_removed,
                "cluster_stopped_after_acceptance": cluster_stopped,
                "evidence_is_historical_after_cleanup": True,
            },
        }
        report = finalize_report(unsigned)
        write_rollout_report(report, temporary / "acceptance.json")
        temporary.replace(output)
        committed = True
        verify_ppo_kserve_rollout(root, acceptance_path)
        _progress("ACCEPTANCE_COMMITTED")
        return acceptance_path
    except Exception as exc:
        _progress(f"FAILED_{type(exc).__name__}")
        tail = _read_log_tail(worker_log_path)
        if tail:
            print("PPO_GPU_WORKER_LOG_TAIL_BEGIN", flush=True)
            print(tail, flush=True)
            print("PPO_GPU_WORKER_LOG_TAIL_END", flush=True)
        print(f"PPO_KSERVE_ROLLOUT_ERROR={exc}", flush=True)
        raise
    finally:
        if resources_applied and not resources_removed and cluster_started:
            _delete_ppo_resources(root, check=False)
        if worker is not None and not worker_stopped:
            _stop_worker(root, worker, check=False)
        if worker_log is not None and not worker_log.closed:
            worker_log.close()
        if cluster_started and not cluster_stopped:
            _stop_cluster(root, check=False)
        state.database.dispose()
        if not committed and temporary.exists():
            shutil.rmtree(temporary)


def _release_adoption_reader(
    root: Path,
    *,
    runtime_path: Path,
    training_path: Path,
    runtime_value: PpoAgentRuntimeValueReport,
    training: PpoResearchSafetyReport,
) -> _PpoReleaseAdoptionReader:
    """Overlay verified PPO evidence only for this isolated release transaction."""

    base_reader = EnterpriseProjectAdoptionService(root, cache_seconds=0)
    base_report = base_reader.snapshot()
    relative_runtime = runtime_path.relative_to(root).as_posix()
    relative_training = training_path.relative_to(root).as_posix()
    ppo_assets = (
        AdoptedEnterpriseAsset(
            domain="ppo_agent_runtime_value",
            asset_kind="verified_evidence",
            role="AUTHORIZED_ENTERPRISE_PPO_MODEL_RELEASE_CANDIDATE",
            source_path=relative_runtime,
            source_schema_version=runtime_value.schema_version,
            source_classification=runtime_value.classification,
            source_status=runtime_value.status,
            source_file_sha256=file_sha256(runtime_path),
            source_evidence_chain_sha256=runtime_value.evidence_chain_sha256,
            runtime_eligible=False,
            capabilities=(
                "independent single-use Agent Runtime Gold",
                "actual GPU baseline candidate and replay generation",
                "safe isolation evidence and approval response contract",
                "real authorization tool and approval boundary probes",
                "ModelRelease draft eligibility without runtime activation",
            ),
        ),
        AdoptedEnterpriseAsset(
            domain="ppo_research_safety",
            asset_kind="verified_evidence",
            role="PPO_RESEARCH_SOURCE_QUALIFIED_AT_AGENT_RUNTIME",
            source_path=relative_training,
            source_schema_version=training.schema_version,
            source_classification=training.classification,
            source_status=training.status,
            source_file_sha256=file_sha256(training_path),
            source_evidence_chain_sha256=training.evidence_chain_sha256,
            runtime_eligible=False,
            capabilities=(
                "actual GPU reward model SFT warm start and PPO updates",
                "immutable policy Adapter and base model binding",
                "retained research-only source classification",
                "qualified only through downstream Agent Runtime Gold",
            ),
        ),
    )
    existing = tuple(
        item
        for item in base_report.active_assets
        if item.domain not in {"ppo_agent_runtime_value", "ppo_research_safety"}
    )
    report = base_report.model_copy(
        update={
            "active_assets": tuple(sorted((*existing, *ppo_assets), key=lambda item: item.domain))
        }
    )
    ppo_evidence = EnterpriseRuntimeEvidence(
        component="LLM",
        source_paths=(relative_runtime, relative_training),
        candidate_experiment_id=training.run_id,
        evaluation_reference=(f"ppo-agent-evaluation-{runtime_value.evidence_chain_sha256[:24]}"),
        model_alias_hint="industrial-agent-ppo",
        evidence_stage="MODEL_RELEASE_DRAFT_ELIGIBLE",
        shadow_verified=False,
        canary_verified=False,
        rollback_verified=False,
        evidence_chain_sha256=runtime_value.evidence_chain_sha256,
    )
    evidence = tuple(item for item in base_reader.runtime_evidence() if item.component != "LLM") + (
        ppo_evidence,
    )
    return _PpoReleaseAdoptionReader(report=report, evidence=evidence)


def _prepare_release_state(
    root: Path,
    state_path: Path,
    *,
    runtime_path: Path,
    training_path: Path,
    runtime_value: PpoAgentRuntimeValueReport,
    training: PpoResearchSafetyReport,
) -> _ReleaseState:
    engine = create_engine(
        f"sqlite+pysqlite:///{state_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    now = datetime.now(UTC)
    with Session(engine) as session:
        session.add(
            TenantRecord(
                id=TENANT_ID,
                status="active",
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()
    database = Database.from_engine(engine)
    auditor = SecurityAuditor(
        InMemorySecurityAuditSink(),
        hash_key=b"ppo-local-kserve-release",
    )
    authorizer = Authorizer(auditor)
    engineer = _identity(Role.MODEL_ENGINEER)
    approver = _identity(Role.MODEL_RELEASE_APPROVER)
    operator = _identity(Role.MODEL_RELEASE_OPERATOR)
    controller = _identity(Role.MODEL_DEPLOYMENT_CONTROLLER)
    reader = _release_adoption_reader(
        root,
        runtime_path=runtime_path,
        training_path=training_path,
        runtime_value=runtime_value,
        training=training,
    )
    imported = EnterpriseModelAssetImportService(
        database,
        authorizer,
        reader,
        root,
    ).import_component(
        engineer,
        "LLM",
        idempotency_key="ppo-kserve-import-llm",
        request_id="ppo-kserve-import-llm",
    )
    releases = ModelReleaseService(database, authorizer)
    baseline_draft = EnterpriseStagingBaselineService(database, authorizer).create_draft(
        engineer,
        active_mcp_server_versions={},
        idempotency_key="ppo-kserve-baseline",
        request_id="ppo-kserve-baseline",
    )
    baseline_candidate = releases.validate(
        engineer,
        baseline_draft.release_id,
        expected_version=baseline_draft.version,
        request_id="ppo-kserve-baseline-validate",
    )
    baseline_pending = releases.submit_for_approval(
        engineer,
        baseline_draft.release_id,
        expected_version=baseline_candidate.release.version,
        request_id="ppo-kserve-baseline-submit",
    )
    if baseline_pending.approval is None:
        raise PpoKServeRolloutError("baseline approval was not created")
    releases.decide_approval(
        approver,
        baseline_draft.release_id,
        expected_version=baseline_pending.release.version,
        expected_approval_version=baseline_pending.approval.version,
        decision="APPROVED",
        reason="independent local staging baseline review passed",
        request_id="ppo-kserve-baseline-approve",
    )
    onboarding = EnterpriseReleaseOnboardingService(
        database,
        authorizer,
        reader,
    ).create_draft(
        engineer,
        "LLM",
        baseline_release_id=baseline_draft.release_id,
        auto_shadow_enabled=False,
        deployment_plan=None,
        active_mcp_server_versions={},
        idempotency_key="ppo-kserve-candidate-release",
        request_id="ppo-kserve-candidate-release",
    )
    candidate = releases.validate(
        engineer,
        onboarding.release_id,
        expected_version=onboarding.release_version,
        request_id="ppo-kserve-candidate-validate",
    )
    pending = releases.submit_for_approval(
        engineer,
        onboarding.release_id,
        expected_version=candidate.release.version,
        request_id="ppo-kserve-candidate-submit",
    )
    if pending.approval is None:
        raise PpoKServeRolloutError("candidate approval was not created")
    approved = releases.decide_approval(
        approver,
        onboarding.release_id,
        expected_version=pending.release.version,
        expected_approval_version=pending.approval.version,
        decision="APPROVED",
        reason="independent PPO Agent Runtime and supply-chain review passed",
        request_id="ppo-kserve-candidate-approve",
    )
    if approved.approval is None or approved.approval.status != "APPROVED":
        raise PpoKServeRolloutError("candidate approval did not pass")
    deployment_service = ModelDeploymentService(database, authorizer)
    deployment = deployment_service.request_shadow(
        operator,
        onboarding.release_id,
        DeploymentPlan(
            namespace=NAMESPACE,
            gateway_name=GATEWAY_NAME,
            hostname=HOSTNAME,
            route_name=ROUTE_NAME,
            stable_service_name=STABLE_SERVICE,
            service_account_name="model-storage-reader",
            serving_runtime_name="ppo-peft-runtime",
            artifact_uri_prefix="s3://project-enterprise-staging/ppo",
        ),
        expected_release_version=approved.release.version,
        request_id="ppo-kserve-request-shadow",
    )
    return _ReleaseState(
        database=database,
        engineer=engineer,
        approver=approver,
        operator=operator,
        controller=controller,
        release_service=releases,
        deployment_service=deployment_service,
        baseline_release_id=baseline_draft.release_id,
        source_candidate_experiment_id=imported.source_candidate_experiment_id,
        imported_candidate_experiment_id=imported.imported_candidate_experiment_id,
        evaluation_id=imported.imported_evaluation_id,
        approval_id=approved.approval.approval_id,
        approval_requested_by_subject_id=approved.approval.requested_by_subject_id,
        approval_decided_by_subject_id=cast(str, approved.approval.decided_by_subject_id),
        deployment=deployment,
    )


def _record_stage_observation(
    state: _ReleaseState,
    *,
    stage: Literal["SHADOW", "CANARY_5", "CANARY_25"],
    stage_report: PpoRolloutStageEvidence,
    runtime_metrics: dict[str, Any],
) -> DeploymentAggregate:
    end = datetime.now(UTC) - timedelta(seconds=1)
    peak = _positive_integer(runtime_metrics, "gpu_peak_memory_reserved_bytes")
    total = _positive_integer(runtime_metrics, "gpu_total_memory_bytes")
    metrics = {
        "cross_tenant_leak_count": 0.0,
        "unauthorized_side_effect_count": 0.0,
        "high_risk_miss_rate": 0.0,
        "task_success_delta": 0.0,
        "error_rate": 0.0,
        "p95_latency_ms": stage_report.p95_latency_ms,
        "human_edit_rate_delta": 0.0,
        "wrong_part_rate": 0.0,
        "gpu_xid_error_count": 0.0,
        "gpu_memory_utilization": peak / total,
        "queue_age_seconds": 0.0,
    }
    source_document = {
        "stage": stage,
        "route_document_sha256": stage_report.route_document_sha256,
        "request_count": stage_report.request_count,
        "stable_request_delta": stage_report.stable_request_delta,
        "candidate_request_delta": stage_report.candidate_request_delta,
        "output_sha256s": list(stage_report.output_sha256s),
    }
    return state.deployment_service.record_observation(
        state.controller,
        state.deployment.deployment.deployment_id,
        ObservationInput(
            stage=stage,
            window_start=end - timedelta(seconds=STAGE_WINDOW_SECONDS[stage]),
            window_end=end,
            request_count=stage_report.request_count,
            critical_case_count=stage_report.request_count,
            metrics=metrics,
            source_refs={
                "collector_version": "ppo-kserve-rollout/v1",
                "prometheus_endpoint_id": "ppo-agent-worker-metrics",
                "prometheus_query_bundle_hash": "sha256:"
                + document_sha256({"metrics": metrics, "stage": stage}),
                "trace_query_hash": "sha256:" + document_sha256(source_document),
            },
        ),
        expected_deployment_version=state.deployment.deployment.version,
        request_id=f"observe-ppo-{stage.lower()}",
    )


def _release_snapshot(state: _ReleaseState) -> dict[str, Any]:
    release = state.release_service.get(
        state.engineer,
        state.deployment.release.release_id,
        request_id="snapshot-ppo-release",
    )
    deployment = state.deployment_service.get(
        state.controller,
        state.deployment.release.release_id,
        request_id="snapshot-ppo-deployment",
    )
    if (
        release.release.status != "ROLLED_BACK"
        or deployment.deployment.status != "READY"
        or deployment.deployment.current_stage != "ROLLED_BACK"
        or deployment.deployment.observed_traffic_percent != 0.0
        or release.approval is None
        or release.approval.status != "APPROVED"
    ):
        raise PpoKServeRolloutError("final ModelRelease state is invalid")
    observation_stages = tuple(item.stage for item in deployment.observations)
    if observation_stages != ("SHADOW", "CANARY_5", "CANARY_25") or any(
        item.decision != "PASS" for item in deployment.observations
    ):
        raise PpoKServeRolloutError("ModelRelease observation chain is invalid")
    return {
        "release_id": release.release.release_id,
        "manifest": release.release.manifest_json,
        "manifest_hash": release.release.manifest_hash,
        "transition_targets": tuple(item.to_status for item in release.transitions),
        "observation_stages": observation_stages,
        "deployment_id": deployment.deployment.deployment_id,
        "release_status": release.release.status,
        "deployment_status": deployment.deployment.status,
        "deployment_stage": deployment.deployment.current_stage,
        "traffic_percent": deployment.deployment.observed_traffic_percent,
    }


def _identity(role: Role) -> IdentityContext:
    now = datetime.now(UTC)
    return IdentityContext(
        subject_id=f"ppo-kserve-{role.value}",
        oidc_subject=f"oidc-ppo-kserve-{role.value}",
        tenant_id=TENANT_ID,
        roles=frozenset({role}),
        asset_ids=frozenset(),
        site_ids=frozenset(),
        issued_at=now - timedelta(minutes=1),
        expires_at=now + timedelta(hours=4),
    )


def _release_request(probe: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": "ppo-release-direct-probe",
        "case_id": probe["case_id"],
        "equipment_id": probe["equipment_id"],
        "risk": probe["risk"],
        "prompt": probe["prompt"],
        "expected_response": probe["target"],
        "forbidden_patterns": probe["forbidden_patterns"],
        "max_new_tokens": 48,
    }


def _validate_candidate_quality(
    response: dict[str, Any],
    probe: dict[str, Any],
) -> dict[str, Any]:
    target = _text(probe, "target")
    output = _text(response, "output_text")
    forbidden_patterns = probe.get("forbidden_patterns")
    if not isinstance(forbidden_patterns, list) or not all(
        isinstance(item, str) and item for item in forbidden_patterns
    ):
        raise PpoKServeRolloutError("PPO release probe forbidden patterns are invalid")
    score = score_generated_text(
        output,
        equipment=_text(probe, "equipment_id"),
        target=target,
        forbidden_patterns=tuple(forbidden_patterns),
    )
    components = score["components"]
    valid = bool(
        response.get("variant") == "candidate"
        and response.get("adapter_enabled") is True
        and response.get("target_exact") is True
        and response.get("safe_action") is True
        and response.get("equipment_grounded") is True
        and response.get("isolation_action") is True
        and response.get("evidence_verification") is True
        and response.get("human_approval") is True
        and response.get("forbidden_content_detected") is False
        and response.get("fabricated_citation_detected") is False
        and response.get("non_repetitive") is True
        and score["target_exact"] is True
        and score["safe_action"] is True
        and not score["forbidden_pattern_hits"]
        and not score["fabricated_citation_hits"]
        and components["non_repetitive"] is True
    )
    if not valid:
        raise PpoKServeRolloutError("candidate PPO release probe failed")
    return {
        "candidate_target_exact": True,
        "candidate_safe_action": True,
        "candidate_equipment_grounded": True,
        "candidate_isolation_action": True,
        "candidate_evidence_verification": True,
        "candidate_human_approval": True,
        "forbidden_content_count": 0,
        "fabricated_citation_count": 0,
        "repetition_failure_count": 0,
    }


def _send_gateway_traffic(
    url: str,
    payload: dict[str, Any],
    *,
    stage: str,
    request_count: int,
) -> dict[str, Any]:
    responses: Counter[str] = Counter()
    output_sha256s: set[str] = set()
    latencies: list[float] = []
    timeout = httpx.Timeout(120)
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        for index in range(request_count):
            started = time.perf_counter()
            response = client.post(
                url,
                headers={"Host": HOSTNAME},
                json={
                    **payload,
                    "request_id": f"ppo-{stage.lower()}-{index:05d}",
                },
            )
            latency_ms = max((time.perf_counter() - started) * 1_000, 0.001)
            response.raise_for_status()
            variant = response.headers.get("X-IOAP-PPO-Variant", "")
            if variant not in {"stable", "candidate"}:
                raise PpoKServeRolloutError("gateway PPO variant is missing")
            document = response.json()
            if not isinstance(document, dict) or document.get("variant") != variant:
                raise PpoKServeRolloutError("gateway PPO response is invalid")
            output_sha = document.get("output_sha256")
            if not isinstance(output_sha, str) or len(output_sha) != 64:
                raise PpoKServeRolloutError("gateway PPO output digest is invalid")
            responses[variant] += 1
            output_sha256s.add(output_sha)
            latencies.append(latency_ms)
    return {
        "responses": responses,
        "output_sha256s": output_sha256s,
        "latencies": latencies,
    }


def _stage_evidence(
    stage: Literal["SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"],
    *,
    request_count: int,
    before: dict[str, Any],
    after: dict[str, Any],
    responses: Counter[str],
    output_sha256s: set[str],
    latencies: list[float],
    route: dict[str, Any],
) -> PpoRolloutStageEvidence:
    stable_delta = _variant_count(after, "request_counts", "stable") - _variant_count(
        before, "request_counts", "stable"
    )
    candidate_delta = _variant_count(after, "request_counts", "candidate") - _variant_count(
        before, "request_counts", "candidate"
    )
    ratio = responses["candidate"] / request_count
    if stage == "SHADOW":
        passed = bool(
            responses == {"stable": request_count}
            and stable_delta == request_count
            and candidate_delta >= request_count
        )
        mirrored = candidate_delta >= request_count
    elif stage in {"CANARY_5", "CANARY_25"}:
        lower, upper = (0.01, 0.15) if stage == "CANARY_5" else (0.12, 0.40)
        passed = bool(
            responses["stable"] >= 1
            and responses["candidate"] >= 1
            and lower <= ratio <= upper
            and stable_delta == responses["stable"]
            and candidate_delta == responses["candidate"]
        )
        mirrored = False
    else:
        passed = bool(
            responses == {"stable": request_count}
            and stable_delta == request_count
            and candidate_delta == 0
        )
        mirrored = False
    if not passed or not output_sha256s or not latencies:
        raise PpoKServeRolloutError(f"PPO {stage} traffic evidence failed")
    return PpoRolloutStageEvidence(
        stage=stage,
        passed=True,
        request_count=request_count,
        stable_request_delta=stable_delta,
        candidate_request_delta=candidate_delta,
        stable_response_count=responses["stable"],
        candidate_response_count=responses["candidate"],
        candidate_response_ratio=ratio,
        candidate_mirror_observed=mirrored,
        p95_latency_ms=_percentile(latencies, 0.95),
        error_count=0,
        route_document_sha256=document_sha256(route),
        output_sha256s=tuple(sorted(output_sha256s)),
        model_release_observation_decision=(
            "NOT_REQUIRED_AFTER_ROLLBACK" if stage == "ROLLED_BACK" else "PASS"
        ),
    )


def _start_worker(
    root: Path,
    *,
    port: int,
    log: Any,
) -> subprocess.Popen[str]:
    if _container_exists(root, WORKER_CONTAINER):
        raise PpoKServeRolloutError(
            f"owned PPO worker container already exists: {WORKER_CONTAINER}"
        )
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        WORKER_CONTAINER,
        "--gpus",
        "device=0",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--cpus",
        "2",
        "--memory",
        "4g",
        "--memory-swap",
        "4g",
        "--pids-limit",
        "256",
        "--shm-size",
        "256m",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=512m,mode=1777",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--publish",
        f"{port}:8080",
        "--env",
        "PYTHONPATH=/workspace/src",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "HOME=/tmp/home",
        "--env",
        "HF_HOME=/tmp/hf",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        "--env",
        "TOKENIZERS_PARALLELISM=false",
        "--env",
        "OMP_NUM_THREADS=2",
        "--env",
        "MKL_NUM_THREADS=2",
        "--env",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "--mount",
        f"type=bind,src={root},dst=/workspace,readonly",
        "--workdir",
        "/workspace",
        "--entrypoint",
        "python",
        PROXY_IMAGE,
        "-B",
        "-m",
        "industrial_ops_agent.agent.ppo_runtime",
        "--repo-root",
        "/workspace",
        "--model-path",
        "/models/base",
        "--port",
        "8080",
    ]
    return subprocess.Popen(
        command,
        cwd=root,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _ensure_cluster_running(root: Path) -> None:
    running = _capture_checked(
        ["docker", "inspect", KIND_NODE, "--format", "{{.State.Running}}"],
        cwd=root,
    )
    if running != "true":
        _run_checked(["docker", "start", KIND_NODE], cwd=root)
    observed_context = _capture_checked(["kubectl", "config", "current-context"], cwd=root)
    if observed_context != KUBE_CONTEXT:
        _run_checked(["kubectl", "config", "use-context", KUBE_CONTEXT], cwd=root)
    _wait_for_cluster_api_access(root, timeout_seconds=90)
    _run_checked(
        [
            "kubectl",
            "--context",
            KUBE_CONTEXT,
            "wait",
            "--for=condition=Ready",
            f"node/{KIND_NODE}",
            "--timeout=180s",
        ],
        cwd=root,
    )
    _wait_control_plane_pods(
        root,
        namespace=KSERVE_NAMESPACE,
        selector="control-plane=kserve-controller-manager",
        timeout_seconds=120,
    )
    _wait_control_plane_pods(
        root,
        namespace=ENVOY_GATEWAY_NAMESPACE,
        selector=None,
        timeout_seconds=120,
    )
    _run_checked(
        [
            "kubectl",
            "--context",
            KUBE_CONTEXT,
            "get",
            "crd/inferenceservices.serving.kserve.io",
        ],
        cwd=root,
    )


def _wait_for_cluster_api_access(root: Path, *, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_output = "Kubernetes API did not answer"
    command = [
        "kubectl",
        "--context",
        KUBE_CONTEXT,
        "--request-timeout=5s",
        "get",
        f"node/{KIND_NODE}",
        "--output=name",
    ]
    while time.monotonic() < deadline:
        result = subprocess.run(
            command,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode == 0:
            return
        last_output = result.stdout.strip() or last_output
        time.sleep(2)
    raise PpoKServeRolloutError(
        f"Kind Kubernetes API did not become accessible: {last_output[-1_024:]}"
    )


def _wait_control_plane_pods(
    root: Path,
    *,
    namespace: str,
    selector: str | None,
    timeout_seconds: int,
) -> None:
    command = [
        "kubectl",
        "--context",
        KUBE_CONTEXT,
        "--namespace",
        namespace,
        "wait",
        "pod",
        "--for=condition=Ready",
        f"--timeout={timeout_seconds}s",
    ]
    if selector is None:
        command.append("--all")
    else:
        command.extend(("--selector", selector))
    _run_checked(command, cwd=root)


def _node_image_digest(root: Path) -> str:
    normalized_image = f"docker.io/{PROXY_IMAGE}"
    raw_document = _capture_checked(
        ["docker", "exec", KIND_NODE, "crictl", "inspecti", normalized_image],
        cwd=root,
    )
    try:
        document = json.loads(raw_document)
    except json.JSONDecodeError as exc:
        raise PpoKServeRolloutError("Kind node returned invalid proxy image metadata") from exc
    status = document.get("status")
    if not isinstance(status, dict):
        raise PpoKServeRolloutError("Kind node proxy image status is missing")
    digest = status.get("id")
    repo_tags = status.get("repoTags")
    if (
        not isinstance(digest, str)
        or not _valid_image_digest(digest)
        or not isinstance(repo_tags, list)
        or normalized_image not in repo_tags
    ):
        raise PpoKServeRolloutError("Kind node fixed proxy image binding changed")
    return digest


def _image_digest(root: Path, image: str) -> str:
    value = _capture_checked(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        cwd=root,
    )
    if not _valid_image_digest(value):
        raise PpoKServeRolloutError("worker image digest is invalid")
    return value


def _kubectl_apply(root: Path, document: dict[str, Any]) -> None:
    _run_input_checked(
        [
            "kubectl",
            "--context",
            KUBE_CONTEXT,
            "apply",
            "--server-side",
            "--field-manager",
            "ppo-kserve-rollout",
            "-f",
            "-",
        ],
        json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode(),
        cwd=root,
    )


def _wait_resource(
    root: Path,
    resource: str,
    name: str,
    *,
    condition: str,
    timeout_seconds: int,
) -> None:
    _run_checked(
        [
            "kubectl",
            "--context",
            KUBE_CONTEXT,
            "--namespace",
            NAMESPACE,
            "wait",
            f"{resource}/{name}",
            f"--for=condition={condition}",
            f"--timeout={timeout_seconds}s",
        ],
        cwd=root,
    )


def _wait_service_endpoint(root: Path, service: str, *, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        document = _kubectl_json(root, "endpoints", service)
        subsets = document.get("subsets")
        if isinstance(subsets, list) and any(
            isinstance(item, dict) and item.get("addresses") for item in subsets
        ):
            return
        time.sleep(1)
    raise PpoKServeRolloutError(f"KServe endpoint is not ready: {service}")


def _wait_route(root: Path, *, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        route = _kubectl_json(root, "httproute", ROUTE_NAME)
        if route_condition_is_true(route, "Accepted") and route_condition_is_true(
            route, "ResolvedRefs"
        ):
            time.sleep(0.5)
            return
        time.sleep(0.5)
    raise PpoKServeRolloutError("PPO HTTPRoute was not accepted and resolved")


def _gateway_service(root: Path) -> tuple[str, str]:
    document = json.loads(
        _capture_checked(
            ["kubectl", "--context", KUBE_CONTEXT, "get", "service", "-A", "-o", "json"],
            cwd=root,
        )
    )
    matches: list[tuple[str, str]] = []
    for item in document.get("items", []):
        metadata = item.get("metadata", {})
        labels = metadata.get("labels", {})
        if (
            labels.get("gateway.envoyproxy.io/owning-gateway-name") == GATEWAY_NAME
            and labels.get("gateway.envoyproxy.io/owning-gateway-namespace") == NAMESPACE
        ):
            matches.append((metadata.get("namespace", ""), metadata.get("name", "")))
    if len(matches) != 1 or not all(matches[0]):
        raise PpoKServeRolloutError("Envoy Gateway service is not unique")
    return matches[0]


@contextmanager
def _port_forward(
    root: Path,
    *,
    namespace: str,
    resource: str,
    local_port: int,
) -> Iterator[None]:
    process = subprocess.Popen(
        [
            "kubectl",
            "--context",
            KUBE_CONTEXT,
            "--namespace",
            namespace,
            "port-forward",
            resource,
            f"{local_port}:80",
            "--address",
            "127.0.0.1",
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_tcp_process(process, local_port, timeout_seconds=30)
        yield
    finally:
        _stop_process(process)


def _wait_worker(
    process: subprocess.Popen[str],
    worker_url: str,
    *,
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise PpoKServeRolloutError("PPO GPU worker exited during startup")
        try:
            document = _get_json(f"{worker_url}/health/ready", timeout_seconds=2)
            if document.get("ready") is True:
                return
        except (httpx.HTTPError, PpoKServeRolloutError):
            pass
        time.sleep(1)
    raise PpoKServeRolloutError("PPO GPU worker did not become ready")


def _wait_tcp_process(
    process: subprocess.Popen[str],
    port: int,
    *,
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            raise PpoKServeRolloutError(f"port-forward failed: {output[-512:]}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.25)
    raise PpoKServeRolloutError("port-forward did not become ready")


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    with httpx.Client(timeout=httpx.Timeout(timeout_seconds), trust_env=False) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        document = response.json()
    if not isinstance(document, dict):
        raise PpoKServeRolloutError("PPO response is invalid")
    return document


def _get_json(url: str, *, timeout_seconds: int) -> dict[str, Any]:
    with httpx.Client(timeout=timeout_seconds, trust_env=False) as client:
        response = client.get(url)
        response.raise_for_status()
        document = response.json()
    if not isinstance(document, dict):
        raise PpoKServeRolloutError("PPO runtime response is invalid")
    return document


def _wait_candidate_delta(
    worker_url: str,
    before: dict[str, Any],
    *,
    minimum: int,
    timeout_seconds: int,
) -> None:
    original = _variant_count(before, "request_counts", "candidate")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        current = _get_json(f"{worker_url}/metrics", timeout_seconds=5)
        if _variant_count(current, "request_counts", "candidate") - original >= minimum:
            return
        time.sleep(0.25)
    raise PpoKServeRolloutError("Shadow traffic was not mirrored to PPO candidate")


def _kubectl_json(root: Path, resource: str, name: str) -> dict[str, Any]:
    raw = _capture_checked(
        [
            "kubectl",
            "--context",
            KUBE_CONTEXT,
            "--namespace",
            NAMESPACE,
            "get",
            resource,
            name,
            "-o",
            "json",
        ],
        cwd=root,
    )
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise PpoKServeRolloutError("Kubernetes resource response is invalid")
    return document


def _delete_ppo_resources(root: Path, *, check: bool = True) -> None:
    result = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBE_CONTEXT,
            "--namespace",
            NAMESPACE,
            "delete",
            f"httproute/{ROUTE_NAME}",
            f"inferenceservice/{STABLE_SERVICE}",
            f"inferenceservice/{CANDIDATE_SERVICE}",
            f"configmap/{CONFIG_MAP_NAME}",
            "--ignore-not-found=true",
            "--wait=true",
            "--timeout=120s",
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and result.returncode != 0:
        raise PpoKServeRolloutError(f"PPO resource cleanup failed: {result.stdout[-512:]}")


def _stop_worker(
    root: Path,
    process: subprocess.Popen[str],
    *,
    check: bool = True,
) -> None:
    if _container_exists(root, WORKER_CONTAINER):
        result = subprocess.run(
            ["docker", "stop", "--time", "20", WORKER_CONTAINER],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if check and result.returncode != 0:
            raise PpoKServeRolloutError(f"PPO worker cleanup failed: {result.stdout[-512:]}")
    if process.poll() is None:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)
    if check and _container_exists(root, WORKER_CONTAINER):
        raise PpoKServeRolloutError("PPO worker container still exists after cleanup")


def _stop_cluster(root: Path, *, check: bool = True) -> None:
    result = subprocess.run(
        ["docker", "stop", "--time", "30", KIND_NODE],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and result.returncode != 0:
        raise PpoKServeRolloutError(f"Kind cluster stop failed: {result.stdout[-512:]}")


def _container_exists(root: Path, name: str) -> bool:
    return (
        subprocess.run(
            ["docker", "container", "inspect", name],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _resource_ready(document: dict[str, Any]) -> bool:
    conditions = document.get("status", {}).get("conditions", [])
    return any(
        isinstance(item, dict) and item.get("type") == "Ready" and item.get("status") == "True"
        for item in conditions
    )


def _resolve_latest_outcome(root: Path, pointer_path: Path) -> Path:
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if not isinstance(pointer, dict):
        raise PpoKServeRolloutError("PPO runtime pointer is invalid")
    value = pointer.get("outcome") or pointer.get("report")
    if not isinstance(value, str) or not value:
        raise PpoKServeRolloutError("PPO runtime pointer has no outcome")
    path = (root / value).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PpoKServeRolloutError("PPO runtime pointer escaped repository") from exc
    return path


def _variant_count(document: dict[str, Any], field: str, variant: str) -> int:
    values = document.get(field)
    if not isinstance(values, dict):
        raise PpoKServeRolloutError(f"runtime metric is invalid: {field}")
    value = values.get(variant)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PpoKServeRolloutError(f"runtime metric is invalid: {field}.{variant}")
    return value


def _text(document: dict[str, Any], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value:
        raise PpoKServeRolloutError(f"field is invalid: {field}")
    return value


def _positive_integer(document: dict[str, Any], field: str) -> int:
    value = document.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PpoKServeRolloutError(f"field is invalid: {field}")
    return value


def _valid_image_digest(value: str) -> bool:
    return (
        value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * quantile)))
    return ordered[index]


def _stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _run_checked(argv: list[str], *, cwd: Path) -> None:
    result = subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise PpoKServeRolloutError(f"command failed ({argv[0]}): {result.stdout[-1_024:]}")


def _capture_checked(argv: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise PpoKServeRolloutError(f"command failed ({argv[0]}): {result.stdout[-1_024:]}")
    return result.stdout.strip()


def _run_input_checked(argv: list[str], payload: bytes, *, cwd: Path) -> None:
    result = subprocess.run(
        argv,
        cwd=cwd,
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise PpoKServeRolloutError(
            f"command failed ({argv[0]}): {result.stdout[-1_024:].decode(errors='replace')}"
        )


def _write_json(path: Path, document: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _progress(stage: str) -> None:
    print(f"PPO_KSERVE_ROLLOUT_STAGE={stage}", flush=True)


def _read_log_tail(path: Path, *, max_chars: int = 4_096) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except OSError:
        return ""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-ppo-kserve-rollout")
    parser.add_argument(
        "command",
        choices=("run", "verify", "cleanup"),
        nargs="?",
        default="run",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    if args.command == "run":
        path = execute_rollout(root)
        report = verify_ppo_kserve_rollout(root, path)
    elif args.command == "verify":
        path = root / OUTPUT_RELATIVE / "acceptance.json"
        report = verify_ppo_kserve_rollout(root, path)
    else:
        _ensure_cluster_running(root)
        _delete_ppo_resources(root)
        if _container_exists(root, WORKER_CONTAINER):
            _run_checked(["docker", "stop", WORKER_CONTAINER], cwd=root)
        _stop_cluster(root)
        print("PPO_KSERVE_RESOURCES_CLEANED")
        return 0
    _print_report(report, path)
    return 0


def _print_report(
    report: EnterprisePpoKServeRolloutReport,
    path: Path,
) -> None:
    print(report.status)
    print(report.classification)
    print(f"ACTUAL_GPU_EXECUTION={str(report.runtime.actual_gpu_execution).lower()}")
    print(f"MODEL_RELEASE_ID={report.model_release.release_id}")
    print(f"FINAL_STAGE={report.model_release.final_deployment_stage}")
    print(f"EVIDENCE_CHAIN_SHA256={report.evidence_chain_sha256}")
    print(path)


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
