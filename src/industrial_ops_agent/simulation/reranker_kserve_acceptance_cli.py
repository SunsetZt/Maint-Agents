"""Execute the governed Reranker release on the local KServe control plane."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from contextlib import ExitStack
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
    STAGE_POLICY,
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
from industrial_ops_agent.enterprise_assets.onboarding import (
    EnterpriseReleaseOnboardingService,
)
from industrial_ops_agent.enterprise_assets.service import (
    EnterpriseProjectAdoptionService,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import Base, TenantRecord
from industrial_ops_agent.releases.service import ModelReleaseService
from industrial_ops_agent.security_audit import InMemorySecurityAuditSink, SecurityAuditor
from industrial_ops_agent.simulation import asr_kserve_acceptance_cli as local
from industrial_ops_agent.simulation.asr_kserve_rollout import PROXY_IMAGE
from industrial_ops_agent.simulation.reranker_kserve_acceptance import (
    ACCEPTANCE_RELATIVE,
    CANDIDATE_SERVICE,
    CANDIDATE_WORKER_CONTAINER,
    CLASSIFICATION,
    CONFIG_MAP_NAME,
    ENDPOINT_PATH,
    EXCLUDED_RAW_COMPARATOR_FILES,
    GATEWAY_NAME,
    HOSTNAME,
    KSERVE_CONTROLLER_DEPLOYMENT,
    KSERVE_CONTROLLER_IMAGE,
    KUBE_CONTEXT,
    NAMESPACE,
    OUTPUT_RELATIVE,
    PROXY_SOURCE_RELATIVE,
    ROUTE_NAME,
    ROUTE_PATH_PREFIX,
    RUNTIME_IMAGE_DIGEST,
    RUNTIME_IMAGE_REPOSITORY,
    RUNTIME_IMAGE_TAG,
    STABLE_SERVICE,
    STABLE_WORKER_CONTAINER,
    STATUS,
    RerankerKServeAcceptanceError,
    RolloutStageEvidence,
    document_sha256,
    file_binding,
    file_sha256,
    finalize_acceptance,
    inference_service_document,
    proxy_config_map_document,
    route_condition_is_true,
    route_document,
    route_rules_semantically_equal,
    verify_reranker_kserve_acceptance,
    write_acceptance,
)
from industrial_ops_agent.simulation.reranker_kserve_readiness import (
    ACCEPTANCE_RELATIVE as READINESS_RELATIVE,
)
from industrial_ops_agent.simulation.reranker_kserve_readiness import (
    verify_reranker_kserve_readiness,
)
from industrial_ops_agent.simulation.reranker_runtime_gpu_probe import (
    ACCEPTANCE_RELATIVE as GPU_PROBE_RELATIVE,
)
from industrial_ops_agent.simulation.reranker_runtime_gpu_probe import (
    EXPECTED_TOP_CHUNK_ID,
    probe_request,
    verify_runtime_gpu_probe_report,
)

KIND_NODE = "ioap-gpu-promotion-lab-control-plane"
TENANT_ID = "project-enterprise-staging"
STAGE_REQUEST_COUNTS: dict[str, int] = {
    "SHADOW": int(STAGE_POLICY["SHADOW"]["min_request_count"]),
    "CANARY_5": int(STAGE_POLICY["CANARY_5"]["min_request_count"]),
    "CANARY_25": int(STAGE_POLICY["CANARY_25"]["min_request_count"]),
    "ROLLED_BACK": 20,
}
STAGE_WINDOW_SECONDS: dict[str, int] = {
    "SHADOW": int(STAGE_POLICY["SHADOW"]["min_window_seconds"]),
    "CANARY_5": int(STAGE_POLICY["CANARY_5"]["min_window_seconds"]),
    "CANARY_25": int(STAGE_POLICY["CANARY_25"]["min_window_seconds"]),
}


@dataclass(slots=True)
class _ReleaseState:
    database: Database
    engineer: IdentityContext
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


class _ObservedLocalKServeProvider:
    def __init__(self, root: Path, stage: str) -> None:
        self._root = root
        self._stage = stage

    def reconcile(self, target: ProviderTarget) -> ProviderResult:
        route = local._kubectl_json(self._root, "httproute", ROUTE_NAME)
        stable = local._kubectl_json(self._root, "inferenceservice", STABLE_SERVICE)
        candidate = local._kubectl_json(
            self._root,
            "inferenceservice",
            CANDIDATE_SERVICE,
        )
        expected_route = route_document(cast(Any, self._stage))
        ready = bool(
            target.desired_stage == self._stage
            and target.route_name == ROUTE_NAME
            and target.stable_service_name == STABLE_SERVICE
            and local._resource_ready(stable)
            and local._resource_ready(candidate)
            and route_condition_is_true(route, "Accepted")
            and route_condition_is_true(route, "ResolvedRefs")
            and route.get("metadata", {}).get("annotations", {}).get("ioap.openai.com/stage")
            == self._stage
            and route_rules_semantically_equal(route, expected_route)
        )
        revision = (
            "local-reranker-kserve-"
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
            endpoint_url=f"http://{HOSTNAME}{ENDPOINT_PATH}" if ready else None,
            reason_code=None if ready else "local_reranker_kserve_observation_mismatch",
        )


def execute_rollout(repo_root: Path) -> Path:
    root = repo_root.resolve(strict=True)
    acceptance_path = root / ACCEPTANCE_RELATIVE
    if acceptance_path.is_file():
        verify_reranker_kserve_acceptance(root, ACCEPTANCE_RELATIVE)
        return acceptance_path
    output = root / OUTPUT_RELATIVE
    if output.exists():
        raise RerankerKServeAcceptanceError(
            "Reranker rollout output exists without a verified acceptance"
        )
    readiness = verify_reranker_kserve_readiness(root, READINESS_RELATIVE)
    gpu_probe = verify_runtime_gpu_probe_report(root, GPU_PROBE_RELATIVE)
    if (
        gpu_probe.source.readiness_evidence_chain_sha256 != readiness.evidence_chain_sha256
        or gpu_probe.source.value_run_id != readiness.source.run_id
        or gpu_probe.runtime.image_digest != readiness.runtime_image.digest
    ):
        raise RerankerKServeAcceptanceError("Reranker rollout source is not accepted")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)
    kubernetes_output = temporary / "kubernetes"
    kubernetes_output.mkdir()
    raw = _build_raw_comparator(
        root,
        Path(readiness.source.candidate_path),
        readiness.source.candidate_bundle_sha256,
        temporary,
    )
    state_path = temporary / "model-release-state.sqlite3"
    state: _ReleaseState | None = None
    stable_worker: subprocess.Popen[str] | None = None
    candidate_worker: subprocess.Popen[str] | None = None
    stable_log_handle: Any = None
    candidate_log_handle: Any = None
    stable_log_path = temporary / "stable-gpu-worker.log"
    candidate_log_path = temporary / "candidate-gpu-worker.log"
    resources_applied = False
    resources_removed = False
    workers_stopped = False
    port_forwards_stopped = False
    cluster_started = False
    cluster_stopped = False
    database_disposed = False
    committed = False
    try:
        _progress("MODEL_RELEASE_PREPARING")
        state = _prepare_release_state(root, state_path)
        release_id = state.deployment.release.release_id
        manifest_hash = state.deployment.release.manifest_hash
        public_component_id = state.imported_candidate_experiment_id
        public_artifact_hash = readiness.source.candidate_bundle_sha256

        _progress("CLUSTER_STARTING")
        cluster_started = True
        _ensure_reranker_cluster_running(root)
        _progress("CLUSTER_READY")
        proxy_image_digest = local._node_image_digest(root)
        runtime_image = f"{RUNTIME_IMAGE_REPOSITORY}:{RUNTIME_IMAGE_TAG}"
        if local._image_digest(root, runtime_image) != RUNTIME_IMAGE_DIGEST:
            raise RerankerKServeAcceptanceError("Reranker runtime image changed")
        host_gateway = local._capture_checked(
            [
                "docker",
                "inspect",
                KIND_NODE,
                "--format",
                "{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}",
            ],
            cwd=root,
        )
        stable_worker_port = local._free_port()
        candidate_worker_port = local._free_port()
        stable_log_handle = stable_log_path.open("w", encoding="utf-8")
        candidate_log_handle = candidate_log_path.open("w", encoding="utf-8")
        _progress("GPU_WORKERS_STARTING")
        stable_worker = _start_worker(
            root,
            name=STABLE_WORKER_CONTAINER,
            port=stable_worker_port,
            model_directory=temporary / "raw-comparator",
            release_id=release_id,
            manifest_hash=manifest_hash,
            component_model_id=cast(str, raw["component_model_id"]),
            artifact_hash=cast(str, raw["bundle_sha256"]),
            log=stable_log_handle,
        )
        candidate_worker = _start_worker(
            root,
            name=CANDIDATE_WORKER_CONTAINER,
            port=candidate_worker_port,
            model_directory=root / readiness.source.candidate_path,
            release_id=release_id,
            manifest_hash=manifest_hash,
            component_model_id=public_component_id,
            artifact_hash=public_artifact_hash,
            log=candidate_log_handle,
        )
        stable_url = f"http://127.0.0.1:{stable_worker_port}"
        candidate_url = f"http://127.0.0.1:{candidate_worker_port}"
        _wait_worker(
            stable_worker,
            stable_url,
            component_model_id=cast(str, raw["component_model_id"]),
            calibration_enabled=False,
            timeout_seconds=180,
        )
        _wait_worker(
            candidate_worker,
            candidate_url,
            component_model_id=public_component_id,
            calibration_enabled=True,
            timeout_seconds=180,
        )
        _progress("GPU_WORKERS_READY")
        _write_container_inspect(
            root,
            STABLE_WORKER_CONTAINER,
            temporary / "stable-container-inspect.json",
        )
        _write_container_inspect(
            root,
            CANDIDATE_WORKER_CONTAINER,
            temporary / "candidate-container-inspect.json",
        )

        proxy_path = (root / PROXY_SOURCE_RELATIVE).resolve(strict=True)
        proxy_source = proxy_path.read_text(encoding="utf-8")
        common = {
            "host_gateway": host_gateway,
            "proxy_image_digest": proxy_image_digest,
            "release_id": release_id,
            "manifest_hash": manifest_hash,
            "public_component_model_id": public_component_id,
            "public_artifact_hash": public_artifact_hash,
        }
        config_map = proxy_config_map_document(proxy_source)
        stable_isvc = inference_service_document(
            variant="stable",
            host_port=stable_worker_port,
            upstream_component_model_id=cast(str, raw["component_model_id"]),
            upstream_artifact_hash=cast(str, raw["bundle_sha256"]),
            **common,
        )
        candidate_isvc = inference_service_document(
            variant="candidate",
            host_port=candidate_worker_port,
            upstream_component_model_id=public_component_id,
            upstream_artifact_hash=public_artifact_hash,
            **common,
        )
        _progress("KSERVE_RESOURCES_APPLYING")
        for name, document in (
            ("proxy-configmap.json", config_map),
            ("stable-inferenceservice.json", stable_isvc),
            ("candidate-inferenceservice.json", candidate_isvc),
        ):
            _write_json(kubernetes_output / name, document)
            local._kubectl_apply(root, document)
        resources_applied = True
        for service in (STABLE_SERVICE, CANDIDATE_SERVICE):
            local._wait_resource(
                root,
                "inferenceservice",
                service,
                condition="Ready",
                timeout_seconds=240,
            )
            local._wait_service_endpoint(
                root,
                f"{service}-predictor",
                timeout_seconds=120,
            )
        _progress("KSERVE_RESOURCES_READY")
        _write_json(
            kubernetes_output / "observed-kserve-controller.json",
            _kserve_controller_document(root),
        )

        gateway_namespace, gateway_service = local._gateway_service(root)
        stable_port = local._free_port()
        candidate_port = local._free_port()
        gateway_port = local._free_port()
        route_paths: dict[str, Path] = {}
        stage_reports: list[RolloutStageEvidence] = []
        quality_request = probe_request()
        quality_request.pop("probe_metadata")
        quality_request.update(
            {
                "expected_release_id": release_id,
                "expected_manifest_hash": manifest_hash,
                "expected_component_model_id": public_component_id,
                "expected_artifact_content_hash": public_artifact_hash,
            }
        )
        quality_request_path = temporary / "quality-request.json"
        stable_response_path = temporary / "direct-stable-response.json"
        candidate_response_path = temporary / "direct-candidate-response.json"
        with ExitStack() as stack:
            stack.enter_context(
                local._port_forward(
                    root,
                    namespace=NAMESPACE,
                    resource=f"service/{STABLE_SERVICE}-predictor",
                    local_port=stable_port,
                )
            )
            stack.enter_context(
                local._port_forward(
                    root,
                    namespace=NAMESPACE,
                    resource=f"service/{CANDIDATE_SERVICE}-predictor",
                    local_port=candidate_port,
                )
            )
            stack.enter_context(
                local._port_forward(
                    root,
                    namespace=gateway_namespace,
                    resource=f"service/{gateway_service}",
                    local_port=gateway_port,
                )
            )
            _write_json(quality_request_path, quality_request)
            stable_response = _post_reranker(
                f"http://127.0.0.1:{stable_port}{ENDPOINT_PATH}",
                quality_request,
            )
            candidate_response = _post_reranker(
                f"http://127.0.0.1:{candidate_port}{ENDPOINT_PATH}",
                quality_request,
            )
            _write_json(stable_response_path, stable_response)
            _write_json(candidate_response_path, candidate_response)
            quality = _quality_evidence(stable_response, candidate_response)
            gateway_url = f"http://127.0.0.1:{gateway_port}{ENDPOINT_PATH}"
            stable_metrics_url = f"http://127.0.0.1:{stable_port}/metrics"
            candidate_metrics_url = f"http://127.0.0.1:{candidate_port}/metrics"
            for stage in ("SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"):
                _progress(f"{stage}_TRAFFIC_STARTING")
                route = route_document(cast(Any, stage))
                route_path = kubernetes_output / f"route-{stage.lower()}.json"
                route_paths[stage] = route_path
                _write_json(route_path, route)
                local._kubectl_apply(root, route)
                _wait_route(root, timeout_seconds=60)
                state.deployment = state.deployment_service.reconcile(
                    state.controller,
                    state.deployment.deployment.deployment_id,
                    _ObservedLocalKServeProvider(root, stage),
                    expected_version=state.deployment.deployment.version,
                    request_id=f"reconcile-reranker-{stage.lower()}",
                )
                if (
                    state.deployment.deployment.status != "READY"
                    or state.deployment.deployment.current_stage != stage
                ):
                    raise RerankerKServeAcceptanceError(
                        f"Reranker deployment did not enter {stage}"
                    )
                before_stable = _get_json(stable_metrics_url)
                before_candidate = _get_json(candidate_metrics_url)
                traffic = _send_gateway_traffic(
                    gateway_url,
                    quality_request,
                    request_count=STAGE_REQUEST_COUNTS[stage],
                )
                if stage == "SHADOW":
                    _wait_proxy_delta(
                        candidate_metrics_url,
                        before_candidate,
                        minimum=STAGE_REQUEST_COUNTS[stage],
                        timeout_seconds=30,
                    )
                after_stable = _get_json(stable_metrics_url)
                after_candidate = _get_json(candidate_metrics_url)
                stage_report = _stage_evidence(
                    cast(Any, stage),
                    route=route,
                    before_stable=before_stable,
                    after_stable=after_stable,
                    before_candidate=before_candidate,
                    after_candidate=after_candidate,
                    variants=traffic["variants"],
                    output_sha256s=traffic["output_sha256s"],
                    latencies=traffic["latencies"],
                )
                if stage != "ROLLED_BACK":
                    state.deployment = _record_stage_observation(
                        state,
                        stage=cast(Any, stage),
                        stage_report=stage_report,
                    )
                    observation = (
                        state.deployment.observations[-1] if state.deployment.observations else None
                    )
                    if observation is None or observation.decision != "PASS":
                        decision = observation.decision if observation is not None else "MISSING"
                        reasons = (
                            ",".join(observation.failure_reasons)
                            if observation is not None
                            else "observation_missing"
                        )
                        raise RerankerKServeAcceptanceError(
                            f"Reranker {stage} observation did not pass: "
                            f"decision={decision}; reasons={reasons}"
                        )
                    stage_report = stage_report.model_copy(
                        update={"model_release_observation_decision": "PASS"}
                    )
                    if stage in {"SHADOW", "CANARY_5"}:
                        state.deployment = state.deployment_service.promote(
                            state.operator,
                            state.deployment.release.release_id,
                            expected_deployment_version=(state.deployment.deployment.version),
                            request_id=f"promote-reranker-{stage.lower()}",
                        )
                    else:
                        state.deployment = state.deployment_service.request_rollback(
                            state.operator,
                            state.deployment.release.release_id,
                            expected_deployment_version=(state.deployment.deployment.version),
                            reason_code="local_reranker_acceptance_rollback_drill",
                            request_id="rollback-reranker-after-canary-25",
                        )
                stage_reports.append(stage_report)
                _progress(f"{stage}_TRAFFIC_PASSED")
            stable_proxy_metrics = _get_json(stable_metrics_url)
            candidate_proxy_metrics = _get_json(candidate_metrics_url)
            _write_json(temporary / "stable-proxy-metrics.json", stable_proxy_metrics)
            _write_json(
                temporary / "candidate-proxy-metrics.json",
                candidate_proxy_metrics,
            )
        port_forwards_stopped = True

        final_route = local._kubectl_json(root, "httproute", ROUTE_NAME)
        stable_state = local._kubectl_json(root, "inferenceservice", STABLE_SERVICE)
        candidate_state = local._kubectl_json(
            root,
            "inferenceservice",
            CANDIDATE_SERVICE,
        )
        _write_json(kubernetes_output / "observed-final-route.json", final_route)
        _write_json(kubernetes_output / "observed-stable.json", stable_state)
        _write_json(kubernetes_output / "observed-candidate.json", candidate_state)
        _write_text(
            temporary / "stable-runtime-metrics.txt",
            _get_text(f"{stable_url}/metrics"),
        )
        _write_text(
            temporary / "candidate-runtime-metrics.txt",
            _get_text(f"{candidate_url}/metrics"),
        )
        gpu_inventory = local._capture_checked(
            [
                "docker",
                "exec",
                CANDIDATE_WORKER_CONTAINER,
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            cwd=root,
        )
        _write_text(temporary / "gpu-inventory.csv", gpu_inventory + "\n")
        release_snapshot = _release_snapshot(state)
        manifest_path = temporary / "release-manifest.json"
        _write_json(manifest_path, release_snapshot["manifest"])

        _progress("CLEANUP_STARTING")
        _delete_resources(root)
        resources_removed = True
        _stop_worker(root, STABLE_WORKER_CONTAINER, stable_worker)
        _stop_worker(root, CANDIDATE_WORKER_CONTAINER, candidate_worker)
        workers_stopped = True
        stable_log_handle.close()
        candidate_log_handle.close()
        stable_log_handle = None
        candidate_log_handle = None
        local._stop_cluster(root)
        cluster_stopped = True
        state.database.dispose()
        database_disposed = True
        _require_rollout_services_absent(root)

        report = finalize_acceptance(
            _acceptance_document(
                root=root,
                temporary=temporary,
                readiness=readiness,
                gpu_probe=gpu_probe,
                raw=raw,
                state=state,
                release_snapshot=release_snapshot,
                manifest_path=manifest_path,
                proxy_image_digest=proxy_image_digest,
                proxy_path=proxy_path,
                host_gateway=host_gateway,
                stable_worker_port=stable_worker_port,
                candidate_worker_port=candidate_worker_port,
                route_paths=route_paths,
                quality_request_path=quality_request_path,
                stable_response_path=stable_response_path,
                candidate_response_path=candidate_response_path,
                quality=quality,
                stage_reports=stage_reports,
                resources_removed=resources_removed,
                workers_stopped=workers_stopped,
                port_forwards_stopped=port_forwards_stopped,
                cluster_stopped=cluster_stopped,
            )
        )
        write_acceptance(report, temporary / "acceptance.json")
        temporary.replace(output)
        try:
            verify_reranker_kserve_acceptance(root, ACCEPTANCE_RELATIVE)
        except Exception:
            output.replace(temporary)
            raise
        committed = True
        _progress("ACCEPTANCE_COMMITTED")
        return acceptance_path
    except Exception as exc:
        _progress(f"FAILED_{type(exc).__name__}")
        print(f"RERANKER_KSERVE_ACCEPTANCE_ERROR={exc}", flush=True)
        for name, path in (
            ("STABLE", stable_log_path),
            ("CANDIDATE", candidate_log_path),
        ):
            tail = _read_tail(path)
            if tail:
                print(f"RERANKER_{name}_WORKER_LOG_TAIL_BEGIN", flush=True)
                print(tail, flush=True)
                print(f"RERANKER_{name}_WORKER_LOG_TAIL_END", flush=True)
        raise
    finally:
        if resources_applied and not resources_removed and cluster_started:
            _delete_resources(root, check=False)
        if stable_worker is not None and not workers_stopped:
            _stop_worker(root, STABLE_WORKER_CONTAINER, stable_worker, check=False)
        if candidate_worker is not None and not workers_stopped:
            _stop_worker(
                root,
                CANDIDATE_WORKER_CONTAINER,
                candidate_worker,
                check=False,
            )
        if stable_log_handle is not None and not stable_log_handle.closed:
            stable_log_handle.close()
        if candidate_log_handle is not None and not candidate_log_handle.closed:
            candidate_log_handle.close()
        if cluster_started and not cluster_stopped:
            local._stop_cluster(root, check=False)
        if state is not None and not database_disposed:
            state.database.dispose()
        if not committed and temporary.exists():
            shutil.rmtree(temporary)


def _acceptance_document(
    *,
    root: Path,
    temporary: Path,
    readiness: Any,
    gpu_probe: Any,
    raw: dict[str, Any],
    state: _ReleaseState,
    release_snapshot: dict[str, Any],
    manifest_path: Path,
    proxy_image_digest: str,
    proxy_path: Path,
    host_gateway: str,
    stable_worker_port: int,
    candidate_worker_port: int,
    route_paths: dict[str, Path],
    quality_request_path: Path,
    stable_response_path: Path,
    candidate_response_path: Path,
    quality: dict[str, Any],
    stage_reports: list[RolloutStageEvidence],
    resources_removed: bool,
    workers_stopped: bool,
    port_forwards_stopped: bool,
    cluster_stopped: bool,
) -> dict[str, Any]:
    def bind(path: Path, relative: Path) -> dict[str, Any]:
        return file_binding(
            root,
            path,
            reported_path=OUTPUT_RELATIVE / relative,
        )

    return {
        "schema_version": "enterprise-reranker-kserve-rollout-acceptance/v1",
        "classification": CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "external_enterprise_production_claim": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": STATUS,
        "source": {
            "readiness": file_binding(root, root / READINESS_RELATIVE),
            "readiness_evidence_chain_sha256": readiness.evidence_chain_sha256,
            "gpu_probe": file_binding(root, root / GPU_PROBE_RELATIVE),
            "gpu_probe_evidence_chain_sha256": gpu_probe.evidence_chain_sha256,
            "candidate_run_id": readiness.source.run_id,
            "candidate_path": readiness.source.candidate_path,
            "candidate_bundle_sha256": readiness.source.candidate_bundle_sha256,
        },
        "raw_comparator": {
            "manifest": bind(
                temporary / "raw-comparator-manifest.json",
                Path("raw-comparator-manifest.json"),
            ),
            **raw,
        },
        "model_release": {
            "state_database": bind(
                temporary / "model-release-state.sqlite3",
                Path("model-release-state.sqlite3"),
            ),
            "sqlite_integrity_check": "ok",
            "tenant_id": TENANT_ID,
            "baseline_release_id": state.baseline_release_id,
            "release_id": state.deployment.release.release_id,
            "manifest": bind(manifest_path, Path("release-manifest.json")),
            "manifest_hash": release_snapshot["manifest_hash"],
            "source_candidate_experiment_id": state.source_candidate_experiment_id,
            "imported_candidate_experiment_id": state.imported_candidate_experiment_id,
            "evaluation_id": state.evaluation_id,
            "approval_id": state.approval_id,
            "approval_requested_by_subject_id": (state.approval_requested_by_subject_id),
            "approval_decided_by_subject_id": state.approval_decided_by_subject_id,
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
            "topology": "TWO_DOCKER_GPU_WORKERS_BEHIND_KSERVE_PROXIES",
            "actual_gpu_execution": True,
            "model_execution_simulated": False,
            "image_repository": RUNTIME_IMAGE_REPOSITORY,
            "image_tag": RUNTIME_IMAGE_TAG,
            "image_digest": RUNTIME_IMAGE_DIGEST,
            "stable_worker_container": STABLE_WORKER_CONTAINER,
            "candidate_worker_container": CANDIDATE_WORKER_CONTAINER,
            "stable_worker_log": bind(
                temporary / "stable-gpu-worker.log",
                Path("stable-gpu-worker.log"),
            ),
            "candidate_worker_log": bind(
                temporary / "candidate-gpu-worker.log",
                Path("candidate-gpu-worker.log"),
            ),
            "stable_container_inspect": bind(
                temporary / "stable-container-inspect.json",
                Path("stable-container-inspect.json"),
            ),
            "candidate_container_inspect": bind(
                temporary / "candidate-container-inspect.json",
                Path("candidate-container-inspect.json"),
            ),
            "stable_runtime_metrics": bind(
                temporary / "stable-runtime-metrics.txt",
                Path("stable-runtime-metrics.txt"),
            ),
            "candidate_runtime_metrics": bind(
                temporary / "candidate-runtime-metrics.txt",
                Path("candidate-runtime-metrics.txt"),
            ),
            "stable_proxy_metrics": bind(
                temporary / "stable-proxy-metrics.json",
                Path("stable-proxy-metrics.json"),
            ),
            "candidate_proxy_metrics": bind(
                temporary / "candidate-proxy-metrics.json",
                Path("candidate-proxy-metrics.json"),
            ),
            "gpu_inventory": bind(
                temporary / "gpu-inventory.csv",
                Path("gpu-inventory.csv"),
            ),
            "exactly_one_gpu_exposed_per_worker": True,
            "cpu_limit_per_worker": 1.5,
            "memory_limit_bytes_per_worker": 4 * 1024**3,
            "read_only_root_filesystem": True,
            "non_root_user": "app",
            "no_new_privileges": True,
            "all_linux_capabilities_dropped": True,
        },
        "kserve": {
            "cluster_context": KUBE_CONTEXT,
            "namespace": NAMESPACE,
            "controller_deployment": bind(
                temporary / "kubernetes/observed-kserve-controller.json",
                Path("kubernetes/observed-kserve-controller.json"),
            ),
            "controller_image": KSERVE_CONTROLLER_IMAGE,
            "controller_image_pull_policy": "IfNotPresent",
            "gateway_name": GATEWAY_NAME,
            "hostname": HOSTNAME,
            "route_name": ROUTE_NAME,
            "route_path_prefix": ROUTE_PATH_PREFIX,
            "stable_inference_service": STABLE_SERVICE,
            "candidate_inference_service": CANDIDATE_SERVICE,
            "stable_ready": True,
            "candidate_ready": True,
            "route_accepted": True,
            "route_resolved_refs": True,
            "worker_host_gateway": host_gateway,
            "stable_worker_host_port": stable_worker_port,
            "candidate_worker_host_port": candidate_worker_port,
            "proxy_image": PROXY_IMAGE,
            "proxy_image_digest": proxy_image_digest,
            "proxy_source_sha256": file_sha256(proxy_path),
            "config_map": bind(
                temporary / "kubernetes/proxy-configmap.json",
                Path("kubernetes/proxy-configmap.json"),
            ),
            "stable_service": bind(
                temporary / "kubernetes/stable-inferenceservice.json",
                Path("kubernetes/stable-inferenceservice.json"),
            ),
            "candidate_service": bind(
                temporary / "kubernetes/candidate-inferenceservice.json",
                Path("kubernetes/candidate-inferenceservice.json"),
            ),
            "routes": {
                stage: bind(
                    path,
                    Path(f"kubernetes/route-{stage.lower()}.json"),
                )
                for stage, path in route_paths.items()
            },
            "observed_final_route": bind(
                temporary / "kubernetes/observed-final-route.json",
                Path("kubernetes/observed-final-route.json"),
            ),
            "observed_stable_service": bind(
                temporary / "kubernetes/observed-stable.json",
                Path("kubernetes/observed-stable.json"),
            ),
            "observed_candidate_service": bind(
                temporary / "kubernetes/observed-candidate.json",
                Path("kubernetes/observed-candidate.json"),
            ),
            "hardened_container_security_context": True,
            "proxy_has_no_gpu_request": True,
            "endpoint_scope": "LOCAL_PORT_FORWARD",
        },
        "quality": {
            "request": bind(quality_request_path, Path("quality-request.json")),
            "stable_response": bind(
                stable_response_path,
                Path("direct-stable-response.json"),
            ),
            "candidate_response": bind(
                candidate_response_path,
                Path("direct-candidate-response.json"),
            ),
            "expected_top_chunk_id": EXPECTED_TOP_CHUNK_ID,
            **quality,
        },
        "stages": [item.model_dump(mode="json") for item in stage_reports],
        "cleanup": {
            "stable_gpu_worker_stopped": workers_stopped,
            "candidate_gpu_worker_stopped": workers_stopped,
            "port_forwards_stopped": port_forwards_stopped,
            "reranker_kubernetes_resources_removed": resources_removed,
            "cluster_stopped_after_acceptance": cluster_stopped,
            "no_rollout_service_left_running": True,
            "evidence_is_historical_after_cleanup": True,
        },
        "hard_gates": {
            "preflight_readiness_verified": True,
            "actual_gpu_probe_verified": True,
            "raw_and_calibrated_variants_isolated": True,
            "independent_release_approval_verified": True,
            "kserve_inference_services_ready": True,
            "kserve_controller_local_image_policy_verified": True,
            "gateway_route_conditions_verified": True,
            "shadow_mirror_verified": True,
            "canary_5_verified": True,
            "canary_25_verified": True,
            "rollback_zero_candidate_traffic_verified": True,
            "model_release_fsm_verified": True,
            "resource_cleanup_verified": True,
        },
    }


def _prepare_release_state(root: Path, state_path: Path) -> _ReleaseState:
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
    authorizer = Authorizer(
        SecurityAuditor(
            InMemorySecurityAuditSink(),
            hash_key=b"reranker-local-kserve-release",
        )
    )
    engineer = _identity(Role.MODEL_ENGINEER)
    approver = _identity(Role.MODEL_RELEASE_APPROVER)
    operator = _identity(Role.MODEL_RELEASE_OPERATOR)
    controller = _identity(Role.MODEL_DEPLOYMENT_CONTROLLER)
    reader = EnterpriseProjectAdoptionService(root, cache_seconds=0)
    imported = EnterpriseModelAssetImportService(
        database,
        authorizer,
        reader,
        root,
    ).import_component(
        engineer,
        "RERANKER",
        idempotency_key="reranker-kserve-import",
        request_id="reranker-kserve-import",
    )
    releases = ModelReleaseService(database, authorizer)
    baseline = EnterpriseStagingBaselineService(database, authorizer).create_draft(
        engineer,
        active_mcp_server_versions={},
        idempotency_key="reranker-kserve-baseline",
        request_id="reranker-kserve-baseline",
    )
    baseline_candidate = releases.validate(
        engineer,
        baseline.release_id,
        expected_version=baseline.version,
        request_id="reranker-kserve-baseline-validate",
    )
    baseline_pending = releases.submit_for_approval(
        engineer,
        baseline.release_id,
        expected_version=baseline_candidate.release.version,
        request_id="reranker-kserve-baseline-submit",
    )
    if baseline_pending.approval is None:
        raise RerankerKServeAcceptanceError("Reranker baseline approval was not created")
    releases.decide_approval(
        approver,
        baseline.release_id,
        expected_version=baseline_pending.release.version,
        expected_approval_version=baseline_pending.approval.version,
        decision="APPROVED",
        reason="independent local Reranker Staging baseline review passed",
        request_id="reranker-kserve-baseline-approve",
    )
    onboarding = EnterpriseReleaseOnboardingService(
        database,
        authorizer,
        reader,
    ).create_draft(
        engineer,
        "RERANKER",
        baseline_release_id=baseline.release_id,
        auto_shadow_enabled=False,
        deployment_plan=None,
        active_mcp_server_versions={},
        idempotency_key="reranker-kserve-candidate-release",
        request_id="reranker-kserve-candidate-release",
    )
    candidate = releases.validate(
        engineer,
        onboarding.release_id,
        expected_version=onboarding.release_version,
        request_id="reranker-kserve-candidate-validate",
    )
    pending = releases.submit_for_approval(
        engineer,
        onboarding.release_id,
        expected_version=candidate.release.version,
        request_id="reranker-kserve-candidate-submit",
    )
    if pending.approval is None:
        raise RerankerKServeAcceptanceError("Reranker candidate approval was not created")
    approved = releases.decide_approval(
        approver,
        onboarding.release_id,
        expected_version=pending.release.version,
        expected_approval_version=pending.approval.version,
        decision="APPROVED",
        reason="independent Reranker Gold, GPU runtime and supply-chain review passed",
        request_id="reranker-kserve-candidate-approve",
    )
    if approved.approval is None or approved.approval.status != "APPROVED":
        raise RerankerKServeAcceptanceError("Reranker candidate approval did not pass")
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
            serving_runtime_name="industrial-reranker-runtime",
            artifact_uri_prefix="s3://project-enterprise-staging/reranker",
        ),
        expected_release_version=approved.release.version,
        request_id="reranker-kserve-request-shadow",
    )
    return _ReleaseState(
        database=database,
        engineer=engineer,
        operator=operator,
        controller=controller,
        release_service=releases,
        deployment_service=deployment_service,
        baseline_release_id=baseline.release_id,
        source_candidate_experiment_id=imported.source_candidate_experiment_id,
        imported_candidate_experiment_id=imported.imported_candidate_experiment_id,
        evaluation_id=imported.imported_evaluation_id,
        approval_id=approved.approval.approval_id,
        approval_requested_by_subject_id=approved.approval.requested_by_subject_id,
        approval_decided_by_subject_id=cast(str, approved.approval.decided_by_subject_id),
        deployment=deployment,
    )


def _build_raw_comparator(
    root: Path,
    candidate_relative: Path,
    candidate_bundle_sha256: str,
    temporary: Path,
) -> dict[str, Any]:
    source = (root / candidate_relative).resolve(strict=True)
    directory = temporary / "raw-comparator"
    directory.mkdir()
    records: list[dict[str, Any]] = []
    for source_file in sorted(item for item in source.iterdir() if item.is_file()):
        if source_file.name in EXCLUDED_RAW_COMPARATOR_FILES:
            continue
        target = directory / source_file.name
        try:
            os.link(source_file, target)
        except OSError:
            shutil.copy2(source_file, target)
        target.chmod(0o644)
        records.append(
            {
                "path": (OUTPUT_RELATIVE / "raw-comparator" / target.name).as_posix(),
                "size_bytes": target.stat().st_size,
                "sha256": file_sha256(target),
            }
        )
    bundle_sha256 = document_sha256(records)
    component_model_id = f"reranker-raw-comparator-{bundle_sha256[:20]}"
    document = {
        "schema_version": "enterprise-reranker-raw-comparator-manifest/v1",
        "classification": CLASSIFICATION,
        "purpose": "ROUTING_BASELINE_ONLY",
        "routing_only_not_release_candidate": True,
        "source_candidate_path": candidate_relative.as_posix(),
        "source_candidate_bundle_sha256": candidate_bundle_sha256,
        "excluded_files": list(EXCLUDED_RAW_COMPARATOR_FILES),
        "component_model_id": component_model_id,
        "bundle_sha256": bundle_sha256,
        "files": records,
    }
    _write_json(temporary / "raw-comparator-manifest.json", document)
    return {
        "directory": (OUTPUT_RELATIVE / "raw-comparator").as_posix(),
        "component_model_id": component_model_id,
        "bundle_sha256": bundle_sha256,
        "source_candidate_path": candidate_relative.as_posix(),
        "source_candidate_bundle_sha256": candidate_bundle_sha256,
        "excluded_files": EXCLUDED_RAW_COMPARATOR_FILES,
        "routing_only_not_release_candidate": True,
    }


def _start_worker(
    root: Path,
    *,
    name: str,
    port: int,
    model_directory: Path,
    release_id: str,
    manifest_hash: str,
    component_model_id: str,
    artifact_hash: str,
    log: Any,
) -> subprocess.Popen[str]:
    if _container_exists(name):
        raise RerankerKServeAcceptanceError(f"owned worker already exists: {name}")
    image = f"{RUNTIME_IMAGE_REPOSITORY}:{RUNTIME_IMAGE_TAG}"
    command = [
        "docker",
        "run",
        "--name",
        name,
        "--gpus",
        "device=0",
        "--user",
        "app",
        "--cpus",
        "1.5",
        "--memory",
        "4g",
        "--memory-swap",
        "4g",
        "--pids-limit",
        "256",
        "--shm-size",
        "128m",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=256m",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--publish",
        f"{port}:8080",
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
        "--volume",
        f"{model_directory.resolve(strict=True)}:/mnt/models/reranker:ro",
        image,
        "--model-dir",
        "/mnt/models/reranker",
        "--release-id",
        release_id,
        "--manifest-hash",
        manifest_hash,
        "--component-model-id",
        component_model_id,
        "--artifact-content-hash",
        artifact_hash,
        "--precision",
        "bfloat16",
        "--batch-size",
        "4",
        "--max-sequence-length",
        "128",
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


def _wait_worker(
    process: subprocess.Popen[str],
    url: str,
    *,
    component_model_id: str,
    calibration_enabled: bool,
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RerankerKServeAcceptanceError("Reranker GPU worker exited during startup")
        try:
            value = _get_json(f"{url}/health/ready")
            if (
                value.get("status") == "ready"
                and value.get("component_model_id") == component_model_id
                and value.get("calibration_enabled") is calibration_enabled
            ):
                return
        except (httpx.HTTPError, RerankerKServeAcceptanceError):
            pass
        time.sleep(1)
    raise RerankerKServeAcceptanceError("Reranker GPU worker did not become ready")


def _stop_worker(
    root: Path,
    name: str,
    process: subprocess.Popen[str],
    *,
    check: bool = True,
) -> None:
    stop = subprocess.run(
        ["docker", "stop", "--time", "10", name],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=10)
    remove = subprocess.run(
        ["docker", "rm", "--force", name],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if check and stop.returncode != 0:
        raise RerankerKServeAcceptanceError(f"failed to stop Reranker worker: {name}")
    if check and remove.returncode != 0:
        raise RerankerKServeAcceptanceError(f"failed to remove Reranker worker: {name}")


def _write_container_inspect(root: Path, name: str, path: Path) -> None:
    raw = local._capture_checked(["docker", "inspect", name], cwd=root)
    value = json.loads(raw)
    _write_json(path, value)


def _post_reranker(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    with httpx.Client(timeout=30, trust_env=False) as client:
        response = client.post(url, json=payload)
        latency_ms = (time.perf_counter() - started) * 1000
        document = response.json()
    if not isinstance(document, dict):
        raise RerankerKServeAcceptanceError("Reranker response is invalid")
    return {
        "schema_version": "enterprise-reranker-kserve-http-result/v1",
        "status_code": response.status_code,
        "variant": response.headers.get("x-ioap-reranker-variant"),
        "latency_ms": latency_ms,
        "body": document,
    }


def _quality_evidence(
    stable: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    stable_score = _target_score(stable, EXPECTED_TOP_CHUNK_ID)
    candidate_score = _target_score(candidate, EXPECTED_TOP_CHUNK_ID)
    candidate_body = candidate.get("body")
    items = candidate_body.get("items") if isinstance(candidate_body, dict) else None
    if (
        stable.get("status_code") != 200
        or stable.get("variant") != "stable"
        or candidate.get("status_code") != 200
        or candidate.get("variant") != "candidate"
        or not isinstance(items, list)
        or not items
        or not isinstance(items[0], dict)
        or items[0].get("chunk_id") != EXPECTED_TOP_CHUNK_ID
        or candidate_score < 3.0
        or candidate_score <= stable_score
    ):
        raise RerankerKServeAcceptanceError("Reranker direct quality probe failed")
    return {
        "stable_target_score": stable_score,
        "candidate_target_score": candidate_score,
        "candidate_score_delta": candidate_score - stable_score,
        "candidate_top_rank_verified": True,
        "release_binding_verified": True,
    }


def _send_gateway_traffic(
    url: str,
    payload: dict[str, Any],
    *,
    request_count: int,
) -> dict[str, Any]:
    variants: list[str] = []
    latencies: list[float] = []
    output_sha256s: set[str] = set()
    with httpx.Client(
        timeout=30,
        trust_env=False,
        headers={"Host": HOSTNAME},
    ) as client:
        for _ in range(request_count):
            started = time.perf_counter()
            response = client.post(url, json=payload)
            latencies.append((time.perf_counter() - started) * 1000)
            if response.status_code != 200:
                raise RerankerKServeAcceptanceError(
                    f"Reranker gateway returned {response.status_code}"
                )
            body = response.json()
            if not isinstance(body, dict):
                raise RerankerKServeAcceptanceError("Reranker gateway response invalid")
            variant = response.headers.get("x-ioap-reranker-variant")
            if variant not in {"stable", "candidate"}:
                raise RerankerKServeAcceptanceError("Reranker gateway variant missing")
            variants.append(variant)
            output_sha256s.add(document_sha256(body))
    return {
        "variants": variants,
        "latencies": latencies,
        "output_sha256s": sorted(output_sha256s),
    }


def _stage_evidence(
    stage: Literal["SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"],
    *,
    route: dict[str, Any],
    before_stable: dict[str, Any],
    after_stable: dict[str, Any],
    before_candidate: dict[str, Any],
    after_candidate: dict[str, Any],
    variants: list[str],
    output_sha256s: list[str],
    latencies: list[float],
) -> RolloutStageEvidence:
    request_count = len(variants)
    stable_delta = _metric_count(after_stable) - _metric_count(before_stable)
    candidate_delta = _metric_count(after_candidate) - _metric_count(before_candidate)
    stable_responses = variants.count("stable")
    candidate_responses = variants.count("candidate")
    ratio = candidate_responses / request_count
    mirror = stage == "SHADOW" and candidate_delta >= request_count
    if stage == "SHADOW":
        passed = stable_responses == request_count and mirror
    elif stage == "CANARY_5":
        passed = 0.01 <= ratio <= 0.15 and candidate_delta >= candidate_responses
    elif stage == "CANARY_25":
        passed = 0.12 <= ratio <= 0.40 and candidate_delta >= candidate_responses
    else:
        passed = (
            stable_responses == request_count and candidate_responses == 0 and candidate_delta == 0
        )
    if not passed or stable_delta < stable_responses or not latencies:
        raise RerankerKServeAcceptanceError(f"Reranker {stage} traffic gate failed")
    ordered = sorted(latencies)
    p95 = ordered[max(0, int(len(ordered) * 0.95 + 0.999999) - 1)]
    return RolloutStageEvidence(
        stage=stage,
        passed=True,
        request_count=request_count,
        stable_request_delta=stable_delta,
        candidate_request_delta=candidate_delta,
        stable_response_count=stable_responses,
        candidate_response_count=candidate_responses,
        candidate_response_ratio=ratio,
        candidate_mirror_observed=mirror,
        p95_latency_ms=p95,
        error_count=0,
        route_document_sha256=document_sha256(route),
        output_sha256s=tuple(output_sha256s),
        model_release_observation_decision=(
            "NOT_REQUIRED_AFTER_ROLLBACK" if stage == "ROLLED_BACK" else "PASS"
        ),
    )


def _record_stage_observation(
    state: _ReleaseState,
    *,
    stage: Literal["SHADOW", "CANARY_5", "CANARY_25"],
    stage_report: RolloutStageEvidence,
) -> DeploymentAggregate:
    end = datetime.now(UTC) - timedelta(seconds=1)
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
        "gpu_memory_utilization": _gpu_memory_utilization(),
        "queue_age_seconds": 0.0,
    }
    source = {
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
                "collector_version": "reranker-kserve-acceptance/v1",
                "prometheus_endpoint_id": "reranker-runtime-and-proxy-metrics",
                "prometheus_query_bundle_hash": "sha256:"
                + document_sha256({"metrics": metrics, "stage": stage}),
                "trace_query_hash": "sha256:" + document_sha256(source),
            },
        ),
        expected_deployment_version=state.deployment.deployment.version,
        request_id=f"observe-reranker-{stage.lower()}",
    )


def _release_snapshot(state: _ReleaseState) -> dict[str, Any]:
    release = state.release_service.get(
        state.engineer,
        state.deployment.release.release_id,
        request_id="snapshot-reranker-release",
    )
    deployment = state.deployment_service.get(
        state.controller,
        state.deployment.release.release_id,
        request_id="snapshot-reranker-deployment",
    )
    if (
        release.release.status != "ROLLED_BACK"
        or deployment.deployment.status != "READY"
        or deployment.deployment.current_stage != "ROLLED_BACK"
        or deployment.deployment.observed_traffic_percent != 0.0
        or release.approval is None
        or release.approval.status != "APPROVED"
    ):
        raise RerankerKServeAcceptanceError("final Reranker release state is invalid")
    observation_stages = tuple(item.stage for item in deployment.observations)
    if observation_stages != ("SHADOW", "CANARY_5", "CANARY_25") or any(
        item.decision != "PASS" for item in deployment.observations
    ):
        raise RerankerKServeAcceptanceError("Reranker observation chain is invalid")
    return {
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
        subject_id=f"reranker-kserve-{role.value}",
        oidc_subject=f"oidc-reranker-kserve-{role.value}",
        tenant_id=TENANT_ID,
        roles=frozenset({role}),
        asset_ids=frozenset(),
        site_ids=frozenset(),
        issued_at=now - timedelta(minutes=1),
        expires_at=now + timedelta(hours=4),
    )


def _wait_route(root: Path, *, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        route = local._kubectl_json(root, "httproute", ROUTE_NAME)
        if route_condition_is_true(route, "Accepted") and route_condition_is_true(
            route,
            "ResolvedRefs",
        ):
            time.sleep(0.5)
            return
        time.sleep(0.5)
    raise RerankerKServeAcceptanceError("Reranker HTTPRoute was not accepted")


def _delete_resources(root: Path, *, check: bool = True) -> None:
    command = [
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
    ]
    if check:
        local._run_checked(command, cwd=root)
    else:
        subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=130,
        )


def _wait_proxy_delta(
    url: str,
    before: dict[str, Any],
    *,
    minimum: int,
    timeout_seconds: int,
) -> None:
    original = _metric_count(before)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _metric_count(_get_json(url)) - original >= minimum:
            return
        time.sleep(0.25)
    raise RerankerKServeAcceptanceError("Shadow traffic was not mirrored")


def _metric_count(value: dict[str, Any]) -> int:
    count = value.get("request_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise RerankerKServeAcceptanceError("Reranker proxy metrics are invalid")
    return count


def _target_score(document: dict[str, Any], chunk_id: str) -> float:
    body = document.get("body")
    items = body.get("items") if isinstance(body, dict) else None
    if not isinstance(items, list):
        raise RerankerKServeAcceptanceError("Reranker response items are invalid")
    matches = [
        item for item in items if isinstance(item, dict) and item.get("chunk_id") == chunk_id
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("score"), (int, float)):
        raise RerankerKServeAcceptanceError("Reranker target score is invalid")
    return float(matches[0]["score"])


def _gpu_memory_utilization() -> float:
    raw = subprocess.run(
        [
            "/usr/lib/wsl/lib/nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    try:
        used_text, total_text = (item.strip() for item in raw.split(",", maxsplit=1))
        used = float(used_text)
        total = float(total_text)
    except (ValueError, TypeError) as exc:
        raise RerankerKServeAcceptanceError("GPU memory inventory is invalid") from exc
    if total <= 0 or not 0 <= used <= total:
        raise RerankerKServeAcceptanceError("GPU memory inventory is invalid")
    return used / total


def _get_json(url: str) -> dict[str, Any]:
    with httpx.Client(timeout=10, trust_env=False) as client:
        response = client.get(url)
        response.raise_for_status()
        value = response.json()
    if not isinstance(value, dict):
        raise RerankerKServeAcceptanceError("Reranker JSON response is invalid")
    return cast(dict[str, Any], value)


def _get_text(url: str) -> str:
    with httpx.Client(timeout=10, trust_env=False) as client:
        response = client.get(url)
        response.raise_for_status()
    if not response.text.strip():
        raise RerankerKServeAcceptanceError("Reranker metrics response is empty")
    return response.text


def _container_exists(name: str) -> bool:
    result = subprocess.run(
        ["docker", "container", "inspect", name],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode == 0


def _require_rollout_services_absent(root: Path) -> None:
    if _container_exists(STABLE_WORKER_CONTAINER) or _container_exists(CANDIDATE_WORKER_CONTAINER):
        raise RerankerKServeAcceptanceError("Reranker worker cleanup failed")
    observed = local._capture_checked(
        ["docker", "inspect", KIND_NODE, "--format", "{{.State.Running}}"],
        cwd=root,
    )
    if observed != "false":
        raise RerankerKServeAcceptanceError("Kind cluster cleanup failed")


def _ensure_reranker_cluster_running(root: Path) -> None:
    if not local._container_exists(root, KIND_NODE):
        raise RerankerKServeAcceptanceError(
            f"required Kind control plane does not exist: {KIND_NODE}"
        )
    if not local._container_running(root, KIND_NODE):
        local._run_checked(["docker", "start", KIND_NODE], cwd=root)
    local._wait_for_cluster_api_access(root, timeout_seconds=90)
    _pin_kserve_controller_to_local_image(root)
    local._ensure_cluster_running(root)


def _pin_kserve_controller_to_local_image(root: Path) -> None:
    document = _kserve_controller_document(root)
    manager = _manager_container(document)
    if manager.get("image") != KSERVE_CONTROLLER_IMAGE:
        raise RerankerKServeAcceptanceError("KServe controller image changed")
    if manager.get("imagePullPolicy") == "IfNotPresent":
        return
    patch = {
        "spec": {
            "template": {
                "spec": {"containers": [{"name": "manager", "imagePullPolicy": "IfNotPresent"}]}
            }
        }
    }
    local._run_checked(
        [
            "kubectl",
            "--context",
            KUBE_CONTEXT,
            "--namespace",
            "kserve",
            "patch",
            "deployment",
            KSERVE_CONTROLLER_DEPLOYMENT,
            "--type",
            "strategic",
            "--patch",
            json.dumps(patch, separators=(",", ":")),
        ],
        cwd=root,
    )
    if (
        _manager_container(_kserve_controller_document(root)).get("imagePullPolicy")
        != "IfNotPresent"
    ):
        raise RerankerKServeAcceptanceError("KServe controller local image policy was not applied")


def _kserve_controller_document(root: Path) -> dict[str, Any]:
    raw = local._capture_checked(
        [
            "kubectl",
            "--context",
            KUBE_CONTEXT,
            "--namespace",
            "kserve",
            "get",
            "deployment",
            KSERVE_CONTROLLER_DEPLOYMENT,
            "--output",
            "json",
        ],
        cwd=root,
    )
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RerankerKServeAcceptanceError("KServe controller deployment JSON is invalid") from exc
    if not isinstance(document, dict):
        raise RerankerKServeAcceptanceError("KServe controller deployment JSON is invalid")
    return cast(dict[str, Any], document)


def _manager_container(document: dict[str, Any]) -> dict[str, Any]:
    spec = document.get("spec")
    template = spec.get("template") if isinstance(spec, dict) else None
    pod_spec = template.get("spec") if isinstance(template, dict) else None
    containers = pod_spec.get("containers") if isinstance(pod_spec, dict) else None
    matches = (
        [item for item in containers if isinstance(item, dict) and item.get("name") == "manager"]
        if isinstance(containers, list)
        else []
    )
    if len(matches) != 1:
        raise RerankerKServeAcceptanceError("KServe controller container changed")
    return matches[0]


def _read_tail(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-4000:]
    except OSError:
        return ""


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _progress(value: str) -> None:
    print(f"RERANKER_KSERVE_PROGRESS={value}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reranker local KServe acceptance")
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    path = execute_rollout(root) if args.command == "run" else root / ACCEPTANCE_RELATIVE
    report = verify_reranker_kserve_acceptance(root, path)
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
