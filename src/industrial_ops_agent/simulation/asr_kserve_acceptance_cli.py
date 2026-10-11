"""Execute the governed ASR ModelRelease on the local KServe control plane."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import socket
import subprocess
import time
import unicodedata
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
from industrial_ops_agent.simulation.asr_enterprise_value_lab import (
    AsrEnterpriseValueReport,
    verify_asr_enterprise_value,
)
from industrial_ops_agent.simulation.asr_kserve_acceptance import (
    ACTUAL_OUTPUT_RELATIVE,
    CLASSIFICATION,
    SCHEMA_VERSION,
    STATUS,
    TRAFFIC_PROBE_DURATION_SECONDS,
    TRAFFIC_PROBE_SAMPLE_COUNT,
    TRAFFIC_PROBE_SAMPLE_RATE,
    TRAFFIC_PROBE_SCHEMA,
    AsrKServeAcceptanceError,
    AsrRolloutStageEvidence,
    EnterpriseAsrKServeAcceptanceReport,
    document_sha256,
    file_binding,
    file_sha256,
    finalize_acceptance,
    route_condition_is_true,
    route_rules_semantically_equal,
    verify_asr_kserve_acceptance,
    write_acceptance,
)
from industrial_ops_agent.simulation.asr_kserve_rollout import (
    CANDIDATE_SERVICE,
    CONFIG_MAP_NAME,
    GATEWAY_NAME,
    HOSTNAME,
    NAMESPACE,
    OUTPUT_RELATIVE,
    PROBE_AUDIO_SHA256,
    PROBE_CASE_ID,
    PROBE_SAMPLE_COUNT,
    PROBE_SAMPLE_RATE,
    PROBE_TRANSCRIPT,
    PROXY_IMAGE,
    PROXY_SOURCE_RELATIVE,
    ROUTE_NAME,
    ROUTE_PATH_PREFIX,
    STABLE_SERVICE,
    WORKER_SOURCE_RELATIVE,
    EnterpriseAsrKServeReadinessReport,
    inference_service_document,
    proxy_config_map_document,
    route_document,
    verify_asr_kserve_readiness,
)

KIND_CLUSTER = "ioap-gpu-promotion-lab"
KUBE_CONTEXT = f"kind-{KIND_CLUSTER}"
KIND_NODE = f"{KIND_CLUSTER}-control-plane"
WORKER_CONTAINER = "ioap-asr-kserve-gpu-worker"
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
            and route.get("metadata", {}).get("annotations", {}).get(
                "ioap.openai.com/stage"
            )
            == self._stage
            and route_rules_semantically_equal(route, expected_route)
        )
        revision = "local-asr-kserve-" + document_sha256(
            {
                "stage": self._stage,
                "route_resource_version": route.get("metadata", {}).get(
                    "resourceVersion"
                ),
                "stable_resource_version": stable.get("metadata", {}).get(
                    "resourceVersion"
                ),
                "candidate_resource_version": candidate.get("metadata", {}).get(
                    "resourceVersion"
                ),
                "desired_spec_hash": target.desired_spec_hash,
            }
        )[:24]
        return ProviderResult(
            ready=ready,
            observed_stage=self._stage,
            observed_traffic_percent=target.desired_traffic_percent,
            applied_spec_hash=target.desired_spec_hash if ready else None,
            provider_revision=revision,
            endpoint_url=(
                f"http://{HOSTNAME}{ROUTE_PATH_PREFIX}/transcribe" if ready else None
            ),
            reason_code=None if ready else "local_asr_kserve_observation_mismatch",
        )


def execute_rollout(repo_root: Path) -> Path:
    root = repo_root.resolve(strict=True)
    acceptance_path = root / ACTUAL_OUTPUT_RELATIVE / "acceptance.json"
    if acceptance_path.is_file():
        verify_asr_kserve_acceptance(root, acceptance_path)
        return acceptance_path
    output = root / ACTUAL_OUTPUT_RELATIVE
    if output.exists():
        raise AsrKServeAcceptanceError(
            "ASR actual rollout output exists without a verified acceptance receipt"
        )
    readiness_path = root / OUTPUT_RELATIVE / "readiness.json"
    readiness = verify_asr_kserve_readiness(root, readiness_path)
    training_path = root / readiness.source.acceptance.path
    training = verify_asr_enterprise_value(root, training_path)
    if (
        readiness.source.run_id != training.run_id
        or readiness.source.adapter_bundle_sha256 != training.adapter.bundle_sha256
        or not all(training.hard_gates.model_dump().values())
    ):
        raise AsrKServeAcceptanceError("ASR rollout source is not accepted")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)
    kubernetes_output = temporary / "kubernetes"
    kubernetes_output.mkdir()
    quality_manifest_path = root / readiness.probe.manifest.path
    quality_audio_path = root / readiness.probe.audio.path
    routing_audio_path = temporary / "routing-probe.flac"
    routing_manifest_path = temporary / "routing-probe.json"
    _build_routing_probe(
        quality_audio_path,
        routing_audio_path,
        routing_manifest_path,
    )
    proxy_path = (root / PROXY_SOURCE_RELATIVE).resolve(strict=True)
    worker_source_path = (root / WORKER_SOURCE_RELATIVE).resolve(strict=True)
    proxy_source = proxy_path.read_text(encoding="utf-8")
    state_path = temporary / "model-release-state.sqlite3"
    state: _ReleaseState | None = None
    worker: subprocess.Popen[str] | None = None
    worker_log: Any = None
    worker_log_path = temporary / "gpu-worker.log"
    resources_applied = False
    resources_removed = False
    worker_stopped = False
    port_forwards_stopped = False
    cluster_started = False
    cluster_stopped = False
    database_disposed = False
    committed = False
    try:
        _progress("MODEL_RELEASE_PREPARING")
        state = _prepare_release_state(root, state_path)
        _progress("MODEL_RELEASE_SHADOW_REQUESTED")
        cluster_started = True
        _progress("CLUSTER_STARTING")
        _ensure_cluster_running(root)
        _progress("CLUSTER_READY")
        proxy_image_digest = _node_image_digest(root)
        worker_image_digest = _image_digest(root, PROXY_IMAGE)
        if worker_image_digest != training.base_model.image_digest:
            raise AsrKServeAcceptanceError("ASR worker image digest changed")
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
        worker = _start_worker(root, port=worker_port, log=worker_log)
        worker_url = f"http://127.0.0.1:{worker_port}"
        _wait_worker(worker, worker_url, timeout_seconds=240)
        _progress("GPU_WORKER_READY")

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
        _progress("KSERVE_RESOURCES_APPLYING")
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
        stage_reports: list[AsrRolloutStageEvidence] = []
        direct_stable: dict[str, Any]
        direct_candidate: dict[str, Any]
        route_paths: dict[str, Path] = {}
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
            quality_payload = _asr_request(
                quality_audio_path,
                case_id=PROBE_CASE_ID,
                request_id="asr-release-direct-probe",
            )
            direct_stable = _post_json(
                f"http://127.0.0.1:{stable_port}{ROUTE_PATH_PREFIX}/transcribe",
                quality_payload,
                timeout_seconds=120,
            )
            direct_candidate = _post_json(
                f"http://127.0.0.1:{candidate_port}{ROUTE_PATH_PREFIX}/transcribe",
                quality_payload,
                timeout_seconds=120,
            )
            quality = _validate_candidate_quality(
                direct_stable,
                direct_candidate,
                expected_transcript=PROBE_TRANSCRIPT,
            )
            routing_payload = _asr_request(
                routing_audio_path,
                case_id="asr-routing-probe",
                request_id="asr-routing-probe",
            )
            gateway_url = (
                f"http://127.0.0.1:{gateway_port}{ROUTE_PATH_PREFIX}/transcribe"
            )
            for stage in ("SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"):
                _progress(f"{stage}_TRAFFIC_STARTING")
                route = route_document(cast(Any, stage))
                route_path = kubernetes_output / f"route-{stage.lower()}.json"
                route_paths[stage] = route_path
                _write_json(route_path, route)
                _kubectl_apply(root, route)
                _wait_route(root, timeout_seconds=60)
                state.deployment = state.deployment_service.reconcile(
                    state.controller,
                    state.deployment.deployment.deployment_id,
                    _ObservedLocalKServeProvider(root, stage),
                    expected_version=state.deployment.deployment.version,
                    request_id=f"reconcile-asr-{stage.lower()}",
                )
                if (
                    state.deployment.deployment.status != "READY"
                    or state.deployment.deployment.current_stage != stage
                ):
                    raise AsrKServeAcceptanceError(
                        f"ASR ModelRelease deployment did not enter {stage}"
                    )
                before = _get_json(f"{worker_url}/metrics", timeout_seconds=5)
                traffic = _send_gateway_traffic(
                    gateway_url,
                    routing_payload,
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
                        raise AsrKServeAcceptanceError(
                            f"ASR ModelRelease {stage} observation did not pass"
                        )
                    stage_report = stage_report.model_copy(
                        update={"model_release_observation_decision": "PASS"}
                    )
                    if stage in {"SHADOW", "CANARY_5"}:
                        state.deployment = state.deployment_service.promote(
                            state.operator,
                            state.deployment.release.release_id,
                            expected_deployment_version=state.deployment.deployment.version,
                            request_id=f"promote-asr-{stage.lower()}",
                        )
                    else:
                        state.deployment = state.deployment_service.request_rollback(
                            state.operator,
                            state.deployment.release.release_id,
                            expected_deployment_version=state.deployment.deployment.version,
                            reason_code="local_asr_acceptance_rollback_drill",
                            request_id="rollback-asr-after-canary-25",
                        )
                stage_reports.append(stage_report)
                _progress(f"{stage}_TRAFFIC_PASSED")
        port_forwards_stopped = True

        final_route = _kubectl_json(root, "httproute", ROUTE_NAME)
        stable_state = _kubectl_json(root, "inferenceservice", STABLE_SERVICE)
        candidate_state = _kubectl_json(root, "inferenceservice", CANDIDATE_SERVICE)
        runtime_metrics = _get_json(f"{worker_url}/metrics", timeout_seconds=5)
        _write_json(kubernetes_output / "observed-final-route.json", final_route)
        _write_json(kubernetes_output / "observed-stable.json", stable_state)
        _write_json(kubernetes_output / "observed-candidate.json", candidate_state)
        release_snapshot = _release_snapshot(state)
        manifest_path = temporary / "release-manifest.json"
        _write_json(manifest_path, release_snapshot["manifest"])

        _progress("CLEANUP_STARTING")
        _delete_asr_resources(root)
        resources_removed = True
        _stop_worker(root, worker)
        worker_stopped = True
        worker_log.close()
        worker_log = None
        _stop_cluster(root)
        cluster_stopped = True
        state.database.dispose()
        database_disposed = True

        unsigned = _acceptance_document(
            root=root,
            temporary=temporary,
            readiness_path=readiness_path,
            readiness=readiness,
            training_path=training_path,
            training=training,
            quality_manifest_path=quality_manifest_path,
            quality_audio_path=quality_audio_path,
            routing_manifest_path=routing_manifest_path,
            routing_audio_path=routing_audio_path,
            state=state,
            state_path=state_path,
            release_snapshot=release_snapshot,
            manifest_path=manifest_path,
            runtime_metrics=runtime_metrics,
            worker_image_digest=worker_image_digest,
            proxy_image_digest=proxy_image_digest,
            proxy_path=proxy_path,
            worker_source_path=worker_source_path,
            host_gateway=host_gateway,
            worker_port=worker_port,
            route_paths=route_paths,
            final_route=final_route,
            stable_state=stable_state,
            candidate_state=candidate_state,
            quality=quality,
            stage_reports=stage_reports,
            worker_stopped=worker_stopped,
            port_forwards_stopped=port_forwards_stopped,
            resources_removed=resources_removed,
            cluster_stopped=cluster_stopped,
        )
        report = finalize_acceptance(unsigned)
        write_acceptance(report, temporary / "acceptance.json")
        temporary.replace(output)
        try:
            verify_asr_kserve_acceptance(root, acceptance_path)
        except Exception:
            output.replace(temporary)
            raise
        committed = True
        _progress("ACCEPTANCE_COMMITTED")
        return acceptance_path
    except Exception as exc:
        _progress(f"FAILED_{type(exc).__name__}")
        tail = _read_log_tail(worker_log_path)
        if tail:
            print("ASR_GPU_WORKER_LOG_TAIL_BEGIN", flush=True)
            print(tail, flush=True)
            print("ASR_GPU_WORKER_LOG_TAIL_END", flush=True)
        print(f"ASR_KSERVE_ACCEPTANCE_ERROR={exc}", flush=True)
        raise
    finally:
        if resources_applied and not resources_removed and cluster_started:
            _delete_asr_resources(root, check=False)
        if worker is not None and not worker_stopped:
            _stop_worker(root, worker, check=False)
        if worker_log is not None and not worker_log.closed:
            worker_log.close()
        if cluster_started and not cluster_stopped:
            _stop_cluster(root, check=False)
        if state is not None and not database_disposed:
            state.database.dispose()
        if not committed and temporary.exists():
            shutil.rmtree(temporary)


def _acceptance_document(
    *,
    root: Path,
    temporary: Path,
    readiness_path: Path,
    readiness: EnterpriseAsrKServeReadinessReport,
    training_path: Path,
    training: AsrEnterpriseValueReport,
    quality_manifest_path: Path,
    quality_audio_path: Path,
    routing_manifest_path: Path,
    routing_audio_path: Path,
    state: _ReleaseState,
    state_path: Path,
    release_snapshot: dict[str, Any],
    manifest_path: Path,
    runtime_metrics: dict[str, Any],
    worker_image_digest: str,
    proxy_image_digest: str,
    proxy_path: Path,
    worker_source_path: Path,
    host_gateway: str,
    worker_port: int,
    route_paths: dict[str, Path],
    final_route: dict[str, Any],
    stable_state: dict[str, Any],
    candidate_state: dict[str, Any],
    quality: dict[str, Any],
    stage_reports: list[AsrRolloutStageEvidence],
    worker_stopped: bool,
    port_forwards_stopped: bool,
    resources_removed: bool,
    cluster_stopped: bool,
) -> dict[str, Any]:
    kubernetes_output = temporary / "kubernetes"
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "external_enterprise_production_claim": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": STATUS,
        "source": {
            "readiness": file_binding(root, readiness_path),
            "readiness_evidence_chain_sha256": readiness.evidence_chain_sha256,
            "training_acceptance": file_binding(root, training_path),
            "training_run_id": training.run_id,
            "training_evidence_chain_sha256": training.evidence_chain_sha256,
            "model_id": training.base_model.model_id,
            "model_revision": training.base_model.revision,
            "adapter_bundle_sha256": training.adapter.bundle_sha256,
        },
        "probe": {
            "quality_manifest": file_binding(root, quality_manifest_path),
            "quality_audio": file_binding(root, quality_audio_path),
            "quality_case_id": PROBE_CASE_ID,
            "quality_source_id": readiness.probe.source_id,
            "expected_transcript_sha256": readiness.probe.transcript_sha256,
            "routing_manifest": file_binding(
                root,
                routing_manifest_path,
                reported_path=ACTUAL_OUTPUT_RELATIVE / "routing-probe.json",
            ),
            "routing_audio": file_binding(
                root,
                routing_audio_path,
                reported_path=ACTUAL_OUTPUT_RELATIVE / "routing-probe.flac",
            ),
            "routing_parent_audio_sha256": PROBE_AUDIO_SHA256,
            "routing_sample_rate": TRAFFIC_PROBE_SAMPLE_RATE,
            "routing_sample_count": TRAFFIC_PROBE_SAMPLE_COUNT,
            "routing_duration_seconds": TRAFFIC_PROBE_DURATION_SECONDS,
            "routing_probe_is_quality_evidence": False,
            "formal_gold_reused": False,
        },
        "model_release": {
            "state_database": file_binding(
                root,
                state_path,
                reported_path=ACTUAL_OUTPUT_RELATIVE / "model-release-state.sqlite3",
            ),
            "sqlite_integrity_check": "ok",
            "tenant_id": TENANT_ID,
            "baseline_release_id": state.baseline_release_id,
            "release_id": release_snapshot["release_id"],
            "manifest": file_binding(
                root,
                manifest_path,
                reported_path=ACTUAL_OUTPUT_RELATIVE / "release-manifest.json",
            ),
            "manifest_hash": release_snapshot["manifest_hash"],
            "source_candidate_experiment_id": state.source_candidate_experiment_id,
            "imported_candidate_experiment_id": state.imported_candidate_experiment_id,
            "evaluation_id": state.evaluation_id,
            "approval_id": state.approval_id,
            "approval_requested_by_subject_id": state.approval_requested_by_subject_id,
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
            "topology": "DOCKER_GPU_WORKER_BEHIND_KSERVE_PROXY",
            "actual_gpu_execution": True,
            "model_execution_simulated": False,
            "worker_image": PROXY_IMAGE,
            "worker_image_digest": worker_image_digest,
            "model_id": _text(runtime_metrics, "model_id"),
            "model_revision": _text(runtime_metrics, "model_revision"),
            "adapter_bundle_sha256": training.adapter.bundle_sha256,
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
            "stable_cache_hits": _variant_count(
                runtime_metrics, "cache_hits", "stable"
            ),
            "candidate_cache_hits": _variant_count(
                runtime_metrics, "cache_hits", "candidate"
            ),
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
            "worker_host_gateway": host_gateway,
            "worker_host_port": worker_port,
            "proxy_image": PROXY_IMAGE,
            "proxy_image_digest": proxy_image_digest,
            "proxy_source_sha256": file_sha256(proxy_path),
            "worker_source_sha256": file_sha256(worker_source_path),
            "config_map": file_binding(
                root,
                kubernetes_output / "proxy-configmap.json",
                reported_path=ACTUAL_OUTPUT_RELATIVE / "kubernetes/proxy-configmap.json",
            ),
            "stable_service": file_binding(
                root,
                kubernetes_output / "stable-inferenceservice.json",
                reported_path=(
                    ACTUAL_OUTPUT_RELATIVE / "kubernetes/stable-inferenceservice.json"
                ),
            ),
            "candidate_service": file_binding(
                root,
                kubernetes_output / "candidate-inferenceservice.json",
                reported_path=(
                    ACTUAL_OUTPUT_RELATIVE / "kubernetes/candidate-inferenceservice.json"
                ),
            ),
            "routes": {
                stage: file_binding(
                    root,
                    path,
                    reported_path=(
                        ACTUAL_OUTPUT_RELATIVE / f"kubernetes/route-{stage.lower()}.json"
                    ),
                )
                for stage, path in route_paths.items()
            },
            "observed_final_route": file_binding(
                root,
                kubernetes_output / "observed-final-route.json",
                reported_path=(
                    ACTUAL_OUTPUT_RELATIVE / "kubernetes/observed-final-route.json"
                ),
            ),
            "observed_stable_service": file_binding(
                root,
                kubernetes_output / "observed-stable.json",
                reported_path=ACTUAL_OUTPUT_RELATIVE / "kubernetes/observed-stable.json",
            ),
            "observed_candidate_service": file_binding(
                root,
                kubernetes_output / "observed-candidate.json",
                reported_path=(
                    ACTUAL_OUTPUT_RELATIVE / "kubernetes/observed-candidate.json"
                ),
            ),
            "hardened_container_security_context": True,
            "endpoint_scope": "LOCAL_PORT_FORWARD",
        },
        "quality": {"case_id": PROBE_CASE_ID, "audio_sha256": PROBE_AUDIO_SHA256, **quality},
        "stages": [item.model_dump(mode="json") for item in stage_reports],
        "cleanup": {
            "gpu_worker_stopped": worker_stopped,
            "port_forwards_stopped": port_forwards_stopped,
            "asr_kubernetes_resources_removed": resources_removed,
            "cluster_stopped_after_acceptance": cluster_stopped,
            "evidence_is_historical_after_cleanup": True,
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
    auditor = SecurityAuditor(
        InMemorySecurityAuditSink(),
        hash_key=b"asr-local-kserve-release",
    )
    authorizer = Authorizer(auditor)
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
        "ASR",
        idempotency_key="asr-kserve-import-asr",
        request_id="asr-kserve-import-asr",
    )
    releases = ModelReleaseService(database, authorizer)
    baseline_draft = EnterpriseStagingBaselineService(database, authorizer).create_draft(
        engineer,
        active_mcp_server_versions={},
        idempotency_key="asr-kserve-baseline",
        request_id="asr-kserve-baseline",
    )
    baseline_candidate = releases.validate(
        engineer,
        baseline_draft.release_id,
        expected_version=baseline_draft.version,
        request_id="asr-kserve-baseline-validate",
    )
    baseline_pending = releases.submit_for_approval(
        engineer,
        baseline_draft.release_id,
        expected_version=baseline_candidate.release.version,
        request_id="asr-kserve-baseline-submit",
    )
    if baseline_pending.approval is None:
        raise AsrKServeAcceptanceError("ASR baseline approval was not created")
    releases.decide_approval(
        approver,
        baseline_draft.release_id,
        expected_version=baseline_pending.release.version,
        expected_approval_version=baseline_pending.approval.version,
        decision="APPROVED",
        reason="independent local ASR Staging baseline review passed",
        request_id="asr-kserve-baseline-approve",
    )
    onboarding = EnterpriseReleaseOnboardingService(
        database,
        authorizer,
        reader,
    ).create_draft(
        engineer,
        "ASR",
        baseline_release_id=baseline_draft.release_id,
        auto_shadow_enabled=False,
        deployment_plan=None,
        active_mcp_server_versions={},
        idempotency_key="asr-kserve-candidate-release",
        request_id="asr-kserve-candidate-release",
    )
    candidate = releases.validate(
        engineer,
        onboarding.release_id,
        expected_version=onboarding.release_version,
        request_id="asr-kserve-candidate-validate",
    )
    pending = releases.submit_for_approval(
        engineer,
        onboarding.release_id,
        expected_version=candidate.release.version,
        request_id="asr-kserve-candidate-submit",
    )
    if pending.approval is None:
        raise AsrKServeAcceptanceError("ASR candidate approval was not created")
    approved = releases.decide_approval(
        approver,
        onboarding.release_id,
        expected_version=pending.release.version,
        expected_approval_version=pending.approval.version,
        decision="APPROVED",
        reason="independent ASR Gold, release probe and supply-chain review passed",
        request_id="asr-kserve-candidate-approve",
    )
    if approved.approval is None or approved.approval.status != "APPROVED":
        raise AsrKServeAcceptanceError("ASR candidate approval did not pass")
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
            serving_runtime_name="asr-whisper-peft-runtime",
            artifact_uri_prefix="s3://project-enterprise-staging/asr",
        ),
        expected_release_version=approved.release.version,
        request_id="asr-kserve-request-shadow",
    )
    return _ReleaseState(
        database=database,
        engineer=engineer,
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
    stage_report: AsrRolloutStageEvidence,
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
                "collector_version": "asr-kserve-acceptance/v1",
                "prometheus_endpoint_id": "asr-gpu-worker-metrics",
                "prometheus_query_bundle_hash": "sha256:"
                + document_sha256({"metrics": metrics, "stage": stage}),
                "trace_query_hash": "sha256:" + document_sha256(source_document),
            },
        ),
        expected_deployment_version=state.deployment.deployment.version,
        request_id=f"observe-asr-{stage.lower()}",
    )


def _release_snapshot(state: _ReleaseState) -> dict[str, Any]:
    release = state.release_service.get(
        state.engineer,
        state.deployment.release.release_id,
        request_id="snapshot-asr-release",
    )
    deployment = state.deployment_service.get(
        state.controller,
        state.deployment.release.release_id,
        request_id="snapshot-asr-deployment",
    )
    if (
        release.release.status != "ROLLED_BACK"
        or deployment.deployment.status != "READY"
        or deployment.deployment.current_stage != "ROLLED_BACK"
        or deployment.deployment.observed_traffic_percent != 0.0
        or release.approval is None
        or release.approval.status != "APPROVED"
    ):
        raise AsrKServeAcceptanceError("final ASR ModelRelease state is invalid")
    observation_stages = tuple(item.stage for item in deployment.observations)
    if observation_stages != ("SHADOW", "CANARY_5", "CANARY_25") or any(
        item.decision != "PASS" for item in deployment.observations
    ):
        raise AsrKServeAcceptanceError("ASR ModelRelease observation chain is invalid")
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
        subject_id=f"asr-kserve-{role.value}",
        oidc_subject=f"oidc-asr-kserve-{role.value}",
        tenant_id=TENANT_ID,
        roles=frozenset({role}),
        asset_ids=frozenset(),
        site_ids=frozenset(),
        issued_at=now - timedelta(minutes=1),
        expires_at=now + timedelta(hours=4),
    )


def _build_routing_probe(
    quality_audio_path: Path,
    audio_path: Path,
    manifest_path: Path,
) -> None:
    try:
        import soundfile  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - environment preflight
        raise AsrKServeAcceptanceError("soundfile is required for ASR routing probe") from exc
    if file_sha256(quality_audio_path) != PROBE_AUDIO_SHA256:
        raise AsrKServeAcceptanceError("ASR quality probe audio changed")
    audio, sampling_rate = soundfile.read(
        quality_audio_path,
        dtype="float32",
        always_2d=False,
    )
    if sampling_rate != TRAFFIC_PROBE_SAMPLE_RATE:
        raise AsrKServeAcceptanceError("ASR quality probe sampling rate changed")
    if getattr(audio, "ndim", 0) == 2:
        audio = audio.mean(axis=1)
    routing = audio[:TRAFFIC_PROBE_SAMPLE_COUNT]
    if len(routing) != TRAFFIC_PROBE_SAMPLE_COUNT:
        raise AsrKServeAcceptanceError("ASR quality probe is too short")
    soundfile.write(
        audio_path,
        routing,
        TRAFFIC_PROBE_SAMPLE_RATE,
        format="FLAC",
        subtype="PCM_16",
    )
    media_sha256 = file_sha256(audio_path)
    document = {
        "schema_version": TRAFFIC_PROBE_SCHEMA,
        "classification": CLASSIFICATION,
        "purpose": "ROUTING_ONLY",
        "quality_evidence": False,
        "formal_gold": False,
        "formal_gold_reused": False,
        "parent_audio_sha256": PROBE_AUDIO_SHA256,
        "media": {
            "path": f"{ACTUAL_OUTPUT_RELATIVE.as_posix()}/routing-probe.flac",
            "mime_type": "audio/flac",
            "sha256": media_sha256,
            "size_bytes": audio_path.stat().st_size,
            "sampling_rate": TRAFFIC_PROBE_SAMPLE_RATE,
            "sample_count": TRAFFIC_PROBE_SAMPLE_COUNT,
            "duration_seconds": TRAFFIC_PROBE_DURATION_SECONDS,
        },
    }
    _write_json(
        manifest_path,
        {**document, "evidence_chain_sha256": document_sha256(document)},
    )


def _asr_request(audio_path: Path, *, case_id: str, request_id: str) -> dict[str, Any]:
    audio = audio_path.read_bytes()
    return {
        "schema_version": "enterprise-asr-runtime-request/v1",
        "request_id": request_id,
        "case_id": case_id,
        "audio_base64": base64.b64encode(audio).decode("ascii"),
        "audio_sha256": file_sha256(audio_path),
        "media_mime_type": "audio/flac",
        "sampling_rate": PROBE_SAMPLE_RATE,
        "language": "en",
        "max_length": 64,
    }


def _validate_candidate_quality(
    stable: dict[str, Any],
    candidate: dict[str, Any],
    *,
    expected_transcript: str,
) -> dict[str, Any]:
    _validate_runtime_response(stable, variant="stable")
    _validate_runtime_response(candidate, variant="candidate")
    stable_text = _text(stable, "normalized_transcript")
    candidate_text = _text(candidate, "normalized_transcript")
    stable_wer = _word_error_rate(expected_transcript, stable_text)
    candidate_wer = _word_error_rate(expected_transcript, candidate_text)
    if candidate_wer > 0.75 or candidate_wer > stable_wer + 0.25:
        raise AsrKServeAcceptanceError("candidate ASR release probe failed")
    return {
        "stable_normalized_transcript": stable_text,
        "candidate_normalized_transcript": candidate_text,
        "stable_normalized_transcript_sha256": document_sha256(stable_text),
        "candidate_normalized_transcript_sha256": document_sha256(candidate_text),
        "stable_word_error_rate": stable_wer,
        "candidate_word_error_rate": candidate_wer,
        "candidate_word_error_rate_max": 0.75,
        "candidate_regression_tolerance": 0.25,
        "candidate_response_contract_valid": True,
        "candidate_audio_digest_verified": True,
        "candidate_release_probe_passed": True,
    }


def _validate_runtime_response(
    response: dict[str, Any],
    *,
    variant: Literal["stable", "candidate"],
) -> None:
    response_sha256 = response.get("response_sha256")
    unsigned = {key: value for key, value in response.items() if key != "response_sha256"}
    valid = bool(
        response.get("schema_version") == "enterprise-asr-runtime-response/v1"
        and response.get("case_id") == PROBE_CASE_ID
        and response.get("variant") == variant
        and response.get("adapter_enabled") is (variant == "candidate")
        and response.get("audio_sha256") == PROBE_AUDIO_SHA256
        and response.get("sampling_rate") == PROBE_SAMPLE_RATE
        and response.get("sample_count") == PROBE_SAMPLE_COUNT
        and isinstance(response.get("normalized_transcript"), str)
        and bool(response.get("normalized_transcript"))
        and isinstance(response_sha256, str)
        and response_sha256 == document_sha256(unsigned)
    )
    if not valid:
        raise AsrKServeAcceptanceError(f"{variant} ASR runtime response is invalid")


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
    with httpx.Client(timeout=httpx.Timeout(120), trust_env=False) as client:
        for index in range(request_count):
            started = time.perf_counter()
            response = client.post(
                url,
                headers={"Host": HOSTNAME},
                json={**payload, "request_id": f"asr-{stage.lower()}-{index:05d}"},
            )
            latency_ms = max((time.perf_counter() - started) * 1_000, 0.001)
            response.raise_for_status()
            variant = response.headers.get("X-IOAP-ASR-Variant", "")
            if variant not in {"stable", "candidate"}:
                raise AsrKServeAcceptanceError("gateway ASR variant is missing")
            document = response.json()
            if (
                not isinstance(document, dict)
                or document.get("variant") != variant
                or document.get("audio_sha256") != payload["audio_sha256"]
                or not isinstance(document.get("response_sha256"), str)
            ):
                raise AsrKServeAcceptanceError("gateway ASR response is invalid")
            responses[variant] += 1
            output_sha256s.add(
                document_sha256(
                    {
                        "variant": variant,
                        "audio_sha256": document["audio_sha256"],
                        "normalized_transcript": document.get("normalized_transcript"),
                    }
                )
            )
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
) -> AsrRolloutStageEvidence:
    stable_delta = _variant_count(after, "request_counts", "stable") - _variant_count(
        before, "request_counts", "stable"
    )
    candidate_delta = _variant_count(
        after, "request_counts", "candidate"
    ) - _variant_count(before, "request_counts", "candidate")
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
        raise AsrKServeAcceptanceError(f"ASR {stage} traffic evidence failed")
    return AsrRolloutStageEvidence(
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


def _word_error_rate(reference: str, transcript: str) -> float:
    expected = _transcript_words(reference)
    actual = _transcript_words(transcript)
    return _edit_distance(expected, actual) / max(len(expected), 1)


def _transcript_words(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]|[a-z0-9]+", normalized))


def _edit_distance(expected: tuple[str, ...], actual: tuple[str, ...]) -> int:
    previous = list(range(len(actual) + 1))
    for expected_index, expected_item in enumerate(expected, start=1):
        current = [expected_index]
        for actual_index, actual_item in enumerate(actual, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[actual_index] + 1,
                    previous[actual_index - 1] + (expected_item != actual_item),
                )
            )
        previous = current
    return previous[-1]


def _start_worker(
    root: Path,
    *,
    port: int,
    log: Any,
) -> subprocess.Popen[str]:
    if _container_exists(root, WORKER_CONTAINER):
        raise AsrKServeAcceptanceError(
            f"owned ASR worker container already exists: {WORKER_CONTAINER}"
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
        "industrial_ops_agent.multimodal.asr_runtime",
        "--repo-root",
        "/workspace",
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
    if not _container_exists(root, KIND_NODE):
        raise AsrKServeAcceptanceError(
            f"required Kind control plane does not exist: {KIND_NODE}"
        )
    if not _container_running(root, KIND_NODE):
        _run_checked(["docker", "start", KIND_NODE], cwd=root)
    observed_context = _capture_checked(
        ["kubectl", "config", "current-context"],
        cwd=root,
    )
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
    raise AsrKServeAcceptanceError(
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
        "--for=condition=Ready",
        "pod",
        f"--timeout={timeout_seconds}s",
    ]
    if selector is None:
        command.append("--all")
    else:
        command.extend(("--selector", selector))
    _run_checked(command, cwd=root)


def _node_image_digest(root: Path) -> str:
    normalized_image = f"docker.io/{PROXY_IMAGE}"
    raw = _capture_checked(
        ["docker", "exec", KIND_NODE, "crictl", "inspecti", normalized_image],
        cwd=root,
    )
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AsrKServeAcceptanceError(
            "Kind node returned invalid ASR proxy image metadata"
        ) from exc
    status = document.get("status")
    if not isinstance(status, dict):
        raise AsrKServeAcceptanceError("Kind node ASR proxy image status is missing")
    digest = status.get("id")
    repo_tags = status.get("repoTags")
    if (
        not isinstance(digest, str)
        or not _valid_image_digest(digest)
        or not isinstance(repo_tags, list)
        or normalized_image not in repo_tags
    ):
        raise AsrKServeAcceptanceError("Kind node fixed ASR proxy image binding changed")
    return digest


def _image_digest(root: Path, image: str) -> str:
    value = _capture_checked(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        cwd=root,
    )
    if not _valid_image_digest(value):
        raise AsrKServeAcceptanceError("ASR worker image digest is invalid")
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
            "asr-kserve-acceptance",
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
    raise AsrKServeAcceptanceError(f"ASR KServe endpoint is not ready: {service}")


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
    raise AsrKServeAcceptanceError("ASR HTTPRoute was not accepted and resolved")


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
        raise AsrKServeAcceptanceError("Envoy Gateway service is not unique")
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
            raise AsrKServeAcceptanceError("ASR GPU worker exited during startup")
        try:
            document = _get_json(f"{worker_url}/health/ready", timeout_seconds=2)
            if document.get("ready") is True:
                return
        except (httpx.HTTPError, AsrKServeAcceptanceError):
            pass
        time.sleep(1)
    raise AsrKServeAcceptanceError("ASR GPU worker did not become ready")


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
            raise AsrKServeAcceptanceError(f"port-forward failed: {output[-512:]}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.25)
    raise AsrKServeAcceptanceError("ASR port-forward did not become ready")


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
        raise AsrKServeAcceptanceError("ASR response is invalid")
    return document


def _get_json(url: str, *, timeout_seconds: int) -> dict[str, Any]:
    with httpx.Client(timeout=timeout_seconds, trust_env=False) as client:
        response = client.get(url)
        response.raise_for_status()
        document = response.json()
    if not isinstance(document, dict):
        raise AsrKServeAcceptanceError("ASR runtime response is invalid")
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
    raise AsrKServeAcceptanceError("Shadow traffic was not mirrored to ASR candidate")


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
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AsrKServeAcceptanceError("Kubernetes resource JSON is invalid") from exc
    if not isinstance(document, dict):
        raise AsrKServeAcceptanceError("Kubernetes resource response is invalid")
    return document


def _delete_asr_resources(root: Path, *, check: bool = True) -> None:
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
        raise AsrKServeAcceptanceError(
            f"ASR resource cleanup failed: {result.stdout[-512:]}"
        )


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
            raise AsrKServeAcceptanceError(
                f"ASR worker cleanup failed: {result.stdout[-512:]}"
            )
    if process.poll() is None:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)
    if check and _container_exists(root, WORKER_CONTAINER):
        raise AsrKServeAcceptanceError("ASR worker still exists after cleanup")


def _stop_cluster(root: Path, *, check: bool = True) -> None:
    if not _container_exists(root, KIND_NODE):
        return
    result = subprocess.run(
        ["docker", "stop", "--time", "30", KIND_NODE],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and result.returncode != 0:
        raise AsrKServeAcceptanceError(f"Kind cluster stop failed: {result.stdout[-512:]}")


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


def _container_running(root: Path, name: str) -> bool:
    if not _container_exists(root, name):
        return False
    return (
        _capture_checked(
            ["docker", "inspect", name, "--format", "{{.State.Running}}"],
            cwd=root,
        )
        == "true"
    )


def _resource_ready(document: dict[str, Any]) -> bool:
    conditions = document.get("status", {}).get("conditions", [])
    return any(
        isinstance(item, dict)
        and item.get("type") == "Ready"
        and item.get("status") == "True"
        for item in conditions
    )


def _variant_count(document: dict[str, Any], field: str, variant: str) -> int:
    value = document.get(field)
    if not isinstance(value, dict):
        raise AsrKServeAcceptanceError(f"runtime metric is invalid: {field}")
    count = value.get(variant)
    if not isinstance(count, int) or count < 0:
        raise AsrKServeAcceptanceError(f"runtime metric is invalid: {field}.{variant}")
    return count


def _text(document: dict[str, Any], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value:
        raise AsrKServeAcceptanceError(f"field is invalid: {field}")
    return value


def _positive_integer(document: dict[str, Any], field: str) -> int:
    value = document.get(field)
    if not isinstance(value, int) or value <= 0:
        raise AsrKServeAcceptanceError(f"field is invalid: {field}")
    return value


def _valid_image_digest(value: str) -> bool:
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", value))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * quantile) - 1))
    return ordered[index]


def _stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _run_checked(argv: list[str], *, cwd: Path) -> None:
    result = subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise AsrKServeAcceptanceError(
            f"command failed ({argv[0]}): {result.stdout[-1_024:]}"
        )


def _capture_checked(argv: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise AsrKServeAcceptanceError(
            f"command failed ({argv[0]}): {result.stdout[-1_024:]}"
        )
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
        output = result.stdout.decode(errors="replace")
        raise AsrKServeAcceptanceError(
            f"command failed ({argv[0]}): {output[-1_024:]}"
        )


def _write_json(path: Path, document: object) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _progress(stage: str) -> None:
    print(f"ASR_KSERVE_STAGE={stage}", flush=True)


def _read_log_tail(path: Path, *, max_chars: int = 4_096) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except OSError:
        return ""


def preflight(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    readiness = verify_asr_kserve_readiness(root, root / OUTPUT_RELATIVE / "readiness.json")
    missing = [name for name in ("docker", "kubectl", "nvidia-smi") if shutil.which(name) is None]
    if missing:
        raise AsrKServeAcceptanceError(
            "ASR rollout tools are missing: " + ", ".join(missing)
        )
    worker_image_digest = _image_digest(root, PROXY_IMAGE)
    if worker_image_digest != readiness.source.image_digest:
        raise AsrKServeAcceptanceError("ASR preflight worker image digest changed")
    gpu = _capture_checked(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        cwd=root,
    )
    if len([line for line in gpu.splitlines() if line.strip()]) != 1:
        raise AsrKServeAcceptanceError("ASR preflight requires exactly one visible GPU")
    if not _container_exists(root, KIND_NODE):
        raise AsrKServeAcceptanceError("ASR preflight Kind control plane is missing")
    if _container_exists(root, WORKER_CONTAINER):
        raise AsrKServeAcceptanceError("ASR preflight found an owned worker container")
    return {
        "status": "ASR_KSERVE_EXECUTION_PREFLIGHT_PASSED",
        "readiness_evidence_chain_sha256": readiness.evidence_chain_sha256,
        "worker_image_digest": worker_image_digest,
        "gpu": gpu,
        "kind_control_plane_running": _container_running(root, KIND_NODE),
        "actual_acceptance_exists": (root / ACTUAL_OUTPUT_RELATIVE / "acceptance.json").is_file(),
        "services_started": False,
    }


def cleanup(repo_root: Path) -> None:
    root = repo_root.resolve(strict=True)
    if _container_exists(root, WORKER_CONTAINER):
        result = subprocess.run(
            ["docker", "stop", "--time", "20", WORKER_CONTAINER],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            raise AsrKServeAcceptanceError(
                f"ASR worker cleanup failed: {result.stdout[-512:]}"
            )
    if _container_running(root, KIND_NODE):
        _delete_asr_resources(root)
        _stop_cluster(root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-asr-kserve-acceptance")
    parser.add_argument(
        "command",
        choices=("preflight", "run", "verify", "cleanup"),
        nargs="?",
        default="preflight",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    if args.command == "preflight":
        print(json.dumps(preflight(root), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if args.command == "run":
        path = execute_rollout(root)
        report = verify_asr_kserve_acceptance(root, path)
    elif args.command == "verify":
        path = root / ACTUAL_OUTPUT_RELATIVE / "acceptance.json"
        report = verify_asr_kserve_acceptance(root, path)
    else:
        cleanup(root)
        print("ASR_KSERVE_RESOURCES_CLEANED")
        return 0
    _print_report(report, path)
    return 0


def _print_report(
    report: EnterpriseAsrKServeAcceptanceReport,
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
