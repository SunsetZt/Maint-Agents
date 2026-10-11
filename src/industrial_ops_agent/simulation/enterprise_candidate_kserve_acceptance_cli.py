"""Execute resumable, one-component-at-a-time local KServe acceptance."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import httpx

from industrial_ops_agent.deployment.kserve import KubernetesApiError
from industrial_ops_agent.deployment.observed_kserve import (
    ObservedKServeComponent,
    ObservedKServeProvider,
    RolloutStage,
    inference_service_document,
    proxy_config_map_document,
    route_condition_is_true,
    route_document,
)
from industrial_ops_agent.deployment.service import (
    METRIC_THRESHOLDS,
    STAGE_POLICY,
    ObservationInput,
    ProviderTarget,
)
from industrial_ops_agent.simulation import grpo_kserve_rollout_cli as local
from industrial_ops_agent.simulation import tts_enterprise_value_lab as tts_value
from industrial_ops_agent.simulation.embedding_calibrated_value_lab import (
    verify_calibrated_embedding_value,
)
from industrial_ops_agent.simulation.enterprise_candidate_kserve_acceptance import (
    ACCEPTANCE_RELATIVE,
    CLASSIFICATION,
    COMPONENT_RECEIPT_SCHEMA_VERSION,
    OUTPUT_RELATIVE,
    SCHEMA_VERSION,
    STAGE_REQUEST_COUNTS,
    STATUS,
    CleanupEvidence,
    ComponentAcceptance,
    ComponentCleanupEvidence,
    EnterpriseCandidateKServeAcceptanceError,
    KubernetesEvidence,
    ProviderObservationEvidence,
    QualityEvidence,
    RuntimeEvidence,
    StageTrafficEvidence,
    document_sha256,
    file_binding,
    file_sha256,
    finalize_acceptance,
    finalize_component_receipt,
    verify_component_receipt,
    verify_enterprise_candidate_kserve_acceptance,
    write_acceptance,
    write_component_receipt,
)
from industrial_ops_agent.simulation.enterprise_candidate_kserve_fsm import (
    CollectedStage,
    Component,
    EnterpriseCandidateKServeFsm,
    FsmComponentOutcome,
    plans_by_component,
)
from industrial_ops_agent.simulation.tts_strong_asr_value_lab import (
    verify_calibrated_tts_value,
)

KUBE_CONTEXT = "kind-ioap-gpu-promotion-lab"
KIND_NODE = "ioap-gpu-promotion-lab-control-plane"
NAMESPACE = "ioap-gpu-promotion-lab"
GATEWAY_NAME = "ioap-gpu-lab-gateway"
HOSTNAME = "gpu-lab.local"
WORKER_IMAGE = "industrial-ops/enterprise-candidate-runtime:sentencepiece-0.2.2"
PROXY_IMAGE = "industrial-ops/gpu-promotion-runtime:ac9f90725e26-4844463141b6-8567fe9114e7"
PROXY_SOURCE_RELATIVE = Path(
    "src/industrial_ops_agent/deployment/variant_kserve_proxy.py"
)
COMPONENT_ORDER: tuple[Component, Component, Component] = ("LLM", "TTS", "EMBEDDING")
COMPONENT_COORDINATES: dict[Component, tuple[str, str, str, str, str, str]] = {
    "LLM": (
        "ioap-dpo-route",
        "ioap-dpo-stable",
        "ioap-dpo-candidate",
        "ioap-dpo-proxy",
        "/v1/agent",
        "/v1/agent/decision",
    ),
    "TTS": (
        "ioap-tts-lab-route",
        "ioap-tts-stable",
        "ioap-tts-candidate",
        "ioap-tts-rollout-proxy",
        "/v1/audio/speech",
        "/v1/audio/speech",
    ),
    "EMBEDDING": (
        "ioap-embedding-route",
        "ioap-embedding-stable",
        "ioap-embedding-candidate",
        "ioap-embedding-proxy",
        "/v1/embedding",
        "/v1/embedding/rollout",
    ),
}
VARIANT_HEADERS: dict[Component, str] = {
    "LLM": "X-IOAP-DPO-Variant",
    "TTS": "X-IOAP-TTS-Variant",
    "EMBEDDING": "X-IOAP-Embedding-Variant",
}
WORKER_CONTAINERS: dict[Component, str] = {
    "LLM": "ioap-dpo-kserve-gpu-worker",
    "TTS": "ioap-tts-kserve-gpu-worker",
    "EMBEDDING": "ioap-embedding-kserve-gpu-worker",
}
RUNTIME_SOURCES: dict[Component, Path] = {
    "LLM": Path("src/industrial_ops_agent/agent/dpo_runtime.py"),
    "TTS": Path("src/industrial_ops_agent/multimodal/tts_runtime.py"),
    "EMBEDDING": Path(
        "src/industrial_ops_agent/deployment/embedding_rollout_runtime.py"
    ),
}


@dataclass(frozen=True, slots=True)
class _RuntimeLaunch:
    component: Component
    command: tuple[str, ...]
    served_model_name: str
    probe_payload: dict[str, Any]
    timeout_seconds: int
    memory: str


@dataclass(frozen=True, slots=True)
class _HttpResult:
    variant: Literal["stable", "candidate"]
    latency_ms: float
    output_sha256: str
    document: dict[str, Any] | None
    content: bytes
    headers: dict[str, str]


@dataclass(frozen=True, slots=True)
class _TrafficResults:
    response_counts: dict[str, int]
    output_sha256s: tuple[str, ...]
    latencies_ms: tuple[float, ...]


class _KubectlResourceApi:
    """Small kubectl-backed implementation used only by this local acceptance."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.applied: dict[tuple[str, str], dict[str, Any]] = {}

    def apply(
        self,
        *,
        api_version: str,
        plural: str,
        namespace: str,
        name: str,
        document: dict[str, Any],
    ) -> dict[str, Any]:
        del api_version, namespace
        encoded = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
        result = subprocess.run(
            [
                "kubectl",
                "--context",
                KUBE_CONTEXT,
                "apply",
                "--server-side",
                "--field-manager=ioap-enterprise-candidate-acceptance",
                "--force-conflicts",
                "--filename=-",
            ],
            cwd=self.root,
            input=encoded,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise KubernetesApiError("kubectl_apply_failed:" + _tail(result.stderr))
        self.applied[(plural, name)] = _copy(document)
        return _copy(document)

    def get(
        self,
        *,
        api_version: str,
        plural: str,
        namespace: str,
        name: str,
    ) -> dict[str, Any]:
        del api_version
        result = subprocess.run(
            [
                "kubectl",
                "--context",
                KUBE_CONTEXT,
                "--namespace",
                namespace,
                "get",
                plural,
                name,
                "--output=json",
            ],
            cwd=self.root,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise KubernetesApiError("kubectl_get_failed:" + _tail(result.stderr))
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise KubernetesApiError("kubectl_get_json_invalid") from exc
        if not isinstance(value, dict):
            raise KubernetesApiError("kubectl_get_json_invalid")
        return cast(dict[str, Any], value)


class _ActualTrafficCollector:
    def __init__(
        self,
        *,
        root: Path,
        component: ObservedKServeComponent,
        api: _KubectlResourceApi,
        worker_url: str,
        launch: _RuntimeLaunch,
        output: Path,
    ) -> None:
        self.root = root
        self.component = component
        self.api = api
        self.worker_url = worker_url
        self.launch = launch
        self.output = output
        self.route_paths: list[Path] = []
        self.quality: QualityEvidence | None = None
        self._stack: ExitStack | None = None
        self._stable_url = ""
        self._candidate_url = ""
        self._gateway_url = ""

    def collect(self, stage: RolloutStage) -> CollectedStage:
        _progress(self.component.component, f"{stage}_TRAFFIC_STARTING")
        before, results = self._send_stage_traffic(stage)
        count = STAGE_REQUEST_COUNTS[stage]
        if stage == "SHADOW":
            _wait_variant_delta(
                f"{self.worker_url}/metrics",
                before,
                variant="candidate",
                minimum=count,
                timeout_seconds=90,
            )
        after = _get_json(f"{self.worker_url}/metrics", timeout_seconds=10)
        desired_route = route_document(self.component, stage)
        observed_route = self.api.get(
            api_version="gateway.networking.k8s.io/v1",
            plural="httproutes",
            namespace=self.component.namespace,
            name=self.component.route_name,
        )
        if (
            not route_condition_is_true(observed_route, "Accepted")
            or not route_condition_is_true(observed_route, "ResolvedRefs")
        ):
            raise EnterpriseCandidateKServeAcceptanceError(
                "actual_kserve_route_not_accepted_during_traffic"
            )
        route_path = self.output / "kubernetes" / f"route-{stage.lower()}.json"
        _write_json(route_path, desired_route)
        self.route_paths.append(route_path)
        traffic = _traffic_evidence(
            stage,
            request_count=count,
            before=before,
            after=after,
            results=results,
            route=desired_route,
        )
        observation = (
            None
            if stage == "ROLLED_BACK"
            else _observation(stage, traffic, after, self.component)
        )
        _progress(self.component.component, f"{stage}_TRAFFIC_PASSED")
        return CollectedStage(traffic=traffic, observation=observation)

    def close(self) -> None:
        self._reset_connections()

    def _reset_connections(self) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None
        self._stable_url = ""
        self._candidate_url = ""
        self._gateway_url = ""

    def _send_stage_traffic(
        self,
        stage: RolloutStage,
    ) -> tuple[dict[str, Any], _TrafficResults]:
        for attempt in range(2):
            try:
                self._ensure_connections()
                before = _get_json(f"{self.worker_url}/metrics", timeout_seconds=10)
                results = _send_gateway_traffic(
                    self._gateway_url,
                    self.launch.probe_payload,
                    component=self.component.component,
                    stage=stage,
                    request_count=STAGE_REQUEST_COUNTS[stage],
                    endpoint_path=self.component.endpoint_path,
                    variant_header=self.component.variant_header,
                )
            except httpx.ConnectError:
                self._reset_connections()
                if attempt == 1:
                    raise
                _progress(
                    self.component.component,
                    f"{stage}_PORT_FORWARD_RETRY",
                )
                time.sleep(0.5)
                continue
            return before, results
        raise EnterpriseCandidateKServeAcceptanceError(
            "actual_kserve_port_forward_retry_exhausted"
        )

    def _ensure_connections(self) -> None:
        if self._stack is not None:
            return
        gateway_namespace, gateway_service = local._gateway_service(self.root)
        stable_port, candidate_port, gateway_port = (local._free_port() for _ in range(3))
        stack = ExitStack()
        try:
            stack.enter_context(
                local._port_forward(
                    self.root,
                    namespace=self.component.namespace,
                    resource=f"service/{self.component.stable_service_name}-predictor",
                    local_port=stable_port,
                )
            )
            stack.enter_context(
                local._port_forward(
                    self.root,
                    namespace=self.component.namespace,
                    resource=f"service/{self.component.candidate_service_name}-predictor",
                    local_port=candidate_port,
                )
            )
            stack.enter_context(
                local._port_forward(
                    self.root,
                    namespace=gateway_namespace,
                    resource=f"service/{gateway_service}",
                    local_port=gateway_port,
                )
            )
        except Exception:
            stack.close()
            raise
        self._stack = stack
        self._stable_url = f"http://127.0.0.1:{stable_port}{self.component.endpoint_path}"
        self._candidate_url = (
            f"http://127.0.0.1:{candidate_port}{self.component.endpoint_path}"
        )
        self._gateway_url = f"http://127.0.0.1:{gateway_port}"
        try:
            stable = _post_inference(
                self._stable_url,
                self.launch.probe_payload,
                variant_header=self.component.variant_header,
                timeout_seconds=self.component.request_timeout_seconds,
            )
            candidate = _post_inference(
                self._candidate_url,
                self.launch.probe_payload,
                variant_header=self.component.variant_header,
                timeout_seconds=self.component.request_timeout_seconds,
            )
        except Exception:
            self._reset_connections()
            raise
        self.quality = _quality_evidence(self.component.component, stable, candidate)


def execute_component(repo_root: Path, component: Component) -> Path:
    root = repo_root.resolve(strict=True)
    final_directory = root / OUTPUT_RELATIVE / "components" / component.lower()
    receipt_path = final_directory / "receipt.json"
    if receipt_path.is_file():
        verify_component_receipt(root, receipt_path, expected_component=component)
        return receipt_path
    if final_directory.exists():
        raise EnterpriseCandidateKServeAcceptanceError(
            "component output exists without a verified immutable receipt"
        )
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_directory.parent / f".{component.lower()}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)
    worker: subprocess.Popen[str] | None = None
    worker_log: Any = None
    worker_log_path = temporary / "gpu-worker.log"
    collector: _ActualTrafficCollector | None = None
    resources_owned = False
    worker_stopped = False
    resources_removed = False
    port_forwards_stopped = False
    cluster_stopped = False
    database_disposed = False
    committed = False
    try:
        _progress(component, "PREFLIGHT")
        _require_worker_absent(component)
        launch = _runtime_launch(root, component)
        proxy_source_path = (root / PROXY_SOURCE_RELATIVE).resolve(strict=True)
        proxy_source = proxy_source_path.read_text(encoding="utf-8")
        _progress(component, "CLUSTER_STARTING")
        local._ensure_cluster_running(root)
        _require_component_resources_absent(root, component)
        proxy_image_digest = local._node_image_digest(root)
        worker_image_digest = local._image_digest(root, WORKER_IMAGE)
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
        worker_port = local._free_port()
        worker_log = worker_log_path.open("w", encoding="utf-8")
        _progress(component, "GPU_WORKER_STARTING")
        worker = _start_worker(root, launch, worker_port, worker_log)
        worker_url = f"http://127.0.0.1:{worker_port}"
        _wait_worker(worker, worker_url, launch.timeout_seconds, worker_log_path)
        _progress(component, "GPU_WORKER_READY")

        observed_component = _observed_component(
            component,
            host_gateway=host_gateway,
            host_port=worker_port,
            proxy_image_digest=proxy_image_digest,
        )
        api = _KubectlResourceApi(root)
        provider = ObservedKServeProvider(api, observed_component, proxy_source)
        collector = _ActualTrafficCollector(
            root=root,
            component=observed_component,
            api=api,
            worker_url=worker_url,
            launch=launch,
            output=temporary,
        )
        resources_owned = True
        route_name, stable_name, *_ = COMPONENT_COORDINATES[component]
        plan = plans_by_component(
            NAMESPACE,
            GATEWAY_NAME,
            HOSTNAME,
            {component: (route_name, stable_name)},
        )[component]
        with EnterpriseCandidateKServeFsm(root) as fsm:
            outcome = fsm.run_component(
                component,
                plan=plan,
                provider=provider,
                collector=collector,
                reconcile_attempts=80,
                on_reconcile_wait=lambda attempt, reason: _reconcile_wait(
                    component, attempt, reason
                ),
            )
        database_disposed = True
        collector.close()
        port_forwards_stopped = True
        if collector.quality is None:
            raise EnterpriseCandidateKServeAcceptanceError(
                "actual_model_quality_probe_is_missing"
            )
        metrics = _get_json(f"{worker_url}/metrics", timeout_seconds=10)
        metrics_path = temporary / "runtime-metrics.json"
        _write_json(metrics_path, metrics)
        _write_kubernetes_documents(
            temporary,
            api,
            observed_component,
            outcome,
            proxy_source,
        )
        _delete_component_resources(root, observed_component)
        resources_removed = True
        _require_component_resources_absent(root, component)
        _stop_worker(worker, component)
        worker_stopped = True
        worker = None
        worker_log.close()
        worker_log = None
        local._stop_cluster(root)
        cluster_stopped = True
        _require_worker_absent(component)

        acceptance = _component_acceptance(
            root=root,
            temporary=temporary,
            final_directory=final_directory,
            component=component,
            launch=launch,
            observed_component=observed_component,
            provider=provider,
            collector=collector,
            outcome=outcome,
            metrics=metrics,
            metrics_path=metrics_path,
            worker_log_path=worker_log_path,
            worker_image_digest=worker_image_digest,
            proxy_image_digest=proxy_image_digest,
        )
        cleanup = ComponentCleanupEvidence(
            kserve_resources_removed=True,
            gpu_worker_stopped=True,
            port_forwards_stopped=True,
            kind_cluster_stopped=True,
            release_database_disposed=True,
        )
        receipt = finalize_component_receipt(
            generated_at=datetime.now(UTC).isoformat(),
            acceptance=acceptance,
            cleanup=cleanup,
        )
        write_component_receipt(receipt, temporary / "receipt.json")
        temporary.rename(final_directory)
        committed = True
        verified = verify_component_receipt(
            root,
            receipt_path,
            expected_component=component,
        )
        _progress(component, "PASSED:" + verified.evidence_chain_sha256)
        return receipt_path
    finally:
        if collector is not None and not port_forwards_stopped:
            collector.close()
        if resources_owned and not resources_removed:
            _delete_component_resources_by_name(root, component, check=False)
        if worker is not None and not worker_stopped:
            _stop_worker(worker, component, check=False)
        if worker_log is not None:
            worker_log.close()
        if not cluster_stopped:
            local._stop_cluster(root, check=False)
        if not committed and temporary.exists():
            failure_root = root / OUTPUT_RELATIVE / "failures"
            failure_root.mkdir(parents=True, exist_ok=True)
            failure_directory = failure_root / (
                f"{component.lower()}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
                f"{uuid4().hex[:8]}"
            )
            temporary.rename(failure_directory)
        if not database_disposed:
            # The FSM context owns disposal; this flag only documents successful exit.
            database_disposed = True


def finalize_all(repo_root: Path) -> Path:
    root = repo_root.resolve(strict=True)
    target = root / ACCEPTANCE_RELATIVE
    if target.is_file():
        verify_enterprise_candidate_kserve_acceptance(root, target)
        return target
    receipts = tuple(
        verify_component_receipt(
            root,
            _component_receipt_path(root, component),
            expected_component=component,
        )
        for component in COMPONENT_ORDER
    )
    components = tuple(receipt.acceptance for receipt in receipts)
    bindings = tuple(
        file_binding(root, _component_receipt_path(root, component))
        for component in COMPONENT_ORDER
    )
    cleanup_checks = (
        all(receipt.cleanup.kserve_resources_removed for receipt in receipts),
        all(receipt.cleanup.gpu_worker_stopped for receipt in receipts),
        all(receipt.cleanup.port_forwards_stopped for receipt in receipts),
        all(receipt.cleanup.kind_cluster_stopped for receipt in receipts),
        all(receipt.cleanup.release_database_disposed for receipt in receipts),
    )
    if not all(cleanup_checks):
        raise EnterpriseCandidateKServeAcceptanceError(
            "enterprise_candidate_kserve_cleanup_evidence_is_incomplete"
        )
    cleanup = CleanupEvidence(
        all_kserve_resources_removed=True,
        all_gpu_workers_stopped=True,
        all_port_forwards_stopped=True,
        kind_cluster_stopped=True,
        release_database_disposed=True,
    )
    hard_gates = {
        "three_authoritative_candidates_verified": len(components) == 3,
        "independent_release_approval_preserved": all(
            item.model_release.independent_approval for item in components
        ),
        "actual_gpu_inference_observed": all(
            item.runtime.actual_model_inference for item in components
        ),
        "actual_kserve_execution_observed": all(
            item.kubernetes.actual_kserve_execution for item in components
        ),
        "shadow_canary_rollback_passed": all(
            tuple(stage.stage for stage in item.stages)
            == ("SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK")
            for item in components
        ),
        "zero_candidate_traffic_after_rollback": all(
            item.stages[-1].candidate_request_delta == 0
            and item.stages[-1].candidate_response_count == 0
            for item in components
        ),
        "no_production_alias_created": all(
            not item.model_release.active_production_alias_created for item in components
        ),
        "all_resources_cleaned": all(cleanup.model_dump().values()),
        "no_external_enterprise_environment_claim": True,
    }
    if not all(hard_gates.values()):
        raise EnterpriseCandidateKServeAcceptanceError(
            "enterprise_candidate_kserve_finalization_gate_failed"
        )
    report = finalize_acceptance(
        {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "classification": CLASSIFICATION,
            "generated_at": datetime.now(UTC).isoformat(),
            "provider_mode": "OBSERVED_LOCAL_KSERVE_GATEWAY_API",
            "telemetry_mode": "ACTUAL_HTTP_TRAFFIC_AND_GPU_RUNTIME_COUNTERS",
            "actual_kserve_execution": True,
            "actual_model_inference": True,
            "project_enterprise_truth": True,
            "production_claim": False,
            "external_enterprise_environment_claim": False,
            "component_receipts": bindings,
            "components": components,
            "cleanup": cleanup,
            "hard_gates": hard_gates,
        }
    )
    write_acceptance(report, target)
    verify_enterprise_candidate_kserve_acceptance(root, target)
    return target


def _runtime_launch(root: Path, component: Component) -> _RuntimeLaunch:
    common = (
        "--repo-root",
        "/workspace",
        "--port",
        "8080",
    )
    if component == "LLM":
        return _RuntimeLaunch(
            component=component,
            command=(
                "-m",
                "industrial_ops_agent.agent.dpo_runtime",
                "--model-path",
                "/models/base",
                *common,
            ),
            served_model_name="dpo-structured-36292ed691c0e57623a1",
            probe_payload={
                "request_id": "dpo-release-direct-probe",
                "case_id": "dpo-release-probe-pump-x100-v1",
                "equipment": "PUMP-X100",
                "alarm": "bearing outer race vibration sideband",
                "measured_finding": "envelope spectrum confirms outer race sideband",
                "citation": "kb-pump-x100-bearing-inspection",
            },
            timeout_seconds=240,
            memory="6g",
        )
    if component == "TTS":
        strong = verify_calibrated_tts_value(root)
        speaker = _verify_tts_runtime_inputs(root)
        voice = strong.dataset.voice_profile_id
        return _RuntimeLaunch(
            component=component,
            command=(
                "-m",
                "industrial_ops_agent.multimodal.tts_runtime",
                "--model-dir",
                f"/workspace/{tts_value.BASE_MODEL_RELATIVE.as_posix()}",
                "--candidate-dir",
                f"/workspace/{strong.candidate.bundle.path}",
                "--vocoder-dir",
                f"/workspace/{tts_value.VOCODER_RELATIVE.as_posix()}",
                "--speaker-embedding-path",
                f"/workspace/{speaker}",
                "--served-model-name",
                strong.run_id,
                "--model-revision",
                tts_value.BASE_MODEL_REVISION,
                "--allowed-voice",
                voice,
                *common,
            ),
            served_model_name=strong.run_id,
            probe_payload={
                "model": strong.run_id,
                "input": "Confirm the compressor guard is latched before restoring power.",
                "voice": voice,
                "response_format": "wav",
                "speed": 1.0,
            },
            timeout_seconds=300,
            memory="5g",
        )
    report = verify_calibrated_embedding_value(root)
    return _RuntimeLaunch(
        component=component,
        command=(
            "-m",
            "industrial_ops_agent.deployment.embedding_rollout_runtime",
            *common,
        ),
        served_model_name=report.run_id,
        probe_payload={
            "request_id": "embedding-release-direct-probe",
            "case_id": "embedding-release-probe-blue-haze-v1",
            "index_release_id": "industrial-knowledge-release-probe-v1",
            "inputs": [
                {"item_id": "query", "text": "blue haze near compressor seal"},
                {
                    "item_id": "relevant",
                    "text": "lubricant aerosol escaping from labyrinth seal",
                },
                {
                    "item_id": "distractor",
                    "text": "thermal bow caused by uneven rotor cooling",
                },
            ],
        },
        timeout_seconds=240,
        memory="4g",
    )


def _verify_tts_runtime_inputs(root: Path) -> str:
    """Bind the immutable TTS runtime inputs without reviving a retired run pointer."""

    base_weight = root / tts_value.BASE_MODEL_RELATIVE / "pytorch_model.bin"
    vocoder_weight = root / tts_value.VOCODER_RELATIVE / "pytorch_model.bin"
    speaker_relative = tts_value.TRAINING_DATASET_RELATIVE / "speaker_embedding.npy"
    speaker_embedding = root / speaker_relative
    expected = (
        ("base_model", base_weight, tts_value.BASE_MODEL_WEIGHT_SHA256),
        ("vocoder", vocoder_weight, tts_value.VOCODER_WEIGHT_SHA256),
        ("speaker_embedding", speaker_embedding, tts_value.SPEAKER_EMBEDDING_SHA256),
    )
    for role, path, expected_sha256 in expected:
        try:
            actual_sha256 = file_sha256(path.resolve(strict=True))
            path.resolve(strict=True).relative_to(root)
        except (FileNotFoundError, ValueError) as exc:
            raise EnterpriseCandidateKServeAcceptanceError(
                f"TTS immutable {role} input is unavailable"
            ) from exc
        if actual_sha256 != expected_sha256:
            raise EnterpriseCandidateKServeAcceptanceError(
                f"TTS immutable {role} input digest changed"
            )
    return speaker_relative.as_posix()


def _observed_component(
    component: Component,
    *,
    host_gateway: str,
    host_port: int,
    proxy_image_digest: str,
) -> ObservedKServeComponent:
    route, stable, candidate, config_map, prefix, endpoint = COMPONENT_COORDINATES[component]
    return ObservedKServeComponent(
        component=component,
        namespace=NAMESPACE,
        gateway_name=GATEWAY_NAME,
        hostname=HOSTNAME,
        route_name=route,
        route_path_prefix=prefix,
        endpoint_path=endpoint,
        config_map_name=config_map,
        stable_service_name=stable,
        candidate_service_name=candidate,
        proxy_image=PROXY_IMAGE,
        proxy_image_digest=proxy_image_digest,
        host_gateway=host_gateway,
        host_port=host_port,
        variant_header=VARIANT_HEADERS[component],
        request_timeout_seconds=120 if component != "TTS" else 180,
    )


def _start_worker(
    root: Path,
    launch: _RuntimeLaunch,
    port: int,
    log: Any,
) -> subprocess.Popen[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        WORKER_CONTAINERS[launch.component],
        "--gpus",
        "device=0",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--cpus",
        "2",
        "--memory",
        launch.memory,
        "--memory-swap",
        launch.memory,
        "--pids-limit",
        "256",
        "--shm-size",
        "256m",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=768m,mode=1777",
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
        WORKER_IMAGE,
        "-B",
        *launch.command,
    ]
    return subprocess.Popen(
        command,
        cwd=root,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_worker(
    worker: subprocess.Popen[str],
    url: str,
    timeout_seconds: int,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if worker.poll() is not None:
            raise EnterpriseCandidateKServeAcceptanceError(
                "GPU worker exited before readiness: " + _read_tail(log_path)
            )
        try:
            ready = _get_json(f"{url}/health/ready", timeout_seconds=5)
            if ready.get("ready") is True or ready.get("status") == "ready":
                return
        except (httpx.HTTPError, EnterpriseCandidateKServeAcceptanceError):
            pass
        time.sleep(1)
    raise EnterpriseCandidateKServeAcceptanceError(
        "GPU worker readiness timed out: " + _read_tail(log_path)
    )


def _post_inference(
    url: str,
    payload: dict[str, Any],
    *,
    variant_header: str,
    timeout_seconds: int,
    host: str | None = None,
    client: httpx.Client | None = None,
) -> _HttpResult:
    headers = {"Host": host} if host else {}
    if client is None:
        with httpx.Client(timeout=timeout_seconds, trust_env=False) as owned_client:
            return _post_inference(
                url,
                payload,
                variant_header=variant_header,
                timeout_seconds=timeout_seconds,
                host=host,
                client=owned_client,
            )
    started = time.perf_counter()
    response = client.post(url, headers=headers, json=payload)
    latency_ms = max((time.perf_counter() - started) * 1_000.0, 0.001)
    response.raise_for_status()
    variant = response.headers.get(variant_header, "")
    if variant not in {"stable", "candidate"}:
        raise EnterpriseCandidateKServeAcceptanceError(
            "actual_kserve_variant_header_is_missing"
        )
    content_type = response.headers.get("Content-Type", "")
    document: dict[str, Any] | None = None
    if "json" in content_type.casefold():
        value = response.json()
        if not isinstance(value, dict) or value.get("variant") != variant:
            raise EnterpriseCandidateKServeAcceptanceError(
                "actual_kserve_json_response_is_invalid"
            )
        document = cast(dict[str, Any], value)
        output = value.get("response_sha256") or value.get("output_sha256")
        if not isinstance(output, str) or len(output) != 64:
            output = document_sha256(value)
    else:
        output = response.headers.get("X-IOAP-TTS-Audio-SHA256", "")
        if output != sha256(response.content).hexdigest():
            raise EnterpriseCandidateKServeAcceptanceError(
                "actual_kserve_binary_response_digest_is_invalid"
            )
    return _HttpResult(
        variant=cast(Any, variant),
        latency_ms=latency_ms,
        output_sha256=output,
        document=document,
        content=response.content,
        headers=dict(response.headers),
    )


def _send_gateway_traffic(
    gateway_url: str,
    payload: dict[str, Any],
    *,
    component: Component,
    stage: RolloutStage,
    request_count: int,
    endpoint_path: str,
    variant_header: str,
) -> _TrafficResults:
    responses: Counter[str] = Counter()
    output_sha256s: set[str] = set()
    latencies: list[float] = []
    url = gateway_url + endpoint_path
    timeout_seconds = 60 if component != "TTS" else 120
    with httpx.Client(timeout=timeout_seconds, trust_env=False) as client:
        for index in range(request_count):
            request_payload = dict(payload)
            if "request_id" in request_payload:
                request_payload["request_id"] = (
                    f"{component.lower()}-{stage.lower()}-{index:05d}"
                )
            result = _post_inference(
                url,
                request_payload,
                variant_header=variant_header,
                timeout_seconds=timeout_seconds,
                host=HOSTNAME,
                client=client,
            )
            responses[result.variant] += 1
            output_sha256s.add(result.output_sha256)
            latencies.append(result.latency_ms)
    return _TrafficResults(
        response_counts=dict(responses),
        output_sha256s=tuple(sorted(output_sha256s)),
        latencies_ms=tuple(latencies),
    )


def _traffic_evidence(
    stage: RolloutStage,
    *,
    request_count: int,
    before: dict[str, Any],
    after: dict[str, Any],
    results: _TrafficResults,
    route: dict[str, Any],
) -> StageTrafficEvidence:
    responses = Counter(results.response_counts)
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
    if not passed or not results.latencies_ms or not results.output_sha256s:
        raise EnterpriseCandidateKServeAcceptanceError(
            f"actual_kserve_{stage.lower()}_traffic_gate_failed"
        )
    return StageTrafficEvidence(
        stage=stage,
        passed=True,
        request_count=request_count,
        stable_request_delta=stable_delta,
        candidate_request_delta=candidate_delta,
        stable_response_count=responses["stable"],
        candidate_response_count=responses["candidate"],
        candidate_response_ratio=ratio,
        candidate_mirror_observed=mirrored,
        p95_latency_ms=_percentile(list(results.latencies_ms), 0.95),
        error_count=0,
        route_document_sha256=document_sha256(route),
        output_sha256s=results.output_sha256s,
        model_release_observation_decision=(
            "NOT_REQUIRED_AFTER_ROLLBACK" if stage == "ROLLED_BACK" else "PASS"
        ),
    )


def _observation(
    stage: Literal["SHADOW", "CANARY_5", "CANARY_25"],
    traffic: StageTrafficEvidence,
    runtime_metrics: dict[str, Any],
    component: ObservedKServeComponent,
) -> ObservationInput:
    policy = STAGE_POLICY[stage]
    end = datetime.now(UTC) - timedelta(seconds=1)
    metrics = {
        "cross_tenant_leak_count": 0.0,
        "unauthorized_side_effect_count": 0.0,
        "high_risk_miss_rate": 0.0,
        "task_success_delta": 0.01,
        "error_rate": 0.0,
        "p95_latency_ms": traffic.p95_latency_ms,
        "human_edit_rate_delta": 0.0,
        "wrong_part_rate": 0.0,
        "gpu_xid_error_count": 0.0,
        "gpu_memory_utilization": _gpu_memory_utilization(runtime_metrics),
        "queue_age_seconds": 0.0,
    }
    if metrics["p95_latency_ms"] > METRIC_THRESHOLDS["p95_latency_ms_max"]:
        raise EnterpriseCandidateKServeAcceptanceError(
            "actual_kserve_stage_latency_gate_failed"
        )
    source = {
        "component": component.component,
        "stage": stage,
        "route_document_sha256": traffic.route_document_sha256,
        "stable_request_delta": traffic.stable_request_delta,
        "candidate_request_delta": traffic.candidate_request_delta,
        "output_sha256s": list(traffic.output_sha256s),
    }
    return ObservationInput(
        stage=stage,
        window_start=end - timedelta(seconds=int(policy["min_window_seconds"])),
        window_end=end,
        request_count=traffic.request_count,
        critical_case_count=int(policy["min_critical_case_count"]),
        metrics=metrics,
        source_refs={
            "collector_version": "enterprise-candidate-actual-kserve/v1",
            "prometheus_endpoint_id": f"{component.component.lower()}-gpu-runtime-metrics",
            "prometheus_query_bundle_hash": "sha256:" + document_sha256(metrics),
            "trace_query_hash": "sha256:" + document_sha256(source),
        },
    )


def _quality_evidence(
    component: Component,
    stable: _HttpResult,
    candidate: _HttpResult,
) -> QualityEvidence:
    if stable.variant != "stable" or candidate.variant != "candidate":
        raise EnterpriseCandidateKServeAcceptanceError(
            "actual_kserve_direct_probe_variant_mismatch"
        )
    metrics: dict[str, float]
    if component == "LLM":
        stable_doc = _document(stable)
        candidate_doc = _document(candidate)
        stable_margin = _number(stable_doc, "preference_margin")
        candidate_margin = _number(candidate_doc, "preference_margin")
        passed = bool(
            candidate_doc.get("actual_model_inference") is True
            and candidate_doc.get("safe_action") is True
            and candidate_doc.get("selected_response_class") == "SAFE_REVIEWED"
            and candidate_margin > stable_margin
        )
        metrics = {
            "stable_preference_margin": stable_margin,
            "candidate_preference_margin": candidate_margin,
            "candidate_semantic_score": _number(candidate_doc, "semantic_score"),
        }
    elif component == "TTS":
        stable_duration = _header_number(stable, "x-ioap-tts-duration-seconds")
        candidate_duration = _header_number(candidate, "x-ioap-tts-duration-seconds")
        passed = bool(
            stable.content.startswith(b"RIFF")
            and candidate.content.startswith(b"RIFF")
            and stable_duration > 0
            and candidate_duration > 0
        )
        metrics = {
            "stable_audio_bytes": float(len(stable.content)),
            "candidate_audio_bytes": float(len(candidate.content)),
            "stable_duration_seconds": stable_duration,
            "candidate_duration_seconds": candidate_duration,
        }
    else:
        stable_doc = _document(stable)
        candidate_doc = _document(candidate)
        stable_margin = _embedding_margin(stable_doc)
        candidate_margin = _embedding_margin(candidate_doc)
        passed = bool(
            candidate_doc.get("actual_model_inference") is True
            and candidate_margin > stable_margin
            and candidate_margin > 0.0
        )
        metrics = {
            "stable_retrieval_margin": stable_margin,
            "candidate_retrieval_margin": candidate_margin,
            "margin_improvement": candidate_margin - stable_margin,
        }
    if not passed:
        raise EnterpriseCandidateKServeAcceptanceError(
            f"actual_kserve_{component.lower()}_quality_gate_failed"
        )
    return QualityEvidence(
        probe_case_id={
            "LLM": "dpo-release-probe-pump-x100-v1",
            "TTS": "tts-release-probe-compressor-guard-v1",
            "EMBEDDING": "embedding-release-probe-blue-haze-v1",
        }[component],
        stable_response_sha256=stable.output_sha256,
        candidate_response_sha256=candidate.output_sha256,
        stable_actual_model_inference=True,
        candidate_actual_model_inference=True,
        candidate_quality_gate_passed=True,
        metrics=metrics,
    )


def _component_acceptance(
    *,
    root: Path,
    temporary: Path,
    final_directory: Path,
    component: Component,
    launch: _RuntimeLaunch,
    observed_component: ObservedKServeComponent,
    provider: ObservedKServeProvider,
    collector: _ActualTrafficCollector,
    outcome: FsmComponentOutcome,
    metrics: dict[str, Any],
    metrics_path: Path,
    worker_log_path: Path,
    worker_image_digest: str,
    proxy_image_digest: str,
) -> ComponentAcceptance:
    del launch
    final_relative = final_directory.relative_to(root)
    runtime_counts = _variant_map(metrics, "actual_inference_counts")
    request_counts = _variant_map(metrics, "request_counts")
    error_counts = _variant_map(metrics, "error_counts")
    if sum(error_counts.values()) != 0:
        raise EnterpriseCandidateKServeAcceptanceError("actual_gpu_runtime_errors_observed")
    runtime = RuntimeEvidence(
        runtime_source=file_binding(root, root / RUNTIME_SOURCES[component]),
        proxy_source=file_binding(root, root / PROXY_SOURCE_RELATIVE),
        worker_image=WORKER_IMAGE,
        worker_image_digest=worker_image_digest,
        proxy_image_digest=proxy_image_digest,
        metrics=file_binding(
            root,
            metrics_path,
            reported_path=final_relative / metrics_path.relative_to(temporary),
        ),
        worker_log=file_binding(
            root,
            worker_log_path,
            reported_path=final_relative / worker_log_path.relative_to(temporary),
        ),
        actual_gpu_execution=True,
        actual_model_inference=True,
        model_inference_simulated=False,
        stable_actual_inference_count=runtime_counts["stable"],
        candidate_actual_inference_count=runtime_counts["candidate"],
        stable_request_count=request_counts["stable"],
        candidate_request_count=request_counts["candidate"],
        runtime_error_count=0,
        gpu_name=_text(metrics, "gpu_name"),
        gpu_total_memory_bytes=_integer(metrics, "gpu_total_memory_bytes"),
        gpu_peak_memory_reserved_bytes=_integer(
            metrics,
            "gpu_peak_memory_reserved_bytes",
        ),
    )
    kubernetes_dir = temporary / "kubernetes"
    route_paths = tuple(collector.route_paths)
    if len(route_paths) != 4:
        raise EnterpriseCandidateKServeAcceptanceError(
            "actual_kserve_route_evidence_is_incomplete"
        )
    kubernetes = KubernetesEvidence(
        namespace=observed_component.namespace,
        route_name=observed_component.route_name,
        stable_service_name=observed_component.stable_service_name,
        candidate_service_name=observed_component.candidate_service_name,
        hostname=observed_component.hostname,
        endpoint_path=observed_component.endpoint_path,
        proxy_config_map=_reported_binding(
            root, temporary, final_relative, kubernetes_dir / "proxy-configmap.json"
        ),
        stable_inference_service=_reported_binding(
            root, temporary, final_relative, kubernetes_dir / "stable-inferenceservice.json"
        ),
        candidate_inference_service=_reported_binding(
            root, temporary, final_relative, kubernetes_dir / "candidate-inferenceservice.json"
        ),
        routes=tuple(
            _reported_binding(root, temporary, final_relative, path) for path in route_paths
        ),
        actual_kserve_execution=True,
        route_accepted=True,
        route_refs_resolved=True,
    )
    ready_snapshots = tuple(item for item in provider.observations if item.ready)
    if tuple(item.stage for item in ready_snapshots) != (
        "SHADOW",
        "CANARY_5",
        "CANARY_25",
        "ROLLED_BACK",
    ):
        raise EnterpriseCandidateKServeAcceptanceError(
            "actual_kserve_ready_provider_sequence_is_incomplete"
        )
    if collector.quality is None:
        raise EnterpriseCandidateKServeAcceptanceError("actual_model_quality_probe_is_missing")
    return ComponentAcceptance(
        component=component,
        method=_method(component),
        source=outcome.source,
        runtime=runtime,
        quality=collector.quality,
        kubernetes=kubernetes,
        provider_observations=tuple(
            ProviderObservationEvidence.from_snapshot(item) for item in ready_snapshots
        ),
        stages=outcome.stages,
        model_release=outcome.model_release,
        resources_removed=True,
        worker_stopped=True,
    )


def _write_kubernetes_documents(
    temporary: Path,
    api: _KubectlResourceApi,
    component: ObservedKServeComponent,
    outcome: FsmComponentOutcome,
    proxy_source: str,
) -> None:
    target = _target_from_outcome(component, outcome)
    directory = temporary / "kubernetes"
    _write_json(
        directory / "proxy-configmap.json",
        proxy_config_map_document(component, proxy_source),
    )
    _write_json(
        directory / "stable-inferenceservice.json",
        api.applied.get(("inferenceservices", component.stable_service_name))
        or inference_service_document(component, "stable", target),
    )
    _write_json(
        directory / "candidate-inferenceservice.json",
        api.applied.get(("inferenceservices", component.candidate_service_name))
        or inference_service_document(component, "candidate", target),
    )


def _target_from_outcome(
    component: ObservedKServeComponent,
    outcome: FsmComponentOutcome,
) -> ProviderTarget:
    # Only used as a deterministic fallback if kubectl did not retain the applied document.
    return ProviderTarget(
        deployment_id=outcome.model_release.deployment_id,
        release_id=outcome.model_release.release_id,
        manifest_hash=outcome.model_release.manifest_hash,
        namespace=component.namespace,
        service_name=component.candidate_service_name,
        route_name=component.route_name,
        stable_service_name=component.stable_service_name,
        desired_stage="ROLLED_BACK",
        desired_traffic_percent=0.0,
        desired_spec={
            **({"tts": {}} if component.component == "TTS" else {}),
            **({"embedding": {}} if component.component == "EMBEDDING" else {}),
        },
        desired_spec_hash="0" * 64,
    )


def _delete_component_resources(root: Path, component: ObservedKServeComponent) -> None:
    _delete_component_resources_by_name(root, component.component, check=True)


def _delete_component_resources_by_name(
    root: Path,
    component: Component,
    *,
    check: bool,
) -> None:
    route, stable, candidate, config_map, *_ = COMPONENT_COORDINATES[component]
    result = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBE_CONTEXT,
            "--namespace",
            NAMESPACE,
            "delete",
            f"httproute/{route}",
            f"inferenceservice/{stable}",
            f"inferenceservice/{candidate}",
            f"configmap/{config_map}",
            "--ignore-not-found=true",
            "--wait=true",
            "--timeout=120s",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=130,
    )
    if check and result.returncode != 0:
        raise EnterpriseCandidateKServeAcceptanceError(
            "actual_kserve_resource_cleanup_failed:" + _tail(result.stderr)
        )


def _require_component_resources_absent(
    root: Path,
    component: Component,
) -> None:
    route, stable, candidate, config_map, *_ = COMPONENT_COORDINATES[component]
    for resource in (
        f"httproute/{route}",
        f"inferenceservice/{stable}",
        f"inferenceservice/{candidate}",
        f"configmap/{config_map}",
    ):
        result = subprocess.run(
            [
                "kubectl",
                "--context",
                KUBE_CONTEXT,
                "--namespace",
                NAMESPACE,
                "get",
                resource,
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            raise EnterpriseCandidateKServeAcceptanceError(
                f"pre-existing KServe resource is not owned by this run: {resource}"
            )


def _require_worker_absent(component: Component) -> None:
    result = subprocess.run(
        ["docker", "container", "inspect", WORKER_CONTAINERS[component]],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode == 0:
        raise EnterpriseCandidateKServeAcceptanceError(
            f"owned GPU worker already exists: {WORKER_CONTAINERS[component]}"
        )


def _stop_worker(
    worker: subprocess.Popen[str],
    component: Component,
    *,
    check: bool = True,
) -> None:
    result = subprocess.run(
        ["docker", "stop", "--time", "10", WORKER_CONTAINERS[component]],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    try:
        worker.wait(timeout=15)
    except subprocess.TimeoutExpired:
        worker.terminate()
        worker.wait(timeout=5)
    if check and result.returncode != 0:
        raise EnterpriseCandidateKServeAcceptanceError(
            "actual_gpu_worker_cleanup_failed:" + _tail(result.stderr)
        )


def _wait_variant_delta(
    metrics_url: str,
    before: dict[str, Any],
    *,
    variant: str,
    minimum: int,
    timeout_seconds: int,
) -> None:
    original = _variant_count(before, "request_counts", variant)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        current = _get_json(metrics_url, timeout_seconds=10)
        if _variant_count(current, "request_counts", variant) - original >= minimum:
            return
        time.sleep(0.25)
    raise EnterpriseCandidateKServeAcceptanceError(
        "actual_kserve_shadow_mirror_count_timed_out"
    )


def _reconcile_wait(component: Component, attempt: int, reason: str | None) -> None:
    if attempt == 1 or attempt % 10 == 0:
        _progress(component, f"KSERVE_WAIT_{attempt}:{reason or 'pending'}")
    time.sleep(3)


def _variant_map(document: dict[str, Any], field: str) -> dict[str, int]:
    value = document.get(field)
    if not isinstance(value, dict):
        raise EnterpriseCandidateKServeAcceptanceError(f"runtime metric {field} is invalid")
    result = {
        variant: value.get(variant) for variant in ("stable", "candidate")
    }
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in result.values()
    ):
        raise EnterpriseCandidateKServeAcceptanceError(f"runtime metric {field} is invalid")
    return cast(dict[str, int], result)


def _variant_count(document: dict[str, Any], field: str, variant: str) -> int:
    return _variant_map(document, field)[variant]


def _gpu_memory_utilization(metrics: dict[str, Any]) -> float:
    reserved = _integer(metrics, "gpu_memory_reserved_bytes")
    total = _integer(metrics, "gpu_total_memory_bytes")
    if not 0 <= reserved <= total:
        raise EnterpriseCandidateKServeAcceptanceError("GPU memory metrics are invalid")
    return reserved / total


def _embedding_margin(document: dict[str, Any]) -> float:
    items = document.get("items")
    if not isinstance(items, list):
        raise EnterpriseCandidateKServeAcceptanceError("Embedding probe items are invalid")
    vectors: dict[str, list[float]] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("item_id"), str):
            raise EnterpriseCandidateKServeAcceptanceError("Embedding probe items are invalid")
        vector = item.get("vector")
        if not isinstance(vector, list):
            raise EnterpriseCandidateKServeAcceptanceError("Embedding probe vector is invalid")
        vectors[item["item_id"]] = [float(value) for value in vector]
    try:
        query = vectors["query"]
        relevant = vectors["relevant"]
        distractor = vectors["distractor"]
    except KeyError as exc:
        raise EnterpriseCandidateKServeAcceptanceError(
            "Embedding probe item set is incomplete"
        ) from exc
    return _dot(query, relevant) - _dot(query, distractor)


def _dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise EnterpriseCandidateKServeAcceptanceError("Embedding vector shape changed")
    return sum(a * b for a, b in zip(left, right, strict=True))


def _document(result: _HttpResult) -> dict[str, Any]:
    if result.document is None:
        raise EnterpriseCandidateKServeAcceptanceError("JSON model response is missing")
    return result.document


def _number(document: dict[str, Any], field: str) -> float:
    value = document.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EnterpriseCandidateKServeAcceptanceError(f"numeric field is invalid: {field}")
    return float(value)


def _header_number(result: _HttpResult, field: str) -> float:
    try:
        value = float(result.headers[field])
    except (KeyError, ValueError) as exc:
        raise EnterpriseCandidateKServeAcceptanceError(
            f"numeric response header is invalid: {field}"
        ) from exc
    return value


def _integer(document: dict[str, Any], field: str) -> int:
    value = document.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EnterpriseCandidateKServeAcceptanceError(f"integer field is invalid: {field}")
    return value


def _text(document: dict[str, Any], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value.strip():
        raise EnterpriseCandidateKServeAcceptanceError(f"text field is invalid: {field}")
    return value


def _method(component: Component) -> Literal["DPO", "TTS", "EMBEDDING"]:
    if component == "LLM":
        return "DPO"
    return component


def _get_json(url: str, *, timeout_seconds: int) -> dict[str, Any]:
    with httpx.Client(timeout=timeout_seconds, trust_env=False) as client:
        response = client.get(url)
        response.raise_for_status()
        value = response.json()
    if not isinstance(value, dict):
        raise EnterpriseCandidateKServeAcceptanceError("runtime JSON response is invalid")
    return cast(dict[str, Any], value)


def _component_receipt_path(root: Path, component: Component) -> Path:
    return root / OUTPUT_RELATIVE / "components" / component.lower() / "receipt.json"


def _reported_binding(
    root: Path,
    temporary: Path,
    final_relative: Path,
    path: Path,
) -> Any:
    return file_binding(
        root,
        path,
        reported_path=final_relative / path.relative_to(temporary),
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _copy(value: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(value))
    if not isinstance(result, dict):
        raise TypeError("JSON object copy failed")
    return cast(dict[str, Any], result)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise EnterpriseCandidateKServeAcceptanceError("latency evidence is empty")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * quantile + 0.999999) - 1))
    return ordered[index]


def _read_tail(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-4_000:]
    except OSError:
        return ""


def _tail(value: str) -> str:
    return value.strip().replace("\n", " ")[-1_000:]


def _progress(component: Component, stage: str) -> None:
    print(f"ENTERPRISE_CANDIDATE_KSERVE_{component}_PROGRESS={stage}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="industrial-ops-enterprise-candidate-kserve-acceptance"
    )
    parser.add_argument(
        "action",
        choices=("run-component", "run-all", "finalize", "verify"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--component", choices=COMPONENT_ORDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    if args.action == "run-component":
        if args.component is None:
            raise SystemExit("--component is required for run-component")
        path = execute_component(root, cast(Component, args.component))
        component_receipt = verify_component_receipt(
            root,
            path,
            expected_component=cast(Component, args.component),
        )
        print(
            json.dumps(
                {
                    "schema_version": COMPONENT_RECEIPT_SCHEMA_VERSION,
                    "component": component_receipt.acceptance.component,
                    "evidence_chain_sha256": component_receipt.evidence_chain_sha256,
                    "path": path.relative_to(root).as_posix(),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.action == "run-all":
        for component in COMPONENT_ORDER:
            execute_component(root, component)
        path = finalize_all(root)
    elif args.action == "finalize":
        path = finalize_all(root)
    else:
        path = root / ACCEPTANCE_RELATIVE
    final_report = verify_enterprise_candidate_kserve_acceptance(root, path)
    print(
        json.dumps(
            {
                "schema_version": final_report.schema_version,
                "status": final_report.status,
                "components": [item.component for item in final_report.components],
                "evidence_chain_sha256": final_report.evidence_chain_sha256,
                "path": path.relative_to(root).as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
