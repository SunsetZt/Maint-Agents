"""Actual llama.cpp traffic collection and fail-closed KServe rollout evidence."""

from __future__ import annotations

import json
import math
import socket
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

import httpx

from industrial_ops_agent.deployment.kserve import KubectlResourceApi
from industrial_ops_agent.training.gpu_promotion_gguf import (
    AUTHORIZATION_CLASSIFICATION,
    CLASSIFICATION,
    DATA_CLASSIFICATION,
    LLAMA_CPP_COMMIT,
    ROLE_ATTESTATION,
    GgufPromotionEvidence,
    load_gguf_promotion_evidence,
)

SCHEMA_VERSION = "local-gpu-promotion-kserve/v2"
STATUS = "GPU_MODEL_KSERVE_ROLLOUT_PASSED"
KUBERNETES_CONTEXT = "kind-ioap-gpu-promotion-lab"
KIND_NODE = "ioap-gpu-promotion-lab-control-plane"
KSERVE_VERSION = "v0.19.0"
NAMESPACE = "ioap-gpu-promotion-lab"
GATEWAY_NAME = "ioap-gpu-lab-gateway"
HOSTNAME = "gpu-lab.local"
ROUTE_NAME = "ioap-gpu-lab-route"
SERVICE_ACCOUNT_NAME = "model-storage-reader"
MODEL_PVC_NAME = "gpu-lab-gguf-models"
RUNTIME_PROFILE_ID = "llama-cpp-local-staging-v1"
CANARY_RATIO_BOUNDS = {
    "CANARY_5": (0.03, 0.07),
    "CANARY_25": (0.22, 0.28),
}

RolloutStage = Literal["SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"]


class GpuPromotionKServeError(RuntimeError):
    """An actual local rollout or its immutable evidence failed closed."""


@dataclass(frozen=True, slots=True)
class TrafficSample:
    stage: RolloutStage
    request_count: int
    critical_case_count: int
    stable_count: int
    candidate_count: int
    error_count: int
    unknown_count: int
    p95_latency_ms: float
    release_ids: tuple[str, ...]
    response_sha256s: tuple[str, ...]
    route_document_sha256: str
    runtime_metrics_sha256: str
    direct_candidate_release_id: str | None = None
    direct_candidate_output_sha256: str | None = None

    @property
    def candidate_ratio(self) -> float:
        successful = self.stable_count + self.candidate_count
        return self.candidate_count / successful if successful else 0.0

    @property
    def error_rate(self) -> float:
        return self.error_count / self.request_count if self.request_count else 0.0

    def evidence(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "candidate_ratio": self.candidate_ratio,
            "error_rate": self.error_rate,
        }


@dataclass(frozen=True, slots=True)
class FaultInjectionEvidence:
    fault_type: Literal["CANDIDATE_DEPLOYMENT_SCALED_TO_ZERO"]
    target_resource: str
    original_state_sha256: str
    injected_state_sha256: str
    observed_endpoint_count: int
    command_output_sha256: str


class ActualLlamaCppTrafficCollector:
    """Collect low-cost real Gateway traffic while preserving release identity."""

    def __init__(
        self,
        *,
        repo_root: Path,
        api: KubectlResourceApi,
        candidate_service_name: str,
        stable_release_id: str,
        candidate_release_id: str,
    ) -> None:
        self._root = repo_root.resolve(strict=True)
        self._api = api
        self._candidate_service = candidate_service_name
        self._stable_release_id = stable_release_id
        self._candidate_release_id = candidate_release_id
        self._processes: list[subprocess.Popen[str]] = []
        self._gateway_url: str | None = None
        self._candidate_url: str | None = None
        self._original_replicas: int | None = None

    def __enter__(self) -> ActualLlamaCppTrafficCollector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._original_replicas is not None:
            self.restore_candidate_service()
        for process in reversed(self._processes):
            _stop_process(process)
        self._processes.clear()
        self._gateway_url = None
        self._candidate_url = None

    def collect(
        self,
        stage: RolloutStage,
        *,
        request_count: int,
        critical_case_count: int,
    ) -> TrafficSample:
        if request_count < 1 or not 0 <= critical_case_count <= request_count:
            raise ValueError("traffic sample counts are invalid")
        self._wait_route_reconciled()
        self._ensure_port_forwards()
        assert self._gateway_url is not None
        direct_release_id: str | None = None
        direct_output_sha256: str | None = None
        if stage == "SHADOW":
            direct_release_id, direct_output_sha256 = self._direct_candidate_probe()

        counts: Counter[str] = Counter()
        response_hashes: set[str] = set()
        latencies_ms: list[float] = []
        with httpx.Client(timeout=httpx.Timeout(30.0), trust_env=False) as client:
            for index in range(request_count):
                started = time.perf_counter()
                try:
                    if index < critical_case_count:
                        response = client.post(
                            f"{self._gateway_url}/v1/chat/completions",
                            headers={"Host": HOSTNAME},
                            json=_critical_probe_payload(stage, index),
                        )
                    else:
                        response = client.get(
                            f"{self._gateway_url}/v1/models",
                            headers={"Host": HOSTNAME},
                        )
                    latencies_ms.append(max((time.perf_counter() - started) * 1000.0, 0.001))
                    if response.status_code >= 400:
                        counts["error"] += 1
                        continue
                    document = _json_object(response)
                    release_id = _response_release_id(document)
                    if release_id == self._stable_release_id:
                        counts["stable"] += 1
                    elif release_id == self._candidate_release_id:
                        counts["candidate"] += 1
                    else:
                        counts["unknown"] += 1
                    response_hashes.add(_digest(document))
                except httpx.HTTPError:
                    latencies_ms.append(max((time.perf_counter() - started) * 1000.0, 0.001))
                    counts["error"] += 1

        route = self._api.get(
            api_version="gateway.networking.k8s.io/v1",
            plural="httproutes",
            namespace=NAMESPACE,
            name=ROUTE_NAME,
        )
        metrics = self._runtime_metrics()
        sample = TrafficSample(
            stage=stage,
            request_count=request_count,
            critical_case_count=critical_case_count,
            stable_count=counts["stable"],
            candidate_count=counts["candidate"],
            error_count=counts["error"],
            unknown_count=counts["unknown"],
            p95_latency_ms=_percentile(latencies_ms, 0.95),
            release_ids=tuple(
                release_id
                for release_id, count in (
                    (self._stable_release_id, counts["stable"]),
                    (self._candidate_release_id, counts["candidate"]),
                )
                if count > 0
            ),
            response_sha256s=tuple(sorted(response_hashes)),
            route_document_sha256=_digest(route),
            runtime_metrics_sha256=_digest(metrics),
            direct_candidate_release_id=direct_release_id,
            direct_candidate_output_sha256=direct_output_sha256,
        )
        _validate_traffic_sample(sample)
        return sample

    def _wait_route_reconciled(self, timeout_seconds: int = 180) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            route = self._api.get(
                api_version="gateway.networking.k8s.io/v1",
                plural="httproutes",
                namespace=NAMESPACE,
                name=ROUTE_NAME,
            )
            if _route_has_current_acceptance(route):
                return
            time.sleep(1)
        raise GpuPromotionKServeError("HTTPRoute_reconciliation_timed_out")

    def inject_candidate_service_fault(self) -> FaultInjectionEvidence:
        if self._original_replicas is not None:
            raise GpuPromotionKServeError("candidate_service_fault_is_already_active")
        deployment_name = f"{self._candidate_service}-predictor"
        deployment = self._kubectl_json(
            "--namespace",
            NAMESPACE,
            "get",
            "deployment",
            deployment_name,
            "--output=json",
        )
        replicas = deployment.get("spec", {}).get("replicas")
        if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 1:
            raise GpuPromotionKServeError("candidate_deployment_replicas_are_invalid")
        self._original_replicas = replicas
        output = self._kubectl(
            "--namespace",
            NAMESPACE,
            "scale",
            "deployment",
            deployment_name,
            "--replicas=0",
        )
        endpoint_count = self._wait_endpoint_count(0, timeout_seconds=45)
        return FaultInjectionEvidence(
            fault_type="CANDIDATE_DEPLOYMENT_SCALED_TO_ZERO",
            target_resource=f"deployment/{deployment_name}",
            original_state_sha256=_digest({"replicas": replicas}),
            injected_state_sha256=_digest({"replicas": 0}),
            observed_endpoint_count=endpoint_count,
            command_output_sha256=sha256(output.encode()).hexdigest(),
        )

    def restore_candidate_service(self) -> None:
        if self._original_replicas is None:
            return
        replicas = self._original_replicas
        deployment_name = f"{self._candidate_service}-predictor"
        self._kubectl(
            "--namespace",
            NAMESPACE,
            "scale",
            "deployment",
            deployment_name,
            f"--replicas={replicas}",
        )
        self._original_replicas = None

    def resource_projection(
        self,
        *,
        stable_service_name: str,
        candidate_service_name: str,
    ) -> dict[str, Any]:
        coordinates = {
            "stable_inference_service": (
                "serving.kserve.io/v1beta1",
                "inferenceservices",
                stable_service_name,
            ),
            "candidate_inference_service": (
                "serving.kserve.io/v1beta1",
                "inferenceservices",
                candidate_service_name,
            ),
            "candidate_serving_runtime": (
                "serving.kserve.io/v1alpha1",
                "servingruntimes",
                f"{candidate_service_name}-runtime",
            ),
            "stable_serving_runtime": (
                "serving.kserve.io/v1alpha1",
                "servingruntimes",
                f"{stable_service_name.removesuffix('-predictor')}-runtime",
            ),
            "route": (
                "gateway.networking.k8s.io/v1",
                "httproutes",
                ROUTE_NAME,
            ),
        }
        projection: dict[str, Any] = {}
        for key, (api_version, plural, name) in coordinates.items():
            document = self._api.get(
                api_version=api_version,
                plural=plural,
                namespace=NAMESPACE,
                name=name,
            )
            metadata = document.get("metadata", {})
            projection[key] = {
                "name": name,
                "uid": metadata.get("uid"),
                "generation": metadata.get("generation"),
                "resource_version": metadata.get("resourceVersion"),
                "document_sha256": _digest(document),
            }
        return projection

    def _direct_candidate_probe(self) -> tuple[str, str]:
        assert self._candidate_url is not None
        with httpx.Client(timeout=httpx.Timeout(60.0), trust_env=False) as client:
            response = client.post(
                f"{self._candidate_url}/v1/chat/completions",
                json=_critical_probe_payload("SHADOW", -1),
            )
            response.raise_for_status()
            document = _json_object(response)
        release_id = _response_release_id(document)
        if release_id != self._candidate_release_id:
            raise GpuPromotionKServeError("direct_candidate_probe_release_id_mismatch")
        return release_id, _digest(document)

    def _runtime_metrics(self) -> dict[str, Any]:
        assert self._candidate_url is not None
        try:
            with httpx.Client(timeout=10.0, trust_env=False) as client:
                response = client.get(f"{self._candidate_url}/metrics")
                response.raise_for_status()
            return {
                "available": True,
                "content_sha256": sha256(response.content).hexdigest(),
                "content_length": len(response.content),
                "status_code": response.status_code,
            }
        except httpx.HTTPError as exc:
            # Candidate metrics are expected to become unreachable while its Service
            # selector is deliberately faulted. Persist the observed unavailability
            # instead of aborting before the existing rollback FSM sees the fault.
            return {
                "available": False,
                "error_type": type(exc).__name__,
                "probe": "candidate-runtime-metrics",
            }

    def _ensure_port_forwards(self) -> None:
        if self._gateway_url is not None and self._candidate_url is not None:
            return
        gateway_namespace, gateway_service = self._gateway_service()
        gateway_port = _free_port()
        candidate_port = _free_port()
        self._processes.extend(
            (
                self._start_port_forward(
                    namespace=gateway_namespace,
                    resource=f"service/{gateway_service}",
                    local_port=gateway_port,
                ),
                self._start_port_forward(
                    namespace=NAMESPACE,
                    resource=f"service/{self._candidate_service}-predictor",
                    local_port=candidate_port,
                ),
            )
        )
        self._gateway_url = f"http://127.0.0.1:{gateway_port}"
        self._candidate_url = f"http://127.0.0.1:{candidate_port}"

    def _gateway_service(self) -> tuple[str, str]:
        document = self._kubectl_json("get", "service", "--all-namespaces", "--output=json")
        matches: list[tuple[str, str]] = []
        for item in document.get("items", []):
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata", {})
            labels = metadata.get("labels", {})
            if (
                isinstance(labels, dict)
                and labels.get("gateway.envoyproxy.io/owning-gateway-name") == GATEWAY_NAME
                and labels.get("gateway.envoyproxy.io/owning-gateway-namespace") == NAMESPACE
                and isinstance(metadata.get("namespace"), str)
                and isinstance(metadata.get("name"), str)
            ):
                matches.append((metadata["namespace"], metadata["name"]))
        if len(matches) != 1:
            raise GpuPromotionKServeError("gateway_service_is_not_unique")
        return matches[0]

    def _start_port_forward(
        self,
        *,
        namespace: str,
        resource: str,
        local_port: int,
    ) -> subprocess.Popen[str]:
        process = subprocess.Popen(
            [
                "kubectl",
                "--context",
                KUBERNETES_CONTEXT,
                "--namespace",
                namespace,
                "port-forward",
                resource,
                f"{local_port}:80",
                "--address=127.0.0.1",
            ],
            cwd=self._root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout is not None else ""
                raise GpuPromotionKServeError("kubectl_port_forward_failed:" + output[-256:])
            try:
                with socket.create_connection(("127.0.0.1", local_port), timeout=0.5):
                    return process
            except OSError:
                time.sleep(0.25)
        _stop_process(process)
        raise GpuPromotionKServeError("kubectl_port_forward_timed_out")

    def _wait_endpoint_count(self, expected: int, *, timeout_seconds: int) -> int:
        deadline = time.monotonic() + timeout_seconds
        observed = -1
        while time.monotonic() < deadline:
            document = self._kubectl_json(
                "--namespace",
                NAMESPACE,
                "get",
                "endpoints",
                f"{self._candidate_service}-predictor",
                "--output=json",
            )
            observed = sum(
                len(subset.get("addresses", []))
                for subset in document.get("subsets", [])
                if isinstance(subset, dict) and isinstance(subset.get("addresses", []), list)
            )
            if observed == expected:
                return observed
            time.sleep(0.5)
        raise GpuPromotionKServeError(
            f"candidate_endpoint_count_did_not_reach_{expected}:observed={observed}"
        )

    def _kubectl_json(self, *arguments: str) -> dict[str, Any]:
        output = self._kubectl(*arguments)
        try:
            value = json.loads(output)
        except json.JSONDecodeError as exc:
            raise GpuPromotionKServeError("kubectl_output_is_not_JSON") from exc
        if not isinstance(value, dict):
            raise GpuPromotionKServeError("kubectl_output_is_not_JSON")
        return cast(dict[str, Any], value)

    def _kubectl(self, *arguments: str) -> str:
        try:
            result = subprocess.run(
                ["kubectl", "--context", KUBERNETES_CONTEXT, *arguments],
                cwd=self._root,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GpuPromotionKServeError("kubectl_command_failed") from exc
        if result.returncode != 0:
            raise GpuPromotionKServeError("kubectl_command_failed")
        return result.stdout


def finalize_rollout_evidence(document: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(document)
    unsigned.pop("evidence_chain_sha256", None)
    return {**unsigned, "evidence_chain_sha256": _digest(unsigned)}


def write_rollout_evidence(path: Path, document: dict[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def load_rollout_evidence(
    path: Path,
    *,
    repo_root: Path,
    gguf_evidence_path: Path,
    training_evidence_path: Path,
) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GpuPromotionKServeError("rollout_evidence_is_invalid") from exc
    if not isinstance(document, dict):
        raise GpuPromotionKServeError("rollout_evidence_is_invalid")
    chain = document.get("evidence_chain_sha256")
    unsigned = dict(document)
    unsigned.pop("evidence_chain_sha256", None)
    serialized = json.dumps(document, sort_keys=True, separators=(",", ":"))
    if (
        chain != _digest(unsigned)
        or document.get("schema_version") != SCHEMA_VERSION
        or document.get("status") != STATUS
        or document.get("classification") != CLASSIFICATION
        or document.get("data_classification") != DATA_CLASSIFICATION
        or document.get("authorization_classification") != AUTHORIZATION_CLASSIFICATION
        or document.get("role_attestation") != ROLE_ATTESTATION
        or document.get("production_claim") is not False
        or document.get("enterprise_production_data") is not False
        or document.get("actual_kserve_execution") is not True
        or document.get("actual_llama_cpp_inference") is not True
        or "ENTERPRISE_PRODUCTION_ACCEPTED" in serialized
    ):
        raise GpuPromotionKServeError("rollout_evidence_contract_changed")
    gguf = load_gguf_promotion_evidence(
        gguf_evidence_path,
        repo_root=repo_root,
        training_evidence_path=training_evidence_path,
    )
    _verify_source(document.get("source_gguf"), gguf)
    _verify_releases(document.get("releases"), gguf)
    _verify_stages(document.get("stages"), document.get("releases"), gguf)
    _verify_final_state(document.get("final_state"), document.get("releases"), gguf)
    kubernetes = document.get("kubernetes")
    runtime = document.get("runtime")
    if (
        not isinstance(kubernetes, dict)
        or kubernetes.get("context") != KUBERNETES_CONTEXT
        or kubernetes.get("kserve_version") != KSERVE_VERSION
        or kubernetes.get("deployment_mode") != "Standard"
        or not isinstance(kubernetes.get("resources"), dict)
        or not isinstance(runtime, dict)
        or runtime.get("llama_cpp_commit") != LLAMA_CPP_COMMIT
        or not _image_digest(runtime.get("image_digest"))
        or not _sha256(runtime.get("dockerfile_sha256"))
    ):
        raise GpuPromotionKServeError("rollout_runtime_binding_failed")
    _verify_kubernetes_resources(kubernetes["resources"])
    return document


def _verify_source(value: object, gguf: GgufPromotionEvidence) -> None:
    if not isinstance(value, dict) or (
        value.get("evidence_chain_sha256") != gguf.evidence_chain_sha256
        or value.get("training_evidence_chain_sha256")
        != gguf.training_evidence_chain_sha256
    ):
        raise GpuPromotionKServeError("rollout_GGUF_source_binding_failed")


def _verify_releases(value: object, gguf: GgufPromotionEvidence) -> None:
    if not isinstance(value, dict) or set(value) != {"stable", "candidate"}:
        raise GpuPromotionKServeError("rollout_release_evidence_is_invalid")
    expected = {"stable": gguf.stable, "candidate": gguf.candidate}
    release_ids: set[str] = set()
    for role, variant in expected.items():
        item = value.get(role)
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("release_id"), str)
            or not item["release_id"].startswith("model-release-")
            or not _sha256(item.get("manifest_hash"))
            or item.get("evaluation_id") != variant.edge_evaluation.evaluation_id
            or item.get("quantization_experiment_id") != variant.quantization_experiment_id
            or item.get("model_file_sha256") != variant.model_file_sha256
            or item.get("approval_requested_by_subject_id")
            == item.get("approval_decided_by_subject_id")
            or not isinstance(item.get("approval_id"), str)
        ):
            raise GpuPromotionKServeError("rollout_release_binding_failed")
        release_ids.add(cast(str, item["release_id"]))
    if len(release_ids) != 2:
        raise GpuPromotionKServeError("rollout_release_identities_are_not_distinct")


def _verify_stages(
    value: object,
    releases: object,
    gguf: GgufPromotionEvidence,
) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"}
        or not isinstance(releases, dict)
    ):
        raise GpuPromotionKServeError("rollout_stage_evidence_is_invalid")
    stable_id = releases["stable"]["release_id"]
    candidate_id = releases["candidate"]["release_id"]
    shadow = _traffic(value["SHADOW"], "SHADOW")
    canary_5 = _traffic(value["CANARY_5"], "CANARY_5")
    canary_25_value = value["CANARY_25"]
    rollback = _traffic(value["ROLLED_BACK"], "ROLLED_BACK")
    if not isinstance(canary_25_value, dict):
        raise GpuPromotionKServeError("rollout_canary_25_evidence_is_invalid")
    routing = _traffic(canary_25_value.get("routing"), "CANARY_25")
    fault = _traffic(canary_25_value.get("fault_sampling"), "CANARY_25")
    fault_injection = canary_25_value.get("fault_injection")
    if (
        shadow["request_count"] < 500
        or shadow["critical_case_count"] < 50
        or shadow["stable_count"] != shadow["request_count"]
        or shadow["candidate_count"] != 0
        or shadow["error_count"] != 0
        or shadow.get("direct_candidate_release_id") != candidate_id
        or not _sha256(shadow.get("direct_candidate_output_sha256"))
        or canary_5["request_count"] < 1000
        or canary_5["critical_case_count"] < 100
        or not CANARY_RATIO_BOUNDS["CANARY_5"][0]
        <= float(canary_5["candidate_ratio"])
        <= CANARY_RATIO_BOUNDS["CANARY_5"][1]
        or canary_5["error_count"] != 0
        or routing["request_count"] < 5000
        or routing["critical_case_count"] < 250
        or not CANARY_RATIO_BOUNDS["CANARY_25"][0]
        <= float(routing["candidate_ratio"])
        <= CANARY_RATIO_BOUNDS["CANARY_25"][1]
        or routing["error_count"] != 0
        or float(shadow["p95_latency_ms"]) > 5000.0
        or float(canary_5["p95_latency_ms"]) > 5000.0
        or float(routing["p95_latency_ms"]) > 5000.0
        or fault["request_count"] < 500
        or float(fault["error_rate"]) <= 0.01
        or rollback["request_count"] < 500
        or rollback["stable_count"] != rollback["request_count"]
        or rollback["candidate_count"] != 0
        or rollback["error_count"] != 0
        or shadow["release_ids"] != [stable_id]
        or set(canary_5["release_ids"]) != {stable_id, candidate_id}
        or set(routing["release_ids"]) != {stable_id, candidate_id}
        or rollback["release_ids"] != [stable_id]
        or value["SHADOW"].get("observation_decision") != "PASS"
        or value["CANARY_5"].get("observation_decision") != "PASS"
        or canary_25_value.get("observation_decision") != "ROLLBACK"
        or not isinstance(fault_injection, dict)
        or fault_injection.get("fault_type")
        != "CANDIDATE_DEPLOYMENT_SCALED_TO_ZERO"
        or not isinstance(fault_injection.get("target_resource"), str)
        or not str(fault_injection["target_resource"]).startswith("deployment/")
        or not _sha256(fault_injection.get("original_state_sha256"))
        or not _sha256(fault_injection.get("injected_state_sha256"))
        or not _sha256(fault_injection.get("command_output_sha256"))
        or fault_injection.get("observed_endpoint_count") != 0
        or value["ROLLED_BACK"].get("stable_model_file_sha256")
        != gguf.stable.model_file_sha256
    ):
        raise GpuPromotionKServeError("rollout_stage_gate_failed")


def _verify_final_state(
    value: object,
    releases: object,
    gguf: GgufPromotionEvidence,
) -> None:
    if not isinstance(value, dict) or not isinstance(releases, dict) or (
        value.get("release_status") != "ROLLED_BACK"
        or value.get("deployment_status") != "READY"
        or value.get("deployment_stage") != "ROLLED_BACK"
        or value.get("candidate_traffic_percent") != 0.0
        or value.get("stable_release_id") != releases["stable"]["release_id"]
        or value.get("candidate_release_id") != releases["candidate"]["release_id"]
        or value.get("stable_model_file_sha256") != gguf.stable.model_file_sha256
        or value.get("production_alias_created") is not False
    ):
        raise GpuPromotionKServeError("rollout_final_state_gate_failed")


def _verify_kubernetes_resources(value: object) -> None:
    expected = {
        "stable_inference_service",
        "candidate_inference_service",
        "stable_serving_runtime",
        "candidate_serving_runtime",
        "route",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise GpuPromotionKServeError("rollout_kubernetes_resources_are_incomplete")
    identities: set[tuple[str, str]] = set()
    for key, item in value.items():
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not item["name"]
            or not isinstance(item.get("uid"), str)
            or not item["uid"]
            or isinstance(item.get("generation"), bool)
            or not isinstance(item.get("generation"), int)
            or item["generation"] < 1
            or not isinstance(item.get("resource_version"), str)
            or not item["resource_version"]
            or not _sha256(item.get("document_sha256"))
        ):
            raise GpuPromotionKServeError("rollout_kubernetes_resource_binding_failed")
        identity = (str(key).rsplit("_", 2)[-1], cast(str, item["name"]))
        if identity in identities:
            raise GpuPromotionKServeError("rollout_kubernetes_resource_identity_reused")
        identities.add(identity)


def _traffic(value: object, stage: RolloutStage) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("stage") != stage:
        raise GpuPromotionKServeError("rollout_traffic_evidence_is_invalid")
    integer_fields = (
        "request_count",
        "critical_case_count",
        "stable_count",
        "candidate_count",
        "error_count",
        "unknown_count",
    )
    if any(
        isinstance(value.get(field), bool)
        or not isinstance(value.get(field), int)
        or int(value[field]) < 0
        for field in integer_fields
    ):
        raise GpuPromotionKServeError("rollout_traffic_counts_are_invalid")
    successful = int(value["stable_count"]) + int(value["candidate_count"])
    expected_candidate_ratio = (
        int(value["candidate_count"]) / successful if successful else 0.0
    )
    expected_error_rate = (
        int(value["error_count"]) / int(value["request_count"])
        if int(value["request_count"])
        else 0.0
    )
    if (
        value["stable_count"]
        + value["candidate_count"]
        + value["error_count"]
        + value["unknown_count"]
        != value["request_count"]
        or value["critical_case_count"] > value["request_count"]
        or value["unknown_count"] != 0
        or not _finite_non_negative(value.get("p95_latency_ms"))
        or not _finite_non_negative(value.get("candidate_ratio"))
        or not _finite_non_negative(value.get("error_rate"))
        or not math.isclose(
            float(value["candidate_ratio"]),
            expected_candidate_ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(value["error_rate"]),
            expected_error_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not _sha256(value.get("route_document_sha256"))
        or not _sha256(value.get("runtime_metrics_sha256"))
        or not isinstance(value.get("release_ids"), list)
        or not isinstance(value.get("response_sha256s"), list)
        or not value["response_sha256s"]
        or any(not _sha256(item) for item in value["response_sha256s"])
        or any(not isinstance(item, str) or not item for item in value["release_ids"])
        or len(value["release_ids"]) != len(set(value["release_ids"]))
    ):
        raise GpuPromotionKServeError("rollout_traffic_binding_failed")
    return value


def _validate_traffic_sample(sample: TrafficSample) -> None:
    total = sample.stable_count + sample.candidate_count + sample.error_count + sample.unknown_count
    if (
        total != sample.request_count
        or sample.unknown_count != 0
        or not math.isfinite(sample.p95_latency_ms)
        or sample.p95_latency_ms < 0
        or not sample.response_sha256s
        or any(not _sha256(value) for value in sample.response_sha256s)
    ):
        raise GpuPromotionKServeError("actual_traffic_sample_failed")


def _route_has_current_acceptance(document: dict[str, Any]) -> bool:
    metadata = document.get("metadata")
    status = document.get("status")
    if not isinstance(metadata, dict) or not isinstance(status, dict):
        return False
    generation = metadata.get("generation")
    parents = status.get("parents")
    if not isinstance(generation, int) or generation < 1 or not isinstance(parents, list):
        return False
    for parent in parents:
        if not isinstance(parent, dict):
            continue
        parent_ref = parent.get("parentRef")
        conditions = parent.get("conditions")
        if (
            not isinstance(parent_ref, dict)
            or parent_ref.get("name") != GATEWAY_NAME
            or not isinstance(conditions, list)
        ):
            continue
        by_type = {
            condition.get("type"): condition
            for condition in conditions
            if isinstance(condition, dict)
        }
        if all(
            isinstance(by_type.get(condition_type), dict)
            and by_type[condition_type].get("status") == "True"
            and by_type[condition_type].get("observedGeneration") == generation
            for condition_type in ("Accepted", "ResolvedRefs")
        ):
            return True
    return False


def _critical_probe_payload(stage: RolloutStage, index: int) -> dict[str, Any]:
    return {
        "model": "industrial-ops-local-staging",
        "messages": [
            {
                "role": "user",
                "content": f"Return OK. stage={stage}; synthetic_case={index}",
            }
        ],
        "temperature": 0,
        "max_tokens": 1,
        "stream": False,
    }


def _response_release_id(document: dict[str, Any]) -> str:
    model = document.get("model")
    if isinstance(model, str) and model:
        return model
    data = document.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        value = data[0].get("id")
        if isinstance(value, str) and value:
            return value
    raise GpuPromotionKServeError("llama_cpp_response_has_no_release_id")


def _json_object(response: httpx.Response) -> dict[str, Any]:
    try:
        value = response.json()
    except ValueError as exc:
        raise GpuPromotionKServeError("llama_cpp_response_is_not_JSON") from exc
    if not isinstance(value, dict):
        raise GpuPromotionKServeError("llama_cpp_response_is_not_JSON")
    return cast(dict[str, Any], value)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise GpuPromotionKServeError("traffic_latency_evidence_is_empty")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * quantile)))
    return ordered[index]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _digest(value: object) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value.removeprefix("sha256:")) == 64 and all(
        character in "0123456789abcdef" for character in value.removeprefix("sha256:")
    )


def _image_digest(value: object) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and _sha256(value)


def _finite_non_negative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )
