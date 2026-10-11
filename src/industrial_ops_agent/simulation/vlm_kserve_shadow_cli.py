"""Run and verify the project-authorized VLM KServe rollout acceptance."""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

import httpx

from industrial_ops_agent.multimodal.vlm_output import build_vlm_findings_instruction
from industrial_ops_agent.simulation.vlm_enterprise_staging import (
    verify_enterprise_vlm_staging_adoption,
)
from industrial_ops_agent.simulation.vlm_kserve_shadow import (
    ADAPTER_RELATIVE,
    ADOPTION_RELATIVE,
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
    STABLE_SERVICE,
    VlmKServeRolloutError,
    VlmRolloutStageEvidence,
    build_project_release_manifest,
    document_sha256,
    file_sha256,
    finalize_report,
    inference_service_document,
    proxy_config_map_document,
    route_condition_is_true,
    route_document,
    verify_vlm_kserve_rollout,
    write_rollout_report,
)

KIND_CLUSTER = "ioap-gpu-promotion-lab"
KUBE_CONTEXT = f"kind-{KIND_CLUSTER}"
KIND_NODE = f"{KIND_CLUSTER}-control-plane"
DATASET_RELATIVE = Path("artifacts/m7-vlm-curriculum-dataset")
WORKER_LOG_NAME = "gpu-worker.log"


def execute_rollout(
    repo_root: Path,
    *,
    stop_cluster_after_run: bool,
) -> Path:
    root = repo_root.resolve(strict=True)
    output = root / OUTPUT_RELATIVE
    output.mkdir(parents=True, exist_ok=True)
    kubernetes_output = output / "kubernetes"
    kubernetes_output.mkdir(parents=True, exist_ok=True)
    adoption_path = root / ADOPTION_RELATIVE
    adoption = verify_enterprise_vlm_staging_adoption(root, adoption_path)
    source_adoption_sha256 = file_sha256(adoption_path)
    release_manifest = build_project_release_manifest(
        adoption,
        source_adoption_sha256=source_adoption_sha256,
    )
    release_manifest_path = output / "release-manifest.json"
    _write_json(release_manifest_path, release_manifest)
    proxy_path = (root / PROXY_SOURCE_RELATIVE).resolve(strict=True)
    proxy_source = proxy_path.read_text(encoding="utf-8")
    adapter_path = (root / ADAPTER_RELATIVE).resolve(strict=True)
    case, request_payload, expected_label, expected_region = _frozen_request(root)

    _ensure_cluster_running(root)
    proxy_image_digest = _node_image_digest(root)
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
    worker_log_path = output / WORKER_LOG_NAME
    worker_log = worker_log_path.open("w", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    worker = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-m",
            "industrial_ops_agent.multimodal.vlm_runtime",
            "--repo-root",
            str(root),
            "--port",
            str(worker_port),
        ],
        cwd=root,
        env={**environment, "PYTHONPATH": str(root / "src")},
        stdout=worker_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    resources_applied = False
    port_forwards_stopped = False
    resources_removed = False
    cluster_stopped = False
    report_path = output / "acceptance.json"
    try:
        worker_url = f"http://127.0.0.1:{worker_port}"
        _wait_worker(worker, worker_url, timeout_seconds=240)
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
        gateway_namespace, gateway_service = _gateway_service(root)
        stable_port, candidate_port, gateway_port = (_free_port() for _ in range(3))
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
            direct_stable = _post_json(
                f"http://127.0.0.1:{stable_port}{ROUTE_PATH_PREFIX}/predict",
                request_payload,
                timeout_seconds=300,
            )
            direct_candidate = _post_json(
                f"http://127.0.0.1:{candidate_port}{ROUTE_PATH_PREFIX}/predict",
                request_payload,
                timeout_seconds=300,
            )
            candidate_iou = _validate_candidate_quality(
                direct_candidate,
                expected_label=expected_label,
                expected_region=expected_region,
            )
            stage_reports: list[VlmRolloutStageEvidence] = []
            gateway_url = (
                f"http://127.0.0.1:{gateway_port}{ROUTE_PATH_PREFIX}/predict"
            )
            for stage, request_count in (
                ("SHADOW", 5),
                ("CANARY_5", 200),
                ("CANARY_25", 120),
                ("ROLLED_BACK", 40),
            ):
                route = route_document(cast(Any, stage))
                _write_json(kubernetes_output / f"route-{stage.lower()}.json", route)
                _kubectl_apply(root, route)
                _wait_route(root, timeout_seconds=60)
                before = _get_json(f"{worker_url}/metrics", timeout_seconds=5)
                response_counter: Counter[str] = Counter()
                response_hashes: set[str] = set()
                for index in range(request_count):
                    response, variant = _post_gateway(
                        gateway_url,
                        {**request_payload, "request_id": f"m7-vlm-{stage.lower()}-{index}"},
                    )
                    response_counter[variant] += 1
                    response_hash = response.get("response_sha256")
                    if isinstance(response_hash, str):
                        response_hashes.add(response_hash)
                if stage == "SHADOW":
                    _wait_candidate_delta(worker_url, before, minimum=1, timeout_seconds=30)
                after = _get_json(f"{worker_url}/metrics", timeout_seconds=5)
                stage_reports.append(
                    _stage_evidence(
                        cast(Any, stage),
                        request_count=request_count,
                        before=before,
                        after=after,
                        responses=response_counter,
                        response_hashes=response_hashes,
                        route=route,
                    )
                )
            port_forwards_stopped = True
        route_state = _kubectl_json(root, "httproute", ROUTE_NAME)
        stable_state = _kubectl_json(root, "inferenceservice", STABLE_SERVICE)
        candidate_state = _kubectl_json(root, "inferenceservice", CANDIDATE_SERVICE)
        metrics = _get_json(f"{worker_url}/metrics", timeout_seconds=5)
        runtime_versions = _runtime_versions()
        _delete_vlm_resources(root)
        resources_removed = True
        _stop_process(worker)
        worker_log.close()
        if stop_cluster_after_run:
            _run_checked(["docker", "stop", KIND_NODE], cwd=root)
            cluster_stopped = True
        unsigned: dict[str, Any] = {
            "schema_version": "enterprise-vlm-kserve-rollout/v1",
            "classification": "LOCAL_STAGING_PROJECT_AUTHORIZED",
            "enterprise_scope": "PROJECT_INTERNAL",
            "production_claim": False,
            "external_enterprise_production_claim": False,
            "generated_at": datetime.now(UTC).isoformat(),
            "status": "VLM_LOCAL_KSERVE_SHADOW_CANARY_ROLLBACK_PASSED",
            "release": {
                "schema_version": release_manifest["schema_version"],
                "release_id": release_manifest["release_id"],
                "manifest_path": release_manifest_path.relative_to(root).as_posix(),
                "manifest_sha256": file_sha256(release_manifest_path),
                "target_environment": "ENTERPRISE_STAGING",
                "release_scope": "PROJECT_INTERNAL",
                "candidate_experiment_id": adoption.candidate.experiment_id,
                "evaluation_suite_id": adoption.candidate.evaluation_suite_id,
                "source_adoption_path": ADOPTION_RELATIVE.as_posix(),
                "source_adoption_sha256": source_adoption_sha256,
                "source_evidence_chain_sha256": adoption.evidence_chain_sha256,
                "adapter_bundle_sha256": adoption.candidate.artifact_sha256,
                "formal_project_release_identity_created": True,
                "external_production_release_created": False,
            },
            "runtime": {
                "topology": "WSL_HOST_GPU_BEHIND_KSERVE_PROXY",
                "actual_gpu_execution": True,
                "model_execution_simulated": False,
                "model_id": adoption.candidate.model_id,
                "model_revision": adoption.candidate.model_revision,
                "adapter_model_sha256": file_sha256(adapter_path),
                "gpu_name": _text(metrics, "gpu_name"),
                "gpu_total_memory_bytes": _positive_integer(
                    metrics, "gpu_total_memory_bytes"
                ),
                "gpu_peak_reserved_memory_bytes": _positive_integer(
                    metrics, "gpu_peak_memory_reserved_bytes"
                ),
                "torch_version": runtime_versions["torch"],
                "transformers_version": runtime_versions["transformers"],
                "peft_version": runtime_versions["peft"],
                "stable_actual_inferences": _variant_count(
                    metrics, "actual_inference_counts", "stable"
                ),
                "candidate_actual_inferences": _variant_count(
                    metrics, "actual_inference_counts", "candidate"
                ),
                "replay_cache_used_for_route_load": True,
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
                "route_accepted": route_condition_is_true(route_state, "Accepted"),
                "route_resolved_refs": route_condition_is_true(
                    route_state, "ResolvedRefs"
                ),
                "proxy_image": PROXY_IMAGE,
                "proxy_image_digest": proxy_image_digest,
                "proxy_source_sha256": file_sha256(proxy_path),
                "hardened_container_security_context": True,
                "kserve_shadow_endpoint_verified": True,
                "kserve_gpu_pod_verified": False,
                "endpoint_scope": "LOCAL_PORT_FORWARD",
            },
            "quality": {
                "case_id": case["case_id"],
                "image_sha256": case["image"]["sha256"],
                "expected_label": expected_label,
                "stable_response_sha256": _text(direct_stable, "response_sha256"),
                "candidate_response_sha256": _text(
                    direct_candidate, "response_sha256"
                ),
                "candidate_structured_output_valid": True,
                "candidate_expected_label_present": True,
                "candidate_region_iou": candidate_iou,
                "human_confirmation_required": True,
            },
            "stages": [item.model_dump(mode="json") for item in stage_reports],
            "cleanup": {
                "gpu_worker_stopped": True,
                "port_forwards_stopped": port_forwards_stopped,
                "vlm_kubernetes_resources_removed": resources_removed,
                "cluster_stopped_after_acceptance": cluster_stopped,
                "evidence_is_historical_after_cleanup": True,
            },
        }
        report = finalize_report(unsigned)
        write_rollout_report(report, report_path)
        verify_vlm_kserve_rollout(root, report_path)
        return report_path
    finally:
        if resources_applied and not resources_removed:
            _delete_vlm_resources(root, check=False)
        if worker.poll() is None:
            _stop_process(worker)
        if not worker_log.closed:
            worker_log.close()


def _frozen_request(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, float]]:
    cases_path = root / DATASET_RELATIVE / "cases/simulation_evaluation.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
    if len(cases) != 6 or any(not isinstance(item, dict) for item in cases):
        raise VlmKServeRolloutError("frozen VLM evaluation cases are invalid")
    case = cast(dict[str, Any], cases[0])
    labels = tuple(
        sorted(
            {
                finding["label"]
                for item in cases
                for finding in item["target"]["expected_findings"]
            }
        )
    )
    image_path = (root / DATASET_RELATIVE / case["image"]["path"]).resolve(strict=True)
    image = image_path.read_bytes()
    if sha256(image).hexdigest() != case["image"]["sha256"]:
        raise VlmKServeRolloutError("frozen VLM image digest changed")
    instruction = build_vlm_findings_instruction(
        "定位图中异常区域并输出受控可见故障标签。",
        allowed_labels=labels,
    )
    finding = case["target"]["expected_findings"][0]
    request = {
        "request_id": "m7-vlm-direct-probe",
        "case_id": f"sim-feedback-{case['case_id']}",
        "instruction": instruction,
        "allowed_labels": list(labels),
        "image_base64": base64.b64encode(image).decode(),
        "image_sha256": case["image"]["sha256"],
        "max_new_tokens": 96,
    }
    return case, request, finding["label"], finding["region"]


def _stage_evidence(
    stage: Literal["SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"],
    *,
    request_count: int,
    before: dict[str, Any],
    after: dict[str, Any],
    responses: Counter[str],
    response_hashes: set[str],
    route: dict[str, Any],
) -> VlmRolloutStageEvidence:
    stable_delta = _variant_count(after, "request_counts", "stable") - _variant_count(
        before, "request_counts", "stable"
    )
    candidate_delta = _variant_count(
        after, "request_counts", "candidate"
    ) - _variant_count(before, "request_counts", "candidate")
    if stage == "SHADOW":
        passed = (
            responses == {"stable": request_count}
            and stable_delta >= request_count
            and candidate_delta >= 1
        )
        mirrored = candidate_delta >= 1
    elif stage in {"CANARY_5", "CANARY_25"}:
        candidate_ratio = responses["candidate"] / request_count
        lower, upper = ((0.01, 0.15) if stage == "CANARY_5" else (0.12, 0.40))
        passed = (
            responses["stable"] >= 1
            and responses["candidate"] >= 1
            and lower <= candidate_ratio <= upper
            and stable_delta == responses["stable"]
            and candidate_delta == responses["candidate"]
        )
        mirrored = False
    else:
        passed = (
            responses == {"stable": request_count}
            and stable_delta == request_count
            and candidate_delta == 0
        )
        mirrored = False
    if not passed or not response_hashes:
        raise VlmKServeRolloutError(f"VLM {stage} traffic evidence failed")
    return VlmRolloutStageEvidence(
        stage=stage,
        passed=True,
        request_count=request_count,
        stable_request_delta=stable_delta,
        candidate_request_delta=candidate_delta,
        stable_response_count=responses["stable"],
        candidate_response_count=responses["candidate"],
        route_document_sha256=document_sha256(route),
        response_sha256s=tuple(sorted(response_hashes)),
        candidate_mirror_observed=mirrored,
    )


def _validate_candidate_quality(
    response: dict[str, Any],
    *,
    expected_label: str,
    expected_region: dict[str, float],
) -> float:
    if response.get("variant") != "candidate" or response.get(
        "structured_output_valid"
    ) is not True:
        raise VlmKServeRolloutError("candidate VLM response is invalid")
    findings = response.get("findings")
    if not isinstance(findings, list):
        raise VlmKServeRolloutError("candidate VLM findings are invalid")
    matches = [
        item
        for item in findings
        if isinstance(item, dict) and item.get("label") == expected_label
    ]
    if not matches or not isinstance(matches[0].get("region"), dict):
        raise VlmKServeRolloutError("candidate VLM expected label is missing")
    score = _region_iou(cast(dict[str, float], matches[0]["region"]), expected_region)
    if score < 0.5:
        raise VlmKServeRolloutError("candidate VLM region IoU is below gate")
    return score


def _region_iou(left: dict[str, float], right: dict[str, float]) -> float:
    left_x2, left_y2 = left["x"] + left["width"], left["y"] + left["height"]
    right_x2, right_y2 = right["x"] + right["width"], right["y"] + right["height"]
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left["x"], right["x"]))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left["y"], right["y"]))
    intersection = intersection_width * intersection_height
    union = left["width"] * left["height"] + right["width"] * right["height"] - intersection
    return intersection / union if union > 0 else 0.0


def _ensure_cluster_running(root: Path) -> None:
    running = _capture_checked(
        ["docker", "inspect", KIND_NODE, "--format", "{{.State.Running}}"],
        cwd=root,
    )
    if running != "true":
        _run_checked(["docker", "start", KIND_NODE], cwd=root)
        time.sleep(10)
    observed_context = _capture_checked(
        ["kubectl", "config", "current-context"], cwd=root
    )
    if observed_context != KUBE_CONTEXT:
        _run_checked(["kubectl", "config", "use-context", KUBE_CONTEXT], cwd=root)


def _node_image_digest(root: Path) -> str:
    normalized_image = f"docker.io/{PROXY_IMAGE}"
    raw_document = _capture_checked(
        ["docker", "exec", KIND_NODE, "crictl", "inspecti", normalized_image],
        cwd=root,
    )
    try:
        document = json.loads(raw_document)
    except json.JSONDecodeError as exc:
        raise VlmKServeRolloutError(
            "Kind node returned invalid proxy image metadata"
        ) from exc
    status = document.get("status")
    if not isinstance(status, dict):
        raise VlmKServeRolloutError("Kind node proxy image status is missing")
    digest = status.get("id")
    if not isinstance(digest, str) or not _valid_image_digest(digest):
        raise VlmKServeRolloutError("Kind node proxy image digest is invalid")
    repo_tags = status.get("repoTags")
    if not isinstance(repo_tags, list) or normalized_image not in repo_tags:
        raise VlmKServeRolloutError(
            "Kind node does not contain the fixed VLM proxy image tag"
        )
    return digest
    _run_checked(
        [
            "kubectl",
            "--context",
            KUBE_CONTEXT,
            "wait",
            "--for=condition=Ready",
            "node",
            KIND_NODE,
            "--timeout=120s",
        ],
        cwd=root,
    )


def _kubectl_apply(root: Path, document: dict[str, Any]) -> None:
    _run_input_checked(
        [
            "kubectl",
            "--context",
            KUBE_CONTEXT,
            "apply",
            "--server-side",
            "--field-manager",
            "vlm-kserve-rollout",
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
    raise VlmKServeRolloutError(f"KServe endpoint is not ready: {service}")


def _wait_route(root: Path, *, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        route = _kubectl_json(root, "httproute", ROUTE_NAME)
        if route_condition_is_true(route, "Accepted") and route_condition_is_true(
            route, "ResolvedRefs"
        ):
            time.sleep(1)
            return
        time.sleep(1)
    raise VlmKServeRolloutError("VLM HTTPRoute was not accepted and resolved")


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
        raise VlmKServeRolloutError("Envoy Gateway service is not unique")
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
            raise VlmKServeRolloutError("VLM GPU worker exited during startup")
        try:
            document = _get_json(f"{worker_url}/health/ready", timeout_seconds=2)
            if document.get("ready") is True:
                return
        except (httpx.HTTPError, VlmKServeRolloutError):
            pass
        time.sleep(1)
    raise VlmKServeRolloutError("VLM GPU worker did not become ready")


def _wait_tcp_process(
    process: subprocess.Popen[str], port: int, *, timeout_seconds: int
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            raise VlmKServeRolloutError(f"port-forward failed: {output[-512:]}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.25)
    raise VlmKServeRolloutError("port-forward did not become ready")


def _post_gateway(url: str, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    with httpx.Client(timeout=httpx.Timeout(300), trust_env=False) as client:
        response = client.post(url, headers={"Host": HOSTNAME}, json=payload)
        response.raise_for_status()
        variant = response.headers.get("X-IOAP-VLM-Variant", "")
        if variant not in {"stable", "candidate"}:
            raise VlmKServeRolloutError("gateway response variant is missing")
        document = response.json()
    if not isinstance(document, dict):
        raise VlmKServeRolloutError("gateway response is invalid")
    return document, variant


def _post_json(url: str, payload: dict[str, Any], *, timeout_seconds: int) -> dict[str, Any]:
    with httpx.Client(timeout=httpx.Timeout(timeout_seconds), trust_env=False) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        document = response.json()
    if not isinstance(document, dict):
        raise VlmKServeRolloutError("VLM response is invalid")
    return document


def _get_json(url: str, *, timeout_seconds: int) -> dict[str, Any]:
    with httpx.Client(timeout=timeout_seconds, trust_env=False) as client:
        response = client.get(url)
        response.raise_for_status()
        document = response.json()
    if not isinstance(document, dict):
        raise VlmKServeRolloutError("VLM runtime response is invalid")
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
    raise VlmKServeRolloutError("Shadow request was not mirrored to VLM candidate")


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
        raise VlmKServeRolloutError("Kubernetes resource response is invalid")
    return document


def _delete_vlm_resources(root: Path, *, check: bool = True) -> None:
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
        raise VlmKServeRolloutError(f"VLM resource cleanup failed: {result.stdout[-512:]}")


def _resource_ready(document: dict[str, Any]) -> bool:
    conditions = document.get("status", {}).get("conditions", [])
    return any(
        isinstance(item, dict)
        and item.get("type") == "Ready"
        and item.get("status") == "True"
        for item in conditions
    )


def _runtime_versions() -> dict[str, str]:
    import peft
    import torch
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }


def _variant_count(document: dict[str, Any], field: str, variant: str) -> int:
    values = document.get(field)
    if not isinstance(values, dict):
        raise VlmKServeRolloutError(f"runtime metric is invalid: {field}")
    value = values.get(variant)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise VlmKServeRolloutError(f"runtime metric is invalid: {field}.{variant}")
    return value


def _text(document: dict[str, Any], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value:
        raise VlmKServeRolloutError(f"field is invalid: {field}")
    return value


def _positive_integer(document: dict[str, Any], field: str) -> int:
    value = document.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VlmKServeRolloutError(f"field is invalid: {field}")
    return value


def _valid_image_digest(value: str) -> bool:
    return value.startswith("sha256:") and len(value) == 71 and all(
        character in "0123456789abcdef" for character in value[7:]
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


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
        raise VlmKServeRolloutError(
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
        raise VlmKServeRolloutError(
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
        raise VlmKServeRolloutError(
            f"command failed ({argv[0]}): {result.stdout[-1_024:].decode(errors='replace')}"
        )


def _write_json(path: Path, document: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-vlm-kserve-shadow")
    parser.add_argument("command", choices=("run", "verify", "cleanup"), nargs="?", default="run")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--stop-cluster-after-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    if args.command == "run":
        path = execute_rollout(
            root,
            stop_cluster_after_run=args.stop_cluster_after_run,
        )
        report = verify_vlm_kserve_rollout(root, path)
    elif args.command == "verify":
        path = root / OUTPUT_RELATIVE / "acceptance.json"
        report = verify_vlm_kserve_rollout(root, path)
    else:
        _ensure_cluster_running(root)
        _delete_vlm_resources(root)
        if args.stop_cluster_after_run:
            _run_checked(["docker", "stop", KIND_NODE], cwd=root)
        print("VLM_KSERVE_RESOURCES_CLEANED")
        return 0
    print(report.status)
    print(report.classification)
    print(f"ACTUAL_GPU_EXECUTION={report.runtime.actual_gpu_execution}")
    print(f"KSERVE_SHADOW_VERIFIED={report.kserve.kserve_shadow_endpoint_verified}")
    print(f"CANARY_VERIFIED={all(item.passed for item in report.stages)}")
    print(f"EVIDENCE_CHAIN_SHA256={report.evidence_chain_sha256}")
    print(path)
    return 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
