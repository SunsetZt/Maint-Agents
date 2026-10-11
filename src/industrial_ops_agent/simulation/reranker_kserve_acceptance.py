"""Immutable evidence for an executed Reranker ModelRelease KServe rollout."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from industrial_ops_agent.simulation.asr_kserve_rollout import PROXY_IMAGE
from industrial_ops_agent.simulation.reranker_kserve_readiness import (
    RUNTIME_IMAGE_DIGEST,
    RUNTIME_IMAGE_REPOSITORY,
    RUNTIME_IMAGE_TAG,
    verify_reranker_kserve_readiness,
)
from industrial_ops_agent.simulation.reranker_runtime_gpu_probe import (
    EXPECTED_TOP_CHUNK_ID,
    verify_runtime_gpu_probe_report,
)
from industrial_ops_agent.training.gpu_promotion_security import (
    hardened_container_security_context,
)

SCHEMA_VERSION: Literal["enterprise-reranker-kserve-rollout-acceptance/v1"] = (
    "enterprise-reranker-kserve-rollout-acceptance/v1"
)
STATUS: Literal["RERANKER_MODEL_RELEASE_KSERVE_SHADOW_CANARY_ROLLBACK_PASSED"] = (
    "RERANKER_MODEL_RELEASE_KSERVE_SHADOW_CANARY_ROLLBACK_PASSED"
)
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = "LOCAL_STAGING_PROJECT_AUTHORIZED"
OUTPUT_RELATIVE = Path("artifacts/m7-reranker-kserve-acceptance")
ACCEPTANCE_RELATIVE = OUTPUT_RELATIVE / "acceptance.json"
PROXY_SOURCE_RELATIVE = Path("src/industrial_ops_agent/deployment/reranker_kserve_proxy.py")
NAMESPACE = "ioap-gpu-promotion-lab"
GATEWAY_NAME = "ioap-gpu-lab-gateway"
HOSTNAME = "gpu-lab.local"
ROUTE_PATH_PREFIX = "/v1/retrieval"
ENDPOINT_PATH = f"{ROUTE_PATH_PREFIX}/rerank"
CONFIG_MAP_NAME = "ioap-reranker-rollout-proxy"
STABLE_SERVICE = "ioap-reranker-stable"
CANDIDATE_SERVICE = "ioap-reranker-candidate"
ROUTE_NAME = "ioap-reranker-lab-route"
STABLE_WORKER_CONTAINER = "ioap-reranker-stable-gpu-worker"
CANDIDATE_WORKER_CONTAINER = "ioap-reranker-candidate-gpu-worker"
KUBE_CONTEXT = "kind-ioap-gpu-promotion-lab"
KSERVE_CONTROLLER_DEPLOYMENT = "kserve-controller-manager"
KSERVE_CONTROLLER_IMAGE = "kserve/kserve-controller:v0.19.0"
EXCLUDED_RAW_COMPARATOR_FILES = (
    "industrial-ops-calibrated-runtime.json",
    "industrial-ops-terminology-registry.json",
)


class RerankerKServeAcceptanceError(RuntimeError):
    """Executed Reranker rollout evidence is missing or inconsistent."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileBinding(_ClosedModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AcceptanceSource(_ClosedModel):
    readiness: FileBinding
    readiness_evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gpu_probe: FileBinding
    gpu_probe_evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_run_id: Literal["reranker-calibrated-64f493aa97c1a51cf422"]
    candidate_path: str = Field(min_length=1)
    candidate_bundle_sha256: Literal[
        "937a9db83f62a4db201d2a5928f80b9982fd814c87be36c121afff3880a318a9"
    ]


class RawComparatorEvidence(_ClosedModel):
    manifest: FileBinding
    directory: str = Field(min_length=1)
    component_model_id: str = Field(pattern=r"^reranker-raw-comparator-[0-9a-f]{20}$")
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_candidate_path: str = Field(min_length=1)
    source_candidate_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    excluded_files: tuple[str, ...] = Field(min_length=2, max_length=2)
    routing_only_not_release_candidate: Literal[True] = True


class ModelReleaseEvidence(_ClosedModel):
    state_database: FileBinding
    sqlite_integrity_check: Literal["ok"]
    tenant_id: Literal["project-enterprise-staging"]
    baseline_release_id: str = Field(pattern=r"^model-release-[0-9a-f]{32}$")
    release_id: str = Field(pattern=r"^model-release-[0-9a-f]{32}$")
    manifest: FileBinding
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_candidate_experiment_id: Literal["reranker-calibrated-64f493aa97c1a51cf422"]
    imported_candidate_experiment_id: str = Field(min_length=1)
    evaluation_id: str = Field(min_length=1)
    approval_id: str = Field(pattern=r"^release-approval-[0-9a-f]{32}$")
    approval_requested_by_subject_id: str = Field(min_length=1)
    approval_decided_by_subject_id: str = Field(min_length=1)
    independent_approval_verified: Literal[True] = True
    deployment_id: str = Field(pattern=r"^model-deployment-[0-9a-f]{32}$")
    deployment_provider: Literal["KSERVE_GATEWAY_API"]
    transition_targets: tuple[str, ...] = Field(min_length=8)
    observation_stages: tuple[str, ...] = Field(min_length=3, max_length=3)
    all_stage_observations_passed: Literal[True] = True
    final_release_status: Literal["ROLLED_BACK"]
    final_deployment_status: Literal["READY"]
    final_deployment_stage: Literal["ROLLED_BACK"]
    final_traffic_percent: float = Field(ge=0.0, le=0.0)


class RuntimeEvidence(_ClosedModel):
    topology: Literal["TWO_DOCKER_GPU_WORKERS_BEHIND_KSERVE_PROXIES"]
    actual_gpu_execution: Literal[True] = True
    model_execution_simulated: Literal[False] = False
    image_repository: Literal["industrial-ops/reranker-runtime"]
    image_tag: Literal["calibrated-v1-95fb9107de7a"]
    image_digest: Literal["sha256:7386510182721f4f086773ad8d2dd7f5713855c483f43e7d572252f88c9776eb"]
    stable_worker_container: Literal["ioap-reranker-stable-gpu-worker"]
    candidate_worker_container: Literal["ioap-reranker-candidate-gpu-worker"]
    stable_worker_log: FileBinding
    candidate_worker_log: FileBinding
    stable_container_inspect: FileBinding
    candidate_container_inspect: FileBinding
    stable_runtime_metrics: FileBinding
    candidate_runtime_metrics: FileBinding
    stable_proxy_metrics: FileBinding
    candidate_proxy_metrics: FileBinding
    gpu_inventory: FileBinding
    exactly_one_gpu_exposed_per_worker: Literal[True] = True
    cpu_limit_per_worker: float = Field(ge=1.5, le=1.5)
    memory_limit_bytes_per_worker: Literal[4294967296]
    read_only_root_filesystem: Literal[True] = True
    non_root_user: Literal["app"]
    no_new_privileges: Literal[True] = True
    all_linux_capabilities_dropped: Literal[True] = True


class KServeEvidence(_ClosedModel):
    cluster_context: Literal["kind-ioap-gpu-promotion-lab"]
    namespace: Literal["ioap-gpu-promotion-lab"]
    controller_deployment: FileBinding
    controller_image: Literal["kserve/kserve-controller:v0.19.0"]
    controller_image_pull_policy: Literal["IfNotPresent"]
    gateway_name: Literal["ioap-gpu-lab-gateway"]
    hostname: Literal["gpu-lab.local"]
    route_name: Literal["ioap-reranker-lab-route"]
    route_path_prefix: Literal["/v1/retrieval"]
    stable_inference_service: Literal["ioap-reranker-stable"]
    candidate_inference_service: Literal["ioap-reranker-candidate"]
    stable_ready: Literal[True] = True
    candidate_ready: Literal[True] = True
    route_accepted: Literal[True] = True
    route_resolved_refs: Literal[True] = True
    worker_host_gateway: str = Field(min_length=1)
    stable_worker_host_port: int = Field(ge=1, le=65535)
    candidate_worker_host_port: int = Field(ge=1, le=65535)
    proxy_image: str = Field(min_length=1)
    proxy_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    proxy_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_map: FileBinding
    stable_service: FileBinding
    candidate_service: FileBinding
    routes: dict[str, FileBinding]
    observed_final_route: FileBinding
    observed_stable_service: FileBinding
    observed_candidate_service: FileBinding
    hardened_container_security_context: Literal[True] = True
    proxy_has_no_gpu_request: Literal[True] = True
    endpoint_scope: Literal["LOCAL_PORT_FORWARD"]


class QualityEvidence(_ClosedModel):
    request: FileBinding
    stable_response: FileBinding
    candidate_response: FileBinding
    expected_top_chunk_id: Literal["probe-gearbox-controlled-action"]
    stable_target_score: float
    candidate_target_score: float = Field(ge=3.0)
    candidate_score_delta: float = Field(gt=0.0)
    candidate_top_rank_verified: Literal[True] = True
    release_binding_verified: Literal[True] = True


class RolloutStageEvidence(_ClosedModel):
    stage: Literal["SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"]
    passed: Literal[True] = True
    request_count: int = Field(ge=1)
    stable_request_delta: int = Field(ge=0)
    candidate_request_delta: int = Field(ge=0)
    stable_response_count: int = Field(ge=0)
    candidate_response_count: int = Field(ge=0)
    candidate_response_ratio: float = Field(ge=0.0, le=1.0)
    candidate_mirror_observed: bool
    p95_latency_ms: float = Field(gt=0.0)
    error_count: Literal[0] = 0
    route_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256s: tuple[str, ...] = Field(min_length=1, max_length=2)
    model_release_observation_decision: Literal["PASS", "NOT_REQUIRED_AFTER_ROLLBACK"]


class CleanupEvidence(_ClosedModel):
    stable_gpu_worker_stopped: Literal[True] = True
    candidate_gpu_worker_stopped: Literal[True] = True
    port_forwards_stopped: Literal[True] = True
    reranker_kubernetes_resources_removed: Literal[True] = True
    cluster_stopped_after_acceptance: Literal[True] = True
    no_rollout_service_left_running: Literal[True] = True
    evidence_is_historical_after_cleanup: Literal[True] = True


class EnterpriseRerankerKServeAcceptanceReport(_ClosedModel):
    schema_version: Literal["enterprise-reranker-kserve-rollout-acceptance/v1"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal["RERANKER_MODEL_RELEASE_KSERVE_SHADOW_CANARY_ROLLBACK_PASSED"] = STATUS
    source: AcceptanceSource
    raw_comparator: RawComparatorEvidence
    model_release: ModelReleaseEvidence
    runtime: RuntimeEvidence
    kserve: KServeEvidence
    quality: QualityEvidence
    stages: tuple[RolloutStageEvidence, ...] = Field(min_length=4, max_length=4)
    cleanup: CleanupEvidence
    hard_gates: dict[str, Literal[True]]
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def proxy_config_map_document(proxy_source: str) -> dict[str, Any]:
    if not proxy_source.strip():
        raise ValueError("Reranker proxy source is empty")
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIG_MAP_NAME, "namespace": NAMESPACE},
        "data": {"reranker_kserve_proxy.py": proxy_source},
    }


def inference_service_document(
    *,
    variant: Literal["stable", "candidate"],
    host_gateway: str,
    host_port: int,
    proxy_image_digest: str,
    release_id: str,
    manifest_hash: str,
    public_component_model_id: str,
    public_artifact_hash: str,
    upstream_component_model_id: str,
    upstream_artifact_hash: str,
) -> dict[str, Any]:
    if not host_gateway or any(character.isspace() for character in host_gateway):
        raise ValueError("Reranker host GPU endpoint is invalid")
    if not 1 <= host_port <= 65_535 or not proxy_image_digest.startswith("sha256:"):
        raise ValueError("Reranker host GPU endpoint is invalid")
    name = STABLE_SERVICE if variant == "stable" else CANDIDATE_SERVICE
    env = {
        "PORT": "8080",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": "/tmp/pycache",
        "IOAP_RERANKER_VARIANT": variant,
        "IOAP_RERANKER_UPSTREAM_URL": f"http://{host_gateway}:{host_port}",
        "IOAP_RERANKER_UPSTREAM_TIMEOUT_SECONDS": "30",
        "IOAP_RERANKER_RELEASE_ID": release_id,
        "IOAP_RERANKER_MANIFEST_HASH": manifest_hash,
        "IOAP_RERANKER_PUBLIC_COMPONENT_MODEL_ID": public_component_model_id,
        "IOAP_RERANKER_PUBLIC_ARTIFACT_CONTENT_HASH": public_artifact_hash,
        "IOAP_RERANKER_UPSTREAM_COMPONENT_MODEL_ID": upstream_component_model_id,
        "IOAP_RERANKER_UPSTREAM_ARTIFACT_CONTENT_HASH": upstream_artifact_hash,
    }
    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "annotations": {"serving.kserve.io/deploymentMode": "Standard"},
            "labels": {
                "ioap.openai.com/acceptance": "enterprise-reranker-rollout-v1",
                "ioap.openai.com/model-variant": variant,
                "ioap.openai.com/runtime-image-digest": proxy_image_digest.removeprefix("sha256:")[
                    :63
                ],
            },
        },
        "spec": {
            "predictor": {
                "minReplicas": 1,
                "maxReplicas": 1,
                "containers": [
                    {
                        "name": "kserve-container",
                        "image": PROXY_IMAGE,
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["python", "-u", "/app/reranker_kserve_proxy.py"],
                        "env": [{"name": key, "value": value} for key, value in env.items()],
                        "ports": [{"name": "http1", "containerPort": 8080}],
                        "startupProbe": {
                            "httpGet": {"path": "/health/ready", "port": 8080},
                            "periodSeconds": 2,
                            "failureThreshold": 60,
                        },
                        "readinessProbe": {
                            "httpGet": {"path": "/health/ready", "port": 8080},
                            "periodSeconds": 3,
                            "failureThreshold": 10,
                        },
                        "securityContext": {
                            **hardened_container_security_context(),
                            "readOnlyRootFilesystem": True,
                        },
                        "resources": {
                            "requests": {"cpu": "25m", "memory": "48Mi"},
                            "limits": {"cpu": "250m", "memory": "192Mi"},
                        },
                        "volumeMounts": [
                            {
                                "name": "proxy-code",
                                "mountPath": "/app/reranker_kserve_proxy.py",
                                "subPath": "reranker_kserve_proxy.py",
                                "readOnly": True,
                            },
                            {"name": "runtime-tmp", "mountPath": "/tmp"},
                        ],
                    }
                ],
                "volumes": [
                    {"name": "proxy-code", "configMap": {"name": CONFIG_MAP_NAME}},
                    {"name": "runtime-tmp", "emptyDir": {}},
                ],
            }
        },
    }


def route_document(
    stage: Literal["SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"],
) -> dict[str, Any]:
    stable = {"name": f"{STABLE_SERVICE}-predictor", "port": 80}
    candidate = {"name": f"{CANDIDATE_SERVICE}-predictor", "port": 80}
    if stage == "SHADOW":
        rule: dict[str, Any] = {
            "backendRefs": [{**stable, "weight": 100}],
            "filters": [{"type": "RequestMirror", "requestMirror": {"backendRef": candidate}}],
        }
    elif stage in {"CANARY_5", "CANARY_25"}:
        candidate_weight = 5 if stage == "CANARY_5" else 25
        rule = {
            "backendRefs": [
                {**stable, "weight": 100 - candidate_weight},
                {**candidate, "weight": candidate_weight},
            ]
        }
    elif stage == "ROLLED_BACK":
        rule = {"backendRefs": [{**stable, "weight": 100}]}
    else:  # pragma: no cover - Literal plus runtime defense
        raise ValueError("Reranker rollout stage is invalid")
    rule["matches"] = [{"path": {"type": "PathPrefix", "value": ROUTE_PATH_PREFIX}}]
    rule["timeouts"] = {"request": "30s", "backendRequest": "30s"}
    return {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "HTTPRoute",
        "metadata": {
            "name": ROUTE_NAME,
            "namespace": NAMESPACE,
            "annotations": {"ioap.openai.com/stage": stage},
        },
        "spec": {
            "parentRefs": [{"name": GATEWAY_NAME}],
            "hostnames": [HOSTNAME],
            "rules": [rule],
        },
    }


def finalize_acceptance(
    unsigned: dict[str, Any],
) -> EnterpriseRerankerKServeAcceptanceReport:
    draft = EnterpriseRerankerKServeAcceptanceReport.model_validate(
        {**unsigned, "evidence_chain_sha256": "0" * 64}
    )
    payload = draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    return draft.model_copy(update={"evidence_chain_sha256": document_sha256(payload)})


def write_acceptance(
    report: EnterpriseRerankerKServeAcceptanceReport,
    path: Path,
) -> None:
    _write_json(path, report.model_dump(mode="json"))


def verify_reranker_kserve_acceptance(
    repo_root: Path,
    acceptance_path: Path = ACCEPTANCE_RELATIVE,
) -> EnterpriseRerankerKServeAcceptanceReport:
    root = repo_root.resolve(strict=True)
    path = _inside_file(root, acceptance_path)
    try:
        report = EnterpriseRerankerKServeAcceptanceReport.model_validate_json(path.read_bytes())
    except (OSError, UnicodeError, ValidationError) as exc:
        raise RerankerKServeAcceptanceError("reranker_kserve_acceptance_is_invalid") from exc
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != document_sha256(unsigned):
        raise RerankerKServeAcceptanceError("reranker_kserve_acceptance_chain_changed")
    readiness = verify_reranker_kserve_readiness(
        root,
        Path(report.source.readiness.path),
    )
    gpu_probe = verify_runtime_gpu_probe_report(
        root,
        Path(report.source.gpu_probe.path),
    )
    if (
        readiness.evidence_chain_sha256 != report.source.readiness_evidence_chain_sha256
        or gpu_probe.evidence_chain_sha256 != report.source.gpu_probe_evidence_chain_sha256
        or gpu_probe.source.readiness_evidence_chain_sha256 != readiness.evidence_chain_sha256
        or readiness.source.run_id != report.source.candidate_run_id
        or readiness.source.candidate_path != report.source.candidate_path
        or readiness.source.candidate_bundle_sha256 != report.source.candidate_bundle_sha256
        or report.runtime.image_digest != RUNTIME_IMAGE_DIGEST
        or report.runtime.image_repository != RUNTIME_IMAGE_REPOSITORY
        or report.runtime.image_tag != RUNTIME_IMAGE_TAG
    ):
        raise RerankerKServeAcceptanceError("reranker_kserve_source_binding_changed")
    for binding in (
        report.source.readiness,
        report.source.gpu_probe,
        report.raw_comparator.manifest,
        report.model_release.state_database,
        report.model_release.manifest,
        report.runtime.stable_worker_log,
        report.runtime.candidate_worker_log,
        report.runtime.stable_container_inspect,
        report.runtime.candidate_container_inspect,
        report.runtime.stable_runtime_metrics,
        report.runtime.candidate_runtime_metrics,
        report.runtime.stable_proxy_metrics,
        report.runtime.candidate_proxy_metrics,
        report.runtime.gpu_inventory,
        report.quality.request,
        report.quality.stable_response,
        report.quality.candidate_response,
        report.kserve.config_map,
        report.kserve.controller_deployment,
        report.kserve.stable_service,
        report.kserve.candidate_service,
        report.kserve.observed_final_route,
        report.kserve.observed_stable_service,
        report.kserve.observed_candidate_service,
        *report.kserve.routes.values(),
    ):
        _verify_binding(root, binding)
    _verify_raw_comparator(root, report)
    manifest = _load_object(_inside_file(root, Path(report.model_release.manifest.path)))
    if document_sha256(manifest) != report.model_release.manifest_hash:
        raise RerankerKServeAcceptanceError("reranker_release_manifest_changed")
    _verify_release_manifest(report, manifest)
    _verify_release_database(
        _inside_file(root, Path(report.model_release.state_database.path)),
        report,
        manifest,
    )
    _verify_quality(root, report)
    _verify_runtime(root, report)
    _verify_kserve(root, report)
    _verify_stages(report)
    if set(report.hard_gates.values()) != {True}:
        raise RerankerKServeAcceptanceError("reranker_acceptance_hard_gate_changed")
    return report


def _verify_raw_comparator(
    root: Path,
    report: EnterpriseRerankerKServeAcceptanceReport,
) -> None:
    evidence = report.raw_comparator
    manifest = _load_object(_inside_file(root, Path(evidence.manifest.path)))
    directory = _inside_directory(root, Path(evidence.directory))
    source = _inside_directory(root, Path(evidence.source_candidate_path))
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != "enterprise-reranker-raw-comparator-manifest/v1"
        or manifest.get("source_candidate_path") != evidence.source_candidate_path
        or manifest.get("source_candidate_bundle_sha256") != evidence.source_candidate_bundle_sha256
        or tuple(manifest.get("excluded_files", [])) != EXCLUDED_RAW_COMPARATOR_FILES
        or manifest.get("bundle_sha256") != evidence.bundle_sha256
        or not isinstance(files, list)
    ):
        raise RerankerKServeAcceptanceError("reranker_raw_comparator_manifest_changed")
    expected: list[dict[str, Any]] = []
    for source_file in sorted(item for item in source.iterdir() if item.is_file()):
        if source_file.name in EXCLUDED_RAW_COMPARATOR_FILES:
            continue
        target = directory / source_file.name
        if (
            target.is_symlink()
            or not target.is_file()
            or target.stat().st_mode & 0o004 != 0o004
            or file_sha256(target) != file_sha256(source_file)
        ):
            raise RerankerKServeAcceptanceError("reranker_raw_comparator_file_changed")
        expected.append(
            {
                "path": target.relative_to(root).as_posix(),
                "size_bytes": target.stat().st_size,
                "sha256": file_sha256(target),
            }
        )
    if files != expected or evidence.bundle_sha256 != document_sha256(expected):
        raise RerankerKServeAcceptanceError("reranker_raw_comparator_bundle_changed")
    if any((directory / item).exists() for item in EXCLUDED_RAW_COMPARATOR_FILES):
        raise RerankerKServeAcceptanceError("reranker_raw_comparator_policy_changed")


def _verify_release_manifest(
    report: EnterpriseRerankerKServeAcceptanceReport,
    manifest: dict[str, Any],
) -> None:
    component = manifest.get("specialized_components", {}).get("reranker", {})
    project_import = component.get("project_authorized_import", {})
    evaluation = component.get("evaluation", {})
    if (
        component.get("runtime_model_id") != report.model_release.imported_candidate_experiment_id
        or project_import.get("source_candidate_experiment_id")
        != report.model_release.source_candidate_experiment_id
        or evaluation.get("evaluation_id") != report.model_release.evaluation_id
    ):
        raise RerankerKServeAcceptanceError("reranker_release_component_binding_changed")


def _verify_release_database(
    path: Path,
    report: EnterpriseRerankerKServeAcceptanceReport,
    manifest: dict[str, Any],
) -> None:
    release = report.model_release
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        release_row = connection.execute(
            "SELECT status, manifest_hash, traffic_percent, candidate_experiment_id "
            "FROM model_releases WHERE tenant_id = ? AND release_id = ?",
            (release.tenant_id, release.release_id),
        ).fetchone()
        deployment_row = connection.execute(
            "SELECT deployment_id, provider, status, current_stage, "
            "observed_traffic_percent FROM model_deployments "
            "WHERE tenant_id = ? AND release_id = ?",
            (release.tenant_id, release.release_id),
        ).fetchone()
        approval_row = connection.execute(
            "SELECT approval_id, status, requested_by_subject_id, decided_by_subject_id "
            "FROM model_release_approvals WHERE tenant_id = ? AND release_id = ?",
            (release.tenant_id, release.release_id),
        ).fetchone()
        transitions = tuple(
            row[0]
            for row in connection.execute(
                "SELECT to_status FROM model_release_transitions "
                "WHERE tenant_id = ? AND release_id = ? ORDER BY sequence",
                (release.tenant_id, release.release_id),
            )
        )
        observations = tuple(
            row[0]
            for row in connection.execute(
                "SELECT stage FROM model_release_observations "
                "WHERE tenant_id = ? AND release_id = ? AND decision = 'PASS' "
                "ORDER BY created_at, sequence",
                (release.tenant_id, release.release_id),
            )
        )
    finally:
        connection.close()
    candidate_id = manifest.get("training", {}).get("experiment_id")
    if (
        integrity != ("ok",)
        or release.sqlite_integrity_check != "ok"
        or not isinstance(candidate_id, str)
        or release_row != ("ROLLED_BACK", release.manifest_hash, 0.0, candidate_id)
        or deployment_row
        != (
            release.deployment_id,
            "KSERVE_GATEWAY_API",
            "READY",
            "ROLLED_BACK",
            0.0,
        )
        or approval_row
        != (
            release.approval_id,
            "APPROVED",
            release.approval_requested_by_subject_id,
            release.approval_decided_by_subject_id,
        )
        or transitions != release.transition_targets
        or observations != release.observation_stages
    ):
        raise RerankerKServeAcceptanceError("reranker_release_database_changed")


def _verify_quality(
    root: Path,
    report: EnterpriseRerankerKServeAcceptanceReport,
) -> None:
    request = _load_object(_inside_file(root, Path(report.quality.request.path)))
    stable = _load_object(_inside_file(root, Path(report.quality.stable_response.path)))
    candidate = _load_object(_inside_file(root, Path(report.quality.candidate_response.path)))
    stable_score = _target_score(stable, EXPECTED_TOP_CHUNK_ID)
    candidate_score = _target_score(candidate, EXPECTED_TOP_CHUNK_ID)
    candidate_body = candidate.get("body")
    if (
        request.get("expected_release_id") != report.model_release.release_id
        or request.get("expected_manifest_hash") != report.model_release.manifest_hash
        or stable.get("status_code") != 200
        or stable.get("variant") != "stable"
        or candidate.get("status_code") != 200
        or candidate.get("variant") != "candidate"
        or not isinstance(candidate_body, dict)
        or candidate_body.get("items", [{}])[0].get("chunk_id") != EXPECTED_TOP_CHUNK_ID
        or abs(stable_score - report.quality.stable_target_score) > 1e-12
        or abs(candidate_score - report.quality.candidate_target_score) > 1e-12
        or abs(candidate_score - stable_score - report.quality.candidate_score_delta) > 1e-12
        or candidate_score < 3.0
        or candidate_score <= stable_score
    ):
        raise RerankerKServeAcceptanceError("reranker_release_quality_changed")


def _verify_runtime(
    root: Path,
    report: EnterpriseRerankerKServeAcceptanceReport,
) -> None:
    stable_proxy = _load_object(_inside_file(root, Path(report.runtime.stable_proxy_metrics.path)))
    candidate_proxy = _load_object(
        _inside_file(root, Path(report.runtime.candidate_proxy_metrics.path))
    )
    gpu_inventory = _inside_file(
        root,
        Path(report.runtime.gpu_inventory.path),
    ).read_text(encoding="utf-8")
    stable_inspect = _load_array(
        _inside_file(root, Path(report.runtime.stable_container_inspect.path))
    )
    candidate_inspect = _load_array(
        _inside_file(root, Path(report.runtime.candidate_container_inspect.path))
    )
    if (
        stable_proxy.get("variant") != "stable"
        or candidate_proxy.get("variant") != "candidate"
        or not isinstance(stable_proxy.get("request_count"), int)
        or not isinstance(candidate_proxy.get("request_count"), int)
        or stable_proxy.get("error_count") != 0
        or candidate_proxy.get("error_count") != 0
        or "NVIDIA GeForce RTX 4070 SUPER" not in gpu_inventory
    ):
        raise RerankerKServeAcceptanceError("reranker_runtime_evidence_changed")
    _validate_worker_container(stable_inspect, STABLE_WORKER_CONTAINER)
    _validate_worker_container(candidate_inspect, CANDIDATE_WORKER_CONTAINER)


def _validate_worker_container(document: dict[str, Any], name: str) -> None:
    host = document.get("HostConfig")
    config = document.get("Config")
    requests = host.get("DeviceRequests") if isinstance(host, dict) else None
    request = requests[0] if isinstance(requests, list) and len(requests) == 1 else None
    security_options_value = host.get("SecurityOpt") if isinstance(host, dict) else []
    security_options = security_options_value if isinstance(security_options_value, list) else []
    if (
        document.get("Name") != f"/{name}"
        or document.get("Image") != RUNTIME_IMAGE_DIGEST
        or not isinstance(host, dict)
        or host.get("NanoCpus") != 1_500_000_000
        or host.get("Memory") != 4 * 1024**3
        or host.get("MemorySwap") != 4 * 1024**3
        or host.get("ReadonlyRootfs") is not True
        or host.get("PidsLimit") != 256
        or "ALL" not in (host.get("CapDrop") or [])
        or not any(str(item).startswith("no-new-privileges") for item in security_options)
        or not isinstance(request, dict)
        or request.get("DeviceIDs") != ["0"]
        or not isinstance(config, dict)
        or config.get("User") != "app"
    ):
        raise RerankerKServeAcceptanceError("reranker_worker_container_changed")


def _verify_kserve(
    root: Path,
    report: EnterpriseRerankerKServeAcceptanceReport,
) -> None:
    proxy_source = _inside_file(root, PROXY_SOURCE_RELATIVE).read_text(encoding="utf-8")
    if file_sha256(_inside_file(root, PROXY_SOURCE_RELATIVE)) != (
        report.kserve.proxy_source_sha256
    ):
        raise RerankerKServeAcceptanceError("reranker_proxy_source_changed")
    config_map = _load_object(_inside_file(root, Path(report.kserve.config_map.path)))
    stable = _load_object(_inside_file(root, Path(report.kserve.stable_service.path)))
    candidate = _load_object(_inside_file(root, Path(report.kserve.candidate_service.path)))
    common = {
        "host_gateway": report.kserve.worker_host_gateway,
        "proxy_image_digest": report.kserve.proxy_image_digest,
        "release_id": report.model_release.release_id,
        "manifest_hash": report.model_release.manifest_hash,
        "public_component_model_id": (report.model_release.imported_candidate_experiment_id),
        "public_artifact_hash": report.source.candidate_bundle_sha256,
    }
    if (
        config_map != proxy_config_map_document(proxy_source)
        or stable
        != inference_service_document(
            variant="stable",
            host_port=report.kserve.stable_worker_host_port,
            upstream_component_model_id=report.raw_comparator.component_model_id,
            upstream_artifact_hash=report.raw_comparator.bundle_sha256,
            **common,
        )
        or candidate
        != inference_service_document(
            variant="candidate",
            host_port=report.kserve.candidate_worker_host_port,
            upstream_component_model_id=(report.model_release.imported_candidate_experiment_id),
            upstream_artifact_hash=report.source.candidate_bundle_sha256,
            **common,
        )
    ):
        raise RerankerKServeAcceptanceError("reranker_kserve_contract_changed")
    if set(report.kserve.routes) != {"SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"}:
        raise RerankerKServeAcceptanceError("reranker_route_set_changed")
    for stage, binding in report.kserve.routes.items():
        expected = route_document(cast(Any, stage))
        if _load_object(_inside_file(root, Path(binding.path))) != expected:
            raise RerankerKServeAcceptanceError("reranker_route_document_changed")
    final_route = _load_object(_inside_file(root, Path(report.kserve.observed_final_route.path)))
    stable_state = _load_object(
        _inside_file(root, Path(report.kserve.observed_stable_service.path))
    )
    candidate_state = _load_object(
        _inside_file(root, Path(report.kserve.observed_candidate_service.path))
    )
    controller = _load_object(_inside_file(root, Path(report.kserve.controller_deployment.path)))
    manager = _named_container(controller, "manager")
    if not route_condition_is_true(final_route, "Accepted") or not route_condition_is_true(
        final_route, "ResolvedRefs"
    ):
        raise RerankerKServeAcceptanceError("reranker_gateway_route_not_accepted")
    if not route_rules_semantically_equal(final_route, route_document("ROLLED_BACK")):
        raise RerankerKServeAcceptanceError("reranker_final_rollback_route_changed")
    if not resource_ready(stable_state) or not resource_ready(candidate_state):
        raise RerankerKServeAcceptanceError("reranker_inference_service_not_ready")
    if (
        controller.get("metadata", {}).get("name") != KSERVE_CONTROLLER_DEPLOYMENT
        or controller.get("metadata", {}).get("namespace") != "kserve"
        or manager.get("image") != KSERVE_CONTROLLER_IMAGE
        or manager.get("imagePullPolicy") != "IfNotPresent"
    ):
        raise RerankerKServeAcceptanceError("reranker_kserve_controller_changed")


def _verify_stages(report: EnterpriseRerankerKServeAcceptanceReport) -> None:
    stages = {item.stage: item for item in report.stages}
    if set(stages) != {"SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"}:
        raise RerankerKServeAcceptanceError("reranker_stage_evidence_incomplete")
    if (
        stages["SHADOW"].request_count < 500
        or stages["SHADOW"].candidate_request_delta < 500
        or not stages["SHADOW"].candidate_mirror_observed
        or stages["CANARY_5"].request_count < 1_000
        or stages["CANARY_5"].candidate_response_count < 1
        or not 0.01 <= stages["CANARY_5"].candidate_response_ratio <= 0.15
        or stages["CANARY_25"].request_count < 5_000
        or stages["CANARY_25"].candidate_response_count < 1
        or not 0.12 <= stages["CANARY_25"].candidate_response_ratio <= 0.40
        or stages["ROLLED_BACK"].request_count < 20
        or stages["ROLLED_BACK"].candidate_request_delta != 0
        or stages["ROLLED_BACK"].candidate_response_count != 0
        or stages["ROLLED_BACK"].stable_response_count < 1
    ):
        raise RerankerKServeAcceptanceError("reranker_traffic_evidence_invalid")


def route_condition_is_true(document: dict[str, Any], condition_type: str) -> bool:
    generation = document.get("metadata", {}).get("generation")
    parents = document.get("status", {}).get("parents", [])
    if not isinstance(generation, int) or not isinstance(parents, list):
        return False
    for parent in parents:
        if not isinstance(parent, dict):
            continue
        conditions = parent.get("conditions", [])
        if not isinstance(conditions, list):
            continue
        if any(
            isinstance(condition, dict)
            and condition.get("type") == condition_type
            and condition.get("status") == "True"
            and condition.get("observedGeneration") == generation
            for condition in conditions
        ):
            return True
    return False


def route_rules_semantically_equal(
    observed: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    return _normalized_route_rules(observed) == _normalized_route_rules(expected)


def _normalized_route_rules(document: dict[str, Any]) -> list[dict[str, Any]]:
    rules = document.get("spec", {}).get("rules", [])
    if not isinstance(rules, list):
        return []
    normalized: list[dict[str, Any]] = []
    for raw_rule in rules:
        if not isinstance(raw_rule, dict):
            continue
        rule = json.loads(json.dumps(raw_rule))
        for backend in rule.get("backendRefs", []):
            if isinstance(backend, dict):
                backend.pop("group", None)
                backend.pop("kind", None)
        for item in rule.get("filters", []):
            reference = item.get("requestMirror", {}).get("backendRef", {})
            if isinstance(reference, dict):
                reference.pop("group", None)
                reference.pop("kind", None)
        normalized.append(rule)
    return normalized


def resource_ready(document: dict[str, Any]) -> bool:
    conditions = document.get("status", {}).get("conditions", [])
    return isinstance(conditions, list) and any(
        isinstance(condition, dict)
        and condition.get("type") == "Ready"
        and condition.get("status") == "True"
        for condition in conditions
    )


def _named_container(document: dict[str, Any], name: str) -> dict[str, Any]:
    spec = document.get("spec")
    template = spec.get("template") if isinstance(spec, dict) else None
    pod_spec = template.get("spec") if isinstance(template, dict) else None
    containers = pod_spec.get("containers") if isinstance(pod_spec, dict) else None
    matches = (
        [item for item in containers if isinstance(item, dict) and item.get("name") == name]
        if isinstance(containers, list)
        else []
    )
    if len(matches) != 1:
        raise RerankerKServeAcceptanceError("reranker_kserve_controller_changed")
    return matches[0]


def _target_score(document: dict[str, Any], chunk_id: str) -> float:
    body = document.get("body")
    items = body.get("items") if isinstance(body, dict) else None
    if not isinstance(items, list):
        raise RerankerKServeAcceptanceError("reranker_quality_response_invalid")
    matches = [
        item for item in items if isinstance(item, dict) and item.get("chunk_id") == chunk_id
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("score"), (int, float)):
        raise RerankerKServeAcceptanceError("reranker_quality_response_invalid")
    return float(matches[0]["score"])


def file_binding(
    root: Path,
    path: Path,
    *,
    reported_path: Path | None = None,
) -> dict[str, Any]:
    target = path.resolve(strict=True)
    if reported_path is None:
        try:
            reported_path = target.relative_to(root)
        except ValueError as exc:
            raise RerankerKServeAcceptanceError(
                "Reranker acceptance file escaped repository root"
            ) from exc
    return {
        "path": reported_path.as_posix(),
        "size_bytes": target.stat().st_size,
        "sha256": file_sha256(target),
    }


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_sha256(document: object) -> str:
    return sha256(
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _verify_binding(root: Path, binding: FileBinding) -> Path:
    path = _inside_file(root, Path(binding.path))
    if path.stat().st_size != binding.size_bytes or file_sha256(path) != binding.sha256:
        raise RerankerKServeAcceptanceError("reranker_acceptance_file_binding_changed")
    return path


def _inside_file(root: Path, value: Path) -> Path:
    target = value if value.is_absolute() else root / value
    try:
        resolved = target.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise RerankerKServeAcceptanceError("Reranker acceptance path escaped repository") from exc
    if not resolved.is_file():
        raise RerankerKServeAcceptanceError("Reranker acceptance file is missing")
    return resolved


def _inside_directory(root: Path, value: Path) -> Path:
    target = value if value.is_absolute() else root / value
    try:
        resolved = target.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise RerankerKServeAcceptanceError(
            "Reranker acceptance directory escaped repository"
        ) from exc
    if not resolved.is_dir():
        raise RerankerKServeAcceptanceError("Reranker acceptance directory is missing")
    return resolved


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RerankerKServeAcceptanceError("Reranker acceptance JSON is invalid") from exc
    if not isinstance(value, dict):
        raise RerankerKServeAcceptanceError("Reranker acceptance JSON is invalid")
    return cast(dict[str, Any], value)


def _load_array(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RerankerKServeAcceptanceError("Reranker acceptance JSON is invalid") from exc
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise RerankerKServeAcceptanceError("Reranker container inspection is invalid")
    return cast(dict[str, Any], value[0])


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "ACCEPTANCE_RELATIVE",
    "CANDIDATE_SERVICE",
    "CANDIDATE_WORKER_CONTAINER",
    "CLASSIFICATION",
    "CONFIG_MAP_NAME",
    "ENDPOINT_PATH",
    "EXCLUDED_RAW_COMPARATOR_FILES",
    "EnterpriseRerankerKServeAcceptanceReport",
    "GATEWAY_NAME",
    "HOSTNAME",
    "KUBE_CONTEXT",
    "NAMESPACE",
    "OUTPUT_RELATIVE",
    "PROXY_SOURCE_RELATIVE",
    "ROUTE_NAME",
    "ROUTE_PATH_PREFIX",
    "RUNTIME_IMAGE_DIGEST",
    "RUNTIME_IMAGE_REPOSITORY",
    "RUNTIME_IMAGE_TAG",
    "RerankerKServeAcceptanceError",
    "RolloutStageEvidence",
    "STABLE_SERVICE",
    "STABLE_WORKER_CONTAINER",
    "STATUS",
    "document_sha256",
    "file_binding",
    "file_sha256",
    "finalize_acceptance",
    "inference_service_document",
    "proxy_config_map_document",
    "route_condition_is_true",
    "route_document",
    "route_rules_semantically_equal",
    "verify_reranker_kserve_acceptance",
    "write_acceptance",
]
