"""Contracts for the project-authorized VLM KServe rollout acceptance."""

from __future__ import annotations

import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.simulation.vlm_enterprise_staging import (
    EnterpriseVlmStagingAdoptionReport,
    verify_enterprise_vlm_staging_adoption,
)
from industrial_ops_agent.training.gpu_promotion_security import (
    hardened_container_security_context,
)

SCHEMA_VERSION: Literal["enterprise-vlm-kserve-rollout/v1"] = (
    "enterprise-vlm-kserve-rollout/v1"
)
STATUS: Literal["VLM_LOCAL_KSERVE_SHADOW_CANARY_ROLLBACK_PASSED"] = (
    "VLM_LOCAL_KSERVE_SHADOW_CANARY_ROLLBACK_PASSED"
)
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = (
    "LOCAL_STAGING_PROJECT_AUTHORIZED"
)
NAMESPACE = "ioap-gpu-promotion-lab"
GATEWAY_NAME = "ioap-gpu-lab-gateway"
HOSTNAME = "gpu-lab.local"
ROUTE_PATH_PREFIX = "/v1/vlm"
CONFIG_MAP_NAME = "ioap-vlm-shadow-proxy"
STABLE_SERVICE = "ioap-vlm-stable"
CANDIDATE_SERVICE = "ioap-vlm-candidate"
ROUTE_NAME = "ioap-vlm-lab-route"
PROXY_IMAGE = (
    "industrial-ops/gpu-promotion-runtime:ac9f90725e26-4844463141b6-8567fe9114e7"
)
OUTPUT_RELATIVE = Path("artifacts/m7-vlm-kserve-shadow")
ADOPTION_RELATIVE = Path("artifacts/m7-vlm-enterprise-staging/acceptance.json")
ADAPTER_RELATIVE = Path(
    "artifacts/m7-vlm-checkpoint-continuation-rescored-lab/"
    "adapters/lora-continuation/adapter_model.safetensors"
)
PROXY_SOURCE_RELATIVE = Path(
    "src/industrial_ops_agent/multimodal/vlm_kserve_proxy.py"
)


class VlmKServeRolloutError(RuntimeError):
    """The dynamic VLM rollout evidence is incomplete, stale, or inconsistent."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VlmProjectReleaseEvidence(_ClosedModel):
    schema_version: Literal["project-model-release-manifest/v1"]
    release_id: str = Field(min_length=1)
    manifest_path: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_environment: Literal["ENTERPRISE_STAGING"]
    release_scope: Literal["PROJECT_INTERNAL"]
    candidate_experiment_id: str = Field(min_length=1)
    evaluation_suite_id: str = Field(min_length=1)
    source_adoption_path: str = Field(min_length=1)
    source_adoption_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    formal_project_release_identity_created: Literal[True] = True
    external_production_release_created: Literal[False] = False


class VlmGpuRuntimeEvidence(_ClosedModel):
    topology: Literal["WSL_HOST_GPU_BEHIND_KSERVE_PROXY"]
    actual_gpu_execution: Literal[True] = True
    model_execution_simulated: Literal[False] = False
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    adapter_model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gpu_name: str = Field(min_length=1)
    gpu_total_memory_bytes: int = Field(gt=0)
    gpu_peak_reserved_memory_bytes: int = Field(gt=0)
    torch_version: str = Field(min_length=1)
    transformers_version: str = Field(min_length=1)
    peft_version: str = Field(min_length=1)
    stable_actual_inferences: int = Field(ge=1)
    candidate_actual_inferences: int = Field(ge=1)
    replay_cache_used_for_route_load: Literal[True] = True


class VlmKServeEvidence(_ClosedModel):
    cluster_context: Literal["kind-ioap-gpu-promotion-lab"]
    namespace: Literal["ioap-gpu-promotion-lab"]
    gateway_name: Literal["ioap-gpu-lab-gateway"]
    hostname: Literal["gpu-lab.local"]
    route_name: Literal["ioap-vlm-lab-route"]
    route_path_prefix: Literal["/v1/vlm"]
    stable_inference_service: Literal["ioap-vlm-stable"]
    candidate_inference_service: Literal["ioap-vlm-candidate"]
    stable_ready: Literal[True] = True
    candidate_ready: Literal[True] = True
    route_accepted: Literal[True] = True
    route_resolved_refs: Literal[True] = True
    proxy_image: Literal[
        "industrial-ops/gpu-promotion-runtime:ac9f90725e26-4844463141b6-8567fe9114e7"
    ]
    proxy_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    proxy_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hardened_container_security_context: Literal[True] = True
    kserve_shadow_endpoint_verified: Literal[True] = True
    kserve_gpu_pod_verified: Literal[False] = False
    endpoint_scope: Literal["LOCAL_PORT_FORWARD"]


class VlmRolloutStageEvidence(_ClosedModel):
    stage: Literal["SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"]
    passed: Literal[True] = True
    request_count: int = Field(ge=1)
    stable_request_delta: int = Field(ge=0)
    candidate_request_delta: int = Field(ge=0)
    stable_response_count: int = Field(ge=0)
    candidate_response_count: int = Field(ge=0)
    route_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_sha256s: tuple[str, ...] = Field(min_length=1)
    candidate_mirror_observed: bool


class VlmQualityEvidence(_ClosedModel):
    case_id: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_label: str = Field(min_length=1)
    stable_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_structured_output_valid: Literal[True] = True
    candidate_expected_label_present: Literal[True] = True
    candidate_region_iou: float = Field(ge=0.5, le=1.0)
    human_confirmation_required: Literal[True] = True


class VlmRolloutCleanupEvidence(_ClosedModel):
    gpu_worker_stopped: Literal[True] = True
    port_forwards_stopped: Literal[True] = True
    vlm_kubernetes_resources_removed: Literal[True] = True
    cluster_stopped_after_acceptance: bool
    evidence_is_historical_after_cleanup: Literal[True] = True


class EnterpriseVlmKServeRolloutReport(_ClosedModel):
    schema_version: Literal["enterprise-vlm-kserve-rollout/v1"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal["VLM_LOCAL_KSERVE_SHADOW_CANARY_ROLLBACK_PASSED"] = STATUS
    release: VlmProjectReleaseEvidence
    runtime: VlmGpuRuntimeEvidence
    kserve: VlmKServeEvidence
    quality: VlmQualityEvidence
    stages: tuple[VlmRolloutStageEvidence, ...] = Field(min_length=4, max_length=4)
    cleanup: VlmRolloutCleanupEvidence
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_project_release_manifest(
    adoption: EnterpriseVlmStagingAdoptionReport,
    *,
    source_adoption_sha256: str,
) -> dict[str, Any]:
    base = {
        "schema_version": "project-model-release-manifest/v1",
        "classification": CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "target_environment": "ENTERPRISE_STAGING",
        "component": "VLM",
        "model_id": adoption.candidate.model_id,
        "model_revision": adoption.candidate.model_revision,
        "candidate_experiment_id": adoption.candidate.experiment_id,
        "evaluation_suite_id": adoption.candidate.evaluation_suite_id,
        "adapter_bundle_sha256": adoption.candidate.artifact_sha256,
        "source_adoption_path": ADOPTION_RELATIVE.as_posix(),
        "source_adoption_sha256": source_adoption_sha256,
        "source_evidence_chain_sha256": adoption.evidence_chain_sha256,
        "model_alias": adoption.runtime.model_alias,
        "human_confirmation_required": True,
        "deployment": {
            "provider": "KSERVE_GATEWAY_API",
            "namespace": NAMESPACE,
            "gateway_name": GATEWAY_NAME,
            "hostname": HOSTNAME,
            "route_name": ROUTE_NAME,
            "stable_service_name": STABLE_SERVICE,
            "candidate_service_name": CANDIDATE_SERVICE,
        },
        "external_production_release": False,
    }
    identity = _digest(base)
    return {**base, "release_id": f"model-release-vlm-{identity[:24]}"}


def proxy_config_map_document(proxy_source: str) -> dict[str, Any]:
    if not proxy_source.strip():
        raise ValueError("proxy source is empty")
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIG_MAP_NAME, "namespace": NAMESPACE},
        "data": {"vlm_kserve_proxy.py": proxy_source},
    }


def inference_service_document(
    *,
    variant: Literal["stable", "candidate"],
    host_gateway: str,
    host_port: int,
    proxy_image_digest: str,
) -> dict[str, Any]:
    if not host_gateway or not 1 <= host_port <= 65_535:
        raise ValueError("host GPU endpoint is invalid")
    if not proxy_image_digest.startswith("sha256:"):
        raise ValueError("proxy image digest is invalid")
    name = STABLE_SERVICE if variant == "stable" else CANDIDATE_SERVICE
    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "annotations": {"serving.kserve.io/deploymentMode": "Standard"},
            "labels": {
                "ioap.openai.com/acceptance": "enterprise-vlm-kserve-rollout-v1",
                "ioap.openai.com/model-variant": variant,
                "ioap.openai.com/runtime-image-digest": proxy_image_digest.removeprefix(
                    "sha256:"
                )[:63],
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
                        "command": ["python", "-u", "/app/vlm_kserve_proxy.py"],
                        "env": [
                            {"name": "PORT", "value": "8080"},
                            {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                            {"name": "PYTHONPYCACHEPREFIX", "value": "/tmp/pycache"},
                            {"name": "IOAP_VLM_VARIANT", "value": variant},
                            {
                                "name": "IOAP_VLM_UPSTREAM_URL",
                                "value": f"http://{host_gateway}:{host_port}",
                            },
                            {
                                "name": "IOAP_VLM_UPSTREAM_TIMEOUT_SECONDS",
                                "value": "300",
                            },
                        ],
                        "ports": [{"name": "http1", "containerPort": 8080}],
                        "startupProbe": {
                            "httpGet": {"path": "/health/ready", "port": 8080},
                            "periodSeconds": 3,
                            "failureThreshold": 100,
                        },
                        "readinessProbe": {
                            "httpGet": {"path": "/health/ready", "port": 8080},
                            "periodSeconds": 5,
                            "failureThreshold": 6,
                        },
                        "securityContext": hardened_container_security_context(),
                        "resources": {
                            "requests": {"cpu": "50m", "memory": "64Mi"},
                            "limits": {"cpu": "500m", "memory": "256Mi"},
                        },
                        "volumeMounts": [
                            {
                                "name": "proxy-code",
                                "mountPath": "/app/vlm_kserve_proxy.py",
                                "subPath": "vlm_kserve_proxy.py",
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
            "filters": [
                {"type": "RequestMirror", "requestMirror": {"backendRef": candidate}}
            ],
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
        raise ValueError("VLM rollout stage is invalid")
    rule["matches"] = [
        {"path": {"type": "PathPrefix", "value": ROUTE_PATH_PREFIX}}
    ]
    rule["timeouts"] = {"request": "300s", "backendRequest": "300s"}
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


def finalize_report(
    unsigned: dict[str, Any],
) -> EnterpriseVlmKServeRolloutReport:
    draft = EnterpriseVlmKServeRolloutReport.model_validate(
        {**unsigned, "evidence_chain_sha256": "0" * 64}
    )
    payload = draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    return draft.model_copy(update={"evidence_chain_sha256": _digest(payload)})


def write_rollout_report(
    report: EnterpriseVlmKServeRolloutReport,
    path: Path,
) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def verify_vlm_kserve_rollout(
    repo_root: Path,
    acceptance_path: Path = OUTPUT_RELATIVE / "acceptance.json",
) -> EnterpriseVlmKServeRolloutReport:
    root = repo_root.resolve(strict=True)
    report_path = _inside_file(root, acceptance_path)
    report = EnterpriseVlmKServeRolloutReport.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != _digest(unsigned):
        raise VlmKServeRolloutError("vlm_kserve_rollout_evidence_chain_changed")
    adoption_path = _inside_file(root, Path(report.release.source_adoption_path))
    adoption = verify_enterprise_vlm_staging_adoption(root, adoption_path)
    if (
        sha256(adoption_path.read_bytes()).hexdigest()
        != report.release.source_adoption_sha256
        or adoption.evidence_chain_sha256
        != report.release.source_evidence_chain_sha256
        or adoption.candidate.experiment_id
        != report.release.candidate_experiment_id
        or adoption.candidate.evaluation_suite_id
        != report.release.evaluation_suite_id
        or adoption.candidate.artifact_sha256
        != report.release.adapter_bundle_sha256
    ):
        raise VlmKServeRolloutError("vlm_kserve_release_source_binding_changed")
    manifest_path = _inside_file(root, Path(report.release.manifest_path))
    manifest = _load_object(manifest_path)
    if (
        sha256(manifest_path.read_bytes()).hexdigest()
        != report.release.manifest_sha256
        or manifest.get("release_id") != report.release.release_id
        or manifest
        != build_project_release_manifest(
            adoption,
            source_adoption_sha256=report.release.source_adoption_sha256,
        )
    ):
        raise VlmKServeRolloutError("vlm_project_release_manifest_changed")
    adapter_path = _inside_file(root, ADAPTER_RELATIVE)
    proxy_path = _inside_file(root, PROXY_SOURCE_RELATIVE)
    if (
        sha256(adapter_path.read_bytes()).hexdigest()
        != report.runtime.adapter_model_sha256
        or sha256(proxy_path.read_bytes()).hexdigest()
        != report.kserve.proxy_source_sha256
    ):
        raise VlmKServeRolloutError("vlm_runtime_source_binding_changed")
    stages = {item.stage: item for item in report.stages}
    if set(stages) != {"SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"}:
        raise VlmKServeRolloutError("vlm_rollout_stage_evidence_is_incomplete")
    if (
        stages["SHADOW"].candidate_request_delta < 1
        or not stages["SHADOW"].candidate_mirror_observed
        or stages["CANARY_5"].stable_response_count < 1
        or stages["CANARY_5"].candidate_response_count < 1
        or stages["CANARY_25"].stable_response_count < 1
        or stages["CANARY_25"].candidate_response_count < 1
        or stages["ROLLED_BACK"].candidate_request_delta != 0
        or stages["ROLLED_BACK"].candidate_response_count != 0
        or stages["ROLLED_BACK"].stable_response_count < 1
    ):
        raise VlmKServeRolloutError("vlm_rollout_traffic_evidence_is_invalid")
    return report


def document_sha256(document: object) -> str:
    return _digest(document)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_file(root: Path, value: Path) -> Path:
    target = value if value.is_absolute() else root / value
    target = target.resolve(strict=True)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise VlmKServeRolloutError("VLM rollout path escaped repository root") from exc
    if not target.is_file():
        raise VlmKServeRolloutError("VLM rollout evidence path is not a file")
    return target


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VlmKServeRolloutError("VLM rollout JSON is invalid") from exc
    if not isinstance(value, dict):
        raise VlmKServeRolloutError("VLM rollout JSON root must be an object")
    return value


def _digest(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
