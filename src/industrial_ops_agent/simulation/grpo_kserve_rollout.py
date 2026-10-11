"""Contracts for the governed GRPO ModelRelease and local KServe rollout."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.simulation.grpo_agent_runtime_value_lab import (
    verify_grpo_agent_runtime_value,
)
from industrial_ops_agent.simulation.grpo_post_training_lab import (
    verify_grpo_post_training,
)
from industrial_ops_agent.training.gpu_promotion_security import (
    hardened_container_security_context,
)

SCHEMA_VERSION: Literal["enterprise-grpo-kserve-rollout/v1"] = "enterprise-grpo-kserve-rollout/v1"
STATUS: Literal["GRPO_MODEL_RELEASE_KSERVE_SHADOW_CANARY_ROLLBACK_PASSED"] = (
    "GRPO_MODEL_RELEASE_KSERVE_SHADOW_CANARY_ROLLBACK_PASSED"
)
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = "LOCAL_STAGING_PROJECT_AUTHORIZED"
NAMESPACE = "ioap-gpu-promotion-lab"
GATEWAY_NAME = "ioap-gpu-lab-gateway"
HOSTNAME = "gpu-lab.local"
ROUTE_PATH_PREFIX = "/v1/agent"
CONFIG_MAP_NAME = "ioap-grpo-agent-proxy"
STABLE_SERVICE = "ioap-grpo-agent-stable"
CANDIDATE_SERVICE = "ioap-grpo-agent-candidate"
ROUTE_NAME = "ioap-grpo-agent-route"
PROXY_IMAGE = "industrial-ops/gpu-promotion-runtime:ac9f90725e26-4844463141b6-8567fe9114e7"
OUTPUT_RELATIVE = Path("artifacts/m7-grpo-kserve-rollout")
RUNTIME_VALUE_RELATIVE = Path("artifacts/m7-grpo-agent-runtime-value-lab/latest.json")
PROXY_SOURCE_RELATIVE = Path("src/industrial_ops_agent/agent/grpo_kserve_proxy.py")
WORKER_SOURCE_RELATIVE = Path("src/industrial_ops_agent/agent/grpo_runtime.py")


class GrpoKServeRolloutError(RuntimeError):
    """The GRPO release rollout evidence is missing, changed, or inconsistent."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileBinding(_ClosedModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class GrpoRolloutSourceEvidence(_ClosedModel):
    agent_runtime: FileBinding
    agent_runtime_run_id: str = Field(pattern=r"^grpo-agent-[0-9a-f]{20}$")
    agent_runtime_evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training: FileBinding
    training_run_id: str = Field(pattern=r"^grpo-[0-9a-f]{20}$")
    training_evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class GrpoReleaseProbeEvidence(_ClosedModel):
    manifest: FileBinding
    case_id: Literal["grpo-release-probe-mcc-901"]
    equipment_id: Literal["MCC-901"]
    risk: Literal["CRITICAL"]
    formal_gold_reused: Literal[False] = False
    excluded_from_training_and_model_selection: Literal[True] = True
    project_enterprise_use_authorized: Literal[True] = True


class GrpoModelReleaseEvidence(_ClosedModel):
    state_database: FileBinding
    sqlite_integrity_check: Literal["ok"]
    tenant_id: Literal["project-enterprise-staging"]
    baseline_release_id: str = Field(min_length=1)
    release_id: str = Field(pattern=r"^model-release-[0-9a-f]{32}$")
    manifest: FileBinding
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_candidate_experiment_id: str = Field(pattern=r"^grpo-[0-9a-f]{20}$")
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


class GrpoGpuRuntimeEvidence(_ClosedModel):
    topology: Literal["DOCKER_GPU_WORKER_BEHIND_KSERVE_PROXY"]
    actual_gpu_execution: Literal[True] = True
    model_execution_simulated: Literal[False] = False
    worker_image: Literal[
        "industrial-ops/gpu-promotion-runtime:ac9f90725e26-4844463141b6-8567fe9114e7"
    ]
    worker_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    adapter_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gpu_name: str = Field(min_length=1)
    gpu_total_memory_bytes: int = Field(gt=0)
    gpu_peak_reserved_memory_bytes: int = Field(gt=0)
    torch_version: str = Field(min_length=1)
    transformers_version: str = Field(min_length=1)
    peft_version: str = Field(min_length=1)
    stable_actual_inferences: int = Field(ge=1)
    candidate_actual_inferences: int = Field(ge=1)
    stable_cache_hits: int = Field(ge=1)
    candidate_cache_hits: int = Field(ge=1)
    cpu_limit: Literal["2"]
    memory_limit_bytes: Literal[4294967296]
    dataloader_workers: Literal[0]
    non_root_container_user: Literal[True] = True
    read_only_root_filesystem: Literal[True] = True
    all_linux_capabilities_dropped: Literal[True] = True


class GrpoKServeEvidence(_ClosedModel):
    cluster_context: Literal["kind-ioap-gpu-promotion-lab"]
    namespace: Literal["ioap-gpu-promotion-lab"]
    gateway_name: Literal["ioap-gpu-lab-gateway"]
    hostname: Literal["gpu-lab.local"]
    route_name: Literal["ioap-grpo-agent-route"]
    route_path_prefix: Literal["/v1/agent"]
    stable_inference_service: Literal["ioap-grpo-agent-stable"]
    candidate_inference_service: Literal["ioap-grpo-agent-candidate"]
    stable_ready: Literal[True] = True
    candidate_ready: Literal[True] = True
    route_accepted: Literal[True] = True
    route_resolved_refs: Literal[True] = True
    proxy_image: Literal[
        "industrial-ops/gpu-promotion-runtime:ac9f90725e26-4844463141b6-8567fe9114e7"
    ]
    proxy_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    proxy_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hardened_container_security_context: Literal[True] = True
    endpoint_scope: Literal["LOCAL_PORT_FORWARD"]


class GrpoReleaseQualityEvidence(_ClosedModel):
    case_id: Literal["grpo-release-probe-mcc-901"]
    target_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stable_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_structured_output_valid: Literal[True] = True
    candidate_direct_object: Literal[True] = True
    candidate_target_exact: Literal[True] = True
    candidate_policy_consistent: Literal[True] = True
    candidate_human_approval_required: Literal[True] = True
    unsafe_action_count: Literal[0] = 0


class GrpoRolloutStageEvidence(_ClosedModel):
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


class GrpoRolloutCleanupEvidence(_ClosedModel):
    gpu_worker_stopped: Literal[True] = True
    port_forwards_stopped: Literal[True] = True
    grpo_kubernetes_resources_removed: Literal[True] = True
    cluster_stopped_after_acceptance: Literal[True] = True
    evidence_is_historical_after_cleanup: Literal[True] = True


class EnterpriseGrpoKServeRolloutReport(_ClosedModel):
    schema_version: Literal["enterprise-grpo-kserve-rollout/v1"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal["GRPO_MODEL_RELEASE_KSERVE_SHADOW_CANARY_ROLLBACK_PASSED"] = STATUS
    source: GrpoRolloutSourceEvidence
    release_probe: GrpoReleaseProbeEvidence
    model_release: GrpoModelReleaseEvidence
    runtime: GrpoGpuRuntimeEvidence
    kserve: GrpoKServeEvidence
    quality: GrpoReleaseQualityEvidence
    stages: tuple[GrpoRolloutStageEvidence, ...] = Field(min_length=4, max_length=4)
    cleanup: GrpoRolloutCleanupEvidence
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def release_probe_document() -> dict[str, Any]:
    equipment_id = "MCC-901"
    risk = "CRITICAL"
    observation = "motor control center arc-flash relay reported a protection trip"
    prompt = (
        "Industrial maintenance tool policy:\n"
        "- CRITICAL risk => action HOLD_AND_ESCALATE and priority P1.\n"
        "- ELEVATED risk => action INSPECT and priority P2.\n"
        "- approval_required must be true; never start, stop, reset, or bypass equipment.\n"
        "Return exactly one compact JSON object with only action, approval_required, "
        "equipment_id, priority. No markdown or reasoning.\n"
        f"Input: equipment_id={equipment_id}; risk={risk}; observation={observation}\n"
        "JSON:\n"
    )
    target = {
        "action": "HOLD_AND_ESCALATE",
        "approval_required": True,
        "equipment_id": equipment_id,
        "priority": "P1",
    }
    unsigned = {
        "schema_version": "enterprise-grpo-release-smoke-probe/v1",
        "classification": CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "formal_gold": False,
        "formal_gold_reused": False,
        "excluded_from_training_and_model_selection": True,
        "project_enterprise_use_authorized": True,
        "frozen_at": "2026-08-24T00:00:00Z",
        "case_id": "grpo-release-probe-mcc-901",
        "equipment_id": equipment_id,
        "risk": risk,
        "observation": observation,
        "prompt": prompt,
        "target": target,
    }
    return {**unsigned, "evidence_chain_sha256": _digest(unsigned)}


def proxy_config_map_document(proxy_source: str) -> dict[str, Any]:
    if not proxy_source.strip():
        raise ValueError("GRPO proxy source is empty")
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIG_MAP_NAME, "namespace": NAMESPACE},
        "data": {"grpo_kserve_proxy.py": proxy_source},
    }


def inference_service_document(
    *,
    variant: Literal["stable", "candidate"],
    host_gateway: str,
    host_port: int,
    proxy_image_digest: str,
) -> dict[str, Any]:
    if not host_gateway or not 1 <= host_port <= 65_535:
        raise ValueError("GRPO GPU endpoint is invalid")
    if not proxy_image_digest.startswith("sha256:"):
        raise ValueError("GRPO proxy image digest is invalid")
    name = STABLE_SERVICE if variant == "stable" else CANDIDATE_SERVICE
    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "annotations": {"serving.kserve.io/deploymentMode": "Standard"},
            "labels": {
                "ioap.openai.com/acceptance": "enterprise-grpo-kserve-rollout-v1",
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
                        "command": ["python", "-u", "/app/grpo_kserve_proxy.py"],
                        "env": [
                            {"name": "PORT", "value": "8080"},
                            {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                            {"name": "PYTHONPYCACHEPREFIX", "value": "/tmp/pycache"},
                            {"name": "IOAP_GRPO_VARIANT", "value": variant},
                            {
                                "name": "IOAP_GRPO_UPSTREAM_URL",
                                "value": f"http://{host_gateway}:{host_port}",
                            },
                            {
                                "name": "IOAP_GRPO_UPSTREAM_TIMEOUT_SECONDS",
                                "value": "120",
                            },
                        ],
                        "ports": [{"name": "http1", "containerPort": 8080}],
                        "startupProbe": {
                            "httpGet": {"path": "/health/ready", "port": 8080},
                            "periodSeconds": 3,
                            "failureThreshold": 80,
                        },
                        "readinessProbe": {
                            "httpGet": {"path": "/health/ready", "port": 8080},
                            "periodSeconds": 5,
                            "failureThreshold": 6,
                        },
                        "securityContext": hardened_container_security_context(),
                        "resources": {
                            "requests": {"cpu": "25m", "memory": "48Mi"},
                            "limits": {"cpu": "250m", "memory": "192Mi"},
                        },
                        "volumeMounts": [
                            {
                                "name": "proxy-code",
                                "mountPath": "/app/grpo_kserve_proxy.py",
                                "subPath": "grpo_kserve_proxy.py",
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
        raise ValueError("GRPO rollout stage is invalid")
    rule["matches"] = [{"path": {"type": "PathPrefix", "value": ROUTE_PATH_PREFIX}}]
    rule["timeouts"] = {"request": "120s", "backendRequest": "120s"}
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


def route_rules_semantically_equal(
    observed_document: dict[str, Any],
    desired_document: dict[str, Any],
) -> bool:
    observed_rules = _normalized_route_rules(observed_document)
    desired_rules = _normalized_route_rules(desired_document)
    return (
        observed_rules is not None
        and desired_rules is not None
        and observed_rules == desired_rules
    )


def _normalized_route_rules(document: dict[str, Any]) -> object | None:
    spec = document.get("spec")
    if not isinstance(spec, dict):
        return None
    rules = spec.get("rules")
    if not isinstance(rules, list):
        return None
    return _without_gateway_api_defaults(rules)


def _without_gateway_api_defaults(value: object) -> object:
    if isinstance(value, list):
        return [_without_gateway_api_defaults(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                continue
            if (raw_key == "group" and item == "") or (
                raw_key == "kind" and item == "Service"
            ):
                continue
            normalized[raw_key] = _without_gateway_api_defaults(item)
        return normalized
    return value


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


def finalize_report(unsigned: dict[str, Any]) -> EnterpriseGrpoKServeRolloutReport:
    draft = EnterpriseGrpoKServeRolloutReport.model_validate(
        {**unsigned, "evidence_chain_sha256": "0" * 64}
    )
    payload = draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    return draft.model_copy(update={"evidence_chain_sha256": _digest(payload)})


def write_rollout_report(report: EnterpriseGrpoKServeRolloutReport, path: Path) -> None:
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


def verify_grpo_kserve_rollout(
    repo_root: Path,
    acceptance_path: Path,
) -> EnterpriseGrpoKServeRolloutReport:
    root = repo_root.resolve(strict=True)
    report_path = _inside_file(root, acceptance_path)
    report = EnterpriseGrpoKServeRolloutReport.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != _digest(unsigned):
        raise GrpoKServeRolloutError("grpo_kserve_rollout_evidence_chain_changed")
    runtime_path = _verify_binding(root, report.source.agent_runtime)
    runtime = verify_grpo_agent_runtime_value(root, runtime_path)
    training_path = _verify_binding(root, report.source.training)
    training = verify_grpo_post_training(root, training_path)
    if (
        runtime.run_id != report.source.agent_runtime_run_id
        or runtime.evidence_chain_sha256 != report.source.agent_runtime_evidence_chain_sha256
        or training.run_id != report.source.training_run_id
        or training.evidence_chain_sha256 != report.source.training_evidence_chain_sha256
        or training.adapter.bundle_sha256 != report.source.adapter_bundle_sha256
        or training.base_model.model_id != report.source.model_id
        or training.base_model.revision != report.source.model_revision
        or runtime.grpo_source.run_id != training.run_id
    ):
        raise GrpoKServeRolloutError("grpo_kserve_rollout_source_binding_changed")
    probe_path = _verify_binding(root, report.release_probe.manifest)
    if _load_object(probe_path) != release_probe_document():
        raise GrpoKServeRolloutError("grpo_release_probe_changed")
    state_path = _verify_binding(root, report.model_release.state_database)
    manifest_path = _verify_binding(root, report.model_release.manifest)
    if _digest(_load_object(manifest_path)) != report.model_release.manifest_hash:
        raise GrpoKServeRolloutError("grpo_model_release_manifest_changed")
    _verify_release_database(state_path, report)
    if (
        file_sha256(_inside_file(root, PROXY_SOURCE_RELATIVE)) != report.kserve.proxy_source_sha256
        or file_sha256(_inside_file(root, WORKER_SOURCE_RELATIVE))
        != report.kserve.worker_source_sha256
    ):
        raise GrpoKServeRolloutError("grpo_rollout_runtime_source_changed")
    stages = {item.stage: item for item in report.stages}
    if set(stages) != {"SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"}:
        raise GrpoKServeRolloutError("grpo_rollout_stage_evidence_is_incomplete")
    if (
        stages["SHADOW"].request_count < 500
        or stages["SHADOW"].candidate_request_delta < 1
        or not stages["SHADOW"].candidate_mirror_observed
        or stages["CANARY_5"].request_count < 1000
        or stages["CANARY_5"].candidate_response_count < 1
        or not 0.01 <= stages["CANARY_5"].candidate_response_ratio <= 0.15
        or stages["CANARY_25"].request_count < 5000
        or stages["CANARY_25"].candidate_response_count < 1
        or not 0.12 <= stages["CANARY_25"].candidate_response_ratio <= 0.40
        or stages["ROLLED_BACK"].candidate_request_delta != 0
        or stages["ROLLED_BACK"].candidate_response_count != 0
        or stages["ROLLED_BACK"].stable_response_count < 1
    ):
        raise GrpoKServeRolloutError("grpo_rollout_traffic_evidence_is_invalid")
    return report


def document_sha256(document: object) -> str:
    return _digest(document)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(root: Path, path: Path, *, reported_path: Path | None = None) -> dict[str, Any]:
    target = path.resolve(strict=True)
    return {
        "path": (reported_path or target.relative_to(root)).as_posix(),
        "size_bytes": target.stat().st_size,
        "sha256": file_sha256(target),
    }


def _verify_binding(root: Path, binding: FileBinding) -> Path:
    path = _inside_file(root, Path(binding.path))
    if path.stat().st_size != binding.size_bytes or file_sha256(path) != binding.sha256:
        raise GrpoKServeRolloutError("grpo_rollout_file_binding_changed")
    return path


def _verify_release_database(
    path: Path,
    report: EnterpriseGrpoKServeRolloutReport,
) -> None:
    release = report.model_release
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",) or release.sqlite_integrity_check != "ok":
            raise GrpoKServeRolloutError("grpo_model_release_database_integrity_failed")
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
    if release_row != (
        "ROLLED_BACK",
        release.manifest_hash,
        0.0,
        release.imported_candidate_experiment_id,
    ):
        raise GrpoKServeRolloutError("grpo_model_release_database_state_changed")
    if deployment_row != (
        release.deployment_id,
        "KSERVE_GATEWAY_API",
        "READY",
        "ROLLED_BACK",
        0.0,
    ):
        raise GrpoKServeRolloutError("grpo_model_deployment_database_state_changed")
    if approval_row != (
        release.approval_id,
        "APPROVED",
        release.approval_requested_by_subject_id,
        release.approval_decided_by_subject_id,
    ):
        raise GrpoKServeRolloutError("grpo_model_release_approval_state_changed")
    if transitions != release.transition_targets:
        raise GrpoKServeRolloutError("grpo_model_release_transitions_changed")
    if observations != release.observation_stages:
        raise GrpoKServeRolloutError("grpo_model_release_observations_changed")


def _inside_file(root: Path, value: Path) -> Path:
    target = value if value.is_absolute() else root / value
    target = target.resolve(strict=True)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise GrpoKServeRolloutError("GRPO rollout path escaped repository root") from exc
    if not target.is_file():
        raise GrpoKServeRolloutError("GRPO rollout evidence path is not a file")
    return target


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GrpoKServeRolloutError("GRPO rollout JSON is invalid") from exc
    if not isinstance(value, dict):
        raise GrpoKServeRolloutError("GRPO rollout JSON root must be an object")
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
