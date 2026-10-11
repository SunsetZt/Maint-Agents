"""Immutable readiness contracts for the governed ASR ModelRelease rollout."""

from __future__ import annotations

import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.simulation.asr_enterprise_value_lab import (
    AsrEnterpriseValueLabError,
    verify_asr_enterprise_value,
)
from industrial_ops_agent.training.gpu_promotion_security import (
    hardened_container_security_context,
)

SCHEMA_VERSION: Literal["enterprise-asr-kserve-rollout-readiness/v1"] = (
    "enterprise-asr-kserve-rollout-readiness/v1"
)
STATUS: Literal["ASR_MODEL_RELEASE_KSERVE_ROLLOUT_READY"] = (
    "ASR_MODEL_RELEASE_KSERVE_ROLLOUT_READY"
)
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = (
    "LOCAL_STAGING_PROJECT_AUTHORIZED"
)
NAMESPACE = "ioap-gpu-promotion-lab"
GATEWAY_NAME = "ioap-gpu-lab-gateway"
HOSTNAME = "gpu-lab.local"
ROUTE_PATH_PREFIX = "/v1/asr"
CONFIG_MAP_NAME = "ioap-asr-rollout-proxy"
STABLE_SERVICE = "ioap-asr-stable"
CANDIDATE_SERVICE = "ioap-asr-candidate"
ROUTE_NAME = "ioap-asr-lab-route"
PROXY_IMAGE = "industrial-ops/gpu-promotion-runtime:ac9f90725e26-4844463141b6-8567fe9114e7"
OUTPUT_RELATIVE = Path("artifacts/m7-asr-kserve-rollout")
PROXY_SOURCE_RELATIVE = Path("src/industrial_ops_agent/multimodal/asr_kserve_proxy.py")
WORKER_SOURCE_RELATIVE = Path("src/industrial_ops_agent/multimodal/asr_runtime.py")

PROBE_SOURCE_ID = "1272-135031-0006"
PROBE_CASE_ID = "asr-release-probe-1272-135031-0006"
PROBE_TRANSCRIPT = "I ALSO OFFERED TO HELP YOUR BROTHER TO ESCAPE BUT HE WOULD NOT GO"
PROBE_TRANSCRIPT_SHA256 = "cec64d4fba4eb13f5b1800fb208fd024f31c3194be80dd09537eb0fd8e5c69c5"
PROBE_AUDIO_SHA256 = "65ae3678b861333fb9ba2e5d7db9a0f77fe58ad89d4480872eeb62d22e894dd3"
PROBE_AUDIO_SIZE_BYTES = 87_310
PROBE_SAMPLE_RATE = 16_000
PROBE_SAMPLE_COUNT = 65_440
PROBE_DURATION_SECONDS = 4.09


class AsrKServeRolloutError(RuntimeError):
    """ASR rollout readiness evidence is missing, changed, or inconsistent."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileBinding(_ClosedModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AsrReadinessSource(_ClosedModel):
    acceptance: FileBinding
    run_id: str = Field(pattern=r"^asr-[0-9a-f]{20}$")
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: Literal["openai/whisper-tiny"]
    model_revision: Literal["c4f62d5fce5d73978a1dbfcac01926312acd1a51"]
    image: Literal[
        "industrial-ops/gpu-promotion-runtime:ac9f90725e26-4844463141b6-8567fe9114e7"
    ]
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    adapter_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    actual_gpu_training: Literal[True] = True
    frozen_gold_passed: Literal[True] = True


class AsrReadinessProbe(_ClosedModel):
    manifest: FileBinding
    audio: FileBinding
    case_id: Literal["asr-release-probe-1272-135031-0006"]
    source_id: Literal["1272-135031-0006"]
    transcript_sha256: Literal[
        "cec64d4fba4eb13f5b1800fb208fd024f31c3194be80dd09537eb0fd8e5c69c5"
    ]
    sampling_rate: Literal[16000]
    sample_count: Literal[65440]
    duration_seconds: float = Field(ge=4.09, le=4.09)
    formal_gold: Literal[False] = False
    formal_gold_reused: Literal[False] = False
    excluded_from_training_and_model_selection: Literal[True] = True
    source_id_disjoint_from_governed_splits: Literal[True] = True
    project_enterprise_use_authorized: Literal[True] = True


class AsrReadinessRuntime(_ClosedModel):
    worker_source: FileBinding
    proxy_source: FileBinding
    endpoint_path: Literal["/v1/asr/transcribe"]
    stable_disables_adapter: Literal[True] = True
    candidate_enables_adapter: Literal[True] = True
    request_media_digest_required: Literal[True] = True
    runtime_cache_variant_bound: Literal[True] = True
    exactly_one_gpu_required: Literal[True] = True
    cpu_threads: Literal[2]
    memory_limit_bytes: Literal[4294967296]
    actual_rollout_execution_completed: Literal[False] = False


class AsrKubernetesReadiness(_ClosedModel):
    namespace: Literal["ioap-gpu-promotion-lab"]
    gateway_name: Literal["ioap-gpu-lab-gateway"]
    hostname: Literal["gpu-lab.local"]
    route_name: Literal["ioap-asr-lab-route"]
    stable_inference_service: Literal["ioap-asr-stable"]
    candidate_inference_service: Literal["ioap-asr-candidate"]
    config_map: FileBinding
    stable_service: FileBinding
    candidate_service: FileBinding
    routes: dict[str, FileBinding]
    hardened_container_security_context: Literal[True] = True
    proxy_has_no_gpu_request: Literal[True] = True
    min_replicas_per_variant: Literal[1] = 1
    max_replicas_per_variant: Literal[1] = 1


class AsrModelReleaseReadiness(_ClosedModel):
    component: Literal["ASR"]
    method: Literal["ASR"]
    artifact_kind: Literal["asr_adapter_bundle"]
    target_profile: Literal["ASR_COMPONENT"]
    target_environment: Literal["STAGING"]
    import_api: Literal[
        "/api/v1/enterprise-assets/runtime-bindings/ASR/imports"
    ]
    preview_api: Literal[
        "/api/v1/enterprise-assets/runtime-bindings/ASR/release-draft-preview"
    ]
    existing_approval_fsm_reused: Literal[True] = True
    existing_deployment_fsm_reused: Literal[True] = True
    shadow_canary_rollback_stages: tuple[
        Literal["SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"], ...
    ] = Field(min_length=4, max_length=4)
    formal_model_release_created: Literal[False] = False
    shadow_claim: Literal[False] = False
    canary_claim: Literal[False] = False
    rollback_claim: Literal[False] = False


class EnterpriseAsrKServeReadinessReport(_ClosedModel):
    schema_version: Literal["enterprise-asr-kserve-rollout-readiness/v1"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal["ASR_MODEL_RELEASE_KSERVE_ROLLOUT_READY"] = STATUS
    source: AsrReadinessSource
    probe: AsrReadinessProbe
    runtime: AsrReadinessRuntime
    kubernetes: AsrKubernetesReadiness
    model_release: AsrModelReleaseReadiness
    hard_gates: dict[str, Literal[True]]
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def release_probe_document() -> dict[str, Any]:
    unsigned: dict[str, Any] = {
        "schema_version": "enterprise-asr-release-probe/v1",
        "classification": CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "formal_gold": False,
        "formal_gold_reused": False,
        "excluded_from_training_and_model_selection": True,
        "source_id_disjoint_from_governed_splits": True,
        "project_enterprise_use_authorized": True,
        "frozen_at": "2026-08-24T00:00:00Z",
        "case_id": PROBE_CASE_ID,
        "source_id": PROBE_SOURCE_ID,
        "media": {
            "path": f"{OUTPUT_RELATIVE.as_posix()}/release-probe.flac",
            "mime_type": "audio/flac",
            "sha256": PROBE_AUDIO_SHA256,
            "size_bytes": PROBE_AUDIO_SIZE_BYTES,
            "sampling_rate": PROBE_SAMPLE_RATE,
            "sample_count": PROBE_SAMPLE_COUNT,
            "duration_seconds": PROBE_DURATION_SECONDS,
        },
        "expected_transcript": PROBE_TRANSCRIPT,
        "expected_transcript_sha256": PROBE_TRANSCRIPT_SHA256,
    }
    return {**unsigned, "evidence_chain_sha256": document_sha256(unsigned)}


def proxy_config_map_document(proxy_source: str) -> dict[str, Any]:
    if not proxy_source.strip():
        raise ValueError("ASR proxy source is empty")
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIG_MAP_NAME, "namespace": NAMESPACE},
        "data": {"asr_kserve_proxy.py": proxy_source},
    }


def inference_service_document(
    *,
    variant: Literal["stable", "candidate"],
    host_gateway: str,
    host_port: int,
    proxy_image_digest: str,
) -> dict[str, Any]:
    if not host_gateway or any(character.isspace() for character in host_gateway):
        raise ValueError("ASR host GPU endpoint is invalid")
    if not 1 <= host_port <= 65_535:
        raise ValueError("ASR host GPU endpoint is invalid")
    if not proxy_image_digest.startswith("sha256:"):
        raise ValueError("ASR proxy image digest is invalid")
    name = STABLE_SERVICE if variant == "stable" else CANDIDATE_SERVICE
    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "annotations": {"serving.kserve.io/deploymentMode": "Standard"},
            "labels": {
                "ioap.openai.com/acceptance": "enterprise-asr-kserve-rollout-v1",
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
                        "command": ["python", "-u", "/app/asr_kserve_proxy.py"],
                        "env": [
                            {"name": "PORT", "value": "8080"},
                            {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                            {"name": "PYTHONPYCACHEPREFIX", "value": "/tmp/pycache"},
                            {"name": "IOAP_ASR_VARIANT", "value": variant},
                            {
                                "name": "IOAP_ASR_UPSTREAM_URL",
                                "value": f"http://{host_gateway}:{host_port}",
                            },
                            {
                                "name": "IOAP_ASR_UPSTREAM_TIMEOUT_SECONDS",
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
                                "mountPath": "/app/asr_kserve_proxy.py",
                                "subPath": "asr_kserve_proxy.py",
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
        raise ValueError("ASR rollout stage is invalid")
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


def finalize_readiness(unsigned: dict[str, Any]) -> EnterpriseAsrKServeReadinessReport:
    draft = EnterpriseAsrKServeReadinessReport.model_validate(
        {**unsigned, "evidence_chain_sha256": "0" * 64}
    )
    payload = draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    return draft.model_copy(update={"evidence_chain_sha256": document_sha256(payload)})


def write_readiness_report(
    report: EnterpriseAsrKServeReadinessReport,
    path: Path,
) -> None:
    _write_json(path, report.model_dump(mode="json"))


def verify_asr_kserve_readiness(
    repo_root: Path,
    readiness_path: Path = OUTPUT_RELATIVE / "readiness.json",
) -> EnterpriseAsrKServeReadinessReport:
    root = repo_root.resolve(strict=True)
    path = _inside_file(root, readiness_path)
    report = EnterpriseAsrKServeReadinessReport.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != document_sha256(unsigned):
        raise AsrKServeRolloutError("asr_rollout_readiness_chain_changed")
    acceptance_path = _verify_binding(root, report.source.acceptance)
    try:
        source = verify_asr_enterprise_value(root, acceptance_path)
    except AsrEnterpriseValueLabError as exc:
        raise AsrKServeRolloutError("asr_rollout_source_is_invalid") from exc
    if (
        source.run_id != report.source.run_id
        or source.evidence_chain_sha256 != report.source.evidence_chain_sha256
        or source.base_model.model_id != report.source.model_id
        or source.base_model.revision != report.source.model_revision
        or source.base_model.image_digest != report.source.image_digest
        or source.adapter.bundle_sha256 != report.source.adapter_bundle_sha256
        or not all(source.hard_gates.model_dump().values())
    ):
        raise AsrKServeRolloutError("asr_rollout_source_binding_changed")
    dataset_manifest = _load_object(
        _inside_file(root, Path(source.dataset.manifest.path))
    )
    splits = dataset_manifest.get("splits")
    if not isinstance(splits, dict) or any(
        PROBE_SOURCE_ID in split.get("source_ids", [])
        for split in splits.values()
        if isinstance(split, dict)
    ):
        raise AsrKServeRolloutError("asr_release_probe_is_not_split_independent")
    manifest_path = _verify_binding(root, report.probe.manifest)
    audio_path = _verify_binding(root, report.probe.audio)
    if (
        _load_object(manifest_path) != release_probe_document()
        or file_sha256(audio_path) != PROBE_AUDIO_SHA256
        or audio_path.stat().st_size != PROBE_AUDIO_SIZE_BYTES
    ):
        raise AsrKServeRolloutError("asr_release_probe_changed")
    if (
        file_sha256(_verify_binding(root, report.runtime.worker_source))
        != report.runtime.worker_source.sha256
        or file_sha256(_verify_binding(root, report.runtime.proxy_source))
        != report.runtime.proxy_source.sha256
    ):
        raise AsrKServeRolloutError("asr_runtime_source_changed")
    _verify_kubernetes_documents(root, report)
    if set(report.hard_gates) != {
        "authoritative_training_evidence",
        "release_probe_independence",
        "runtime_source_integrity",
        "stable_candidate_isolation",
        "hardened_proxy_security",
        "model_release_fsm_reuse",
        "no_unexecuted_rollout_claim",
    } or not all(report.hard_gates.values()):
        raise AsrKServeRolloutError("asr_rollout_readiness_gates_changed")
    return report


def _verify_kubernetes_documents(
    root: Path,
    report: EnterpriseAsrKServeReadinessReport,
) -> None:
    config_map = _load_object(_verify_binding(root, report.kubernetes.config_map))
    stable = _load_object(_verify_binding(root, report.kubernetes.stable_service))
    candidate = _load_object(_verify_binding(root, report.kubernetes.candidate_service))
    proxy_source = _inside_file(root, PROXY_SOURCE_RELATIVE).read_text(encoding="utf-8")
    if config_map != proxy_config_map_document(proxy_source):
        raise AsrKServeRolloutError("asr_proxy_config_map_changed")
    stable_container = stable["spec"]["predictor"]["containers"][0]
    candidate_container = candidate["spec"]["predictor"]["containers"][0]
    stable_env = {item["name"]: item["value"] for item in stable_container["env"]}
    candidate_env = {item["name"]: item["value"] for item in candidate_container["env"]}
    if (
        stable_env.get("IOAP_ASR_VARIANT") != "stable"
        or candidate_env.get("IOAP_ASR_VARIANT") != "candidate"
        or stable_env.get("IOAP_ASR_UPSTREAM_URL")
        != candidate_env.get("IOAP_ASR_UPSTREAM_URL")
        or stable_container["securityContext"] != hardened_container_security_context()
        or candidate_container["securityContext"] != hardened_container_security_context()
        or "nvidia.com/gpu" in stable_container["resources"]["limits"]
        or "nvidia.com/gpu" in candidate_container["resources"]["limits"]
    ):
        raise AsrKServeRolloutError("asr_inference_service_contract_changed")
    if set(report.kubernetes.routes) != {"SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"}:
        raise AsrKServeRolloutError("asr_route_plan_is_incomplete")
    for stage, binding in report.kubernetes.routes.items():
        observed = _load_object(_verify_binding(root, binding))
        if observed != route_document(stage):  # type: ignore[arg-type]
            raise AsrKServeRolloutError("asr_route_document_changed")


def file_binding(root: Path, path: Path) -> dict[str, Any]:
    target = path.resolve(strict=True)
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise AsrKServeRolloutError("ASR rollout file escaped repository root") from exc
    return {
        "path": relative.as_posix(),
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
        raise AsrKServeRolloutError("asr_rollout_file_binding_changed")
    return path


def _inside_file(root: Path, value: Path) -> Path:
    target = value if value.is_absolute() else root / value
    target = target.resolve(strict=True)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise AsrKServeRolloutError("ASR rollout path escaped repository root") from exc
    if not target.is_file():
        raise AsrKServeRolloutError("ASR rollout path is not a file")
    return target


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AsrKServeRolloutError("ASR rollout JSON is invalid") from exc
    if not isinstance(value, dict):
        raise AsrKServeRolloutError("ASR rollout JSON root must be an object")
    return value


def _write_json(path: Path, value: object) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
