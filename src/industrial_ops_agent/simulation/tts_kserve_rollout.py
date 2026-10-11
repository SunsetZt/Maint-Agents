"""Immutable readiness contracts for a governed TTS ModelRelease rollout."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.simulation.tts_enterprise_value_lab import (
    TtsEnterpriseValueLabError,
    TtsEnterpriseValueReport,
    verify_tts_enterprise_value,
)
from industrial_ops_agent.training.gpu_promotion_security import (
    hardened_container_security_context,
)

RolloutStage = Literal["SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"]
SCHEMA_VERSION: Literal["enterprise-tts-kserve-rollout-readiness/v1"] = (
    "enterprise-tts-kserve-rollout-readiness/v1"
)
STATUS: Literal["TTS_MODEL_RELEASE_KSERVE_ROLLOUT_READY"] = "TTS_MODEL_RELEASE_KSERVE_ROLLOUT_READY"
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = "LOCAL_STAGING_PROJECT_AUTHORIZED"
NAMESPACE = "ioap-gpu-promotion-lab"
GATEWAY_NAME = "ioap-gpu-lab-gateway"
HOSTNAME = "gpu-lab.local"
ROUTE_PATH_PREFIX = "/v1/audio/speech"
CONFIG_MAP_NAME = "ioap-tts-rollout-proxy"
STABLE_SERVICE = "ioap-tts-stable"
CANDIDATE_SERVICE = "ioap-tts-candidate"
ROUTE_NAME = "ioap-tts-lab-route"
PROXY_IMAGE = "industrial-ops/gpu-promotion-runtime:ac9f90725e26-4844463141b6-8567fe9114e7"
OUTPUT_RELATIVE = Path("artifacts/m7-tts-kserve-rollout")
PROXY_SOURCE_RELATIVE = Path("src/industrial_ops_agent/multimodal/tts_kserve_proxy.py")
WORKER_SOURCE_RELATIVE = Path("src/industrial_ops_agent/multimodal/tts_runtime.py")
PROBE_CASE_ID = "tts-release-probe-compressor-guard-v1"
PROBE_TEXT = "Confirm the compressor guard is latched before restoring auxiliary power."
PROBE_TEXT_SHA256 = "3be84338c2714ad107cb17c5b947140698095fb7c492230971f994f8da7d74ff"


class TtsKServeRolloutError(RuntimeError):
    """TTS rollout readiness evidence is missing, changed, or inconsistent."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileBinding(_ClosedModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TtsReadinessSource(_ClosedModel):
    acceptance: FileBinding
    run_id: str = Field(pattern=r"^tts-[0-9a-f]{20}$")
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    image: str = Field(min_length=1)
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    candidate_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    voice_profile_id: str = Field(min_length=1)
    actual_gpu_training: Literal[True] = True
    frozen_gold_passed: Literal[True] = True


class TtsReadinessProbe(_ClosedModel):
    manifest: FileBinding
    case_id: Literal["tts-release-probe-compressor-guard-v1"]
    text_sha256: Literal["3be84338c2714ad107cb17c5b947140698095fb7c492230971f994f8da7d74ff"]
    voice: Literal["default"] = "default"
    formal_gold: Literal[False] = False
    formal_gold_reused: Literal[False] = False
    excluded_from_training_and_model_selection: Literal[True] = True
    text_disjoint_from_governed_splits: Literal[True] = True
    project_enterprise_use_authorized: Literal[True] = True


class TtsReadinessRuntime(_ClosedModel):
    worker_source: FileBinding
    proxy_source: FileBinding
    endpoint_path: Literal["/v1/audio/speech"]
    openai_compatible_binary_wav: Literal[True] = True
    stable_uses_base_postnet: Literal[True] = True
    candidate_uses_approved_postnet: Literal[True] = True
    runtime_cache_variant_bound: Literal[True] = True
    unauthorized_variant_override_rejected: Literal[True] = True
    exactly_one_gpu_required: Literal[True] = True
    cpu_threads: Literal[2]
    memory_limit_bytes: Literal[4294967296]
    actual_rollout_execution_completed: Literal[False] = False


class TtsKubernetesReadiness(_ClosedModel):
    namespace: Literal["ioap-gpu-promotion-lab"]
    gateway_name: Literal["ioap-gpu-lab-gateway"]
    hostname: Literal["gpu-lab.local"]
    route_name: Literal["ioap-tts-lab-route"]
    stable_inference_service: Literal["ioap-tts-stable"]
    candidate_inference_service: Literal["ioap-tts-candidate"]
    host_gateway: str = Field(min_length=1)
    host_port: int = Field(ge=1, le=65535)
    proxy_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config_map: FileBinding
    stable_service: FileBinding
    candidate_service: FileBinding
    routes: dict[str, FileBinding]
    hardened_container_security_context: Literal[True] = True
    proxy_has_no_gpu_request: Literal[True] = True
    min_replicas_per_variant: Literal[1] = 1
    max_replicas_per_variant: Literal[1] = 1


class TtsModelReleaseReadiness(_ClosedModel):
    component: Literal["TTS"]
    method: Literal["TTS"]
    artifact_kind: Literal["tts_model_bundle"]
    target_profile: Literal["TTS_COMPONENT"]
    target_environment: Literal["STAGING"]
    import_api: Literal["/api/v1/enterprise-assets/runtime-bindings/TTS/imports"]
    preview_api: Literal["/api/v1/enterprise-assets/runtime-bindings/TTS/release-draft-preview"]
    existing_approval_fsm_reused: Literal[True] = True
    existing_deployment_fsm_reused: Literal[True] = True
    shadow_canary_rollback_stages: tuple[RolloutStage, ...] = Field(min_length=4, max_length=4)
    formal_model_release_created: Literal[False] = False
    shadow_claim: Literal[False] = False
    canary_claim: Literal[False] = False
    rollback_claim: Literal[False] = False


class EnterpriseTtsKServeReadinessReport(_ClosedModel):
    schema_version: Literal["enterprise-tts-kserve-rollout-readiness/v1"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal["TTS_MODEL_RELEASE_KSERVE_ROLLOUT_READY"] = STATUS
    source: TtsReadinessSource
    probe: TtsReadinessProbe
    runtime: TtsReadinessRuntime
    kubernetes: TtsKubernetesReadiness
    model_release: TtsModelReleaseReadiness
    hard_gates: dict[str, Literal[True]]
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def release_probe_document() -> dict[str, Any]:
    unsigned: dict[str, Any] = {
        "schema_version": "enterprise-tts-release-probe/v1",
        "classification": CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "formal_gold": False,
        "formal_gold_reused": False,
        "excluded_from_training_and_model_selection": True,
        "text_disjoint_from_governed_splits": True,
        "project_enterprise_use_authorized": True,
        "frozen_at": "2026-08-25T00:00:00Z",
        "case_id": PROBE_CASE_ID,
        "input": PROBE_TEXT,
        "input_sha256": PROBE_TEXT_SHA256,
        "voice": "default",
        "response_format": "wav",
    }
    return {**unsigned, "evidence_chain_sha256": document_sha256(unsigned)}


def proxy_config_map_document(proxy_source: str) -> dict[str, Any]:
    if not proxy_source.strip():
        raise ValueError("TTS proxy source is empty")
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIG_MAP_NAME, "namespace": NAMESPACE},
        "data": {"tts_kserve_proxy.py": proxy_source},
    }


def inference_service_document(
    *,
    variant: Literal["stable", "candidate"],
    host_gateway: str,
    host_port: int,
    proxy_image_digest: str,
) -> dict[str, Any]:
    if not host_gateway or any(character.isspace() for character in host_gateway):
        raise ValueError("TTS host GPU endpoint is invalid")
    if not 1 <= host_port <= 65_535:
        raise ValueError("TTS host GPU endpoint is invalid")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", proxy_image_digest):
        raise ValueError("TTS proxy image digest is invalid")
    name = STABLE_SERVICE if variant == "stable" else CANDIDATE_SERVICE
    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "annotations": {"serving.kserve.io/deploymentMode": "Standard"},
            "labels": {
                "ioap.openai.com/acceptance": "enterprise-tts-kserve-rollout-v1",
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
                        "command": ["python", "-u", "/app/tts_kserve_proxy.py"],
                        "env": [
                            {"name": "PORT", "value": "8080"},
                            {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                            {"name": "PYTHONPYCACHEPREFIX", "value": "/tmp/pycache"},
                            {"name": "IOAP_TTS_VARIANT", "value": variant},
                            {
                                "name": "IOAP_TTS_UPSTREAM_URL",
                                "value": f"http://{host_gateway}:{host_port}",
                            },
                            {
                                "name": "IOAP_TTS_UPSTREAM_TIMEOUT_SECONDS",
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
                                "mountPath": "/app/tts_kserve_proxy.py",
                                "subPath": "tts_kserve_proxy.py",
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


def route_document(stage: RolloutStage) -> dict[str, Any]:
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
    else:  # pragma: no cover
        raise ValueError("TTS rollout stage is invalid")
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


def build_tts_kserve_readiness(
    repo_root: Path,
    *,
    acceptance_path: Path,
    host_gateway: str,
    host_port: int,
    proxy_image_digest: str,
) -> Path:
    root = repo_root.resolve(strict=True)
    source = verify_tts_enterprise_value(root, acceptance_path)
    _require_probe_independent(root, source)
    output = root / OUTPUT_RELATIVE
    kubernetes = output / "kubernetes"
    kubernetes.mkdir(parents=True, exist_ok=True)
    probe_path = output / "release-probe.json"
    _write_json(probe_path, release_probe_document())
    proxy_path = (root / PROXY_SOURCE_RELATIVE).resolve(strict=True)
    worker_path = (root / WORKER_SOURCE_RELATIVE).resolve(strict=True)
    config_path = kubernetes / "proxy-configmap.json"
    stable_path = kubernetes / "stable-inferenceservice.json"
    candidate_path = kubernetes / "candidate-inferenceservice.json"
    _write_json(
        config_path,
        proxy_config_map_document(proxy_path.read_text(encoding="utf-8")),
    )
    _write_json(
        stable_path,
        inference_service_document(
            variant="stable",
            host_gateway=host_gateway,
            host_port=host_port,
            proxy_image_digest=proxy_image_digest,
        ),
    )
    _write_json(
        candidate_path,
        inference_service_document(
            variant="candidate",
            host_gateway=host_gateway,
            host_port=host_port,
            proxy_image_digest=proxy_image_digest,
        ),
    )
    route_paths: dict[str, Path] = {}
    for stage in ("SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"):
        route_path = kubernetes / f"route-{stage.lower()}.json"
        _write_json(route_path, route_document(stage))
        route_paths[stage] = route_path
    base_model = _base_model(source)
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "external_enterprise_production_claim": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": STATUS,
        "source": {
            "acceptance": file_binding(root, _inside_file(root, acceptance_path)),
            "run_id": source.run_id,
            "evidence_chain_sha256": source.evidence_chain_sha256,
            "model_id": base_model.model_id,
            "model_revision": base_model.revision,
            "image": source.image,
            "image_digest": source.image_digest,
            "candidate_bundle_sha256": source.candidate.bundle_sha256,
            "voice_profile_id": source.dataset.voice_profile_id,
            "actual_gpu_training": True,
            "frozen_gold_passed": True,
        },
        "probe": {
            "manifest": file_binding(root, probe_path),
            "case_id": PROBE_CASE_ID,
            "text_sha256": PROBE_TEXT_SHA256,
            "voice": "default",
            "formal_gold": False,
            "formal_gold_reused": False,
            "excluded_from_training_and_model_selection": True,
            "text_disjoint_from_governed_splits": True,
            "project_enterprise_use_authorized": True,
        },
        "runtime": {
            "worker_source": file_binding(root, worker_path),
            "proxy_source": file_binding(root, proxy_path),
            "endpoint_path": "/v1/audio/speech",
            "openai_compatible_binary_wav": True,
            "stable_uses_base_postnet": True,
            "candidate_uses_approved_postnet": True,
            "runtime_cache_variant_bound": True,
            "unauthorized_variant_override_rejected": True,
            "exactly_one_gpu_required": True,
            "cpu_threads": 2,
            "memory_limit_bytes": 4 * 1024**3,
            "actual_rollout_execution_completed": False,
        },
        "kubernetes": {
            "namespace": NAMESPACE,
            "gateway_name": GATEWAY_NAME,
            "hostname": HOSTNAME,
            "route_name": ROUTE_NAME,
            "stable_inference_service": STABLE_SERVICE,
            "candidate_inference_service": CANDIDATE_SERVICE,
            "host_gateway": host_gateway,
            "host_port": host_port,
            "proxy_image_digest": proxy_image_digest,
            "config_map": file_binding(root, config_path),
            "stable_service": file_binding(root, stable_path),
            "candidate_service": file_binding(root, candidate_path),
            "routes": {stage: file_binding(root, path) for stage, path in route_paths.items()},
            "hardened_container_security_context": True,
            "proxy_has_no_gpu_request": True,
            "min_replicas_per_variant": 1,
            "max_replicas_per_variant": 1,
        },
        "model_release": {
            "component": "TTS",
            "method": "TTS",
            "artifact_kind": "tts_model_bundle",
            "target_profile": "TTS_COMPONENT",
            "target_environment": "STAGING",
            "import_api": "/api/v1/enterprise-assets/runtime-bindings/TTS/imports",
            "preview_api": ("/api/v1/enterprise-assets/runtime-bindings/TTS/release-draft-preview"),
            "existing_approval_fsm_reused": True,
            "existing_deployment_fsm_reused": True,
            "shadow_canary_rollback_stages": [
                "SHADOW",
                "CANARY_5",
                "CANARY_25",
                "ROLLED_BACK",
            ],
            "formal_model_release_created": False,
            "shadow_claim": False,
            "canary_claim": False,
            "rollback_claim": False,
        },
        "hard_gates": {
            "authoritative_training_evidence": True,
            "release_probe_independence": True,
            "runtime_source_integrity": True,
            "stable_candidate_isolation": True,
            "hardened_proxy_security": True,
            "model_release_fsm_reuse": True,
            "no_unexecuted_rollout_claim": True,
        },
    }
    report = finalize_readiness(unsigned)
    readiness_path = output / "readiness.json"
    _write_json(readiness_path, report.model_dump(mode="json"))
    verify_tts_kserve_readiness(root, readiness_path)
    return readiness_path


def finalize_readiness(unsigned: dict[str, Any]) -> EnterpriseTtsKServeReadinessReport:
    draft = EnterpriseTtsKServeReadinessReport.model_validate(
        {**unsigned, "evidence_chain_sha256": "0" * 64}
    )
    payload = draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    return draft.model_copy(update={"evidence_chain_sha256": document_sha256(payload)})


def verify_tts_kserve_readiness(
    repo_root: Path,
    readiness_path: Path = OUTPUT_RELATIVE / "readiness.json",
) -> EnterpriseTtsKServeReadinessReport:
    root = repo_root.resolve(strict=True)
    path = _inside_file(root, readiness_path)
    report = EnterpriseTtsKServeReadinessReport.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != document_sha256(unsigned):
        raise TtsKServeRolloutError("tts_rollout_readiness_chain_changed")
    acceptance_path = _verify_binding(root, report.source.acceptance)
    try:
        source = verify_tts_enterprise_value(root, acceptance_path)
    except TtsEnterpriseValueLabError as exc:
        raise TtsKServeRolloutError("tts_rollout_source_is_invalid") from exc
    base_model = _base_model(source)
    if (
        source.run_id != report.source.run_id
        or source.evidence_chain_sha256 != report.source.evidence_chain_sha256
        or base_model.model_id != report.source.model_id
        or base_model.revision != report.source.model_revision
        or source.image != report.source.image
        or source.image_digest != report.source.image_digest
        or source.candidate.bundle_sha256 != report.source.candidate_bundle_sha256
        or source.dataset.voice_profile_id != report.source.voice_profile_id
        or not all(source.hard_gates.model_dump().values())
    ):
        raise TtsKServeRolloutError("tts_rollout_source_binding_changed")
    _require_probe_independent(root, source)
    if _load_object(_verify_binding(root, report.probe.manifest)) != release_probe_document():
        raise TtsKServeRolloutError("tts_release_probe_changed")
    _verify_binding(root, report.runtime.worker_source)
    _verify_binding(root, report.runtime.proxy_source)
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
        raise TtsKServeRolloutError("tts_rollout_readiness_gates_changed")
    return report


def _verify_kubernetes_documents(
    root: Path,
    report: EnterpriseTtsKServeReadinessReport,
) -> None:
    proxy_source = _inside_file(root, PROXY_SOURCE_RELATIVE).read_text(encoding="utf-8")
    if _load_object(_verify_binding(root, report.kubernetes.config_map)) != (
        proxy_config_map_document(proxy_source)
    ):
        raise TtsKServeRolloutError("tts_proxy_config_map_changed")
    for variant, binding in (
        ("stable", report.kubernetes.stable_service),
        ("candidate", report.kubernetes.candidate_service),
    ):
        observed = _load_object(_verify_binding(root, binding))
        expected = inference_service_document(
            variant=cast(Any, variant),
            host_gateway=report.kubernetes.host_gateway,
            host_port=report.kubernetes.host_port,
            proxy_image_digest=report.kubernetes.proxy_image_digest,
        )
        if observed != expected:
            raise TtsKServeRolloutError("tts_inference_service_contract_changed")
        container = observed["spec"]["predictor"]["containers"][0]
        if (
            container["securityContext"] != hardened_container_security_context()
            or "nvidia.com/gpu" in container["resources"]["limits"]
        ):
            raise TtsKServeRolloutError("tts_proxy_security_contract_changed")
    if set(report.kubernetes.routes) != {
        "SHADOW",
        "CANARY_5",
        "CANARY_25",
        "ROLLED_BACK",
    }:
        raise TtsKServeRolloutError("tts_route_plan_is_incomplete")
    for stage, binding in report.kubernetes.routes.items():
        if _load_object(_verify_binding(root, binding)) != route_document(
            cast(RolloutStage, stage)
        ):
            raise TtsKServeRolloutError("tts_route_document_changed")


def _require_probe_independent(root: Path, source: TtsEnterpriseValueReport) -> None:
    training = _load_object(_inside_file(root, Path(source.dataset.training_manifest.path)))
    gold = _load_object(_inside_file(root, Path(source.dataset.final_gold_manifest.path)))
    governed_texts = {_normalize_text(item) for item in _all_text_values(training, gold)}
    if _normalize_text(PROBE_TEXT) in governed_texts:
        raise TtsKServeRolloutError("tts_release_probe_is_not_split_independent")


def _all_text_values(*documents: object) -> tuple[str, ...]:
    values: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "text" and isinstance(item, str):
                    values.append(item)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for document in documents:
        visit(document)
    return tuple(values)


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _base_model(source: TtsEnterpriseValueReport) -> Any:
    matches = [item for item in source.input_components if item.role == "base_model"]
    if len(matches) != 1:
        raise TtsKServeRolloutError("tts_base_model_binding_changed")
    return matches[0]


def file_binding(root: Path, path: Path) -> dict[str, Any]:
    target = path.resolve(strict=True)
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise TtsKServeRolloutError("TTS rollout file escaped repository root") from exc
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
        raise TtsKServeRolloutError("tts_rollout_file_binding_changed")
    return path


def _inside_file(root: Path, value: Path) -> Path:
    target = value if value.is_absolute() else root / value
    target = target.resolve(strict=True)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise TtsKServeRolloutError("TTS rollout path escaped repository root") from exc
    if not target.is_file():
        raise TtsKServeRolloutError("TTS rollout path is not a file")
    return target


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TtsKServeRolloutError("TTS rollout JSON is invalid") from exc
    if not isinstance(value, dict):
        raise TtsKServeRolloutError("TTS rollout JSON root must be an object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
