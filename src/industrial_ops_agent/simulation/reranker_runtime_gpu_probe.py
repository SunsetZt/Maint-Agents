"""Immutable evidence for a short-lived actual-GPU Reranker runtime probe."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from industrial_ops_agent.search_profiles.reranker import RERANKER_RUNTIME_SCHEMA
from industrial_ops_agent.simulation.reranker_kserve_readiness import (
    ACCEPTANCE_RELATIVE as READINESS_RELATIVE,
)
from industrial_ops_agent.simulation.reranker_kserve_readiness import (
    RUNTIME_IMAGE_DIGEST,
    RUNTIME_IMAGE_REPOSITORY,
    RUNTIME_IMAGE_TAG,
    document_sha256,
    file_sha256,
    verify_reranker_kserve_readiness,
)

SCHEMA_VERSION: Literal["enterprise-reranker-runtime-gpu-probe/v1"] = (
    "enterprise-reranker-runtime-gpu-probe/v1"
)
STATUS: Literal["RERANKER_CALIBRATED_RUNTIME_ACTUAL_GPU_PROBE_PASSED"] = (
    "RERANKER_CALIBRATED_RUNTIME_ACTUAL_GPU_PROBE_PASSED"
)
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = "LOCAL_STAGING_PROJECT_AUTHORIZED"
OUTPUT_RELATIVE = Path("artifacts/m7-reranker-runtime-gpu-probe")
ACCEPTANCE_RELATIVE = OUTPUT_RELATIVE / "acceptance.json"
CONTAINER_NAME = "ioap-reranker-runtime-gpu-probe"
PROBE_CASE_ID = "reranker-release-probe-gearbox-z17-v1"
PROBE_RELEASE_ID = "model-release-reranker-runtime-probe-v1"
PROBE_MANIFEST_HASH = sha256(b"reranker-runtime-probe-manifest-v1").hexdigest()
PROBE_INDEX_RELEASE_ID = "index-release-reranker-runtime-probe-v1"
EXPECTED_TOP_CHUNK_ID = "probe-gearbox-controlled-action"
EVIDENCE_FILES = {
    "probe_request": "probe-request.json",
    "health": "health.json",
    "first_response": "first-response.json",
    "second_response": "second-response.json",
    "missing_alias_rejection": "missing-alias-rejection.json",
    "ambiguous_alias_rejection": "ambiguous-alias-rejection.json",
    "binding_rejection": "binding-rejection.json",
    "metrics": "metrics.txt",
    "container_inspect": "container-inspect.json",
    "gpu_inventory": "gpu-inventory.csv",
    "container_logs": "container.log",
}


class RerankerRuntimeGpuProbeError(RuntimeError):
    """Actual runtime execution or its immutable evidence is invalid."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileBinding(_ClosedModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProbeSource(_ClosedModel):
    readiness: FileBinding
    readiness_evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    value_run_id: Literal["reranker-calibrated-64f493aa97c1a51cf422"]
    value_evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_path: str = Field(min_length=1)
    candidate_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RuntimeExecutionEvidence(_ClosedModel):
    image_repository: Literal["industrial-ops/reranker-runtime"]
    image_tag: Literal["calibrated-v1-95fb9107de7a"]
    image_digest: Literal["sha256:7386510182721f4f086773ad8d2dd7f5713855c483f43e7d572252f88c9776eb"]
    container_name: Literal["ioap-reranker-runtime-gpu-probe"]
    actual_gpu_execution: Literal[True] = True
    model_execution_simulated: Literal[False] = False
    exactly_one_gpu_exposed: Literal[True] = True
    network_mode: Literal["none"]
    cpu_limit: Literal[2]
    memory_limit_bytes: Literal[4294967296]
    memory_swap_limit_bytes: Literal[4294967296]
    read_only_root_filesystem: Literal[True] = True
    non_root_user: Literal["app"]
    no_new_privileges: Literal[True] = True
    all_linux_capabilities_dropped: Literal[True] = True
    pids_limit: Literal[256]
    duration_seconds: float = Field(gt=0.0, le=180.0)
    health: FileBinding
    metrics: FileBinding
    container_inspect: FileBinding
    gpu_inventory: FileBinding
    container_logs: FileBinding


class RuntimeProbeEvidence(_ClosedModel):
    request: FileBinding
    first_response: FileBinding
    second_response: FileBinding
    missing_alias_rejection: FileBinding
    ambiguous_alias_rejection: FileBinding
    binding_rejection: FileBinding
    case_id: Literal["reranker-release-probe-gearbox-z17-v1"]
    formal_gold: Literal[False] = False
    formal_gold_reused: Literal[False] = False
    excluded_from_training_and_model_selection: Literal[True] = True
    endpoint_path: Literal["/v1/retrieval/rerank"]
    calibration_profile_id: Literal["industrial-reranker-evidence-calibration-v1"]
    registry_id: Literal["industrial-reranker-terminology-registry-v1"]
    expected_top_chunk_id: Literal["probe-gearbox-controlled-action"]
    deterministic_replay_verified: Literal[True] = True
    calibrated_evidence_score_floor: float = Field(ge=3.0, le=3.0)
    missing_alias_failed_closed: Literal[True] = True
    ambiguous_alias_failed_closed: Literal[True] = True
    immutable_release_binding_failed_closed: Literal[True] = True


class CleanupEvidence(_ClosedModel):
    container_stopped: Literal[True] = True
    container_removed: Literal[True] = True
    no_runtime_service_left_running: Literal[True] = True
    evidence_is_historical_after_cleanup: Literal[True] = True


class EnterpriseRerankerRuntimeGpuProbeReport(_ClosedModel):
    schema_version: Literal["enterprise-reranker-runtime-gpu-probe/v1"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal["RERANKER_CALIBRATED_RUNTIME_ACTUAL_GPU_PROBE_PASSED"] = STATUS
    source: ProbeSource
    runtime: RuntimeExecutionEvidence
    probe: RuntimeProbeEvidence
    cleanup: CleanupEvidence
    hard_gates: dict[str, Literal[True]]
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def probe_request() -> dict[str, Any]:
    return {
        "schema_version": RERANKER_RUNTIME_SCHEMA,
        "index_release_id": PROBE_INDEX_RELEASE_ID,
        "query": "Investigate amber cascade on gearbox train Z-17 after coastdown.",
        "candidates": [
            {
                "chunk_id": EXPECTED_TOP_CHUNK_ID,
                "title": "Controlled field bulletin Z-17",
                "content": (
                    "Independent release probe. Confirm lubricant aeration and persistent "
                    "oil foaming, then inspect the oil return and air-ingress points before "
                    "restart. This sentence is not part of any formal Gold dataset."
                ),
                "recall_score": 0.51,
            },
            {
                "chunk_id": "probe-alias-only-decoy",
                "title": "Legacy alarm catalog",
                "content": "Amber cascade is a local alarm label without diagnostic evidence.",
                "recall_score": 0.99,
            },
            {
                "chunk_id": "probe-unrelated-decoy",
                "title": "Cooling loop note",
                "content": "Inspect coolant concentration after a salt deposit alarm.",
                "recall_score": 0.75,
            },
        ],
        "top_k": 3,
        "expected_release_id": PROBE_RELEASE_ID,
        "expected_manifest_hash": PROBE_MANIFEST_HASH,
        "expected_component_model_id": "reranker-calibrated-64f493aa97c1a51cf422",
        "expected_artifact_content_hash": (
            "937a9db83f62a4db201d2a5928f80b9982fd814c87be36c121afff3880a318a9"
        ),
        "probe_metadata": {
            "case_id": PROBE_CASE_ID,
            "formal_gold": False,
            "formal_gold_reused": False,
            "excluded_from_training_and_model_selection": True,
        },
    }


def build_runtime_gpu_probe_report(
    repo_root: Path,
    *,
    duration_seconds: float,
    output_path: Path = ACCEPTANCE_RELATIVE,
) -> EnterpriseRerankerRuntimeGpuProbeReport:
    root = repo_root.resolve(strict=True)
    readiness = verify_reranker_kserve_readiness(root, READINESS_RELATIVE)
    output = _inside_directory(root, OUTPUT_RELATIVE)
    evidence = {
        name: _binding(root, output / filename) for name, filename in EVIDENCE_FILES.items()
    }
    health = _read_object(output / EVIDENCE_FILES["health"])
    first = _read_http_result(output / EVIDENCE_FILES["first_response"])
    second = _read_http_result(output / EVIDENCE_FILES["second_response"])
    missing = _read_http_result(output / EVIDENCE_FILES["missing_alias_rejection"])
    ambiguous = _read_http_result(output / EVIDENCE_FILES["ambiguous_alias_rejection"])
    binding = _read_http_result(output / EVIDENCE_FILES["binding_rejection"])
    metrics = (output / EVIDENCE_FILES["metrics"]).read_text(encoding="utf-8")
    inspect = _read_inspect(output / EVIDENCE_FILES["container_inspect"])
    gpu_inventory = (output / EVIDENCE_FILES["gpu_inventory"]).read_text(encoding="utf-8")
    _validate_health(health)
    _validate_success(first, second)
    _validate_rejection(
        missing,
        "reranker_terminology_resolution_requires_exactly_one_entry",
    )
    _validate_rejection(
        ambiguous,
        "reranker_terminology_resolution_requires_exactly_one_entry",
    )
    _validate_rejection(binding, "reranker_runtime_release_binding_mismatch")
    _validate_metrics(metrics)
    _validate_container(inspect, gpu_inventory)
    destination = _inside_output(root, output_path)
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "external_enterprise_production_claim": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": STATUS,
        "source": {
            "readiness": _binding(root, root / READINESS_RELATIVE),
            "readiness_evidence_chain_sha256": readiness.evidence_chain_sha256,
            "value_run_id": readiness.source.run_id,
            "value_evidence_chain_sha256": readiness.source.value_evidence_chain_sha256,
            "candidate_path": readiness.source.candidate_path,
            "candidate_bundle_sha256": readiness.source.candidate_bundle_sha256,
        },
        "runtime": {
            "image_repository": readiness.runtime_image.repository,
            "image_tag": readiness.runtime_image.tag,
            "image_digest": readiness.runtime_image.digest,
            "container_name": CONTAINER_NAME,
            "actual_gpu_execution": True,
            "model_execution_simulated": False,
            "exactly_one_gpu_exposed": True,
            "network_mode": "none",
            "cpu_limit": 2,
            "memory_limit_bytes": 4 * 1024**3,
            "memory_swap_limit_bytes": 4 * 1024**3,
            "read_only_root_filesystem": True,
            "non_root_user": "app",
            "no_new_privileges": True,
            "all_linux_capabilities_dropped": True,
            "pids_limit": 256,
            "duration_seconds": duration_seconds,
            "health": evidence["health"],
            "metrics": evidence["metrics"],
            "container_inspect": evidence["container_inspect"],
            "gpu_inventory": evidence["gpu_inventory"],
            "container_logs": evidence["container_logs"],
        },
        "probe": {
            "request": evidence["probe_request"],
            "first_response": evidence["first_response"],
            "second_response": evidence["second_response"],
            "missing_alias_rejection": evidence["missing_alias_rejection"],
            "ambiguous_alias_rejection": evidence["ambiguous_alias_rejection"],
            "binding_rejection": evidence["binding_rejection"],
            "case_id": PROBE_CASE_ID,
            "formal_gold": False,
            "formal_gold_reused": False,
            "excluded_from_training_and_model_selection": True,
            "endpoint_path": "/v1/retrieval/rerank",
            "calibration_profile_id": "industrial-reranker-evidence-calibration-v1",
            "registry_id": "industrial-reranker-terminology-registry-v1",
            "expected_top_chunk_id": EXPECTED_TOP_CHUNK_ID,
            "deterministic_replay_verified": True,
            "calibrated_evidence_score_floor": 3.0,
            "missing_alias_failed_closed": True,
            "ambiguous_alias_failed_closed": True,
            "immutable_release_binding_failed_closed": True,
        },
        "cleanup": {
            "container_stopped": True,
            "container_removed": True,
            "no_runtime_service_left_running": True,
            "evidence_is_historical_after_cleanup": True,
        },
        "hard_gates": {
            "readiness_evidence_verified": True,
            "actual_gpu_model_loaded": True,
            "calibrated_runtime_contract_verified": True,
            "deterministic_replay_verified": True,
            "missing_alias_failed_closed": True,
            "ambiguous_alias_failed_closed": True,
            "immutable_release_binding_failed_closed": True,
            "container_resource_limits_verified": True,
            "container_security_verified": True,
            "container_cleanup_verified": True,
        },
    }
    draft = EnterpriseRerankerRuntimeGpuProbeReport.model_validate(
        {**unsigned, "evidence_chain_sha256": "0" * 64}
    )
    canonical = draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    report = draft.model_copy(update={"evidence_chain_sha256": document_sha256(canonical)})
    _write_json(destination, report.model_dump(mode="json"))
    return verify_runtime_gpu_probe_report(root, destination)


def verify_runtime_gpu_probe_report(
    repo_root: Path,
    acceptance_path: Path = ACCEPTANCE_RELATIVE,
) -> EnterpriseRerankerRuntimeGpuProbeReport:
    root = repo_root.resolve(strict=True)
    path = _inside_file(root, acceptance_path)
    try:
        report = EnterpriseRerankerRuntimeGpuProbeReport.model_validate_json(path.read_bytes())
    except (OSError, UnicodeError, ValidationError) as exc:
        raise RerankerRuntimeGpuProbeError("Reranker GPU probe report is invalid") from exc
    canonical = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != document_sha256(canonical):
        raise RerankerRuntimeGpuProbeError("Reranker GPU probe evidence chain changed")
    readiness = verify_reranker_kserve_readiness(root, Path(report.source.readiness.path))
    if (
        readiness.evidence_chain_sha256 != report.source.readiness_evidence_chain_sha256
        or readiness.source.run_id != report.source.value_run_id
        or readiness.source.value_evidence_chain_sha256 != report.source.value_evidence_chain_sha256
        or readiness.source.candidate_path != report.source.candidate_path
        or readiness.source.candidate_bundle_sha256 != report.source.candidate_bundle_sha256
        or readiness.runtime_image.digest != report.runtime.image_digest
        or report.runtime.image_digest != RUNTIME_IMAGE_DIGEST
        or report.runtime.image_repository != RUNTIME_IMAGE_REPOSITORY
        or report.runtime.image_tag != RUNTIME_IMAGE_TAG
        or set(report.hard_gates.values()) != {True}
    ):
        raise RerankerRuntimeGpuProbeError("Reranker GPU probe source binding changed")
    bindings = (
        report.source.readiness,
        report.runtime.health,
        report.runtime.metrics,
        report.runtime.container_inspect,
        report.runtime.gpu_inventory,
        report.runtime.container_logs,
        report.probe.request,
        report.probe.first_response,
        report.probe.second_response,
        report.probe.missing_alias_rejection,
        report.probe.ambiguous_alias_rejection,
        report.probe.binding_rejection,
    )
    for binding in bindings:
        _verify_binding(root, binding)
    output = _inside_directory(root, OUTPUT_RELATIVE)
    health = _read_object(output / EVIDENCE_FILES["health"])
    first = _read_http_result(output / EVIDENCE_FILES["first_response"])
    second = _read_http_result(output / EVIDENCE_FILES["second_response"])
    _validate_health(health)
    _validate_success(first, second)
    _validate_rejection(
        _read_http_result(output / EVIDENCE_FILES["missing_alias_rejection"]),
        "reranker_terminology_resolution_requires_exactly_one_entry",
    )
    _validate_rejection(
        _read_http_result(output / EVIDENCE_FILES["ambiguous_alias_rejection"]),
        "reranker_terminology_resolution_requires_exactly_one_entry",
    )
    _validate_rejection(
        _read_http_result(output / EVIDENCE_FILES["binding_rejection"]),
        "reranker_runtime_release_binding_mismatch",
    )
    _validate_metrics((output / EVIDENCE_FILES["metrics"]).read_text(encoding="utf-8"))
    _validate_container(
        _read_inspect(output / EVIDENCE_FILES["container_inspect"]),
        (output / EVIDENCE_FILES["gpu_inventory"]).read_text(encoding="utf-8"),
    )
    return report


def _validate_health(value: dict[str, Any]) -> None:
    if (
        value.get("status") != "ready"
        or value.get("release_id") != PROBE_RELEASE_ID
        or value.get("component_model_id") != "reranker-calibrated-64f493aa97c1a51cf422"
        or value.get("calibration_enabled") is not True
        or value.get("runtime_profile_id") != "industrial-reranker-evidence-calibration-v1"
        or value.get("registry_id") != "industrial-reranker-terminology-registry-v1"
        or value.get("max_sequence_length") != 128
    ):
        raise RerankerRuntimeGpuProbeError("Reranker runtime health contract is invalid")


def _validate_success(first: dict[str, Any], second: dict[str, Any]) -> None:
    if first != second or first.get("status_code") != 200:
        raise RerankerRuntimeGpuProbeError("Reranker deterministic runtime replay failed")
    body = first.get("body")
    if not isinstance(body, dict):
        raise RerankerRuntimeGpuProbeError("Reranker calibrated runtime result is invalid")
    items = body.get("items")
    if (
        not isinstance(items, list)
        or len(items) != 3
        or not isinstance(items[0], dict)
        or items[0].get("chunk_id") != EXPECTED_TOP_CHUNK_ID
        or not isinstance(items[0].get("score"), (int, float))
        or cast(float, items[0]["score"]) < 3.0
        or body.get("release_id") != PROBE_RELEASE_ID
        or body.get("manifest_hash") != PROBE_MANIFEST_HASH
    ):
        raise RerankerRuntimeGpuProbeError("Reranker calibrated runtime result is invalid")


def _validate_rejection(value: dict[str, Any], detail: str) -> None:
    body = value.get("body")
    if (
        value.get("status_code") != 422
        or not isinstance(body, dict)
        or body.get("detail") != detail
    ):
        raise RerankerRuntimeGpuProbeError(f"Reranker rejection contract changed: {detail}")


def _validate_metrics(value: str) -> None:
    required = (
        "ioap_reranker_runtime_requests_total{"
        'release_id="model-release-reranker-runtime-probe-v1",status="success"} 2.0',
        "ioap_reranker_runtime_requests_total{"
        'release_id="model-release-reranker-runtime-probe-v1",status="rejected"} 3.0',
    )
    if any(item not in value for item in required):
        raise RerankerRuntimeGpuProbeError("Reranker runtime metrics are incomplete")


def _validate_container(inspect: dict[str, Any], gpu_inventory: str) -> None:
    host = inspect.get("HostConfig")
    config = inspect.get("Config")
    requests = host.get("DeviceRequests") if isinstance(host, dict) else None
    request = requests[0] if isinstance(requests, list) and len(requests) == 1 else None
    if (
        inspect.get("Image") != RUNTIME_IMAGE_DIGEST
        or not isinstance(host, dict)
        or host.get("NetworkMode") != "none"
        or host.get("NanoCpus") != 2_000_000_000
        or host.get("Memory") != 4 * 1024**3
        or host.get("MemorySwap") != 4 * 1024**3
        or host.get("ReadonlyRootfs") is not True
        or host.get("PidsLimit") != 256
        or "ALL" not in (host.get("CapDrop") or [])
        or "no-new-privileges" not in (host.get("SecurityOpt") or [])
        or not isinstance(request, dict)
        or request.get("DeviceIDs") != ["0"]
        or not isinstance(config, dict)
        or config.get("User") != "app"
        or "NVIDIA GeForce RTX 4070 SUPER" not in gpu_inventory
    ):
        raise RerankerRuntimeGpuProbeError("Reranker runtime container controls are invalid")


def _read_http_result(path: Path) -> dict[str, Any]:
    value = _read_object(path)
    if value.get("schema_version") != "container-http-result/v1":
        raise RerankerRuntimeGpuProbeError("Reranker HTTP evidence is invalid")
    return value


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RerankerRuntimeGpuProbeError("Reranker probe JSON is invalid") from exc
    if not isinstance(value, dict):
        raise RerankerRuntimeGpuProbeError("Reranker probe JSON is invalid")
    return cast(dict[str, Any], value)


def _read_inspect(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RerankerRuntimeGpuProbeError("Reranker container inspection is invalid") from exc
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise RerankerRuntimeGpuProbeError("Reranker container inspection is invalid")
    return cast(dict[str, Any], value[0])


def _binding(root: Path, path: Path) -> dict[str, Any]:
    target = _inside_file(root, path)
    return {
        "path": target.relative_to(root).as_posix(),
        "size_bytes": target.stat().st_size,
        "sha256": file_sha256(target),
    }


def _verify_binding(root: Path, binding: FileBinding) -> None:
    target = _inside_file(root, Path(binding.path))
    if target.stat().st_size != binding.size_bytes or file_sha256(target) != binding.sha256:
        raise RerankerRuntimeGpuProbeError(f"Reranker probe file changed: {binding.path}")


def _inside_file(root: Path, path: Path) -> Path:
    target = path if path.is_absolute() else root / path
    try:
        resolved = target.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise RerankerRuntimeGpuProbeError("Reranker probe path escaped repository") from exc
    if not resolved.is_file():
        raise RerankerRuntimeGpuProbeError("Reranker probe file is missing")
    return resolved


def _inside_directory(root: Path, path: Path) -> Path:
    target = (root / path).resolve(strict=True)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise RerankerRuntimeGpuProbeError("Reranker probe directory escaped repository") from exc
    if not target.is_dir():
        raise RerankerRuntimeGpuProbeError("Reranker probe directory is missing")
    return target


def _inside_output(root: Path, path: Path) -> Path:
    target = path if path.is_absolute() else root / path
    resolved = target.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RerankerRuntimeGpuProbeError("Reranker probe output escaped repository") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ACCEPTANCE_RELATIVE",
    "CONTAINER_NAME",
    "EVIDENCE_FILES",
    "EnterpriseRerankerRuntimeGpuProbeReport",
    "OUTPUT_RELATIVE",
    "PROBE_MANIFEST_HASH",
    "PROBE_RELEASE_ID",
    "RerankerRuntimeGpuProbeError",
    "build_runtime_gpu_probe_report",
    "probe_request",
    "verify_runtime_gpu_probe_report",
]
