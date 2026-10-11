"""Immutable evidence contract for an executed ASR ModelRelease rollout."""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.simulation.asr_enterprise_value_lab import (
    verify_asr_enterprise_value,
)
from industrial_ops_agent.simulation.asr_kserve_rollout import (
    OUTPUT_RELATIVE,
    PROBE_AUDIO_SHA256,
    PROBE_CASE_ID,
    PROBE_SOURCE_ID,
    PROBE_TRANSCRIPT,
    PROBE_TRANSCRIPT_SHA256,
    PROXY_SOURCE_RELATIVE,
    WORKER_SOURCE_RELATIVE,
    inference_service_document,
    proxy_config_map_document,
    release_probe_document,
    route_document,
    verify_asr_kserve_readiness,
)

SCHEMA_VERSION: Literal["enterprise-asr-kserve-rollout-acceptance/v1"] = (
    "enterprise-asr-kserve-rollout-acceptance/v1"
)
STATUS: Literal["ASR_MODEL_RELEASE_KSERVE_SHADOW_CANARY_ROLLBACK_PASSED"] = (
    "ASR_MODEL_RELEASE_KSERVE_SHADOW_CANARY_ROLLBACK_PASSED"
)
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = (
    "LOCAL_STAGING_PROJECT_AUTHORIZED"
)
ACTUAL_OUTPUT_RELATIVE = OUTPUT_RELATIVE / "actual"
TRAFFIC_PROBE_SCHEMA = "enterprise-asr-routing-probe/v1"
TRAFFIC_PROBE_SAMPLE_RATE = 16_000
TRAFFIC_PROBE_SAMPLE_COUNT = 1_600
TRAFFIC_PROBE_DURATION_SECONDS = 0.1


class AsrKServeAcceptanceError(RuntimeError):
    """Executed ASR rollout evidence is missing, changed, or inconsistent."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileBinding(_ClosedModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AsrAcceptanceSource(_ClosedModel):
    readiness: FileBinding
    readiness_evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_acceptance: FileBinding
    training_run_id: str = Field(pattern=r"^asr-[0-9a-f]{20}$")
    training_evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: Literal["openai/whisper-tiny"]
    model_revision: Literal["c4f62d5fce5d73978a1dbfcac01926312acd1a51"]
    adapter_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AsrAcceptanceProbe(_ClosedModel):
    quality_manifest: FileBinding
    quality_audio: FileBinding
    quality_case_id: Literal["asr-release-probe-1272-135031-0006"]
    quality_source_id: Literal["1272-135031-0006"]
    expected_transcript_sha256: Literal[
        "cec64d4fba4eb13f5b1800fb208fd024f31c3194be80dd09537eb0fd8e5c69c5"
    ]
    routing_manifest: FileBinding
    routing_audio: FileBinding
    routing_parent_audio_sha256: Literal[
        "65ae3678b861333fb9ba2e5d7db9a0f77fe58ad89d4480872eeb62d22e894dd3"
    ]
    routing_sample_rate: Literal[16000]
    routing_sample_count: Literal[1600]
    routing_duration_seconds: float = Field(ge=0.1, le=0.1)
    routing_probe_is_quality_evidence: Literal[False] = False
    formal_gold_reused: Literal[False] = False


class AsrModelReleaseEvidence(_ClosedModel):
    state_database: FileBinding
    sqlite_integrity_check: Literal["ok"]
    tenant_id: Literal["project-enterprise-staging"]
    baseline_release_id: str = Field(pattern=r"^model-release-[0-9a-f]{32}$")
    release_id: str = Field(pattern=r"^model-release-[0-9a-f]{32}$")
    manifest: FileBinding
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_candidate_experiment_id: str = Field(pattern=r"^asr-[0-9a-f]{20}$")
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


class AsrGpuRuntimeEvidence(_ClosedModel):
    topology: Literal["DOCKER_GPU_WORKER_BEHIND_KSERVE_PROXY"]
    actual_gpu_execution: Literal[True] = True
    model_execution_simulated: Literal[False] = False
    worker_image: Literal[
        "industrial-ops/gpu-promotion-runtime:ac9f90725e26-4844463141b6-8567fe9114e7"
    ]
    worker_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    model_id: Literal["openai/whisper-tiny"]
    model_revision: Literal["c4f62d5fce5d73978a1dbfcac01926312acd1a51"]
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


class AsrKServeEvidence(_ClosedModel):
    cluster_context: Literal["kind-ioap-gpu-promotion-lab"]
    namespace: Literal["ioap-gpu-promotion-lab"]
    gateway_name: Literal["ioap-gpu-lab-gateway"]
    hostname: Literal["gpu-lab.local"]
    route_name: Literal["ioap-asr-lab-route"]
    route_path_prefix: Literal["/v1/asr"]
    stable_inference_service: Literal["ioap-asr-stable"]
    candidate_inference_service: Literal["ioap-asr-candidate"]
    stable_ready: Literal[True] = True
    candidate_ready: Literal[True] = True
    route_accepted: Literal[True] = True
    route_resolved_refs: Literal[True] = True
    worker_host_gateway: str = Field(min_length=1)
    worker_host_port: int = Field(ge=1, le=65535)
    proxy_image: Literal[
        "industrial-ops/gpu-promotion-runtime:ac9f90725e26-4844463141b6-8567fe9114e7"
    ]
    proxy_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    proxy_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_map: FileBinding
    stable_service: FileBinding
    candidate_service: FileBinding
    routes: dict[str, FileBinding]
    observed_final_route: FileBinding
    observed_stable_service: FileBinding
    observed_candidate_service: FileBinding
    hardened_container_security_context: Literal[True] = True
    endpoint_scope: Literal["LOCAL_PORT_FORWARD"]


class AsrReleaseQualityEvidence(_ClosedModel):
    case_id: Literal["asr-release-probe-1272-135031-0006"]
    audio_sha256: Literal[
        "65ae3678b861333fb9ba2e5d7db9a0f77fe58ad89d4480872eeb62d22e894dd3"
    ]
    stable_normalized_transcript: str = Field(min_length=1)
    candidate_normalized_transcript: str = Field(min_length=1)
    stable_normalized_transcript_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_normalized_transcript_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stable_word_error_rate: float = Field(ge=0.0)
    candidate_word_error_rate: float = Field(ge=0.0)
    candidate_word_error_rate_max: float = Field(ge=0.75, le=0.75)
    candidate_regression_tolerance: float = Field(ge=0.25, le=0.25)
    candidate_response_contract_valid: Literal[True] = True
    candidate_audio_digest_verified: Literal[True] = True
    candidate_release_probe_passed: Literal[True] = True


class AsrRolloutStageEvidence(_ClosedModel):
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


class AsrRolloutCleanupEvidence(_ClosedModel):
    gpu_worker_stopped: Literal[True] = True
    port_forwards_stopped: Literal[True] = True
    asr_kubernetes_resources_removed: Literal[True] = True
    cluster_stopped_after_acceptance: Literal[True] = True
    evidence_is_historical_after_cleanup: Literal[True] = True


class EnterpriseAsrKServeAcceptanceReport(_ClosedModel):
    schema_version: Literal["enterprise-asr-kserve-rollout-acceptance/v1"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal["ASR_MODEL_RELEASE_KSERVE_SHADOW_CANARY_ROLLBACK_PASSED"] = STATUS
    source: AsrAcceptanceSource
    probe: AsrAcceptanceProbe
    model_release: AsrModelReleaseEvidence
    runtime: AsrGpuRuntimeEvidence
    kserve: AsrKServeEvidence
    quality: AsrReleaseQualityEvidence
    stages: tuple[AsrRolloutStageEvidence, ...] = Field(min_length=4, max_length=4)
    cleanup: AsrRolloutCleanupEvidence
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def finalize_acceptance(unsigned: dict[str, Any]) -> EnterpriseAsrKServeAcceptanceReport:
    draft = EnterpriseAsrKServeAcceptanceReport.model_validate(
        {**unsigned, "evidence_chain_sha256": "0" * 64}
    )
    payload = draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    return draft.model_copy(update={"evidence_chain_sha256": document_sha256(payload)})


def write_acceptance(
    report: EnterpriseAsrKServeAcceptanceReport,
    path: Path,
) -> None:
    _write_json(path, report.model_dump(mode="json"))


def verify_asr_kserve_acceptance(
    repo_root: Path,
    acceptance_path: Path = ACTUAL_OUTPUT_RELATIVE / "acceptance.json",
) -> EnterpriseAsrKServeAcceptanceReport:
    root = repo_root.resolve(strict=True)
    path = _inside_file(root, acceptance_path)
    report = EnterpriseAsrKServeAcceptanceReport.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != document_sha256(unsigned):
        raise AsrKServeAcceptanceError("asr_kserve_acceptance_chain_changed")
    readiness_path = _verify_binding(root, report.source.readiness)
    readiness = verify_asr_kserve_readiness(root, readiness_path)
    training_path = _verify_binding(root, report.source.training_acceptance)
    training = verify_asr_enterprise_value(root, training_path)
    if (
        readiness.evidence_chain_sha256 != report.source.readiness_evidence_chain_sha256
        or training.run_id != report.source.training_run_id
        or training.evidence_chain_sha256
        != report.source.training_evidence_chain_sha256
        or training.base_model.model_id != report.source.model_id
        or training.base_model.revision != report.source.model_revision
        or training.adapter.bundle_sha256 != report.source.adapter_bundle_sha256
        or readiness.source.run_id != training.run_id
    ):
        raise AsrKServeAcceptanceError("asr_kserve_acceptance_source_changed")
    _verify_probes(root, report)
    state_path = _verify_binding(root, report.model_release.state_database)
    manifest_path = _verify_binding(root, report.model_release.manifest)
    manifest = _load_object(manifest_path)
    if document_sha256(manifest) != report.model_release.manifest_hash:
        raise AsrKServeAcceptanceError("asr_model_release_manifest_changed")
    specialized_asr = manifest.get("specialized_components", {}).get("asr", {})
    asr_training = specialized_asr.get("training", {})
    asr_import = specialized_asr.get("project_authorized_import", {})
    asr_evaluation = specialized_asr.get("evaluation", {})
    if (
        specialized_asr.get("runtime_model_id")
        != report.model_release.imported_candidate_experiment_id
        or asr_training.get("experiment_id")
        != report.model_release.imported_candidate_experiment_id
        or asr_import.get("source_candidate_experiment_id")
        != report.model_release.source_candidate_experiment_id
        or asr_evaluation.get("evaluation_id") != report.model_release.evaluation_id
    ):
        raise AsrKServeAcceptanceError("asr_specialized_release_binding_changed")
    _verify_release_database(state_path, report, manifest)
    _verify_runtime_and_kserve(root, report)
    _verify_stages(report)
    if (
        report.quality.candidate_word_error_rate
        > report.quality.candidate_word_error_rate_max
        or report.quality.candidate_word_error_rate
        > report.quality.stable_word_error_rate
        + report.quality.candidate_regression_tolerance
    ):
        raise AsrKServeAcceptanceError("asr_release_probe_quality_gate_changed")
    stable_wer = _word_error_rate(
        PROBE_TRANSCRIPT,
        report.quality.stable_normalized_transcript,
    )
    candidate_wer = _word_error_rate(
        PROBE_TRANSCRIPT,
        report.quality.candidate_normalized_transcript,
    )
    if (
        document_sha256(report.quality.stable_normalized_transcript)
        != report.quality.stable_normalized_transcript_sha256
        or document_sha256(report.quality.candidate_normalized_transcript)
        != report.quality.candidate_normalized_transcript_sha256
        or abs(stable_wer - report.quality.stable_word_error_rate) > 1e-12
        or abs(candidate_wer - report.quality.candidate_word_error_rate) > 1e-12
    ):
        raise AsrKServeAcceptanceError("asr_release_probe_transcript_evidence_changed")
    return report


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
            raise AsrKServeAcceptanceError(
                "ASR acceptance file escaped repository root"
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


def _verify_probes(
    root: Path,
    report: EnterpriseAsrKServeAcceptanceReport,
) -> None:
    quality_manifest = _load_object(_verify_binding(root, report.probe.quality_manifest))
    quality_audio = _verify_binding(root, report.probe.quality_audio)
    if (
        quality_manifest != release_probe_document()
        or file_sha256(quality_audio) != PROBE_AUDIO_SHA256
        or report.probe.quality_case_id != PROBE_CASE_ID
        or report.probe.quality_source_id != PROBE_SOURCE_ID
        or report.probe.expected_transcript_sha256 != PROBE_TRANSCRIPT_SHA256
    ):
        raise AsrKServeAcceptanceError("asr_acceptance_quality_probe_changed")
    routing_manifest = _load_object(_verify_binding(root, report.probe.routing_manifest))
    routing_audio = _verify_binding(root, report.probe.routing_audio)
    media = routing_manifest.get("media", {})
    if (
        routing_manifest.get("schema_version") != TRAFFIC_PROBE_SCHEMA
        or routing_manifest.get("purpose") != "ROUTING_ONLY"
        or routing_manifest.get("quality_evidence") is not False
        or routing_manifest.get("formal_gold_reused") is not False
        or routing_manifest.get("parent_audio_sha256") != PROBE_AUDIO_SHA256
        or media.get("path") != report.probe.routing_audio.path
        or media.get("sha256") != file_sha256(routing_audio)
        or media.get("size_bytes") != routing_audio.stat().st_size
        or media.get("sampling_rate") != TRAFFIC_PROBE_SAMPLE_RATE
        or media.get("sample_count") != TRAFFIC_PROBE_SAMPLE_COUNT
        or media.get("duration_seconds") != TRAFFIC_PROBE_DURATION_SECONDS
        or routing_audio.stat().st_size > 16_384
    ):
        raise AsrKServeAcceptanceError("asr_acceptance_routing_probe_changed")
    routing_unsigned = {
        key: value
        for key, value in routing_manifest.items()
        if key != "evidence_chain_sha256"
    }
    if routing_manifest.get("evidence_chain_sha256") != document_sha256(
        routing_unsigned
    ):
        raise AsrKServeAcceptanceError("asr_acceptance_routing_probe_chain_changed")


def _verify_runtime_and_kserve(
    root: Path,
    report: EnterpriseAsrKServeAcceptanceReport,
) -> None:
    if (
        file_sha256(_inside_file(root, PROXY_SOURCE_RELATIVE))
        != report.kserve.proxy_source_sha256
        or file_sha256(_inside_file(root, WORKER_SOURCE_RELATIVE))
        != report.kserve.worker_source_sha256
    ):
        raise AsrKServeAcceptanceError("asr_acceptance_runtime_source_changed")
    proxy_source = _inside_file(root, PROXY_SOURCE_RELATIVE).read_text(encoding="utf-8")
    config_map = _load_object(_verify_binding(root, report.kserve.config_map))
    stable = _load_object(_verify_binding(root, report.kserve.stable_service))
    candidate = _load_object(_verify_binding(root, report.kserve.candidate_service))
    if (
        config_map != proxy_config_map_document(proxy_source)
        or stable
        != inference_service_document(
            variant="stable",
            host_gateway=report.kserve.worker_host_gateway,
            host_port=report.kserve.worker_host_port,
            proxy_image_digest=report.kserve.proxy_image_digest,
        )
        or candidate
        != inference_service_document(
            variant="candidate",
            host_gateway=report.kserve.worker_host_gateway,
            host_port=report.kserve.worker_host_port,
            proxy_image_digest=report.kserve.proxy_image_digest,
        )
    ):
        raise AsrKServeAcceptanceError("asr_acceptance_kserve_contract_changed")
    if set(report.kserve.routes) != {"SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"}:
        raise AsrKServeAcceptanceError("asr_acceptance_route_set_changed")
    for stage, binding in report.kserve.routes.items():
        if _load_object(_verify_binding(root, binding)) != route_document(stage):  # type: ignore[arg-type]
            raise AsrKServeAcceptanceError("asr_acceptance_route_document_changed")
    final_route = _load_object(_verify_binding(root, report.kserve.observed_final_route))
    stable_state = _load_object(
        _verify_binding(root, report.kserve.observed_stable_service)
    )
    candidate_state = _load_object(
        _verify_binding(root, report.kserve.observed_candidate_service)
    )
    if (
        not route_condition_is_true(final_route, "Accepted")
        or not route_condition_is_true(final_route, "ResolvedRefs")
        or not route_rules_semantically_equal(final_route, route_document("ROLLED_BACK"))
        or not _resource_ready(stable_state)
        or not _resource_ready(candidate_state)
    ):
        raise AsrKServeAcceptanceError("asr_acceptance_observed_kserve_state_changed")


def _verify_stages(report: EnterpriseAsrKServeAcceptanceReport) -> None:
    stages = {item.stage: item for item in report.stages}
    if set(stages) != {"SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"}:
        raise AsrKServeAcceptanceError("asr_acceptance_stage_evidence_is_incomplete")
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
        or stages["ROLLED_BACK"].candidate_request_delta != 0
        or stages["ROLLED_BACK"].candidate_response_count != 0
        or stages["ROLLED_BACK"].stable_response_count < 1
    ):
        raise AsrKServeAcceptanceError("asr_acceptance_traffic_evidence_is_invalid")


def _verify_release_database(
    path: Path,
    report: EnterpriseAsrKServeAcceptanceReport,
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
    if integrity != ("ok",) or release.sqlite_integrity_check != "ok":
        raise AsrKServeAcceptanceError("asr_acceptance_database_integrity_failed")
    release_candidate_experiment_id = manifest.get("training", {}).get("experiment_id")
    if not isinstance(release_candidate_experiment_id, str):
        raise AsrKServeAcceptanceError("asr_baseline_release_binding_changed")
    if release_row != (
        "ROLLED_BACK",
        release.manifest_hash,
        0.0,
        release_candidate_experiment_id,
    ):
        raise AsrKServeAcceptanceError("asr_acceptance_release_state_changed")
    if deployment_row != (
        release.deployment_id,
        "KSERVE_GATEWAY_API",
        "READY",
        "ROLLED_BACK",
        0.0,
    ):
        raise AsrKServeAcceptanceError("asr_acceptance_deployment_state_changed")
    if approval_row != (
        release.approval_id,
        "APPROVED",
        release.approval_requested_by_subject_id,
        release.approval_decided_by_subject_id,
    ):
        raise AsrKServeAcceptanceError("asr_acceptance_approval_state_changed")
    if transitions != release.transition_targets or observations != release.observation_stages:
        raise AsrKServeAcceptanceError("asr_acceptance_history_changed")


def _verify_binding(root: Path, binding: FileBinding) -> Path:
    path = _inside_file(root, Path(binding.path))
    if path.stat().st_size != binding.size_bytes or file_sha256(path) != binding.sha256:
        raise AsrKServeAcceptanceError("asr_acceptance_file_binding_changed")
    return path


def _inside_file(root: Path, value: Path) -> Path:
    target = value if value.is_absolute() else root / value
    target = target.resolve(strict=True)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise AsrKServeAcceptanceError(
            "ASR acceptance path escaped repository root"
        ) from exc
    if not target.is_file():
        raise AsrKServeAcceptanceError("ASR acceptance path is not a file")
    return target


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AsrKServeAcceptanceError("ASR acceptance JSON is invalid") from exc
    if not isinstance(value, dict):
        raise AsrKServeAcceptanceError("ASR acceptance JSON root must be an object")
    return value


def _resource_ready(document: dict[str, Any]) -> bool:
    conditions = document.get("status", {}).get("conditions", [])
    return any(
        isinstance(item, dict)
        and item.get("type") == "Ready"
        and item.get("status") == "True"
        for item in conditions
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


def _normalized_route_rules(document: dict[str, Any]) -> object | None:
    rules = document.get("spec", {}).get("rules")
    return _without_gateway_api_defaults(rules)


def _without_gateway_api_defaults(value: object) -> object:
    if isinstance(value, list):
        return [_without_gateway_api_defaults(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized: dict[str, object] = {}
    for key, item in value.items():
        if key in {"group", "kind"} and item in {"", "Core", "Service"}:
            continue
        normalized[key] = _without_gateway_api_defaults(item)
    return normalized


def _write_json(path: Path, value: object) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
