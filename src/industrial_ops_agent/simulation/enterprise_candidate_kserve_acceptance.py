"""Failure-closed evidence contract for the DPO/TTS/Embedding KServe rollout."""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from industrial_ops_agent.deployment.observed_kserve import ObservedKServeSnapshot
from industrial_ops_agent.simulation.enterprise_candidate_rollout import (
    SourceBinding,
    _verify_sources,
)

SCHEMA_VERSION: Literal["enterprise-candidate-kserve-acceptance/v1"] = (
    "enterprise-candidate-kserve-acceptance/v1"
)
COMPONENT_RECEIPT_SCHEMA_VERSION: Literal[
    "enterprise-candidate-kserve-component-receipt/v1"
] = "enterprise-candidate-kserve-component-receipt/v1"
STATUS = "ENTERPRISE_DPO_TTS_EMBEDDING_ACTUAL_GPU_KSERVE_ROLLOUT_PASSED"
CLASSIFICATION = "LOCAL_STAGING_PROJECT_AUTHORIZED"
OUTPUT_RELATIVE = Path("artifacts/m7-enterprise-candidate-kserve-acceptance")
ACCEPTANCE_RELATIVE = OUTPUT_RELATIVE / "acceptance.json"
STAGE_REQUEST_COUNTS = {
    "SHADOW": 500,
    "CANARY_5": 1_000,
    "CANARY_25": 5_000,
    "ROLLED_BACK": 64,
}
EXPECTED_COMPONENTS: tuple[
    Literal["LLM", "TTS", "EMBEDDING"],
    Literal["LLM", "TTS", "EMBEDDING"],
    Literal["LLM", "TTS", "EMBEDDING"],
] = ("LLM", "TTS", "EMBEDDING")


class EnterpriseCandidateKServeAcceptanceError(RuntimeError):
    """Actual rollout evidence is absent, mutable, or internally inconsistent."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileBinding(_ClosedModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RuntimeEvidence(_ClosedModel):
    runtime_source: FileBinding
    proxy_source: FileBinding
    worker_image: str = Field(min_length=1)
    worker_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    proxy_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    metrics: FileBinding
    worker_log: FileBinding
    actual_gpu_execution: Literal[True]
    actual_model_inference: Literal[True]
    model_inference_simulated: Literal[False]
    stable_actual_inference_count: int = Field(ge=1)
    candidate_actual_inference_count: int = Field(ge=1)
    stable_request_count: int = Field(ge=1)
    candidate_request_count: int = Field(ge=1)
    runtime_error_count: Literal[0]
    gpu_name: str = Field(min_length=1)
    gpu_total_memory_bytes: int = Field(gt=0)
    gpu_peak_memory_reserved_bytes: int = Field(gt=0)


class QualityEvidence(_ClosedModel):
    probe_case_id: str = Field(min_length=1)
    stable_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stable_actual_model_inference: Literal[True]
    candidate_actual_model_inference: Literal[True]
    candidate_quality_gate_passed: Literal[True]
    metrics: dict[str, float]


class ProviderObservationEvidence(_ClosedModel):
    component: Literal["LLM", "TTS", "EMBEDDING"]
    stage: Literal["SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"]
    ready: Literal[True]
    route_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_route_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stable_resource_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_resource_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_revision: str = Field(min_length=1)
    route_resource_version: str = Field(min_length=1)
    stable_resource_version: str = Field(min_length=1)
    candidate_resource_version: str = Field(min_length=1)
    reason_code: None

    @classmethod
    def from_snapshot(
        cls,
        value: ObservedKServeSnapshot,
    ) -> ProviderObservationEvidence:
        if not value.ready or value.reason_code is not None:
            raise EnterpriseCandidateKServeAcceptanceError(
                "observed_kserve_provider_snapshot_is_not_ready"
            )
        return cls(**asdict(value))


class StageTrafficEvidence(_ClosedModel):
    stage: Literal["SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"]
    passed: Literal[True]
    request_count: int = Field(gt=0)
    stable_request_delta: int = Field(ge=0)
    candidate_request_delta: int = Field(ge=0)
    stable_response_count: int = Field(ge=0)
    candidate_response_count: int = Field(ge=0)
    candidate_response_ratio: float = Field(ge=0.0, le=1.0)
    candidate_mirror_observed: bool
    p95_latency_ms: float = Field(gt=0.0)
    error_count: Literal[0]
    route_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256s: tuple[str, ...] = Field(min_length=1)
    model_release_observation_decision: Literal["PASS", "NOT_REQUIRED_AFTER_ROLLBACK"]


class ModelReleaseEvidence(_ClosedModel):
    release_id: str = Field(min_length=1)
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_id: str = Field(min_length=1)
    approval_requested_by_subject_id: str = Field(min_length=1)
    approval_decided_by_subject_id: str = Field(min_length=1)
    independent_approval: Literal[True]
    deployment_id: str = Field(min_length=1)
    transition_reason_codes: tuple[str, ...] = Field(min_length=4)
    observation_stages: tuple[
        Literal["SHADOW", "CANARY_5", "CANARY_25"], ...
    ] = Field(min_length=3, max_length=3)
    final_release_status: Literal["ROLLED_BACK"]
    final_deployment_status: Literal["READY"]
    final_deployment_stage: Literal["ROLLED_BACK"]
    final_traffic_percent: float = Field(ge=0.0, le=0.0)
    active_production_alias_created: Literal[False]


class KubernetesEvidence(_ClosedModel):
    namespace: str = Field(min_length=1)
    route_name: str = Field(min_length=1)
    stable_service_name: str = Field(min_length=1)
    candidate_service_name: str = Field(min_length=1)
    hostname: str = Field(min_length=1)
    endpoint_path: str = Field(min_length=1)
    proxy_config_map: FileBinding
    stable_inference_service: FileBinding
    candidate_inference_service: FileBinding
    routes: tuple[FileBinding, ...] = Field(min_length=4, max_length=4)
    actual_kserve_execution: Literal[True]
    route_accepted: Literal[True]
    route_refs_resolved: Literal[True]


class ComponentAcceptance(_ClosedModel):
    component: Literal["LLM", "TTS", "EMBEDDING"]
    method: Literal["DPO", "TTS", "EMBEDDING"]
    source: SourceBinding
    runtime: RuntimeEvidence
    quality: QualityEvidence
    kubernetes: KubernetesEvidence
    provider_observations: tuple[ProviderObservationEvidence, ...]
    stages: tuple[StageTrafficEvidence, ...]
    model_release: ModelReleaseEvidence
    resources_removed: Literal[True]
    worker_stopped: Literal[True]


class ComponentCleanupEvidence(_ClosedModel):
    kserve_resources_removed: Literal[True]
    gpu_worker_stopped: Literal[True]
    port_forwards_stopped: Literal[True]
    kind_cluster_stopped: Literal[True]
    release_database_disposed: Literal[True]


class ComponentAcceptanceReceipt(_ClosedModel):
    schema_version: Literal["enterprise-candidate-kserve-component-receipt/v1"]
    generated_at: str = Field(min_length=1)
    acceptance: ComponentAcceptance
    cleanup: ComponentCleanupEvidence
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CleanupEvidence(_ClosedModel):
    all_kserve_resources_removed: Literal[True]
    all_gpu_workers_stopped: Literal[True]
    all_port_forwards_stopped: Literal[True]
    kind_cluster_stopped: Literal[True]
    release_database_disposed: Literal[True]


class EnterpriseCandidateKServeAcceptanceReport(_ClosedModel):
    schema_version: Literal["enterprise-candidate-kserve-acceptance/v1"]
    status: Literal[
        "ENTERPRISE_DPO_TTS_EMBEDDING_ACTUAL_GPU_KSERVE_ROLLOUT_PASSED"
    ]
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"]
    generated_at: str = Field(min_length=1)
    provider_mode: Literal["OBSERVED_LOCAL_KSERVE_GATEWAY_API"]
    telemetry_mode: Literal["ACTUAL_HTTP_TRAFFIC_AND_GPU_RUNTIME_COUNTERS"]
    actual_kserve_execution: Literal[True]
    actual_model_inference: Literal[True]
    project_enterprise_truth: Literal[True]
    production_claim: Literal[False]
    external_enterprise_environment_claim: Literal[False]
    component_receipts: tuple[FileBinding, ...] = Field(min_length=3, max_length=3)
    components: tuple[ComponentAcceptance, ...]
    cleanup: CleanupEvidence
    hard_gates: dict[str, bool]
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def finalize_acceptance(
    unsigned: dict[str, Any],
) -> EnterpriseCandidateKServeAcceptanceReport:
    provisional = EnterpriseCandidateKServeAcceptanceReport(
        **unsigned,
        evidence_chain_sha256="0" * 64,
    )
    return provisional.model_copy(
        update={
            "evidence_chain_sha256": document_sha256(
                provisional.model_dump(mode="json", exclude={"evidence_chain_sha256"})
            )
        }
    )


def finalize_component_receipt(
    *,
    generated_at: str,
    acceptance: ComponentAcceptance,
    cleanup: ComponentCleanupEvidence,
) -> ComponentAcceptanceReceipt:
    provisional = ComponentAcceptanceReceipt(
        schema_version=COMPONENT_RECEIPT_SCHEMA_VERSION,
        generated_at=generated_at,
        acceptance=acceptance,
        cleanup=cleanup,
        evidence_chain_sha256="0" * 64,
    )
    return provisional.model_copy(
        update={
            "evidence_chain_sha256": document_sha256(
                provisional.model_dump(mode="json", exclude={"evidence_chain_sha256"})
            )
        }
    )


def write_component_receipt(report: ComponentAcceptanceReceipt, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def verify_component_receipt(
    repo_root: Path,
    path: Path,
    *,
    expected_component: Literal["LLM", "TTS", "EMBEDDING"] | None = None,
) -> ComponentAcceptanceReceipt:
    root = repo_root.resolve(strict=True)
    target = _inside_file(root, path)
    try:
        receipt = ComponentAcceptanceReceipt.model_validate_json(
            target.read_text(encoding="utf-8")
        )
    except (OSError, ValueError, ValidationError) as exc:
        raise EnterpriseCandidateKServeAcceptanceError(
            "enterprise_candidate_kserve_component_receipt_is_invalid"
        ) from exc
    calculated = document_sha256(
        receipt.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    )
    if calculated != receipt.evidence_chain_sha256:
        raise EnterpriseCandidateKServeAcceptanceError(
            "enterprise_candidate_kserve_component_receipt_chain_changed"
        )
    if expected_component is not None and receipt.acceptance.component != expected_component:
        raise EnterpriseCandidateKServeAcceptanceError(
            "enterprise_candidate_kserve_component_receipt_mismatch"
        )
    expected_sources = dict(_verified_source_bindings(str(root)))
    if receipt.acceptance.source != expected_sources[receipt.acceptance.component]:
        raise EnterpriseCandidateKServeAcceptanceError(
            "enterprise_candidate_kserve_source_binding_changed"
        )
    _verify_component(root, receipt.acceptance)
    return receipt


def write_acceptance(
    report: EnterpriseCandidateKServeAcceptanceReport,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def verify_enterprise_candidate_kserve_acceptance(
    repo_root: Path,
    report_path: Path = ACCEPTANCE_RELATIVE,
) -> EnterpriseCandidateKServeAcceptanceReport:
    root = repo_root.resolve(strict=True)
    path = _inside_file(root, report_path)
    try:
        report = EnterpriseCandidateKServeAcceptanceReport.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError, ValidationError) as exc:
        raise EnterpriseCandidateKServeAcceptanceError(
            "enterprise_candidate_kserve_acceptance_is_invalid"
        ) from exc
    calculated = document_sha256(
        report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    )
    if calculated != report.evidence_chain_sha256:
        raise EnterpriseCandidateKServeAcceptanceError(
            "enterprise_candidate_kserve_acceptance_chain_changed"
        )
    if tuple(item.component for item in report.components) != (
        "LLM",
        "TTS",
        "EMBEDDING",
    ):
        raise EnterpriseCandidateKServeAcceptanceError(
            "enterprise_candidate_kserve_component_set_changed"
        )
    if len(report.component_receipts) != 3:
        raise EnterpriseCandidateKServeAcceptanceError(
            "enterprise_candidate_kserve_component_receipts_changed"
        )
    receipt_components: list[ComponentAcceptance] = []
    for binding, expected_component in zip(
        report.component_receipts,
        EXPECTED_COMPONENTS,
        strict=True,
    ):
        receipt_path = _verify_binding(root, binding)
        receipt = verify_component_receipt(
            root,
            receipt_path,
            expected_component=expected_component,
        )
        receipt_components.append(receipt.acceptance)
    if tuple(receipt_components) != report.components:
        raise EnterpriseCandidateKServeAcceptanceError(
            "enterprise_candidate_kserve_component_projection_changed"
        )
    expected_sources = dict(_verified_source_bindings(str(root)))
    for item in report.components:
        if item.source != expected_sources[item.component]:
            raise EnterpriseCandidateKServeAcceptanceError(
                "enterprise_candidate_kserve_source_binding_changed"
            )
        _verify_component(root, item)
    if not report.hard_gates or not all(report.hard_gates.values()):
        raise EnterpriseCandidateKServeAcceptanceError(
            "enterprise_candidate_kserve_hard_gate_failed"
        )
    return report


@lru_cache(maxsize=4)
def _verified_source_bindings(root_value: str) -> tuple[tuple[str, SourceBinding], ...]:
    """Run heavyweight model-source verification once per acceptance process."""

    root = Path(root_value).resolve(strict=True)
    return tuple((source.component, source.binding(root)) for source in _verify_sources(root))


def _verify_component(root: Path, item: ComponentAcceptance) -> None:
    expected_method = {"LLM": "DPO", "TTS": "TTS", "EMBEDDING": "EMBEDDING"}[
        item.component
    ]
    if item.method != expected_method:
        raise EnterpriseCandidateKServeAcceptanceError(
            "enterprise_candidate_kserve_method_changed"
        )
    for binding in (
        item.runtime.runtime_source,
        item.runtime.proxy_source,
        item.runtime.metrics,
        item.runtime.worker_log,
        item.kubernetes.proxy_config_map,
        item.kubernetes.stable_inference_service,
        item.kubernetes.candidate_inference_service,
        *item.kubernetes.routes,
    ):
        _verify_binding(root, binding)
    stages = tuple(stage.stage for stage in item.stages)
    provider_stages = tuple(stage.stage for stage in item.provider_observations)
    expected_stages = ("SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK")
    if stages != expected_stages or provider_stages != expected_stages:
        raise EnterpriseCandidateKServeAcceptanceError(
            "enterprise_candidate_kserve_stage_sequence_changed"
        )
    for stage in item.stages:
        _verify_stage(stage)
    release = item.model_release
    reasons = set(release.transition_reason_codes)
    if (
        release.approval_requested_by_subject_id == release.approval_decided_by_subject_id
        or release.observation_stages != ("SHADOW", "CANARY_5", "CANARY_25")
        or not {
            "shadow_route_verified",
            "canary_5_route_verified",
            "canary_25_route_verified",
            "rollback_route_verified",
        }.issubset(reasons)
    ):
        raise EnterpriseCandidateKServeAcceptanceError(
            "enterprise_candidate_kserve_release_state_changed"
        )


def _verify_stage(stage: StageTrafficEvidence) -> None:
    if stage.request_count != STAGE_REQUEST_COUNTS[stage.stage]:
        raise EnterpriseCandidateKServeAcceptanceError(
            "enterprise_candidate_kserve_request_count_changed"
        )
    if stage.stage == "SHADOW":
        passed = bool(
            stage.stable_response_count == stage.request_count
            and stage.candidate_response_count == 0
            and stage.stable_request_delta == stage.request_count
            and stage.candidate_request_delta >= stage.request_count
            and stage.candidate_mirror_observed
            and stage.model_release_observation_decision == "PASS"
        )
    elif stage.stage in {"CANARY_5", "CANARY_25"}:
        lower, upper = (0.01, 0.15) if stage.stage == "CANARY_5" else (0.12, 0.40)
        passed = bool(
            lower <= stage.candidate_response_ratio <= upper
            and stage.stable_response_count >= 1
            and stage.candidate_response_count >= 1
            and stage.stable_request_delta == stage.stable_response_count
            and stage.candidate_request_delta == stage.candidate_response_count
            and not stage.candidate_mirror_observed
            and stage.model_release_observation_decision == "PASS"
        )
    else:
        passed = bool(
            stage.stable_response_count == stage.request_count
            and stage.candidate_response_count == 0
            and stage.stable_request_delta == stage.request_count
            and stage.candidate_request_delta == 0
            and stage.candidate_response_ratio == 0.0
            and not stage.candidate_mirror_observed
            and stage.model_release_observation_decision == "NOT_REQUIRED_AFTER_ROLLBACK"
        )
    if not passed:
        raise EnterpriseCandidateKServeAcceptanceError(
            "enterprise_candidate_kserve_stage_evidence_changed"
        )


def file_binding(root: Path, path: Path, *, reported_path: Path | None = None) -> FileBinding:
    source = _inside_file(root, path)
    return FileBinding(
        path=(reported_path or source.relative_to(root)).as_posix(),
        size_bytes=source.stat().st_size,
        sha256=file_sha256(source),
    )


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
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
    if file_binding(root, path) != binding:
        raise EnterpriseCandidateKServeAcceptanceError(
            "enterprise_candidate_kserve_file_binding_changed"
        )
    return path


def _inside_file(root: Path, value: Path) -> Path:
    path = value if value.is_absolute() else root / value
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EnterpriseCandidateKServeAcceptanceError(
            "enterprise_candidate_kserve_path_escapes_repository"
        ) from exc
    if not resolved.is_file():
        raise EnterpriseCandidateKServeAcceptanceError(
            "enterprise_candidate_kserve_file_is_missing"
        )
    return resolved
