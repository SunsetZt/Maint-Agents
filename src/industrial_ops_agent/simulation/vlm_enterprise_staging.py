"""Adopt one verified VLM candidate as the enterprise project's staging default."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.simulation.vlm_continuation_lab import (
    SimulatedVlmContinuationLabError,
    SimulatedVlmContinuationReport,
    verify_simulated_vlm_continuation_lab,
)

CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = (
    "LOCAL_STAGING_PROJECT_AUTHORIZED"
)
SCHEMA_VERSION: Literal["enterprise-vlm-staging-adoption/v1"] = (
    "enterprise-vlm-staging-adoption/v1"
)
STATUS: Literal["ENTERPRISE_VLM_STAGING_DEFAULT_ADOPTED"] = (
    "ENTERPRISE_VLM_STAGING_DEFAULT_ADOPTED"
)
SOURCE_CLASSIFICATION: Literal["SIMULATED_NON_PRODUCTION"] = "SIMULATED_NON_PRODUCTION"
SOURCE_DECISION: Literal["SIMULATION_CONTINUATION_ELIGIBLE"] = (
    "SIMULATION_CONTINUATION_ELIGIBLE"
)

REQUIRED_HARD_GATES = (
    "data_governance",
    "no_blocking_regressions",
    "vlm_hallucination",
    "vlm_media_integrity",
    "vlm_region_grounding",
)
AUTHORIZED_BUSINESS_USES = (
    "industrial_incident_image_triage",
    "field_work_order_photo_review",
    "video_keyframe_visual_evidence",
    "after_sales_remote_assistance",
)
PROHIBITED_AUTOMATIONS = (
    "autonomous_physical_control",
    "unreviewed_final_diagnosis",
    "safety_critical_action_without_human_confirmation",
    "external_enterprise_production_release_claim",
)


class EnterpriseVlmStagingAdoptionError(RuntimeError):
    """The requested enterprise-staging adoption is invalid or has been tampered with."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnterpriseVlmCandidateMetrics(_ClosedModel):
    evaluation_cases: int = Field(gt=0)
    diagnostic_accuracy: float = Field(ge=0, le=1)
    mean_region_iou: float = Field(ge=0, le=1)
    hallucination_rate: float = Field(ge=0, le=1)
    accuracy_delta: float
    region_iou_delta: float
    hard_gates: dict[str, Literal[True]]


class EnterpriseVlmCandidateBinding(_ClosedModel):
    source_directory: str = Field(min_length=1)
    source_acceptance_path: str = Field(min_length=1)
    source_acceptance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_classification: Literal["SIMULATED_NON_PRODUCTION"] = SOURCE_CLASSIFICATION
    source_decision: Literal["SIMULATION_CONTINUATION_ELIGIBLE"] = SOURCE_DECISION
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    model_license: str = Field(min_length=1)
    training_snapshot_id: str = Field(min_length=1)
    training_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_snapshot_id: str = Field(min_length=1)
    evaluation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_suite_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    artifact_object_key: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_size_bytes: int = Field(gt=0)
    metrics: EnterpriseVlmCandidateMetrics


class EnterpriseVlmRuntimeBinding(_ClosedModel):
    environment: Literal["ENTERPRISE_STAGING"] = "ENTERPRISE_STAGING"
    component: Literal["vlm"] = "vlm"
    model_alias: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )
    gateway_provider: Literal["GatewayVlmProvider"] = "GatewayVlmProvider"
    request_class: Literal["VLM"] = "VLM"
    selection_status: Literal["ACTIVE_PROJECT_DEFAULT"] = "ACTIVE_PROJECT_DEFAULT"
    deployment_status: Literal["READY_FOR_LOCAL_KSERVE_SHADOW"] = (
        "READY_FOR_LOCAL_KSERVE_SHADOW"
    )
    default_for_enterprise_project: Literal[True] = True
    business_use_authorized: Literal[True] = True
    human_confirmation_required: Literal[True] = True
    kserve_shadow_endpoint_verified: Literal[False] = False


class EnterpriseVlmStagingAdoptionReport(_ClosedModel):
    schema_version: Literal["enterprise-vlm-staging-adoption/v1"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    operational_tier: Literal["ENTERPRISE_PROJECT_STAGING"] = "ENTERPRISE_PROJECT_STAGING"
    status: Literal["ENTERPRISE_VLM_STAGING_DEFAULT_ADOPTED"] = STATUS
    authorization_basis: Literal["PROJECT_OWNER_EXPLICIT_ADOPTION"] = (
        "PROJECT_OWNER_EXPLICIT_ADOPTION"
    )
    enterprise_candidate: Literal[True] = True
    project_enterprise_use_authorized: Literal[True] = True
    production_claim: Literal[False] = False
    enterprise_production_data: Literal[False] = False
    project_generated_data: Literal[True] = True
    formal_production_release_created: Literal[False] = False
    adopted_at: datetime
    candidate: EnterpriseVlmCandidateBinding
    runtime: EnterpriseVlmRuntimeBinding
    authorized_business_uses: tuple[str, ...]
    prohibited_automations: tuple[str, ...]
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def adopt_vlm_for_enterprise_staging(
    repo_root: Path,
    *,
    source_dir: Path,
    output_path: Path,
    model_alias: str = "industrial-diagnosis",
) -> tuple[EnterpriseVlmStagingAdoptionReport, bool]:
    """Create one immutable project-owner decision over a verified VLM candidate."""

    root = repo_root.resolve(strict=True)
    source = _inside_repo_directory(root, source_dir)
    source_report = _verify_source(source)
    candidate = _candidate_binding(root, source, source_report)
    runtime = EnterpriseVlmRuntimeBinding(model_alias=model_alias)
    target = _output_file(root, output_path)
    if target.exists():
        existing = verify_enterprise_vlm_staging_adoption(root, target)
        if existing.candidate != candidate or existing.runtime != runtime:
            raise EnterpriseVlmStagingAdoptionError(
                "enterprise_vlm_adoption_output_has_different_binding"
            )
        return existing, False

    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "operational_tier": "ENTERPRISE_PROJECT_STAGING",
        "status": STATUS,
        "authorization_basis": "PROJECT_OWNER_EXPLICIT_ADOPTION",
        "enterprise_candidate": True,
        "project_enterprise_use_authorized": True,
        "production_claim": False,
        "enterprise_production_data": False,
        "project_generated_data": True,
        "formal_production_release_created": False,
        "adopted_at": datetime.now(UTC).isoformat(),
        "candidate": candidate.model_dump(mode="json"),
        "runtime": runtime.model_dump(mode="json"),
        "authorized_business_uses": list(AUTHORIZED_BUSINESS_USES),
        "prohibited_automations": list(PROHIBITED_AUTOMATIONS),
    }
    draft = EnterpriseVlmStagingAdoptionReport.model_validate(
        {**unsigned, "evidence_chain_sha256": "0" * 64}
    )
    report = draft.model_copy(update={"evidence_chain_sha256": _report_digest(draft)})
    _write_atomic(report, target)
    return verify_enterprise_vlm_staging_adoption(root, target), True


def verify_enterprise_vlm_staging_adoption(
    repo_root: Path,
    acceptance_path: Path,
) -> EnterpriseVlmStagingAdoptionReport:
    """Verify the decision receipt and the full immutable source candidate."""

    root = repo_root.resolve(strict=True)
    path = _inside_repo_file(root, acceptance_path)
    try:
        report = EnterpriseVlmStagingAdoptionReport.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise EnterpriseVlmStagingAdoptionError(
            "enterprise_vlm_adoption_acceptance_is_invalid"
        ) from exc
    if report.evidence_chain_sha256 != _report_digest(report):
        raise EnterpriseVlmStagingAdoptionError(
            "enterprise_vlm_adoption_acceptance_digest_mismatch"
        )
    source = _inside_repo_directory(root, Path(report.candidate.source_directory))
    acceptance = _inside_repo_file(root, Path(report.candidate.source_acceptance_path))
    if acceptance != source / "acceptance.json":
        raise EnterpriseVlmStagingAdoptionError(
            "enterprise_vlm_adoption_source_acceptance_path_is_invalid"
        )
    if sha256(acceptance.read_bytes()).hexdigest() != report.candidate.source_acceptance_sha256:
        raise EnterpriseVlmStagingAdoptionError(
            "enterprise_vlm_adoption_source_acceptance_changed"
        )
    source_report = _verify_source(source)
    expected = _candidate_binding(root, source, source_report)
    if report.candidate != expected:
        raise EnterpriseVlmStagingAdoptionError(
            "enterprise_vlm_adoption_candidate_binding_changed"
        )
    if (
        report.authorized_business_uses != AUTHORIZED_BUSINESS_USES
        or report.prohibited_automations != PROHIBITED_AUTOMATIONS
    ):
        raise EnterpriseVlmStagingAdoptionError(
            "enterprise_vlm_adoption_business_scope_changed"
        )
    return report


def _verify_source(source: Path) -> SimulatedVlmContinuationReport:
    try:
        report = verify_simulated_vlm_continuation_lab(source)
    except SimulatedVlmContinuationLabError as exc:
        raise EnterpriseVlmStagingAdoptionError(
            "enterprise_vlm_adoption_source_verification_failed"
        ) from exc
    gates = report.evaluation.hard_gates
    if (
        report.classification != SOURCE_CLASSIFICATION
        or report.production_claim is not False
        or report.enterprise_production_data is not False
        or report.production_release_eligible is not False
        or report.formal_release_created is not False
        or report.decision != SOURCE_DECISION
        or report.evaluation.decision != SOURCE_DECISION
        or set(gates) != set(REQUIRED_HARD_GATES)
        or not all(gates.values())
    ):
        raise EnterpriseVlmStagingAdoptionError(
            "enterprise_vlm_adoption_source_is_not_eligible"
        )
    return report


def _candidate_binding(
    root: Path,
    source: Path,
    report: SimulatedVlmContinuationReport,
) -> EnterpriseVlmCandidateBinding:
    acceptance = source / "acceptance.json"
    evaluation = report.evaluation
    return EnterpriseVlmCandidateBinding(
        source_directory=source.relative_to(root).as_posix(),
        source_acceptance_path=acceptance.relative_to(root).as_posix(),
        source_acceptance_sha256=sha256(acceptance.read_bytes()).hexdigest(),
        source_evidence_chain_sha256=report.evidence_chain_sha256,
        model_id=report.model_id,
        model_revision=report.model_revision,
        model_license=report.model_license,
        training_snapshot_id=report.training_snapshot_id,
        training_manifest_sha256=report.training_manifest_sha256,
        evaluation_snapshot_id=report.evaluation_snapshot_id,
        evaluation_manifest_sha256=report.evaluation_manifest_sha256,
        evaluation_suite_id=report.simulation_reference_suite_id,
        experiment_id=report.training.child_experiment_id,
        artifact_id=report.training.child_artifact_id,
        artifact_object_key=report.training.child_artifact_object_key,
        artifact_sha256=report.training.child_artifact_sha256,
        artifact_size_bytes=report.training.child_artifact_size_bytes,
        metrics=EnterpriseVlmCandidateMetrics(
            evaluation_cases=report.evaluation_cases,
            diagnostic_accuracy=evaluation.child_accuracy,
            mean_region_iou=evaluation.child_mean_region_iou,
            hallucination_rate=evaluation.child_hallucination_rate,
            accuracy_delta=evaluation.accuracy_delta,
            region_iou_delta=evaluation.region_iou_delta,
            hard_gates={name: True for name in REQUIRED_HARD_GATES},
        ),
    )


def _inside_repo_directory(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise EnterpriseVlmStagingAdoptionError(
            "enterprise_vlm_adoption_path_is_outside_repository"
        ) from exc
    if not resolved.is_dir() or resolved.is_symlink():
        raise EnterpriseVlmStagingAdoptionError(
            "enterprise_vlm_adoption_source_is_not_a_directory"
        )
    return resolved


def _inside_repo_file(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise EnterpriseVlmStagingAdoptionError(
            "enterprise_vlm_adoption_path_is_outside_repository"
        ) from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise EnterpriseVlmStagingAdoptionError(
            "enterprise_vlm_adoption_path_is_not_a_file"
        )
    return resolved


def _output_file(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EnterpriseVlmStagingAdoptionError(
            "enterprise_vlm_adoption_path_is_outside_repository"
        ) from exc
    if resolved.exists() and not resolved.is_file():
        raise EnterpriseVlmStagingAdoptionError(
            "enterprise_vlm_adoption_output_is_not_a_file"
        )
    return resolved


def _write_atomic(report: EnterpriseVlmStagingAdoptionReport, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_bytes(_canonical(report.model_dump(mode="json")) + b"\n")
    temporary.replace(target)


def _report_digest(report: EnterpriseVlmStagingAdoptionReport) -> str:
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    return sha256(_canonical(unsigned)).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
