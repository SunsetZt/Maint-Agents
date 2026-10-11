"""Read and verify historical AGT-007 project acceptance receipts."""

from __future__ import annotations

import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: Literal["maintenance-planning-value-lab/v2"] = "maintenance-planning-value-lab/v2"
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = "LOCAL_STAGING_PROJECT_AUTHORIZED"
STATUS: Literal["MAINTENANCE_PLANNING_VALUE_EVALUATION_PASSED"] = (
    "MAINTENANCE_PLANNING_VALUE_EVALUATION_PASSED"
)
OUTPUT_RELATIVE = Path("artifacts/m7-maintenance-planning-value-lab/acceptance.json")
_LEGACY_TEST_PATH = "tests/integration/test_maintenance_planning_value_evaluation.py"
_LEGACY_TEST_SNAPSHOT = Path(
    "contracts/evidence/maintenance-planning-value-lab-test-source.txt"
)
_LEGACY_DOCUMENT_PATH = "docs/m7-multi-agent-maintenance-planning.md"
_LEGACY_DOCUMENT_SNAPSHOT = Path(
    "contracts/evidence/maintenance-planning-value-lab-v1-document.txt"
)
_IMPLEMENTATION_PATHS = (
    Path("src/industrial_ops_agent/maintenance_planning/activation.py"),
    Path("src/industrial_ops_agent/maintenance_planning/evaluation.py"),
    Path("src/industrial_ops_agent/maintenance_planning/service.py"),
    Path("src/industrial_ops_agent/api/routes/maintenance_planning.py"),
    Path("src/industrial_ops_agent/api/routes/maintenance_planning_activations.py"),
    Path("src/industrial_ops_agent/api/routes/maintenance_planning_evaluations.py"),
    Path("src/industrial_ops_agent/api/app.py"),
    Path("src/industrial_ops_agent/runtime.py"),
    Path("src/industrial_ops_agent/persistence/models.py"),
    Path("alembic/versions/0080_maintenance_planning_value_evaluation.py"),
    Path("alembic/versions/0081_maintenance_planning_activation.py"),
    Path("web/app/maintenance-planning/page.tsx"),
    Path("web/lib/api/client.ts"),
    Path("contracts/openapi.json"),
    Path("web/generated/api/schema.d.ts"),
    Path(_LEGACY_TEST_PATH),
)


class MaintenancePlanningValueLabError(RuntimeError):
    """The project value-evaluation acceptance receipt is invalid."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ImplementationArtifact(_ClosedModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TargetedTestEvidence(_ClosedModel):
    command: tuple[str, ...] = Field(min_length=4)
    selected_test_count: Literal[1] = 1
    exit_code: Literal[0] = 0
    passed: Literal[True] = True
    combined_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completion_marker: Literal["1 passed"] = "1 passed"


class BlindEvaluationEvidence(_ClosedModel):
    execution_mode: Literal["IN_MEMORY_SQLITE_FASTAPI"] = "IN_MEMORY_SQLITE_FASTAPI"
    gold_tier: Literal["PROJECT_STAGING_GOLD"] = "PROJECT_STAGING_GOLD"
    sample_count: Literal[3] = 3
    anonymous_variant_count: Literal[2] = 2
    independent_judge_count: Literal[2] = 2
    judgment_count: Literal[6] = 6
    preference_disagreement_requires_third_judge: Literal[True] = True
    assignment_hidden_before_completion: Literal[True] = True
    assignment_revealed_after_completion: Literal[True] = True
    quality_delta: float = Field(ge=1.79, le=1.81)
    quality_ci_low: float = Field(gt=0)
    quality_ci_high: float = Field(gt=0)
    candidate_latency_ratio: float = Field(gt=0, lt=4.5)
    candidate_token_ratio: float = Field(gt=0, lt=5.0)
    safety_omission_regression: Literal[False] = False
    factual_error_regression: Literal[False] = False
    parts_false_positive_regression: Literal[False] = False
    dispatch_executability_regression: Literal[False] = False
    expert_review_time_regression: Literal[False] = False
    controlled_model_binding: Literal[True] = True
    performance_measurements_complete: Literal[True] = True
    project_sample_gate_passed: Literal[True] = True
    production_sample_gate_passed: Literal[False] = False
    enterprise_gold_gate_passed: Literal[False] = False
    decision: Literal["PROJECT_CANDIDATE_ELIGIBLE"] = "PROJECT_CANDIDATE_ELIGIBLE"


class BoundaryEvidence(_ClosedModel):
    advisory_only: Literal[True] = True
    runtime_activation_changed: Literal[False] = False
    work_order_mutation_count: Literal[0] = 0
    procurement_mutation_count: Literal[0] = 0
    inventory_mutation_count: Literal[0] = 0
    dispatch_mutation_count: Literal[0] = 0
    notification_mutation_count: Literal[0] = 0
    equipment_control_count: Literal[0] = 0
    background_services_started: tuple[str, ...] = ()


class MaintenancePlanningValueLabReport(_ClosedModel):
    schema_version: Literal[
        "maintenance-planning-value-lab/v1", "maintenance-planning-value-lab/v2"
    ] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    enterprise_production_data: Literal[False] = False
    project_generated_gold_data: Literal[True] = True
    project_enterprise_use_authorized: Literal[True] = True
    ready_for_project_enterprise_staging: Literal[True] = True
    ready_for_external_enterprise_production: Literal[False] = False
    generated_at: datetime
    git_commit: str = Field(min_length=7, max_length=64)
    status: Literal["MAINTENANCE_PLANNING_VALUE_EVALUATION_PASSED"] = STATUS
    evaluation: BlindEvaluationEvidence
    targeted_test: TargetedTestEvidence
    boundary: BoundaryEvidence
    implementation_artifacts: tuple[ImplementationArtifact, ...] = Field(min_length=10)
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")



def verify_maintenance_planning_value_lab(
    repo_root: Path,
    report_path: Path,
) -> MaintenancePlanningValueLabReport:
    root = repo_root.resolve(strict=True)
    path = _inside_file(root, report_path)
    try:
        report = MaintenancePlanningValueLabReport.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise MaintenancePlanningValueLabError(
            "maintenance_planning_value_receipt_is_invalid"
        ) from exc
    if report.evidence_chain_sha256 != _digest(report, exclude_chain=True):
        raise MaintenancePlanningValueLabError("maintenance_planning_value_evidence_chain_changed")
    expected_paths = {item.as_posix() for item in _IMPLEMENTATION_PATHS}
    if report.schema_version == "maintenance-planning-value-lab/v1":
        expected_paths.add(_LEGACY_DOCUMENT_PATH)
    observed_paths = {item.path for item in report.implementation_artifacts}
    if observed_paths != expected_paths:
        raise MaintenancePlanningValueLabError(
            "maintenance_planning_value_implementation_scope_changed"
        )
    for artifact in report.implementation_artifacts:
        # Resolve retired sources to byte-identical, read-only snapshots.
        # Original logical paths and all receipt checksums remain unchanged.
        source_path = {
            _LEGACY_DOCUMENT_PATH: _LEGACY_DOCUMENT_SNAPSHOT,
            _LEGACY_TEST_PATH: _LEGACY_TEST_SNAPSHOT,
        }.get(artifact.path, Path(artifact.path))
        source = _inside_file(root, source_path)
        if sha256(source.read_bytes()).hexdigest() != artifact.sha256:
            raise MaintenancePlanningValueLabError(
                "maintenance_planning_value_implementation_changed"
            )
    if (
        report.evaluation.decision != "PROJECT_CANDIDATE_ELIGIBLE"
        or report.evaluation.quality_ci_low <= 0
        or report.evaluation.production_sample_gate_passed is not False
        or report.boundary.runtime_activation_changed is not False
        or report.boundary.background_services_started
    ):
        raise MaintenancePlanningValueLabError(
            "maintenance_planning_value_acceptance_boundary_changed"
        )
    return report


def _inside_file(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise MaintenancePlanningValueLabError(
            "maintenance_planning_value_path_is_outside_repository"
        ) from exc
    if not resolved.is_file():
        raise MaintenancePlanningValueLabError("maintenance_planning_value_path_is_not_a_file")
    return resolved


def _digest(report: MaintenancePlanningValueLabReport, *, exclude_chain: bool) -> str:
    value = report.model_dump(
        mode="json",
        exclude={"evidence_chain_sha256"} if exclude_chain else None,
    )
    return sha256(_canonical(value)).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
