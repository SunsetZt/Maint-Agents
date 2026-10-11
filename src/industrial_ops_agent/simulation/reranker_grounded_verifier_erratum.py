"""Additive verifier erratum for the grounded Reranker v4 observation schema."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.evaluation.metrics import (
    ModelObservation,
    score_retrieval_paired_observations,
)
from industrial_ops_agent.simulation import reranker_grounded_value_lab as lab
from industrial_ops_agent.simulation import reranker_recovery_value_lab as v3

SCHEMA_VERSION: Literal["enterprise-reranker-grounded-verifier-erratum/v1"] = (
    "enterprise-reranker-grounded-verifier-erratum/v1"
)
STATUS: Literal["RERANKER_V4_REJECTION_VERIFIED_WITH_V4_OBSERVATION_SCHEMA"] = (
    "RERANKER_V4_REJECTION_VERIFIED_WITH_V4_OBSERVATION_SCHEMA"
)
AUTHORITATIVE_LATEST_RELATIVE = lab.OUTPUT_RELATIVE / "authoritative-latest.json"


class RerankerGroundedVerifierErratumError(RuntimeError):
    """The immutable v4 outcome or schema-aware verification is inconsistent."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileBinding(_ClosedModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CorrectedVerification(_ClosedModel):
    original_v3_parser_rejects_v4_schema: Literal[True] = True
    v4_schema_aware_parser_accepts_rows: Literal[True] = True
    outcome_semantics_changed: Literal[False] = False
    baseline_recall_at_k: float = Field(ge=0.0, le=1.0)
    candidate_recall_at_k: float = Field(ge=0.0, le=1.0)
    baseline_mrr: float = Field(ge=0.0, le=1.0)
    candidate_mrr: float = Field(ge=0.0, le=1.0)
    baseline_ndcg_at_k: float = Field(ge=0.0, le=1.0)
    candidate_ndcg_at_k: float = Field(ge=0.0, le=1.0)
    primary_improvement: float
    failed_hard_gates: tuple[str, ...]
    candidate_accepted: Literal[False] = False


class RerankerGroundedVerifierErratumReport(_ClosedModel):
    schema_version: Literal["enterprise-reranker-grounded-verifier-erratum/v1"] = (
        SCHEMA_VERSION
    )
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = lab.CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal["RERANKER_V4_REJECTION_VERIFIED_WITH_V4_OBSERVATION_SCHEMA"] = (
        STATUS
    )
    decision: Literal["REJECTION_UPHELD_DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD"] = (
        "REJECTION_UPHELD_DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD"
    )
    run_id: Literal["reranker-grounded-36ac19976887f824161a"]
    defect_id: Literal["V3_OBSERVATION_SCHEMA_REUSED_FOR_V4"] = (
        "V3_OBSERVATION_SCHEMA_REUSED_FOR_V4"
    )
    defect_scope: Literal["FINAL_OFFLINE_VERIFIER_ONLY"] = "FINAL_OFFLINE_VERIFIER_ONLY"
    correction_method: Literal["V4_SCHEMA_AWARE_ROW_PARSER"] = (
        "V4_SCHEMA_AWARE_ROW_PARSER"
    )
    source: FileBinding
    immutable_original_verifier_source: FileBinding
    original_outcome: FileBinding
    original_latest_pointer: FileBinding
    formal_evaluation: FileBinding
    observations: FileBinding
    terminology_registry: FileBinding
    final_gold_v4: FileBinding
    gold_v4_retirement: FileBinding
    original_evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corrected_verification: CorrectedVerification
    original_outcome_mutated: Literal[False] = False
    formal_gold_replayed: Literal[False] = False
    original_gold_retirement_preserved: Literal[True] = True
    formal_model_release_created: Literal[False] = False
    runtime_eligible: Literal[False] = False
    same_gold_reuse_permitted: Literal[False] = False
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def observations_from_v4_document(
    document: dict[str, Any], key: str
) -> tuple[ModelObservation, ...]:
    rows = document.get(key)
    if (
        document.get("schema_version") != "enterprise-reranker-grounded-observations/v4"
        or document.get("classification") != lab.CLASSIFICATION
        or document.get("component") != "RERANKER"
        or not isinstance(rows, list)
        or len(rows) != 12
    ):
        raise RerankerGroundedVerifierErratumError(
            "grounded Reranker v4 observations are invalid"
        )
    return tuple(v3._observation_from_document(item) for item in rows)


def verify_original_with_v4_observation_schema(
    repo_root: Path,
    outcome_path: Path | None = None,
) -> lab.RerankerGroundedValueReport:
    """Verify persisted v4 evidence without loading a model or replaying Gold."""

    root = repo_root.resolve(strict=True)
    latest_path = lab._latest_outcome_path(root)
    path = latest_path if outcome_path is None else v3._inside_file(root, outcome_path)
    if path != latest_path:
        raise RerankerGroundedVerifierErratumError(
            "original grounded outcome is not frozen latest"
        )
    try:
        report = lab.RerankerGroundedValueReport.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise RerankerGroundedVerifierErratumError(
            "original grounded outcome is invalid"
        ) from exc
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != v3._digest(unsigned):
        raise RerankerGroundedVerifierErratumError("original evidence chain changed")
    source_path = v3._inside_file(root, Path(report.source.path))
    if v3._file_evidence(
        Path(report.source.path), source_path
    ) != report.source.model_dump(mode="json"):
        raise RerankerGroundedVerifierErratumError("original verifier source changed")
    registry_path, gold_path = lab.prepare_inputs(root)
    if lab.planned_run_id(root, report.image_digest) != report.run_id:
        raise RerankerGroundedVerifierErratumError("original run identity changed")
    if report.predecessor != lab._predecessor_evidence(root):
        raise RerankerGroundedVerifierErratumError("original predecessor changed")
    model_path = v3._inside_directory(root, v3.BASE_MODEL_RELATIVE)
    if report.input_model.model_dump(mode="json") != v3._input_model_evidence(
        root, model_path
    ):
        raise RerankerGroundedVerifierErratumError("original base model changed")
    if report.registry.model_dump(mode="json") != lab._registry_evidence(
        root, registry_path
    ):
        raise RerankerGroundedVerifierErratumError("original registry changed")
    if report.dataset.model_dump(mode="json") != lab._dataset_evidence(root, gold_path):
        raise RerankerGroundedVerifierErratumError("original Gold changed")
    lab._verify_candidate(root, report)

    formal_path = v3._verify_file_evidence(root, report.evaluation.formal_report)
    observations_path = v3._verify_file_evidence(root, report.evaluation.observations)
    observations = v3._load_object(observations_path)
    baseline = observations_from_v4_document(observations, "baseline")
    candidate = observations_from_v4_document(observations, "candidate")
    cases = v3._gold_cases(v3._load_object(gold_path))
    formal = score_retrieval_paired_observations(
        cases,
        candidate,
        baseline,
        top_k=lab._CONFIG["top_k"],
        primary_metric="ndcg_at_k",
    )
    expected_formal = {
        "schema_version": "enterprise-reranker-grounded-formal-evaluation/v4",
        "classification": lab.CLASSIFICATION,
        "component": "RERANKER",
        "report": asdict(formal),
    }
    if lab._canonical_json(v3._load_object(formal_path)) != lab._canonical_json(
        expected_formal
    ):
        raise RerankerGroundedVerifierErratumError("formal evaluation changed")
    summary = lab._evaluation_summary(
        formal,
        evaluation_peak=report.evaluation.evaluation_peak_gpu_memory_reserved_bytes,
        candidate=candidate,
    )
    failures = lab._evaluation_failures(summary)
    expected_evaluation = {
        **summary,
        "primary_metric": "ndcg_at_k",
        "formal_report": report.evaluation.formal_report.model_dump(mode="json"),
        "observations": report.evaluation.observations.model_dump(mode="json"),
        "failed_quality_gates": list(failures),
    }
    if report.evaluation.model_dump(mode="json") != expected_evaluation:
        raise RerankerGroundedVerifierErratumError("evaluation metrics changed")
    hard_gates = lab._hard_gates(
        runtime=report.runtime.model_dump(mode="json"), evaluation=summary
    )
    expected_failed = tuple(
        name for name in lab._HARD_GATE_NAMES if not hard_gates[name]
    )
    if (
        report.hard_gates.model_dump(mode="json") != hard_gates
        or report.failed_hard_gates != expected_failed
        or expected_failed != ("reranker_quality_improved",)
        or report.candidate_accepted is not False
        or report.status != "RERANKER_GROUNDED_ACTUAL_GPU_CANDIDATE_REJECTED"
        or report.decision != "DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD"
    ):
        raise RerankerGroundedVerifierErratumError("rejection decision changed")
    lab._verify_gold_retirement(root, path, report, gold_path)
    return report


def build_reranker_grounded_verifier_erratum(
    repo_root: Path,
) -> Path:
    root = repo_root.resolve(strict=True)
    original_path = lab._latest_outcome_path(root)
    original = verify_original_with_v4_observation_schema(root, original_path)
    final_directory = root / lab.OUTPUT_RELATIVE / "errata" / original.run_id
    final_path = final_directory / "verification.json"
    if final_path.is_file():
        verify_reranker_grounded_verifier_erratum(root, final_path)
        return final_path
    if final_directory.exists():
        raise RerankerGroundedVerifierErratumError(
            "immutable grounded erratum directory already exists"
        )
    source_path = Path(__file__).resolve(strict=True)
    original_source = v3._inside_file(root, Path(original.source.path))
    formal_path = v3._inside_file(root, Path(original.evaluation.formal_report.path))
    observations_path = v3._inside_file(
        root, Path(original.evaluation.observations.path)
    )
    registry_path = v3._inside_file(root, lab.REGISTRY_RELATIVE)
    gold_path = v3._inside_file(root, lab.GOLD_RELATIVE)
    retirement_path = v3._inside_file(root, lab.GOLD_RETIREMENT_RELATIVE)
    latest_path = v3._inside_file(root, lab.LATEST_RELATIVE)
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": lab.CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "external_enterprise_production_claim": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": STATUS,
        "decision": "REJECTION_UPHELD_DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD",
        "run_id": original.run_id,
        "defect_id": "V3_OBSERVATION_SCHEMA_REUSED_FOR_V4",
        "defect_scope": "FINAL_OFFLINE_VERIFIER_ONLY",
        "correction_method": "V4_SCHEMA_AWARE_ROW_PARSER",
        "source": _binding(root, source_path),
        "immutable_original_verifier_source": _binding(root, original_source),
        "original_outcome": _binding(root, original_path),
        "original_latest_pointer": _binding(root, latest_path),
        "formal_evaluation": _binding(root, formal_path),
        "observations": _binding(root, observations_path),
        "terminology_registry": _binding(root, registry_path),
        "final_gold_v4": _binding(root, gold_path),
        "gold_v4_retirement": _binding(root, retirement_path),
        "original_evidence_chain_sha256": original.evidence_chain_sha256,
        "corrected_verification": _corrected_projection(original),
        "original_outcome_mutated": False,
        "formal_gold_replayed": False,
        "original_gold_retirement_preserved": True,
        "formal_model_release_created": False,
        "runtime_eligible": False,
        "same_gold_reuse_permitted": False,
    }
    draft = RerankerGroundedVerifierErratumReport.model_validate(
        {**unsigned, "evidence_chain_sha256": "0" * 64}
    )
    report = draft.model_copy(
        update={
            "evidence_chain_sha256": v3._digest(
                draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
            )
        }
    )
    temporary = root / lab.OUTPUT_RELATIVE / f".{original.run_id}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        v3._write_json(temporary / "verification.json", report.model_dump(mode="json"))
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(final_directory)
    finally:
        if temporary.exists():
            temporary.rmdir()
    _write_authoritative_latest(root, final_path, report)
    verify_reranker_grounded_verifier_erratum(root, final_path)
    return final_path


def verify_reranker_grounded_verifier_erratum(
    repo_root: Path,
    erratum_path: Path | None = None,
) -> RerankerGroundedVerifierErratumReport:
    root = repo_root.resolve(strict=True)
    path = (
        _authoritative_latest_path(root)
        if erratum_path is None
        else v3._inside_file(root, erratum_path)
    )
    try:
        report = RerankerGroundedVerifierErratumReport.model_validate_json(
            path.read_bytes()
        )
    except (OSError, ValueError) as exc:
        raise RerankerGroundedVerifierErratumError(
            "grounded verifier erratum is invalid"
        ) from exc
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != v3._digest(unsigned):
        raise RerankerGroundedVerifierErratumError("grounded erratum chain changed")
    bindings = (
        report.source,
        report.immutable_original_verifier_source,
        report.original_outcome,
        report.original_latest_pointer,
        report.formal_evaluation,
        report.observations,
        report.terminology_registry,
        report.final_gold_v4,
        report.gold_v4_retirement,
    )
    if any(
        _binding(root, v3._inside_file(root, Path(item.path))) != item
        for item in bindings
    ):
        raise RerankerGroundedVerifierErratumError("grounded erratum input changed")
    original_path = v3._inside_file(root, Path(report.original_outcome.path))
    original = verify_original_with_v4_observation_schema(root, original_path)
    if (
        original.run_id != report.run_id
        or original.evidence_chain_sha256 != report.original_evidence_chain_sha256
        or Path(original.source.path)
        != Path(report.immutable_original_verifier_source.path)
        or Path(original.evaluation.formal_report.path)
        != Path(report.formal_evaluation.path)
        or Path(original.evaluation.observations.path) != Path(report.observations.path)
        or _corrected_projection(original)
        != report.corrected_verification.model_dump(mode="json")
    ):
        raise RerankerGroundedVerifierErratumError(
            "grounded corrected verification changed"
        )
    return report


def _corrected_projection(
    report: lab.RerankerGroundedValueReport,
) -> dict[str, Any]:
    return {
        "original_v3_parser_rejects_v4_schema": True,
        "v4_schema_aware_parser_accepts_rows": True,
        "outcome_semantics_changed": False,
        "baseline_recall_at_k": report.evaluation.baseline_recall_at_k,
        "candidate_recall_at_k": report.evaluation.candidate_recall_at_k,
        "baseline_mrr": report.evaluation.baseline_mrr,
        "candidate_mrr": report.evaluation.candidate_mrr,
        "baseline_ndcg_at_k": report.evaluation.baseline_ndcg_at_k,
        "candidate_ndcg_at_k": report.evaluation.candidate_ndcg_at_k,
        "primary_improvement": report.evaluation.primary_improvement,
        "failed_hard_gates": list(report.failed_hard_gates),
        "candidate_accepted": False,
    }


def _binding(root: Path, path: Path) -> FileBinding:
    resolved = v3._inside_file(root, path)
    return FileBinding(
        path=resolved.relative_to(root).as_posix(),
        size_bytes=resolved.stat().st_size,
        sha256=v3._file_sha256(resolved),
    )


def _write_authoritative_latest(
    root: Path,
    report_path: Path,
    report: RerankerGroundedVerifierErratumReport,
) -> None:
    unsigned = {
        "schema_version": "enterprise-reranker-grounded-authoritative-latest/v1",
        "classification": lab.CLASSIFICATION,
        "run_id": report.run_id,
        "status": report.status,
        "report": report_path.relative_to(root).as_posix(),
        "report_sha256": v3._file_sha256(report_path),
        "evidence_chain_sha256": report.evidence_chain_sha256,
    }
    v3._write_json(
        root / AUTHORITATIVE_LATEST_RELATIVE,
        {**unsigned, "pointer_chain_sha256": v3._digest(unsigned)},
    )


def _authoritative_latest_path(root: Path) -> Path:
    path = v3._inside_file(root, AUTHORITATIVE_LATEST_RELATIVE)
    pointer = v3._load_object(path)
    unsigned = {
        key: value for key, value in pointer.items() if key != "pointer_chain_sha256"
    }
    value = pointer.get("report")
    if (
        pointer.get("schema_version")
        != "enterprise-reranker-grounded-authoritative-latest/v1"
        or pointer.get("classification") != lab.CLASSIFICATION
        or not isinstance(value, str)
        or not value
        or pointer.get("pointer_chain_sha256") != v3._digest(unsigned)
    ):
        raise RerankerGroundedVerifierErratumError(
            "grounded authoritative pointer is invalid"
        )
    report_path = v3._inside_file(root, Path(value))
    if pointer.get("report_sha256") != v3._file_sha256(report_path):
        raise RerankerGroundedVerifierErratumError(
            "grounded authoritative pointer changed"
        )
    return report_path
