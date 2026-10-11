"""Additive verifier erratum for the Reranker v3 JSON tuple/list mismatch."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.evaluation.metrics import score_retrieval_paired_observations
from industrial_ops_agent.simulation import reranker_recovery_value_lab as lab

SCHEMA_VERSION: Literal["enterprise-reranker-recovery-verifier-erratum/v1"] = (
    "enterprise-reranker-recovery-verifier-erratum/v1"
)
STATUS: Literal["RERANKER_V3_REJECTION_VERIFIED_AFTER_JSON_NORMALIZATION"] = (
    "RERANKER_V3_REJECTION_VERIFIED_AFTER_JSON_NORMALIZATION"
)
AUTHORITATIVE_LATEST_RELATIVE = lab.OUTPUT_RELATIVE / "authoritative-latest.json"
EXPECTED_ORIGINAL_SOURCE_SHA256 = "ec920e986cf6b240c5b0bd0d8c19e6e0b65170640132d5d47882edb407bff667"


class RerankerRecoveryVerifierErratumError(RuntimeError):
    """The immutable v3 outcome or normalized verification is inconsistent."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileBinding(_ClosedModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CorrectedVerification(_ClosedModel):
    raw_python_structure_equal: Literal[False] = False
    canonical_json_structure_equal: Literal[True] = True
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


class RerankerRecoveryVerifierErratumReport(_ClosedModel):
    schema_version: Literal["enterprise-reranker-recovery-verifier-erratum/v1"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = lab.CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal["RERANKER_V3_REJECTION_VERIFIED_AFTER_JSON_NORMALIZATION"] = STATUS
    decision: Literal["REJECTION_UPHELD_DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD"] = (
        "REJECTION_UPHELD_DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD"
    )
    run_id: str = Field(pattern=r"^reranker-recovery-[0-9a-f]{20}$")
    defect_id: Literal["JSON_TUPLE_LIST_NORMALIZATION_MISMATCH"] = (
        "JSON_TUPLE_LIST_NORMALIZATION_MISMATCH"
    )
    defect_scope: Literal["FINAL_VERIFIER_REPRESENTATION_ONLY"] = (
        "FINAL_VERIFIER_REPRESENTATION_ONLY"
    )
    correction_method: Literal["CANONICAL_JSON_ROUND_TRIP_COMPARISON"] = (
        "CANONICAL_JSON_ROUND_TRIP_COMPARISON"
    )
    source: FileBinding
    immutable_original_verifier_source: FileBinding
    original_outcome: FileBinding
    original_latest_pointer: FileBinding
    formal_evaluation: FileBinding
    observations: FileBinding
    interrupted_gold_v2_retirement: FileBinding
    gold_v3_retirement: FileBinding
    original_evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corrected_verification: CorrectedVerification
    original_outcome_mutated: Literal[False] = False
    formal_gold_replayed: Literal[False] = False
    original_gold_retirement_preserved: Literal[True] = True
    formal_model_release_created: Literal[False] = False
    runtime_eligible: Literal[False] = False
    same_gold_reuse_permitted: Literal[False] = False
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def canonical_json_value(value: object) -> Any:
    """Normalize tuples and other JSON-compatible containers exactly as persistence does."""

    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def verify_original_with_json_normalization(
    repo_root: Path,
    outcome_path: Path | None = None,
) -> lab.RerankerRecoveryValueReport:
    """Verify the immutable v3 result while correcting only JSON representation equality."""

    root = repo_root.resolve(strict=True)
    latest_path = lab._latest_outcome_path(root)
    path = latest_path if outcome_path is None else lab._inside_file(root, outcome_path)
    if path != latest_path:
        raise RerankerRecoveryVerifierErratumError("original outcome is not the frozen latest")
    try:
        report = lab.RerankerRecoveryValueReport.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise RerankerRecoveryVerifierErratumError("original Reranker outcome is invalid") from exc
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != lab._digest(unsigned):
        raise RerankerRecoveryVerifierErratumError("original evidence chain changed")

    source_path = lab._inside_file(root, Path(report.source.path))
    if report.source.sha256 != EXPECTED_ORIGINAL_SOURCE_SHA256 or lab._file_evidence(
        Path(report.source.path), source_path
    ) != report.source.model_dump(mode="json"):
        raise RerankerRecoveryVerifierErratumError("immutable original verifier source changed")
    training_path, gold_path = lab.prepare_inputs(root)
    if lab.planned_run_id(root, report.image_digest) != report.run_id:
        raise RerankerRecoveryVerifierErratumError("original run identity changed")
    if report.predecessor != lab._verify_predecessor(root):
        raise RerankerRecoveryVerifierErratumError("original predecessor binding changed")
    model_path = lab._inside_directory(root, lab.BASE_MODEL_RELATIVE)
    if report.input_model.model_dump(mode="json") != lab._input_model_evidence(root, model_path):
        raise RerankerRecoveryVerifierErratumError("original base model changed")
    if report.dataset.model_dump(mode="json") != lab._dataset_evidence(
        root, training_path, gold_path
    ):
        raise RerankerRecoveryVerifierErratumError("original Reranker data changed")
    _verify_candidate(root, report)

    formal_path = lab._verify_file_evidence(root, report.evaluation.formal_report)
    observations_path = lab._verify_file_evidence(root, report.evaluation.observations)
    observations = lab._load_object(observations_path)
    cases = lab._gold_cases(lab._load_object(gold_path))
    baseline = lab._observations_from_document(observations, "baseline")
    candidate = lab._observations_from_document(observations, "candidate")
    formal = score_retrieval_paired_observations(
        cases,
        candidate,
        baseline,
        top_k=lab._CONFIG["top_k"],
        primary_metric="ndcg_at_k",
    )
    expected_formal = {
        "schema_version": "enterprise-reranker-formal-evaluation/v3",
        "classification": lab.CLASSIFICATION,
        "component": "RERANKER",
        "report": asdict(formal),
    }
    actual_formal = lab._load_object(formal_path)
    if actual_formal == expected_formal:
        raise RerankerRecoveryVerifierErratumError("documented raw representation defect vanished")
    if actual_formal != canonical_json_value(expected_formal):
        raise RerankerRecoveryVerifierErratumError("normalized formal evaluation changed")

    summary = lab._evaluation_summary(
        formal,
        evaluation_peak=report.evaluation.evaluation_peak_gpu_memory_reserved_bytes,
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
        raise RerankerRecoveryVerifierErratumError("normalized evaluation metrics changed")
    hard_gates = lab._hard_gates(
        runtime=report.runtime.model_dump(mode="json"),
        evaluation=summary,
    )
    expected_failed = tuple(name for name in lab._HARD_GATE_NAMES if not hard_gates[name])
    if (
        hard_gates != report.hard_gates.model_dump(mode="json")
        or expected_failed != report.failed_hard_gates
    ):
        raise RerankerRecoveryVerifierErratumError("normalized hard-gate result changed")
    accepted = not expected_failed
    if (
        report.candidate_accepted is not accepted
        or (accepted and report.status != "RERANKER_RECOVERY_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED")
        or (accepted and report.decision != "RERANKER_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT")
        or (not accepted and report.status != "RERANKER_RECOVERY_ACTUAL_GPU_CANDIDATE_REJECTED")
        or (not accepted and report.decision != "DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD")
    ):
        raise RerankerRecoveryVerifierErratumError("normalized decision is inconsistent")
    _verify_retirement(root, path, report, gold_path, accepted)
    return report


def build_reranker_recovery_verifier_erratum(
    repo_root: Path,
    outcome_path: Path | None = None,
) -> Path:
    """Create additive proof without mutating the original outcome or replaying Gold."""

    root = repo_root.resolve(strict=True)
    original_path = (
        lab._latest_outcome_path(root)
        if outcome_path is None
        else lab._inside_file(root, outcome_path)
    )
    original = verify_original_with_json_normalization(root, original_path)
    if original.candidate_accepted is not False or original.failed_hard_gates != (
        "candidate_absolute_quality",
    ):
        raise RerankerRecoveryVerifierErratumError(
            "erratum only applies to the frozen v3 absolute-quality rejection"
        )
    final_directory = root / lab.OUTPUT_RELATIVE / "errata" / original.run_id
    final_path = final_directory / "verification.json"
    if final_path.is_file():
        verify_reranker_recovery_verifier_erratum(root, final_path)
        return final_path
    if final_directory.exists():
        raise RerankerRecoveryVerifierErratumError("immutable erratum directory already exists")
    source_path = Path(__file__).resolve(strict=True)
    original_source = lab._inside_file(root, Path(original.source.path))
    formal_path = lab._inside_file(root, Path(original.evaluation.formal_report.path))
    observations_path = lab._inside_file(root, Path(original.evaluation.observations.path))
    interruption_path = lab._inside_file(root, lab._PREDECESSOR_INTERRUPTION)
    retirement_path = lab._inside_file(root, lab.GOLD_RETIREMENT_RELATIVE)
    latest_pointer = lab._inside_file(root, lab.LATEST_RELATIVE)
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
        "defect_id": "JSON_TUPLE_LIST_NORMALIZATION_MISMATCH",
        "defect_scope": "FINAL_VERIFIER_REPRESENTATION_ONLY",
        "correction_method": "CANONICAL_JSON_ROUND_TRIP_COMPARISON",
        "source": _file_binding(root, source_path),
        "immutable_original_verifier_source": _file_binding(root, original_source),
        "original_outcome": _file_binding(root, original_path),
        "original_latest_pointer": _file_binding(root, latest_pointer),
        "formal_evaluation": _file_binding(root, formal_path),
        "observations": _file_binding(root, observations_path),
        "interrupted_gold_v2_retirement": _file_binding(root, interruption_path),
        "gold_v3_retirement": _file_binding(root, retirement_path),
        "original_evidence_chain_sha256": original.evidence_chain_sha256,
        "corrected_verification": _corrected_projection(original),
        "original_outcome_mutated": False,
        "formal_gold_replayed": False,
        "original_gold_retirement_preserved": True,
        "formal_model_release_created": False,
        "runtime_eligible": False,
        "same_gold_reuse_permitted": False,
    }
    draft = RerankerRecoveryVerifierErratumReport.model_validate(
        {**unsigned, "evidence_chain_sha256": "0" * 64}
    )
    report = draft.model_copy(
        update={
            "evidence_chain_sha256": _digest(
                draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
            )
        }
    )
    temporary = final_directory.with_name(f".{final_directory.name}.{uuid4().hex}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        _write_json(temporary / "verification.json", report.model_dump(mode="json"))
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(final_directory)
    finally:
        if temporary.exists():
            temporary.rmdir()
    _write_authoritative_latest(root, final_path, report)
    verify_reranker_recovery_verifier_erratum(root, final_path)
    return final_path


def verify_reranker_recovery_verifier_erratum(
    repo_root: Path,
    erratum_path: Path | None = None,
) -> RerankerRecoveryVerifierErratumReport:
    root = repo_root.resolve(strict=True)
    path = (
        _authoritative_latest_path(root)
        if erratum_path is None
        else lab._inside_file(root, erratum_path)
    )
    try:
        report = RerankerRecoveryVerifierErratumReport.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise RerankerRecoveryVerifierErratumError("Reranker verifier erratum is invalid") from exc
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != _digest(unsigned):
        raise RerankerRecoveryVerifierErratumError("Reranker verifier erratum chain changed")
    bindings = (
        report.source,
        report.immutable_original_verifier_source,
        report.original_outcome,
        report.original_latest_pointer,
        report.formal_evaluation,
        report.observations,
        report.interrupted_gold_v2_retirement,
        report.gold_v3_retirement,
    )
    if any(
        _file_binding(root, lab._inside_file(root, Path(item.path))) != item for item in bindings
    ):
        raise RerankerRecoveryVerifierErratumError("Reranker verifier erratum input changed")
    original_path = lab._inside_file(root, Path(report.original_outcome.path))
    original = verify_original_with_json_normalization(root, original_path)
    if (
        original.run_id != report.run_id
        or original.evidence_chain_sha256 != report.original_evidence_chain_sha256
        or Path(original.source.path) != Path(report.immutable_original_verifier_source.path)
        or Path(original.evaluation.formal_report.path) != Path(report.formal_evaluation.path)
        or Path(original.evaluation.observations.path) != Path(report.observations.path)
        or _corrected_projection(original) != report.corrected_verification.model_dump(mode="json")
        or report.corrected_verification.failed_hard_gates != ("candidate_absolute_quality",)
    ):
        raise RerankerRecoveryVerifierErratumError("Reranker corrected verification changed")
    return report


def _verify_candidate(root: Path, report: lab.RerankerRecoveryValueReport) -> None:
    candidate_path = lab._inside_directory(root, Path(report.candidate.path))
    manifest = lab._directory_manifest(candidate_path)
    expected_files = [
        {
            "path": Path(item.path).relative_to(Path(report.candidate.path)).as_posix(),
            "size_bytes": item.size_bytes,
            "sha256": item.sha256,
        }
        for item in report.candidate.files
    ]
    if (
        lab._manifest_digest(manifest) != report.candidate.bundle_sha256
        or sum(int(item["size_bytes"]) for item in manifest) != report.candidate.size_bytes
        or manifest != expected_files
    ):
        raise RerankerRecoveryVerifierErratumError("original candidate bundle changed")


def _verify_retirement(
    root: Path,
    outcome_path: Path,
    report: lab.RerankerRecoveryValueReport,
    gold_path: Path,
    accepted: bool,
) -> None:
    retirement_path = lab._inside_file(root, Path(report.gold_retirement_path))
    retirement = lab._load_chained_object(
        retirement_path,
        schema="enterprise-reranker-gold-retirement/v3",
        run_id=report.run_id,
    )
    if (
        retirement.get("outcome") != ("PASSED" if accepted else "REJECTED")
        or retirement.get("formal_evaluation_count") != 1
        or retirement.get("gold_manifest_sha256") != lab._file_sha256(gold_path)
        or retirement.get("outcome_path") != outcome_path.relative_to(root).as_posix()
        or retirement.get("outcome_sha256") != lab._file_sha256(outcome_path)
        or retirement.get("reuse_permitted") is not False
        or retirement.get("retired_v1_gold_reused") is not False
        or retirement.get("interrupted_v2_gold_reused") is not False
        or retirement.get("model_release_created") is not False
        or retirement.get("runtime_eligible") is not False
    ):
        raise RerankerRecoveryVerifierErratumError("original Gold v3 retirement changed")


def _corrected_projection(
    report: lab.RerankerRecoveryValueReport,
) -> dict[str, Any]:
    return {
        "raw_python_structure_equal": False,
        "canonical_json_structure_equal": True,
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


def _write_authoritative_latest(
    root: Path,
    erratum_path: Path,
    report: RerankerRecoveryVerifierErratumReport,
) -> None:
    unsigned = {
        "schema_version": "enterprise-reranker-recovery-authoritative-latest/v1",
        "classification": lab.CLASSIFICATION,
        "production_claim": False,
        "run_id": report.run_id,
        "status": report.status,
        "outcome": erratum_path.relative_to(root).as_posix(),
        "outcome_sha256": _file_sha256(erratum_path),
        "evidence_chain_sha256": report.evidence_chain_sha256,
    }
    _write_json(
        root / AUTHORITATIVE_LATEST_RELATIVE,
        {**unsigned, "pointer_sha256": _digest(unsigned)},
    )


def _authoritative_latest_path(root: Path) -> Path:
    pointer_path = lab._inside_file(root, AUTHORITATIVE_LATEST_RELATIVE)
    pointer = _load_object(pointer_path)
    unsigned = {key: value for key, value in pointer.items() if key != "pointer_sha256"}
    value = pointer.get("outcome")
    if (
        pointer.get("schema_version") != "enterprise-reranker-recovery-authoritative-latest/v1"
        or pointer.get("pointer_sha256") != _digest(unsigned)
        or not isinstance(value, str)
        or not value
    ):
        raise RerankerRecoveryVerifierErratumError("authoritative latest pointer is invalid")
    path = lab._inside_file(root, Path(value))
    if pointer.get("outcome_sha256") != _file_sha256(path):
        raise RerankerRecoveryVerifierErratumError("authoritative erratum changed")
    return path


def _file_binding(root: Path, path: Path) -> FileBinding:
    actual = path.resolve(strict=True)
    try:
        relative = actual.relative_to(root)
    except ValueError as exc:
        raise RerankerRecoveryVerifierErratumError("evidence escaped repository root") from exc
    return FileBinding(
        path=relative.as_posix(),
        size_bytes=actual.stat().st_size,
        sha256=_file_sha256(actual),
    )


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RerankerRecoveryVerifierErratumError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise RerankerRecoveryVerifierErratumError(f"JSON evidence is not an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
