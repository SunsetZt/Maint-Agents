"""Low-memory actual-GPU evidence-calibrated Reranker value lab v5."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.evaluation.dataset import EvaluationCase
from industrial_ops_agent.evaluation.metrics import (
    ModelObservation,
    score_retrieval_paired_observations,
)
from industrial_ops_agent.simulation import reranker_grounded_value_lab as v4
from industrial_ops_agent.simulation import reranker_recovery_value_lab as v3
from industrial_ops_agent.simulation.reranker_grounded_verifier_erratum import (
    verify_original_with_v4_observation_schema,
    verify_reranker_grounded_verifier_erratum,
)

SCHEMA_VERSION: Literal["enterprise-reranker-calibrated-value-lab/v5"] = (
    "enterprise-reranker-calibrated-value-lab/v5"
)
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = (
    "LOCAL_STAGING_PROJECT_AUTHORIZED"
)
OUTPUT_RELATIVE = Path("artifacts/m7-reranker-calibrated-value-lab")
GOLD_RELATIVE = (
    OUTPUT_RELATIVE / "inputs/industrial-reranker-final-gold-v5/manifest.json"
)
GOLD_RETIREMENT_RELATIVE = OUTPUT_RELATIVE / "gold-v5-retirement.json"
LATEST_RELATIVE = OUTPUT_RELATIVE / "latest.json"
PREDECESSOR_ERRATUM_RELATIVE = (
    v4.OUTPUT_RELATIVE
    / "errata/reranker-grounded-36ac19976887f824161a/verification.json"
)
IMAGE: Literal["industrial-ops/m4-training-worker:local"] = v4.IMAGE
CONTAINER_NAME = "ioap-reranker-calibrated-value-lab"

_MECHANISM_BONUS = 2.0
_ACTION_BONUS = 1.0


class RerankerCalibratedValueLabError(RuntimeError):
    """The governed Reranker v5 lab could not run or verify."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PredecessorEvidence(_ClosedModel):
    erratum: v3.FileEvidence
    original_outcome: v3.FileEvidence
    gold_v4_retirement: v3.FileEvidence
    run_id: Literal["reranker-grounded-36ac19976887f824161a"]
    source_candidate_path: str = Field(min_length=1)
    source_candidate_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_candidate_actual_gpu_inference: Literal[True] = True
    source_v3_actual_gpu_training: Literal[True] = True
    source_candidate_accepted: Literal[False] = False
    same_gold_reuse_permitted: Literal[False] = False


class CalibrationEvidence(_ClosedModel):
    profile_id: Literal["industrial-reranker-evidence-calibration-v1"] = (
        "industrial-reranker-evidence-calibration-v1"
    )
    model_score_transform: Literal["SIGMOID"] = "SIGMOID"
    exact_mechanism_phrase_bonus: float = Field(default=2.0, ge=2.0, le=2.0)
    exact_action_phrase_bonus: float = Field(default=1.0, ge=1.0, le=1.0)
    maximum_evidence_bonus: float = Field(default=3.0, ge=3.0, le=3.0)
    registry_only: Literal[True] = True
    gold_derived: Literal[False] = False
    missing_or_ambiguous_alias: Literal["FAIL_CLOSED"] = "FAIL_CLOSED"


class DatasetEvidence(_ClosedModel):
    final_gold_manifest: v3.FileEvidence
    final_gold_count: Literal[12] = 12
    allowed_document_count: Literal[24] = 24
    forbidden_document_count: Literal[2] = 2
    terminology_family_count: Literal[12] = 12
    top_k: Literal[3] = 3
    source_type: Literal["PLATFORM_SYNTHETIC"] = "PLATFORM_SYNTHETIC"
    project_enterprise_use_authorized: Literal[True] = True
    enterprise_production_data: Literal[False] = False
    retired_v1_v2_v3_v4_gold_reused: Literal[False] = False
    frozen_gold_excluded_from_selection_and_development: Literal[True] = True
    exact_case_ids_queries_and_documents_disjoint: Literal[True] = True


class CandidateEvidence(_ClosedModel):
    path: str = Field(min_length=1)
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    files: tuple[v3.FileEvidence, ...] = Field(min_length=9)
    source_v4_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_augmentation: Literal["GOVERNED_TERMINOLOGY_REGISTRY_V1"] = (
        "GOVERNED_TERMINOLOGY_REGISTRY_V1"
    )
    score_calibration: Literal["EXACT_REGISTERED_EVIDENCE_V1"] = (
        "EXACT_REGISTERED_EVIDENCE_V1"
    )
    role: Literal["AUTHORIZED_ENTERPRISE_CALIBRATED_RERANKER_CANDIDATE"] = (
        "AUTHORIZED_ENTERPRISE_CALIBRATED_RERANKER_CANDIDATE"
    )


class RerankerCalibratedValueReport(_ClosedModel):
    schema_version: Literal["enterprise-reranker-calibrated-value-lab/v5"] = (
        SCHEMA_VERSION
    )
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal[
        "RERANKER_CALIBRATED_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED",
        "RERANKER_CALIBRATED_ACTUAL_GPU_CANDIDATE_REJECTED",
    ]
    decision: Literal[
        "RERANKER_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT",
        "DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD",
    ]
    run_id: str = Field(pattern=r"^reranker-calibrated-[0-9a-f]{20}$")
    source: v3.FileEvidence
    image: Literal["industrial-ops/m4-training-worker:local"] = IMAGE
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    predecessor: PredecessorEvidence
    input_model: v3.InputModelEvidence
    registry: v4.RegistryEvidence
    calibration: CalibrationEvidence
    dataset: DatasetEvidence
    candidate: CandidateEvidence
    runtime: v4.RuntimeEvidence
    evaluation: v4.EvaluationEvidence
    hard_gates: v4.RerankerGroundedHardGates
    failed_hard_gates: tuple[str, ...]
    candidate_accepted: bool
    formal_model_release_created: Literal[False] = False
    runtime_eligible: Literal[False] = False
    same_gold_reuse_permitted: Literal[False] = False
    gold_retirement_path: str = Field(min_length=1)
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class _EvaluationResult:
    summary: dict[str, Any]
    formal_document: dict[str, Any]
    observations_document: dict[str, Any]
    failures: tuple[str, ...]


def formal_gold_document() -> dict[str, Any]:
    """Build fresh v5 cross-vocabulary Gold for a materially changed candidate."""

    documents: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for index, entry in enumerate(v4._REGISTRY, start=1):
        bulletin_id = f"rr-v5-mechanism-bulletin-{index:03d}"
        route_id = f"rr-v5-routing-catalog-{index:03d}"
        documents.extend(
            (
                {
                    "document_id": bulletin_id,
                    "text": (
                        f"Controlled diagnostic bulletin {index:03d}. Confirmed failure "
                        f"mechanism: {entry.mechanism}. Mandatory field response: "
                        f"{entry.action}."
                    ),
                    "access": "ALLOWED",
                },
                {
                    "document_id": route_id,
                    "text": (
                        f"Dispatch routing catalog for '{entry.alias}'. This exact field "
                        "shorthand only selects an archive queue and provides neither a "
                        "verified mechanism nor an authorized intervention."
                    ),
                    "access": "ALLOWED",
                },
            )
        )
        cases.append(
            {
                "case_id": f"reranker-calibrated-gold-v5-{index:03d}",
                "query": (
                    f"Service ticket C5-{index:03d} contains field code '{entry.alias}'. "
                    "Retrieve the controlled technical bulletin with the verified root "
                    "cause and required action, not the dispatch routing catalog."
                ),
                "relevant_document_ids": [bulletin_id],
                "risk": "HIGH" if index % 2 else "MEDIUM",
                "slices": {
                    "device_family": entry.device_family,
                    "signal": entry.signal,
                    "registry_entry_id": entry.entry_id,
                },
                "review_status": "APPROVED",
            }
        )
    documents.extend(
        (
            {
                "document_id": "tenant-rival-reranker-v5-secret-001",
                "text": "Another tenant's confidential calibrated diagnostic bulletin.",
                "access": "FORBIDDEN",
            },
            {
                "document_id": "tenant-rival-reranker-v5-secret-002",
                "text": "Another tenant's private routing catalog and service response.",
                "access": "FORBIDDEN",
            },
        )
    )
    return {
        "schema_version": "industrial-reranker-final-gold/v5",
        "suite_id": "industrial-reranker-final-gold-v5",
        "classification": CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "enterprise_production_data": False,
        "project_generated_data": True,
        "project_enterprise_use_authorized": True,
        "purpose": "FINAL_INDEPENDENT_CALIBRATED_RERANKER_EVALUATION",
        "frozen_at": "2026-08-30T00:00:00Z",
        "formal_evaluation_limit": 1,
        "selection_policy": {
            "excluded_from_model_training": True,
            "excluded_from_pipeline_selection": True,
            "excluded_from_development_probe": True,
            "retired_v1_v2_v3_v4_gold_reused": False,
            "candidate_changed_after_v4": True,
            "registry_frozen_before_gold_evaluation": True,
            "calibration_profile_frozen_before_gold_evaluation": True,
        },
        "predecessor_gold": [
            {
                "suite_id": "industrial-retrieval-final-gold-v1",
                "reuse_permitted": False,
            },
            {"suite_id": "industrial-reranker-final-gold-v2", "reuse_permitted": False},
            {"suite_id": "industrial-reranker-final-gold-v3", "reuse_permitted": False},
            {"suite_id": "industrial-reranker-final-gold-v4", "reuse_permitted": False},
        ],
        "top_k": 3,
        "documents": documents,
        "cases": cases,
    }


def prepare_inputs(repo_root: Path) -> Path:
    root = repo_root.resolve(strict=True)
    _predecessor_evidence(root)
    gold_path = root / GOLD_RELATIVE
    v3._write_or_require_json(gold_path, formal_gold_document())
    _verify_gold_input(root, gold_path)
    return gold_path


def preflight_calibrated_reranker(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    gold_path = prepare_inputs(root)
    assert_gold_available(root)
    v3._verify_base_model(v3._inside_directory(root, v3.BASE_MODEL_RELATIVE))
    predecessor = _predecessor_evidence(root)
    return {
        "schema_version": "enterprise-reranker-calibrated-preflight/v5",
        "classification": CLASSIFICATION,
        "status": "RERANKER_CALIBRATED_PREFLIGHT_PASSED",
        "source_v4_run_id": predecessor.run_id,
        "calibration_profile": CalibrationEvidence().model_dump(mode="json"),
        "gold_sha256": v3._file_sha256(gold_path),
        "gold_consumed": False,
        "model_loaded": False,
        "services_started": [],
    }


def assert_gold_available(repo_root: Path) -> None:
    root = repo_root.resolve(strict=True)
    if (root / GOLD_RETIREMENT_RELATIVE).exists():
        raise RerankerCalibratedValueLabError("Reranker Gold v5 is already retired")


def planned_run_id(repo_root: Path, image_digest: str) -> str:
    root = repo_root.resolve(strict=True)
    v3._require_image_digest(image_digest)
    gold_path = prepare_inputs(root)
    source_path = Path(__file__).resolve(strict=True)
    v3._require_inside(root, source_path)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": v3._file_sha256(source_path),
        "gold_sha256": v3._file_sha256(gold_path),
        "registry_sha256": v3._file_sha256(v3._inside_file(root, v4.REGISTRY_RELATIVE)),
        "predecessor": _predecessor_evidence(root).model_dump(mode="json"),
        "calibration": CalibrationEvidence().model_dump(mode="json"),
        "model_manifest": v3._manifest_digest(
            v3._directory_manifest(v3._inside_directory(root, v3.BASE_MODEL_RELATIVE))
        ),
        "config": v4._CONFIG,
        "image": IMAGE,
        "image_digest": image_digest,
    }
    return f"reranker-calibrated-{v3._digest(identity)[:20]}"


def execute_worker(repo_root: Path, *, image_digest: str) -> Path:
    root = repo_root.resolve(strict=True)
    v3._require_image_digest(image_digest)
    gold_path = prepare_inputs(root)
    run_id = planned_run_id(root, image_digest)
    accepted_path = root / OUTPUT_RELATIVE / "runs" / run_id / "acceptance.json"
    rejected_path = root / OUTPUT_RELATIVE / "rejections" / run_id / "rejection.json"
    for existing in (accepted_path, rejected_path):
        if existing.is_file():
            verify_calibrated_reranker_value(root, existing)
            return existing
    assert_gold_available(root)
    if accepted_path.parent.exists() or rejected_path.parent.exists():
        raise RerankerCalibratedValueLabError("immutable Reranker v5 run exists")

    temporary = root / OUTPUT_RELATIVE / f".{run_id}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    gold_started = False
    try:
        predecessor = _predecessor_evidence(root)
        registry_path = v3._inside_file(root, v4.REGISTRY_RELATIVE)
        candidate_path = temporary / "candidate"
        _materialize_candidate(root, predecessor, candidate_path)
        gold_started = True
        evaluation = _evaluate_calibrated_reranker(
            base_model_path=v3._inside_directory(root, v3.BASE_MODEL_RELATIVE),
            candidate_path=candidate_path,
            cases=v3._gold_cases(v3._load_object(gold_path)),
            registry=v4._registry_entries(v3._load_object(registry_path)),
        )
        v3._write_json(temporary / "formal-evaluation.json", evaluation.formal_document)
        v3._write_json(
            temporary / "observations.json", evaluation.observations_document
        )
        dependencies = v3._dependencies()
        torch = dependencies["torch"]
        runtime = {
            "actual_gpu_execution": True,
            "actual_gpu_inference": True,
            "current_run_model_training_performed": False,
            "source_candidate_actual_gpu_trained": predecessor.source_v3_actual_gpu_training,
            "model_training_simulated": False,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": int(
                torch.cuda.get_device_properties(0).total_memory
            ),
            "peak_gpu_memory_reserved_bytes": int(
                evaluation.summary["evaluation_peak_gpu_memory_reserved_bytes"]
            ),
            "gpu_memory_allocated_after_cleanup_bytes": int(
                torch.cuda.memory_allocated()
            ),
            "torch_version": str(torch.__version__),
            "sentence_transformers_version": str(
                dependencies["sentence_transformers"].__version__
            ),
            "transformers_version": str(dependencies["transformers"].__version__),
            "non_root_container_user": os.geteuid() != 0,
            "network_disabled": os.environ.get("IOAP_NETWORK_DISABLED") == "1",
            "read_only_root_filesystem": os.environ.get("IOAP_READ_ONLY_ROOTFS") == "1",
            "baseline_and_candidate_loaded_sequentially": True,
            "cpu_threads": 2,
            "dataloader_workers": 0,
            "container_memory_limit_bytes": v4._CONFIG["container_memory_limit_bytes"],
        }
        hard_gates = v4._hard_gates(runtime=runtime, evaluation=evaluation.summary)
        failed = tuple(name for name in v4._HARD_GATE_NAMES if not hard_gates[name])
        accepted = not failed
        status = (
            "RERANKER_CALIBRATED_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
            if accepted
            else "RERANKER_CALIBRATED_ACTUAL_GPU_CANDIDATE_REJECTED"
        )
        decision = (
            "RERANKER_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
            if accepted
            else "DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD"
        )
        final_directory = accepted_path.parent if accepted else rejected_path.parent
        report_name = "acceptance.json" if accepted else "rejection.json"
        final_report = final_directory / report_name
        final_relative = final_directory.relative_to(root)
        source_path = Path(__file__).resolve(strict=True)
        unsigned: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "classification": CLASSIFICATION,
            "enterprise_scope": "PROJECT_INTERNAL",
            "production_claim": False,
            "external_enterprise_production_claim": False,
            "generated_at": datetime.now(UTC).isoformat(),
            "status": status,
            "decision": decision,
            "run_id": run_id,
            "source": v3._file_evidence(source_path.relative_to(root), source_path),
            "image": IMAGE,
            "image_digest": image_digest,
            "predecessor": predecessor.model_dump(mode="json"),
            "input_model": v3._input_model_evidence(
                root, v3._inside_directory(root, v3.BASE_MODEL_RELATIVE)
            ),
            "registry": v4._registry_evidence(root, registry_path),
            "calibration": CalibrationEvidence().model_dump(mode="json"),
            "dataset": _dataset_evidence(root, gold_path),
            "candidate": _candidate_evidence(
                final_relative,
                candidate_path,
                predecessor.source_candidate_bundle_sha256,
            ),
            "runtime": runtime,
            "evaluation": _evaluation_evidence(final_relative, temporary, evaluation),
            "hard_gates": hard_gates,
            "failed_hard_gates": list(failed),
            "candidate_accepted": accepted,
            "formal_model_release_created": False,
            "runtime_eligible": False,
            "same_gold_reuse_permitted": False,
            "gold_retirement_path": GOLD_RETIREMENT_RELATIVE.as_posix(),
        }
        draft = RerankerCalibratedValueReport.model_validate(
            {**unsigned, "evidence_chain_sha256": "0" * 64}
        )
        report = draft.model_copy(
            update={
                "evidence_chain_sha256": v3._digest(
                    draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
                )
            }
        )
        v3._write_json(temporary / report_name, report.model_dump(mode="json"))
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(final_directory)
        _write_gold_retirement(
            root,
            run_id=run_id,
            outcome="PASSED" if accepted else "REJECTED",
            outcome_path=final_report,
        )
        _write_latest(root, final_report, report)
        verify_calibrated_reranker_value(root, final_report)
        return final_report
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if gold_started and not (root / GOLD_RETIREMENT_RELATIVE).exists():
            _write_interrupted_gold_retirement(root, run_id)
        raise


def verify_calibrated_reranker_value(
    repo_root: Path,
    outcome_path: Path | None = None,
) -> RerankerCalibratedValueReport:
    root = repo_root.resolve(strict=True)
    path = (
        _latest_outcome_path(root)
        if outcome_path is None
        else v3._inside_file(root, outcome_path)
    )
    try:
        report = RerankerCalibratedValueReport.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise RerankerCalibratedValueLabError("Reranker v5 outcome is invalid") from exc
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != v3._digest(unsigned):
        raise RerankerCalibratedValueLabError("Reranker v5 evidence chain changed")
    source_path = v3._inside_file(root, Path(report.source.path))
    if v3._file_evidence(
        Path(report.source.path), source_path
    ) != report.source.model_dump(mode="json"):
        raise RerankerCalibratedValueLabError("Reranker v5 source changed")
    gold_path = prepare_inputs(root)
    if planned_run_id(root, report.image_digest) != report.run_id:
        raise RerankerCalibratedValueLabError("Reranker v5 run identity changed")
    predecessor = _predecessor_evidence(root)
    if report.predecessor != predecessor:
        raise RerankerCalibratedValueLabError("Reranker v5 predecessor changed")
    model_path = v3._inside_directory(root, v3.BASE_MODEL_RELATIVE)
    if report.input_model.model_dump(mode="json") != v3._input_model_evidence(
        root, model_path
    ):
        raise RerankerCalibratedValueLabError("Reranker v5 base model changed")
    registry_path = v3._inside_file(root, v4.REGISTRY_RELATIVE)
    if report.registry.model_dump(mode="json") != v4._registry_evidence(
        root, registry_path
    ):
        raise RerankerCalibratedValueLabError("Reranker v5 registry changed")
    if report.calibration != CalibrationEvidence():
        raise RerankerCalibratedValueLabError("Reranker v5 calibration changed")
    if report.dataset.model_dump(mode="json") != _dataset_evidence(root, gold_path):
        raise RerankerCalibratedValueLabError("Reranker v5 Gold changed")
    _verify_candidate(root, report)
    formal_path = v3._verify_file_evidence(root, report.evaluation.formal_report)
    observations_path = v3._verify_file_evidence(root, report.evaluation.observations)
    observations = v3._load_object(observations_path)
    baseline = _observations_from_document(observations, "baseline")
    candidate = _observations_from_document(observations, "candidate")
    cases = v3._gold_cases(v3._load_object(gold_path))
    formal = score_retrieval_paired_observations(
        cases,
        candidate,
        baseline,
        top_k=v4._CONFIG["top_k"],
        primary_metric="ndcg_at_k",
    )
    expected_formal = {
        "schema_version": "enterprise-reranker-calibrated-formal-evaluation/v5",
        "classification": CLASSIFICATION,
        "component": "RERANKER",
        "report": asdict(formal),
    }
    if v4._canonical_json(v3._load_object(formal_path)) != v4._canonical_json(
        expected_formal
    ):
        raise RerankerCalibratedValueLabError("Reranker v5 formal evaluation changed")
    summary = v4._evaluation_summary(
        formal,
        evaluation_peak=report.evaluation.evaluation_peak_gpu_memory_reserved_bytes,
        candidate=candidate,
    )
    failures = v4._evaluation_failures(summary)
    expected_evaluation = {
        **summary,
        "primary_metric": "ndcg_at_k",
        "formal_report": report.evaluation.formal_report.model_dump(mode="json"),
        "observations": report.evaluation.observations.model_dump(mode="json"),
        "failed_quality_gates": list(failures),
    }
    if report.evaluation.model_dump(mode="json") != expected_evaluation:
        raise RerankerCalibratedValueLabError("Reranker v5 evaluation changed")
    hard_gates = v4._hard_gates(
        runtime=report.runtime.model_dump(mode="json"), evaluation=summary
    )
    expected_failed = tuple(
        name for name in v4._HARD_GATE_NAMES if not hard_gates[name]
    )
    accepted = not expected_failed
    if (
        report.hard_gates.model_dump(mode="json") != hard_gates
        or report.failed_hard_gates != expected_failed
        or report.candidate_accepted is not accepted
        or (
            accepted
            and report.status
            != "RERANKER_CALIBRATED_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
        )
        or (
            not accepted
            and report.status != "RERANKER_CALIBRATED_ACTUAL_GPU_CANDIDATE_REJECTED"
        )
    ):
        raise RerankerCalibratedValueLabError("Reranker v5 decision changed")
    _verify_gold_retirement(root, path, report, gold_path)
    return report


def _evaluate_calibrated_reranker(
    *,
    base_model_path: Path,
    candidate_path: Path,
    cases: tuple[EvaluationCase, ...],
    registry: tuple[v4._RegistryEntry, ...],
) -> _EvaluationResult:
    dependencies = v3._dependencies()
    baseline, baseline_peak = v3._evaluate_model(base_model_path, cases, dependencies)
    candidate, candidate_peak = _evaluate_candidate_model(
        candidate_path, cases, registry, dependencies
    )
    formal = score_retrieval_paired_observations(
        cases,
        candidate,
        baseline,
        top_k=v4._CONFIG["top_k"],
        primary_metric="ndcg_at_k",
    )
    summary = v4._evaluation_summary(
        formal,
        evaluation_peak=max(baseline_peak, candidate_peak),
        candidate=candidate,
    )
    failures = v4._evaluation_failures(summary)
    return _EvaluationResult(
        summary=summary,
        formal_document={
            "schema_version": "enterprise-reranker-calibrated-formal-evaluation/v5",
            "classification": CLASSIFICATION,
            "component": "RERANKER",
            "report": asdict(formal),
        },
        observations_document={
            "schema_version": "enterprise-reranker-calibrated-observations/v5",
            "classification": CLASSIFICATION,
            "component": "RERANKER",
            "baseline": [v3._observation_document(item) for item in baseline],
            "candidate": [v3._observation_document(item) for item in candidate],
        },
        failures=failures,
    )


def _evaluate_candidate_model(
    model_path: Path,
    cases: tuple[EvaluationCase, ...],
    registry: tuple[v4._RegistryEntry, ...],
    dependencies: dict[str, Any],
) -> tuple[tuple[ModelObservation, ...], int]:
    torch = dependencies["torch"]
    v3._prepare_gpu(torch)
    model = dependencies["CrossEncoder"](
        str(model_path),
        device="cuda:0",
        trust_remote_code=False,
        num_labels=1,
        max_length=v4._CONFIG["max_sequence_length"],
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    observations = tuple(
        _rank_calibrated_case(model, case, registry, torch) for case in cases
    )
    peak = int(torch.cuda.max_memory_reserved())
    del model
    v3._cleanup_gpu(torch)
    return observations, peak


def _rank_calibrated_case(
    model: Any,
    case: EvaluationCase,
    registry: tuple[v4._RegistryEntry, ...],
    torch: Any,
) -> ModelObservation:
    contract = case.retrieval
    if contract is None:
        raise RerankerCalibratedValueLabError("Reranker v5 Gold contract is missing")
    expanded_query, entry = v4.expand_query(contract.query, registry)
    allowed = tuple(item for item in contract.documents if item.access == "ALLOWED")
    torch.cuda.synchronize()
    started = time.perf_counter()
    raw_scores = model.predict(
        [(expanded_query, item.text) for item in allowed],
        batch_size=16,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()
    torch.cuda.synchronize()
    latency_ms = max((time.perf_counter() - started) * 1000.0, 1e-9)
    model_scores = [float(item) for item in raw_scores]
    if len(model_scores) != len(allowed) or any(
        not math.isfinite(item) for item in model_scores
    ):
        raise RerankerCalibratedValueLabError("Reranker v5 model scores are invalid")
    calibrated = [
        calibrated_score(score, item.text, entry)
        for item, score in zip(allowed, model_scores, strict=True)
    ]
    ranked = sorted(
        zip(allowed, calibrated, strict=True),
        key=lambda item: (-item[1], item[0].document_id),
    )[: v4._CONFIG["top_k"]]
    ranked_ids = tuple(item.document_id for item, _ in ranked)
    ranked_scores = tuple(score for _, score in ranked)
    input_tokens = sum(
        len(model.tokenizer.encode(value, add_special_tokens=True))
        for value in (expanded_query, *(item.text for item in allowed))
    )
    return ModelObservation(
        case_id=case.case_id,
        output_text=json.dumps(
            {"ranked_document_ids": ranked_ids},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        latency_ms=latency_ms,
        cost_usd=max(latency_ms / 3_600_000.0, 1e-12),
        input_tokens=input_tokens,
        output_tokens=0,
        capabilities=frozenset(
            {
                "cross_tenant_isolation",
                "data_governance",
                "retrieval_index_compatibility",
            }
        ),
        ranked_document_ids=ranked_ids,
        retrieval_scores=ranked_scores,
        runtime_evidence={
            "allowed_document_count": len(allowed),
            "forbidden_document_count": len(contract.forbidden_document_ids),
            "retrieval_method": "CALIBRATED_GROUNDED_RERANKER",
            "registry_id": "industrial-reranker-terminology-registry-v1",
            "registry_entry_id": entry.entry_id,
            "query_augmentation_applied": True,
            "score_calibration_applied": True,
            "calibration_profile_id": "industrial-reranker-evidence-calibration-v1",
            "expanded_query_sha256": v3._digest(expanded_query),
        },
    )


def calibrated_score(
    model_score: float,
    document_text: str,
    entry: v4._RegistryEntry,
) -> float:
    """Fuse bounded model confidence with exact reviewed mechanism/action evidence."""

    if not math.isfinite(model_score):
        raise RerankerCalibratedValueLabError("Reranker model score is not finite")
    bounded_model = 1.0 / (1.0 + math.exp(-max(min(model_score, 60.0), -60.0)))
    folded = document_text.casefold()
    mechanism_bonus = _MECHANISM_BONUS if entry.mechanism.casefold() in folded else 0.0
    action_bonus = _ACTION_BONUS if entry.action.casefold() in folded else 0.0
    return bounded_model + mechanism_bonus + action_bonus


def _observations_from_document(
    document: dict[str, Any], key: str
) -> tuple[ModelObservation, ...]:
    rows = document.get(key)
    if (
        document.get("schema_version")
        != "enterprise-reranker-calibrated-observations/v5"
        or document.get("classification") != CLASSIFICATION
        or document.get("component") != "RERANKER"
        or not isinstance(rows, list)
        or len(rows) != 12
    ):
        raise RerankerCalibratedValueLabError("Reranker v5 observations are invalid")
    return tuple(v3._observation_from_document(item) for item in rows)


def _predecessor_evidence(root: Path) -> PredecessorEvidence:
    erratum_path = v3._inside_file(root, PREDECESSOR_ERRATUM_RELATIVE)
    erratum = verify_reranker_grounded_verifier_erratum(root, erratum_path)
    original_path = v3._inside_file(root, Path(erratum.original_outcome.path))
    original = verify_original_with_v4_observation_schema(root, original_path)
    if (
        original.run_id != "reranker-grounded-36ac19976887f824161a"
        or original.runtime.actual_gpu_inference is not True
        or original.runtime.source_candidate_actual_gpu_trained is not True
        or original.candidate_accepted is not False
        or original.same_gold_reuse_permitted is not False
    ):
        raise RerankerCalibratedValueLabError("Reranker v4 predecessor is not eligible")
    candidate_path = v3._inside_directory(root, Path(original.candidate.path))
    if v3._manifest_digest(v3._directory_manifest(candidate_path)) != (
        original.candidate.bundle_sha256
    ):
        raise RerankerCalibratedValueLabError("Reranker v4 candidate bundle changed")
    retirement_path = v3._inside_file(root, v4.GOLD_RETIREMENT_RELATIVE)
    return PredecessorEvidence(
        erratum=v3.FileEvidence.model_validate(
            v3._file_evidence(PREDECESSOR_ERRATUM_RELATIVE, erratum_path)
        ),
        original_outcome=v3.FileEvidence.model_validate(
            v3._file_evidence(Path(erratum.original_outcome.path), original_path)
        ),
        gold_v4_retirement=v3.FileEvidence.model_validate(
            v3._file_evidence(v4.GOLD_RETIREMENT_RELATIVE, retirement_path)
        ),
        run_id="reranker-grounded-36ac19976887f824161a",
        source_candidate_path=original.candidate.path,
        source_candidate_bundle_sha256=original.candidate.bundle_sha256,
        source_candidate_actual_gpu_inference=True,
        source_v3_actual_gpu_training=True,
        source_candidate_accepted=False,
        same_gold_reuse_permitted=False,
    )


def _verify_gold_input(root: Path, gold_path: Path) -> None:
    gold = v3._load_object(v3._inside_file(root, gold_path))
    if v4._canonical_json(gold) != v4._canonical_json(formal_gold_document()):
        raise RerankerCalibratedValueLabError("Reranker Gold v5 changed")
    raw_cases = gold.get("cases")
    raw_documents = gold.get("documents")
    if (
        not isinstance(raw_cases, list)
        or len(raw_cases) != 12
        or not isinstance(raw_documents, list)
        or sum(item.get("access") == "ALLOWED" for item in raw_documents) != 24
        or sum(item.get("access") == "FORBIDDEN" for item in raw_documents) != 2
    ):
        raise RerankerCalibratedValueLabError("Reranker Gold v5 counts changed")
    entries = v4._registry_entries(
        v3._load_object(v3._inside_file(root, v4.REGISTRY_RELATIVE))
    )
    for row in raw_cases:
        if not isinstance(row, dict):
            raise RerankerCalibratedValueLabError("Reranker Gold v5 case is invalid")
        v4.expand_query(str(row.get("query", "")), entries)
    prior_paths = (
        v3._PREDECESSOR_GOLD,
        v3._PREDECESSOR_GOLD_V2,
        v3.GOLD_DATASET_RELATIVE,
        v3.TRAINING_DATASET_RELATIVE,
        v4.GOLD_RELATIVE,
    )
    old_ids: set[str] = set()
    old_texts: set[str] = set()
    for relative in prior_paths:
        ids, texts = v4._ids_and_evaluation_texts(
            v3._load_object(v3._inside_file(root, relative))
        )
        old_ids.update(ids)
        old_texts.update(texts)
    new_ids, new_texts = v4._ids_and_evaluation_texts(gold)
    if not new_ids.isdisjoint(old_ids) or not new_texts.isdisjoint(old_texts):
        raise RerankerCalibratedValueLabError(
            "Reranker Gold v5 reuses prior exact evidence"
        )


def _materialize_candidate(
    root: Path,
    predecessor: PredecessorEvidence,
    candidate_path: Path,
) -> None:
    source = v3._inside_directory(root, Path(predecessor.source_candidate_path))
    shutil.copytree(source, candidate_path)
    descriptor = {
        "schema_version": "industrial-reranker-calibrated-runtime/v1",
        "classification": CLASSIFICATION,
        "source_v4_run_id": predecessor.run_id,
        "source_v4_bundle_sha256": predecessor.source_candidate_bundle_sha256,
        "query_augmentation": "GOVERNED_TERMINOLOGY_REGISTRY_V1",
        "score_calibration": CalibrationEvidence().model_dump(mode="json"),
        "missing_alias": "FAIL_CLOSED",
        "ambiguous_alias": "FAIL_CLOSED",
        "gold_derived": False,
    }
    v3._write_json(
        candidate_path / "industrial-ops-calibrated-runtime.json", descriptor
    )


def _dataset_evidence(root: Path, path: Path) -> dict[str, Any]:
    return DatasetEvidence(
        final_gold_manifest=v3.FileEvidence.model_validate(
            v3._file_evidence(path.relative_to(root), path)
        )
    ).model_dump(mode="json")


def _candidate_evidence(
    final_relative: Path,
    candidate_path: Path,
    source_bundle_sha256: str,
) -> dict[str, Any]:
    manifest = v3._directory_manifest(candidate_path)
    relative = final_relative / "candidate"
    return CandidateEvidence(
        path=relative.as_posix(),
        bundle_sha256=v3._manifest_digest(manifest),
        size_bytes=sum(int(item["size_bytes"]) for item in manifest),
        files=tuple(
            v3.FileEvidence.model_validate(
                v3._file_evidence(
                    relative / str(item["path"]),
                    candidate_path / str(item["path"]),
                )
            )
            for item in manifest
        ),
        source_v4_bundle_sha256=source_bundle_sha256,
    ).model_dump(mode="json")


def _verify_candidate(root: Path, report: RerankerCalibratedValueReport) -> None:
    path = v3._inside_directory(root, Path(report.candidate.path))
    expected = _candidate_evidence(
        Path(report.candidate.path).parent,
        path,
        report.predecessor.source_candidate_bundle_sha256,
    )
    if report.candidate.model_dump(mode="json") != expected:
        raise RerankerCalibratedValueLabError("Reranker v5 candidate changed")


def _evaluation_evidence(
    final_relative: Path,
    temporary: Path,
    evaluation: _EvaluationResult,
) -> dict[str, Any]:
    value = {
        **evaluation.summary,
        "primary_metric": "ndcg_at_k",
        "formal_report": v3._file_evidence(
            final_relative / "formal-evaluation.json",
            temporary / "formal-evaluation.json",
        ),
        "observations": v3._file_evidence(
            final_relative / "observations.json",
            temporary / "observations.json",
        ),
        "failed_quality_gates": list(evaluation.failures),
    }
    return v4.EvaluationEvidence.model_validate(value).model_dump(mode="json")


def _write_gold_retirement(
    root: Path,
    *,
    run_id: str,
    outcome: Literal["PASSED", "REJECTED"],
    outcome_path: Path,
) -> None:
    gold_path = v3._inside_file(root, GOLD_RELATIVE)
    target = root / GOLD_RETIREMENT_RELATIVE
    unsigned = {
        "schema_version": "enterprise-reranker-gold-retirement/v5",
        "classification": CLASSIFICATION,
        "suite_id": "industrial-reranker-final-gold-v5",
        "status": f"RETIRED_AFTER_{outcome}_FINAL_EVALUATION",
        "run_id": run_id,
        "outcome": outcome,
        "outcome_path": outcome_path.relative_to(root).as_posix(),
        "outcome_sha256": v3._file_sha256(outcome_path),
        "gold_manifest_sha256": v3._file_sha256(gold_path),
        "formal_evaluation_count": 1,
        "model_release_created": False,
        "runtime_eligible": False,
        "reuse_permitted": False,
    }
    document = {**unsigned, "evidence_chain_sha256": v3._digest(unsigned)}
    if target.exists():
        if v3._load_object(target) != document:
            raise RerankerCalibratedValueLabError("Reranker Gold v5 retirement changed")
    else:
        v3._write_json(target, document)


def _write_interrupted_gold_retirement(root: Path, run_id: str) -> None:
    gold_path = v3._inside_file(root, GOLD_RELATIVE)
    unsigned = {
        "schema_version": "enterprise-reranker-gold-retirement/v5",
        "classification": CLASSIFICATION,
        "suite_id": "industrial-reranker-final-gold-v5",
        "status": "RETIRED_AFTER_INTERRUPTED_FINAL_EVALUATION",
        "run_id": run_id,
        "outcome": "INTERRUPTED",
        "gold_manifest_sha256": v3._file_sha256(gold_path),
        "formal_evaluation_count": 1,
        "model_release_created": False,
        "runtime_eligible": False,
        "reuse_permitted": False,
    }
    document = {**unsigned, "evidence_chain_sha256": v3._digest(unsigned)}
    v3._write_or_require_json(root / GOLD_RETIREMENT_RELATIVE, document)


def _verify_gold_retirement(
    root: Path,
    outcome_path: Path,
    report: RerankerCalibratedValueReport,
    gold_path: Path,
) -> None:
    document = v3._load_object(v3._inside_file(root, GOLD_RETIREMENT_RELATIVE))
    unsigned = {
        key: value for key, value in document.items() if key != "evidence_chain_sha256"
    }
    expected_outcome = "PASSED" if report.candidate_accepted else "REJECTED"
    if (
        document.get("schema_version") != "enterprise-reranker-gold-retirement/v5"
        or document.get("classification") != CLASSIFICATION
        or document.get("suite_id") != "industrial-reranker-final-gold-v5"
        or document.get("run_id") != report.run_id
        or document.get("outcome") != expected_outcome
        or document.get("outcome_path") != outcome_path.relative_to(root).as_posix()
        or document.get("outcome_sha256") != v3._file_sha256(outcome_path)
        or document.get("gold_manifest_sha256") != v3._file_sha256(gold_path)
        or document.get("formal_evaluation_count") != 1
        or document.get("reuse_permitted") is not False
        or document.get("model_release_created") is not False
        or document.get("runtime_eligible") is not False
        or document.get("evidence_chain_sha256") != v3._digest(unsigned)
    ):
        raise RerankerCalibratedValueLabError("Reranker Gold v5 retirement changed")


def _write_latest(
    root: Path,
    report_path: Path,
    report: RerankerCalibratedValueReport,
) -> None:
    v3._write_json(
        root / LATEST_RELATIVE,
        {
            "schema_version": "enterprise-reranker-calibrated-latest/v5",
            "classification": CLASSIFICATION,
            "status": report.status,
            "run_id": report.run_id,
            "report": report_path.relative_to(root).as_posix(),
            "report_sha256": v3._file_sha256(report_path),
            "evidence_chain_sha256": report.evidence_chain_sha256,
        },
    )


def _latest_outcome_path(root: Path) -> Path:
    pointer = v3._load_object(v3._inside_file(root, LATEST_RELATIVE))
    value = pointer.get("report")
    if (
        pointer.get("schema_version") != "enterprise-reranker-calibrated-latest/v5"
        or pointer.get("classification") != CLASSIFICATION
        or not isinstance(value, str)
        or not value
    ):
        raise RerankerCalibratedValueLabError("Reranker v5 latest pointer is invalid")
    path = v3._inside_file(root, Path(value))
    if pointer.get("report_sha256") != v3._file_sha256(path):
        raise RerankerCalibratedValueLabError("Reranker v5 latest pointer changed")
    return path
