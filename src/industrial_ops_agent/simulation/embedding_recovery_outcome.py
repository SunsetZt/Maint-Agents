"""Verify and bind the rejected Embedding v2 GPU experiment as audit history."""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, TypeGuard
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: Literal["embedding-recovery-rejection-acceptance/v1"] = (
    "embedding-recovery-rejection-acceptance/v1"
)
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = "LOCAL_STAGING_PROJECT_AUTHORIZED"
STATUS: Literal["EMBEDDING_RECOVERY_REJECTION_VERIFIED"] = "EMBEDDING_RECOVERY_REJECTION_VERIFIED"
OUTPUT_RELATIVE = Path("artifacts/m7-embedding-recovery-value-lab/rejection-acceptance.json")
BASE_MODEL_WEIGHT_RELATIVE = Path(
    "artifacts/m7-retrieval-enterprise-value-lab/inputs/all-MiniLM-L6-v2-1110a243/model.safetensors"
)
TRAINING_MANIFEST_RELATIVE = Path(
    "artifacts/m7-embedding-recovery-value-lab/inputs/"
    "industrial-embedding-domain-training-v2/manifest.json"
)
GOLD_MANIFEST_RELATIVE = Path(
    "artifacts/m7-embedding-recovery-value-lab/inputs/"
    "industrial-embedding-final-gold-v2/manifest.json"
)
MODEL_WEIGHT_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
TRAINING_MANIFEST_SHA256 = "117d3541df927c4f9d682c7f1e5fbb0b57a4751f3619fcee649b4555a4263ca9"
GOLD_MANIFEST_SHA256 = "693ca26301cd2f8aed6ee1fc3d11a24b1a0e3fcc049827fe96d86a798e489802"
EXPECTED_IMAGE_DIGEST = "sha256:7bcd50262cf62f33f8a3728a53f49ee36cbcc206f2bc0970776442f0b105effe"
EXPECTED_RETRIEVAL_GATES = frozenset(
    {
        "cross_tenant_isolation",
        "data_governance",
        "no_blocking_regressions",
        "retrieval_index_compatibility",
    }
)


class EmbeddingRecoveryOutcomeError(RuntimeError):
    """Embedding v2 rejection evidence is missing, changed, or inconsistent."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileBinding(_ClosedModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DirectoryBinding(_ClosedModel):
    path: str = Field(min_length=1)
    file_count: int = Field(gt=0)
    size_bytes: int = Field(gt=0)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TrainingOutcome(_ClosedModel):
    trainer_family: Literal["sentence-transformers-5"]
    loss_family: Literal["MultipleNegativesRankingLoss"]
    optimizer_steps: Literal[120]
    selected_validation_step: int = Field(ge=20, le=120)
    train_loss: float = Field(ge=0.0)
    selected_validation_loss: float = Field(ge=0.0)
    trainable_parameters: int = Field(gt=0)
    total_parameters: int = Field(gt=0)
    peak_gpu_memory_allocated_bytes: int = Field(gt=0)
    peak_gpu_memory_reserved_bytes: int = Field(gt=0)
    gpu_memory_allocated_after_cleanup_bytes: int = Field(ge=0)


class EvaluationOutcome(_ClosedModel):
    primary_metric: Literal["mrr"]
    baseline_recall_at_k: float = Field(ge=0.0, le=1.0)
    candidate_recall_at_k: float = Field(ge=0.0, le=1.0)
    baseline_mrr: float = Field(ge=0.0, le=1.0)
    candidate_mrr: float = Field(ge=0.0, le=1.0)
    baseline_ndcg_at_k: float = Field(ge=0.0, le=1.0)
    candidate_ndcg_at_k: float = Field(ge=0.0, le=1.0)
    primary_improvement: float
    baseline_p95_latency_ms: float = Field(gt=0.0)
    candidate_p95_latency_ms: float = Field(gt=0.0)
    candidate_latency_ratio: float = Field(gt=0.0)
    hard_gate_results: dict[str, bool]
    evaluation_peak_gpu_memory_reserved_bytes: int = Field(gt=0)


class RejectionGovernanceGates(_ClosedModel):
    source_integrity: Literal[True] = True
    fixed_image_model_and_data: Literal[True] = True
    actual_gpu_training: Literal[True] = True
    frozen_gold_evaluated_once: Literal[True] = True
    retired_v1_gold_not_reused: Literal[True] = True
    formal_gate_failure_preserved: Literal[True] = True
    failed_candidate_not_registered: Literal[True] = True
    failed_candidate_not_runtime_activated: Literal[True] = True
    detailed_candidate_and_observations_retained: Literal[True] = True


class EmbeddingRecoveryRejectionAcceptance(_ClosedModel):
    schema_version: Literal["embedding-recovery-rejection-acceptance/v1"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal["EMBEDDING_RECOVERY_REJECTION_VERIFIED"] = STATUS
    decision: Literal["RETAIN_AS_AUDIT_HISTORY_DO_NOT_DEPLOY"] = (
        "RETAIN_AS_AUDIT_HISTORY_DO_NOT_DEPLOY"
    )
    run_id: str = Field(pattern=r"^embedding-recovery-[0-9a-f]{20}$")
    image_digest: Literal[
        "sha256:7bcd50262cf62f33f8a3728a53f49ee36cbcc206f2bc0970776442f0b105effe"
    ] = "sha256:7bcd50262cf62f33f8a3728a53f49ee36cbcc206f2bc0970776442f0b105effe"
    source: FileBinding
    base_model_weight: FileBinding
    training_manifest: FileBinding
    final_gold_manifest: FileBinding
    rejection: FileBinding
    gold_retirement: FileBinding
    rejected_candidate_bundle: DirectoryBinding
    formal_evaluation: FileBinding
    observations: FileBinding
    training: TrainingOutcome
    evaluation: EvaluationOutcome
    failed_gates: tuple[str, ...] = Field(min_length=1)
    actual_gpu_training: Literal[True] = True
    independent_frozen_gold_evaluation_count: Literal[1] = 1
    same_gold_reuse_permitted: Literal[False] = False
    candidate_accepted: Literal[False] = False
    formal_model_release_created: Literal[False] = False
    runtime_eligible: Literal[False] = False
    governance_gates: RejectionGovernanceGates
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_embedding_recovery_rejection_acceptance(
    repo_root: Path,
    *,
    rejection_path: Path,
    retirement_path: Path,
) -> EmbeddingRecoveryRejectionAcceptance:
    root = repo_root.resolve(strict=True)
    rejection_file = _inside_file(root, rejection_path)
    retirement_file = _inside_file(root, retirement_path)
    rejection = _load_object(rejection_file)
    retirement = _load_object(retirement_file)
    _verify_rejection_document(root, rejection)
    _verify_retirement_document(rejection, retirement)

    run_id = str(rejection["run_id"])
    source = _inside_file(root, Path(str(rejection["source_path"])))
    base_model_weight = _inside_file(root, BASE_MODEL_WEIGHT_RELATIVE)
    training_manifest = _inside_file(root, TRAINING_MANIFEST_RELATIVE)
    gold_manifest = _inside_file(root, GOLD_MANIFEST_RELATIVE)
    if (
        _file_sha256(source) != rejection["source_sha256"]
        or _file_sha256(base_model_weight) != MODEL_WEIGHT_SHA256
        or _file_sha256(training_manifest) != TRAINING_MANIFEST_SHA256
        or _file_sha256(gold_manifest) != GOLD_MANIFEST_SHA256
    ):
        raise EmbeddingRecoveryOutcomeError("Embedding v2 fixed source evidence changed")

    candidate_directory = _inside_directory(root, rejection_file.parent / "candidate")
    formal_file = _inside_file(root, rejection_file.parent / "formal-evaluation.json")
    observations_file = _inside_file(root, rejection_file.parent / "observations.json")
    candidate_training_file = _inside_file(
        root,
        candidate_directory / "industrial-ops-training-report.json",
    )
    training = TrainingOutcome.model_validate(rejection.get("training"))
    evaluation = EvaluationOutcome.model_validate(rejection.get("evaluation"))
    failed_gates = rejection.get("failures")
    if not _string_list(failed_gates):
        raise EmbeddingRecoveryOutcomeError("Embedding v2 failed gates are missing")
    if _load_object(candidate_training_file) != training.model_dump(mode="json"):
        raise EmbeddingRecoveryOutcomeError("Embedding v2 candidate training changed")
    _verify_formal_evaluation(formal_file, evaluation)
    _verify_observations(observations_file)
    if tuple(failed_gates) != _expected_failures(evaluation):
        raise EmbeddingRecoveryOutcomeError("Embedding v2 failed gates changed")

    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": STATUS,
        "decision": "RETAIN_AS_AUDIT_HISTORY_DO_NOT_DEPLOY",
        "run_id": run_id,
        "image_digest": EXPECTED_IMAGE_DIGEST,
        "source": _file_binding(root, source).model_dump(mode="json"),
        "base_model_weight": _file_binding(root, base_model_weight).model_dump(mode="json"),
        "training_manifest": _file_binding(root, training_manifest).model_dump(mode="json"),
        "final_gold_manifest": _file_binding(root, gold_manifest).model_dump(mode="json"),
        "rejection": _file_binding(root, rejection_file).model_dump(mode="json"),
        "gold_retirement": _file_binding(root, retirement_file).model_dump(mode="json"),
        "rejected_candidate_bundle": _directory_binding(root, candidate_directory).model_dump(
            mode="json"
        ),
        "formal_evaluation": _file_binding(root, formal_file).model_dump(mode="json"),
        "observations": _file_binding(root, observations_file).model_dump(mode="json"),
        "training": training.model_dump(mode="json"),
        "evaluation": evaluation.model_dump(mode="json"),
        "failed_gates": failed_gates,
        "actual_gpu_training": True,
        "independent_frozen_gold_evaluation_count": 1,
        "same_gold_reuse_permitted": False,
        "candidate_accepted": False,
        "formal_model_release_created": False,
        "runtime_eligible": False,
        "governance_gates": RejectionGovernanceGates().model_dump(mode="json"),
    }
    draft = EmbeddingRecoveryRejectionAcceptance.model_validate(
        {**unsigned, "evidence_chain_sha256": "0" * 64}
    )
    normalized = draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    return draft.model_copy(update={"evidence_chain_sha256": _digest(normalized)})


def write_embedding_recovery_rejection_acceptance(
    report: EmbeddingRecoveryRejectionAcceptance,
    output_path: Path,
) -> Path:
    target = output_path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(_canonical(report.model_dump(mode="json")) + b"\n")
    os.replace(temporary, target)
    return target


def verify_embedding_recovery_rejection_acceptance(
    repo_root: Path,
    acceptance_path: Path | None = None,
) -> EmbeddingRecoveryRejectionAcceptance:
    root = repo_root.resolve(strict=True)
    report_path = _inside_file(root, acceptance_path or OUTPUT_RELATIVE)
    try:
        report = EmbeddingRecoveryRejectionAcceptance.model_validate_json(report_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise EmbeddingRecoveryOutcomeError("Embedding v2 rejection acceptance is invalid") from exc
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != _digest(unsigned):
        raise EmbeddingRecoveryOutcomeError("Embedding v2 rejection acceptance changed")
    rebuilt = build_embedding_recovery_rejection_acceptance(
        root,
        rejection_path=Path(report.rejection.path),
        retirement_path=Path(report.gold_retirement.path),
    )
    if _stable_projection(report) != _stable_projection(rebuilt):
        raise EmbeddingRecoveryOutcomeError("Embedding v2 rejection sources changed")
    return report


def _verify_rejection_document(root: Path, rejection: dict[str, Any]) -> None:
    unsigned = dict(rejection)
    chain = unsigned.pop("evidence_chain_sha256", None)
    training = rejection.get("training")
    evaluation = rejection.get("evaluation")
    source_path = rejection.get("source_path")
    if (
        rejection.get("schema_version") != "enterprise-embedding-recovery-rejection/v2"
        or rejection.get("classification") != CLASSIFICATION
        or rejection.get("outcome") != "REJECTED"
        or rejection.get("reuse_permitted") is not False
        or rejection.get("image") != "industrial-ops/m4-training-worker:local"
        or rejection.get("image_digest") != EXPECTED_IMAGE_DIGEST
        or rejection.get("formal_model_release_created") is not False
        or rejection.get("runtime_eligible") is not False
        or not isinstance(source_path, str)
        or not isinstance(rejection.get("source_sha256"), str)
        or not isinstance(training, dict)
        or not isinstance(evaluation, dict)
        or not isinstance(chain, str)
        or chain != _source_digest(unsigned)
    ):
        raise EmbeddingRecoveryOutcomeError("Embedding v2 rejection is invalid")
    source = _inside_file(root, Path(source_path))
    if _file_sha256(source) != rejection["source_sha256"]:
        raise EmbeddingRecoveryOutcomeError("Embedding v2 rejection source changed")


def _verify_retirement_document(
    rejection: dict[str, Any],
    retirement: dict[str, Any],
) -> None:
    unsigned = dict(retirement)
    chain = unsigned.pop("evidence_chain_sha256", None)
    if (
        retirement.get("schema_version") != "embedding-gold-retirement/v2"
        or retirement.get("suite_id") != "industrial-embedding-final-gold-v2"
        or retirement.get("run_id") != rejection.get("run_id")
        or retirement.get("outcome") != "REJECTED"
        or retirement.get("reuse_permitted") is not False
        or retirement.get("retired_v1_suite_reused") is not False
        or not isinstance(chain, str)
        or chain != _source_digest(unsigned)
    ):
        raise EmbeddingRecoveryOutcomeError("Embedding v2 Gold retirement is invalid")


def _verify_formal_evaluation(
    path: Path,
    evaluation: EvaluationOutcome,
) -> None:
    document = _load_object(path)
    report = document.get("report")
    if (
        document.get("schema_version") != "enterprise-embedding-recovery-formal-evaluation/v2"
        or document.get("classification") != CLASSIFICATION
        or document.get("suite_id") != "industrial-embedding-final-gold-v2"
        or document.get("single_final_evaluation") is not True
        or not isinstance(report, dict)
    ):
        raise EmbeddingRecoveryOutcomeError("Embedding v2 formal evaluation is invalid")
    evidence = report.get("evidence")
    case_metrics = report.get("case_metrics")
    if (
        not isinstance(evidence, dict)
        or evidence.get("hard_gate_results") != evaluation.hard_gate_results
        or not _same_number(
            evidence.get("baseline_p95_latency_ms"),
            evaluation.baseline_p95_latency_ms,
        )
        or not _same_number(
            evidence.get("candidate_p95_latency_ms"),
            evaluation.candidate_p95_latency_ms,
        )
        or not isinstance(case_metrics, list)
        or len(case_metrics) != 12
    ):
        raise EmbeddingRecoveryOutcomeError("Embedding v2 formal evaluation summary changed")
    retrieval = evidence.get("slice_metrics")
    retrieval = retrieval.get("retrieval") if isinstance(retrieval, dict) else None
    if not isinstance(retrieval, dict):
        raise EmbeddingRecoveryOutcomeError("Embedding v2 retrieval metrics are missing")
    expected = {
        "mrr": (evaluation.baseline_mrr, evaluation.candidate_mrr),
        "ndcg_at_k": (
            evaluation.baseline_ndcg_at_k,
            evaluation.candidate_ndcg_at_k,
        ),
        "recall_at_k": (
            evaluation.baseline_recall_at_k,
            evaluation.candidate_recall_at_k,
        ),
    }
    for name, (baseline, candidate) in expected.items():
        metric = retrieval.get(name)
        if (
            not isinstance(metric, dict)
            or not _same_number(metric.get("baseline"), baseline)
            or not _same_number(metric.get("candidate"), candidate)
        ):
            raise EmbeddingRecoveryOutcomeError(f"Embedding v2 {name} evidence changed")


def _verify_observations(path: Path) -> None:
    document = _load_object(path)
    baseline = document.get("baseline")
    candidate = document.get("candidate")
    if (
        document.get("schema_version") != "enterprise-embedding-recovery-observations/v2"
        or document.get("classification") != CLASSIFICATION
        or document.get("suite_id") != "industrial-embedding-final-gold-v2"
        or not isinstance(baseline, list)
        or not isinstance(candidate, list)
        or len(baseline) != 12
        or len(candidate) != 12
    ):
        raise EmbeddingRecoveryOutcomeError("Embedding v2 observations are invalid")
    baseline_ids = {item.get("case_id") for item in baseline if isinstance(item, dict)}
    candidate_ids = {item.get("case_id") for item in candidate if isinstance(item, dict)}
    if (
        len(baseline_ids) != 12
        or baseline_ids != candidate_ids
        or any(not _valid_observation(item) for item in (*baseline, *candidate))
    ):
        raise EmbeddingRecoveryOutcomeError("Embedding v2 observations changed")


def _valid_observation(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    capabilities = value.get("capabilities")
    ranked_ids = value.get("ranked_document_ids")
    runtime = value.get("runtime_evidence")
    return (
        isinstance(capabilities, list)
        and set(capabilities)
        == {
            "cross_tenant_isolation",
            "data_governance",
            "retrieval_index_compatibility",
        }
        and isinstance(ranked_ids, list)
        and len(ranked_ids) == 3
        and all(
            isinstance(item, str) and not item.startswith("tenant-rival") for item in ranked_ids
        )
        and isinstance(runtime, dict)
        and runtime.get("allowed_document_count") == 20
        and runtime.get("forbidden_document_count") == 2
        and runtime.get("retrieval_method") == "EMBEDDING"
    )


def _expected_failures(evaluation: EvaluationOutcome) -> tuple[str, ...]:
    if set(evaluation.hard_gate_results) != EXPECTED_RETRIEVAL_GATES:
        raise EmbeddingRecoveryOutcomeError("Embedding v2 hard-gate contract changed")
    failures = [
        f"embedding_{name}"
        for name, passed in sorted(evaluation.hard_gate_results.items())
        if passed is not True
    ]
    if evaluation.primary_improvement < 0.05:
        failures.append("embedding_primary_metric_improvement")
    if evaluation.candidate_recall_at_k < 0.85:
        failures.append("embedding_recall_at_k")
    if evaluation.candidate_mrr < 0.85:
        failures.append("embedding_mrr")
    if evaluation.candidate_latency_ratio > 4.0:
        failures.append("embedding_latency_budget")
    if not failures:
        raise EmbeddingRecoveryOutcomeError(
            "Embedding v2 rejection has no reproducible gate failure"
        )
    return tuple(failures)


def _file_binding(root: Path, path: Path) -> FileBinding:
    resolved = _inside_file(root, path)
    return FileBinding(
        path=resolved.relative_to(root).as_posix(),
        size_bytes=resolved.stat().st_size,
        sha256=_file_sha256(resolved),
    )


def _directory_binding(root: Path, path: Path) -> DirectoryBinding:
    resolved = _inside_directory(root, path)
    manifest: list[dict[str, Any]] = []
    for candidate in sorted(resolved.rglob("*")):
        if candidate.is_symlink():
            raise EmbeddingRecoveryOutcomeError("Embedding v2 candidate contains a symlink")
        if candidate.is_file():
            manifest.append(
                {
                    "path": candidate.relative_to(resolved).as_posix(),
                    "size_bytes": candidate.stat().st_size,
                    "sha256": _file_sha256(candidate),
                }
            )
    if not manifest:
        raise EmbeddingRecoveryOutcomeError("Embedding v2 candidate is empty")
    return DirectoryBinding(
        path=resolved.relative_to(root).as_posix(),
        file_count=len(manifest),
        size_bytes=sum(int(item["size_bytes"]) for item in manifest),
        manifest_sha256=_digest(manifest),
    )


def _stable_projection(
    report: EmbeddingRecoveryRejectionAcceptance,
) -> dict[str, Any]:
    return report.model_dump(
        mode="json",
        exclude={"generated_at", "evidence_chain_sha256"},
    )


def _string_list(value: Any) -> TypeGuard[list[str]]:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item) for item in value)
    )


def _same_number(value: Any, expected: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and abs(float(value) - expected) <= 1e-12
    )


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EmbeddingRecoveryOutcomeError(f"invalid Embedding v2 JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise EmbeddingRecoveryOutcomeError(f"invalid Embedding v2 JSON: {path.name}")
    return value


def _inside_file(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise EmbeddingRecoveryOutcomeError("Embedding v2 file escapes repository") from exc
    if not resolved.is_file():
        raise EmbeddingRecoveryOutcomeError("Embedding v2 evidence is not a file")
    return resolved


def _inside_directory(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise EmbeddingRecoveryOutcomeError("Embedding v2 directory escapes repository") from exc
    if not resolved.is_dir():
        raise EmbeddingRecoveryOutcomeError("Embedding v2 evidence is not a directory")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _source_digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
