"""Bind rejected and accepted actual-GPU model experiments into one receipt."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, TypeGuard

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.simulation.embedding_calibrated_value_lab import (
    EmbeddingCalibratedValueReport,
    verify_calibrated_embedding_value,
)
from industrial_ops_agent.simulation.embedding_recovery_outcome import (
    verify_embedding_recovery_rejection_acceptance,
)
from industrial_ops_agent.simulation.reranker_calibrated_value_lab import (
    RerankerCalibratedValueReport,
    verify_calibrated_reranker_value,
)
from industrial_ops_agent.simulation.reranker_grounded_verifier_erratum import (
    verify_original_with_v4_observation_schema,
    verify_reranker_grounded_verifier_erratum,
)
from industrial_ops_agent.simulation.reranker_recovery_verifier_erratum import (
    verify_original_with_json_normalization,
    verify_reranker_recovery_verifier_erratum,
)
from industrial_ops_agent.simulation.tts_recovery_value_lab import (
    FileEvidence as TtsFileEvidence,
)
from industrial_ops_agent.simulation.tts_recovery_value_lab import (
    verify_tts_recovery_rejection,
)
from industrial_ops_agent.simulation.tts_strong_asr_value_lab import (
    TtsCalibratedValueReport,
    verify_calibrated_tts_value,
)

SCHEMA_VERSION: Literal["actual-gpu-model-experiment-outcomes/v3"] = (
    "actual-gpu-model-experiment-outcomes/v3"
)
CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = (
    "LOCAL_STAGING_PROJECT_AUTHORIZED"
)
STATUS: Literal["ACTUAL_GPU_MODEL_EXPERIMENT_OUTCOMES_RECORDED"] = (
    "ACTUAL_GPU_MODEL_EXPERIMENT_OUTCOMES_RECORDED"
)
OUTPUT_RELATIVE = Path("artifacts/m7-model-experiment-outcomes/acceptance.json")


class ModelExperimentOutcomesError(RuntimeError):
    """Experiment evidence is missing, changed, or internally inconsistent."""


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


class RetrievalMetricOutcome(_ClosedModel):
    primary_metric: Literal["mrr", "ndcg_at_k"]
    baseline_primary: float = Field(ge=0.0, le=1.0)
    candidate_primary: float = Field(ge=0.0, le=1.0)
    primary_improvement: float
    candidate_recall_at_k: float = Field(ge=0.0, le=1.0)
    candidate_mrr: float = Field(ge=0.0, le=1.0)
    candidate_ndcg_at_k: float = Field(ge=0.0, le=1.0)


class TtsMetricOutcome(_ClosedModel):
    baseline_voice_similarity: float = Field(ge=-1.0, le=1.0)
    candidate_voice_similarity: float = Field(ge=-1.0, le=1.0)
    voice_similarity_improvement: float
    candidate_wer: float = Field(ge=0.0)
    candidate_safety_phrase_completeness: float = Field(ge=0.0, le=1.0)
    candidate_terminology_recall: float = Field(ge=0.0, le=1.0)
    candidate_intelligibility: float = Field(ge=0.0, le=1.0)
    candidate_latency_ratio: float = Field(gt=0.0)


class RejectedExperimentOutcome(_ClosedModel):
    method: Literal["TTS", "EMBEDDING", "RERANKER"]
    run_id: str = Field(
        pattern=(
            r"^(tts|tts-recovery|retrieval|embedding-recovery|reranker-recovery|"
            r"reranker-grounded)-"
            r"[0-9a-f]{20}$"
        )
    )
    result: Literal["REJECTED_BY_FORMAL_GATES"] = "REJECTED_BY_FORMAL_GATES"
    actual_gpu_training: Literal[True] = True
    independent_frozen_gold_evaluation_count: Literal[1] = 1
    same_gold_reuse_permitted: Literal[False] = False
    candidate_accepted: Literal[False] = False
    formal_model_release_created: Literal[False] = False
    runtime_eligible: Literal[False] = False
    retained_detailed_evidence: bool
    failed_gates: tuple[str, ...] = Field(min_length=1)
    rejection: FileBinding
    gold_retirement: FileBinding
    rejected_candidate_bundle: DirectoryBinding | None = None
    formal_evaluation: FileBinding | None = None
    observations: FileBinding | None = None
    retrieval_metrics: RetrievalMetricOutcome | None = None
    governance_acceptance: FileBinding | None = None


class AcceptedRerankerOutcome(_ClosedModel):
    method: Literal["RERANKER"] = "RERANKER"
    run_id: str = Field(pattern=r"^reranker-calibrated-[0-9a-f]{20}$")
    result: Literal["ACCEPTED_BY_FORMAL_GATES"] = "ACCEPTED_BY_FORMAL_GATES"
    actual_gpu_training: Literal[True] = True
    actual_gpu_inference: Literal[True] = True
    current_run_training_performed: Literal[False] = False
    independent_frozen_gold_evaluation_count: Literal[1] = 1
    same_gold_reuse_permitted: Literal[False] = False
    candidate_accepted: Literal[True] = True
    formal_model_release_created: Literal[False] = False
    runtime_eligible: Literal[False] = False
    release_draft_eligible: Literal[True] = True
    retained_detailed_evidence: Literal[True] = True
    failed_gates: tuple[str, ...] = Field(default=(), max_length=0)
    candidate_bundle: DirectoryBinding
    formal_evaluation: FileBinding
    observations: FileBinding
    retrieval_metrics: RetrievalMetricOutcome
    governance_acceptance: FileBinding
    predecessor_governance: FileBinding
    gold_retirement: FileBinding


class AcceptedTtsOutcome(_ClosedModel):
    method: Literal["TTS"] = "TTS"
    run_id: str = Field(pattern=r"^tts-strong-asr-[0-9a-f]{20}$")
    result: Literal["ACCEPTED_BY_FORMAL_GATES"] = "ACCEPTED_BY_FORMAL_GATES"
    actual_gpu_training: Literal[True] = True
    actual_gpu_inference: Literal[True] = True
    current_run_training_performed: Literal[False] = False
    independent_frozen_gold_evaluation_count: Literal[1] = 1
    same_gold_reuse_permitted: Literal[False] = False
    candidate_accepted: Literal[True] = True
    formal_model_release_created: Literal[False] = False
    runtime_eligible: Literal[False] = False
    release_draft_eligible: Literal[True] = True
    retained_detailed_evidence: Literal[True] = True
    failed_gates: tuple[str, ...] = Field(default=(), max_length=0)
    candidate_bundle: DirectoryBinding
    runtime_policy: FileBinding
    asr_verifier: DirectoryBinding
    formal_evaluation: FileBinding
    observations: FileBinding
    tts_metrics: TtsMetricOutcome
    governance_acceptance: FileBinding
    predecessor_rejection: FileBinding
    predecessor_gold_retirement: FileBinding
    gold_retirement: FileBinding


class AcceptedEmbeddingOutcome(_ClosedModel):
    method: Literal["EMBEDDING"] = "EMBEDDING"
    run_id: str = Field(pattern=r"^embedding-calibrated-[0-9a-f]{20}$")
    result: Literal["ACCEPTED_BY_FORMAL_GATES"] = "ACCEPTED_BY_FORMAL_GATES"
    actual_gpu_training: Literal[True] = True
    actual_gpu_inference: Literal[True] = True
    current_run_training_performed: Literal[False] = False
    independent_frozen_gold_evaluation_count: Literal[1] = 1
    same_gold_reuse_permitted: Literal[False] = False
    candidate_accepted: Literal[True] = True
    formal_model_release_created: Literal[False] = False
    runtime_eligible: Literal[False] = False
    release_draft_eligible: Literal[True] = True
    retained_detailed_evidence: Literal[True] = True
    failed_gates: tuple[str, ...] = Field(default=(), max_length=0)
    candidate_bundle: DirectoryBinding
    formal_evaluation: FileBinding
    observations: FileBinding
    retrieval_metrics: RetrievalMetricOutcome
    governance_acceptance: FileBinding
    predecessor_governance: FileBinding
    predecessor_rejection: FileBinding
    predecessor_gold_retirement: FileBinding
    historical_retrieval_rejection: FileBinding
    historical_retrieval_gold_retirement: FileBinding
    gold_retirement: FileBinding


ExperimentOutcome = (
    RejectedExperimentOutcome
    | AcceptedTtsOutcome
    | AcceptedEmbeddingOutcome
    | AcceptedRerankerOutcome
)


class OutcomeGovernanceGates(_ClosedModel):
    source_integrity: Literal[True] = True
    actual_gpu_execution: Literal[True] = True
    independent_gold_used_once: Literal[True] = True
    failed_candidates_rejected: Literal[True] = True
    failed_candidates_not_registered: Literal[True] = True
    failed_candidates_not_runtime_activated: Literal[True] = True
    rejection_evidence_retained: Literal[True] = True
    passed_candidates_formally_accepted: Literal[True] = True
    passed_candidates_release_draft_only: Literal[True] = True
    passed_candidates_not_runtime_activated: Literal[True] = True


class ModelExperimentOutcomesReport(_ClosedModel):
    schema_version: Literal["actual-gpu-model-experiment-outcomes/v3"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal["ACTUAL_GPU_MODEL_EXPERIMENT_OUTCOMES_RECORDED"] = STATUS
    decision: Literal[
        "RETAIN_AS_AUDIT_HISTORY_DO_NOT_DEPLOY",
        "RETAIN_REJECTIONS_AND_ADVANCE_ACCEPTED_CANDIDATES_TO_RELEASE_DRAFT",
    ]
    rejected_method_count: int = Field(ge=0, le=3)
    active_candidate_count: int = Field(ge=0, le=3)
    verified_enterprise_value_methods: tuple[
        Literal[
            "TTS_ENTERPRISE_VALUE",
            "EMBEDDING_ENTERPRISE_VALUE",
            "RERANKER_ENTERPRISE_VALUE",
        ],
        ...,
    ]
    intentionally_unverified_methods: tuple[
        Literal[
            "TTS_ENTERPRISE_VALUE",
            "EMBEDDING_ENTERPRISE_VALUE",
            "RERANKER_ENTERPRISE_VALUE",
        ],
        ...,
    ] = Field(max_length=3)
    outcomes: tuple[ExperimentOutcome, ...] = Field(min_length=3, max_length=3)
    governance_gates: OutcomeGovernanceGates
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_model_experiment_outcomes(
    repo_root: Path,
    *,
    tts_rejection_path: Path,
    tts_retirement_path: Path,
    retrieval_rejection_path: Path,
    retrieval_retirement_path: Path,
    tts_strong_asr_acceptance_path: Path | None = None,
    embedding_recovery_acceptance_path: Path | None = None,
    embedding_calibrated_acceptance_path: Path | None = None,
    reranker_recovery_erratum_path: Path | None = None,
    reranker_calibrated_acceptance_path: Path | None = None,
) -> ModelExperimentOutcomesReport:
    """Validate model experiments and produce their aggregate audit receipt."""

    root = repo_root.resolve(strict=True)
    tts_rejection = _inside_file(root, tts_rejection_path)
    tts_retirement = _inside_file(root, tts_retirement_path)
    retrieval_rejection = _inside_file(root, retrieval_rejection_path)
    retrieval_retirement = _inside_file(root, retrieval_retirement_path)
    historical_tts = _tts_outcome(root, tts_rejection, tts_retirement)
    tts_outcome: ExperimentOutcome = (
        historical_tts
        if tts_strong_asr_acceptance_path is None
        else _tts_strong_asr_outcome(
            root,
            _inside_file(root, tts_strong_asr_acceptance_path),
            historical_tts,
        )
    )
    retrieval_outcomes = _retrieval_outcomes(
        root,
        retrieval_rejection,
        retrieval_retirement,
    )
    embedding_outcome: ExperimentOutcome = (
        retrieval_outcomes[0]
        if embedding_recovery_acceptance_path is None
        else _embedding_recovery_outcome(
            root,
            _inside_file(root, embedding_recovery_acceptance_path),
        )
    )
    if embedding_calibrated_acceptance_path is not None:
        embedding_outcome = _embedding_calibrated_outcome(
            root,
            _inside_file(root, embedding_calibrated_acceptance_path),
            retrieval_rejection,
            retrieval_retirement,
        )
    if (
        reranker_recovery_erratum_path is not None
        and reranker_calibrated_acceptance_path is not None
    ):
        raise ModelExperimentOutcomesError(
            "Reranker rejection and accepted candidate cannot both be authoritative"
        )
    if reranker_calibrated_acceptance_path is not None:
        reranker_outcome: ExperimentOutcome = _reranker_calibrated_outcome(
            root,
            _inside_file(root, reranker_calibrated_acceptance_path),
            retrieval_rejection,
            retrieval_retirement,
        )
    elif reranker_recovery_erratum_path is not None:
        reranker_outcome = _reranker_recovery_outcome(
            root,
            _inside_file(root, reranker_recovery_erratum_path),
        )
    else:
        reranker_outcome = retrieval_outcomes[1]
    outcomes = (
        tts_outcome,
        embedding_outcome,
        reranker_outcome,
    )
    rejected_count = sum(
        isinstance(item, RejectedExperimentOutcome) for item in outcomes
    )
    accepted = tuple(
        item for item in outcomes if not isinstance(item, RejectedExperimentOutcome)
    )
    verified_methods = [f"{item.method}_ENTERPRISE_VALUE" for item in accepted]
    unverified_methods = [
        f"{item.method}_ENTERPRISE_VALUE"
        for item in outcomes
        if isinstance(item, RejectedExperimentOutcome)
    ]
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "external_enterprise_production_claim": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": STATUS,
        "decision": (
            "RETAIN_REJECTIONS_AND_ADVANCE_ACCEPTED_CANDIDATES_TO_RELEASE_DRAFT"
            if accepted
            else "RETAIN_AS_AUDIT_HISTORY_DO_NOT_DEPLOY"
        ),
        "rejected_method_count": rejected_count,
        "active_candidate_count": len(accepted),
        "verified_enterprise_value_methods": verified_methods,
        "intentionally_unverified_methods": unverified_methods,
        "outcomes": [item.model_dump(mode="json") for item in outcomes],
        "governance_gates": OutcomeGovernanceGates().model_dump(mode="json"),
    }
    draft = ModelExperimentOutcomesReport.model_validate(
        {**unsigned, "evidence_chain_sha256": "0" * 64}
    )
    normalized = draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    return draft.model_copy(update={"evidence_chain_sha256": _digest(normalized)})


def write_model_experiment_outcomes(
    report: ModelExperimentOutcomesReport,
    output_path: Path,
) -> Path:
    target = output_path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_bytes(_canonical(report.model_dump(mode="json")) + b"\n")
    temporary.replace(target)
    return target


def verify_model_experiment_outcomes(
    repo_root: Path,
    acceptance_path: Path | None = None,
) -> ModelExperimentOutcomesReport:
    root = repo_root.resolve(strict=True)
    report_path = _inside_file(root, acceptance_path or OUTPUT_RELATIVE)
    try:
        report = ModelExperimentOutcomesReport.model_validate_json(
            report_path.read_bytes()
        )
    except (OSError, ValueError) as exc:
        raise ModelExperimentOutcomesError(
            "model experiment outcome receipt is invalid"
        ) from exc
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != _digest(unsigned):
        raise ModelExperimentOutcomesError("model experiment outcome receipt changed")
    by_method = {item.method: item for item in report.outcomes}
    if set(by_method) != {"TTS", "EMBEDDING", "RERANKER"}:
        raise ModelExperimentOutcomesError("model experiment outcome methods changed")
    _verify_report_summary(report)
    tts_outcome = by_method["TTS"]
    embedding_outcome = by_method["EMBEDDING"]
    reranker_outcome = by_method["RERANKER"]
    tts_strong_asr_acceptance_path: Path | None = None
    if isinstance(tts_outcome, AcceptedTtsOutcome):
        tts_rejection_path = Path(tts_outcome.predecessor_rejection.path)
        tts_retirement_path = Path(tts_outcome.predecessor_gold_retirement.path)
        tts_strong_asr_acceptance_path = Path(
            tts_outcome.governance_acceptance.path
        )
    elif isinstance(tts_outcome, RejectedExperimentOutcome):
        if tts_outcome.method != "TTS":
            raise ModelExperimentOutcomesError("TTS outcome type changed")
        tts_rejection_path = Path(tts_outcome.rejection.path)
        tts_retirement_path = Path(tts_outcome.gold_retirement.path)
    else:
        raise ModelExperimentOutcomesError("TTS outcome type changed")

    embedding_recovery_acceptance_path: Path | None = None
    embedding_calibrated_acceptance_path: Path | None = None
    if isinstance(embedding_outcome, AcceptedEmbeddingOutcome):
        retrieval_rejection_path = Path(
            embedding_outcome.historical_retrieval_rejection.path
        )
        retrieval_retirement_path = Path(
            embedding_outcome.historical_retrieval_gold_retirement.path
        )
        embedding_recovery_acceptance_path = Path(
            embedding_outcome.predecessor_governance.path
        )
        embedding_calibrated_acceptance_path = Path(
            embedding_outcome.governance_acceptance.path
        )
    elif isinstance(embedding_outcome, RejectedExperimentOutcome):
        if embedding_outcome.method != "EMBEDDING":
            raise ModelExperimentOutcomesError("Embedding outcome type changed")
        embedding_recovery_acceptance_path = (
            Path(embedding_outcome.governance_acceptance.path)
            if embedding_outcome.governance_acceptance is not None
            else None
        )
    else:
        raise ModelExperimentOutcomesError("Embedding outcome type changed")

    reranker_calibrated_acceptance_path: Path | None = None
    if isinstance(reranker_outcome, AcceptedRerankerOutcome):
        reranker_recovery_erratum_path = None
        reranker_calibrated_acceptance_path = Path(
            reranker_outcome.governance_acceptance.path
        )
        calibrated = verify_calibrated_reranker_value(
            root,
            reranker_calibrated_acceptance_path,
        )
        retrieval_rejection_path, retrieval_retirement_path = (
            _calibrated_predecessor_retrieval_sources(root, calibrated)
        )
    elif not isinstance(reranker_outcome, RejectedExperimentOutcome):
        raise ModelExperimentOutcomesError("Reranker outcome type changed")
    elif reranker_outcome.governance_acceptance is None:
        retrieval_rejection_path = Path(reranker_outcome.rejection.path)
        retrieval_retirement_path = Path(reranker_outcome.gold_retirement.path)
        reranker_recovery_erratum_path = None
    else:
        reranker_erratum_path = Path(reranker_outcome.governance_acceptance.path)
        erratum = verify_reranker_recovery_verifier_erratum(root, reranker_erratum_path)
        original = verify_original_with_json_normalization(
            root,
            Path(erratum.original_outcome.path),
        )
        retrieval_rejection_path = Path(original.predecessor.rejection.path)
        retrieval_retirement_path = Path(original.predecessor.gold_retirement.path)
        reranker_recovery_erratum_path = reranker_erratum_path
    rebuilt = build_model_experiment_outcomes(
        root,
        tts_rejection_path=tts_rejection_path,
        tts_retirement_path=tts_retirement_path,
        tts_strong_asr_acceptance_path=tts_strong_asr_acceptance_path,
        retrieval_rejection_path=retrieval_rejection_path,
        retrieval_retirement_path=retrieval_retirement_path,
        embedding_recovery_acceptance_path=embedding_recovery_acceptance_path,
        embedding_calibrated_acceptance_path=embedding_calibrated_acceptance_path,
        reranker_recovery_erratum_path=reranker_recovery_erratum_path,
        reranker_calibrated_acceptance_path=reranker_calibrated_acceptance_path,
    )
    if _stable_projection(report) != _stable_projection(rebuilt):
        raise ModelExperimentOutcomesError("model experiment source evidence changed")
    return report


def _tts_outcome(
    root: Path,
    rejection_path: Path,
    retirement_path: Path,
) -> RejectedExperimentOutcome:
    rejection = _load_object(rejection_path)
    if rejection.get("schema_version") == "enterprise-tts-recovery-rejection/v2":
        return _tts_recovery_outcome(root, rejection_path, retirement_path)
    retirement = _load_object(retirement_path)
    inputs = rejection.get("governed_inputs")
    resources = rejection.get("resource_profile")
    failed = rejection.get("failed_hard_gates")
    if (
        rejection.get("schema_version") != "enterprise-tts-formal-rejection/v1"
        or rejection.get("classification") != CLASSIFICATION
        or rejection.get("production_claim") is not False
        or rejection.get("status") != "TTS_ACTUAL_GPU_CANDIDATE_REJECTED"
        or rejection.get("decision") != "DO_NOT_REGISTER_DEPLOY_OR_REUSE_GOLD"
        or rejection.get("actual_gpu_training") is not True
        or rejection.get("candidate_accepted") is not False
        or rejection.get("formal_model_release_created") is not False
        or rejection.get("runtime_eligible") is not False
        or rejection.get("same_gold_reuse_permitted") is not False
        or not isinstance(inputs, dict)
        or inputs.get("gold_evaluation_pass_count") != 1
        or not isinstance(resources, dict)
        or resources.get("gpu_count") != 1
        or resources.get("dataloader_workers") != 0
        or resources.get("network_disabled") is not True
        or resources.get("read_only_root_filesystem") is not True
        or not _string_list(failed)
    ):
        raise ModelExperimentOutcomesError("TTS rejection evidence is invalid")
    run_id = rejection.get("run_id")
    if (
        retirement.get("schema_version") != "enterprise-tts-gold-suite-retirement/v1"
        or retirement.get("classification") != CLASSIFICATION
        or retirement.get("status") != "RETIRED_AFTER_SINGLE_FINAL_EVALUATION"
        or retirement.get("failed_run_id") != run_id
        or retirement.get("failed_gates") != failed
        or retirement.get("candidate_accepted") is not False
        or retirement.get("model_release_created") is not False
        or retirement.get("runtime_eligible") is not False
        or retirement.get("reuse_permitted") is not False
    ):
        raise ModelExperimentOutcomesError("TTS Gold retirement evidence is invalid")
    return RejectedExperimentOutcome(
        method="TTS",
        run_id=str(run_id),
        retained_detailed_evidence=False,
        failed_gates=tuple(str(item) for item in failed),
        rejection=_file_binding(root, rejection_path),
        gold_retirement=_file_binding(root, retirement_path),
    )


def _tts_recovery_outcome(
    root: Path,
    rejection_path: Path,
    retirement_path: Path,
) -> RejectedExperimentOutcome:
    rejection = verify_tts_recovery_rejection(root, rejection_path)
    retirement = _load_object(retirement_path)
    failed = list(rejection.failed_hard_gates)
    if (
        retirement.get("schema_version") != "enterprise-tts-gold-suite-retirement/v2"
        or retirement.get("classification") != CLASSIFICATION
        or retirement.get("suite_id") != "industrial-safety-tts-final-gold-v3"
        or retirement.get("status") != "RETIRED_AFTER_REJECTED_FINAL_EVALUATION"
        or retirement.get("run_id") != rejection.run_id
        or retirement.get("outcome") != "REJECTED"
        or retirement.get("failed_hard_gates") != failed
        or retirement.get("candidate_accepted") is not False
        or retirement.get("model_release_created") is not False
        or retirement.get("runtime_eligible") is not False
        or retirement.get("reuse_permitted") is not False
        or retirement.get("outcome_path") != rejection_path.relative_to(root).as_posix()
        or retirement.get("outcome_sha256")
        != sha256(rejection_path.read_bytes()).hexdigest()
    ):
        raise ModelExperimentOutcomesError(
            "TTS recovery Gold retirement evidence is invalid"
        )
    formal = TtsFileEvidence.model_validate(rejection.evaluation.get("formal_report"))
    observations = TtsFileEvidence.model_validate(
        rejection.evaluation.get("observations")
    )
    return RejectedExperimentOutcome(
        method="TTS",
        run_id=rejection.run_id,
        retained_detailed_evidence=True,
        failed_gates=rejection.failed_hard_gates,
        rejection=_file_binding(root, rejection_path),
        gold_retirement=_file_binding(root, retirement_path),
        rejected_candidate_bundle=_directory_binding(
            root,
            Path(rejection.candidate.path),
        ),
        formal_evaluation=_file_binding(root, Path(formal.path)),
        observations=_file_binding(root, Path(observations.path)),
    )


def _tts_strong_asr_outcome(
    root: Path,
    acceptance_path: Path,
    predecessor: RejectedExperimentOutcome,
) -> AcceptedTtsOutcome:
    acceptance: TtsCalibratedValueReport = verify_calibrated_tts_value(
        root,
        acceptance_path,
    )
    if (
        acceptance.status != "TTS_STRONG_ASR_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
        or acceptance.decision
        != "TTS_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
        or acceptance.candidate_accepted is not True
        or acceptance.release_draft_eligible is not True
        or acceptance.failed_hard_gates
        or acceptance.runtime.actual_gpu_inference is not True
        or acceptance.runtime.predecessor_actual_gpu_training is not True
        or acceptance.runtime.current_run_model_training_performed is not False
        or acceptance.formal_model_release_created is not False
        or acceptance.runtime_eligible is not False
        or acceptance.same_gold_reuse_permitted is not False
        or acceptance.predecessor.rejection.path != predecessor.rejection.path
        or predecessor.method != "TTS"
    ):
        raise ModelExperimentOutcomesError(
            "accepted TTS enterprise-value evidence is invalid"
        )
    evaluation = acceptance.evaluation
    return AcceptedTtsOutcome(
        run_id=acceptance.run_id,
        candidate_bundle=_directory_binding(
            root,
            Path(acceptance.candidate.bundle.path),
        ),
        runtime_policy=_file_binding(
            root,
            Path(acceptance.candidate.runtime_policy_file.path),
        ),
        asr_verifier=_directory_binding(
            root,
            Path(acceptance.asr_verifier.bundle.path),
        ),
        formal_evaluation=_file_binding(
            root,
            Path(evaluation.formal_report.path),
        ),
        observations=_file_binding(root, Path(evaluation.observations.path)),
        tts_metrics=TtsMetricOutcome(
            baseline_voice_similarity=evaluation.baseline_voice_similarity,
            candidate_voice_similarity=evaluation.candidate_voice_similarity,
            voice_similarity_improvement=evaluation.voice_similarity_improvement,
            candidate_wer=evaluation.candidate_wer,
            candidate_safety_phrase_completeness=(
                evaluation.candidate_safety_phrase_completeness
            ),
            candidate_terminology_recall=evaluation.candidate_terminology_recall,
            candidate_intelligibility=evaluation.candidate_intelligibility,
            candidate_latency_ratio=evaluation.candidate_latency_ratio,
        ),
        governance_acceptance=_file_binding(root, acceptance_path),
        predecessor_rejection=predecessor.rejection,
        predecessor_gold_retirement=predecessor.gold_retirement,
        gold_retirement=_file_binding(
            root,
            Path(acceptance.gold_retirement_path),
        ),
    )


def _embedding_recovery_outcome(
    root: Path,
    acceptance_path: Path,
) -> RejectedExperimentOutcome:
    acceptance = verify_embedding_recovery_rejection_acceptance(
        root,
        acceptance_path,
    )
    evaluation = acceptance.evaluation
    return RejectedExperimentOutcome(
        method="EMBEDDING",
        run_id=acceptance.run_id,
        retained_detailed_evidence=True,
        failed_gates=acceptance.failed_gates,
        rejection=_file_binding(root, Path(acceptance.rejection.path)),
        gold_retirement=_file_binding(root, Path(acceptance.gold_retirement.path)),
        rejected_candidate_bundle=_directory_binding(
            root,
            Path(acceptance.rejected_candidate_bundle.path),
        ),
        formal_evaluation=_file_binding(
            root,
            Path(acceptance.formal_evaluation.path),
        ),
        observations=_file_binding(root, Path(acceptance.observations.path)),
        retrieval_metrics=RetrievalMetricOutcome(
            primary_metric="mrr",
            baseline_primary=evaluation.baseline_mrr,
            candidate_primary=evaluation.candidate_mrr,
            primary_improvement=evaluation.primary_improvement,
            candidate_recall_at_k=evaluation.candidate_recall_at_k,
            candidate_mrr=evaluation.candidate_mrr,
            candidate_ndcg_at_k=evaluation.candidate_ndcg_at_k,
        ),
        governance_acceptance=_file_binding(root, acceptance_path),
    )


def _embedding_calibrated_outcome(
    root: Path,
    acceptance_path: Path,
    retrieval_rejection_path: Path,
    retrieval_retirement_path: Path,
) -> AcceptedEmbeddingOutcome:
    acceptance: EmbeddingCalibratedValueReport = verify_calibrated_embedding_value(
        root,
        acceptance_path,
    )
    predecessor_path = _inside_file(
        root,
        Path(acceptance.predecessor.governance_acceptance.path),
    )
    predecessor = _embedding_recovery_outcome(root, predecessor_path)
    if (
        acceptance.status
        != "EMBEDDING_CALIBRATED_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
        or acceptance.decision
        != "EMBEDDING_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
        or acceptance.candidate_accepted is not True
        or acceptance.release_draft_eligible is not True
        or acceptance.failed_hard_gates
        or acceptance.runtime.actual_gpu_inference is not True
        or acceptance.runtime.source_candidate_actual_gpu_trained is not True
        or acceptance.runtime.current_run_model_training_performed is not False
        or acceptance.formal_model_release_created is not False
        or acceptance.runtime_eligible is not False
        or acceptance.same_gold_reuse_permitted is not False
    ):
        raise ModelExperimentOutcomesError(
            "accepted Embedding enterprise-value evidence is invalid"
        )
    evaluation = acceptance.evaluation
    return AcceptedEmbeddingOutcome(
        run_id=acceptance.run_id,
        candidate_bundle=_directory_binding(root, Path(acceptance.candidate.path)),
        formal_evaluation=_file_binding(
            root,
            Path(evaluation.formal_report.path),
        ),
        observations=_file_binding(root, Path(evaluation.observations.path)),
        retrieval_metrics=RetrievalMetricOutcome(
            primary_metric="mrr",
            baseline_primary=evaluation.baseline_mrr,
            candidate_primary=evaluation.candidate_mrr,
            primary_improvement=evaluation.primary_improvement,
            candidate_recall_at_k=evaluation.candidate_recall_at_k,
            candidate_mrr=evaluation.candidate_mrr,
            candidate_ndcg_at_k=evaluation.candidate_ndcg_at_k,
        ),
        governance_acceptance=_file_binding(root, acceptance_path),
        predecessor_governance=_file_binding(root, predecessor_path),
        predecessor_rejection=predecessor.rejection,
        predecessor_gold_retirement=predecessor.gold_retirement,
        historical_retrieval_rejection=_file_binding(
            root,
            retrieval_rejection_path,
        ),
        historical_retrieval_gold_retirement=_file_binding(
            root,
            retrieval_retirement_path,
        ),
        gold_retirement=_file_binding(
            root,
            Path(acceptance.gold_retirement_path),
        ),
    )


def _reranker_recovery_outcome(
    root: Path,
    erratum_path: Path,
) -> RejectedExperimentOutcome:
    erratum = verify_reranker_recovery_verifier_erratum(root, erratum_path)
    original = verify_original_with_json_normalization(
        root,
        Path(erratum.original_outcome.path),
    )
    evaluation = original.evaluation
    return RejectedExperimentOutcome(
        method="RERANKER",
        run_id=original.run_id,
        retained_detailed_evidence=True,
        failed_gates=original.failed_hard_gates,
        rejection=_file_binding(root, Path(erratum.original_outcome.path)),
        gold_retirement=_file_binding(root, Path(erratum.gold_v3_retirement.path)),
        rejected_candidate_bundle=_directory_binding(
            root, Path(original.candidate.path)
        ),
        formal_evaluation=_file_binding(
            root,
            Path(original.evaluation.formal_report.path),
        ),
        observations=_file_binding(root, Path(original.evaluation.observations.path)),
        retrieval_metrics=RetrievalMetricOutcome(
            primary_metric="ndcg_at_k",
            baseline_primary=evaluation.baseline_ndcg_at_k,
            candidate_primary=evaluation.candidate_ndcg_at_k,
            primary_improvement=evaluation.primary_improvement,
            candidate_recall_at_k=evaluation.candidate_recall_at_k,
            candidate_mrr=evaluation.candidate_mrr,
            candidate_ndcg_at_k=evaluation.candidate_ndcg_at_k,
        ),
        governance_acceptance=_file_binding(root, erratum_path),
    )


def _reranker_calibrated_outcome(
    root: Path,
    acceptance_path: Path,
    retrieval_rejection_path: Path,
    retrieval_retirement_path: Path,
) -> AcceptedRerankerOutcome:
    acceptance = verify_calibrated_reranker_value(root, acceptance_path)
    predecessor_rejection, predecessor_retirement = (
        _calibrated_predecessor_retrieval_sources(root, acceptance)
    )
    if (
        acceptance.status != "RERANKER_CALIBRATED_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
        or acceptance.decision != "RERANKER_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
        or acceptance.candidate_accepted is not True
        or acceptance.failed_hard_gates
        or acceptance.runtime.actual_gpu_inference is not True
        or acceptance.runtime.source_candidate_actual_gpu_trained is not True
        or acceptance.runtime.current_run_model_training_performed is not False
        or acceptance.formal_model_release_created is not False
        or acceptance.runtime_eligible is not False
        or acceptance.same_gold_reuse_permitted is not False
        or predecessor_rejection != retrieval_rejection_path
        or predecessor_retirement != retrieval_retirement_path
    ):
        raise ModelExperimentOutcomesError(
            "accepted Reranker enterprise-value evidence is invalid"
        )
    evaluation = acceptance.evaluation
    return AcceptedRerankerOutcome(
        run_id=acceptance.run_id,
        candidate_bundle=_directory_binding(root, Path(acceptance.candidate.path)),
        formal_evaluation=_file_binding(
            root,
            Path(acceptance.evaluation.formal_report.path),
        ),
        observations=_file_binding(
            root,
            Path(acceptance.evaluation.observations.path),
        ),
        retrieval_metrics=RetrievalMetricOutcome(
            primary_metric="ndcg_at_k",
            baseline_primary=evaluation.baseline_ndcg_at_k,
            candidate_primary=evaluation.candidate_ndcg_at_k,
            primary_improvement=evaluation.primary_improvement,
            candidate_recall_at_k=evaluation.candidate_recall_at_k,
            candidate_mrr=evaluation.candidate_mrr,
            candidate_ndcg_at_k=evaluation.candidate_ndcg_at_k,
        ),
        governance_acceptance=_file_binding(root, acceptance_path),
        predecessor_governance=_file_binding(
            root,
            Path(acceptance.predecessor.erratum.path),
        ),
        gold_retirement=_file_binding(
            root,
            Path(acceptance.gold_retirement_path),
        ),
    )


def _calibrated_predecessor_retrieval_sources(
    root: Path,
    acceptance: RerankerCalibratedValueReport,
) -> tuple[Path, Path]:
    grounded_erratum = verify_reranker_grounded_verifier_erratum(
        root,
        Path(acceptance.predecessor.erratum.path),
    )
    grounded = verify_original_with_v4_observation_schema(
        root,
        Path(grounded_erratum.original_outcome.path),
    )
    recovery_erratum = verify_reranker_recovery_verifier_erratum(
        root,
        Path(grounded.predecessor.erratum.path),
    )
    recovery = verify_original_with_json_normalization(
        root,
        Path(recovery_erratum.original_outcome.path),
    )
    return (
        _inside_file(root, Path(recovery.predecessor.rejection.path)),
        _inside_file(root, Path(recovery.predecessor.gold_retirement.path)),
    )


def _retrieval_outcomes(
    root: Path,
    rejection_path: Path,
    retirement_path: Path,
) -> tuple[RejectedExperimentOutcome, RejectedExperimentOutcome]:
    rejection = _load_object(rejection_path)
    retirement = _load_object(retirement_path)
    unsigned = dict(rejection)
    chain = unsigned.pop("evidence_chain_sha256", None)
    training = rejection.get("training")
    evaluations = rejection.get("evaluations")
    failed = rejection.get("failed_gates")
    if (
        rejection.get("schema_version") != "enterprise-retrieval-formal-rejection/v1"
        or rejection.get("classification") != CLASSIFICATION
        or rejection.get("production_claim") is not False
        or rejection.get("status") != "RETRIEVAL_ACTUAL_GPU_CANDIDATES_REJECTED"
        or rejection.get("decision") != "DO_NOT_REGISTER_OR_DEPLOY"
        or rejection.get("candidate_accepted") is not False
        or rejection.get("formal_model_release_created") is not False
        or rejection.get("runtime_eligible") is not False
        or not isinstance(chain, str)
        or chain != _source_digest(unsigned)
        or not isinstance(training, list)
        or not isinstance(evaluations, list)
        or not _string_list(failed)
    ):
        raise ModelExperimentOutcomesError("Retrieval rejection evidence is invalid")
    run_id = rejection.get("run_id")
    if (
        retirement.get("schema_version")
        != "enterprise-retrieval-gold-suite-retirement/v1"
        or retirement.get("classification") != CLASSIFICATION
        or retirement.get("run_id") != run_id
        or retirement.get("outcome") != "REJECTED"
        or retirement.get("reuse_permitted") is not False
    ):
        raise ModelExperimentOutcomesError(
            "Retrieval Gold retirement evidence is invalid"
        )
    training_by_component = _by_component(training, "Retrieval training")
    evaluation_by_component = _by_component(evaluations, "Retrieval evaluation")
    result: list[RejectedExperimentOutcome] = []
    for component, method in (("EMBEDDING", "EMBEDDING"), ("RERANKER", "RERANKER")):
        component_training = training_by_component.get(component)
        evaluation = evaluation_by_component.get(component)
        component_failures = tuple(
            str(item) for item in failed if str(item).startswith(component.lower())
        )
        if (
            not isinstance(component_training, dict)
            or component_training.get("optimizer_steps") != 48
            or not _positive_number(
                component_training.get("peak_gpu_memory_allocated_bytes")
            )
            or not _positive_number(
                component_training.get("peak_gpu_memory_reserved_bytes")
            )
            or not isinstance(evaluation, dict)
            or not component_failures
            or not _all_true(evaluation.get("hard_gate_results"))
        ):
            raise ModelExperimentOutcomesError(
                f"{component} actual GPU rejection evidence is invalid"
            )
        metrics = _retrieval_metrics(evaluation)
        name = component.lower()
        candidate_directory = rejection_path.parent / "candidates" / name
        formal_path = rejection_path.parent / f"{name}-formal-evaluation.json"
        observations_path = rejection_path.parent / f"{name}-observations.json"
        result.append(
            RejectedExperimentOutcome(
                method=method,
                run_id=str(run_id),
                retained_detailed_evidence=True,
                failed_gates=component_failures,
                rejection=_file_binding(root, rejection_path),
                gold_retirement=_file_binding(root, retirement_path),
                rejected_candidate_bundle=_directory_binding(root, candidate_directory),
                formal_evaluation=_file_binding(root, formal_path),
                observations=_file_binding(root, observations_path),
                retrieval_metrics=metrics,
            )
        )
    return result[0], result[1]


def _retrieval_metrics(document: dict[str, Any]) -> RetrievalMetricOutcome:
    keys = (
        "baseline_primary",
        "candidate_primary",
        "primary_improvement",
        "candidate_recall_at_k",
        "candidate_mrr",
        "candidate_ndcg_at_k",
    )
    if any(not _finite_number(document.get(key)) for key in keys):
        raise ModelExperimentOutcomesError("Retrieval metric evidence is invalid")
    primary_metric = document.get("primary_metric")
    if primary_metric not in {"mrr", "ndcg_at_k"}:
        raise ModelExperimentOutcomesError("Retrieval primary metric is invalid")
    baseline = float(document["baseline_primary"])
    candidate = float(document["candidate_primary"])
    improvement = float(document["primary_improvement"])
    if abs((candidate - baseline) - improvement) > 1e-9:
        raise ModelExperimentOutcomesError(
            "Retrieval primary improvement is inconsistent"
        )
    return RetrievalMetricOutcome(
        primary_metric=primary_metric,
        baseline_primary=baseline,
        candidate_primary=candidate,
        primary_improvement=improvement,
        candidate_recall_at_k=float(document["candidate_recall_at_k"]),
        candidate_mrr=float(document["candidate_mrr"]),
        candidate_ndcg_at_k=float(document["candidate_ndcg_at_k"]),
    )


def _by_component(rows: list[Any], label: str) -> dict[str, dict[str, Any]]:
    result = {
        str(item["component"]): item
        for item in rows
        if isinstance(item, dict) and isinstance(item.get("component"), str)
    }
    if set(result) != {"EMBEDDING", "RERANKER"} or len(rows) != 2:
        raise ModelExperimentOutcomesError(f"{label} components changed")
    return result


def _file_binding(root: Path, path: Path) -> FileBinding:
    resolved = _inside_file(root, path)
    return FileBinding(
        path=resolved.relative_to(root).as_posix(),
        size_bytes=resolved.stat().st_size,
        sha256=sha256(resolved.read_bytes()).hexdigest(),
    )


def _directory_binding(root: Path, path: Path) -> DirectoryBinding:
    resolved = _inside_directory(root, path)
    manifest: list[dict[str, Any]] = []
    for candidate in sorted(resolved.rglob("*")):
        if candidate.is_symlink():
            raise ModelExperimentOutcomesError("rejected candidate contains a symlink")
        if candidate.is_file():
            manifest.append(
                {
                    "path": candidate.relative_to(resolved).as_posix(),
                    "size_bytes": candidate.stat().st_size,
                    "sha256": sha256(candidate.read_bytes()).hexdigest(),
                }
            )
    if not manifest:
        raise ModelExperimentOutcomesError("rejected candidate evidence is empty")
    return DirectoryBinding(
        path=resolved.relative_to(root).as_posix(),
        file_count=len(manifest),
        size_bytes=sum(int(item["size_bytes"]) for item in manifest),
        manifest_sha256=_digest(manifest),
    )


def _stable_projection(report: ModelExperimentOutcomesReport) -> dict[str, Any]:
    return report.model_dump(
        mode="json",
        exclude={"generated_at", "evidence_chain_sha256"},
    )


def _verify_report_summary(report: ModelExperimentOutcomesReport) -> None:
    rejected = tuple(
        item for item in report.outcomes if isinstance(item, RejectedExperimentOutcome)
    )
    accepted = tuple(
        item
        for item in report.outcomes
        if not isinstance(item, RejectedExperimentOutcome)
    )
    expected_verified = tuple(
        f"{item.method}_ENTERPRISE_VALUE" for item in accepted
    )
    expected_unverified = tuple(
        f"{item.method}_ENTERPRISE_VALUE" for item in rejected
    )
    expected_decision = (
        "RETAIN_REJECTIONS_AND_ADVANCE_ACCEPTED_CANDIDATES_TO_RELEASE_DRAFT"
        if accepted
        else "RETAIN_AS_AUDIT_HISTORY_DO_NOT_DEPLOY"
    )
    if (
        report.rejected_method_count != len(rejected)
        or report.active_candidate_count != len(accepted)
        or report.verified_enterprise_value_methods != expected_verified
        or report.intentionally_unverified_methods != expected_unverified
        or report.decision != expected_decision
        or any(item.candidate_accepted is not False for item in rejected)
        or any(item.release_draft_eligible is not True for item in accepted)
        or any(item.runtime_eligible is not False for item in report.outcomes)
        or not all(report.governance_gates.model_dump().values())
    ):
        raise ModelExperimentOutcomesError(
            "model experiment outcome summary is inconsistent"
        )


def _string_list(value: Any) -> TypeGuard[list[str]]:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item) for item in value)
    )


def _all_true(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(item is True for item in value.values())
    )


def _positive_number(value: Any) -> bool:
    return _finite_number(value) and float(value) > 0.0


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and float("-inf") < float(value) < float("inf")
    )


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelExperimentOutcomesError(
            f"invalid experiment JSON: {path.name}"
        ) from exc
    if not isinstance(value, dict):
        raise ModelExperimentOutcomesError(f"invalid experiment JSON: {path.name}")
    return value


def _inside_file(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ModelExperimentOutcomesError(
            "experiment file escapes repository"
        ) from exc
    if not resolved.is_file():
        raise ModelExperimentOutcomesError("experiment evidence is not a file")
    return resolved


def _inside_directory(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ModelExperimentOutcomesError(
            "experiment directory escapes repository"
        ) from exc
    if not resolved.is_dir():
        raise ModelExperimentOutcomesError("experiment evidence is not a directory")
    return resolved


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
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return sha256(payload).hexdigest()
