"""Read-only verification of retained TTS experiment evidence; no training or GPU execution."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.model_evidence.report_files import file_sha256
from industrial_ops_agent.model_evidence.retired_sources import retired_source_binding
from industrial_ops_agent.model_evidence.tts_legacy_files import (
    LegacyTtsEvidenceFiles,
    legacy_ascii_digest,
)

_LEGACY_SOURCE_PATH = "src/industrial_ops_agent/simulation/tts_enterprise_value_lab.py"

SCHEMA_VERSION: Literal["enterprise-tts-value-lab/v1"] = "enterprise-tts-value-lab/v1"


CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = "LOCAL_STAGING_PROJECT_AUTHORIZED"


STATUS: Literal["TTS_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"] = "TTS_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"


DECISION: Literal["TTS_CANDIDATE_ELIGIBLE_FOR_PROJECT_EVALUATION"] = (
    "TTS_CANDIDATE_ELIGIBLE_FOR_PROJECT_EVALUATION"
)


OUTPUT_RELATIVE = Path("artifacts/m7-tts-enterprise-value-lab")


BASE_MODEL_RELATIVE = OUTPUT_RELATIVE / "inputs" / "speecht5-tts-30fcde30"


VOCODER_RELATIVE = OUTPUT_RELATIVE / "inputs" / "speecht5-hifigan-e8b38625"


ASR_RELATIVE = Path("artifacts/m7-asr-enterprise-value-lab/inputs/whisper-tiny-c4f62d5f")


XVECTOR_RELATIVE = OUTPUT_RELATIVE / "inputs" / "spkrec-xvect-voxceleb-56895a2d"


TRAINING_DATASET_RELATIVE = OUTPUT_RELATIVE / "inputs" / "industrial-safety-voice-v1"


GOLD_DATASET_RELATIVE = OUTPUT_RELATIVE / "inputs" / "industrial-safety-tts-final-gold-v2"


GOLD_V1_RETIREMENT_RELATIVE = OUTPUT_RELATIVE / "gold-v1-retirement.json"


IMAGE: Literal["industrial-ops/tts-xvector-runtime:speechbrain-1.0.3"] = (
    "industrial-ops/tts-xvector-runtime:speechbrain-1.0.3"
)


BASE_MODEL_ID = "microsoft/speecht5_tts"


BASE_MODEL_REVISION = "30fcde30f19b87502b8435427b5f5068e401d5f6"


BASE_MODEL_WEIGHT_SHA256 = "d60d28067349ef66b50d8cd643ae56b6d6b8f27def929bc4ef6fcad907954190"


VOCODER_MODEL_ID = "microsoft/speecht5_hifigan"


VOCODER_REVISION = "e8b38625359976b675c3b3e2a41351058f5ba377"


VOCODER_WEIGHT_SHA256 = "b171e9bcd8a2b50dc9780040478dfa26783a9ee4be012cf5776914f091d6887b"


ASR_MODEL_ID = "openai/whisper-tiny"


ASR_REVISION = "c4f62d5fce5d73978a1dbfcac01926312acd1a51"


ASR_WEIGHT_SHA256 = "9607f98a2b22d9e229ae43c52ecea79dcede9e0c5cfae67e8da6eda86d8aac1d"


XVECTOR_MODEL_ID = "speechbrain/spkrec-xvect-voxceleb"


XVECTOR_REVISION = "56895a2df401be4150a159f3a1c653f00051d477"


XVECTOR_WEIGHT_SHA256 = "9d96cafa0ede1a84799b67dc9b5645f31f5b7d094e7e4775e5d5c12547883a93"


TRAINING_MANIFEST_SHA256 = "92cb8b4a26f3ebaa457289d86db527f00a25d44caac9022eec68fe59b61bf20b"


GOLD_MANIFEST_SHA256 = "cc5d0fe3c3795003f2527ffa3686e9b797994561a5d167c00cbc478205f31882"


GOLD_V1_RETIREMENT_SHA256 = "0342a63bc3adfd6231556051d7a3b67a4a8906cdd108b4ad7edd1a2ce70113f0"


SPEAKER_EMBEDDING_SHA256 = "94318f6c5d7b1061c769b8027955b1a3fb477eb03db9b1f710fdddd3c56fb888"


_TRAINING_CONFIG: dict[str, Any] = {
    "method": "TTS_POSTNET_ADAPTATION",
    "trainable_scope": "speech_decoder_postnet.layers",
    "max_steps": 24,
    "evaluation_interval": 8,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "precision": "bfloat16",
    "sampling_rate": 16_000,
    "max_audio_seconds": 30.0,
    "seed": 20260824,
    "data_seed": 20260825,
    "cpu_threads": 2,
    "dataloader_workers": 0,
    "paired_generation_seed_reset": True,
    "container_memory_limit_bytes": 4 * 1024**3,
    "safety_phrase_completeness_min": 1.0,
    "terminology_recall_min": 0.95,
    "intelligibility_min": 0.90,
    "audio_integrity_rate_min": 1.0,
    "voice_similarity_improvement_min": 0.0001,
}


class TtsEnterpriseValueLabError(RuntimeError):
    """The governed TTS lab could not produce or verify its evidence."""


_FILES = LegacyTtsEvidenceFiles(TtsEnterpriseValueLabError)


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileEvidence(_ClosedModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class InputComponentEvidence(_ClosedModel):
    role: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    weight_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    path: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: tuple[FileEvidence, ...] = Field(min_length=2)


class TtsDatasetEvidence(_ClosedModel):
    training_manifest: FileEvidence
    final_gold_manifest: FileEvidence
    retired_predecessor: FileEvidence
    speaker_embedding: FileEvidence
    train_count: Literal[18]
    validation_count: Literal[6]
    development_count: Literal[10]
    final_gold_count: Literal[10]
    voice_profile_id: Literal["industrial-safety-en-v1"]
    source_type: Literal["PLATFORM_SYNTHETIC"]
    project_enterprise_use_authorized: Literal[True] = True
    enterprise_production_data: Literal[False] = False
    frozen_gold_excluded_from_training_selection_and_development: Literal[True] = True
    source_ids_and_texts_disjoint: Literal[True] = True


class TtsCandidateEvidence(_ClosedModel):
    path: str = Field(min_length=1)
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    files: tuple[FileEvidence, ...] = Field(min_length=6)
    role: Literal["AUTHORIZED_ENTERPRISE_TTS_EVALUATION_CANDIDATE"]


class TtsRuntimeEvidence(_ClosedModel):
    actual_gpu_execution: Literal[True] = True
    model_training_simulated: Literal[False] = False
    method: Literal["TTS_POSTNET_ADAPTATION"]
    trainer_family: Literal["transformers-speecht5-postnet"]
    gpu_name: str = Field(min_length=1)
    gpu_total_memory_bytes: int = Field(gt=0)
    peak_gpu_memory_allocated_bytes: int = Field(gt=0)
    peak_gpu_memory_reserved_bytes: int = Field(gt=0)
    gpu_memory_allocated_after_cleanup_bytes: int = Field(ge=0)
    optimizer_steps: Literal[24]
    selected_validation_step: int = Field(ge=8, le=24)
    trainable_parameters: int = Field(gt=0)
    total_parameters: int = Field(gt=0)
    train_loss: float = Field(ge=0.0)
    selected_validation_loss: float = Field(ge=0.0)
    torch_version: str = Field(min_length=1)
    transformers_version: str = Field(min_length=1)
    speechbrain_version: str = Field(min_length=1)
    non_root_container_user: Literal[True] = True
    network_disabled: Literal[True] = True
    read_only_root_filesystem: Literal[True] = True
    direct_python_entrypoint: Literal[True] = True
    models_loaded_sequentially: Literal[True] = True
    paired_generation_seed_reset: Literal[True] = True
    caches_scoped_to_tmpfs: Literal[True] = True
    cpu_threads: Literal[2] = 2
    dataloader_workers: Literal[0] = 0
    container_memory_limit_bytes: Literal[4294967296] = 4294967296


class TtsEvaluationEvidence(_ClosedModel):
    formal_report: FileEvidence
    observations: FileEvidence
    dataset_receipt: FileEvidence
    frozen_gold_case_count: Literal[10]
    gold_evaluation_pass_count: Literal[1]
    baseline_intelligibility: float = Field(ge=0.0, le=1.0)
    candidate_intelligibility: float = Field(ge=0.0, le=1.0)
    baseline_safety_phrase_completeness: float = Field(ge=0.0, le=1.0)
    candidate_safety_phrase_completeness: float = Field(ge=0.0, le=1.0)
    baseline_terminology_recall: float = Field(ge=0.0, le=1.0)
    candidate_terminology_recall: float = Field(ge=0.0, le=1.0)
    baseline_wer: float = Field(ge=0.0)
    candidate_wer: float = Field(ge=0.0)
    validation_baseline_voice_similarity: float = Field(ge=-1.0, le=1.0)
    validation_candidate_voice_similarity: float = Field(ge=-1.0, le=1.0)
    validation_voice_similarity_improvement: float = Field(gt=0.0)
    gold_baseline_voice_similarity: float = Field(ge=-1.0, le=1.0)
    gold_candidate_voice_similarity: float = Field(ge=-1.0, le=1.0)
    gold_voice_similarity_improvement: float = Field(gt=0.0)
    voice_similarity_improvement_min: float = Field(ge=0.0001, le=0.0001)
    generation_peak_gpu_memory_reserved_bytes: int = Field(gt=0)


class TtsHardGates(_ClosedModel):
    data_governance: Literal[True] = True
    actual_gpu_training: Literal[True] = True
    immutable_model_data_and_image: Literal[True] = True
    frozen_gold_independence: Literal[True] = True
    tts_voice_usage_authorization: Literal[True] = True
    tts_audio_integrity: Literal[True] = True
    tts_safety_warning_completeness: Literal[True] = True
    tts_terminology_accuracy: Literal[True] = True
    tts_intelligibility: Literal[True] = True
    no_blocking_regressions: Literal[True] = True
    validation_voice_similarity_improved: Literal[True] = True
    gold_voice_similarity_improved: Literal[True] = True
    sequential_model_memory_release: Literal[True] = True
    paired_generation_randomness_controlled: Literal[True] = True
    resource_profile_respected: Literal[True] = True


class TtsEnterpriseValueReport(_ClosedModel):
    schema_version: Literal["enterprise-tts-value-lab/v1"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal["TTS_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"] = STATUS
    decision: Literal["TTS_CANDIDATE_ELIGIBLE_FOR_PROJECT_EVALUATION"] = DECISION
    run_id: str = Field(pattern=r"^tts-[0-9a-f]{20}$")
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image: Literal["industrial-ops/tts-xvector-runtime:speechbrain-1.0.3"] = IMAGE
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    input_components: tuple[InputComponentEvidence, ...] = Field(min_length=4, max_length=4)
    dataset: TtsDatasetEvidence
    candidate: TtsCandidateEvidence
    runtime: TtsRuntimeEvidence
    evaluation: TtsEvaluationEvidence
    hard_gates: TtsHardGates
    formal_model_release_created: Literal[False] = False
    runtime_eligible: Literal[False] = False
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def planned_run_id(repo_root: Path, image_digest: str) -> str:
    root = repo_root.resolve(strict=True)
    _FILES.require_image_digest(image_digest)
    paths = _input_paths(root)
    _verify_inputs(paths)
    source_binding = _retired_source(root)
    identity = {
        "training_config": _TRAINING_CONFIG,
        "image": IMAGE,
        "image_digest": image_digest,
        "source_sha256": source_binding["sha256"],
        "components": {
            name: legacy_ascii_digest(_FILES.directory_manifest(path))
            for name, path in paths["components"].items()
        },
        "training_manifest_sha256": file_sha256(paths["training_manifest"]),
        "gold_manifest_sha256": file_sha256(paths["gold_manifest"]),
        "gold_v1_retirement_sha256": file_sha256(paths["gold_v1_retirement"]),
        "speaker_embedding_sha256": file_sha256(paths["speaker_embedding"]),
    }
    return f"tts-{legacy_ascii_digest(identity)[:20]}"


def verify_tts_enterprise_value(
    repo_root: Path, acceptance_path: Path | None = None
) -> TtsEnterpriseValueReport:
    root = repo_root.resolve(strict=True)
    report_path = (
        _latest_report_path(root)
        if acceptance_path is None
        else _FILES.inside_file(root, acceptance_path)
    )
    report = TtsEnterpriseValueReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != legacy_ascii_digest(unsigned):
        raise TtsEnterpriseValueLabError("TTS evidence chain changed")
    source = _retired_source(root)
    if report.source_path != source["path"] or report.source_sha256 != source["sha256"]:
        raise TtsEnterpriseValueLabError("TTS trainer source changed")
    if planned_run_id(root, report.image_digest) != report.run_id:
        raise TtsEnterpriseValueLabError("TTS run identity changed")
    paths = _input_paths(root)
    _verify_inputs(paths)
    expected_components = _component_evidence(root, paths)
    if [item.model_dump(mode="json") for item in report.input_components] != expected_components:
        raise TtsEnterpriseValueLabError("TTS input component changed")
    _verify_file_evidence(root, report.dataset.training_manifest)
    _verify_file_evidence(root, report.dataset.final_gold_manifest)
    _verify_file_evidence(root, report.dataset.retired_predecessor)
    _verify_file_evidence(root, report.dataset.speaker_embedding)
    _verify_file_evidence(root, report.evaluation.formal_report)
    _verify_file_evidence(root, report.evaluation.observations)
    _verify_file_evidence(root, report.evaluation.dataset_receipt)
    candidate_path = _FILES.inside_directory(root, Path(report.candidate.path))
    candidate_files = _FILES.directory_manifest(candidate_path)
    if (
        candidate_files != [item.model_dump(mode="json") for item in report.candidate.files]
        or legacy_ascii_digest(candidate_files) != report.candidate.bundle_sha256
        or sum(item["size_bytes"] for item in candidate_files) != report.candidate.size_bytes
    ):
        raise TtsEnterpriseValueLabError("TTS candidate bundle changed")
    return report


def _input_paths(root: Path) -> dict[str, Any]:
    training_dataset = _FILES.inside_directory(root, TRAINING_DATASET_RELATIVE)
    gold_dataset = _FILES.inside_directory(root, GOLD_DATASET_RELATIVE)
    return {
        "components": {
            "base_model": _FILES.inside_directory(root, BASE_MODEL_RELATIVE),
            "vocoder": _FILES.inside_directory(root, VOCODER_RELATIVE),
            "asr": _FILES.inside_directory(root, ASR_RELATIVE),
            "xvector": _FILES.inside_directory(root, XVECTOR_RELATIVE),
        },
        "training_dataset": training_dataset,
        "training_manifest": _FILES.inside_file(root, training_dataset / "manifest.json"),
        "speaker_embedding": _FILES.inside_file(root, training_dataset / "speaker_embedding.npy"),
        "gold_dataset": gold_dataset,
        "gold_manifest": _FILES.inside_file(root, gold_dataset / "manifest.json"),
        "gold_v1_retirement": _FILES.inside_file(root, GOLD_V1_RETIREMENT_RELATIVE),
    }


def _verify_inputs(paths: dict[str, Any]) -> None:
    expected_weights = {
        "base_model": ("pytorch_model.bin", BASE_MODEL_WEIGHT_SHA256),
        "vocoder": ("pytorch_model.bin", VOCODER_WEIGHT_SHA256),
        "asr": ("pytorch_model.bin", ASR_WEIGHT_SHA256),
        "xvector": ("embedding_model.ckpt", XVECTOR_WEIGHT_SHA256),
    }
    for name, (file_name, digest) in expected_weights.items():
        if file_sha256(paths["components"][name] / file_name) != digest:
            raise TtsEnterpriseValueLabError(f"TTS {name} weight changed")
    if (
        file_sha256(paths["training_manifest"]) != TRAINING_MANIFEST_SHA256
        or file_sha256(paths["gold_manifest"]) != GOLD_MANIFEST_SHA256
        or file_sha256(paths["gold_v1_retirement"]) != GOLD_V1_RETIREMENT_SHA256
        or file_sha256(paths["speaker_embedding"]) != SPEAKER_EMBEDDING_SHA256
    ):
        raise TtsEnterpriseValueLabError("TTS governed data input changed")
    training = _FILES.load_object(paths["training_manifest"])
    gold = _FILES.load_object(paths["gold_manifest"])
    if not _gold_manifest_is_independent(training, gold):
        raise TtsEnterpriseValueLabError("TTS final Gold independence changed")


def _gold_manifest_is_independent(training: dict[str, Any], gold: dict[str, Any]) -> bool:
    training_rows = training.get("rows")
    gold_rows = gold.get("rows")
    policy = gold.get("selection_policy")
    if (
        not isinstance(training_rows, list)
        or not isinstance(gold_rows, list)
        or len(gold_rows) != 10
        or not isinstance(policy, dict)
        or policy
        != {
            "used_for_training": False,
            "used_for_checkpoint_selection": False,
            "used_for_hyperparameter_selection": False,
            "used_for_development_regression": False,
            "single_final_evaluation_only": True,
        }
    ):
        return False
    training_ids = {str(row.get("case_id")) for row in training_rows if isinstance(row, dict)}
    training_texts = {
        str(row.get("text", "")).strip().casefold()
        for row in training_rows
        if isinstance(row, dict)
    }
    gold_ids = {str(row.get("case_id")) for row in gold_rows if isinstance(row, dict)}
    gold_texts = {
        str(row.get("text", "")).strip().casefold() for row in gold_rows if isinstance(row, dict)
    }
    return (
        len(gold_ids) == len(gold_rows)
        and len(gold_texts) == len(gold_rows)
        and training_ids.isdisjoint(gold_ids)
        and training_texts.isdisjoint(gold_texts)
        and all(
            isinstance(row, dict)
            and row.get("risk") == "HIGH"
            and bool(row.get("required_safety_phrases"))
            and bool(row.get("industrial_terms"))
            for row in gold_rows
        )
    )


def _component_evidence(root: Path, paths: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = {
        "base_model": (BASE_MODEL_ID, BASE_MODEL_REVISION, BASE_MODEL_WEIGHT_SHA256),
        "vocoder": (VOCODER_MODEL_ID, VOCODER_REVISION, VOCODER_WEIGHT_SHA256),
        "asr": (ASR_MODEL_ID, ASR_REVISION, ASR_WEIGHT_SHA256),
        "xvector": (XVECTOR_MODEL_ID, XVECTOR_REVISION, XVECTOR_WEIGHT_SHA256),
    }
    return _FILES.component_evidence(root, paths, metadata)


def _verify_file_evidence(root: Path, evidence: FileEvidence) -> Path:
    path = _FILES.inside_file(root, Path(evidence.path))
    if path.stat().st_size != evidence.size_bytes or file_sha256(path) != evidence.sha256:
        raise TtsEnterpriseValueLabError("TTS evidence file changed")
    return path


def _latest_report_path(root: Path) -> Path:
    latest_path = _FILES.inside_file(root, OUTPUT_RELATIVE / "latest.json")
    latest = _FILES.load_object(latest_path)
    value = latest.get("report")
    if not isinstance(value, str) or not value:
        raise TtsEnterpriseValueLabError("TTS latest pointer is invalid")
    report_path = _FILES.inside_file(root, Path(value))
    if latest.get("report_sha256") != file_sha256(report_path):
        raise TtsEnterpriseValueLabError("TTS latest report digest changed")
    return report_path


def _retired_source(root: Path) -> dict[str, Any]:
    return retired_source_binding(root, _LEGACY_SOURCE_PATH, error_type=TtsEnterpriseValueLabError)
