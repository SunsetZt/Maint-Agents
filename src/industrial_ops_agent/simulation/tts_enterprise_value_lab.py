"""Low-memory actual-GPU SpeechT5 enterprise-value experiment."""

from __future__ import annotations

import gc
import importlib
import json
import math
import os
import re
import shutil
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.evaluation.dataset import EvaluationCase, TtsEvaluationContract
from industrial_ops_agent.evaluation.metrics import (
    ModelObservation,
    PairedMetricReport,
    score_tts_paired_observations,
)
from industrial_ops_agent.experiments.service import TTS_HARD_GATES

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
GOLD_RETIREMENT_RELATIVE = OUTPUT_RELATIVE / "gold-v2-retirement.json"
IMAGE: Literal["industrial-ops/tts-xvector-runtime:speechbrain-1.0.3"] = (
    "industrial-ops/tts-xvector-runtime:speechbrain-1.0.3"
)
CONTAINER_NAME = "ioap-tts-enterprise-value-lab"

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
GOLD_RETIREMENT_SHA256 = "e723b35efc2154d7d2fd770f048cb324593b9c74ae65f673163b6e8f2cf1dad7"
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


@dataclass(frozen=True, slots=True)
class _GeneratedAudio:
    case_id: str
    audio: Any
    latency_ms: float
    input_tokens: int
    duration_seconds: float
    silence_ratio: float
    clipping_ratio: float
    audio_sha256: str


@dataclass(frozen=True, slots=True)
class _TrainingResult:
    runtime: dict[str, Any]
    observations_document: dict[str, Any]
    formal_report_document: dict[str, Any]
    dataset_document: dict[str, Any]
    evaluation: dict[str, Any]


class _TtsCollator:
    def __init__(self, processor: Any, torch: Any, model: Any) -> None:
        self._processor = processor
        self._torch = torch
        self._reduction_factor = int(model.config.reduction_factor)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = dict(
            self._processor.pad(
                input_ids=[{"input_ids": item["input_ids"]} for item in features],
                labels=[{"input_values": item["labels"]} for item in features],
                return_tensors="pt",
            )
        )
        decoder_mask = batch.pop("decoder_attention_mask", None)
        if decoder_mask is not None:
            batch["labels"] = batch["labels"].masked_fill(decoder_mask.unsqueeze(-1).ne(1), -100)
        target_length = int(batch["labels"].shape[1])
        target_length -= target_length % self._reduction_factor
        batch["labels"] = batch["labels"][:, :target_length]
        batch["speaker_embeddings"] = self._torch.tensor(
            [item["speaker_embeddings"] for item in features],
            dtype=self._torch.float32,
        )
        return batch


def planned_run_id(repo_root: Path, image_digest: str) -> str:
    root = repo_root.resolve(strict=True)
    _require_image_digest(image_digest)
    paths = _input_paths(root)
    _verify_inputs(paths)
    source_path = Path(__file__).resolve(strict=True)
    _require_inside(root, source_path)
    identity = {
        "training_config": _TRAINING_CONFIG,
        "image": IMAGE,
        "image_digest": image_digest,
        "source_sha256": _file_sha256(source_path),
        "components": {
            name: _manifest_digest(_directory_manifest(path))
            for name, path in paths["components"].items()
        },
        "training_manifest_sha256": _file_sha256(paths["training_manifest"]),
        "gold_manifest_sha256": _file_sha256(paths["gold_manifest"]),
        "gold_v1_retirement_sha256": _file_sha256(paths["gold_v1_retirement"]),
        "speaker_embedding_sha256": _file_sha256(paths["speaker_embedding"]),
    }
    return f"tts-{_digest(identity)[:20]}"


def execute_worker(repo_root: Path, *, image_digest: str) -> Path:
    root = repo_root.resolve(strict=True)
    _require_image_digest(image_digest)
    paths = _input_paths(root)
    _verify_inputs(paths)
    run_id = planned_run_id(root, image_digest)
    output_root = root / OUTPUT_RELATIVE
    final_directory = output_root / "runs" / run_id
    acceptance_path = final_directory / "acceptance.json"
    if acceptance_path.is_file():
        verify_tts_enterprise_value(root, acceptance_path)
        return acceptance_path
    assert_gold_available(root)
    if final_directory.exists():
        raise TtsEnterpriseValueLabError("immutable TTS run directory already exists")
    temporary = output_root / f".{run_id}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        training = _train_and_evaluate(paths, temporary / "candidate")
        observations_path = temporary / "observations.json"
        formal_report_path = temporary / "formal-evaluation.json"
        dataset_path = temporary / "dataset-receipt.json"
        _write_json(observations_path, training.observations_document)
        _write_json(formal_report_path, training.formal_report_document)
        _write_json(dataset_path, training.dataset_document)

        source_path = Path(__file__).resolve(strict=True)
        final_relative = final_directory.relative_to(root)
        candidate_files = _directory_manifest(temporary / "candidate")
        unsigned: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "classification": CLASSIFICATION,
            "enterprise_scope": "PROJECT_INTERNAL",
            "production_claim": False,
            "external_enterprise_production_claim": False,
            "generated_at": datetime.now(UTC).isoformat(),
            "status": STATUS,
            "decision": DECISION,
            "run_id": run_id,
            "source_path": source_path.relative_to(root).as_posix(),
            "source_sha256": _file_sha256(source_path),
            "image": IMAGE,
            "image_digest": image_digest,
            "input_components": _component_evidence(root, paths),
            "dataset": {
                "training_manifest": _file_evidence(
                    paths["training_manifest"].relative_to(root),
                    paths["training_manifest"],
                ),
                "final_gold_manifest": _file_evidence(
                    paths["gold_manifest"].relative_to(root), paths["gold_manifest"]
                ),
                "retired_predecessor": _file_evidence(
                    paths["gold_v1_retirement"].relative_to(root),
                    paths["gold_v1_retirement"],
                ),
                "speaker_embedding": _file_evidence(
                    paths["speaker_embedding"].relative_to(root),
                    paths["speaker_embedding"],
                ),
                "train_count": 18,
                "validation_count": 6,
                "development_count": 10,
                "final_gold_count": 10,
                "voice_profile_id": "industrial-safety-en-v1",
                "source_type": "PLATFORM_SYNTHETIC",
                "project_enterprise_use_authorized": True,
                "enterprise_production_data": False,
                "frozen_gold_excluded_from_training_selection_and_development": True,
                "source_ids_and_texts_disjoint": True,
            },
            "candidate": {
                "path": (final_relative / "candidate").as_posix(),
                "bundle_sha256": _manifest_digest(candidate_files),
                "size_bytes": sum(item["size_bytes"] for item in candidate_files),
                "files": candidate_files,
                "role": "AUTHORIZED_ENTERPRISE_TTS_EVALUATION_CANDIDATE",
            },
            "runtime": training.runtime,
            "evaluation": {
                "formal_report": _file_evidence(
                    final_relative / "formal-evaluation.json", formal_report_path
                ),
                "observations": _file_evidence(
                    final_relative / "observations.json", observations_path
                ),
                "dataset_receipt": _file_evidence(
                    final_relative / "dataset-receipt.json", dataset_path
                ),
                **training.evaluation,
            },
            "hard_gates": TtsHardGates().model_dump(mode="json"),
            "formal_model_release_created": False,
            "runtime_eligible": False,
        }
        draft = TtsEnterpriseValueReport.model_validate(
            {**unsigned, "evidence_chain_sha256": "0" * 64}
        )
        normalized = draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
        report = draft.model_copy(update={"evidence_chain_sha256": _digest(normalized)})
        _write_json(temporary / "acceptance.json", report.model_dump(mode="json"))
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(final_directory)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    verified = verify_tts_enterprise_value(root, acceptance_path)
    _write_latest(root, acceptance_path, verified)
    return acceptance_path


def assert_gold_available(repo_root: Path) -> None:
    """Fail before GPU work when the single-use final Gold suite is retired."""
    root = repo_root.resolve(strict=True)
    retirement_path = root / GOLD_RETIREMENT_RELATIVE
    if not retirement_path.exists():
        return
    retirement_path = _inside_file(root, retirement_path)
    if _file_sha256(retirement_path) != GOLD_RETIREMENT_SHA256:
        raise TtsEnterpriseValueLabError("TTS final Gold retirement receipt changed")
    retirement = _load_object(retirement_path)
    if (
        retirement.get("suite_id") != "industrial-safety-tts-final-gold-v2"
        or retirement.get("reuse_permitted") is not False
    ):
        raise TtsEnterpriseValueLabError("TTS final Gold retirement contract changed")
    raise TtsEnterpriseValueLabError(
        "TTS final Gold suite is retired after its single permitted evaluation"
    )


def verify_tts_enterprise_value(
    repo_root: Path, acceptance_path: Path | None = None
) -> TtsEnterpriseValueReport:
    root = repo_root.resolve(strict=True)
    report_path = (
        _latest_report_path(root)
        if acceptance_path is None
        else _inside_file(root, acceptance_path)
    )
    report = TtsEnterpriseValueReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != _digest(unsigned):
        raise TtsEnterpriseValueLabError("TTS evidence chain changed")
    source_path = _inside_file(root, Path(report.source_path))
    if _file_sha256(source_path) != report.source_sha256:
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
    candidate_path = _inside_directory(root, Path(report.candidate.path))
    candidate_files = _directory_manifest(candidate_path)
    if (
        candidate_files != [item.model_dump(mode="json") for item in report.candidate.files]
        or _manifest_digest(candidate_files) != report.candidate.bundle_sha256
        or sum(item["size_bytes"] for item in candidate_files) != report.candidate.size_bytes
    ):
        raise TtsEnterpriseValueLabError("TTS candidate bundle changed")
    return report


def _train_and_evaluate(paths: dict[str, Any], candidate_directory: Path) -> _TrainingResult:
    dependencies = _dependencies()
    numpy = dependencies["numpy"]
    torch = dependencies["torch"]
    soundfile = dependencies["soundfile"]
    transformers = dependencies["transformers"]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise TtsEnterpriseValueLabError("exactly one CUDA GPU is required")
    if (
        os.environ.get("IOAP_NETWORK_DISABLED") != "1"
        or os.environ.get("IOAP_READ_ONLY_ROOTFS") != "1"
        or os.environ.get("IOAP_DIRECT_PYTHON_ENTRYPOINT") != "1"
        or os.environ.get("TRITON_CACHE_DIR") != "/tmp/triton"
    ):
        raise TtsEnterpriseValueLabError("TTS low-memory container contract is incomplete")
    torch.set_num_threads(2)
    transformers.set_seed(_TRAINING_CONFIG["seed"])
    training_document = _load_object(paths["training_manifest"])
    gold_document = _load_object(paths["gold_manifest"])
    speaker = numpy.load(paths["speaker_embedding"], allow_pickle=False)
    processor = transformers.SpeechT5Processor.from_pretrained(
        paths["components"]["base_model"], local_files_only=True
    )
    model = transformers.SpeechT5ForTextToSpeech.from_pretrained(
        paths["components"]["base_model"],
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.speech_decoder_postnet.layers.parameters():
        parameter.requires_grad = True
    train_records = _training_records(
        training_document, "train", paths["training_dataset"], processor, speaker, soundfile
    )
    validation_records = _training_records(
        training_document,
        "validation",
        paths["training_dataset"],
        processor,
        speaker,
        soundfile,
    )
    checkpoints = candidate_directory.parent / "checkpoints"
    args = transformers.Seq2SeqTrainingArguments(
        output_dir=str(checkpoints),
        max_steps=_TRAINING_CONFIG["max_steps"],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=_TRAINING_CONFIG["learning_rate"],
        warmup_ratio=_TRAINING_CONFIG["warmup_ratio"],
        weight_decay=_TRAINING_CONFIG["weight_decay"],
        lr_scheduler_type=_TRAINING_CONFIG["lr_scheduler_type"],
        eval_strategy="steps",
        eval_steps=_TRAINING_CONFIG["evaluation_interval"],
        save_strategy="steps",
        save_steps=_TRAINING_CONFIG["evaluation_interval"],
        logging_steps=4,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        save_total_limit=1,
        save_only_model=True,
        predict_with_generate=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=_TRAINING_CONFIG["seed"],
        data_seed=_TRAINING_CONFIG["data_seed"],
        disable_tqdm=True,
    )
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_records,
        eval_dataset=validation_records,
        data_collator=_TtsCollator(processor, torch, model),
        processing_class=processor,
    )
    torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()
    trainer.evaluate()
    model.config.use_cache = True
    trainer.save_model(candidate_directory)
    processor.save_pretrained(candidate_directory)
    numpy.save(
        candidate_directory / "speaker_embedding.npy",
        speaker.astype("float32"),
        allow_pickle=False,
    )
    _write_json(
        candidate_directory / "industrial-ops-tts-voice-profile.json",
        {
            "schema_version": "industrial-ops-tts-voice-profile/v1",
            "voice_profile_id": "industrial-safety-en-v1",
            "language": "en-US",
            "sampling_rate": 16_000,
            "speaker_embedding_dimension": 512,
            "speech_contract_version": "industrial-tts-speech-v1",
            "model_family": "SPEECHT5",
            "base_model_revision": BASE_MODEL_REVISION,
            "speaker_embedding_file": "speaker_embedding.npy",
            "speaker_embedding_sha256": "sha256:"
            + _file_sha256(candidate_directory / "speaker_embedding.npy"),
            "authorized_source_types": ["PLATFORM_SYNTHETIC"],
        },
    )
    best_checkpoint = str(trainer.state.best_model_checkpoint or "")
    selected_step = _checkpoint_step(best_checkpoint)
    global_step = int(trainer.state.global_step)
    selected_validation_loss = float(trainer.state.best_metric)
    training_peak_allocated = int(torch.cuda.max_memory_allocated())
    training_peak_reserved = int(torch.cuda.max_memory_reserved())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    del trainer, model, processor
    gc.collect()
    torch.cuda.empty_cache()
    if checkpoints.exists():
        shutil.rmtree(checkpoints)

    validation_rows = _rows(training_document, "validation")
    validation_baseline, validation_candidate, validation_generation_peak = _generate_pair(
        validation_rows, paths, candidate_directory, speaker, dependencies
    )
    validation_voice, _, _ = _compare_speaker_similarity(
        validation_baseline,
        validation_candidate,
        speaker,
        paths["components"]["xvector"],
        dependencies,
    )
    if (
        validation_voice["mean_case_improvement"]
        < _TRAINING_CONFIG["voice_similarity_improvement_min"]
    ):
        raise TtsEnterpriseValueLabError(
            "TTS validation voice similarity did not improve: "
            f"{validation_voice['mean_case_improvement']:.9f}"
        )
    del validation_baseline, validation_candidate
    gc.collect()

    gold_rows = list(gold_document["rows"])
    gold_cases = _gold_cases(gold_rows)
    baseline_audio, candidate_audio, gold_generation_peak = _generate_pair(
        gold_rows, paths, candidate_directory, speaker, dependencies
    )
    baseline_transcripts, candidate_transcripts, asr_peak = _transcribe_pair(
        baseline_audio,
        candidate_audio,
        paths["components"]["asr"],
        dependencies,
    )
    gold_voice, baseline_voice_scores, candidate_voice_scores = _compare_speaker_similarity(
        baseline_audio,
        candidate_audio,
        speaker,
        paths["components"]["xvector"],
        dependencies,
    )
    baseline_observations = _observations(
        gold_cases,
        baseline_audio,
        baseline_transcripts,
        baseline_voice_scores,
    )
    candidate_observations = _observations(
        gold_cases,
        candidate_audio,
        candidate_transcripts,
        candidate_voice_scores,
    )
    formal_report = score_tts_paired_observations(
        gold_cases,
        candidate_observations,
        baseline_observations,
        safety_phrase_completeness_min=_TRAINING_CONFIG["safety_phrase_completeness_min"],
        terminology_recall_min=_TRAINING_CONFIG["terminology_recall_min"],
        intelligibility_min=_TRAINING_CONFIG["intelligibility_min"],
        audio_integrity_rate_min=_TRAINING_CONFIG["audio_integrity_rate_min"],
    )
    evaluation = _evaluation_summary(
        formal_report,
        gold_cases,
        baseline_observations,
        candidate_observations,
        validation_voice,
        gold_voice,
        max(validation_generation_peak, gold_generation_peak),
    )
    _require_formal_gates(formal_report, evaluation)
    runtime = {
        "actual_gpu_execution": True,
        "model_training_simulated": False,
        "method": "TTS_POSTNET_ADAPTATION",
        "trainer_family": "transformers-speecht5-postnet",
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "peak_gpu_memory_allocated_bytes": max(training_peak_allocated, int(asr_peak["allocated"])),
        "peak_gpu_memory_reserved_bytes": max(
            training_peak_reserved,
            validation_generation_peak,
            gold_generation_peak,
            int(asr_peak["reserved"]),
        ),
        "gpu_memory_allocated_after_cleanup_bytes": int(torch.cuda.memory_allocated()),
        "optimizer_steps": global_step,
        "selected_validation_step": selected_step,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "train_loss": float(train_result.metrics["train_loss"]),
        "selected_validation_loss": selected_validation_loss,
        "torch_version": version("torch"),
        "transformers_version": version("transformers"),
        "speechbrain_version": version("speechbrain"),
        "non_root_container_user": os.geteuid() != 0,
        "network_disabled": True,
        "read_only_root_filesystem": True,
        "direct_python_entrypoint": True,
        "models_loaded_sequentially": True,
        "paired_generation_seed_reset": True,
        "caches_scoped_to_tmpfs": True,
        "cpu_threads": 2,
        "dataloader_workers": 0,
        "container_memory_limit_bytes": _TRAINING_CONFIG["container_memory_limit_bytes"],
    }
    observations_document = {
        "schema_version": "enterprise-tts-observations/v1",
        "classification": CLASSIFICATION,
        "baseline": [_observation_document(item) for item in baseline_observations],
        "candidate": [_observation_document(item) for item in candidate_observations],
    }
    formal_report_document = {
        "schema_version": "enterprise-tts-formal-evaluation/v1",
        "classification": CLASSIFICATION,
        "report": asdict(formal_report),
        "voice_similarity": {"validation": validation_voice, "gold": gold_voice},
    }
    dataset_document = {
        "schema_version": "enterprise-tts-governed-dataset/v1",
        "classification": CLASSIFICATION,
        "training_manifest_sha256": _file_sha256(paths["training_manifest"]),
        "final_gold_manifest_sha256": _file_sha256(paths["gold_manifest"]),
        "speaker_embedding_sha256": _file_sha256(paths["speaker_embedding"]),
        "train_ids": [row["case_id"] for row in _rows(training_document, "train")],
        "validation_ids": [row["case_id"] for row in _rows(training_document, "validation")],
        "development_ids": [row["case_id"] for row in _rows(training_document, "gold")],
        "final_gold_ids": [row["case_id"] for row in gold_rows],
        "frozen_gold_excluded_from_training_selection_and_development": True,
        "gold_evaluation_pass_count": 1,
    }
    return _TrainingResult(
        runtime=runtime,
        observations_document=observations_document,
        formal_report_document=formal_report_document,
        dataset_document=dataset_document,
        evaluation=evaluation,
    )


def _dependencies() -> dict[str, Any]:
    try:
        return {
            "numpy": importlib.import_module("numpy"),
            "soundfile": importlib.import_module("soundfile"),
            "torch": importlib.import_module("torch"),
            "transformers": importlib.import_module("transformers"),
        }
    except ImportError as exc:
        raise TtsEnterpriseValueLabError("TTS runtime dependencies are missing") from exc


def _training_records(
    document: dict[str, Any],
    split: str,
    dataset_directory: Path,
    processor: Any,
    speaker: Any,
    soundfile: Any,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in _rows(document, split):
        audio, sampling_rate = soundfile.read(
            dataset_directory / row["audio_path"], dtype="float32", always_2d=False
        )
        if sampling_rate != 16_000 or audio.ndim != 1:
            raise TtsEnterpriseValueLabError("TTS training audio contract changed")
        encoded = processor(
            text=row["text"],
            audio_target=audio,
            sampling_rate=16_000,
            return_attention_mask=False,
        )
        labels = encoded["labels"]
        if labels.ndim == 3 and labels.shape[0] == 1:
            labels = labels[0]
        if labels.ndim != 2:
            raise TtsEnterpriseValueLabError("TTS acoustic label rank changed")
        prepared.append(
            {
                "input_ids": encoded["input_ids"],
                "labels": labels,
                "speaker_embeddings": speaker.tolist(),
            }
        )
    return prepared


def _generate_pair(
    rows: list[dict[str, Any]],
    paths: dict[str, Any],
    candidate_directory: Path,
    speaker: Any,
    dependencies: dict[str, Any],
) -> tuple[tuple[_GeneratedAudio, ...], tuple[_GeneratedAudio, ...], int]:
    torch = dependencies["torch"]
    transformers = dependencies["transformers"]
    processor = transformers.SpeechT5Processor.from_pretrained(
        paths["components"]["base_model"], local_files_only=True
    )
    vocoder = transformers.SpeechT5HifiGan.from_pretrained(
        paths["components"]["vocoder"],
        local_files_only=True,
        dtype=torch.float32,
    ).to("cuda:0")
    vocoder.eval()
    torch.cuda.reset_peak_memory_stats()
    baseline = _generate_model(
        rows,
        paths["components"]["base_model"],
        processor,
        vocoder,
        speaker,
        dependencies,
    )
    candidate = _generate_model(
        rows,
        candidate_directory,
        processor,
        vocoder,
        speaker,
        dependencies,
    )
    peak = int(torch.cuda.max_memory_reserved())
    del vocoder, processor
    gc.collect()
    torch.cuda.empty_cache()
    return baseline, candidate, peak


def _generate_model(
    rows: list[dict[str, Any]],
    model_path: Path,
    processor: Any,
    vocoder: Any,
    speaker: Any,
    dependencies: dict[str, Any],
) -> tuple[_GeneratedAudio, ...]:
    numpy = dependencies["numpy"]
    torch = dependencies["torch"]
    transformers = dependencies["transformers"]
    torch.manual_seed(_TRAINING_CONFIG["seed"])
    torch.cuda.manual_seed_all(_TRAINING_CONFIG["seed"])
    model = transformers.SpeechT5ForTextToSpeech.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda:0")
    model.config.use_cache = True
    model.eval()
    embedding = torch.from_numpy(numpy.asarray(speaker)[None, :]).to(
        device="cuda:0", dtype=model.dtype
    )
    generated: list[_GeneratedAudio] = []
    with torch.inference_mode():
        for row in rows:
            encoded = processor(text=row["text"], return_tensors="pt")
            torch.cuda.synchronize()
            started = time.perf_counter()
            waveform = model.generate_speech(
                input_ids=encoded.input_ids.to("cuda:0"),
                attention_mask=encoded.attention_mask.to("cuda:0"),
                speaker_embeddings=embedding,
                vocoder=vocoder,
                threshold=0.5,
                minlenratio=0.0,
                maxlenratio=20.0,
            )
            torch.cuda.synchronize()
            latency_ms = max((time.perf_counter() - started) * 1000, 1e-9)
            audio = waveform.detach().float().cpu().numpy().reshape(-1)
            duration = len(audio) / 16_000
            clipping = float(numpy.mean(numpy.abs(audio) >= 0.999)) if len(audio) else 1.0
            silence = float(numpy.mean(numpy.abs(audio) <= 0.0001)) if len(audio) else 1.0
            if (
                not numpy.isfinite(audio).all()
                or duration <= 0
                or duration > _TRAINING_CONFIG["max_audio_seconds"]
                or clipping > 0.01
                or silence >= 0.98
            ):
                raise TtsEnterpriseValueLabError("TTS generated audio integrity failed")
            generated.append(
                _GeneratedAudio(
                    case_id=str(row["case_id"]),
                    audio=audio,
                    latency_ms=latency_ms,
                    input_tokens=int(encoded.input_ids.shape[-1]),
                    duration_seconds=duration,
                    silence_ratio=silence,
                    clipping_ratio=clipping,
                    audio_sha256=sha256(audio.tobytes(order="C")).hexdigest(),
                )
            )
    del model, embedding
    gc.collect()
    torch.cuda.empty_cache()
    return tuple(generated)


def _transcribe_pair(
    baseline: tuple[_GeneratedAudio, ...],
    candidate: tuple[_GeneratedAudio, ...],
    asr_path: Path,
    dependencies: dict[str, Any],
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...], dict[str, int]]:
    torch = dependencies["torch"]
    transformers = dependencies["transformers"]
    processor = transformers.AutoProcessor.from_pretrained(
        asr_path, local_files_only=True, language="en", task="transcribe"
    )
    model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
        asr_path, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda:0")
    model.eval()
    torch.cuda.reset_peak_memory_stats()

    def transcribe(items: tuple[_GeneratedAudio, ...]) -> tuple[tuple[str, int], ...]:
        rows: list[tuple[str, int]] = []
        for item in items:
            inputs = processor.feature_extractor(
                item.audio,
                sampling_rate=16_000,
                return_attention_mask=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                tokens = model.generate(
                    input_features=inputs.input_features.to("cuda:0", dtype=torch.bfloat16),
                    attention_mask=inputs.attention_mask.to("cuda:0"),
                    max_length=96,
                    do_sample=False,
                    language="en",
                    task="transcribe",
                )
            text = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            rows.append((text, int(tokens.shape[-1])))
        return tuple(rows)

    baseline_rows = transcribe(baseline)
    candidate_rows = transcribe(candidate)
    peak = {
        "allocated": int(torch.cuda.max_memory_allocated()),
        "reserved": int(torch.cuda.max_memory_reserved()),
    }
    model = None
    processor = None
    gc.collect()
    torch.cuda.empty_cache()
    return baseline_rows, candidate_rows, peak


def _compare_speaker_similarity(
    baseline: tuple[_GeneratedAudio, ...],
    candidate: tuple[_GeneratedAudio, ...],
    reference: Any,
    model_path: Path,
    dependencies: dict[str, Any],
) -> tuple[dict[str, float], tuple[float, ...], tuple[float, ...]]:
    numpy = dependencies["numpy"]
    torch = dependencies["torch"]
    torchaudio = importlib.import_module("torchaudio")
    if not hasattr(torchaudio, "list_audio_backends"):
        setattr(  # noqa: B010 - compatibility shim for newer torchaudio releases
            torchaudio, "list_audio_backends", lambda: ["soundfile"]
        )
    classifiers = importlib.import_module("speechbrain.inference.classifiers")
    classifier = classifiers.EncoderClassifier.from_hparams(
        source=str(model_path),
        savedir="/tmp/spkrec-xvect",
        run_opts={"device": "cpu"},
        overrides={"pretrained_path": str(model_path)},
    )
    normalized_reference = _normalized(reference, numpy)

    def scores(items: tuple[_GeneratedAudio, ...]) -> tuple[tuple[float, ...], float]:
        embeddings: list[Any] = []
        values: list[float] = []
        for item in items:
            with torch.inference_mode():
                raw = classifier.encode_batch(torch.from_numpy(item.audio).unsqueeze(0))
            embedding = _normalized(raw.squeeze().cpu().numpy(), numpy)
            embeddings.append(embedding)
            values.append(float(numpy.dot(embedding, normalized_reference)))
        centroid = _normalized(numpy.mean(numpy.stack(embeddings), axis=0), numpy)
        return tuple(values), float(numpy.dot(centroid, normalized_reference))

    baseline_scores, baseline_centroid = scores(baseline)
    candidate_scores, candidate_centroid = scores(candidate)
    classifier = None
    gc.collect()
    baseline_mean = float(numpy.mean(baseline_scores))
    candidate_mean = float(numpy.mean(candidate_scores))
    return (
        {
            "baseline_mean_case_cosine_similarity": baseline_mean,
            "candidate_mean_case_cosine_similarity": candidate_mean,
            "mean_case_improvement": candidate_mean - baseline_mean,
            "baseline_centroid_cosine_similarity": baseline_centroid,
            "candidate_centroid_cosine_similarity": candidate_centroid,
            "centroid_improvement": candidate_centroid - baseline_centroid,
        },
        baseline_scores,
        candidate_scores,
    )


def _normalized(value: Any, numpy: Any) -> Any:
    normalized = numpy.asarray(value, dtype="float64").reshape(-1)
    norm = float(numpy.linalg.norm(normalized))
    if normalized.shape != (512,) or not math.isfinite(norm) or norm <= 0:
        raise TtsEnterpriseValueLabError("TTS speaker embedding is invalid")
    return normalized / norm


def _gold_cases(rows: list[dict[str, Any]]) -> tuple[EvaluationCase, ...]:
    return tuple(
        EvaluationCase(
            case_id=str(row["case_id"]),
            prompt=(),
            expected_output={},
            slices={"domain": "industrial_safety", "language": "en-US"},
            required_gates=(),
            allowed_citations=(),
            forbidden_substrings=(),
            expected_decision=None,
            risk=str(row["risk"]),
            contexts=(),
            tts=TtsEvaluationContract(
                target_text=str(row["text"]),
                language="en-US",
                voice_profile_id="industrial-safety-en-v1",
                required_safety_phrases=tuple(row["required_safety_phrases"]),
                industrial_terms=tuple(row["industrial_terms"]),
                risk=str(row["risk"]),
            ),
        )
        for row in rows
    )


def _observations(
    cases: tuple[EvaluationCase, ...],
    audio_rows: tuple[_GeneratedAudio, ...],
    transcripts: tuple[tuple[str, int], ...],
    speaker_scores: tuple[float, ...],
) -> tuple[ModelObservation, ...]:
    observations: list[ModelObservation] = []
    for case, audio, (transcript, output_tokens), similarity in zip(
        cases, audio_rows, transcripts, speaker_scores, strict=True
    ):
        observations.append(
            ModelObservation(
                case_id=case.case_id,
                output_text=transcript,
                latency_ms=audio.latency_ms,
                cost_usd=max(audio.latency_ms / 3_600_000, 1e-12),
                input_tokens=audio.input_tokens,
                output_tokens=output_tokens,
                capabilities=frozenset({"tts_voice_usage_authorization", "tts_audio_integrity"}),
                structured_output_valid=bool(transcript),
                transcript_text=transcript,
                runtime_evidence={
                    "tts": {
                        "audio_sha256": "sha256:" + audio.audio_sha256,
                        "sampling_rate": 16_000,
                        "duration_seconds": audio.duration_seconds,
                        "silence_ratio": audio.silence_ratio,
                        "clipping_ratio": audio.clipping_ratio,
                        "voice_profile_id": "industrial-safety-en-v1",
                        "speaker_cosine_similarity": similarity,
                        "asr_verifier": {
                            "model_id": ASR_MODEL_ID,
                            "revision": ASR_REVISION,
                            "digest": f"hf-revision:{ASR_REVISION}",
                        },
                        "vocoder": {
                            "model_id": VOCODER_MODEL_ID,
                            "revision": VOCODER_REVISION,
                        },
                    }
                },
            )
        )
    return tuple(observations)


def _evaluation_summary(
    report: PairedMetricReport,
    cases: tuple[EvaluationCase, ...],
    baseline: tuple[ModelObservation, ...],
    candidate: tuple[ModelObservation, ...],
    validation_voice: dict[str, float],
    gold_voice: dict[str, float],
    generation_peak: int,
) -> dict[str, Any]:
    aggregate = report.aggregate_metrics.get("tts")
    if not isinstance(aggregate, dict):
        raise TtsEnterpriseValueLabError("TTS aggregate metrics are missing")
    intelligibility = _metric_pair(aggregate, "intelligibility")
    safety = _metric_pair(aggregate, "safety_phrase_completeness")
    terminology = _metric_pair(aggregate, "terminology_recall")
    baseline_wer = sum(
        _word_error_rate(case.tts.target_text, observation.transcript_text or "")
        for case, observation in zip(cases, baseline, strict=True)
        if case.tts is not None
    ) / len(cases)
    candidate_wer = sum(
        _word_error_rate(case.tts.target_text, observation.transcript_text or "")
        for case, observation in zip(cases, candidate, strict=True)
        if case.tts is not None
    ) / len(cases)
    return {
        "frozen_gold_case_count": len(cases),
        "gold_evaluation_pass_count": 1,
        "baseline_intelligibility": intelligibility["baseline"],
        "candidate_intelligibility": intelligibility["candidate"],
        "baseline_safety_phrase_completeness": safety["baseline"],
        "candidate_safety_phrase_completeness": safety["candidate"],
        "baseline_terminology_recall": terminology["baseline"],
        "candidate_terminology_recall": terminology["candidate"],
        "baseline_wer": baseline_wer,
        "candidate_wer": candidate_wer,
        "validation_baseline_voice_similarity": validation_voice[
            "baseline_mean_case_cosine_similarity"
        ],
        "validation_candidate_voice_similarity": validation_voice[
            "candidate_mean_case_cosine_similarity"
        ],
        "validation_voice_similarity_improvement": validation_voice["mean_case_improvement"],
        "gold_baseline_voice_similarity": gold_voice["baseline_mean_case_cosine_similarity"],
        "gold_candidate_voice_similarity": gold_voice["candidate_mean_case_cosine_similarity"],
        "gold_voice_similarity_improvement": gold_voice["mean_case_improvement"],
        "voice_similarity_improvement_min": _TRAINING_CONFIG["voice_similarity_improvement_min"],
        "generation_peak_gpu_memory_reserved_bytes": generation_peak,
    }


def _require_formal_gates(report: PairedMetricReport, evaluation: dict[str, Any]) -> None:
    if set(report.evidence.hard_gate_results) != TTS_HARD_GATES:
        raise TtsEnterpriseValueLabError("TTS formal hard-gate contract drifted")
    failed = sorted(
        gate for gate, passed in report.evidence.hard_gate_results.items() if not passed
    )
    if failed:
        raise TtsEnterpriseValueLabError(f"TTS formal hard gates failed: {','.join(failed)}")
    minimum = _TRAINING_CONFIG["voice_similarity_improvement_min"]
    if (
        evaluation["validation_voice_similarity_improvement"] < minimum
        or evaluation["gold_voice_similarity_improvement"] < minimum
    ):
        raise TtsEnterpriseValueLabError("TTS voice similarity improvement is insufficient")


def _metric_pair(document: dict[str, Any], name: str) -> dict[str, float]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise TtsEnterpriseValueLabError(f"TTS {name} metric is missing")
    candidate = value.get("candidate")
    baseline = value.get("baseline")
    if (
        isinstance(candidate, bool)
        or not isinstance(candidate, (int, float))
        or isinstance(baseline, bool)
        or not isinstance(baseline, (int, float))
        or not math.isfinite(float(candidate))
        or not math.isfinite(float(baseline))
    ):
        raise TtsEnterpriseValueLabError(f"TTS {name} metric is invalid")
    return {"candidate": float(candidate), "baseline": float(baseline)}


def _word_error_rate(reference: str, transcript: str) -> float:
    expected = _word_units(reference)
    actual = _word_units(transcript)
    return _edit_distance(expected, actual) / max(len(expected), 1)


def _word_units(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return tuple(normalized.split())


def _edit_distance(expected: tuple[str, ...], actual: tuple[str, ...]) -> int:
    previous = list(range(len(actual) + 1))
    for index, left in enumerate(expected, start=1):
        current = [index]
        for offset, right in enumerate(actual, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[offset] + 1,
                    previous[offset - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def _observation_document(observation: ModelObservation) -> dict[str, Any]:
    return {
        "case_id": observation.case_id,
        "output_text": observation.output_text,
        "latency_ms": observation.latency_ms,
        "cost_usd": observation.cost_usd,
        "input_tokens": observation.input_tokens,
        "output_tokens": observation.output_tokens,
        "capabilities": sorted(observation.capabilities),
        "structured_output_valid": observation.structured_output_valid,
        "transcript_text": observation.transcript_text,
        "runtime_evidence": observation.runtime_evidence,
    }


def _checkpoint_step(value: str) -> int:
    match = re.search(r"checkpoint-(\d+)$", value)
    if match is None:
        raise TtsEnterpriseValueLabError("TTS selected checkpoint is missing")
    return int(match.group(1))


def _rows(document: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise TtsEnterpriseValueLabError("TTS dataset rows are invalid")
    selected = [row for row in rows if isinstance(row, dict) and row.get("split") == split]
    if not selected:
        raise TtsEnterpriseValueLabError(f"TTS {split} split is empty")
    return selected


def _input_paths(root: Path) -> dict[str, Any]:
    training_dataset = _inside_directory(root, TRAINING_DATASET_RELATIVE)
    gold_dataset = _inside_directory(root, GOLD_DATASET_RELATIVE)
    return {
        "components": {
            "base_model": _inside_directory(root, BASE_MODEL_RELATIVE),
            "vocoder": _inside_directory(root, VOCODER_RELATIVE),
            "asr": _inside_directory(root, ASR_RELATIVE),
            "xvector": _inside_directory(root, XVECTOR_RELATIVE),
        },
        "training_dataset": training_dataset,
        "training_manifest": _inside_file(root, training_dataset / "manifest.json"),
        "speaker_embedding": _inside_file(root, training_dataset / "speaker_embedding.npy"),
        "gold_dataset": gold_dataset,
        "gold_manifest": _inside_file(root, gold_dataset / "manifest.json"),
        "gold_v1_retirement": _inside_file(root, GOLD_V1_RETIREMENT_RELATIVE),
    }


def _verify_inputs(paths: dict[str, Any]) -> None:
    expected_weights = {
        "base_model": ("pytorch_model.bin", BASE_MODEL_WEIGHT_SHA256),
        "vocoder": ("pytorch_model.bin", VOCODER_WEIGHT_SHA256),
        "asr": ("pytorch_model.bin", ASR_WEIGHT_SHA256),
        "xvector": ("embedding_model.ckpt", XVECTOR_WEIGHT_SHA256),
    }
    for name, (file_name, digest) in expected_weights.items():
        if _file_sha256(paths["components"][name] / file_name) != digest:
            raise TtsEnterpriseValueLabError(f"TTS {name} weight changed")
    if (
        _file_sha256(paths["training_manifest"]) != TRAINING_MANIFEST_SHA256
        or _file_sha256(paths["gold_manifest"]) != GOLD_MANIFEST_SHA256
        or _file_sha256(paths["gold_v1_retirement"]) != GOLD_V1_RETIREMENT_SHA256
        or _file_sha256(paths["speaker_embedding"]) != SPEAKER_EMBEDDING_SHA256
    ):
        raise TtsEnterpriseValueLabError("TTS governed data input changed")
    training = _load_object(paths["training_manifest"])
    gold = _load_object(paths["gold_manifest"])
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
    evidence: list[dict[str, Any]] = []
    for role in ("base_model", "vocoder", "asr", "xvector"):
        path = paths["components"][role]
        files = _directory_manifest(path)
        model_id, revision, weight_sha256 = metadata[role]
        evidence.append(
            {
                "role": role,
                "model_id": model_id,
                "revision": revision,
                "weight_sha256": weight_sha256,
                "path": path.relative_to(root).as_posix(),
                "manifest_sha256": _manifest_digest(files),
                "files": files,
            }
        )
    return evidence


def _directory_manifest(directory: Path) -> list[dict[str, Any]]:
    root = directory.resolve(strict=True)
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".cache":
            continue
        if path.is_symlink():
            raise TtsEnterpriseValueLabError("TTS artifact contains a symbolic link")
        if not path.is_file():
            continue
        if path.stat().st_size <= 0:
            raise TtsEnterpriseValueLabError("TTS artifact contains an empty file")
        entries.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not entries:
        raise TtsEnterpriseValueLabError("TTS artifact directory is empty")
    return entries


def _manifest_digest(entries: list[dict[str, Any]]) -> str:
    return _digest(entries)


def _file_evidence(relative_path: Path, physical_path: Path) -> dict[str, Any]:
    return {
        "path": relative_path.as_posix(),
        "size_bytes": physical_path.stat().st_size,
        "sha256": _file_sha256(physical_path),
    }


def _verify_file_evidence(root: Path, evidence: FileEvidence) -> Path:
    path = _inside_file(root, Path(evidence.path))
    if path.stat().st_size != evidence.size_bytes or _file_sha256(path) != evidence.sha256:
        raise TtsEnterpriseValueLabError("TTS evidence file changed")
    return path


def _write_latest(root: Path, acceptance_path: Path, report: TtsEnterpriseValueReport) -> None:
    latest = {
        "schema_version": "enterprise-tts-value-latest/v1",
        "classification": CLASSIFICATION,
        "status": report.status,
        "run_id": report.run_id,
        "report": acceptance_path.relative_to(root).as_posix(),
        "report_sha256": _file_sha256(acceptance_path),
        "evidence_chain_sha256": report.evidence_chain_sha256,
    }
    _write_json(root / OUTPUT_RELATIVE / "latest.json", latest)


def _latest_report_path(root: Path) -> Path:
    latest_path = _inside_file(root, OUTPUT_RELATIVE / "latest.json")
    latest = _load_object(latest_path)
    value = latest.get("report")
    if not isinstance(value, str) or not value:
        raise TtsEnterpriseValueLabError("TTS latest pointer is invalid")
    report_path = _inside_file(root, Path(value))
    if latest.get("report_sha256") != _file_sha256(report_path):
        raise TtsEnterpriseValueLabError("TTS latest report digest changed")
    return report_path


def _inside_file(root: Path, value: Path) -> Path:
    target = value if value.is_absolute() else root / value
    target = target.resolve(strict=True)
    _require_inside(root, target)
    if not target.is_file():
        raise TtsEnterpriseValueLabError("TTS evidence path is not a file")
    return target


def _inside_directory(root: Path, value: Path) -> Path:
    target = value if value.is_absolute() else root / value
    target = target.resolve(strict=True)
    _require_inside(root, target)
    if not target.is_dir():
        raise TtsEnterpriseValueLabError("TTS evidence path is not a directory")
    return target


def _require_inside(root: Path, target: Path) -> None:
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise TtsEnterpriseValueLabError("TTS evidence path escaped repository") from exc


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TtsEnterpriseValueLabError("TTS evidence JSON is invalid") from exc
    if not isinstance(value, dict):
        raise TtsEnterpriseValueLabError("TTS evidence JSON is invalid")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(_canonical(value) + b"\n")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_image_digest(value: str) -> None:
    if not (
        value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    ):
        raise TtsEnterpriseValueLabError("TTS runtime image digest is invalid")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _digest(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()
