"""Compile immutable experiment intent into strict training runtime profiles."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Literal, TypeAlias, cast

from industrial_ops_agent.persistence.models import TrainingExperimentRecord
from industrial_ops_agent.training.distributed import (
    CompiledDistributedProfile,
    DistributedProfileError,
    compile_distributed_profile,
)
from industrial_ops_agent.training.reward_contracts import (
    LEGACY_REWARD_CONTRACT_VERSION,
    RewardContractError,
    RewardContractVersion,
    reward_contract,
    validate_reward_contract_digest,
)


class TrainingConfigurationError(ValueError):
    """The registered experiment cannot be executed without changing its intent."""


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    load_in_4bit: bool
    quant_type: Literal["nf4"]
    use_double_quant: bool
    compute_dtype: Literal["bfloat16", "float16"]


@dataclass(frozen=True, slots=True)
class ResumeCheckpointConfig:
    source_experiment_id: str
    artifact_id: str
    content_hash: str
    checkpoint_path: str


@dataclass(frozen=True, slots=True)
class CompiledPeftConfig:
    method: str
    base_model_revision: str
    max_steps: int
    effective_batch_size: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    evaluation_interval: int
    save_steps: int
    logging_steps: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    lr_scheduler_type: str
    max_sequence_length: int
    gradient_checkpointing: bool
    precision: Literal["bfloat16", "float16"]
    seed: int
    data_seed: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_bias: Literal["none", "all", "lora_only"]
    target_modules: tuple[str, ...]
    gpu_hourly_cost_usd: float
    quantization: QuantizationConfig | None
    distributed: CompiledDistributedProfile
    resume_from_checkpoint: ResumeCheckpointConfig | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        import json

        payload = json.dumps(
            self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CompiledSftConfig(CompiledPeftConfig):
    method: Literal["LORA", "QLORA"]


@dataclass(frozen=True, slots=True)
class CompiledDpoConfig(CompiledPeftConfig):
    method: Literal["DPO"]
    preference_dataset_snapshot_id: str
    reference_model_digest: str
    beta: float
    loss_type: Literal["sigmoid", "hinge", "ipo"]


@dataclass(frozen=True, slots=True)
class CompiledGrpoConfig(CompiledPeftConfig):
    method: Literal["GRPO"]
    reward_contract_version: RewardContractVersion
    reward_contract_digest: str
    group_size: int
    beta: float
    max_completion_length: int


@dataclass(frozen=True, slots=True)
class CompiledPpoConfig(CompiledPeftConfig):
    method: Literal["PPO"]
    reward_model_id: str
    reward_model_revision: str
    reward_model_digest: str
    value_model_id: str
    value_model_revision: str
    value_model_digest: str
    kl_coefficient: float
    total_episodes: int
    response_length: int
    num_ppo_epochs: int
    num_mini_batches: int


@dataclass(frozen=True, slots=True)
class CompiledSpecializedConfig:
    method: str
    base_model_revision: str
    max_steps: int
    effective_batch_size: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    evaluation_interval: int
    save_steps: int
    logging_steps: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    lr_scheduler_type: str
    max_sequence_length: int
    precision: Literal["bfloat16", "float16"]
    seed: int
    data_seed: int
    gpu_hourly_cost_usd: float
    retrieval_contract_version: Literal["industrial-retrieval-triplet-v1"]
    hard_negatives_per_query: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        import json

        payload = json.dumps(
            self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CompiledEmbeddingConfig(CompiledSpecializedConfig):
    method: Literal["EMBEDDING"]
    loss_type: Literal["multiple_negatives_ranking"]
    similarity_scale: float


@dataclass(frozen=True, slots=True)
class CompiledRerankerConfig(CompiledSpecializedConfig):
    method: Literal["RERANKER"]
    loss_type: Literal["binary_cross_entropy"]
    positive_weight: float


@dataclass(frozen=True, slots=True)
class VlmCurriculumConfig:
    schema_version: Literal["industrial-vlm-hard-negative-curriculum-v1"]
    target_candidate_ids: tuple[str, ...]
    repeat_factor: int
    selection_digest: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CompiledVlmConfig(CompiledPeftConfig):
    method: Literal["VLM"]
    vision_contract_version: Literal["industrial-vision-instruction-v1"]
    max_image_pixels: int
    curriculum: VlmCurriculumConfig | None
    resume_ignore_data_skip: bool
    resume_optimizer_learning_rate_override: bool


@dataclass(frozen=True, slots=True)
class CompiledAsrConfig(CompiledPeftConfig):
    method: Literal["ASR"]
    audio_contract_version: Literal["industrial-asr-transcript-v1"]
    sampling_rate: Literal[16000]
    language: str
    task: Literal["transcribe"]
    max_audio_seconds: float


@dataclass(frozen=True, slots=True)
class CompiledTtsConfig:
    method: Literal["TTS"]
    base_model_revision: str
    model_family: Literal["SPEECHT5"]
    speech_contract_version: Literal["industrial-tts-speech-v1"]
    sampling_rate: Literal[16000]
    language: str
    voice_profile_id: str
    speaker_embedding_dimension: int
    max_audio_seconds: float
    max_steps: int
    effective_batch_size: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    evaluation_interval: int
    save_steps: int
    logging_steps: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    lr_scheduler_type: str
    max_sequence_length: int
    precision: Literal["bfloat16", "float16"]
    seed: int
    data_seed: int
    gpu_hourly_cost_usd: float
    distributed: CompiledDistributedProfile

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        import json

        payload = json.dumps(
            self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CompiledTimeseriesTransformerConfig:
    method: Literal["TIMESERIES_TRANSFORMER"]
    base_model_revision: Literal["timeseries-transformer-v1"]
    max_steps: int
    effective_batch_size: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    evaluation_interval: int
    save_steps: int
    logging_steps: int
    learning_rate: float
    weight_decay: float
    precision: Literal["bfloat16", "float16"]
    seed: int
    data_seed: int
    gpu_hourly_cost_usd: float
    sequence_contract_version: Literal["industrial-telemetry-sequence-v1"]
    signal_order: tuple[str, ...]
    max_sequence_length: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    objective: Literal["masked_reconstruction"]
    mask_probability: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        import json

        payload = json.dumps(
            self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CompiledRulTransformerConfig:
    method: Literal["RUL_TRANSFORMER"]
    base_model_revision: Literal["rul-transformer-v1"]
    max_steps: int
    effective_batch_size: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    evaluation_interval: int
    save_steps: int
    logging_steps: int
    learning_rate: float
    weight_decay: float
    precision: Literal["bfloat16", "float16"]
    seed: int
    data_seed: int
    gpu_hourly_cost_usd: float
    sequence_contract_version: Literal["industrial-rul-sequence-v1"]
    signal_order: tuple[str, ...]
    max_sequence_length: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    objective: Literal["quantile_regression"]
    quantiles: tuple[float, float, float]
    target_transform: Literal["log1p_minutes"]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        import json

        payload = json.dumps(
            self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        return sha256(payload).hexdigest()


CompiledTrainingConfig: TypeAlias = (
    CompiledSftConfig
    | CompiledDpoConfig
    | CompiledGrpoConfig
    | CompiledPpoConfig
    | CompiledEmbeddingConfig
    | CompiledRerankerConfig
    | CompiledVlmConfig
    | CompiledAsrConfig
    | CompiledTtsConfig
    | CompiledTimeseriesTransformerConfig
    | CompiledRulTransformerConfig
)


def compile_training_config(
    experiment: TrainingExperimentRecord,
) -> CompiledTrainingConfig:
    """Compile the registered method into one executable and immutable profile."""

    if experiment.method in {"LORA", "QLORA"}:
        return compile_sft_config(experiment)
    if experiment.method == "DPO":
        return _compile_dpo_config(experiment)
    if experiment.method == "GRPO":
        return _compile_grpo_config(experiment)
    if experiment.method == "PPO":
        return _compile_ppo_config(experiment)
    if experiment.method == "EMBEDDING":
        return _compile_embedding_config(experiment)
    if experiment.method == "RERANKER":
        return _compile_reranker_config(experiment)
    if experiment.method == "VLM":
        return _compile_vlm_config(experiment)
    if experiment.method == "ASR":
        return _compile_asr_config(experiment)
    if experiment.method == "TTS":
        return _compile_tts_config(experiment)
    if experiment.method == "TIMESERIES_TRANSFORMER":
        return _compile_timeseries_transformer_config(experiment)
    if experiment.method == "RUL_TRANSFORMER":
        return _compile_rul_transformer_config(experiment)
    raise TrainingConfigurationError("training_method_has_no_execution_profile")


def compile_sft_config(experiment: TrainingExperimentRecord) -> CompiledSftConfig:
    """Validate the governed LoRA/QLoRA SFT contract without GPU imports."""

    if experiment.method not in {"LORA", "QLORA"}:
        raise TrainingConfigurationError("sft_profile_requires_lora_or_qlora")
    method = cast(Literal["LORA", "QLORA"], experiment.method)
    common = _compile_peft_common(
        experiment,
        method=method,
        quantized=method == "QLORA",
        default_learning_rate=2e-4,
    )
    return CompiledSftConfig(**common, method=method)


def _compile_dpo_config(experiment: TrainingExperimentRecord) -> CompiledDpoConfig:
    config = experiment.training_config
    snapshot_id = _required_text(config, "preference_dataset_snapshot_id", max_length=255)
    if snapshot_id != experiment.dataset_snapshot_id:
        raise TrainingConfigurationError("preference_snapshot_must_match_experiment_snapshot")
    reference_digest = _required_text(config, "reference_model_digest", max_length=255)
    if reference_digest != experiment.base_model_digest:
        raise TrainingConfigurationError("dpo_reference_must_match_frozen_base_model")
    common = _compile_peft_common(
        experiment,
        method="DPO",
        quantized=_boolean(config, "load_in_4bit", default=False),
        default_learning_rate=5e-7,
    )
    return CompiledDpoConfig(
        **common,
        method="DPO",
        preference_dataset_snapshot_id=snapshot_id,
        reference_model_digest=reference_digest,
        beta=_bounded_float(config, "beta", default=0.1, lower=0.000001),
        loss_type=_choice(config, "loss_type", {"sigmoid", "hinge", "ipo"}, default="sigmoid"),
    )


def _compile_grpo_config(experiment: TrainingExperimentRecord) -> CompiledGrpoConfig:
    config = experiment.training_config
    reward_contract_version = _required_text(config, "reward_contract_version", max_length=64)
    configured_digest = config.get("reward_contract_digest")
    try:
        descriptor = reward_contract(reward_contract_version)
        # Experiments registered before the digest field existed remain replayable.
        if configured_digest is None and reward_contract_version == LEGACY_REWARD_CONTRACT_VERSION:
            configured_digest = descriptor.digest
        if not isinstance(configured_digest, str):
            raise RewardContractError("reward_contract_digest_is_required")
        validate_reward_contract_digest(reward_contract_version, configured_digest)
    except RewardContractError as exc:
        raise TrainingConfigurationError(str(exc)) from exc
    group_size = _positive_integer(config, "group_size")
    common = _compile_peft_common(
        experiment,
        method="GRPO",
        quantized=_boolean(config, "load_in_4bit", default=False),
        default_learning_rate=1e-6,
    )
    if common["effective_batch_size"] % group_size:
        raise TrainingConfigurationError("effective_batch_size_must_be_divisible_by_group_size")
    return CompiledGrpoConfig(
        **common,
        method="GRPO",
        reward_contract_version=descriptor.version,
        reward_contract_digest=descriptor.digest,
        group_size=group_size,
        beta=_bounded_float(config, "beta", default=0.04, lower=0),
        max_completion_length=_positive_integer(config, "max_completion_length", default=256),
    )


def _compile_ppo_config(experiment: TrainingExperimentRecord) -> CompiledPpoConfig:
    config = experiment.training_config
    reward_model_id = _required_text(config, "reward_model_id", max_length=255)
    reward_revision = _immutable_revision(config, "reward_model_revision")
    reward_digest = _model_revision_digest(config, "reward_model_digest", reward_revision)
    value_model_id = _required_text(config, "value_model_id", max_length=255)
    value_revision = _immutable_revision(config, "value_model_revision")
    value_digest = _model_revision_digest(config, "value_model_digest", value_revision)
    kl_controller = config.get("kl_controller")
    if not isinstance(kl_controller, dict) or kl_controller.get("type") != "fixed":
        raise TrainingConfigurationError("ppo_requires_fixed_kl_controller")
    kl_coefficient = _bounded_float(kl_controller, "coefficient", default=0.05, lower=0.000001)
    common = _compile_peft_common(
        experiment,
        method="PPO",
        quantized=False,
        default_learning_rate=3e-6,
    )
    num_mini_batches = _positive_integer(config, "num_mini_batches", default=1)
    if common["effective_batch_size"] % num_mini_batches:
        raise TrainingConfigurationError(
            "effective_batch_size_must_be_divisible_by_num_mini_batches"
        )
    return CompiledPpoConfig(
        **common,
        method="PPO",
        reward_model_id=reward_model_id,
        reward_model_revision=reward_revision,
        reward_model_digest=reward_digest,
        value_model_id=value_model_id,
        value_model_revision=value_revision,
        value_model_digest=value_digest,
        kl_coefficient=kl_coefficient,
        total_episodes=_positive_integer(config, "total_episodes", default=1024),
        response_length=_positive_integer(config, "response_length", default=256),
        num_ppo_epochs=_positive_integer(config, "num_ppo_epochs", default=4),
        num_mini_batches=num_mini_batches,
    )


def _compile_embedding_config(
    experiment: TrainingExperimentRecord,
) -> CompiledEmbeddingConfig:
    config = experiment.training_config
    common = _compile_specialized_common(experiment, default_learning_rate=2e-5)
    loss_type = _required_text(config, "loss_type", max_length=64)
    if loss_type != "multiple_negatives_ranking":
        raise TrainingConfigurationError("embedding_loss_type_is_not_supported")
    return CompiledEmbeddingConfig(
        **common,
        method="EMBEDDING",
        loss_type="multiple_negatives_ranking",
        similarity_scale=_bounded_float(config, "similarity_scale", default=20.0, lower=0.000001),
    )


def _compile_reranker_config(
    experiment: TrainingExperimentRecord,
) -> CompiledRerankerConfig:
    config = experiment.training_config
    common = _compile_specialized_common(experiment, default_learning_rate=2e-5)
    loss_type = _required_text(config, "loss_type", max_length=64)
    if loss_type != "binary_cross_entropy":
        raise TrainingConfigurationError("reranker_loss_type_is_not_supported")
    return CompiledRerankerConfig(
        **common,
        method="RERANKER",
        loss_type="binary_cross_entropy",
        positive_weight=_bounded_float(config, "positive_weight", default=1.0, lower=0.000001),
    )


def _compile_vlm_config(experiment: TrainingExperimentRecord) -> CompiledVlmConfig:
    config = experiment.training_config
    contract = _required_text(config, "vision_contract_version", max_length=64)
    if contract != "industrial-vision-instruction-v1":
        raise TrainingConfigurationError("vision_training_contract_is_not_supported")
    common = _compile_peft_common(
        experiment,
        method="VLM",
        quantized=_boolean(config, "load_in_4bit", default=True),
        default_learning_rate=2e-5,
    )
    curriculum = _compile_vlm_curriculum(config)
    resume_ignore_data_skip = _boolean(
        config,
        "resume_ignore_data_skip",
        default=False,
    )
    resume_optimizer_learning_rate_override = _boolean(
        config,
        "resume_optimizer_learning_rate_override",
        default=False,
    )
    if resume_ignore_data_skip and common["resume_from_checkpoint"] is None:
        raise TrainingConfigurationError("resume_ignore_data_skip_requires_resume_checkpoint")
    if resume_optimizer_learning_rate_override and common["resume_from_checkpoint"] is None:
        raise TrainingConfigurationError(
            "resume_optimizer_learning_rate_override_requires_checkpoint"
        )
    if (
        curriculum is not None
        and common["resume_from_checkpoint"] is not None
        and not resume_ignore_data_skip
    ):
        raise TrainingConfigurationError("resumed_vlm_curriculum_requires_ignore_data_skip")
    return CompiledVlmConfig(
        **common,
        method="VLM",
        vision_contract_version="industrial-vision-instruction-v1",
        max_image_pixels=_positive_integer(config, "max_image_pixels", default=4_194_304),
        curriculum=curriculum,
        resume_ignore_data_skip=resume_ignore_data_skip,
        resume_optimizer_learning_rate_override=(resume_optimizer_learning_rate_override),
    )


def _compile_vlm_curriculum(
    config: dict[str, Any],
) -> VlmCurriculumConfig | None:
    raw = config.get("vlm_curriculum")
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "target_candidate_ids",
        "repeat_factor",
        "selection_digest",
    }:
        raise TrainingConfigurationError("vlm_curriculum_contract_is_invalid")
    if raw.get("schema_version") != "industrial-vlm-hard-negative-curriculum-v1":
        raise TrainingConfigurationError("vlm_curriculum_schema_is_not_supported")
    candidate_ids = raw.get("target_candidate_ids")
    if (
        not isinstance(candidate_ids, list)
        or not candidate_ids
        or len(candidate_ids) > 256
        or not all(
            isinstance(candidate_id, str) and candidate_id.strip() and len(candidate_id) <= 128
            for candidate_id in candidate_ids
        )
    ):
        raise TrainingConfigurationError("vlm_curriculum_candidate_ids_are_invalid")
    normalized_ids = tuple(candidate_id.strip() for candidate_id in candidate_ids)
    if len(set(normalized_ids)) != len(normalized_ids):
        raise TrainingConfigurationError("vlm_curriculum_candidate_ids_are_not_unique")
    repeat_factor = _positive_integer(raw, "repeat_factor")
    if repeat_factor > 8:
        raise TrainingConfigurationError("vlm_curriculum_repeat_factor_is_out_of_range")
    import json

    contract = {
        "schema_version": "industrial-vlm-hard-negative-curriculum-v1",
        "target_candidate_ids": list(normalized_ids),
        "repeat_factor": repeat_factor,
    }
    expected_digest = sha256(
        json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if _required_text(raw, "selection_digest", max_length=64) != expected_digest:
        raise TrainingConfigurationError("vlm_curriculum_selection_digest_mismatch")
    return VlmCurriculumConfig(
        schema_version="industrial-vlm-hard-negative-curriculum-v1",
        target_candidate_ids=normalized_ids,
        repeat_factor=repeat_factor,
        selection_digest=expected_digest,
    )


def _compile_asr_config(experiment: TrainingExperimentRecord) -> CompiledAsrConfig:
    config = experiment.training_config
    contract = _required_text(config, "audio_contract_version", max_length=64)
    if contract != "industrial-asr-transcript-v1":
        raise TrainingConfigurationError("audio_training_contract_is_not_supported")
    sampling_rate = _positive_integer(config, "sampling_rate", default=16_000)
    if sampling_rate != 16_000:
        raise TrainingConfigurationError("asr_sampling_rate_must_be_16000")
    task = _choice(config, "task", {"transcribe"}, default="transcribe")
    common = _compile_peft_common(
        experiment,
        method="ASR",
        quantized=False,
        default_learning_rate=1e-5,
    )
    return CompiledAsrConfig(
        **common,
        method="ASR",
        audio_contract_version="industrial-asr-transcript-v1",
        sampling_rate=16000,
        language=_required_text(config, "language", max_length=32),
        task=task,
        max_audio_seconds=_bounded_float(
            config, "max_audio_seconds", default=30.0, lower=0.1, upper=30.0
        ),
    )


def _compile_tts_config(experiment: TrainingExperimentRecord) -> CompiledTtsConfig:
    config = experiment.training_config
    if _required_text(config, "model_family", max_length=32) != "SPEECHT5":
        raise TrainingConfigurationError("tts_model_family_is_not_supported")
    if (
        _required_text(config, "speech_contract_version", max_length=64)
        != "industrial-tts-speech-v1"
    ):
        raise TrainingConfigurationError("tts_speech_contract_is_not_supported")
    sampling_rate = _positive_integer(config, "sampling_rate", default=16_000)
    if sampling_rate != 16_000:
        raise TrainingConfigurationError("tts_sampling_rate_must_be_16000")
    try:
        distributed = compile_distributed_profile(
            experiment.distributed_profile,
            experiment.hardware_topology,
            method="TTS",
            quantized=False,
        )
    except DistributedProfileError as exc:
        raise TrainingConfigurationError(str(exc)) from exc
    revision = _immutable_revision(config, "base_model_revision")
    if experiment.base_model_digest != f"hf-revision:{revision}":
        raise TrainingConfigurationError("base_model_digest_does_not_bind_revision")
    effective_batch_size = _positive_integer(config, "effective_batch_size")
    per_device_train_batch_size = _positive_integer(
        config, "per_device_train_batch_size", default=1
    )
    distributed_micro_batch = per_device_train_batch_size * distributed.world_size
    if effective_batch_size % distributed_micro_batch:
        raise TrainingConfigurationError(
            "effective_batch_size_must_be_divisible_by_per_device_batch_and_world_size"
        )
    evaluation_interval = _positive_integer(config, "evaluation_interval")
    speaker_embedding_dimension = _positive_integer(config, "speaker_embedding_dimension")
    if not 2 <= speaker_embedding_dimension <= 4096:
        raise TrainingConfigurationError("speaker_embedding_dimension_is_out_of_range")
    return CompiledTtsConfig(
        method="TTS",
        base_model_revision=revision,
        model_family="SPEECHT5",
        speech_contract_version="industrial-tts-speech-v1",
        sampling_rate=16000,
        language=_required_text(config, "language", max_length=32),
        voice_profile_id=_required_text(config, "voice_profile_id", max_length=128),
        speaker_embedding_dimension=speaker_embedding_dimension,
        max_audio_seconds=_bounded_float(
            config, "max_audio_seconds", default=30.0, lower=0.1, upper=30.0
        ),
        max_steps=_positive_integer(config, "max_steps"),
        effective_batch_size=effective_batch_size,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=_positive_integer(
            config,
            "per_device_eval_batch_size",
            default=per_device_train_batch_size,
        ),
        gradient_accumulation_steps=effective_batch_size // distributed_micro_batch,
        evaluation_interval=evaluation_interval,
        save_steps=_positive_integer(config, "save_steps", default=evaluation_interval),
        logging_steps=_positive_integer(config, "logging_steps", default=10),
        learning_rate=_bounded_float(config, "learning_rate", default=1e-5, lower=0.0000000001),
        warmup_ratio=_bounded_float(config, "warmup_ratio", default=0.1, lower=0, upper=1),
        weight_decay=_bounded_float(config, "weight_decay", default=0.01, lower=0, upper=1),
        lr_scheduler_type=_choice(
            config,
            "lr_scheduler_type",
            {"linear", "cosine", "constant", "constant_with_warmup"},
            default="linear",
        ),
        max_sequence_length=_positive_integer(config, "max_sequence_length", default=512),
        precision=_choice(config, "precision", {"bfloat16", "float16"}, default="bfloat16"),
        seed=_first_seed(experiment.random_seeds),
        data_seed=_integer(config, "data_seed", default=_first_seed(experiment.random_seeds)),
        gpu_hourly_cost_usd=_bounded_float(config, "gpu_hourly_cost_usd", default=0.0, lower=0),
        distributed=distributed,
    )


def _compile_timeseries_transformer_config(
    experiment: TrainingExperimentRecord,
) -> CompiledTimeseriesTransformerConfig:
    config = experiment.training_config
    _validate_single_gpu(experiment)
    revision = _required_text(config, "base_model_revision", max_length=128)
    if revision != "timeseries-transformer-v1":
        raise TrainingConfigurationError("timeseries_architecture_revision_is_not_supported")
    if experiment.base_model_digest != f"architecture:{revision}":
        raise TrainingConfigurationError("base_model_digest_does_not_bind_architecture")
    contract = _required_text(config, "sequence_contract_version", max_length=64)
    if contract != "industrial-telemetry-sequence-v1":
        raise TrainingConfigurationError("telemetry_sequence_contract_is_not_supported")
    expected_signals = (
        "vibration_rms_mm_s",
        "bearing_temperature_c",
        "motor_current_a",
        "rpm",
    )
    signal_order_value = config.get("signal_order")
    if signal_order_value != list(expected_signals):
        raise TrainingConfigurationError("telemetry_signal_order_must_match_dataset_contract")
    effective_batch_size = _positive_integer(config, "effective_batch_size")
    per_device_train_batch_size = _positive_integer(
        config, "per_device_train_batch_size", default=16
    )
    if effective_batch_size % per_device_train_batch_size:
        raise TrainingConfigurationError(
            "effective_batch_size_must_be_divisible_by_per_device_batch"
        )
    d_model = _positive_integer(config, "d_model", default=128)
    nhead = _positive_integer(config, "nhead", default=4)
    if d_model % nhead:
        raise TrainingConfigurationError("d_model_must_be_divisible_by_nhead")
    evaluation_interval = _positive_integer(config, "evaluation_interval")
    objective = _choice(
        config,
        "objective",
        {"masked_reconstruction"},
        default="masked_reconstruction",
    )
    return CompiledTimeseriesTransformerConfig(
        method="TIMESERIES_TRANSFORMER",
        base_model_revision="timeseries-transformer-v1",
        max_steps=_positive_integer(config, "max_steps"),
        effective_batch_size=effective_batch_size,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=_positive_integer(
            config,
            "per_device_eval_batch_size",
            default=per_device_train_batch_size,
        ),
        gradient_accumulation_steps=effective_batch_size // per_device_train_batch_size,
        evaluation_interval=evaluation_interval,
        save_steps=_positive_integer(config, "save_steps", default=evaluation_interval),
        logging_steps=_positive_integer(config, "logging_steps", default=10),
        learning_rate=_bounded_float(config, "learning_rate", default=1e-4, lower=0.0000000001),
        weight_decay=_bounded_float(config, "weight_decay", default=0.01, lower=0, upper=1),
        precision=_choice(config, "precision", {"bfloat16", "float16"}, default="bfloat16"),
        seed=_first_seed(experiment.random_seeds),
        data_seed=_integer(config, "data_seed", default=_first_seed(experiment.random_seeds)),
        gpu_hourly_cost_usd=_bounded_float(config, "gpu_hourly_cost_usd", default=0.0, lower=0),
        sequence_contract_version="industrial-telemetry-sequence-v1",
        signal_order=expected_signals,
        max_sequence_length=_positive_integer(config, "max_sequence_length", default=256),
        d_model=d_model,
        nhead=nhead,
        num_layers=_positive_integer(config, "num_layers", default=3),
        dim_feedforward=_positive_integer(config, "dim_feedforward", default=256),
        dropout=_bounded_float(config, "dropout", default=0.1, lower=0, upper=0.9),
        objective=objective,
        mask_probability=_bounded_float(
            config, "mask_probability", default=0.15, lower=0.01, upper=0.9
        ),
    )


def _compile_rul_transformer_config(
    experiment: TrainingExperimentRecord,
) -> CompiledRulTransformerConfig:
    config = experiment.training_config
    _validate_single_gpu(experiment)
    revision = _required_text(config, "base_model_revision", max_length=128)
    if revision != "rul-transformer-v1":
        raise TrainingConfigurationError("rul_architecture_revision_is_not_supported")
    if experiment.base_model_digest != f"architecture:{revision}":
        raise TrainingConfigurationError("base_model_digest_does_not_bind_rul_architecture")
    contract = _required_text(config, "sequence_contract_version", max_length=64)
    if contract != "industrial-rul-sequence-v1":
        raise TrainingConfigurationError("rul_sequence_contract_is_not_supported")
    expected_signals = (
        "vibration_rms_mm_s",
        "bearing_temperature_c",
        "motor_current_a",
        "rpm",
    )
    if config.get("signal_order") != list(expected_signals):
        raise TrainingConfigurationError("rul_signal_order_must_match_dataset_contract")
    if config.get("quantiles") != [0.1, 0.5, 0.9]:
        raise TrainingConfigurationError("rul_quantiles_must_be_p10_p50_p90")
    if config.get("target_transform") != "log1p_minutes":
        raise TrainingConfigurationError("rul_target_transform_is_not_supported")
    effective_batch_size = _positive_integer(config, "effective_batch_size")
    per_device_train_batch_size = _positive_integer(
        config, "per_device_train_batch_size", default=16
    )
    if effective_batch_size % per_device_train_batch_size:
        raise TrainingConfigurationError(
            "effective_batch_size_must_be_divisible_by_per_device_batch"
        )
    d_model = _positive_integer(config, "d_model", default=128)
    nhead = _positive_integer(config, "nhead", default=4)
    if d_model % nhead:
        raise TrainingConfigurationError("d_model_must_be_divisible_by_nhead")
    evaluation_interval = _positive_integer(config, "evaluation_interval")
    objective = _choice(
        config,
        "objective",
        {"quantile_regression"},
        default="quantile_regression",
    )
    return CompiledRulTransformerConfig(
        method="RUL_TRANSFORMER",
        base_model_revision="rul-transformer-v1",
        max_steps=_positive_integer(config, "max_steps"),
        effective_batch_size=effective_batch_size,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=_positive_integer(
            config,
            "per_device_eval_batch_size",
            default=per_device_train_batch_size,
        ),
        gradient_accumulation_steps=effective_batch_size // per_device_train_batch_size,
        evaluation_interval=evaluation_interval,
        save_steps=_positive_integer(config, "save_steps", default=evaluation_interval),
        logging_steps=_positive_integer(config, "logging_steps", default=10),
        learning_rate=_bounded_float(config, "learning_rate", default=1e-4, lower=1e-10),
        weight_decay=_bounded_float(config, "weight_decay", default=0.01, lower=0, upper=1),
        precision=_choice(config, "precision", {"bfloat16", "float16"}, default="bfloat16"),
        seed=_first_seed(experiment.random_seeds),
        data_seed=_integer(config, "data_seed", default=_first_seed(experiment.random_seeds)),
        gpu_hourly_cost_usd=_bounded_float(config, "gpu_hourly_cost_usd", default=0.0, lower=0),
        sequence_contract_version="industrial-rul-sequence-v1",
        signal_order=expected_signals,
        max_sequence_length=_positive_integer(config, "max_sequence_length", default=256),
        d_model=d_model,
        nhead=nhead,
        num_layers=_positive_integer(config, "num_layers", default=3),
        dim_feedforward=_positive_integer(config, "dim_feedforward", default=256),
        dropout=_bounded_float(config, "dropout", default=0.1, lower=0, upper=0.9),
        objective=cast(Literal["quantile_regression"], objective),
        quantiles=(0.1, 0.5, 0.9),
        target_transform="log1p_minutes",
    )


def _compile_specialized_common(
    experiment: TrainingExperimentRecord, *, default_learning_rate: float
) -> dict[str, Any]:
    config = experiment.training_config
    _validate_single_gpu(experiment)
    contract = _required_text(config, "retrieval_contract_version", max_length=64)
    if contract != "industrial-retrieval-triplet-v1":
        raise TrainingConfigurationError("retrieval_training_contract_is_not_supported")
    effective_batch_size = _positive_integer(config, "effective_batch_size")
    per_device_train_batch_size = _positive_integer(
        config, "per_device_train_batch_size", default=8
    )
    if effective_batch_size % per_device_train_batch_size:
        raise TrainingConfigurationError(
            "effective_batch_size_must_be_divisible_by_per_device_batch"
        )
    revision = _immutable_revision(config, "base_model_revision")
    if experiment.base_model_digest != f"hf-revision:{revision}":
        raise TrainingConfigurationError("base_model_digest_does_not_bind_revision")
    evaluation_interval = _positive_integer(config, "evaluation_interval")
    return {
        "base_model_revision": revision,
        "max_steps": _positive_integer(config, "max_steps"),
        "effective_batch_size": effective_batch_size,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": _positive_integer(
            config, "per_device_eval_batch_size", default=per_device_train_batch_size
        ),
        "gradient_accumulation_steps": effective_batch_size // per_device_train_batch_size,
        "evaluation_interval": evaluation_interval,
        "save_steps": _positive_integer(config, "save_steps", default=evaluation_interval),
        "logging_steps": _positive_integer(config, "logging_steps", default=10),
        "learning_rate": _bounded_float(
            config, "learning_rate", default=default_learning_rate, lower=0.0000000001
        ),
        "warmup_ratio": _bounded_float(config, "warmup_ratio", default=0.1, lower=0, upper=1),
        "weight_decay": _bounded_float(config, "weight_decay", default=0.01, lower=0, upper=1),
        "lr_scheduler_type": _choice(
            config,
            "lr_scheduler_type",
            {"linear", "cosine", "constant", "constant_with_warmup"},
            default="linear",
        ),
        "max_sequence_length": _positive_integer(config, "max_sequence_length", default=512),
        "precision": _choice(config, "precision", {"bfloat16", "float16"}, default="bfloat16"),
        "seed": _first_seed(experiment.random_seeds),
        "data_seed": _integer(config, "data_seed", default=_first_seed(experiment.random_seeds)),
        "gpu_hourly_cost_usd": _bounded_float(config, "gpu_hourly_cost_usd", default=0.0, lower=0),
        "retrieval_contract_version": "industrial-retrieval-triplet-v1",
        "hard_negatives_per_query": _positive_integer(
            config, "hard_negatives_per_query", default=1
        ),
    }


def _compile_peft_common(
    experiment: TrainingExperimentRecord,
    *,
    method: str,
    quantized: bool,
    default_learning_rate: float,
) -> dict[str, Any]:
    try:
        distributed = compile_distributed_profile(
            experiment.distributed_profile,
            experiment.hardware_topology,
            method=method,
            quantized=quantized,
        )
    except DistributedProfileError as exc:
        raise TrainingConfigurationError(str(exc)) from exc
    resume_from_checkpoint = _compile_resume_checkpoint(
        experiment.training_config,
        method=method,
    )
    return _compile_peft_common_values(
        experiment,
        quantized=quantized,
        default_learning_rate=default_learning_rate,
        distributed=distributed,
        resume_from_checkpoint=resume_from_checkpoint,
    )


def _validate_single_gpu(experiment: TrainingExperimentRecord) -> None:
    distributed = experiment.distributed_profile
    hardware = experiment.hardware_topology
    if (
        distributed.get("strategy") != "single_gpu"
        or _integer(distributed, "world_size", default=1) != 1
    ):
        raise TrainingConfigurationError("only_single_gpu_profile_is_supported")
    if hardware.get("accelerator") != "NVIDIA_GPU" or _integer(hardware, "count", default=1) != 1:
        raise TrainingConfigurationError("one_nvidia_gpu_is_required")


def _compile_peft_common_values(
    experiment: TrainingExperimentRecord,
    *,
    quantized: bool,
    default_learning_rate: float,
    distributed: CompiledDistributedProfile,
    resume_from_checkpoint: ResumeCheckpointConfig | None,
) -> dict[str, Any]:
    config = experiment.training_config
    effective_batch_size = _positive_integer(config, "effective_batch_size")
    per_device_train_batch_size = _positive_integer(
        config, "per_device_train_batch_size", default=1
    )
    distributed_micro_batch = per_device_train_batch_size * distributed.world_size
    if effective_batch_size % distributed_micro_batch:
        raise TrainingConfigurationError(
            "effective_batch_size_must_be_divisible_by_per_device_batch_and_world_size"
        )
    precision = _choice(config, "precision", {"bfloat16", "float16"}, default="bfloat16")
    target_modules_value = config.get(
        "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
    )
    if (
        not isinstance(target_modules_value, list)
        or not target_modules_value
        or not all(isinstance(item, str) and item.strip() for item in target_modules_value)
    ):
        raise TrainingConfigurationError("target_modules_must_be_a_non_empty_string_list")
    target_modules = tuple(dict.fromkeys(item.strip() for item in target_modules_value))
    base_model_revision = _immutable_revision(config, "base_model_revision")
    if experiment.base_model_digest != f"hf-revision:{base_model_revision}":
        raise TrainingConfigurationError("base_model_digest_does_not_bind_revision")

    quantization = None
    if quantized:
        quantization = QuantizationConfig(
            load_in_4bit=True,
            quant_type="nf4",
            use_double_quant=True,
            compute_dtype=precision,
        )
    evaluation_interval = _positive_integer(config, "evaluation_interval")
    return {
        "base_model_revision": base_model_revision,
        "max_steps": _positive_integer(config, "max_steps"),
        "effective_batch_size": effective_batch_size,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": _positive_integer(
            config, "per_device_eval_batch_size", default=per_device_train_batch_size
        ),
        "gradient_accumulation_steps": effective_batch_size // distributed_micro_batch,
        "evaluation_interval": evaluation_interval,
        "save_steps": _positive_integer(config, "save_steps", default=evaluation_interval),
        "logging_steps": _positive_integer(config, "logging_steps", default=10),
        "learning_rate": _bounded_float(
            config, "learning_rate", default=default_learning_rate, lower=0.0000000001
        ),
        "warmup_ratio": _bounded_float(config, "warmup_ratio", default=0.03, lower=0, upper=1),
        "weight_decay": _bounded_float(config, "weight_decay", default=0.0, lower=0, upper=1),
        "lr_scheduler_type": _choice(
            config,
            "lr_scheduler_type",
            {"linear", "cosine", "constant", "constant_with_warmup"},
            default="cosine",
        ),
        "max_sequence_length": _positive_integer(config, "max_sequence_length", default=2048),
        "gradient_checkpointing": _boolean(config, "gradient_checkpointing", default=True),
        "precision": precision,
        "seed": _first_seed(experiment.random_seeds),
        "data_seed": _integer(config, "data_seed", default=_first_seed(experiment.random_seeds)),
        "lora_rank": _positive_integer(config, "lora_rank", default=16),
        "lora_alpha": _positive_integer(config, "lora_alpha", default=32),
        "lora_dropout": _bounded_float(config, "lora_dropout", default=0.05, lower=0, upper=1),
        "lora_bias": _choice(config, "lora_bias", {"none", "all", "lora_only"}, default="none"),
        "target_modules": target_modules,
        "gpu_hourly_cost_usd": _bounded_float(config, "gpu_hourly_cost_usd", default=0.0, lower=0),
        "quantization": quantization,
        "distributed": distributed,
        "resume_from_checkpoint": resume_from_checkpoint,
    }


def _compile_resume_checkpoint(
    config: dict[str, Any],
    *,
    method: str,
) -> ResumeCheckpointConfig | None:
    raw = config.get("resume_from_checkpoint")
    if raw is None:
        return None
    if method not in {"LORA", "QLORA", "VLM"}:
        raise TrainingConfigurationError("checkpoint_resume_currently_requires_lora_qlora_or_vlm")
    if not isinstance(raw, dict):
        raise TrainingConfigurationError("resume_from_checkpoint_must_be_an_object")
    content_hash = _required_text(raw, "content_hash", max_length=64)
    if len(content_hash) != 64 or any(
        character not in "0123456789abcdef" for character in content_hash
    ):
        raise TrainingConfigurationError("resume_checkpoint_content_hash_must_be_sha256")
    checkpoint_path = _required_text(raw, "checkpoint_path", max_length=64)
    if (
        not checkpoint_path.startswith("checkpoint-")
        or not checkpoint_path.removeprefix("checkpoint-").isdigit()
        or int(checkpoint_path.removeprefix("checkpoint-")) <= 0
    ):
        raise TrainingConfigurationError("resume_checkpoint_path_is_invalid")
    return ResumeCheckpointConfig(
        source_experiment_id=_required_text(raw, "source_experiment_id", max_length=128),
        artifact_id=_required_text(raw, "artifact_id", max_length=128),
        content_hash=content_hash,
        checkpoint_path=checkpoint_path,
    )


def _model_revision_digest(values: dict[str, Any], key: str, revision: str) -> str:
    digest = _required_text(values, key, max_length=255)
    if digest != f"hf-revision:{revision}":
        raise TrainingConfigurationError(f"{key}_does_not_bind_revision")
    return digest


def _immutable_revision(values: dict[str, Any], key: str) -> str:
    revision = _required_text(values, key, max_length=128)
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise TrainingConfigurationError(f"{key}_must_be_a_commit_sha")
    return revision


def _first_seed(values: list[int]) -> int:
    if not values or isinstance(values[0], bool) or not isinstance(values[0], int):
        raise TrainingConfigurationError("one_integer_seed_is_required")
    return values[0]


def _required_text(values: dict[str, Any], key: str, *, max_length: int) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise TrainingConfigurationError(f"{key}_is_required")
    return value.strip()


def _integer(values: dict[str, Any], key: str, *, default: int | None = None) -> int:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingConfigurationError(f"{key}_must_be_an_integer")
    return value


def _positive_integer(values: dict[str, Any], key: str, *, default: int | None = None) -> int:
    value = _integer(values, key, default=default)
    if value <= 0:
        raise TrainingConfigurationError(f"{key}_must_be_positive")
    return value


def _bounded_float(
    values: dict[str, Any],
    key: str,
    *,
    default: float,
    lower: float,
    upper: float | None = None,
) -> float:
    raw = values.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TrainingConfigurationError(f"{key}_must_be_numeric")
    value = float(raw)
    if not math.isfinite(value) or value < lower or (upper is not None and value > upper):
        raise TrainingConfigurationError(f"{key}_is_out_of_range")
    return value


def _choice(values: dict[str, Any], key: str, choices: set[str], *, default: str) -> Any:
    value = values.get(key, default)
    if not isinstance(value, str) or value not in choices:
        raise TrainingConfigurationError(f"{key}_is_not_supported")
    return value


def _boolean(values: dict[str, Any], key: str, *, default: bool) -> bool:
    value = values.get(key, default)
    if not isinstance(value, bool):
        raise TrainingConfigurationError(f"{key}_must_be_boolean")
    return value
