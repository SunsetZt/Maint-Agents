"""Lazy Hugging Face/TRL backend used only inside the GPU training image."""

from __future__ import annotations

import importlib
import json
import math
import time
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, TypeAlias

from industrial_ops_agent.persistence.models import TrainingExperimentRecord
from industrial_ops_agent.predictive_maintenance.rul_training_dataset import (
    RulTrainingDatasetBundle,
)
from industrial_ops_agent.predictive_maintenance.training_dataset import (
    TelemetryTrainingDatasetBundle,
)
from industrial_ops_agent.training.config import (
    CompiledAsrConfig,
    CompiledDpoConfig,
    CompiledEmbeddingConfig,
    CompiledGrpoConfig,
    CompiledPeftConfig,
    CompiledPpoConfig,
    CompiledRerankerConfig,
    CompiledRulTransformerConfig,
    CompiledSftConfig,
    CompiledSpecializedConfig,
    CompiledTimeseriesTransformerConfig,
    CompiledTrainingConfig,
    CompiledTtsConfig,
    CompiledVlmConfig,
    VlmCurriculumConfig,
)
from industrial_ops_agent.training.dataset import TrainingDatasetBundle
from industrial_ops_agent.training.distributed import (
    launched_world_size,
    local_rank,
    process_rank,
    trainer_argument_overrides,
)
from industrial_ops_agent.training.model_contract import (
    verify_encoder_model_contract,
    verify_model_contract,
)
from industrial_ops_agent.training.reward_contracts import (
    LEGACY_REWARD_CONTRACT_VERSION,
    RewardContractError,
    RewardTargetMigration,
    migrate_reward_target,
    score_reward,
    summarize_reward_migrations,
)


class TrainingBackendError(RuntimeError):
    pass


TrainingDataset: TypeAlias = (
    TrainingDatasetBundle | TelemetryTrainingDatasetBundle | RulTrainingDatasetBundle
)


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    metrics: dict[str, float]
    cost_summary: dict[str, Any]
    output_directory: Path
    runtime_metadata: dict[str, Any]


class TrainingBackend(Protocol):
    def train(
        self,
        *,
        experiment: TrainingExperimentRecord,
        config: CompiledTrainingConfig,
        dataset: TrainingDataset,
        output_directory: Path,
        resume_from_checkpoint: Path | None,
    ) -> TrainingOutcome: ...


def _prepare_sft_runtime(
    torch: Any,
    config: CompiledSftConfig,
    output_directory: Path,
) -> None:
    if not torch.cuda.is_available():
        raise TrainingBackendError("cuda_gpu_is_not_available")
    expected = config.distributed.world_size
    visible = int(torch.cuda.device_count())
    if visible != expected:
        raise TrainingBackendError("visible_gpu_count_does_not_match_distributed_profile")
    if config.distributed.distributed:
        if launched_world_size() != expected:
            raise TrainingBackendError("torchrun_world_size_does_not_match_distributed_profile")
        rank = local_rank()
        if rank < 0 or rank >= visible:
            raise TrainingBackendError("local_rank_is_out_of_visible_gpu_range")
        torch.cuda.set_device(rank)
        output_directory.mkdir(parents=True, exist_ok=True)
    else:
        output_directory.mkdir(parents=True, exist_ok=False)


class HuggingFaceSftBackend:
    """Execute completion-only conversational SFT with PEFT adapters."""

    def train(
        self,
        *,
        experiment: TrainingExperimentRecord,
        config: CompiledSftConfig,
        dataset: TrainingDatasetBundle,
        output_directory: Path,
        resume_from_checkpoint: Path | None = None,
    ) -> TrainingOutcome:
        try:
            torch = importlib.import_module("torch")
            Dataset = importlib.import_module("datasets").Dataset
            peft = importlib.import_module("peft")
            LoraConfig = peft.LoraConfig
            prepare_model_for_kbit_training = peft.prepare_model_for_kbit_training
            transformers = importlib.import_module("transformers")
            AutoModelForCausalLM = transformers.AutoModelForCausalLM
            AutoTokenizer = transformers.AutoTokenizer
            BitsAndBytesConfig = transformers.BitsAndBytesConfig
            set_seed = transformers.set_seed
            trl = importlib.import_module("trl")
            SFTConfig = trl.SFTConfig
            SFTTrainer = trl.SFTTrainer
        except ImportError as exc:  # pragma: no cover - exercised in the GPU image
            raise TrainingBackendError("training_dependencies_are_not_installed") from exc

        _prepare_sft_runtime(torch, config, output_directory)
        set_seed(config.seed)
        torch.cuda.reset_peak_memory_stats()

        dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float16
        model_kwargs: dict[str, Any] = {
            "revision": config.base_model_revision,
            "torch_dtype": dtype,
            "trust_remote_code": False,
        }
        if not config.distributed.distributed:
            model_kwargs["device_map"] = {"": 0}
        elif config.quantization is not None:
            # A 4-bit model cannot be moved after loading; each torchrun rank
            # must materialize its replicated QLoRA base on its own device.
            model_kwargs["device_map"] = {"": local_rank()}
        if config.quantization is not None:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=config.quantization.quant_type,
                bnb_4bit_use_double_quant=config.quantization.use_double_quant,
                bnb_4bit_compute_dtype=dtype,
            )
        tokenizer = AutoTokenizer.from_pretrained(
            experiment.base_model_id,
            revision=config.base_model_revision,
            trust_remote_code=False,
        )
        verify_model_contract(
            tokenizer=tokenizer,
            revision=config.base_model_revision,
            base_model_digest=experiment.base_model_digest,
            tokenizer_digest=experiment.tokenizer_digest,
            chat_template_digest=experiment.chat_template_digest,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(experiment.base_model_id, **model_kwargs)
        resolved_revision = getattr(model.config, "_commit_hash", None)
        if resolved_revision and resolved_revision != config.base_model_revision:
            raise TrainingBackendError("resolved_base_model_revision_changed")
        if config.quantization is not None:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=config.gradient_checkpointing,
            )

        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            target_modules=list(config.target_modules),
        )
        args = SFTConfig(
            output_dir=str(output_directory),
            max_steps=config.max_steps,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            eval_strategy="steps",
            eval_steps=config.evaluation_interval,
            save_strategy="steps",
            save_steps=config.save_steps,
            logging_steps=config.logging_steps,
            max_length=config.max_sequence_length,
            gradient_checkpointing=config.gradient_checkpointing,
            bf16=config.precision == "bfloat16",
            fp16=config.precision == "float16",
            seed=config.seed,
            data_seed=config.data_seed,
            completion_only_loss=True,
            report_to="none",
            remove_unused_columns=False,
            save_total_limit=2,
            **trainer_argument_overrides(
                config.distributed,
                precision=config.precision,
            ),
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=Dataset.from_list(dataset.train_records),
            eval_dataset=Dataset.from_list(dataset.validation_records),
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        started = time.monotonic()
        train_result = trainer.train(
            resume_from_checkpoint=(
                str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
            )
        )
        evaluation_metrics = trainer.evaluate()
        runtime_seconds = time.monotonic() - started
        trainer.save_model(str(output_directory))
        if process_rank() == 0:
            tokenizer.save_pretrained(str(output_directory))

        train_metrics = dict(getattr(train_result, "metrics", {}))
        metrics = _finite_metrics(
            {
                "train_loss": train_metrics.get("train_loss"),
                "eval_loss": evaluation_metrics.get("eval_loss"),
                "train_runtime_seconds": runtime_seconds,
                "train_samples_per_second": train_metrics.get("train_samples_per_second", 0.0),
                "peak_gpu_memory_allocated_bytes": float(torch.cuda.max_memory_allocated()),
                "peak_gpu_memory_reserved_bytes": float(torch.cuda.max_memory_reserved()),
                "gpu_hours": runtime_seconds * config.distributed.world_size / 3600.0,
            }
        )
        estimated_cost = metrics["gpu_hours"] * config.gpu_hourly_cost_usd
        device = torch.cuda.get_device_properties(local_rank())
        cost_summary = {
            "currency": "USD",
            "gpu_hourly_cost_usd": config.gpu_hourly_cost_usd,
            "gpu_hours": metrics["gpu_hours"],
            "estimated_compute_cost_usd": estimated_cost,
            "pricing_source": "registered_training_config",
            "energy_kwh": None,
            "energy_source": "not_available_without_dcgm_or_nvml_energy_counter",
        }
        runtime_metadata = {
            "gpu_name": device.name,
            "gpu_total_memory_bytes": int(device.total_memory),
            "gpu_count": config.distributed.world_size,
            "distributed_strategy": config.distributed.strategy,
            "distributed_profile": config.distributed.as_dict(),
            "resumed_from_checkpoint": config.resume_from_checkpoint is not None,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "resolved_base_model_revision": resolved_revision,
            "train_rows": len(dataset.train),
            "validation_rows": len(dataset.validation),
        }
        if process_rank() == 0:
            (output_directory / "industrial-ops-training-report.json").write_text(
                json.dumps(
                    {
                        "metrics": metrics,
                        "cost_summary": cost_summary,
                        "runtime": runtime_metadata,
                        "compiled_config_digest": config.digest,
                        "dataset_manifest_hash": dataset.manifest_hash,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return TrainingOutcome(
            metrics=metrics,
            cost_summary=cost_summary,
            output_directory=output_directory,
            runtime_metadata=runtime_metadata,
        )


class HuggingFaceTrainingBackend:
    """Dispatch one frozen experiment to its method-specific TRL trainer."""

    def train(
        self,
        *,
        experiment: TrainingExperimentRecord,
        config: CompiledTrainingConfig,
        dataset: TrainingDataset,
        output_directory: Path,
        resume_from_checkpoint: Path | None = None,
    ) -> TrainingOutcome:
        if isinstance(config, CompiledTimeseriesTransformerConfig):
            if resume_from_checkpoint is not None:
                raise TrainingBackendError("checkpoint_resume_currently_requires_lora_or_qlora")
            if not isinstance(dataset, TelemetryTrainingDatasetBundle):
                raise TrainingBackendError("timeseries_backend_requires_telemetry_dataset")
            from industrial_ops_agent.predictive_maintenance.transformer_backend import (
                train_timeseries_transformer,
            )

            return train_timeseries_transformer(experiment, config, dataset, output_directory)
        if isinstance(config, CompiledRulTransformerConfig):
            if resume_from_checkpoint is not None:
                raise TrainingBackendError("checkpoint_resume_is_not_supported_for_rul_transformer")
            if not isinstance(dataset, RulTrainingDatasetBundle):
                raise TrainingBackendError("rul_backend_requires_rul_dataset")
            from industrial_ops_agent.predictive_maintenance.rul_transformer_backend import (
                train_rul_transformer,
            )

            return train_rul_transformer(experiment, config, dataset, output_directory)
        if not isinstance(dataset, TrainingDatasetBundle):
            raise TrainingBackendError("language_backend_requires_governed_sft_dataset")
        if isinstance(config, CompiledSftConfig):
            return HuggingFaceSftBackend().train(
                experiment=experiment,
                config=config,
                dataset=dataset,
                output_directory=output_directory,
                resume_from_checkpoint=resume_from_checkpoint,
            )
        if isinstance(config, CompiledVlmConfig):
            return _train_vlm(
                experiment,
                config,
                dataset,
                output_directory,
                resume_from_checkpoint=resume_from_checkpoint,
            )
        if resume_from_checkpoint is not None:
            raise TrainingBackendError("checkpoint_resume_currently_requires_lora_qlora_or_vlm")
        if isinstance(config, CompiledDpoConfig):
            return _train_dpo(experiment, config, dataset, output_directory)
        if isinstance(config, CompiledGrpoConfig):
            return _train_grpo(experiment, config, dataset, output_directory)
        if isinstance(config, CompiledPpoConfig):
            return _train_ppo(experiment, config, dataset, output_directory)
        if isinstance(config, CompiledEmbeddingConfig):
            return _train_embedding(experiment, config, dataset, output_directory)
        if isinstance(config, CompiledRerankerConfig):
            return _train_reranker(experiment, config, dataset, output_directory)
        if isinstance(config, CompiledAsrConfig):
            return _train_asr(experiment, config, dataset, output_directory)
        if isinstance(config, CompiledTtsConfig):
            return _train_tts(experiment, config, dataset, output_directory)
        raise TrainingBackendError("training_method_has_no_backend")


def _train_dpo(
    experiment: TrainingExperimentRecord,
    config: CompiledDpoConfig,
    dataset: TrainingDatasetBundle,
    output_directory: Path,
) -> TrainingOutcome:
    dependencies = _post_training_dependencies("DPO")
    torch = dependencies["torch"]
    Dataset = dependencies["Dataset"]
    DPOConfig = dependencies["DPOConfig"]
    DPOTrainer = dependencies["DPOTrainer"]
    tokenizer, model, peft_config, resolved_revision = _load_policy(
        experiment, config, dependencies, output_directory
    )
    args = DPOConfig(
        output_dir=str(output_directory),
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        eval_strategy="steps",
        eval_steps=config.evaluation_interval,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        max_length=config.max_sequence_length,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.precision == "bfloat16",
        fp16=config.precision == "float16",
        seed=config.seed,
        data_seed=config.data_seed,
        beta=config.beta,
        loss_type=config.loss_type,
        report_to="none",
        remove_unused_columns=False,
        save_total_limit=2,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=Dataset.from_list(dataset.train_records),
        eval_dataset=Dataset.from_list(dataset.validation_records),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    return _run_post_trainer(
        trainer=trainer,
        tokenizer=tokenizer,
        torch=torch,
        experiment=experiment,
        config=config,
        dataset=dataset,
        output_directory=output_directory,
        resolved_revision=resolved_revision,
        evaluate=True,
    )


def _train_grpo(
    experiment: TrainingExperimentRecord,
    config: CompiledGrpoConfig,
    dataset: TrainingDatasetBundle,
    output_directory: Path,
) -> TrainingOutcome:
    train_records, train_migrations = _prepare_grpo_records(dataset.train_records, config)
    validation_records, validation_migrations = _prepare_grpo_records(
        dataset.validation_records, config
    )
    migration_summary = summarize_reward_migrations([*train_migrations, *validation_migrations])
    dependencies = _post_training_dependencies("GRPO")
    torch = dependencies["torch"]
    Dataset = dependencies["Dataset"]
    GRPOConfig = dependencies["GRPOConfig"]
    GRPOTrainer = dependencies["GRPOTrainer"]
    tokenizer, model, peft_config, resolved_revision = _load_policy(
        experiment, config, dependencies, output_directory
    )
    args = GRPOConfig(
        output_dir=str(output_directory),
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        eval_strategy="steps",
        eval_steps=config.evaluation_interval,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        max_completion_length=config.max_completion_length,
        num_generations=config.group_size,
        beta=config.beta,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.precision == "bfloat16",
        fp16=config.precision == "float16",
        seed=config.seed,
        data_seed=config.data_seed,
        report_to="none",
        remove_unused_columns=False,
        save_total_limit=2,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=_reward_function(config.reward_contract_version),
        args=args,
        train_dataset=Dataset.from_list(train_records),
        eval_dataset=Dataset.from_list(validation_records),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    return _run_post_trainer(
        trainer=trainer,
        tokenizer=tokenizer,
        torch=torch,
        experiment=experiment,
        config=config,
        dataset=dataset,
        output_directory=output_directory,
        resolved_revision=resolved_revision,
        evaluate=True,
        runtime_metadata_extra={"reward_contract": migration_summary},
    )


def _train_ppo(
    experiment: TrainingExperimentRecord,
    config: CompiledPpoConfig,
    dataset: TrainingDatasetBundle,
    output_directory: Path,
) -> TrainingOutcome:
    dependencies = _post_training_dependencies("PPO")
    torch = dependencies["torch"]
    Dataset = dependencies["Dataset"]
    AutoModelForSequenceClassification = dependencies["AutoModelForSequenceClassification"]
    PPOConfig = dependencies["PPOConfig"]
    PPOTrainer = dependencies["PPOTrainer"]
    tokenizer, policy, peft_config, resolved_revision = _load_policy(
        experiment, config, dependencies, output_directory
    )
    dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float16
    score_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": False,
        "num_labels": 1,
    }
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_id,
        revision=config.reward_model_revision,
        **score_kwargs,
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        config.value_model_id,
        revision=config.value_model_revision,
        **score_kwargs,
    )
    _verify_resolved_revision(
        reward_model, config.reward_model_revision, "resolved_reward_model_revision_changed"
    )
    _verify_resolved_revision(
        value_model, config.value_model_revision, "resolved_value_model_revision_changed"
    )
    train_records = _tokenize_ppo_records(tokenizer, dataset.train_records)
    validation_records = _tokenize_ppo_records(tokenizer, dataset.validation_records)
    args = PPOConfig(
        output_dir=str(output_directory),
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.precision == "bfloat16",
        fp16=config.precision == "float16",
        seed=config.seed,
        data_seed=config.data_seed,
        report_to="none",
        total_episodes=config.total_episodes,
        response_length=config.response_length,
        num_ppo_epochs=config.num_ppo_epochs,
        num_mini_batches=config.num_mini_batches,
        kl_coef=config.kl_coefficient,
    )
    trainer = PPOTrainer(
        args=args,
        processing_class=tokenizer,
        model=policy,
        ref_model=None,
        reward_model=reward_model,
        train_dataset=Dataset.from_list(train_records),
        value_model=value_model,
        eval_dataset=Dataset.from_list(validation_records),
        peft_config=peft_config,
    )
    return _run_post_trainer(
        trainer=trainer,
        tokenizer=tokenizer,
        torch=torch,
        experiment=experiment,
        config=config,
        dataset=dataset,
        output_directory=output_directory,
        resolved_revision=resolved_revision,
        evaluate=False,
    )


def _train_embedding(
    experiment: TrainingExperimentRecord,
    config: CompiledEmbeddingConfig,
    dataset: TrainingDatasetBundle,
    output_directory: Path,
) -> TrainingOutcome:
    dependencies = _sentence_transformer_dependencies()
    torch = dependencies["torch"]
    _prepare_specialized_runtime(torch, config, output_directory, dependencies["set_seed"])
    dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float16
    model = dependencies["SentenceTransformer"](
        experiment.base_model_id,
        revision=config.base_model_revision,
        trust_remote_code=False,
        model_kwargs={"torch_dtype": dtype},
    )
    model.max_seq_length = config.max_sequence_length
    tokenizer = model.tokenizer
    verify_encoder_model_contract(
        tokenizer=tokenizer,
        revision=config.base_model_revision,
        base_model_digest=experiment.base_model_digest,
        tokenizer_digest=experiment.tokenizer_digest,
        chat_template_digest=experiment.chat_template_digest,
    )
    resolved_revision = _sentence_transformer_revision(model)
    if resolved_revision and resolved_revision != config.base_model_revision:
        raise TrainingBackendError("resolved_base_model_revision_changed")
    args = dependencies["SentenceTransformerTrainingArguments"](
        output_dir=str(output_directory),
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        eval_strategy="steps",
        eval_steps=config.evaluation_interval,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        bf16=config.precision == "bfloat16",
        fp16=config.precision == "float16",
        seed=config.seed,
        data_seed=config.data_seed,
        report_to="none",
        save_total_limit=2,
        batch_sampler=dependencies["BatchSamplers"].NO_DUPLICATES,
    )
    loss = dependencies["EmbeddingLoss"](
        model=model,
        scale=config.similarity_scale,
    )
    trainer = dependencies["SentenceTransformerTrainer"](
        model=model,
        args=args,
        train_dataset=dependencies["Dataset"].from_list(dataset.train_records),
        eval_dataset=dependencies["Dataset"].from_list(dataset.validation_records),
        loss=loss,
    )
    return _run_specialized_trainer(
        trainer=trainer,
        tokenizer=tokenizer,
        torch=torch,
        experiment=experiment,
        config=config,
        dataset=dataset,
        output_directory=output_directory,
        resolved_revision=resolved_revision,
        trainer_family="sentence-transformers-5",
    )


def _train_reranker(
    experiment: TrainingExperimentRecord,
    config: CompiledRerankerConfig,
    dataset: TrainingDatasetBundle,
    output_directory: Path,
) -> TrainingOutcome:
    dependencies = _sentence_transformer_dependencies()
    torch = dependencies["torch"]
    _prepare_specialized_runtime(torch, config, output_directory, dependencies["set_seed"])
    dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float16
    model = dependencies["CrossEncoder"](
        experiment.base_model_id,
        revision=config.base_model_revision,
        trust_remote_code=False,
        num_labels=1,
        max_length=config.max_sequence_length,
        model_kwargs={"torch_dtype": dtype},
    )
    tokenizer = model.tokenizer
    verify_encoder_model_contract(
        tokenizer=tokenizer,
        revision=config.base_model_revision,
        base_model_digest=experiment.base_model_digest,
        tokenizer_digest=experiment.tokenizer_digest,
        chat_template_digest=experiment.chat_template_digest,
    )
    resolved_revision = getattr(model.model.config, "_commit_hash", None)
    if resolved_revision and resolved_revision != config.base_model_revision:
        raise TrainingBackendError("resolved_base_model_revision_changed")
    args = dependencies["CrossEncoderTrainingArguments"](
        output_dir=str(output_directory),
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        eval_strategy="steps",
        eval_steps=config.evaluation_interval,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        bf16=config.precision == "bfloat16",
        fp16=config.precision == "float16",
        seed=config.seed,
        data_seed=config.data_seed,
        report_to="none",
        save_total_limit=2,
    )
    loss = dependencies["RerankerLoss"](
        model=model,
        pos_weight=torch.tensor(config.positive_weight),
    )
    trainer = dependencies["CrossEncoderTrainer"](
        model=model,
        args=args,
        train_dataset=dependencies["Dataset"].from_list(dataset.train_records),
        eval_dataset=dependencies["Dataset"].from_list(dataset.validation_records),
        loss=loss,
    )
    return _run_specialized_trainer(
        trainer=trainer,
        tokenizer=tokenizer,
        torch=torch,
        experiment=experiment,
        config=config,
        dataset=dataset,
        output_directory=output_directory,
        resolved_revision=resolved_revision,
        trainer_family="sentence-transformers-cross-encoder-5",
    )


def _train_vlm(
    experiment: TrainingExperimentRecord,
    config: CompiledVlmConfig,
    dataset: TrainingDatasetBundle,
    output_directory: Path,
    *,
    resume_from_checkpoint: Path | None = None,
) -> TrainingOutcome:
    dependencies = _multimodal_dependencies()
    torch = dependencies["torch"]
    _prepare_peft_runtime(torch, config, output_directory, dependencies["set_seed"])
    dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float16
    model_kwargs: dict[str, Any] = {
        "revision": config.base_model_revision,
        "dtype": dtype,
        "trust_remote_code": False,
        "device_map": {"": 0},
    }
    if config.quantization is not None:
        model_kwargs["quantization_config"] = dependencies["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quantization.quant_type,
            bnb_4bit_use_double_quant=config.quantization.use_double_quant,
            bnb_4bit_compute_dtype=dtype,
        )
    processor = dependencies["AutoProcessor"].from_pretrained(
        experiment.base_model_id,
        revision=config.base_model_revision,
        trust_remote_code=False,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise TrainingBackendError("vlm_processor_has_no_tokenizer")
    verify_model_contract(
        tokenizer=tokenizer,
        revision=config.base_model_revision,
        base_model_digest=experiment.base_model_digest,
        tokenizer_digest=experiment.tokenizer_digest,
        chat_template_digest=experiment.chat_template_digest,
    )
    model = dependencies["AutoModelForImageTextToText"].from_pretrained(
        experiment.base_model_id,
        **model_kwargs,
    )
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision and resolved_revision != config.base_model_revision:
        raise TrainingBackendError("resolved_base_model_revision_changed")
    if config.quantization is not None:
        model = dependencies["prepare_model_for_kbit_training"](
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
    train_records = _vlm_records(
        dataset.train_records,
        config,
        dependencies["Image"],
        curriculum=config.curriculum,
    )
    validation_records = _vlm_records(
        dataset.validation_records,
        config,
        dependencies["Image"],
        curriculum=None,
    )
    peft_config = dependencies["LoraConfig"](
        task_type="CAUSAL_LM",
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        target_modules=list(config.target_modules),
    )
    args = dependencies["SFTConfig"](
        output_dir=str(output_directory),
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        eval_strategy="steps",
        eval_steps=config.evaluation_interval,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        max_length=None,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.precision == "bfloat16",
        fp16=config.precision == "float16",
        seed=config.seed,
        data_seed=config.data_seed,
        report_to="none",
        remove_unused_columns=False,
        save_total_limit=2,
        ignore_data_skip=config.resume_ignore_data_skip,
    )
    trainer = dependencies["SFTTrainer"](
        model=model,
        args=args,
        train_dataset=dependencies["Dataset"].from_list(train_records),
        eval_dataset=dependencies["Dataset"].from_list(validation_records),
        processing_class=processor,
        peft_config=peft_config,
        callbacks=(
            [
                _resume_learning_rate_callback(
                    dependencies["TrainerCallback"],
                    config.learning_rate,
                )
            ]
            if config.resume_optimizer_learning_rate_override
            else None
        ),
    )
    return _run_post_trainer(
        trainer=trainer,
        tokenizer=processor,
        torch=torch,
        experiment=experiment,
        config=config,
        dataset=dataset,
        output_directory=output_directory,
        resolved_revision=resolved_revision,
        evaluate=True,
        resume_from_checkpoint=resume_from_checkpoint,
        trainer_family="trl-vlm-0.28",
        runtime_metadata_extra={
            "real_train_rows": dataset.real_train_count,
            "synthetic_train_rows": dataset.synthetic_train_count,
            "sample_origin_counts": dataset.sample_origin_counts,
            "trainer_train_rows": len(train_records),
            "vlm_curriculum": (
                config.curriculum.as_dict() if config.curriculum is not None else None
            ),
            "resume_ignore_data_skip": config.resume_ignore_data_skip,
            "resume_optimizer_learning_rate_override": (
                config.resume_optimizer_learning_rate_override
            ),
        },
    )


def _train_asr(
    experiment: TrainingExperimentRecord,
    config: CompiledAsrConfig,
    dataset: TrainingDatasetBundle,
    output_directory: Path,
) -> TrainingOutcome:
    dependencies = _multimodal_dependencies()
    torch = dependencies["torch"]
    _prepare_peft_runtime(torch, config, output_directory, dependencies["set_seed"])
    dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float16
    processor = dependencies["AutoProcessor"].from_pretrained(
        experiment.base_model_id,
        revision=config.base_model_revision,
        language=config.language,
        task=config.task,
        trust_remote_code=False,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise TrainingBackendError("asr_processor_has_no_tokenizer")
    verify_encoder_model_contract(
        tokenizer=tokenizer,
        revision=config.base_model_revision,
        base_model_digest=experiment.base_model_digest,
        tokenizer_digest=experiment.tokenizer_digest,
        chat_template_digest=experiment.chat_template_digest,
    )
    model = dependencies["AutoModelForSpeechSeq2Seq"].from_pretrained(
        experiment.base_model_id,
        revision=config.base_model_revision,
        dtype=dtype,
        trust_remote_code=False,
    )
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision and resolved_revision != config.base_model_revision:
        raise TrainingBackendError("resolved_base_model_revision_changed")
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = dependencies["get_peft_model"](
        model,
        dependencies["LoraConfig"](
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            target_modules=list(config.target_modules),
        ),
    )
    train_records = _asr_records(dataset.train_records, processor, config, dependencies)
    validation_records = _asr_records(dataset.validation_records, processor, config, dependencies)
    args = dependencies["Seq2SeqTrainingArguments"](
        output_dir=str(output_directory),
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        eval_strategy="steps",
        eval_steps=config.evaluation_interval,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.precision == "bfloat16",
        fp16=config.precision == "float16",
        seed=config.seed,
        data_seed=config.data_seed,
        report_to="none",
        remove_unused_columns=False,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=config.max_sequence_length,
    )
    trainer = dependencies["Seq2SeqTrainer"](
        model=model,
        args=args,
        train_dataset=dependencies["Dataset"].from_list(train_records),
        eval_dataset=dependencies["Dataset"].from_list(validation_records),
        data_collator=_AsrDataCollator(processor, torch),
        processing_class=processor,
    )
    return _run_post_trainer(
        trainer=trainer,
        tokenizer=processor,
        torch=torch,
        experiment=experiment,
        config=config,
        dataset=dataset,
        output_directory=output_directory,
        resolved_revision=resolved_revision,
        evaluate=True,
        trainer_family="transformers-asr-seq2seq-5",
    )


def _train_tts(
    experiment: TrainingExperimentRecord,
    config: CompiledTtsConfig,
    dataset: TrainingDatasetBundle,
    output_directory: Path,
) -> TrainingOutcome:
    voice_embedding = _tts_voice_embedding(
        [*dataset.train_records, *dataset.validation_records], config
    )
    dependencies = _tts_dependencies()
    torch = dependencies["torch"]
    _prepare_tts_runtime(torch, config, output_directory, dependencies["set_seed"])
    dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float16
    processor = dependencies["SpeechT5Processor"].from_pretrained(
        experiment.base_model_id,
        revision=config.base_model_revision,
        trust_remote_code=False,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise TrainingBackendError("tts_processor_has_no_tokenizer")
    verify_encoder_model_contract(
        tokenizer=tokenizer,
        revision=config.base_model_revision,
        base_model_digest=experiment.base_model_digest,
        tokenizer_digest=experiment.tokenizer_digest,
        chat_template_digest=experiment.chat_template_digest,
    )
    model = dependencies["SpeechT5ForTextToSpeech"].from_pretrained(
        experiment.base_model_id,
        revision=config.base_model_revision,
        dtype=dtype,
        trust_remote_code=False,
    )
    _verify_resolved_revision(
        model,
        config.base_model_revision,
        "resolved_tts_base_model_revision_changed",
    )
    train_records = _tts_records(
        dataset.train_records,
        candidate_ids=[sample.candidate_id for sample in dataset.train],
        processor=processor,
        config=config,
        dependencies=dependencies,
    )
    validation_records = _tts_records(
        dataset.validation_records,
        candidate_ids=[sample.candidate_id for sample in dataset.validation],
        processor=processor,
        config=config,
        dependencies=dependencies,
    )
    args = dependencies["Seq2SeqTrainingArguments"](
        output_dir=str(output_directory),
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        eval_strategy="steps",
        eval_steps=config.evaluation_interval,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        bf16=config.precision == "bfloat16",
        fp16=config.precision == "float16",
        seed=config.seed,
        data_seed=config.data_seed,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        save_total_limit=2,
        predict_with_generate=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    trainer = dependencies["Seq2SeqTrainer"](
        model=model,
        args=args,
        train_dataset=dependencies["Dataset"].from_list(train_records),
        eval_dataset=dependencies["Dataset"].from_list(validation_records),
        data_collator=_TtsDataCollator(processor, torch, model),
        processing_class=processor,
    )
    outcome = _run_post_trainer(
        trainer=trainer,
        tokenizer=processor,
        torch=torch,
        experiment=experiment,
        config=config,
        dataset=dataset,
        output_directory=output_directory,
        resolved_revision=config.base_model_revision,
        evaluate=True,
        trainer_family="transformers-speecht5-seq2seq-5",
        runtime_metadata_extra={
            "voice_profile_id": config.voice_profile_id,
            "language": config.language,
            "sampling_rate": config.sampling_rate,
            "speaker_embedding_dimension": config.speaker_embedding_dimension,
            "speech_contract_version": config.speech_contract_version,
        },
    )
    embedding_digest = _write_tts_speaker_embedding(output_directory, voice_embedding)
    _write_tts_voice_profile(
        output_directory,
        config,
        records=[*dataset.train_records, *dataset.validation_records],
        speaker_embedding_sha256=embedding_digest,
    )
    _verify_tts_output_directory(output_directory)
    return outcome


def _tts_dependencies() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
        datasets = importlib.import_module("datasets")
        soundfile = importlib.import_module("soundfile")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:  # pragma: no cover - exercised in the GPU image
        raise TrainingBackendError("tts_training_dependencies_are_not_installed") from exc
    return {
        "torch": torch,
        "Dataset": datasets.Dataset,
        "SpeechT5Processor": transformers.SpeechT5Processor,
        "SpeechT5ForTextToSpeech": transformers.SpeechT5ForTextToSpeech,
        "Seq2SeqTrainer": transformers.Seq2SeqTrainer,
        "Seq2SeqTrainingArguments": transformers.Seq2SeqTrainingArguments,
        "set_seed": transformers.set_seed,
        "soundfile": soundfile,
    }


def _prepare_tts_runtime(
    torch: Any,
    config: CompiledTtsConfig,
    output_directory: Path,
    set_seed: Any,
) -> None:
    if config.distributed.world_size != 1:
        raise TrainingBackendError("tts_training_requires_single_gpu")
    if not torch.cuda.is_available():
        raise TrainingBackendError("cuda_gpu_is_not_available")
    if torch.cuda.device_count() != 1:
        raise TrainingBackendError("exactly_one_visible_gpu_is_required")
    output_directory.mkdir(parents=True, exist_ok=False)
    set_seed(config.seed)
    torch.cuda.reset_peak_memory_stats()


def _tts_records(
    records: list[dict[str, Any]],
    *,
    candidate_ids: list[str],
    processor: Any,
    config: CompiledTtsConfig,
    dependencies: dict[str, Any],
) -> list[dict[str, Any]]:
    if len(records) != len(candidate_ids):
        raise TrainingBackendError("tts_candidate_binding_is_invalid")
    prepared: list[dict[str, Any]] = []
    soundfile = dependencies["soundfile"]
    for candidate_id, record in zip(candidate_ids, records, strict=True):
        embedding = record.get("speaker_embedding")
        evidence_digest = str(record.get("voice_usage_evidence_sha256", ""))
        if (
            record.get("language") != config.language
            or record.get("voice_profile_id") != config.voice_profile_id
            or record.get("voice_source_type") not in {"PLATFORM_SYNTHETIC", "LICENSED_STUDIO"}
            or len(evidence_digest.removeprefix("sha256:")) != 64
            or not isinstance(embedding, (list, tuple))
            or len(embedding) != config.speaker_embedding_dimension
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in embedding
            )
        ):
            raise TrainingBackendError("tts_voice_profile_binding_is_invalid")
        try:
            audio, sampling_rate = soundfile.read(
                BytesIO(record["media_content"]),
                dtype="float32",
                always_2d=False,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise TrainingBackendError("tts_audio_decode_failed") from exc
        if int(sampling_rate) != config.sampling_rate:
            raise TrainingBackendError("tts_audio_sampling_rate_mismatch")
        if getattr(audio, "ndim", 1) == 2:
            audio = audio.mean(axis=1)
        duration = len(audio) / float(config.sampling_rate)
        if duration <= 0 or duration > config.max_audio_seconds or not _finite_audio(audio):
            raise TrainingBackendError("tts_audio_is_out_of_range")
        encoded = processor(
            text=str(record["target_text"]),
            audio_target=audio,
            sampling_rate=config.sampling_rate,
            return_attention_mask=False,
        )
        try:
            input_ids = encoded["input_ids"]
            labels = encoded["labels"]
        except (KeyError, TypeError) as exc:
            raise TrainingBackendError("tts_processor_output_is_invalid") from exc
        label_shape = getattr(labels, "shape", ())
        if len(label_shape) == 3 and label_shape[0] == 1:
            labels = labels[0]
            label_shape = getattr(labels, "shape", ())
        if len(label_shape) != 2:
            raise TrainingBackendError("tts_processor_output_is_invalid")
        prepared.append(
            {
                "candidate_id": candidate_id,
                "input_ids": input_ids,
                "labels": labels,
                "speaker_embeddings": [float(value) for value in embedding],
            }
        )
    return prepared


def _finite_audio(audio: Any) -> bool:
    values = audio.tolist() if hasattr(audio, "tolist") else audio
    try:
        return all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


class _TtsDataCollator:
    def __init__(self, processor: Any, torch: Any, model: Any) -> None:
        self._processor = processor
        self._torch = torch
        self._reduction_factor = int(getattr(model.config, "reduction_factor", 1))

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = dict(
            self._processor.pad(
                input_ids=[{"input_ids": item["input_ids"]} for item in features],
                labels=[{"input_values": item["labels"]} for item in features],
                return_tensors="pt",
            )
        )
        decoder_mask = batch.pop("decoder_attention_mask", None)
        if decoder_mask is not None:
            batch["labels"] = batch["labels"].masked_fill(decoder_mask.unsqueeze(-1).ne(1), -100)
        labels = batch["labels"]
        shape = getattr(labels, "shape", None)
        if shape is not None and len(shape) > 1 and shape[1] % self._reduction_factor:
            target_length = shape[1] - (shape[1] % self._reduction_factor)
            batch["labels"] = labels[:, :target_length]
        batch["speaker_embeddings"] = self._torch.tensor(
            [item["speaker_embeddings"] for item in features],
            dtype=self._torch.float32,
        )
        return batch


def _write_tts_voice_profile(
    output_directory: Path,
    config: CompiledTtsConfig,
    *,
    records: list[dict[str, Any]],
    speaker_embedding_sha256: str,
) -> None:
    source_types = sorted({str(item["voice_source_type"]) for item in records})
    evidence_digests = sorted({str(item["voice_usage_evidence_sha256"]) for item in records})
    profile = {
        "schema_version": "industrial-ops-tts-voice-profile/v1",
        "voice_profile_id": config.voice_profile_id,
        "language": config.language,
        "sampling_rate": config.sampling_rate,
        "speaker_embedding_dimension": config.speaker_embedding_dimension,
        "speech_contract_version": config.speech_contract_version,
        "model_family": config.model_family,
        "base_model_revision": config.base_model_revision,
        "authorized_source_types": source_types,
        "voice_usage_evidence_sha256": evidence_digests,
        "speaker_embedding_file": "industrial-ops-tts-speaker-embedding.npy",
        "speaker_embedding_sha256": speaker_embedding_sha256,
    }
    (output_directory / "industrial-ops-tts-voice-profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _tts_voice_embedding(
    records: list[dict[str, Any]], config: CompiledTtsConfig
) -> tuple[float, ...]:
    embeddings = {
        tuple(float(value) for value in record.get("speaker_embedding", ())) for record in records
    }
    if (
        len(embeddings) != 1
        or len(next(iter(embeddings), ())) != config.speaker_embedding_dimension
    ):
        raise TrainingBackendError("tts_voice_profile_embedding_changed_between_samples")
    return next(iter(embeddings))


def _write_tts_speaker_embedding(output_directory: Path, embedding: tuple[float, ...]) -> str:
    try:
        numpy = importlib.import_module("numpy")
        path = output_directory / "industrial-ops-tts-speaker-embedding.npy"
        numpy.save(
            path,
            numpy.asarray(embedding, dtype="float32"),
            allow_pickle=False,
        )
        return "sha256:" + sha256(path.read_bytes()).hexdigest()
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise TrainingBackendError("tts_speaker_embedding_artifact_write_failed") from exc


def _verify_tts_output_directory(output_directory: Path) -> None:
    required = {
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "industrial-ops-tts-voice-profile.json",
        "industrial-ops-tts-speaker-embedding.npy",
    }
    if not all((output_directory / name).is_file() for name in required) or not (
        (output_directory / "model.safetensors").is_file()
        or (output_directory / "model.safetensors.index.json").is_file()
    ):
        raise TrainingBackendError("tts_training_output_is_incomplete")


def _sentence_transformer_dependencies() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
        sentence_transformers = importlib.import_module("sentence_transformers")
        cross_encoder = importlib.import_module("sentence_transformers.cross_encoder")
        embedding_losses = importlib.import_module("sentence_transformers.losses")
        reranker_losses = importlib.import_module("sentence_transformers.cross_encoder.losses")
        training_args = importlib.import_module("sentence_transformers.training_args")
        datasets = importlib.import_module("datasets")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:  # pragma: no cover - exercised in the GPU image
        raise TrainingBackendError("specialized_training_dependencies_are_not_installed") from exc
    return {
        "torch": torch,
        "Dataset": datasets.Dataset,
        "set_seed": transformers.set_seed,
        "SentenceTransformer": sentence_transformers.SentenceTransformer,
        "SentenceTransformerTrainer": sentence_transformers.SentenceTransformerTrainer,
        "SentenceTransformerTrainingArguments": (
            sentence_transformers.SentenceTransformerTrainingArguments
        ),
        "BatchSamplers": training_args.BatchSamplers,
        "EmbeddingLoss": embedding_losses.MultipleNegativesRankingLoss,
        "CrossEncoder": cross_encoder.CrossEncoder,
        "CrossEncoderTrainer": cross_encoder.CrossEncoderTrainer,
        "CrossEncoderTrainingArguments": cross_encoder.CrossEncoderTrainingArguments,
        "RerankerLoss": reranker_losses.BinaryCrossEntropyLoss,
    }


def _multimodal_dependencies() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
        datasets = importlib.import_module("datasets")
        peft = importlib.import_module("peft")
        transformers = importlib.import_module("transformers")
        trl = importlib.import_module("trl")
        Image = importlib.import_module("PIL.Image")
        soundfile = importlib.import_module("soundfile")
    except ImportError as exc:  # pragma: no cover - exercised in the GPU image
        raise TrainingBackendError("multimodal_training_dependencies_are_not_installed") from exc
    return {
        "torch": torch,
        "Dataset": datasets.Dataset,
        "LoraConfig": peft.LoraConfig,
        "get_peft_model": peft.get_peft_model,
        "prepare_model_for_kbit_training": peft.prepare_model_for_kbit_training,
        "AutoProcessor": transformers.AutoProcessor,
        "AutoModelForImageTextToText": transformers.AutoModelForImageTextToText,
        "AutoModelForSpeechSeq2Seq": transformers.AutoModelForSpeechSeq2Seq,
        "BitsAndBytesConfig": transformers.BitsAndBytesConfig,
        "Seq2SeqTrainer": transformers.Seq2SeqTrainer,
        "Seq2SeqTrainingArguments": transformers.Seq2SeqTrainingArguments,
        "TrainerCallback": transformers.TrainerCallback,
        "set_seed": transformers.set_seed,
        "SFTConfig": trl.SFTConfig,
        "SFTTrainer": trl.SFTTrainer,
        "Image": Image,
        "soundfile": soundfile,
    }


def _prepare_peft_runtime(
    torch: Any,
    config: CompiledPeftConfig,
    output_directory: Path,
    set_seed: Any,
) -> None:
    if not torch.cuda.is_available():
        raise TrainingBackendError("cuda_gpu_is_not_available")
    if torch.cuda.device_count() != 1:
        raise TrainingBackendError("exactly_one_visible_gpu_is_required")
    output_directory.mkdir(parents=True, exist_ok=False)
    set_seed(config.seed)
    torch.cuda.reset_peak_memory_stats()


def _vlm_records(
    records: list[dict[str, Any]],
    config: CompiledVlmConfig,
    Image: Any,
    *,
    curriculum: VlmCurriculumConfig | None,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    targets = set(curriculum.target_candidate_ids) if curriculum is not None else set()
    matched: set[str] = set()
    for record in records:
        candidate_id = record.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise TrainingBackendError("vlm_candidate_id_is_missing")
        try:
            image = Image.open(BytesIO(record["media_content"]))
            image.load()
            image = image.convert("RGB")
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise TrainingBackendError("vlm_image_decode_failed") from exc
        if image.width * image.height > config.max_image_pixels:
            raise TrainingBackendError("vlm_image_exceeds_registered_pixel_limit")
        repeat = curriculum.repeat_factor if candidate_id in targets else 1
        if candidate_id in targets:
            matched.add(candidate_id)
        prepared.extend({"messages": record["messages"], "images": [image]} for _ in range(repeat))
    if matched != targets:
        raise TrainingBackendError("vlm_curriculum_candidate_is_not_in_training_snapshot")
    return prepared


def _resume_learning_rate_callback(callback_base: Any, learning_rate: float) -> Any:
    class ResumeLearningRateCallback(callback_base):  # type: ignore[misc, valid-type]
        def on_train_begin(
            self,
            args: Any,
            state: Any,
            control: Any,
            **kwargs: Any,
        ) -> Any:
            optimizer = kwargs.get("optimizer")
            if optimizer is None:
                raise TrainingBackendError("resume_optimizer_is_missing_for_learning_rate_override")
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
                group["initial_lr"] = learning_rate
            scheduler = kwargs.get("lr_scheduler")
            if scheduler is not None:
                if hasattr(scheduler, "base_lrs"):
                    scheduler.base_lrs = [learning_rate for _ in scheduler.base_lrs]
                if hasattr(scheduler, "_last_lr"):
                    scheduler._last_lr = [learning_rate for _ in scheduler._last_lr]
            return control

    return ResumeLearningRateCallback()


def _asr_records(
    records: list[dict[str, Any]],
    processor: Any,
    config: CompiledAsrConfig,
    dependencies: dict[str, Any],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    soundfile = dependencies["soundfile"]
    for record in records:
        if record.get("language") != config.language:
            raise TrainingBackendError("asr_sample_language_differs_from_registered_language")
        try:
            audio, sampling_rate = soundfile.read(
                BytesIO(record["media_content"]),
                dtype="float32",
                always_2d=False,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise TrainingBackendError("asr_audio_decode_failed") from exc
        if int(sampling_rate) != config.sampling_rate:
            raise TrainingBackendError("asr_audio_sampling_rate_mismatch")
        if getattr(audio, "ndim", 1) == 2:
            audio = audio.mean(axis=1)
        duration = len(audio) / float(config.sampling_rate)
        if duration <= 0 or duration > config.max_audio_seconds:
            raise TrainingBackendError("asr_audio_duration_is_out_of_range")
        features = processor.feature_extractor(
            audio,
            sampling_rate=config.sampling_rate,
        ).input_features[0]
        labels = processor.tokenizer(str(record["transcript"])).input_ids
        prepared.append(
            {
                "input_features": features.tolist() if hasattr(features, "tolist") else features,
                "labels": labels,
            }
        )
    return prepared


class _AsrDataCollator:
    def __init__(self, processor: Any, torch: Any) -> None:
        self._processor = processor
        self._torch = torch

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        inputs = [{"input_features": feature["input_features"]} for feature in features]
        batch: dict[str, Any] = dict(
            self._processor.feature_extractor.pad(inputs, return_tensors="pt")
        )
        labels = self._processor.tokenizer.pad(
            [{"input_ids": feature["labels"]} for feature in features],
            return_tensors="pt",
        )
        label_ids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        decoder_start = self._processor.tokenizer.bos_token_id
        if (
            decoder_start is not None
            and label_ids.shape[1] > 0
            and (label_ids[:, 0] == decoder_start).all().cpu().item()
        ):
            label_ids = label_ids[:, 1:]
        batch["labels"] = label_ids.to(dtype=self._torch.long)
        return batch


def _prepare_specialized_runtime(
    torch: Any,
    config: CompiledSpecializedConfig,
    output_directory: Path,
    set_seed: Any,
) -> None:
    if not torch.cuda.is_available():
        raise TrainingBackendError("cuda_gpu_is_not_available")
    if torch.cuda.device_count() != 1:
        raise TrainingBackendError("exactly_one_visible_gpu_is_required")
    output_directory.mkdir(parents=True, exist_ok=False)
    set_seed(config.seed)
    torch.cuda.reset_peak_memory_stats()


def _sentence_transformer_revision(model: Any) -> str | None:
    try:
        return getattr(model[0].auto_model.config, "_commit_hash", None)
    except (AttributeError, IndexError, KeyError, TypeError):
        return None


def _run_specialized_trainer(
    *,
    trainer: Any,
    tokenizer: Any,
    torch: Any,
    experiment: TrainingExperimentRecord,
    config: CompiledSpecializedConfig,
    dataset: TrainingDatasetBundle,
    output_directory: Path,
    resolved_revision: str | None,
    trainer_family: str,
) -> TrainingOutcome:
    started = time.monotonic()
    train_result = trainer.train()
    evaluation_metrics = trainer.evaluate()
    runtime_seconds = time.monotonic() - started
    trainer.save_model(str(output_directory))
    tokenizer.save_pretrained(str(output_directory))
    metrics = _numeric_metrics(
        {
            **dict(getattr(train_result, "metrics", {})),
            **dict(evaluation_metrics),
            "train_runtime_seconds": runtime_seconds,
            "peak_gpu_memory_allocated_bytes": float(torch.cuda.max_memory_allocated()),
            "peak_gpu_memory_reserved_bytes": float(torch.cuda.max_memory_reserved()),
            "gpu_hours": runtime_seconds / 3600.0,
        }
    )
    estimated_cost = metrics["gpu_hours"] * config.gpu_hourly_cost_usd
    device = torch.cuda.get_device_properties(0)
    cost_summary = {
        "currency": "USD",
        "gpu_hourly_cost_usd": config.gpu_hourly_cost_usd,
        "gpu_hours": metrics["gpu_hours"],
        "estimated_compute_cost_usd": estimated_cost,
        "pricing_source": "registered_training_config",
        "energy_kwh": None,
        "energy_source": "not_available_without_dcgm_or_nvml_energy_counter",
    }
    runtime_metadata = {
        "training_method": experiment.method,
        "trainer_family": trainer_family,
        "gpu_name": device.name,
        "gpu_total_memory_bytes": int(device.total_memory),
        "gpu_count": 1,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "resolved_base_model_revision": resolved_revision,
        "train_source_rows": len(dataset.train),
        "validation_source_rows": len(dataset.validation),
        "train_trainer_records": len(dataset.train_records),
        "validation_trainer_records": len(dataset.validation_records),
    }
    (output_directory / "industrial-ops-training-report.json").write_text(
        json.dumps(
            {
                "metrics": metrics,
                "cost_summary": cost_summary,
                "runtime": runtime_metadata,
                "compiled_config_digest": config.digest,
                "dataset_manifest_hash": dataset.manifest_hash,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return TrainingOutcome(
        metrics=metrics,
        cost_summary=cost_summary,
        output_directory=output_directory,
        runtime_metadata=runtime_metadata,
    )


def _post_training_dependencies(method: str) -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
        Dataset = importlib.import_module("datasets").Dataset
        peft = importlib.import_module("peft")
        transformers = importlib.import_module("transformers")
        if method == "DPO":
            trl = importlib.import_module("trl")
            trainers = {"DPOConfig": trl.DPOConfig, "DPOTrainer": trl.DPOTrainer}
        elif method == "GRPO":
            trl = importlib.import_module("trl")
            trainers = {"GRPOConfig": trl.GRPOConfig, "GRPOTrainer": trl.GRPOTrainer}
        else:
            ppo = importlib.import_module("trl.experimental.ppo")
            trainers = {"PPOConfig": ppo.PPOConfig, "PPOTrainer": ppo.PPOTrainer}
    except ImportError as exc:  # pragma: no cover - exercised in the GPU image
        raise TrainingBackendError("training_dependencies_are_not_installed") from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": peft.LoraConfig,
        "prepare_model_for_kbit_training": peft.prepare_model_for_kbit_training,
        "AutoModelForCausalLM": transformers.AutoModelForCausalLM,
        "AutoModelForSequenceClassification": transformers.AutoModelForSequenceClassification,
        "AutoTokenizer": transformers.AutoTokenizer,
        "BitsAndBytesConfig": transformers.BitsAndBytesConfig,
        "set_seed": transformers.set_seed,
        **trainers,
    }


def _load_policy(
    experiment: TrainingExperimentRecord,
    config: CompiledPeftConfig,
    dependencies: dict[str, Any],
    output_directory: Path,
) -> tuple[Any, Any, Any, str | None]:
    torch = dependencies["torch"]
    if not torch.cuda.is_available():
        raise TrainingBackendError("cuda_gpu_is_not_available")
    if torch.cuda.device_count() != 1:
        raise TrainingBackendError("exactly_one_visible_gpu_is_required")
    output_directory.mkdir(parents=True, exist_ok=False)
    dependencies["set_seed"](config.seed)
    torch.cuda.reset_peak_memory_stats()
    dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float16
    model_kwargs: dict[str, Any] = {
        "revision": config.base_model_revision,
        "torch_dtype": dtype,
        "trust_remote_code": False,
        "device_map": {"": 0},
    }
    if config.quantization is not None:
        model_kwargs["quantization_config"] = dependencies["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quantization.quant_type,
            bnb_4bit_use_double_quant=config.quantization.use_double_quant,
            bnb_4bit_compute_dtype=dtype,
        )
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        experiment.base_model_id,
        revision=config.base_model_revision,
        trust_remote_code=False,
    )
    verify_model_contract(
        tokenizer=tokenizer,
        revision=config.base_model_revision,
        base_model_digest=experiment.base_model_digest,
        tokenizer_digest=experiment.tokenizer_digest,
        chat_template_digest=experiment.chat_template_digest,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = dependencies["AutoModelForCausalLM"].from_pretrained(
        experiment.base_model_id, **model_kwargs
    )
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision and resolved_revision != config.base_model_revision:
        raise TrainingBackendError("resolved_base_model_revision_changed")
    if config.quantization is not None:
        model = dependencies["prepare_model_for_kbit_training"](
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
    peft_config = dependencies["LoraConfig"](
        task_type="CAUSAL_LM",
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        target_modules=list(config.target_modules),
    )
    return tokenizer, model, peft_config, resolved_revision


def _industrial_json_exact_reward(
    completions: list[Any], ground_truth: list[str], **_: Any
) -> list[float]:
    """Deterministic GRPO reward; no executable user plugin or judge model."""

    return [
        score_reward(LEGACY_REWARD_CONTRACT_VERSION, completion, target)
        for completion, target in zip(completions, ground_truth, strict=True)
    ]


def _reward_function(version: str) -> Any:
    def registered_reward(completions: list[Any], ground_truth: list[str], **_: Any) -> list[float]:
        return [
            score_reward(version, completion, target)
            for completion, target in zip(completions, ground_truth, strict=True)
        ]

    registered_reward.__name__ = version.replace("-", "_")
    return registered_reward


def _prepare_grpo_records(
    records: list[dict[str, Any]], config: CompiledGrpoConfig
) -> tuple[list[dict[str, Any]], list[RewardTargetMigration]]:
    prepared: list[dict[str, Any]] = []
    migrations: list[RewardTargetMigration] = []
    for record in records:
        source_version = record.get("reward_contract_source_version")
        target = record.get("ground_truth")
        if not isinstance(source_version, str) or not isinstance(target, str):
            raise TrainingBackendError("grpo_reward_record_is_incomplete")
        try:
            migration = migrate_reward_target(
                target,
                source_version=source_version,
                target_version=config.reward_contract_version,
            )
        except RewardContractError as exc:
            raise TrainingBackendError(str(exc)) from exc
        migrated_record = dict(record)
        migrated_record["ground_truth"] = migration.canonical_target
        migrated_record["reward_contract_source_version"] = migration.target_version
        prepared.append(migrated_record)
        migrations.append(migration)
    return prepared, migrations


def _tokenize_ppo_records(tokenizer: Any, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokenized: list[dict[str, Any]] = []
    for record in records:
        input_ids = tokenizer.apply_chat_template(
            record["prompt"], tokenize=True, add_generation_prompt=True
        )
        if not isinstance(input_ids, list) or not input_ids:
            raise TrainingBackendError("ppo_prompt_tokenization_failed")
        tokenized.append({"input_ids": input_ids})
    return tokenized


def _verify_resolved_revision(model: Any, expected: str, reason: str) -> None:
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved and resolved != expected:
        raise TrainingBackendError(reason)


def _run_post_trainer(
    *,
    trainer: Any,
    tokenizer: Any,
    torch: Any,
    experiment: TrainingExperimentRecord,
    config: CompiledTrainingConfig,
    dataset: TrainingDatasetBundle,
    output_directory: Path,
    resolved_revision: str | None,
    evaluate: bool,
    resume_from_checkpoint: Path | None = None,
    trainer_family: str = "trl-0.28",
    runtime_metadata_extra: dict[str, Any] | None = None,
) -> TrainingOutcome:
    started = time.monotonic()
    if resume_from_checkpoint is None:
        train_result = trainer.train()
    else:
        train_result = trainer.train(resume_from_checkpoint=str(resume_from_checkpoint))
    evaluation_metrics = trainer.evaluate() if evaluate else {}
    runtime_seconds = time.monotonic() - started
    trainer.save_model(str(output_directory))
    tokenizer.save_pretrained(str(output_directory))
    trained_model = trainer.model
    total_parameters = sum(parameter.numel() for parameter in trained_model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in trained_model.parameters() if parameter.requires_grad
    )
    quantized_modules = sum(
        1
        for module in trained_model.modules()
        if type(module).__name__ in {"Linear4bit", "Linear8bitLt"}
    )
    optimizer = getattr(trainer, "optimizer", None)
    optimizer_learning_rates = (
        sorted(
            {
                float(group["lr"])
                for group in optimizer.param_groups
                if isinstance(group.get("lr"), (int, float))
            }
        )
        if optimizer is not None
        else []
    )
    metrics = _numeric_metrics(
        {
            **dict(getattr(train_result, "metrics", {})),
            **dict(evaluation_metrics),
            "optimizer_steps": float(getattr(trainer.state, "global_step", 0)),
            "total_parameters": float(total_parameters),
            "trainable_parameters": float(trainable_parameters),
            "quantized_modules": float(quantized_modules),
            "train_runtime_seconds": runtime_seconds,
            "peak_gpu_memory_allocated_bytes": float(torch.cuda.max_memory_allocated()),
            "peak_gpu_memory_reserved_bytes": float(torch.cuda.max_memory_reserved()),
            "gpu_hours": runtime_seconds / 3600.0,
        }
    )
    estimated_cost = metrics["gpu_hours"] * config.gpu_hourly_cost_usd
    device = torch.cuda.get_device_properties(0)
    cost_summary = {
        "currency": "USD",
        "gpu_hourly_cost_usd": config.gpu_hourly_cost_usd,
        "gpu_hours": metrics["gpu_hours"],
        "estimated_compute_cost_usd": estimated_cost,
        "pricing_source": "registered_training_config",
        "energy_kwh": None,
        "energy_source": "not_available_without_dcgm_or_nvml_energy_counter",
    }
    resume_descriptor = (
        config.resume_from_checkpoint if isinstance(config, CompiledPeftConfig) else None
    )
    runtime_metadata = {
        "training_method": experiment.method,
        "trainer_family": trainer_family,
        "ppo_profile": "experimental" if experiment.method == "PPO" else "stable",
        "gpu_name": device.name,
        "gpu_total_memory_bytes": int(device.total_memory),
        "gpu_count": 1,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "resolved_base_model_revision": resolved_revision,
        "resumed_from_checkpoint": resume_from_checkpoint is not None,
        "resume_source_experiment_id": (
            resume_descriptor.source_experiment_id if resume_descriptor is not None else None
        ),
        "resume_source_artifact_id": (
            resume_descriptor.artifact_id if resume_descriptor is not None else None
        ),
        "resume_checkpoint_path": (
            resume_descriptor.checkpoint_path if resume_descriptor is not None else None
        ),
        "optimizer_learning_rates": optimizer_learning_rates,
        "train_rows": len(dataset.train),
        "validation_rows": len(dataset.validation),
        **(runtime_metadata_extra or {}),
    }
    (output_directory / "industrial-ops-training-report.json").write_text(
        json.dumps(
            {
                "metrics": metrics,
                "cost_summary": cost_summary,
                "runtime": runtime_metadata,
                "compiled_config_digest": config.digest,
                "dataset_manifest_hash": dataset.manifest_hash,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return TrainingOutcome(
        metrics=metrics,
        cost_summary=cost_summary,
        output_directory=output_directory,
        runtime_metadata=runtime_metadata,
    )


def _numeric_metrics(values: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, raw in values.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        value = float(raw)
        if math.isfinite(value):
            metrics[name.replace("/", "_")[:250]] = value
    if "gpu_hours" not in metrics:
        raise TrainingBackendError("training_runtime_metric_is_missing")
    return metrics


def _finite_metrics(values: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, raw in values.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TrainingBackendError(f"training_metric_is_missing:{name}")
        value = float(raw)
        if not math.isfinite(value):
            raise TrainingBackendError(f"training_metric_is_not_finite:{name}")
        metrics[name] = value
    return metrics
