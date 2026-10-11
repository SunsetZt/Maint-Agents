"""Native PyTorch Transformer training backend for industrial telemetry sequences."""

from __future__ import annotations

import importlib
import json
import math
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from industrial_ops_agent.persistence.models import TrainingExperimentRecord
from industrial_ops_agent.predictive_maintenance.timeseries_model import (
    TimeseriesModelSpec,
    build_timeseries_transformer,
)
from industrial_ops_agent.predictive_maintenance.training_dataset import (
    TelemetrySequenceSample,
    TelemetryTrainingDatasetBundle,
)
from industrial_ops_agent.training.config import CompiledTimeseriesTransformerConfig

if TYPE_CHECKING:
    from industrial_ops_agent.training.backend import TrainingOutcome


def train_timeseries_transformer(
    experiment: TrainingExperimentRecord,
    config: CompiledTimeseriesTransformerConfig,
    dataset: TelemetryTrainingDatasetBundle,
    output_directory: Path,
) -> TrainingOutcome:
    from industrial_ops_agent.training.backend import TrainingBackendError, TrainingOutcome

    try:
        torch = importlib.import_module("torch")
        save_file = importlib.import_module("safetensors.torch").save_file
    except ImportError as exc:  # pragma: no cover - GPU image responsibility
        raise TrainingBackendError("timeseries_training_dependencies_are_not_installed") from exc
    if dataset.contract_version != config.sequence_contract_version:
        raise TrainingBackendError("timeseries_dataset_contract_mismatch")
    if dataset.signal_order != config.signal_order:
        raise TrainingBackendError("timeseries_signal_order_mismatch")
    if not torch.cuda.is_available():
        raise TrainingBackendError("cuda_gpu_is_not_available")
    if torch.cuda.device_count() != 1:
        raise TrainingBackendError("exactly_one_visible_gpu_is_required")

    output_directory.mkdir(parents=True, exist_ok=False)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")

    means, scales = _normalizer(dataset.train, len(config.signal_order))
    model_spec = TimeseriesModelSpec(
        architecture_revision=config.base_model_revision,
        sequence_contract_version=config.sequence_contract_version,
        signal_order=config.signal_order,
        max_sequence_length=config.max_sequence_length,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    )
    model = build_timeseries_transformer(torch, model_spec).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "float16")
    autocast_dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float16
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.data_seed)
    train_batches = _batches(dataset.train, config.per_device_train_batch_size, generator, torch)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    losses: list[float] = []
    for step in range(1, config.max_steps + 1):
        batch = next(train_batches)
        values, observed, padding = _tensor_batch(
            batch,
            max_length=config.max_sequence_length,
            means=means,
            scales=scales,
            torch=torch,
            device=device,
        )
        random_mask = torch.rand(observed.shape, device=device) < config.mask_probability
        reconstruction_mask = observed & random_mask
        if not reconstruction_mask.any():
            reconstruction_mask = observed
        inputs = values.masked_fill(reconstruction_mask, 0.0)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            reconstructed = model(inputs, padding)
            loss = ((reconstructed - values).square() * reconstruction_mask).sum()
            loss = loss / reconstruction_mask.sum().clamp_min(1)
            scaled_loss = loss / config.gradient_accumulation_steps
        scaler.scale(scaled_loss).backward()
        if step % config.gradient_accumulation_steps == 0 or step == config.max_steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))

    validation_loss, labeled_scores, labeled_targets = _evaluate(
        model,
        dataset.validation,
        config=config,
        means=means,
        scales=scales,
        torch=torch,
        device=device,
        dtype=autocast_dtype,
    )
    runtime_seconds = time.monotonic() - started
    metrics = {
        "train_reconstruction_loss": sum(losses[-10:]) / min(10, len(losses)),
        "validation_reconstruction_loss": validation_loss,
        "labeled_validation_windows": float(len(labeled_targets)),
        "supervised_validation_available": float(
            len(set(labeled_targets)) == 2 and len(labeled_targets) >= 2
        ),
        "train_runtime_seconds": runtime_seconds,
        "peak_gpu_memory_allocated_bytes": float(torch.cuda.max_memory_allocated()),
        "peak_gpu_memory_reserved_bytes": float(torch.cuda.max_memory_reserved()),
        "gpu_hours": runtime_seconds / 3600.0,
    }
    if len(set(labeled_targets)) == 2 and len(labeled_targets) >= 2:
        metrics["validation_auroc"] = _auroc(labeled_scores, labeled_targets)
    if any(not math.isfinite(value) for value in metrics.values()):
        raise TrainingBackendError("timeseries_training_produced_non_finite_metrics")

    cpu_state = {
        name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()
    }
    save_file(cpu_state, str(output_directory / "model.safetensors"))
    model_config = {
        "architecture": "IndustrialTelemetryTransformer",
        "architecture_revision": config.base_model_revision,
        "objective": config.objective,
        "signal_order": list(config.signal_order),
        "sequence_contract_version": config.sequence_contract_version,
        "max_sequence_length": config.max_sequence_length,
        "d_model": config.d_model,
        "nhead": config.nhead,
        "num_layers": config.num_layers,
        "dim_feedforward": config.dim_feedforward,
        "dropout": config.dropout,
    }
    _write_json(output_directory / "model-config.json", model_config)
    _write_json(
        output_directory / "normalizer.json",
        {"signal_order": list(config.signal_order), "means": means, "scales": scales},
    )
    gpu = torch.cuda.get_device_properties(0)
    runtime_metadata = {
        "backend": "native-pytorch-transformer-encoder",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu.name,
        "gpu_total_memory_bytes": int(gpu.total_memory),
        "gpu_count": 1,
        "train_rows": len(dataset.train),
        "validation_rows": len(dataset.validation),
        "dataset_snapshot_id": dataset.snapshot_id,
    }
    cost_summary = {
        "currency": "USD",
        "gpu_hourly_cost_usd": config.gpu_hourly_cost_usd,
        "gpu_hours": metrics["gpu_hours"],
        "estimated_compute_cost_usd": metrics["gpu_hours"] * config.gpu_hourly_cost_usd,
        "pricing_source": "registered_training_config",
        "energy_kwh": None,
        "energy_source": "not_available_without_dcgm_or_nvml_energy_counter",
    }
    _write_json(
        output_directory / "industrial-ops-timeseries-training-report.json",
        {
            "experiment_id": experiment.experiment_id,
            "metrics": metrics,
            "cost_summary": cost_summary,
            "runtime": runtime_metadata,
            "compiled_config_digest": config.digest,
            "dataset_manifest_hash": dataset.manifest_hash,
        },
    )
    return TrainingOutcome(
        metrics=metrics,
        cost_summary=cost_summary,
        output_directory=output_directory,
        runtime_metadata=runtime_metadata,
    )


def _normalizer(
    samples: tuple[TelemetrySequenceSample, ...], signal_count: int
) -> tuple[list[float], list[float]]:
    observed: list[list[float]] = [[] for _ in range(signal_count)]
    for sample in samples:
        for values, mask in zip(sample.sequence, sample.mask, strict=True):
            for index, (value, present) in enumerate(zip(values, mask, strict=True)):
                if present:
                    observed[index].append(value)
    if any(not values for values in observed):
        raise ValueError("training_split_has_unobserved_signal")
    means = [sum(values) / len(values) for values in observed]
    scales = []
    for signal_values, mean in zip(observed, means, strict=True):
        variance = sum((value - mean) ** 2 for value in signal_values) / len(signal_values)
        scales.append(max(math.sqrt(variance), 1e-6))
    return means, scales


def _batches(
    samples: tuple[TelemetrySequenceSample, ...], size: int, generator: Any, torch: Any
) -> Any:
    while True:
        order = torch.randperm(len(samples), generator=generator).tolist()
        for start in range(0, len(order), size):
            yield tuple(samples[index] for index in order[start : start + size])


def _tensor_batch(
    samples: tuple[TelemetrySequenceSample, ...],
    *,
    max_length: int,
    means: list[float],
    scales: list[float],
    torch: Any,
    device: Any,
) -> tuple[Any, Any, Any]:
    length = min(max(len(sample.sequence) for sample in samples), max_length)
    signal_count = len(means)
    values = torch.zeros((len(samples), length, signal_count), dtype=torch.float32)
    observed = torch.zeros((len(samples), length, signal_count), dtype=torch.bool)
    padding = torch.ones((len(samples), length), dtype=torch.bool)
    for batch_index, sample in enumerate(samples):
        sample_length = min(len(sample.sequence), length)
        padding[batch_index, :sample_length] = False
        for sequence_index in range(sample_length):
            for signal_index in range(signal_count):
                present = sample.mask[sequence_index][signal_index]
                observed[batch_index, sequence_index, signal_index] = present
                if present:
                    values[batch_index, sequence_index, signal_index] = (
                        sample.sequence[sequence_index][signal_index] - means[signal_index]
                    ) / scales[signal_index]
    return values.to(device), observed.to(device), padding.to(device)


def _evaluate(
    model: Any,
    samples: tuple[TelemetrySequenceSample, ...],
    *,
    config: CompiledTimeseriesTransformerConfig,
    means: list[float],
    scales: list[float],
    torch: Any,
    device: Any,
    dtype: Any,
) -> tuple[float, list[float], list[int]]:
    model.eval()
    total_error = 0.0
    total_values = 0
    scores: list[float] = []
    targets: list[int] = []
    with torch.no_grad():
        for start in range(0, len(samples), config.per_device_eval_batch_size):
            batch = samples[start : start + config.per_device_eval_batch_size]
            values, observed, padding = _tensor_batch(
                batch,
                max_length=config.max_sequence_length,
                means=means,
                scales=scales,
                torch=torch,
                device=device,
            )
            with torch.autocast(device_type="cuda", dtype=dtype):
                prediction = model(values.masked_fill(~observed, 0.0), padding)
            squared = (prediction.float() - values).square() * observed
            total_error += float(squared.sum().cpu())
            total_values += int(observed.sum().cpu())
            per_sample = squared.sum(dim=(1, 2)) / observed.sum(dim=(1, 2)).clamp_min(1)
            for sample, score in zip(batch, per_sample.tolist(), strict=True):
                if sample.label in {0, 1}:
                    scores.append(float(score))
                    targets.append(sample.label)
    model.train()
    return total_error / max(total_values, 1), scores, targets


def _auroc(scores: list[float], targets: list[int]) -> float:
    positives = [score for score, target in zip(scores, targets, strict=True) if target == 1]
    negatives = [score for score, target in zip(scores, targets, strict=True) if target == 0]
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
