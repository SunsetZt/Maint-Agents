"""Native PyTorch quantile-regression backend for supervised RUL sequences."""

from __future__ import annotations

import importlib
import json
import math
import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from industrial_ops_agent.persistence.models import TrainingExperimentRecord
from industrial_ops_agent.predictive_maintenance.rul_model import (
    RulModelSpec,
    build_rul_transformer,
)
from industrial_ops_agent.predictive_maintenance.rul_training_dataset import (
    RulSequenceSample,
    RulTrainingDatasetBundle,
)
from industrial_ops_agent.training.config import CompiledRulTransformerConfig

if TYPE_CHECKING:
    from industrial_ops_agent.training.backend import TrainingOutcome


def train_rul_transformer(
    experiment: TrainingExperimentRecord,
    config: CompiledRulTransformerConfig,
    dataset: RulTrainingDatasetBundle,
    output_directory: Path,
) -> TrainingOutcome:
    from industrial_ops_agent.training.backend import TrainingBackendError, TrainingOutcome

    try:
        torch = importlib.import_module("torch")
        save_file = importlib.import_module("safetensors.torch").save_file
    except ImportError as exc:  # pragma: no cover - GPU image responsibility
        raise TrainingBackendError("rul_training_dependencies_are_not_installed") from exc
    if dataset.contract_version != config.sequence_contract_version:
        raise TrainingBackendError("rul_dataset_contract_mismatch")
    if dataset.signal_order != config.signal_order:
        raise TrainingBackendError("rul_signal_order_mismatch")
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
    spec = RulModelSpec(
        architecture_revision=config.base_model_revision,
        sequence_contract_version=config.sequence_contract_version,
        signal_order=config.signal_order,
        max_sequence_length=config.max_sequence_length,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        quantiles=config.quantiles,
    )
    model = build_rul_transformer(torch, spec).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "float16")
    autocast_dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float16
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.data_seed)
    batches = _batches(dataset.train, config.per_device_train_batch_size, generator, torch)
    quantiles = torch.tensor(config.quantiles, dtype=torch.float32, device=device)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    losses: list[float] = []
    for step in range(1, config.max_steps + 1):
        batch = next(batches)
        values, padding, targets = _tensor_batch(
            batch,
            max_length=config.max_sequence_length,
            means=means,
            scales=scales,
            torch=torch,
            device=device,
        )
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            predicted_log = model(values, padding)
            loss = _pinball_loss(predicted_log, torch.log1p(targets), quantiles, torch)
            scaled_loss = loss / config.gradient_accumulation_steps
        scaler.scale(scaled_loss).backward()
        if step % config.gradient_accumulation_steps == 0 or step == config.max_steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))

    validation = _evaluate(
        model,
        dataset.validation,
        config=config,
        means=means,
        scales=scales,
        torch=torch,
        device=device,
        dtype=autocast_dtype,
    )
    duration_seconds = time.monotonic() - started
    metrics = {
        "train_quantile_loss": sum(losses) / len(losses),
        "validation_pinball_loss": validation[0],
        "validation_median_mae_minutes": validation[1],
        "validation_p10_p90_coverage": validation[2],
        "validation_interval_width_minutes": validation[3],
        "duration_seconds": duration_seconds,
        "gpu_peak_memory_bytes": float(torch.cuda.max_memory_allocated()),
        "gpu_hours": duration_seconds / 3600,
    }
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()},
        str(output_directory / "model.safetensors"),
    )
    _write_json(
        output_directory / "model-config.json",
        {
            "model_type": "IndustrialRulTransformer",
            "architecture_revision": config.base_model_revision,
            "objective": config.objective,
            "target": "lead_time_minutes_from_alert_detection",
            "target_transform": config.target_transform,
            "quantiles": list(config.quantiles),
            "signal_order": list(config.signal_order),
            "sequence_contract_version": config.sequence_contract_version,
            "max_sequence_length": config.max_sequence_length,
            "d_model": config.d_model,
            "nhead": config.nhead,
            "num_layers": config.num_layers,
            "dim_feedforward": config.dim_feedforward,
            "dropout": config.dropout,
        },
    )
    _write_json(
        output_directory / "normalizer.json",
        {"signal_order": list(config.signal_order), "means": means, "scales": scales},
    )
    gpu = torch.cuda.get_device_properties(0)
    runtime_metadata = {
        "backend": "native-pytorch-rul-transformer-encoder",
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
        output_directory / "industrial-ops-rul-training-report.json",
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
    samples: tuple[RulSequenceSample, ...], signal_count: int
) -> tuple[list[float], list[float]]:
    observed: list[list[float]] = [[] for _ in range(signal_count)]
    for sample in samples:
        for values, mask in zip(sample.sequence, sample.mask, strict=True):
            for index, (value, present) in enumerate(zip(values, mask, strict=True)):
                if present:
                    observed[index].append(value)
    if any(not values for values in observed):
        raise ValueError("rul_training_split_has_unobserved_signal")
    means = [sum(values) / len(values) for values in observed]
    scales = []
    for observed_values, mean in zip(observed, means, strict=True):
        variance = sum((value - mean) ** 2 for value in observed_values) / len(
            observed_values
        )
        scales.append(max(math.sqrt(variance), 1e-6))
    return means, scales


def _batches(
    samples: tuple[RulSequenceSample, ...], size: int, generator: Any, torch: Any
) -> Iterator[tuple[RulSequenceSample, ...]]:
    while True:
        order = torch.randperm(len(samples), generator=generator).tolist()
        for start in range(0, len(order), size):
            yield tuple(samples[index] for index in order[start : start + size])


def _tensor_batch(
    samples: tuple[RulSequenceSample, ...],
    *,
    max_length: int,
    means: list[float],
    scales: list[float],
    torch: Any,
    device: Any,
) -> tuple[Any, Any, Any]:
    length = min(max(len(sample.sequence) for sample in samples), max_length)
    values = torch.zeros((len(samples), length, len(means)), dtype=torch.float32, device=device)
    padding = torch.ones((len(samples), length), dtype=torch.bool, device=device)
    targets = torch.empty((len(samples),), dtype=torch.float32, device=device)
    for batch_index, sample in enumerate(samples):
        sequence = sample.sequence[-length:]
        mask = sample.mask[-length:]
        for sequence_index, (signals, present) in enumerate(zip(sequence, mask, strict=True)):
            padding[batch_index, sequence_index] = False
            for signal_index, (value, available) in enumerate(zip(signals, present, strict=True)):
                if available:
                    values[batch_index, sequence_index, signal_index] = (
                        value - means[signal_index]
                    ) / scales[signal_index]
        targets[batch_index] = sample.lead_time_minutes
    return values, padding, targets


def _pinball_loss(predicted: Any, target: Any, quantiles: Any, torch: Any) -> Any:
    error = target.unsqueeze(1) - predicted
    return torch.maximum(quantiles * error, (quantiles - 1) * error).mean()


def _evaluate(
    model: Any,
    samples: tuple[RulSequenceSample, ...],
    *,
    config: CompiledRulTransformerConfig,
    means: list[float],
    scales: list[float],
    torch: Any,
    device: Any,
    dtype: Any,
) -> tuple[float, float, float, float]:
    model.eval()
    losses: list[float] = []
    medians: list[float] = []
    targets_all: list[float] = []
    covered: list[float] = []
    widths: list[float] = []
    quantiles = torch.tensor(config.quantiles, dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, len(samples), config.per_device_eval_batch_size):
            batch = samples[start : start + config.per_device_eval_batch_size]
            values, padding, targets = _tensor_batch(
                batch,
                max_length=config.max_sequence_length,
                means=means,
                scales=scales,
                torch=torch,
                device=device,
            )
            with torch.autocast(device_type="cuda", dtype=dtype):
                predicted_log = model(values, padding)
                loss = _pinball_loss(predicted_log, torch.log1p(targets), quantiles, torch)
            predicted = torch.expm1(predicted_log.float()).clamp_min(0)
            target_values = targets.float()
            losses.append(float(loss.cpu()))
            medians.extend(float(item) for item in predicted[:, 1].cpu())
            targets_all.extend(float(item) for item in target_values.cpu())
            covered.extend(
                float(item)
                for item in (
                    (target_values >= predicted[:, 0]) & (target_values <= predicted[:, 2])
                )
                .float()
                .cpu()
            )
            widths.extend(float(item) for item in (predicted[:, 2] - predicted[:, 0]).cpu())
    model.train()
    mae = sum(
        abs(prediction - target) for prediction, target in zip(medians, targets_all, strict=True)
    )
    return (
        sum(losses) / len(losses),
        mae / len(targets_all),
        sum(covered) / len(covered),
        sum(widths) / len(widths),
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
