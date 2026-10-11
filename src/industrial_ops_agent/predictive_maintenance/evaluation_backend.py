"""Independent rules-vs-Transformer evaluation for governed telemetry windows."""

from __future__ import annotations

import importlib
import json
import math
import time
from pathlib import Path
from typing import Any

from industrial_ops_agent.evaluation.backend import (
    EvaluationBackendError,
    EvaluationRuntimeConfig,
)
from industrial_ops_agent.evaluation.dataset import (
    EvaluationCase,
    RulEvaluationContract,
    TelemetryEvaluationContract,
)
from industrial_ops_agent.evaluation.metrics import ModelObservation
from industrial_ops_agent.persistence.models import TrainingExperimentRecord
from industrial_ops_agent.predictive_maintenance.dataset_contract import SIGNAL_ORDER
from industrial_ops_agent.predictive_maintenance.rul_dataset_contract import (
    SIGNAL_ORDER as RUL_SIGNAL_ORDER,
)
from industrial_ops_agent.predictive_maintenance.rul_model import (
    RulModelSpec,
    build_rul_transformer,
)
from industrial_ops_agent.predictive_maintenance.timeseries_model import (
    TimeseriesModelSpec,
    build_timeseries_transformer,
)


class TimeseriesComponentEvaluationBackend:
    target_profile = "TIMESERIES_COMPONENT"

    def evaluate(
        self,
        *,
        experiment: TrainingExperimentRecord,
        cases: tuple[EvaluationCase, ...],
        config: EvaluationRuntimeConfig,
        adapter_directory: Path | None,
    ) -> tuple[ModelObservation, ...]:
        anomaly_threshold = config.anomaly_threshold
        rule_threshold = config.rule_threshold
        if anomaly_threshold is None or rule_threshold is None:
            raise EvaluationBackendError("timeseries_evaluation_thresholds_are_missing")
        if experiment.method == "TIMESERIES_RULE_BASELINE":
            if adapter_directory is not None:
                raise EvaluationBackendError("timeseries_rule_baseline_has_no_artifact")
            return tuple(_rule_observation(case, rule_threshold) for case in cases)
        if experiment.method != "TIMESERIES_TRANSFORMER" or adapter_directory is None:
            raise EvaluationBackendError("timeseries_candidate_artifact_is_missing")
        return _transformer_observations(
            experiment,
            cases,
            config,
            adapter_directory,
            anomaly_threshold=anomaly_threshold,
        )


class RulComponentEvaluationBackend:
    target_profile = "RUL_COMPONENT"

    def evaluate(
        self,
        *,
        experiment: TrainingExperimentRecord,
        cases: tuple[EvaluationCase, ...],
        config: EvaluationRuntimeConfig,
        adapter_directory: Path | None,
    ) -> tuple[ModelObservation, ...]:
        if experiment.method == "RUL_EMPIRICAL_BASELINE":
            if adapter_directory is not None:
                raise EvaluationBackendError("rul_empirical_baseline_has_no_artifact")
            quantiles = _empirical_rul_quantiles(experiment)
            return tuple(_rul_baseline_observation(case, quantiles) for case in cases)
        if experiment.method != "RUL_TRANSFORMER" or adapter_directory is None:
            raise EvaluationBackendError("rul_candidate_artifact_is_missing")
        return _rul_transformer_observations(experiment, cases, config, adapter_directory)


def _rul_transformer_observations(
    experiment: TrainingExperimentRecord,
    cases: tuple[EvaluationCase, ...],
    config: EvaluationRuntimeConfig,
    directory: Path,
) -> tuple[ModelObservation, ...]:
    try:
        torch = importlib.import_module("torch")
        load_file = importlib.import_module("safetensors.torch").load_file
    except ImportError as exc:  # pragma: no cover - GPU image responsibility
        raise EvaluationBackendError("rul_evaluation_dependencies_are_not_installed") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise EvaluationBackendError("exactly_one_cuda_gpu_is_required")
    model_document = _json_file(directory / "model-config.json")
    normalizer = _json_file(directory / "normalizer.json")
    if (
        model_document.get("model_type") != "IndustrialRulTransformer"
        or model_document.get("objective") != "quantile_regression"
        or model_document.get("target_transform") != "log1p_minutes"
    ):
        raise EvaluationBackendError("rul_model_contract_is_invalid")
    try:
        spec = RulModelSpec.from_document(model_document)
    except ValueError as exc:
        raise EvaluationBackendError(str(exc)) from exc
    if (
        experiment.base_model_digest != "architecture:rul-transformer-v1"
        or experiment.training_config.get("base_model_revision") != spec.architecture_revision
    ):
        raise EvaluationBackendError("rul_experiment_model_binding_changed")
    means, scales = _rul_normalizer(normalizer)
    state_path = directory / "model.safetensors"
    if not state_path.is_file():
        raise EvaluationBackendError("rul_model_weights_are_missing")
    device = torch.device("cuda:0")
    model = build_rul_transformer(torch, spec)
    model.load_state_dict(load_file(str(state_path), device="cpu"), strict=True)
    model.to(device).eval()
    dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float16
    observations: list[ModelObservation] = []
    try:
        with torch.no_grad():
            for case in cases:
                contract = _rul_contract(case)
                values, padding = _rul_tensor(
                    contract,
                    spec=spec,
                    means=means,
                    scales=scales,
                    torch=torch,
                    device=device,
                )
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=dtype):
                    predicted_log = model(values, padding)
                predicted = torch.expm1(predicted_log.float()).clamp_min(0)[0]
                torch.cuda.synchronize()
                latency_ms = max((time.perf_counter() - started) * 1000, 1e-9)
                quantiles = tuple(float(item) for item in predicted.cpu())
                observations.append(
                    _rul_observation(
                        case.case_id,
                        quantiles=(quantiles[0], quantiles[1], quantiles[2]),
                        latency_ms=latency_ms,
                        cost_usd=max(
                            latency_ms / 3_600_000 * config.gpu_hourly_cost_usd,
                            1e-12,
                        ),
                        backend="native-pytorch-rul-transformer-encoder",
                    )
                )
    finally:
        del model
        torch.cuda.empty_cache()
    return tuple(observations)


def _empirical_rul_quantiles(
    experiment: TrainingExperimentRecord,
) -> tuple[float, float, float]:
    raw = experiment.training_config.get("empirical_quantiles_minutes")
    if not isinstance(raw, list) or len(raw) != 3:
        raise EvaluationBackendError("rul_empirical_baseline_quantiles_are_invalid")
    try:
        values = tuple(float(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise EvaluationBackendError("rul_empirical_baseline_quantiles_are_invalid") from exc
    if (
        not all(math.isfinite(item) and item > 0 for item in values)
        or not values[0] <= values[1] <= values[2]
    ):
        raise EvaluationBackendError("rul_empirical_baseline_quantiles_are_invalid")
    return values[0], values[1], values[2]


def _rul_baseline_observation(
    case: EvaluationCase, quantiles: tuple[float, float, float]
) -> ModelObservation:
    _rul_contract(case)
    started = time.perf_counter()
    latency_ms = max((time.perf_counter() - started) * 1000, 1e-9)
    return _rul_observation(
        case.case_id,
        quantiles=quantiles,
        latency_ms=latency_ms,
        cost_usd=1e-12,
        backend="registered-empirical-lead-time-baseline-v1",
    )


def _rul_observation(
    case_id: str,
    *,
    quantiles: tuple[float, float, float],
    latency_ms: float,
    cost_usd: float,
    backend: str,
) -> ModelObservation:
    valid = (
        all(math.isfinite(item) and item >= 0 for item in quantiles)
        and quantiles[0] <= quantiles[1] <= quantiles[2]
    )
    output = json.dumps(
        {
            "p10_minutes": quantiles[0],
            "p50_minutes": quantiles[1],
            "p90_minutes": quantiles[2],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return ModelObservation(
        case_id=case_id,
        output_text=output,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        input_tokens=0,
        output_tokens=0,
        capabilities=frozenset(
            {"data_governance", "rul_artifact_integrity", "rul_target_coverage"}
        ),
        structured_output_valid=valid,
        runtime_evidence={
            "backend": backend,
            "p10_minutes": quantiles[0],
            "p50_minutes": quantiles[1],
            "p90_minutes": quantiles[2],
        },
    )


def _transformer_observations(
    experiment: TrainingExperimentRecord,
    cases: tuple[EvaluationCase, ...],
    config: EvaluationRuntimeConfig,
    directory: Path,
    *,
    anomaly_threshold: float,
) -> tuple[ModelObservation, ...]:
    try:
        torch = importlib.import_module("torch")
        load_file = importlib.import_module("safetensors.torch").load_file
    except ImportError as exc:  # pragma: no cover - GPU image responsibility
        raise EvaluationBackendError(
            "timeseries_evaluation_dependencies_are_not_installed"
        ) from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise EvaluationBackendError("exactly_one_cuda_gpu_is_required")
    model_document = _json_file(directory / "model-config.json")
    normalizer = _json_file(directory / "normalizer.json")
    if (
        model_document.get("architecture") != "IndustrialTelemetryTransformer"
        or model_document.get("objective") != "masked_reconstruction"
    ):
        raise EvaluationBackendError("timeseries_model_contract_is_invalid")
    try:
        spec = TimeseriesModelSpec.from_document(model_document)
    except ValueError as exc:
        raise EvaluationBackendError(str(exc)) from exc
    if (
        experiment.base_model_digest != "architecture:timeseries-transformer-v1"
        or experiment.training_config.get("base_model_revision") != spec.architecture_revision
    ):
        raise EvaluationBackendError("timeseries_experiment_model_binding_changed")
    means, scales = _normalizer(normalizer)
    state_path = directory / "model.safetensors"
    if not state_path.is_file():
        raise EvaluationBackendError("timeseries_model_weights_are_missing")
    device = torch.device("cuda:0")
    model = build_timeseries_transformer(torch, spec)
    state = load_file(str(state_path), device="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float16
    observations: list[ModelObservation] = []
    try:
        with torch.no_grad():
            for case in cases:
                contract = _telemetry_contract(case)
                values, observed, padding = _tensor(
                    contract,
                    spec=spec,
                    means=means,
                    scales=scales,
                    torch=torch,
                    device=device,
                )
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=dtype):
                    prediction = model(values.masked_fill(~observed, 0.0), padding)
                squared = (prediction.float() - values).square() * observed
                score = float((squared.sum() / observed.sum().clamp_min(1)).detach().cpu())
                torch.cuda.synchronize()
                latency_ms = max((time.perf_counter() - started) * 1000, 1e-9)
                if not math.isfinite(score):
                    raise EvaluationBackendError("timeseries_anomaly_score_is_non_finite")
                predicted_label = int(score >= anomaly_threshold)
                observations.append(
                    _observation(
                        case.case_id,
                        predicted_label=predicted_label,
                        score=score,
                        threshold=anomaly_threshold,
                        latency_ms=latency_ms,
                        cost_usd=max(
                            latency_ms / 3_600_000 * config.gpu_hourly_cost_usd,
                            1e-12,
                        ),
                        backend="native-pytorch-transformer-reconstruction",
                    )
                )
    finally:
        del model
        torch.cuda.empty_cache()
    return tuple(observations)


def _rule_observation(case: EvaluationCase, threshold: float) -> ModelObservation:
    contract = _telemetry_contract(case)
    started = time.perf_counter()
    score = _rule_score(contract)
    latency_ms = max((time.perf_counter() - started) * 1000, 1e-9)
    return _observation(
        case.case_id,
        predicted_label=int(score >= threshold),
        score=score,
        threshold=threshold,
        latency_ms=latency_ms,
        cost_usd=1e-12,
        backend="registered-industrial-rule-baseline-v1",
    )


def _observation(
    case_id: str,
    *,
    predicted_label: int,
    score: float,
    threshold: float,
    latency_ms: float,
    cost_usd: float,
    backend: str,
) -> ModelObservation:
    output = json.dumps(
        {"label": predicted_label, "score": score},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return ModelObservation(
        case_id=case_id,
        output_text=output,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        input_tokens=0,
        output_tokens=0,
        capabilities=frozenset(
            {"data_governance", "telemetry_artifact_integrity", "label_coverage"}
        ),
        structured_output_valid=True,
        runtime_evidence={
            "backend": backend,
            "predicted_label": predicted_label,
            "anomaly_score": score,
            "threshold": threshold,
        },
    )


def _rule_score(contract: TelemetryEvaluationContract) -> float:
    values_by_signal: list[list[float]] = [[] for _ in SIGNAL_ORDER]
    for values, present in zip(contract.sequence, contract.mask, strict=True):
        for index, (value, observed) in enumerate(zip(values, present, strict=True)):
            if observed:
                values_by_signal[index].append(value)
    scores: list[float] = []
    vibration = values_by_signal[0]
    if vibration:
        scores.extend(
            (
                _normalized(sum(vibration) / len(vibration), 2.8, 7.1),
                _normalized(max(vibration), 4.5, 11.2),
                _normalized(
                    (vibration[-1] - vibration[0]) / contract.duration_minutes,
                    0.02,
                    0.3,
                ),
            )
        )
    temperature = values_by_signal[1]
    if temperature:
        scores.extend(
            (
                _normalized(sum(temperature) / len(temperature), 65.0, 95.0),
                _normalized(max(temperature), 75.0, 110.0),
                _normalized(
                    (temperature[-1] - temperature[0]) / contract.duration_minutes,
                    0.05,
                    1.0,
                ),
            )
        )
    rpm = values_by_signal[3]
    if rpm and sum(rpm) / len(rpm) > 0:
        average = sum(rpm) / len(rpm)
        scores.append(_normalized((max(rpm) - min(rpm)) / average, 0.08, 0.35))
    return max(scores, default=0.0)


def _normalized(value: float, baseline: float, alarm: float) -> float:
    if value <= baseline:
        return 0.0
    return min(1.0, (value - baseline) / (alarm - baseline))


def _tensor(
    contract: TelemetryEvaluationContract,
    *,
    spec: TimeseriesModelSpec,
    means: tuple[float, ...],
    scales: tuple[float, ...],
    torch: Any,
    device: Any,
) -> tuple[Any, Any, Any]:
    length = min(len(contract.sequence), spec.max_sequence_length)
    values = torch.zeros((1, length, len(SIGNAL_ORDER)), dtype=torch.float32)
    observed = torch.zeros((1, length, len(SIGNAL_ORDER)), dtype=torch.bool)
    for sequence_index in range(length):
        for signal_index in range(len(SIGNAL_ORDER)):
            present = contract.mask[sequence_index][signal_index]
            observed[0, sequence_index, signal_index] = present
            if present:
                values[0, sequence_index, signal_index] = (
                    contract.sequence[sequence_index][signal_index] - means[signal_index]
                ) / scales[signal_index]
    padding = torch.zeros((1, length), dtype=torch.bool)
    return values.to(device), observed.to(device), padding.to(device)


def _rul_tensor(
    contract: RulEvaluationContract,
    *,
    spec: RulModelSpec,
    means: tuple[float, ...],
    scales: tuple[float, ...],
    torch: Any,
    device: Any,
) -> tuple[Any, Any]:
    length = min(len(contract.sequence), spec.max_sequence_length)
    values = torch.zeros((1, length, len(RUL_SIGNAL_ORDER)), dtype=torch.float32)
    padding = torch.zeros((1, length), dtype=torch.bool)
    sequence = contract.sequence[-length:]
    mask = contract.mask[-length:]
    for sequence_index, (signals, present) in enumerate(zip(sequence, mask, strict=True)):
        for signal_index, (value, available) in enumerate(zip(signals, present, strict=True)):
            if available:
                values[0, sequence_index, signal_index] = (
                    value - means[signal_index]
                ) / scales[signal_index]
    return values.to(device), padding.to(device)


def _rul_normalizer(value: dict[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    try:
        signal_order = tuple(str(item) for item in value["signal_order"])
        means = tuple(float(item) for item in value["means"])
        scales = tuple(float(item) for item in value["scales"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationBackendError("rul_normalizer_is_invalid") from exc
    if (
        signal_order != RUL_SIGNAL_ORDER
        or len(means) != len(RUL_SIGNAL_ORDER)
        or len(scales) != len(RUL_SIGNAL_ORDER)
        or not all(math.isfinite(item) for item in means + scales)
        or not all(item > 0 for item in scales)
    ):
        raise EvaluationBackendError("rul_normalizer_is_invalid")
    return means, scales


def _normalizer(value: dict[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    try:
        signal_order = tuple(str(item) for item in value["signal_order"])
        means = tuple(float(item) for item in value["means"])
        scales = tuple(float(item) for item in value["scales"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationBackendError("timeseries_normalizer_is_invalid") from exc
    if (
        signal_order != SIGNAL_ORDER
        or len(means) != len(SIGNAL_ORDER)
        or len(scales) != len(SIGNAL_ORDER)
        or not all(math.isfinite(item) for item in means + scales)
        or not all(item > 0 for item in scales)
    ):
        raise EvaluationBackendError("timeseries_normalizer_is_invalid")
    return means, scales


def _json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationBackendError("timeseries_artifact_json_is_invalid") from exc
    if not isinstance(value, dict):
        raise EvaluationBackendError("timeseries_artifact_json_is_invalid")
    return value


def _telemetry_contract(case: EvaluationCase) -> TelemetryEvaluationContract:
    if case.telemetry is None:
        raise EvaluationBackendError("telemetry_evaluation_contract_is_missing")
    return case.telemetry


def _rul_contract(case: EvaluationCase) -> RulEvaluationContract:
    if case.rul is None:
        raise EvaluationBackendError("rul_evaluation_contract_is_missing")
    return case.rul
