"""GPU inference server for a release-bound industrial RUL Transformer bundle."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from industrial_ops_agent.predictive_maintenance.rul_dataset_contract import (
    CONTRACT_VERSION,
    SIGNAL_ORDER,
)
from industrial_ops_agent.predictive_maintenance.rul_model import (
    RulModelSpec,
    build_rul_transformer,
)
from industrial_ops_agent.runtime_metrics import RulRuntimeMetrics


class RulInferenceRequest(BaseModel):
    schema_version: str = Field(min_length=1, max_length=64)
    feature_snapshot_id: str = Field(min_length=1, max_length=128)
    sequence: list[list[float]] = Field(min_length=3, max_length=4096)
    mask: list[list[int]] = Field(min_length=3, max_length=4096)
    expected_release_id: str = Field(min_length=1, max_length=128)
    expected_manifest_hash: str = Field(min_length=1, max_length=128)
    expected_component_model_id: str = Field(min_length=1, max_length=128)
    expected_artifact_content_hash: str = Field(min_length=1, max_length=128)


class RulInferenceResponse(BaseModel):
    feature_snapshot_id: str
    release_id: str
    manifest_hash: str
    component_model_id: str
    artifact_content_hash: str
    p10_minutes: float
    p50_minutes: float
    p90_minutes: float
    explanation_codes: list[str]


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    release_id: str
    manifest_hash: str
    component_model_id: str
    artifact_content_hash: str


class RulRuntimeModel:
    def __init__(
        self,
        model_directory: Path,
        *,
        identity: RuntimeIdentity,
        precision: str,
        device: str = "cuda",
    ) -> None:
        if device not in {"cuda", "cpu"}:
            raise ValueError("rul_runtime_device_is_invalid")
        if precision not in {"bfloat16", "float16", "float32"}:
            raise ValueError("rul_runtime_precision_is_invalid")
        if device == "cuda" and precision == "float32":
            raise ValueError("rul_runtime_cuda_precision_is_invalid")
        if device == "cpu" and precision != "float32":
            raise ValueError("rul_runtime_cpu_requires_float32")
        try:
            self._torch = importlib.import_module("torch")
            load_file = importlib.import_module("safetensors.torch").load_file
        except ImportError as exc:  # pragma: no cover - runtime image responsibility
            raise RuntimeError("rul_runtime_dependencies_are_not_installed") from exc
        if device == "cuda" and (
            not self._torch.cuda.is_available() or self._torch.cuda.device_count() != 1
        ):
            raise RuntimeError("rul_runtime_requires_exactly_one_cuda_gpu")
        model_document = _json_file(model_directory / "model-config.json")
        if (
            model_document.get("model_type") != "IndustrialRulTransformer"
            or model_document.get("objective") != "quantile_regression"
            or model_document.get("target_transform") != "log1p_minutes"
        ):
            raise RuntimeError("rul_runtime_model_contract_is_invalid")
        try:
            self._spec = RulModelSpec.from_document(model_document)
        except ValueError as exc:
            raise RuntimeError("rul_runtime_model_contract_is_invalid") from exc
        self._means, self._scales = _normalizer(
            _json_file(model_directory / "normalizer.json")
        )
        state_path = model_directory / "model.safetensors"
        if not state_path.is_file():
            raise RuntimeError("rul_runtime_weights_are_missing")
        self._device_type = device
        self._device = self._torch.device("cuda:0" if device == "cuda" else "cpu")
        self._model = build_rul_transformer(self._torch, self._spec)
        self._model.load_state_dict(load_file(str(state_path), device="cpu"), strict=True)
        self._model.to(self._device).eval()
        self._dtype = (
            self._torch.bfloat16
            if precision == "bfloat16"
            else self._torch.float16
            if precision == "float16"
            else self._torch.float32
        )
        self._identity = identity
        self._lock = Lock()

    @property
    def release_id(self) -> str:
        return self._identity.release_id

    def predict(self, request: RulInferenceRequest) -> RulInferenceResponse:
        _require_identity(request, self._identity)
        values, padding = _validated_tensor(
            request.sequence,
            request.mask,
            spec=self._spec,
            means=self._means,
            scales=self._scales,
            torch=self._torch,
            device=self._device,
        )
        with self._lock, self._torch.no_grad():
            precision_context = (
                self._torch.autocast(device_type="cuda", dtype=self._dtype)
                if self._device_type == "cuda"
                else nullcontext()
            )
            with precision_context:
                predicted_log = self._model(values, padding)
            predicted = self._torch.expm1(predicted_log.float()).clamp_min(0)[0]
            quantiles = tuple(float(item) for item in predicted.cpu())
        if (
            len(quantiles) != 3
            or not all(math.isfinite(item) and item >= 0 for item in quantiles)
            or not quantiles[0] <= quantiles[1] <= quantiles[2]
        ):
            raise RuntimeError("rul_runtime_prediction_is_invalid")
        return RulInferenceResponse(
            feature_snapshot_id=request.feature_snapshot_id,
            release_id=self._identity.release_id,
            manifest_hash=self._identity.manifest_hash,
            component_model_id=self._identity.component_model_id,
            artifact_content_hash=self._identity.artifact_content_hash,
            p10_minutes=quantiles[0],
            p50_minutes=quantiles[1],
            p90_minutes=quantiles[2],
            explanation_codes=["RUL_TRANSFORMER_QUANTILE_FORECAST"],
        )


class EmpiricalRulRuntimeModel:
    """Release-bound rollback target used by the local KServe promotion lab."""

    def __init__(
        self,
        *,
        identity: RuntimeIdentity,
        quantiles: tuple[float, float, float],
    ) -> None:
        if (
            not all(math.isfinite(item) and item > 0 for item in quantiles)
            or not quantiles[0] <= quantiles[1] <= quantiles[2]
        ):
            raise ValueError("rul_runtime_empirical_quantiles_are_invalid")
        self._identity = identity
        self._quantiles = quantiles

    @property
    def release_id(self) -> str:
        return self._identity.release_id

    def predict(self, request: RulInferenceRequest) -> RulInferenceResponse:
        _require_identity(request, self._identity)
        return RulInferenceResponse(
            feature_snapshot_id=request.feature_snapshot_id,
            release_id=self._identity.release_id,
            manifest_hash=self._identity.manifest_hash,
            component_model_id=self._identity.component_model_id,
            artifact_content_hash=self._identity.artifact_content_hash,
            p10_minutes=self._quantiles[0],
            p50_minutes=self._quantiles[1],
            p90_minutes=self._quantiles[2],
            explanation_codes=["RUL_EMPIRICAL_BASELINE"],
        )


class RulServingModel(Protocol):
    @property
    def release_id(self) -> str: ...

    def predict(self, request: RulInferenceRequest) -> RulInferenceResponse: ...


def create_rul_runtime_app(
    model: RulServingModel,
    metrics: RulRuntimeMetrics | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    app = FastAPI(title="Industrial RUL Transformer runtime", lifespan=lifespan)
    runtime_metrics = metrics or RulRuntimeMetrics()

    @app.get("/health/ready")
    async def ready() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        return Response(
            content=generate_latest(runtime_metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.post("/v1/rul/forecast", response_model=RulInferenceResponse)
    def forecast(request: RulInferenceRequest) -> RulInferenceResponse:
        started = perf_counter()
        try:
            result = model.predict(request)
        except ValueError as exc:
            runtime_metrics.observe(
                release_id=model.release_id,
                status="rejected",
                seconds=perf_counter() - started,
            )
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            runtime_metrics.observe(
                release_id=model.release_id,
                status="error",
                seconds=perf_counter() - started,
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        runtime_metrics.observe(
            release_id=model.release_id,
            status="success",
            seconds=perf_counter() - started,
        )
        return result

    return app


def _require_identity(request: RulInferenceRequest, identity: RuntimeIdentity) -> None:
    if (
        request.expected_release_id != identity.release_id
        or request.expected_manifest_hash != identity.manifest_hash
        or request.expected_component_model_id != identity.component_model_id
        or request.expected_artifact_content_hash != identity.artifact_content_hash
    ):
        raise ValueError("rul_runtime_release_binding_mismatch")
    if request.schema_version != CONTRACT_VERSION:
        raise ValueError("rul_runtime_sequence_contract_mismatch")


def _validated_tensor(
    sequence: list[list[float]],
    mask: list[list[int]],
    *,
    spec: RulModelSpec,
    means: tuple[float, ...],
    scales: tuple[float, ...],
    torch: Any,
    device: Any,
) -> tuple[Any, Any]:
    if len(sequence) != len(mask) or len(sequence) > spec.max_sequence_length:
        raise ValueError("rul_runtime_sequence_length_is_invalid")
    width = len(SIGNAL_ORDER)
    if any(len(row) != width for row in sequence) or any(len(row) != width for row in mask):
        raise ValueError("rul_runtime_signal_width_is_invalid")
    if any(value not in {0, 1} for row in mask for value in row):
        raise ValueError("rul_runtime_mask_is_invalid")
    if any(not math.isfinite(value) for row in sequence for value in row):
        raise ValueError("rul_runtime_sequence_contains_non_finite_value")
    length = min(len(sequence), spec.max_sequence_length)
    values = torch.zeros((1, length, width), dtype=torch.float32, device=device)
    padding = torch.zeros((1, length), dtype=torch.bool, device=device)
    for sequence_index, (signals, present) in enumerate(
        zip(sequence[-length:], mask[-length:], strict=True)
    ):
        for signal_index, (value, available) in enumerate(zip(signals, present, strict=True)):
            if available:
                values[0, sequence_index, signal_index] = (
                    value - means[signal_index]
                ) / scales[signal_index]
    return values, padding


def _normalizer(document: dict[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    try:
        order = tuple(str(item) for item in document["signal_order"])
        means = tuple(float(item) for item in document["means"])
        scales = tuple(float(item) for item in document["scales"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("rul_runtime_normalizer_is_invalid") from exc
    if (
        order != SIGNAL_ORDER
        or len(means) != len(order)
        or len(scales) != len(order)
        or any(not math.isfinite(item) for item in means + scales)
        or any(item <= 0 for item in scales)
    ):
        raise RuntimeError("rul_runtime_normalizer_is_invalid")
    return means, scales


def _json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("rul_runtime_artifact_json_is_invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeError("rul_runtime_artifact_json_is_invalid")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a governed RUL Transformer")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--manifest-hash", required=True)
    parser.add_argument("--component-model-id", required=True)
    parser.add_argument("--artifact-content-hash", required=True)
    parser.add_argument(
        "--runtime-mode",
        choices=("transformer", "empirical"),
        default="transformer",
    )
    parser.add_argument(
        "--precision",
        choices=("bfloat16", "float16", "float32"),
        required=True,
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--empirical-p10", type=float)
    parser.add_argument("--empirical-p50", type=float)
    parser.add_argument("--empirical-p90", type=float)
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 - container listener
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    identity = RuntimeIdentity(
        release_id=args.release_id,
        manifest_hash=args.manifest_hash,
        component_model_id=args.component_model_id,
        artifact_content_hash=args.artifact_content_hash,
    )
    if args.runtime_mode == "empirical":
        raw_quantiles = (args.empirical_p10, args.empirical_p50, args.empirical_p90)
        if any(item is None for item in raw_quantiles):
            raise ValueError("rul_runtime_empirical_quantiles_are_required")
        model: RulServingModel = EmpiricalRulRuntimeModel(
            identity=identity,
            quantiles=(
                float(raw_quantiles[0]),
                float(raw_quantiles[1]),
                float(raw_quantiles[2]),
            ),
        )
    else:
        model = RulRuntimeModel(
            args.model_dir,
            identity=identity,
            precision=args.precision,
            device=args.device,
        )
    uvicorn = importlib.import_module("uvicorn")
    uvicorn.run(create_rul_runtime_app(model), host=args.host, port=args.port)
    return 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
