"""GPU inference server for a governed industrial telemetry Transformer bundle."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from industrial_ops_agent.predictive_maintenance.dataset_contract import (
    CONTRACT_VERSION,
    SIGNAL_ORDER,
)
from industrial_ops_agent.predictive_maintenance.timeseries_model import (
    TimeseriesModelSpec,
    build_timeseries_transformer,
)
from industrial_ops_agent.runtime_metrics import TimeseriesRuntimeMetrics


class TimeseriesInferenceRequest(BaseModel):
    schema_version: str = Field(min_length=1, max_length=64)
    feature_snapshot_id: str = Field(min_length=1, max_length=128)
    sequence: list[list[float]] = Field(min_length=3, max_length=4096)
    mask: list[list[int]] = Field(min_length=3, max_length=4096)
    expected_release_id: str = Field(min_length=1, max_length=128)
    expected_manifest_hash: str = Field(min_length=1, max_length=128)
    expected_component_model_id: str = Field(min_length=1, max_length=128)
    expected_artifact_content_hash: str = Field(min_length=1, max_length=128)


class TimeseriesInferenceResponse(BaseModel):
    feature_snapshot_id: str
    release_id: str
    manifest_hash: str
    component_model_id: str
    artifact_content_hash: str
    anomaly_score: float
    threshold: float
    anomalous: bool
    explanation_codes: list[str]


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    release_id: str
    manifest_hash: str
    component_model_id: str
    artifact_content_hash: str


class TimeseriesRuntimeModel:
    def __init__(
        self,
        model_directory: Path,
        *,
        identity: RuntimeIdentity,
        precision: str,
        anomaly_threshold: float,
    ) -> None:
        if precision not in {"bfloat16", "float16"}:
            raise ValueError("timeseries_runtime_precision_is_invalid")
        if not math.isfinite(anomaly_threshold) or anomaly_threshold <= 0:
            raise ValueError("timeseries_runtime_threshold_is_invalid")
        try:
            self._torch = importlib.import_module("torch")
            load_file = importlib.import_module("safetensors.torch").load_file
        except ImportError as exc:  # pragma: no cover - runtime image responsibility
            raise RuntimeError("timeseries_runtime_dependencies_are_not_installed") from exc
        if not self._torch.cuda.is_available() or self._torch.cuda.device_count() != 1:
            raise RuntimeError("timeseries_runtime_requires_exactly_one_cuda_gpu")
        model_document = _json_file(model_directory / "model-config.json")
        if (
            model_document.get("architecture") != "IndustrialTelemetryTransformer"
            or model_document.get("objective") != "masked_reconstruction"
        ):
            raise RuntimeError("timeseries_runtime_model_contract_is_invalid")
        self._spec = TimeseriesModelSpec.from_document(model_document)
        normalizer = _json_file(model_directory / "normalizer.json")
        self._means, self._scales = _normalizer(normalizer)
        state_path = model_directory / "model.safetensors"
        if not state_path.is_file():
            raise RuntimeError("timeseries_runtime_weights_are_missing")
        self._device = self._torch.device("cuda:0")
        self._model = build_timeseries_transformer(self._torch, self._spec)
        self._model.load_state_dict(load_file(str(state_path), device="cpu"), strict=True)
        self._model.to(self._device).eval()
        self._dtype = self._torch.bfloat16 if precision == "bfloat16" else self._torch.float16
        self._identity = identity
        self._threshold = anomaly_threshold
        self._lock = Lock()

    @property
    def release_id(self) -> str:
        return self._identity.release_id

    def predict(self, request: TimeseriesInferenceRequest) -> TimeseriesInferenceResponse:
        _require_identity(request, self._identity)
        values, observed, padding = _validated_tensor(
            request.sequence,
            request.mask,
            spec=self._spec,
            means=self._means,
            scales=self._scales,
            torch=self._torch,
            device=self._device,
        )
        with self._lock, self._torch.no_grad():
            with self._torch.autocast(device_type="cuda", dtype=self._dtype):
                prediction = self._model(values.masked_fill(~observed, 0.0), padding)
            squared = (prediction.float() - values).square() * observed
            score = float((squared.sum() / observed.sum().clamp_min(1)).detach().cpu())
        if not math.isfinite(score):
            raise RuntimeError("timeseries_runtime_score_is_non_finite")
        anomalous = score >= self._threshold
        return TimeseriesInferenceResponse(
            feature_snapshot_id=request.feature_snapshot_id,
            release_id=self._identity.release_id,
            manifest_hash=self._identity.manifest_hash,
            component_model_id=self._identity.component_model_id,
            artifact_content_hash=self._identity.artifact_content_hash,
            anomaly_score=score,
            threshold=self._threshold,
            anomalous=anomalous,
            explanation_codes=[
                "TRANSFORMER_RECONSTRUCTION_ANOMALY" if anomalous else "WITHIN_TRANSFORMER_BASELINE"
            ],
        )


def create_timeseries_runtime_app(
    model: TimeseriesRuntimeModel,
    metrics: TimeseriesRuntimeMetrics | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    app = FastAPI(title="Industrial telemetry Transformer runtime", lifespan=lifespan)
    runtime_metrics = metrics or TimeseriesRuntimeMetrics()

    @app.get("/health/ready")
    async def ready() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        return Response(
            content=generate_latest(runtime_metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.post(
        "/v1/timeseries/anomaly",
        response_model=TimeseriesInferenceResponse,
    )
    def anomaly(request: TimeseriesInferenceRequest) -> TimeseriesInferenceResponse:
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


def _require_identity(
    request: TimeseriesInferenceRequest,
    identity: RuntimeIdentity,
) -> None:
    if (
        request.expected_release_id != identity.release_id
        or request.expected_manifest_hash != identity.manifest_hash
        or request.expected_component_model_id != identity.component_model_id
        or request.expected_artifact_content_hash != identity.artifact_content_hash
    ):
        raise ValueError("timeseries_runtime_release_binding_mismatch")
    if request.schema_version != CONTRACT_VERSION:
        raise ValueError("timeseries_runtime_sequence_contract_mismatch")


def _validated_tensor(
    sequence: list[list[float]],
    mask: list[list[int]],
    *,
    spec: TimeseriesModelSpec,
    means: tuple[float, ...],
    scales: tuple[float, ...],
    torch: Any,
    device: Any,
) -> tuple[Any, Any, Any]:
    if len(sequence) != len(mask) or len(sequence) > spec.max_sequence_length:
        raise ValueError("timeseries_runtime_sequence_length_is_invalid")
    width = len(SIGNAL_ORDER)
    if any(len(row) != width for row in sequence) or any(len(row) != width for row in mask):
        raise ValueError("timeseries_runtime_signal_width_is_invalid")
    if any(value not in {0, 1} for row in mask for value in row):
        raise ValueError("timeseries_runtime_mask_is_invalid")
    if any(not math.isfinite(value) for row in sequence for value in row):
        raise ValueError("timeseries_runtime_sequence_contains_non_finite_value")
    values = torch.tensor(sequence, dtype=torch.float32, device=device)
    observed = torch.tensor(mask, dtype=torch.bool, device=device)
    mean_tensor = torch.tensor(means, dtype=torch.float32, device=device)
    scale_tensor = torch.tensor(scales, dtype=torch.float32, device=device)
    values = ((values - mean_tensor) / scale_tensor).unsqueeze(0)
    observed = observed.unsqueeze(0)
    padding = ~observed.any(dim=2)
    return values, observed, padding


def _normalizer(document: dict[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    try:
        order = tuple(str(item) for item in document["signal_order"])
        means = tuple(float(item) for item in document["means"])
        scales = tuple(float(item) for item in document["scales"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("timeseries_runtime_normalizer_is_invalid") from exc
    if (
        order != SIGNAL_ORDER
        or len(means) != len(order)
        or len(scales) != len(order)
        or any(not math.isfinite(item) for item in means + scales)
        or any(item <= 0 for item in scales)
    ):
        raise RuntimeError("timeseries_runtime_normalizer_is_invalid")
    return means, scales


def _json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("timeseries_runtime_artifact_json_is_invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeError("timeseries_runtime_artifact_json_is_invalid")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a governed telemetry Transformer")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--manifest-hash", required=True)
    parser.add_argument("--component-model-id", required=True)
    parser.add_argument("--artifact-content-hash", required=True)
    parser.add_argument("--precision", choices=("bfloat16", "float16"), required=True)
    parser.add_argument("--anomaly-threshold", type=float, required=True)
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 - container listener
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    model = TimeseriesRuntimeModel(
        args.model_dir,
        identity=RuntimeIdentity(
            release_id=args.release_id,
            manifest_hash=args.manifest_hash,
            component_model_id=args.component_model_id,
            artifact_content_hash=args.artifact_content_hash,
        ),
        precision=args.precision,
        anomaly_threshold=args.anomaly_threshold,
    )
    uvicorn = importlib.import_module("uvicorn")
    uvicorn.run(create_timeseries_runtime_app(model), host=args.host, port=args.port)
    return 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
