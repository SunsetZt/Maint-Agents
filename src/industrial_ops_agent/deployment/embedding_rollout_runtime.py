"""Dual-variant actual-GPU runtime for Embedding KServe rollout acceptance."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import threading
import time
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.simulation.embedding_calibrated_value_lab import (
    verify_calibrated_embedding_value,
)
from industrial_ops_agent.simulation.embedding_recovery_value_lab import (
    BASE_MODEL_RELATIVE,
    MODEL_ID,
    MODEL_REVISION,
)


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EmbeddingRolloutInput(_ClosedModel):
    item_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=8_192)


class EmbeddingRolloutRequest(_ClosedModel):
    request_id: str = Field(min_length=1, max_length=128)
    case_id: str = Field(min_length=1, max_length=256)
    index_release_id: str = Field(min_length=1, max_length=128)
    inputs: tuple[EmbeddingRolloutInput, ...] = Field(min_length=1, max_length=32)
    variant: Literal["stable", "candidate"]


class EmbeddingRolloutItem(_ClosedModel):
    item_id: str
    vector: tuple[float, ...] = Field(min_length=8, max_length=8_192)
    vector_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EmbeddingRolloutResponse(_ClosedModel):
    schema_version: Literal["enterprise-embedding-rollout-response/v1"] = (
        "enterprise-embedding-rollout-response/v1"
    )
    request_id: str
    case_id: str
    index_release_id: str
    variant: Literal["stable", "candidate"]
    model_id: str
    model_revision: str
    candidate_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    actual_model_inference: Literal[True] = True
    cache_hit: bool
    dimension: int = Field(ge=8, le=8_192)
    items: tuple[EmbeddingRolloutItem, ...]
    latency_ms: float = Field(gt=0.0)
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EnterpriseEmbeddingRolloutRuntime:
    """Keep the small stable/candidate encoders resident and serialize GPU calls."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        stable_directory: Path,
        candidate_directory: Path,
        candidate_bundle_sha256: str,
        precision: Literal["bfloat16", "float16"] = "bfloat16",
        batch_size: int = 16,
    ) -> None:
        if not model_id.strip() or not model_revision.strip():
            raise ValueError("embedding_rollout_model_identity_is_invalid")
        if len(candidate_bundle_sha256) != 64:
            raise ValueError("embedding_rollout_candidate_bundle_hash_is_invalid")
        if not 1 <= batch_size <= 64:
            raise ValueError("embedding_rollout_batch_size_is_invalid")
        self.model_id = model_id
        self.model_revision = model_revision
        self.stable_directory = stable_directory.resolve(strict=True)
        self.candidate_directory = candidate_directory.resolve(strict=True)
        self.candidate_bundle_sha256 = candidate_bundle_sha256
        self.precision = precision
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._cache: dict[str, EmbeddingRolloutResponse] = {}
        self._request_counts = {"stable": 0, "candidate": 0}
        self._inference_counts = {"stable": 0, "candidate": 0}
        self._cache_hits = {"stable": 0, "candidate": 0}
        self._error_counts = {"stable": 0, "candidate": 0}
        self._started_at = time.time()
        self._torch: Any = None
        self._models: dict[str, Any] = {}
        self._dimension = 0
        self._gpu_name = ""
        self._gpu_total_memory_bytes = 0
        self._torch_version = ""
        self._transformers_version = ""
        self._sentence_transformers_version = ""

    def load(self) -> None:
        try:
            torch = importlib.import_module("torch")
            sentence_transformer = importlib.import_module("sentence_transformers")
        except ImportError as exc:  # pragma: no cover - runtime image responsibility
            raise RuntimeError("embedding_rollout_dependencies_are_not_installed") from exc
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly_one_cuda_gpu_is_required")
        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        dtype = torch.bfloat16 if self.precision == "bfloat16" else torch.float16
        model_class = sentence_transformer.SentenceTransformer
        stable = model_class(
            str(self.stable_directory),
            trust_remote_code=False,
            device="cuda:0",
            model_kwargs={"torch_dtype": dtype},
        )
        candidate = model_class(
            str(self.candidate_directory),
            trust_remote_code=False,
            device="cuda:0",
            model_kwargs={"torch_dtype": dtype},
        )
        stable_dimension = stable.get_sentence_embedding_dimension()
        candidate_dimension = candidate.get_sentence_embedding_dimension()
        if (
            isinstance(stable_dimension, bool)
            or not isinstance(stable_dimension, int)
            or stable_dimension != candidate_dimension
            or not 8 <= stable_dimension <= 8_192
        ):
            raise RuntimeError("embedding_rollout_dimension_contract_changed")
        properties = torch.cuda.get_device_properties(0)
        self._torch = torch
        self._models = {"stable": stable, "candidate": candidate}
        self._dimension = stable_dimension
        self._gpu_name = torch.cuda.get_device_name(0)
        self._gpu_total_memory_bytes = int(properties.total_memory)
        self._torch_version = version("torch")
        self._transformers_version = version("transformers")
        self._sentence_transformers_version = version("sentence-transformers")

    def infer(self, request: EmbeddingRolloutRequest) -> EmbeddingRolloutResponse:
        if not self._models:
            raise RuntimeError("embedding_rollout_runtime_is_not_loaded")
        item_ids = tuple(item.item_id for item in request.inputs)
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("embedding_rollout_input_ids_are_not_unique")
        variant = request.variant
        with self._lock:
            self._request_counts[variant] += 1
            try:
                key = _cache_key(request)
                existing = self._cache.get(key)
                if existing is not None:
                    self._cache_hits[variant] += 1
                    return _copy_cache_hit(existing, request.request_id)
                result = self._execute(request)
                self._cache[key] = result
                self._inference_counts[variant] += 1
                return result
            except Exception:
                self._error_counts[variant] += 1
                raise

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            torch = self._torch
            return {
                "schema_version": "enterprise-embedding-rollout-metrics/v1",
                "ready": bool(self._models),
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "candidate_bundle_sha256": self.candidate_bundle_sha256,
                "dimension": self._dimension,
                "gpu_name": self._gpu_name,
                "gpu_total_memory_bytes": self._gpu_total_memory_bytes,
                "gpu_memory_allocated_bytes": (
                    int(torch.cuda.memory_allocated(0)) if torch is not None else 0
                ),
                "gpu_memory_reserved_bytes": (
                    int(torch.cuda.memory_reserved(0)) if torch is not None else 0
                ),
                "gpu_peak_memory_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved(0)) if torch is not None else 0
                ),
                "request_counts": dict(self._request_counts),
                "actual_inference_counts": dict(self._inference_counts),
                "cache_hits": dict(self._cache_hits),
                "error_counts": dict(self._error_counts),
                "cache_entries": len(self._cache),
                "torch_version": self._torch_version,
                "transformers_version": self._transformers_version,
                "sentence_transformers_version": self._sentence_transformers_version,
                "uptime_seconds": max(time.time() - self._started_at, 0.0),
            }

    def _execute(self, request: EmbeddingRolloutRequest) -> EmbeddingRolloutResponse:
        torch = self._torch
        model = self._models[request.variant]
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            raw = model.encode(
                [item.text for item in request.inputs],
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        torch.cuda.synchronize()
        latency_ms = max((time.perf_counter() - started) * 1_000.0, 0.001)
        vectors = _vectors(raw)
        if len(vectors) != len(request.inputs) or any(
            len(vector) != self._dimension
            or any(not math.isfinite(component) for component in vector)
            or not 0.99 <= math.sqrt(sum(component * component for component in vector)) <= 1.01
            for vector in vectors
        ):
            raise RuntimeError("embedding_rollout_vector_contract_is_invalid")
        items = tuple(
            EmbeddingRolloutItem(
                item_id=item.item_id,
                vector=tuple(vector),
                vector_sha256=_digest(vector),
            )
            for item, vector in zip(request.inputs, vectors, strict=True)
        )
        unsigned: dict[str, Any] = {
            "schema_version": "enterprise-embedding-rollout-response/v1",
            "request_id": request.request_id,
            "case_id": request.case_id,
            "index_release_id": request.index_release_id,
            "variant": request.variant,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "candidate_bundle_sha256": self.candidate_bundle_sha256,
            "actual_model_inference": True,
            "cache_hit": False,
            "dimension": self._dimension,
            "items": tuple(item.model_dump(mode="json") for item in items),
            "latency_ms": latency_ms,
        }
        return EmbeddingRolloutResponse(**unsigned, response_sha256=_digest(unsigned))


def create_app(runtime: EnterpriseEmbeddingRolloutRuntime) -> FastAPI:
    app = FastAPI(title="Enterprise Embedding rollout GPU worker", version="1.0.0")

    @app.get("/health/ready")
    def ready() -> dict[str, object]:
        metrics = runtime.metrics()
        if metrics["ready"] is not True:
            raise HTTPException(status_code=503, detail="embedding_rollout_runtime_not_ready")
        return {
            "ready": True,
            "model_id": metrics["model_id"],
            "model_revision": metrics["model_revision"],
            "dimension": metrics["dimension"],
        }

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return runtime.metrics()

    @app.post("/v1/embedding/rollout", response_model=EmbeddingRolloutResponse)
    def embed(request: EmbeddingRolloutRequest) -> EmbeddingRolloutResponse:
        try:
            return runtime.infer(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


def _vectors(value: Any) -> list[list[float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise RuntimeError("embedding_rollout_vector_shape_is_invalid")
    vectors: list[list[float]] = []
    for item in value:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if not isinstance(item, (list, tuple)):
            raise RuntimeError("embedding_rollout_vector_shape_is_invalid")
        vectors.append([float(component) for component in item])
    return vectors


def _cache_key(request: EmbeddingRolloutRequest) -> str:
    return _digest(request.model_dump(mode="json", exclude={"request_id"}))


def _copy_cache_hit(
    response: EmbeddingRolloutResponse,
    request_id: str,
) -> EmbeddingRolloutResponse:
    unsigned = response.model_dump(
        mode="json",
        exclude={"request_id", "cache_hit", "latency_ms", "response_sha256"},
    )
    unsigned.update({"request_id": request_id, "cache_hit": True, "latency_ms": 0.001})
    return EmbeddingRolloutResponse(**unsigned, response_sha256=_digest(unsigned))


def _digest(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-embedding-rollout-runtime")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--stable-dir", type=Path)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--candidate-bundle-sha256")
    parser.add_argument("--precision", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18097)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.port <= 65_535:
        raise ValueError("port must be between 1 and 65535")
    root = args.repo_root.resolve(strict=True)
    direct = (args.stable_dir, args.candidate_dir)
    if any(value is not None for value in direct):
        if (
            any(value is None for value in direct)
            or not args.model_id
            or not args.model_revision
            or not args.candidate_bundle_sha256
        ):
            raise ValueError(
                "direct Embedding rollout runtime requires complete immutable coordinates"
            )
        stable_directory = args.stable_dir
        candidate_directory = args.candidate_dir
        model_id = args.model_id
        model_revision = args.model_revision
        candidate_bundle_sha256 = args.candidate_bundle_sha256
    else:
        report = verify_calibrated_embedding_value(root)
        if not report.candidate_accepted or not all(report.hard_gates.values()):
            raise RuntimeError("embedding_authoritative_release_source_is_not_accepted")
        stable_directory = root / BASE_MODEL_RELATIVE
        candidate_directory = root / Path(report.candidate.path)
        model_id = MODEL_ID
        model_revision = MODEL_REVISION
        candidate_bundle_sha256 = report.candidate.manifest_sha256
    runtime = EnterpriseEmbeddingRolloutRuntime(
        model_id=model_id,
        model_revision=model_revision,
        stable_directory=stable_directory,
        candidate_directory=candidate_directory,
        candidate_bundle_sha256=candidate_bundle_sha256,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    runtime.load()
    uvicorn = importlib.import_module("uvicorn")
    uvicorn.run(
        create_app(runtime),
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
    )
    return 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
