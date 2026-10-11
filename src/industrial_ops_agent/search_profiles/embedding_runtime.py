"""Single-GPU SentenceTransformer server for an immutable Embedding bundle."""

from __future__ import annotations

import argparse
import importlib
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

from industrial_ops_agent.runtime_metrics import EmbeddingRuntimeMetrics
from industrial_ops_agent.search_profiles.embedding import EMBEDDING_RUNTIME_SCHEMA


class EmbeddingInputRequest(BaseModel):
    item_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=65536)


class EmbeddingInferenceRequest(BaseModel):
    schema_version: str = Field(min_length=1, max_length=64)
    index_release_id: str = Field(min_length=1, max_length=128)
    inputs: list[EmbeddingInputRequest] = Field(min_length=1, max_length=64)
    expected_release_id: str = Field(min_length=1, max_length=128)
    expected_manifest_hash: str = Field(min_length=1, max_length=128)
    expected_component_model_id: str = Field(min_length=1, max_length=128)
    expected_artifact_content_hash: str = Field(min_length=1, max_length=128)


class EmbeddedItemResponse(BaseModel):
    item_id: str
    vector: list[float]


class EmbeddingInferenceResponse(BaseModel):
    schema_version: str
    index_release_id: str
    release_id: str
    manifest_hash: str
    component_model_id: str
    artifact_content_hash: str
    dimension: int
    items: list[EmbeddedItemResponse]


@dataclass(frozen=True, slots=True)
class EmbeddingRuntimeIdentity:
    release_id: str
    manifest_hash: str
    component_model_id: str
    artifact_content_hash: str


class EmbeddingRuntimeModel:
    def __init__(
        self,
        model_directory: Path,
        *,
        identity: EmbeddingRuntimeIdentity,
        precision: str,
        batch_size: int = 32,
    ) -> None:
        if precision not in {"bfloat16", "float16"}:
            raise ValueError("embedding_runtime_precision_is_invalid")
        if not 1 <= batch_size <= 128:
            raise ValueError("embedding_runtime_batch_size_is_invalid")
        if not model_directory.is_dir():
            raise RuntimeError("embedding_runtime_model_directory_is_missing")
        try:
            torch = importlib.import_module("torch")
            SentenceTransformer = importlib.import_module(
                "sentence_transformers"
            ).SentenceTransformer
        except ImportError as exc:  # pragma: no cover - runtime image responsibility
            raise RuntimeError("embedding_runtime_dependencies_are_not_installed") from exc
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("embedding_runtime_requires_exactly_one_cuda_gpu")
        dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
        self._model = SentenceTransformer(
            str(model_directory),
            trust_remote_code=False,
            device="cuda:0",
            model_kwargs={"torch_dtype": dtype},
        )
        dimension = self._model.get_sentence_embedding_dimension()
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or not 8 <= dimension <= 8192
        ):
            raise RuntimeError("embedding_runtime_dimension_is_invalid")
        self._torch = torch
        self._identity = identity
        self._dimension = dimension
        self._batch_size = batch_size
        self._lock = Lock()

    @property
    def release_id(self) -> str:
        return self._identity.release_id

    def predict(self, request: EmbeddingInferenceRequest) -> EmbeddingInferenceResponse:
        _require_contract(request, self._identity)
        item_ids = [item.item_id for item in request.inputs]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("embedding_runtime_input_ids_are_not_unique")
        with self._lock, self._torch.no_grad():
            raw_vectors = self._model.encode(
                [item.text for item in request.inputs],
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        vectors = _vector_list(raw_vectors)
        if len(vectors) != len(request.inputs) or any(
            len(vector) != self._dimension or any(not math.isfinite(value) for value in vector)
            for vector in vectors
        ):
            raise RuntimeError("embedding_runtime_vector_contract_is_invalid")
        return EmbeddingInferenceResponse(
            schema_version=EMBEDDING_RUNTIME_SCHEMA,
            index_release_id=request.index_release_id,
            release_id=self._identity.release_id,
            manifest_hash=self._identity.manifest_hash,
            component_model_id=self._identity.component_model_id,
            artifact_content_hash=self._identity.artifact_content_hash,
            dimension=self._dimension,
            items=[
                EmbeddedItemResponse(item_id=item.item_id, vector=vector)
                for item, vector in zip(request.inputs, vectors, strict=True)
            ],
        )


def create_embedding_runtime_app(
    model: EmbeddingRuntimeModel,
    metrics: EmbeddingRuntimeMetrics | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    app = FastAPI(title="Industrial knowledge Embedding runtime", lifespan=lifespan)
    runtime_metrics = metrics or EmbeddingRuntimeMetrics()

    @app.get("/health/ready")
    async def ready() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        return Response(
            content=generate_latest(runtime_metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.post("/v1/embedding", response_model=EmbeddingInferenceResponse)
    def embed(request: EmbeddingInferenceRequest) -> EmbeddingInferenceResponse:
        started = perf_counter()
        try:
            result = model.predict(request)
        except ValueError as exc:
            runtime_metrics.observe(
                release_id=model.release_id,
                status="rejected",
                seconds=perf_counter() - started,
                item_count=len(request.inputs),
            )
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            runtime_metrics.observe(
                release_id=model.release_id,
                status="error",
                seconds=perf_counter() - started,
                item_count=len(request.inputs),
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        runtime_metrics.observe(
            release_id=model.release_id,
            status="success",
            seconds=perf_counter() - started,
            item_count=len(request.inputs),
        )
        return result

    return app


def _require_contract(
    request: EmbeddingInferenceRequest,
    identity: EmbeddingRuntimeIdentity,
) -> None:
    if request.schema_version != EMBEDDING_RUNTIME_SCHEMA:
        raise ValueError("embedding_runtime_schema_mismatch")
    if (
        request.expected_release_id != identity.release_id
        or request.expected_manifest_hash != identity.manifest_hash
        or request.expected_component_model_id != identity.component_model_id
        or request.expected_artifact_content_hash != identity.artifact_content_hash
    ):
        raise ValueError("embedding_runtime_release_binding_mismatch")


def _vector_list(value: Any) -> list[list[float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise RuntimeError("embedding_runtime_vector_shape_is_invalid")
    vectors: list[list[float]] = []
    for item in value:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if not isinstance(item, (list, tuple)):
            raise RuntimeError("embedding_runtime_vector_shape_is_invalid")
        vectors.append([float(component) for component in item])
    return vectors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a governed knowledge Embedding model")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--manifest-hash", required=True)
    parser.add_argument("--component-model-id", required=True)
    parser.add_argument("--artifact-content-hash", required=True)
    parser.add_argument("--precision", choices=("bfloat16", "float16"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 - container listener
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    model = EmbeddingRuntimeModel(
        args.model_dir,
        identity=EmbeddingRuntimeIdentity(
            release_id=args.release_id,
            manifest_hash=args.manifest_hash,
            component_model_id=args.component_model_id,
            artifact_content_hash=args.artifact_content_hash,
        ),
        precision=args.precision,
        batch_size=args.batch_size,
    )
    uvicorn = importlib.import_module("uvicorn")
    uvicorn.run(create_embedding_runtime_app(model), host=args.host, port=args.port)
    return 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
