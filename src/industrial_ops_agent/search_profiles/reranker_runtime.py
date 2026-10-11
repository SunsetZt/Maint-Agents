"""Single-GPU CrossEncoder server for an immutable governed Reranker bundle."""

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

from industrial_ops_agent.runtime_metrics import RerankerRuntimeMetrics
from industrial_ops_agent.search_profiles.reranker import RERANKER_RUNTIME_SCHEMA


class RerankerCandidateRequest(BaseModel):
    chunk_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=2000)
    content: str = Field(min_length=1, max_length=65536)
    recall_score: float


class RerankerInferenceRequest(BaseModel):
    schema_version: str = Field(min_length=1, max_length=64)
    index_release_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=2000)
    candidates: list[RerankerCandidateRequest] = Field(min_length=1, max_length=50)
    top_k: int = Field(ge=1, le=20)
    expected_release_id: str = Field(min_length=1, max_length=128)
    expected_manifest_hash: str = Field(min_length=1, max_length=128)
    expected_component_model_id: str = Field(min_length=1, max_length=128)
    expected_artifact_content_hash: str = Field(min_length=1, max_length=128)


class RerankedItemResponse(BaseModel):
    chunk_id: str
    score: float


class RerankerInferenceResponse(BaseModel):
    schema_version: str
    index_release_id: str
    release_id: str
    manifest_hash: str
    component_model_id: str
    artifact_content_hash: str
    items: list[RerankedItemResponse]


@dataclass(frozen=True, slots=True)
class RerankerRuntimeIdentity:
    release_id: str
    manifest_hash: str
    component_model_id: str
    artifact_content_hash: str


class RerankerRuntimeModel:
    def __init__(
        self,
        model_directory: Path,
        *,
        identity: RerankerRuntimeIdentity,
        precision: str,
        batch_size: int = 16,
    ) -> None:
        if precision not in {"bfloat16", "float16"}:
            raise ValueError("reranker_runtime_precision_is_invalid")
        if not 1 <= batch_size <= 128:
            raise ValueError("reranker_runtime_batch_size_is_invalid")
        if not model_directory.is_dir():
            raise RuntimeError("reranker_runtime_model_directory_is_missing")
        try:
            torch = importlib.import_module("torch")
            CrossEncoder = importlib.import_module("sentence_transformers").CrossEncoder
        except ImportError as exc:  # pragma: no cover - runtime image responsibility
            raise RuntimeError("reranker_runtime_dependencies_are_not_installed") from exc
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("reranker_runtime_requires_exactly_one_cuda_gpu")
        dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
        self._model = CrossEncoder(
            str(model_directory),
            trust_remote_code=False,
            device="cuda:0",
            num_labels=1,
            model_kwargs={"torch_dtype": dtype},
        )
        self._torch = torch
        self._identity = identity
        self._batch_size = batch_size
        self._lock = Lock()

    @property
    def release_id(self) -> str:
        return self._identity.release_id

    def predict(self, request: RerankerInferenceRequest) -> RerankerInferenceResponse:
        _require_contract(request, self._identity)
        candidate_ids = [item.chunk_id for item in request.candidates]
        if (
            request.top_k > len(request.candidates)
            or len(candidate_ids) != len(set(candidate_ids))
            or any(not math.isfinite(item.recall_score) for item in request.candidates)
        ):
            raise ValueError("reranker_runtime_candidate_contract_is_invalid")
        pairs = [(request.query, f"{item.title}\n{item.content}") for item in request.candidates]
        with self._lock, self._torch.no_grad():
            raw_scores = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
        scores = _score_list(raw_scores)
        if len(scores) != len(request.candidates) or any(
            not math.isfinite(score) for score in scores
        ):
            raise RuntimeError("reranker_runtime_score_contract_is_invalid")
        ranked = sorted(
            zip(request.candidates, scores, strict=True),
            key=lambda item: (-item[1], -item[0].recall_score, item[0].chunk_id),
        )[: request.top_k]
        return RerankerInferenceResponse(
            schema_version=RERANKER_RUNTIME_SCHEMA,
            index_release_id=request.index_release_id,
            release_id=self._identity.release_id,
            manifest_hash=self._identity.manifest_hash,
            component_model_id=self._identity.component_model_id,
            artifact_content_hash=self._identity.artifact_content_hash,
            items=[
                RerankedItemResponse(chunk_id=candidate.chunk_id, score=score)
                for candidate, score in ranked
            ],
        )


def create_reranker_runtime_app(
    model: RerankerRuntimeModel,
    metrics: RerankerRuntimeMetrics | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    app = FastAPI(title="Industrial knowledge CrossEncoder runtime", lifespan=lifespan)
    runtime_metrics = metrics or RerankerRuntimeMetrics()

    @app.get("/health/ready")
    async def ready() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        return Response(
            content=generate_latest(runtime_metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.post("/v1/retrieval/rerank", response_model=RerankerInferenceResponse)
    def rerank(request: RerankerInferenceRequest) -> RerankerInferenceResponse:
        started = perf_counter()
        try:
            result = model.predict(request)
        except ValueError as exc:
            runtime_metrics.observe(
                release_id=model.release_id,
                status="rejected",
                seconds=perf_counter() - started,
                candidate_count=len(request.candidates),
            )
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            runtime_metrics.observe(
                release_id=model.release_id,
                status="error",
                seconds=perf_counter() - started,
                candidate_count=len(request.candidates),
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        runtime_metrics.observe(
            release_id=model.release_id,
            status="success",
            seconds=perf_counter() - started,
            candidate_count=len(request.candidates),
        )
        return result

    return app


def _require_contract(
    request: RerankerInferenceRequest,
    identity: RerankerRuntimeIdentity,
) -> None:
    if request.schema_version != RERANKER_RUNTIME_SCHEMA:
        raise ValueError("reranker_runtime_schema_mismatch")
    if (
        request.expected_release_id != identity.release_id
        or request.expected_manifest_hash != identity.manifest_hash
        or request.expected_component_model_id != identity.component_model_id
        or request.expected_artifact_content_hash != identity.artifact_content_hash
    ):
        raise ValueError("reranker_runtime_release_binding_mismatch")


def _score_list(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return [float(value)]
    scores: list[float] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            if len(item) != 1:
                raise RuntimeError("reranker_runtime_score_shape_is_invalid")
            item = item[0]
        scores.append(float(item))
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a governed knowledge Reranker")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--manifest-hash", required=True)
    parser.add_argument("--component-model-id", required=True)
    parser.add_argument("--artifact-content-hash", required=True)
    parser.add_argument("--precision", choices=("bfloat16", "float16"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 - container listener
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    model = RerankerRuntimeModel(
        args.model_dir,
        identity=RerankerRuntimeIdentity(
            release_id=args.release_id,
            manifest_hash=args.manifest_hash,
            component_model_id=args.component_model_id,
            artifact_content_hash=args.artifact_content_hash,
        ),
        precision=args.precision,
        batch_size=args.batch_size,
    )
    uvicorn = importlib.import_module("uvicorn")
    uvicorn.run(create_reranker_runtime_app(model), host=args.host, port=args.port)
    return 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
