"""Governed GPU runtime for raw and evidence-calibrated Reranker bundles.

This module deliberately lives outside the stable online retrieval package.  It
implements the existing runtime HTTP contract while making the calibration
descriptor stored inside an immutable model bundle executable at serving time.
"""

from __future__ import annotations

import argparse
import importlib
import math
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from industrial_ops_agent.runtime_metrics import RerankerRuntimeMetrics
from industrial_ops_agent.search_profiles.reranker import RERANKER_RUNTIME_SCHEMA
from industrial_ops_agent.search_profiles.reranker_runtime import (
    RerankedItemResponse,
    RerankerInferenceRequest,
    RerankerInferenceResponse,
    RerankerRuntimeIdentity,
)

CALIBRATION_DESCRIPTOR = "industrial-ops-calibrated-runtime.json"
TERMINOLOGY_REGISTRY = "industrial-ops-terminology-registry.json"
CALIBRATION_PROFILE_ID = "industrial-reranker-evidence-calibration-v1"
REGISTRY_ID = "industrial-reranker-terminology-registry-v1"


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CalibrationProfile(_ClosedModel):
    profile_id: Literal["industrial-reranker-evidence-calibration-v1"]
    model_score_transform: Literal["SIGMOID"]
    registry_only: Literal[True]
    exact_mechanism_phrase_bonus: float = Field(ge=2.0, le=2.0)
    exact_action_phrase_bonus: float = Field(ge=1.0, le=1.0)
    maximum_evidence_bonus: float = Field(ge=3.0, le=3.0)
    missing_or_ambiguous_alias: Literal["FAIL_CLOSED"]
    gold_derived: Literal[False]


class CalibratedRuntimeDescriptor(_ClosedModel):
    schema_version: Literal["industrial-reranker-calibrated-runtime/v1"]
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"]
    source_v4_run_id: str = Field(pattern=r"^reranker-grounded-[0-9a-f]{20}$")
    source_v4_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_augmentation: Literal["GOVERNED_TERMINOLOGY_REGISTRY_V1"]
    score_calibration: CalibrationProfile
    missing_alias: Literal["FAIL_CLOSED"]
    ambiguous_alias: Literal["FAIL_CLOSED"]
    gold_derived: Literal[False]


class TerminologyEntry(_ClosedModel):
    entry_id: str = Field(pattern=r"^term-[0-9]{3}$")
    alias: str = Field(min_length=1, max_length=256)
    device_family: str = Field(min_length=1, max_length=128)
    signal: str = Field(min_length=1, max_length=128)
    mechanism: str = Field(min_length=1, max_length=2000)
    action: str = Field(min_length=1, max_length=2000)


class RegistryResolutionPolicy(_ClosedModel):
    match: Literal["EXACT_CASE_INSENSITIVE_ALIAS"]
    missing_alias: Literal["FAIL_CLOSED"]
    ambiguous_alias: Literal["FAIL_CLOSED"]
    candidate_only: Literal[True]


class TerminologyRegistry(_ClosedModel):
    schema_version: Literal["industrial-reranker-terminology-registry/v1"]
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"]
    enterprise_scope: Literal["PROJECT_INTERNAL"]
    production_claim: Literal[False]
    enterprise_production_data: Literal[False]
    project_generated_data: Literal[True]
    project_enterprise_use_authorized: Literal[True]
    registry_id: Literal["industrial-reranker-terminology-registry-v1"]
    review_status: Literal["APPROVED"]
    frozen_before_gold_evaluation: Literal[True]
    derived_from_formal_gold: Literal[False]
    resolution_policy: RegistryResolutionPolicy
    entries: tuple[TerminologyEntry, ...] = Field(min_length=1, max_length=256)


@dataclass(frozen=True, slots=True)
class CalibratedBundle:
    descriptor: CalibratedRuntimeDescriptor
    registry: TerminologyRegistry

    @property
    def profile_id(self) -> str:
        return self.descriptor.score_calibration.profile_id


def load_calibrated_bundle(model_directory: Path) -> CalibratedBundle | None:
    """Load a strict optional calibration profile; partial profiles fail closed."""

    descriptor_path = model_directory / CALIBRATION_DESCRIPTOR
    registry_path = model_directory / TERMINOLOGY_REGISTRY
    if not descriptor_path.exists() and not registry_path.exists():
        return None
    if not descriptor_path.is_file() or not registry_path.is_file():
        raise RuntimeError("reranker_calibration_bundle_is_incomplete")
    try:
        descriptor = CalibratedRuntimeDescriptor.model_validate_json(descriptor_path.read_bytes())
        registry = TerminologyRegistry.model_validate_json(registry_path.read_bytes())
    except (OSError, UnicodeError, ValidationError) as exc:
        raise RuntimeError("reranker_calibration_bundle_is_invalid") from exc
    aliases = [item.alias.casefold() for item in registry.entries]
    entry_ids = [item.entry_id for item in registry.entries]
    if len(aliases) != len(set(aliases)) or len(entry_ids) != len(set(entry_ids)):
        raise RuntimeError("reranker_calibration_registry_is_ambiguous")
    return CalibratedBundle(descriptor=descriptor, registry=registry)


def expand_governed_query(query: str, bundle: CalibratedBundle) -> tuple[str, TerminologyEntry]:
    folded = query.casefold()
    matches = tuple(item for item in bundle.registry.entries if item.alias.casefold() in folded)
    if len(matches) != 1:
        raise ValueError("reranker_terminology_resolution_requires_exactly_one_entry")
    entry = matches[0]
    expanded = (
        f"{query}\nGoverned terminology resolution: {entry.mechanism}. "
        f"Required controlled action: {entry.action}. Device family: "
        f"{entry.device_family}; signal family: {entry.signal}."
    )
    return expanded, entry


def calibrated_score(model_score: float, document_text: str, entry: TerminologyEntry) -> float:
    if not math.isfinite(model_score):
        raise RuntimeError("reranker_runtime_score_contract_is_invalid")
    bounded_model = 1.0 / (1.0 + math.exp(-max(min(model_score, 60.0), -60.0)))
    folded = document_text.casefold()
    mechanism_bonus = 2.0 if entry.mechanism.casefold() in folded else 0.0
    action_bonus = 1.0 if entry.action.casefold() in folded else 0.0
    return bounded_model + mechanism_bonus + action_bonus


class GovernedRerankerRuntimeModel:
    """Serve an immutable CrossEncoder and execute its optional bundle policy."""

    def __init__(
        self,
        model_directory: Path,
        *,
        identity: RerankerRuntimeIdentity,
        precision: str,
        batch_size: int = 16,
        max_sequence_length: int = 128,
    ) -> None:
        if precision not in {"bfloat16", "float16"}:
            raise ValueError("reranker_runtime_precision_is_invalid")
        if not 1 <= batch_size <= 128:
            raise ValueError("reranker_runtime_batch_size_is_invalid")
        if not 128 <= max_sequence_length <= 8192:
            raise ValueError("reranker_runtime_max_sequence_length_is_invalid")
        resolved = model_directory.resolve(strict=True)
        if not resolved.is_dir():
            raise RuntimeError("reranker_runtime_model_directory_is_missing")
        calibration = load_calibrated_bundle(resolved)
        try:
            torch = importlib.import_module("torch")
            cross_encoder = importlib.import_module("sentence_transformers").CrossEncoder
        except ImportError as exc:  # pragma: no cover - runtime image responsibility
            raise RuntimeError("reranker_runtime_dependencies_are_not_installed") from exc
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("reranker_runtime_requires_exactly_one_cuda_gpu")
        torch.set_num_threads(2)
        dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
        self._model = cross_encoder(
            str(resolved),
            trust_remote_code=False,
            device="cuda:0",
            num_labels=1,
            max_length=max_sequence_length,
            model_kwargs={"torch_dtype": dtype},
        )
        self._torch = torch
        self._identity = identity
        self._batch_size = batch_size
        self._max_sequence_length = max_sequence_length
        self._calibration = calibration
        self._lock = Lock()

    @property
    def release_id(self) -> str:
        return self._identity.release_id

    def status(self) -> dict[str, str | bool | int]:
        calibration = self._calibration
        return {
            "status": "ready",
            "release_id": self._identity.release_id,
            "component_model_id": self._identity.component_model_id,
            "calibration_enabled": calibration is not None,
            "runtime_profile_id": (
                calibration.profile_id if calibration is not None else "raw-cross-encoder-v1"
            ),
            "registry_id": (
                calibration.registry.registry_id if calibration is not None else "not-applicable"
            ),
            "max_sequence_length": self._max_sequence_length,
        }

    def predict(self, request: RerankerInferenceRequest) -> RerankerInferenceResponse:
        _require_contract(request, self._identity)
        candidate_ids = [item.chunk_id for item in request.candidates]
        if (
            request.top_k > len(request.candidates)
            or len(candidate_ids) != len(set(candidate_ids))
            or any(not math.isfinite(item.recall_score) for item in request.candidates)
        ):
            raise ValueError("reranker_runtime_candidate_contract_is_invalid")
        calibration = self._calibration
        if calibration is None:
            query = request.query
            entry = None
        else:
            query, entry = expand_governed_query(request.query, calibration)
        document_texts = [f"{item.title}\n{item.content}" for item in request.candidates]
        pairs = list(zip((query for _ in document_texts), document_texts, strict=True))
        with self._lock, self._torch.no_grad():
            raw_scores = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        scores = _score_list(raw_scores)
        if len(scores) != len(request.candidates):
            raise RuntimeError("reranker_runtime_score_contract_is_invalid")
        if entry is not None:
            scores = [
                calibrated_score(score, text, entry)
                for score, text in zip(scores, document_texts, strict=True)
            ]
        elif any(not math.isfinite(score) for score in scores):
            raise RuntimeError("reranker_runtime_score_contract_is_invalid")
        ranked = sorted(
            zip(request.candidates, scores, strict=True),
            key=lambda item: (-item[1], item[0].chunk_id),
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


def create_governed_reranker_runtime_app(
    model: GovernedRerankerRuntimeModel,
    metrics: RerankerRuntimeMetrics | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    app = FastAPI(title="Industrial governed CrossEncoder runtime", lifespan=lifespan)
    runtime_metrics = metrics or RerankerRuntimeMetrics()

    @app.get("/health/ready")
    async def ready() -> dict[str, str | bool | int]:
        return model.status()

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
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return [float(value)]
    scores: list[float] = []
    for item in value:
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            if len(item) != 1:
                raise RuntimeError("reranker_runtime_score_shape_is_invalid")
            item = item[0]
        scores.append(float(item))
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a governed calibrated Reranker")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--manifest-hash", required=True)
    parser.add_argument("--component-model-id", required=True)
    parser.add_argument("--artifact-content-hash", required=True)
    parser.add_argument("--precision", choices=("bfloat16", "float16"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 - container listener
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    model = GovernedRerankerRuntimeModel(
        args.model_dir,
        identity=RerankerRuntimeIdentity(
            release_id=args.release_id,
            manifest_hash=args.manifest_hash,
            component_model_id=args.component_model_id,
            artifact_content_hash=args.artifact_content_hash,
        ),
        precision=args.precision,
        batch_size=args.batch_size,
        max_sequence_length=args.max_sequence_length,
    )
    uvicorn = importlib.import_module("uvicorn")
    uvicorn.run(create_governed_reranker_runtime_app(model), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
