"""Resolve and call the Embedding model bound to the active production release."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import Any, Protocol

import httpx
from sqlalchemy import select

from industrial_ops_agent.deployment.service import PRODUCTION_MODEL_ALIAS
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    ModelAliasRecord,
    ModelDeploymentRecord,
    ModelReleaseRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.runtime_metrics import KnowledgeEmbeddingOnlineMetrics

EMBEDDING_RUNTIME_SCHEMA = "industrial-embedding-runtime/v1"


class EmbeddingUnavailable(RuntimeError):
    def __init__(self, reason: str, *, release_id: str | None = None) -> None:
        self.reason = reason
        self.release_id = release_id
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class EmbeddingInput:
    item_id: str
    text: str


@dataclass(frozen=True, slots=True)
class EmbeddedItem:
    item_id: str
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EmbeddingOutcome:
    model_release_id: str
    manifest_hash: str
    component_model_id: str
    artifact_content_hash: str
    dimension: int
    latency_ms: float
    items: tuple[EmbeddedItem, ...]


class KnowledgeEmbedder(Protocol):
    def embed(
        self,
        context: TenantContext,
        *,
        index_release_id: str,
        inputs: tuple[EmbeddingInput, ...],
    ) -> EmbeddingOutcome: ...


@dataclass(frozen=True, slots=True)
class ResolvedEmbeddingModel:
    release_id: str
    manifest_hash: str
    endpoint_url: str
    component_model_id: str
    artifact_content_hash: str
    index_release_id: str


class ProductionEmbeddingResolver:
    def __init__(self, database: Database) -> None:
        self._database = database

    def resolve(
        self,
        context: TenantContext,
        *,
        index_release_id: str,
    ) -> ResolvedEmbeddingModel:
        with self._database.transaction(context) as session:
            alias = session.scalar(
                select(ModelAliasRecord).where(
                    ModelAliasRecord.tenant_id == context.tenant_id,
                    ModelAliasRecord.alias == PRODUCTION_MODEL_ALIAS,
                    ModelAliasRecord.status == "ACTIVE",
                )
            )
            if alias is None:
                raise EmbeddingUnavailable("production_release_unavailable")
            release = session.scalar(
                select(ModelReleaseRecord).where(
                    ModelReleaseRecord.tenant_id == context.tenant_id,
                    ModelReleaseRecord.release_id == alias.active_release_id,
                    ModelReleaseRecord.status == "PRODUCTION",
                )
            )
            deployment = session.scalar(
                select(ModelDeploymentRecord).where(
                    ModelDeploymentRecord.tenant_id == context.tenant_id,
                    ModelDeploymentRecord.deployment_id == alias.deployment_id,
                    ModelDeploymentRecord.status == "READY",
                )
            )
            deployment_release = (
                session.scalar(
                    select(ModelReleaseRecord).where(
                        ModelReleaseRecord.tenant_id == context.tenant_id,
                        ModelReleaseRecord.release_id == deployment.release_id,
                    )
                )
                if deployment is not None
                else None
            )
            route_matches_release = bool(
                deployment is not None
                and (
                    (
                        deployment.current_stage == "PRODUCTION"
                        and deployment.release_id == alias.active_release_id
                    )
                    or (
                        deployment.current_stage == "ROLLED_BACK"
                        and deployment.release_id != alias.active_release_id
                        and deployment_release is not None
                        and deployment_release.rollback_release_id == alias.active_release_id
                    )
                )
            )
            if (
                release is None
                or deployment is None
                or not route_matches_release
                or alias.manifest_hash != release.manifest_hash
                or alias.endpoint_url != deployment.endpoint_url
                or _digest_json(release.manifest_json) != release.manifest_hash
            ):
                raise EmbeddingUnavailable(
                    "production_release_route_inconsistent",
                    release_id=alias.active_release_id,
                )
            retrieval = release.manifest_json.get("retrieval")
            components = release.manifest_json.get("specialized_components")
            component = components.get("embedding") if isinstance(components, dict) else None
            if not isinstance(retrieval, dict) or not isinstance(component, dict):
                raise EmbeddingUnavailable(
                    "production_embedding_component_unavailable",
                    release_id=release.release_id,
                )
            try:
                manifest_index_release_id = str(retrieval["index_release_id"])
                retrieval_model_id = str(retrieval["embedding_model_id"])
                component_model_id = str(component["runtime_model_id"])
                artifact = component["artifact"]
                artifact_kind = str(artifact["kind"])
                artifact_content_hash = str(artifact["content_hash"])
                artifact_id = str(artifact["artifact_id"])
                runtime = component["runtime"]
                runtime_profile_id = str(runtime["profile_id"])
                runtime_engine = str(runtime["inference_config"]["engine"])
            except (KeyError, TypeError, ValueError) as exc:
                raise EmbeddingUnavailable(
                    "production_embedding_manifest_invalid",
                    release_id=release.release_id,
                ) from exc
            if (
                manifest_index_release_id != index_release_id
                or retrieval_model_id != component_model_id
                or artifact_kind != "embedding_model_bundle"
                or runtime_profile_id != "sentence-transformers-embedding-v1"
                or runtime_engine != "sentence-transformers-embedding"
                or not component_model_id
                or not artifact_content_hash
                or not artifact_id
            ):
                reason = (
                    "production_embedding_index_binding_mismatch"
                    if manifest_index_release_id != index_release_id
                    else "production_embedding_manifest_invalid"
                )
                raise EmbeddingUnavailable(reason, release_id=release.release_id)
            return ResolvedEmbeddingModel(
                release_id=release.release_id,
                manifest_hash=release.manifest_hash,
                endpoint_url=f"{alias.endpoint_url.rstrip('/')}/v1/embedding",
                component_model_id=component_model_id,
                artifact_content_hash=artifact_content_hash,
                index_release_id=manifest_index_release_id,
            )


class HttpKnowledgeEmbedder:
    def __init__(
        self,
        resolver: ProductionEmbeddingResolver,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 10.0,
        allow_plain_http: bool = False,
        metrics: KnowledgeEmbeddingOnlineMetrics | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("embedding inference timeout must be positive")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._resolver = resolver
        self._client = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
        )
        self._allow_plain_http = allow_plain_http
        self._metrics = metrics

    def close(self) -> None:
        self._client.close()

    def embed(
        self,
        context: TenantContext,
        *,
        index_release_id: str,
        inputs: tuple[EmbeddingInput, ...],
    ) -> EmbeddingOutcome:
        started = perf_counter()
        release_id = "unresolved"
        try:
            model = self._resolver.resolve(context, index_release_id=index_release_id)
            release_id = model.release_id
            if not self._allow_plain_http and not model.endpoint_url.startswith("https://"):
                raise EmbeddingUnavailable(
                    "embedding_inference_requires_https",
                    release_id=model.release_id,
                )
            input_ids = [item.item_id for item in inputs]
            if (
                not 1 <= len(inputs) <= 64
                or len(input_ids) != len(set(input_ids))
                or any(not item.text.strip() for item in inputs)
            ):
                raise EmbeddingUnavailable(
                    "embedding_input_contract_invalid",
                    release_id=model.release_id,
                )
            payload = {
                "schema_version": EMBEDDING_RUNTIME_SCHEMA,
                "index_release_id": index_release_id,
                "inputs": [{"item_id": item.item_id, "text": item.text} for item in inputs],
                "expected_release_id": model.release_id,
                "expected_manifest_hash": model.manifest_hash,
                "expected_component_model_id": model.component_model_id,
                "expected_artifact_content_hash": model.artifact_content_hash,
            }
            try:
                response = self._client.post(model.endpoint_url, json=payload)
                response.raise_for_status()
                body = response.json()
            except httpx.TimeoutException as exc:
                raise EmbeddingUnavailable(
                    "embedding_inference_timeout",
                    release_id=model.release_id,
                ) from exc
            except (httpx.HTTPError, ValueError) as exc:
                raise EmbeddingUnavailable(
                    "embedding_inference_transport_failed",
                    release_id=model.release_id,
                ) from exc
            dimension, items = _validated_response(body, model=model, inputs=inputs)
        except EmbeddingUnavailable as exc:
            if self._metrics is not None:
                self._metrics.observe(
                    release_id=exc.release_id or release_id,
                    status="failure",
                    seconds=perf_counter() - started,
                    item_count=len(inputs),
                    reason=exc.reason,
                )
            raise
        latency_ms = (perf_counter() - started) * 1000.0
        if self._metrics is not None:
            self._metrics.observe(
                release_id=model.release_id,
                status="success",
                seconds=latency_ms / 1000.0,
                item_count=len(inputs),
            )
        return EmbeddingOutcome(
            model_release_id=model.release_id,
            manifest_hash=model.manifest_hash,
            component_model_id=model.component_model_id,
            artifact_content_hash=model.artifact_content_hash,
            dimension=dimension,
            latency_ms=latency_ms,
            items=items,
        )


def _validated_response(
    body: Any,
    *,
    model: ResolvedEmbeddingModel,
    inputs: tuple[EmbeddingInput, ...],
) -> tuple[int, tuple[EmbeddedItem, ...]]:
    if not isinstance(body, dict):
        raise EmbeddingUnavailable("embedding_response_invalid", release_id=model.release_id)
    if (
        body.get("schema_version") != EMBEDDING_RUNTIME_SCHEMA
        or body.get("index_release_id") != model.index_release_id
        or body.get("release_id") != model.release_id
        or body.get("manifest_hash") != model.manifest_hash
        or body.get("component_model_id") != model.component_model_id
        or body.get("artifact_content_hash") != model.artifact_content_hash
    ):
        raise EmbeddingUnavailable(
            "embedding_response_binding_failed",
            release_id=model.release_id,
        )
    dimension = body.get("dimension")
    raw_items = body.get("items")
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or not 8 <= dimension <= 8192
        or not isinstance(raw_items, list)
        or len(raw_items) != len(inputs)
    ):
        raise EmbeddingUnavailable("embedding_response_invalid", release_id=model.release_id)
    parsed: list[EmbeddedItem] = []
    try:
        for item in raw_items:
            if not isinstance(item, dict) or not isinstance(item["vector"], list):
                raise TypeError
            vector = tuple(float(value) for value in item["vector"])
            parsed.append(EmbeddedItem(item_id=str(item["item_id"]), vector=vector))
    except (KeyError, TypeError, ValueError) as exc:
        raise EmbeddingUnavailable(
            "embedding_response_invalid",
            release_id=model.release_id,
        ) from exc
    expected_ids = [item.item_id for item in inputs]
    returned_ids = [item.item_id for item in parsed]
    if returned_ids != expected_ids or any(
        len(item.vector) != dimension
        or any(not math.isfinite(value) for value in item.vector)
        or not 0.9 <= math.sqrt(sum(value * value for value in item.vector)) <= 1.1
        for item in parsed
    ):
        raise EmbeddingUnavailable("embedding_response_invalid", release_id=model.release_id)
    return dimension, tuple(parsed)


def _digest_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{sha256(payload).hexdigest()}"
