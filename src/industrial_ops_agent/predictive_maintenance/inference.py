"""Resolve and call the time-series component of the active production release."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
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
from industrial_ops_agent.predictive_maintenance.dataset_contract import CONTRACT_VERSION


class TimeseriesInferenceUnavailable(RuntimeError):
    def __init__(self, reason: str, *, release_id: str | None = None) -> None:
        self.reason = reason
        self.release_id = release_id
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class TimeseriesDetectionInput:
    feature_snapshot_id: str
    sequence: tuple[tuple[float, ...], ...]
    mask: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class TimeseriesDetection:
    model_release_id: str
    component_model_id: str
    detector_kind: str
    anomaly_score: float
    threshold: float
    threshold_policy_id: str
    explanation_codes: tuple[str, ...]


class TimeseriesDetector(Protocol):
    def detect(
        self,
        context: TenantContext,
        value: TimeseriesDetectionInput,
    ) -> TimeseriesDetection: ...


@dataclass(frozen=True, slots=True)
class ResolvedTimeseriesModel:
    release_id: str
    manifest_hash: str
    endpoint_url: str
    component_model_id: str
    artifact_content_hash: str
    threshold: float
    threshold_policy_id: str


class ProductionTimeseriesResolver:
    def __init__(self, database: Database) -> None:
        self._database = database

    def resolve(self, context: TenantContext) -> ResolvedTimeseriesModel:
        with self._database.transaction(context) as session:
            alias = session.scalar(
                select(ModelAliasRecord).where(
                    ModelAliasRecord.tenant_id == context.tenant_id,
                    ModelAliasRecord.alias == PRODUCTION_MODEL_ALIAS,
                    ModelAliasRecord.status == "ACTIVE",
                )
            )
            if alias is None:
                raise TimeseriesInferenceUnavailable("production_release_unavailable")
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
                raise TimeseriesInferenceUnavailable(
                    "production_release_route_inconsistent",
                    release_id=alias.active_release_id,
                )
            components = release.manifest_json.get("specialized_components")
            component = components.get("timeseries") if isinstance(components, dict) else None
            if not isinstance(component, dict):
                raise TimeseriesInferenceUnavailable(
                    "production_timeseries_component_unavailable",
                    release_id=release.release_id,
                )
            try:
                artifact = component["artifact"]
                runtime = component["runtime"]
                inference = runtime["inference_config"]
                evaluation = component["evaluation"]
                component_model_id = str(component["runtime_model_id"])
                artifact_hash = str(artifact["content_hash"])
                threshold = float(inference["anomaly_threshold"])
                threshold_policy_id = str(evaluation["policy_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise TimeseriesInferenceUnavailable(
                    "production_timeseries_manifest_invalid",
                    release_id=release.release_id,
                ) from exc
            if (
                runtime.get("profile_id") != "native-pytorch-timeseries-v1"
                or inference.get("engine") != "native-pytorch-transformer-reconstruction"
                or not math.isfinite(threshold)
                or threshold <= 0
                or not component_model_id
                or not artifact_hash.startswith("sha256:")
                or not threshold_policy_id
            ):
                raise TimeseriesInferenceUnavailable(
                    "production_timeseries_manifest_invalid",
                    release_id=release.release_id,
                )
            return ResolvedTimeseriesModel(
                release_id=release.release_id,
                manifest_hash=release.manifest_hash,
                endpoint_url=f"{alias.endpoint_url.rstrip('/')}/v1/timeseries/anomaly",
                component_model_id=component_model_id,
                artifact_content_hash=artifact_hash,
                threshold=threshold,
                threshold_policy_id=threshold_policy_id,
            )


class HttpTimeseriesDetector:
    def __init__(
        self,
        resolver: ProductionTimeseriesResolver,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 10.0,
        allow_plain_http: bool = False,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeseries inference timeout must be positive")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._resolver = resolver
        self._client = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
        )
        self._allow_plain_http = allow_plain_http

    def close(self) -> None:
        self._client.close()

    def detect(
        self,
        context: TenantContext,
        value: TimeseriesDetectionInput,
    ) -> TimeseriesDetection:
        model = self._resolver.resolve(context)
        if not self._allow_plain_http and not model.endpoint_url.startswith("https://"):
            raise TimeseriesInferenceUnavailable(
                "timeseries_inference_requires_https",
                release_id=model.release_id,
            )
        payload = {
            "schema_version": CONTRACT_VERSION,
            "feature_snapshot_id": value.feature_snapshot_id,
            "sequence": [list(row) for row in value.sequence],
            "mask": [list(row) for row in value.mask],
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
            raise TimeseriesInferenceUnavailable(
                "timeseries_inference_timeout",
                release_id=model.release_id,
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise TimeseriesInferenceUnavailable(
                "timeseries_inference_transport_failed",
                release_id=model.release_id,
            ) from exc
        if not isinstance(body, dict):
            raise TimeseriesInferenceUnavailable(
                "timeseries_inference_response_invalid",
                release_id=model.release_id,
            )
        try:
            score = float(body["anomaly_score"])
            threshold = float(body["threshold"])
            explanations = tuple(str(item) for item in body["explanation_codes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TimeseriesInferenceUnavailable(
                "timeseries_inference_response_invalid",
                release_id=model.release_id,
            ) from exc
        if (
            body.get("feature_snapshot_id") != value.feature_snapshot_id
            or body.get("release_id") != model.release_id
            or body.get("manifest_hash") != model.manifest_hash
            or body.get("component_model_id") != model.component_model_id
            or body.get("artifact_content_hash") != model.artifact_content_hash
            or threshold != model.threshold
            or not math.isfinite(score)
            or score < 0
            or not explanations
        ):
            raise TimeseriesInferenceUnavailable(
                "timeseries_inference_response_binding_failed",
                release_id=model.release_id,
            )
        return TimeseriesDetection(
            model_release_id=model.release_id,
            component_model_id=model.component_model_id,
            detector_kind="TIMESERIES_TRANSFORMER",
            anomaly_score=score,
            threshold=threshold,
            threshold_policy_id=model.threshold_policy_id,
            explanation_codes=explanations,
        )


def _digest_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{sha256(payload).hexdigest()}"
