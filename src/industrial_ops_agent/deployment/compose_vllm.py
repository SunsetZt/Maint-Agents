"""Observed provider for the persistent project-staging Compose/vLLM endpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from industrial_ops_agent.deployment.service import (
    DeploymentProviderUnavailable,
    ProviderResult,
    ProviderTarget,
)

_PROVIDER = "COMPOSE_VLLM_GATEWAY"
_SUPPORTED_STAGE_TRAFFIC = {"SHADOW": 0.0, "PRODUCTION": 100.0}


@dataclass(slots=True)
class ComposeVllmGatewayProvider:
    """Verify exact immutable bindings exposed by the internal model router.

    The provider never starts a container and never invents rollout telemetry.  It
    observes the endpoint assembled by the packaging workflow and reports READY only
    when the router proves that both vLLM services match the requested Release.
    """

    timeout_seconds: float = 10.0
    expected_endpoint_url: str | None = None
    client: httpx.Client | None = None
    provider_kind: str = field(default=_PROVIDER, init=False)
    _owns_client: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.timeout_seconds > 60:
            raise ValueError("Compose/vLLM provider timeout must be between 0 and 60 seconds")
        if self.expected_endpoint_url is not None:
            _readiness_url(self.expected_endpoint_url)
        if self.client is None:
            self.client = httpx.Client(
                follow_redirects=False,
                timeout=httpx.Timeout(self.timeout_seconds),
            )
            self._owns_client = True

    def close(self) -> None:
        if self._owns_client and self.client is not None:
            self.client.close()

    def reconcile(self, target: ProviderTarget) -> ProviderResult:
        if target.provider != _PROVIDER:
            raise DeploymentProviderUnavailable("deployment_provider_mismatch")
        desired = target.desired_spec
        if (
            desired.get("schema_version") != "ioap-compose-vllm-rollout/v1"
            or desired.get("target_environment") != "STAGING"
        ):
            raise DeploymentProviderUnavailable("compose_vllm_desired_spec_invalid")
        expected_traffic = _SUPPORTED_STAGE_TRAFFIC.get(target.desired_stage)
        if expected_traffic is None or target.desired_traffic_percent != expected_traffic:
            raise DeploymentProviderUnavailable("compose_vllm_stage_not_supported")
        endpoint_url = desired.get("endpoint_url")
        if not isinstance(endpoint_url, str):
            raise DeploymentProviderUnavailable("compose_vllm_endpoint_binding_missing")
        if self.expected_endpoint_url is not None and endpoint_url != self.expected_endpoint_url:
            raise DeploymentProviderUnavailable("compose_vllm_endpoint_not_allowlisted")
        readiness_url = _readiness_url(endpoint_url)
        try:
            assert self.client is not None
            response = self.client.get(
                readiness_url,
                headers={"Accept": "application/json"},
            )
        except httpx.TimeoutException as exc:
            raise DeploymentProviderUnavailable("compose_vllm_endpoint_timeout") from exc
        except httpx.HTTPError as exc:
            raise DeploymentProviderUnavailable("compose_vllm_endpoint_unavailable") from exc
        if 300 <= response.status_code < 400:
            raise DeploymentProviderUnavailable("compose_vllm_endpoint_redirect_rejected")
        if response.status_code != 200:
            return _not_ready(target, endpoint_url, "compose_vllm_endpoint_not_ready")
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError):
            return _not_ready(target, endpoint_url, "compose_vllm_readiness_json_invalid")
        if not isinstance(payload, dict) or payload.get("status") not in ("READY", "ACCEPTING"):
            return _not_ready(target, endpoint_url, "compose_vllm_endpoint_not_ready")
        reason = _binding_mismatch_reason(payload, target)
        if reason is not None:
            return _not_ready(target, endpoint_url, reason)
        if payload["status"] == "ACCEPTING" or "lifecycle" in payload:
            lifecycle = payload.get("lifecycle")
            if not isinstance(lifecycle, dict):
                return _not_ready(target, endpoint_url, "compose_vllm_lifecycle_invalid")
            for name in ("diagnosis", "vlm"):
                row = lifecycle.get(name)
                if (not isinstance(row, dict)
                        or row.get("schema_version") != "ioap-owned-model-runtime/v1"
                        or row.get("prepared") is not True
                        or row.get("state") not in ("UNLOADED", "LOADING", "READY", "UNLOADING")
                        or row.get("model_id") != payload["models"].get(name)
                        or row.get("package_digest") != payload["deployment_binding"]["package_digests"].get(name)):
                    return _not_ready(target, endpoint_url, "compose_vllm_lifecycle_invalid")
            loaded = all(lifecycle[name]["state"] == "READY" for name in ("diagnosis", "vlm"))
            if (payload["status"] == "READY") != loaded:
                return _not_ready(target, endpoint_url, "compose_vllm_lifecycle_invalid")
        revision_payload = {
            "deployment_binding": payload["deployment_binding"],
            "models": payload["models"],
            "desired_spec_hash": target.desired_spec_hash,
        }
        revision = sha256(
            json.dumps(
                revision_payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return ProviderResult(
            ready=True,
            observed_stage=target.desired_stage,
            observed_traffic_percent=target.desired_traffic_percent,
            applied_spec_hash=target.desired_spec_hash,
            provider_revision=f"compose-vllm:{revision}",
            endpoint_url=endpoint_url,
        )


def _readiness_url(endpoint_url: str) -> str:
    parsed = urlsplit(endpoint_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/v1"
    ):
        raise DeploymentProviderUnavailable("compose_vllm_endpoint_binding_invalid")
    try:
        port = parsed.port
    except ValueError as exc:
        raise DeploymentProviderUnavailable("compose_vllm_endpoint_binding_invalid") from exc
    if port is not None and not 1 <= port <= 65535:
        raise DeploymentProviderUnavailable("compose_vllm_endpoint_binding_invalid")
    return urlunsplit((parsed.scheme, parsed.netloc, "/health/ready", "", ""))


def _binding_mismatch_reason(payload: dict[str, Any], target: ProviderTarget) -> str | None:
    desired = target.desired_spec
    models = payload.get("models")
    binding = payload.get("deployment_binding")
    runtime_images = desired.get("runtime_images")
    packages = desired.get("packages")
    if not isinstance(runtime_images, dict) or not isinstance(packages, dict):
        return "compose_vllm_release_binding_mismatch"
    if not isinstance(models, dict) or models != desired.get("models"):
        return "compose_vllm_model_binding_mismatch"
    expected_binding = {
        "release_id": target.release_id,
        "manifest_hash": target.manifest_hash,
        "target_environment": "STAGING",
        "serving_image_digests": {
            "diagnosis": runtime_images.get("diagnosis"),
            "vlm": runtime_images.get("vlm"),
        },
        "router_image_digest": runtime_images.get("router"),
        "package_digests": packages,
    }
    if not isinstance(binding, dict) or binding != expected_binding:
        return "compose_vllm_release_binding_mismatch"
    return None


def _not_ready(target: ProviderTarget, endpoint_url: str, reason: str) -> ProviderResult:
    return ProviderResult(
        ready=False,
        observed_stage=target.desired_stage,
        observed_traffic_percent=target.desired_traffic_percent,
        applied_spec_hash=None,
        provider_revision=None,
        endpoint_url=endpoint_url,
        reason_code=reason,
    )
