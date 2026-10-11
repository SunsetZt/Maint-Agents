"""Observed KServe provider for actual stable/candidate rollout execution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Literal, cast

from industrial_ops_agent.deployment.kserve import (
    KubernetesApiError,
    KubernetesResourceApi,
)
from industrial_ops_agent.deployment.service import (
    DeploymentProviderUnavailable,
    ProviderResult,
    ProviderTarget,
)
from industrial_ops_agent.training.gpu_promotion_security import (
    hardened_container_security_context,
)

RolloutStage = Literal["SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"]
Component = Literal["LLM", "TTS", "EMBEDDING"]
Variant = Literal["stable", "candidate"]
VARIANTS: tuple[Variant, Variant] = ("stable", "candidate")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ObservedKServeComponent:
    """Immutable local KServe coordinates for one governed candidate."""

    component: Component
    namespace: str
    gateway_name: str
    hostname: str
    route_name: str
    route_path_prefix: str
    endpoint_path: str
    config_map_name: str
    stable_service_name: str
    candidate_service_name: str
    proxy_image: str
    proxy_image_digest: str
    host_gateway: str
    host_port: int
    variant_header: str
    request_timeout_seconds: int = 120

    def __post_init__(self) -> None:
        names = (
            self.namespace,
            self.gateway_name,
            self.hostname,
            self.route_name,
            self.config_map_name,
            self.stable_service_name,
            self.candidate_service_name,
            self.proxy_image,
            self.host_gateway,
            self.variant_header,
        )
        if any(
            not value.strip() or any(character.isspace() for character in value)
            for value in names
        ):
            raise ValueError("observed_kserve_component_coordinates_are_invalid")
        if (
            not self.route_path_prefix.startswith("/")
            or not self.endpoint_path.startswith(self.route_path_prefix)
            or not 1 <= self.host_port <= 65_535
            or not 1 <= self.request_timeout_seconds <= 300
            or not _SHA256.fullmatch(self.proxy_image_digest)
        ):
            raise ValueError("observed_kserve_component_coordinates_are_invalid")


@dataclass(frozen=True, slots=True)
class ObservedKServeSnapshot:
    component: Component
    stage: RolloutStage
    ready: bool
    route_document_sha256: str
    observed_route_sha256: str
    stable_resource_sha256: str
    candidate_resource_sha256: str
    provider_revision: str
    route_resource_version: str
    stable_resource_version: str
    candidate_resource_version: str
    reason_code: str | None


@dataclass(slots=True)
class ObservedKServeProvider:
    """Apply proxy resources and reconcile the release FSM from observed KServe state."""

    api: KubernetesResourceApi
    component: ObservedKServeComponent
    proxy_source: str
    observations: list[ObservedKServeSnapshot] = field(default_factory=list)

    def reconcile(self, target: ProviderTarget) -> ProviderResult:
        stage = _stage(target.desired_stage)
        desired_route = route_document(self.component, stage)
        try:
            self._apply_static_resources(target)
            self.api.apply(
                api_version="gateway.networking.k8s.io/v1",
                plural="httproutes",
                namespace=self.component.namespace,
                name=self.component.route_name,
                document=desired_route,
            )
            stable = self.api.get(
                api_version="serving.kserve.io/v1beta1",
                plural="inferenceservices",
                namespace=self.component.namespace,
                name=self.component.stable_service_name,
            )
            candidate = self.api.get(
                api_version="serving.kserve.io/v1beta1",
                plural="inferenceservices",
                namespace=self.component.namespace,
                name=self.component.candidate_service_name,
            )
            route = self.api.get(
                api_version="gateway.networking.k8s.io/v1",
                plural="httproutes",
                namespace=self.component.namespace,
                name=self.component.route_name,
            )
        except KubernetesApiError as exc:
            raise DeploymentProviderUnavailable(exc.reason_code) from exc

        reason = _mismatch_reason(
            target,
            self.component,
            stage,
            desired_route,
            stable,
            candidate,
            route,
        )
        ready = reason is None
        revision = "observed-kserve-" + _digest(
            {
                "component": self.component.component,
                "stage": stage,
                "desired_spec_hash": target.desired_spec_hash,
                "stable": _resource_version(stable),
                "candidate": _resource_version(candidate),
                "route": _resource_version(route),
                "ready": ready,
            }
        )[:24]
        self.observations.append(
            ObservedKServeSnapshot(
                component=self.component.component,
                stage=stage,
                ready=ready,
                route_document_sha256=_digest(desired_route),
                observed_route_sha256=_digest(route),
                stable_resource_sha256=_digest(stable),
                candidate_resource_sha256=_digest(candidate),
                provider_revision=revision,
                route_resource_version=_resource_version(route),
                stable_resource_version=_resource_version(stable),
                candidate_resource_version=_resource_version(candidate),
                reason_code=reason,
            )
        )
        return ProviderResult(
            ready=ready,
            observed_stage=stage,
            observed_traffic_percent=(target.desired_traffic_percent if ready else 0.0),
            applied_spec_hash=target.desired_spec_hash if ready else None,
            provider_revision=revision,
            endpoint_url=(
                f"http://{self.component.hostname}{self.component.endpoint_path}"
                if ready
                else None
            ),
            reason_code=reason,
        )

    def _apply_static_resources(self, target: ProviderTarget) -> None:
        config_map = proxy_config_map_document(self.component, self.proxy_source)
        self.api.apply(
            api_version="v1",
            plural="configmaps",
            namespace=self.component.namespace,
            name=self.component.config_map_name,
            document=config_map,
        )
        for variant in VARIANTS:
            document = inference_service_document(self.component, variant, target)
            name = (
                self.component.stable_service_name
                if variant == "stable"
                else self.component.candidate_service_name
            )
            self.api.apply(
                api_version="serving.kserve.io/v1beta1",
                plural="inferenceservices",
                namespace=self.component.namespace,
                name=name,
                document=document,
            )


def proxy_config_map_document(
    component: ObservedKServeComponent,
    proxy_source: str,
) -> dict[str, Any]:
    if not proxy_source.strip():
        raise ValueError("observed_kserve_proxy_source_is_empty")
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": component.config_map_name,
            "namespace": component.namespace,
            "labels": _labels(component),
        },
        "data": {"variant_kserve_proxy.py": proxy_source},
    }


def inference_service_document(
    component: ObservedKServeComponent,
    variant: Variant,
    target: ProviderTarget,
) -> dict[str, Any]:
    name = (
        component.stable_service_name if variant == "stable" else component.candidate_service_name
    )
    labels = {
        **_labels(component),
        "ioap.openai.com/model-variant": variant,
        "ioap.openai.com/release-hash": target.manifest_hash[:63],
    }
    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": name,
            "namespace": component.namespace,
            "annotations": {
                "serving.kserve.io/deploymentMode": "Standard",
                "ioap.openai.com/desired-spec-sha256": target.desired_spec_hash,
            },
            "labels": labels,
        },
        "spec": {
            "predictor": {
                "minReplicas": 1,
                "maxReplicas": 1,
                "containers": [
                    {
                        "name": "kserve-container",
                        "image": component.proxy_image,
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["python", "-u", "/app/variant_kserve_proxy.py"],
                        "env": [
                            {"name": "PORT", "value": "8080"},
                            {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                            {"name": "PYTHONPYCACHEPREFIX", "value": "/tmp/pycache"},
                            {"name": "IOAP_VARIANT", "value": variant},
                            {
                                "name": "IOAP_UPSTREAM_URL",
                                "value": f"http://{component.host_gateway}:{component.host_port}",
                            },
                            {"name": "IOAP_ENDPOINT_PATH", "value": component.endpoint_path},
                            {"name": "IOAP_VARIANT_HEADER", "value": component.variant_header},
                            {
                                "name": "IOAP_UPSTREAM_TIMEOUT_SECONDS",
                                "value": str(component.request_timeout_seconds),
                            },
                        ],
                        "ports": [{"name": "http1", "containerPort": 8080}],
                        "startupProbe": {
                            "httpGet": {"path": "/health/ready", "port": 8080},
                            "periodSeconds": 3,
                            "failureThreshold": 80,
                        },
                        "readinessProbe": {
                            "httpGet": {"path": "/health/ready", "port": 8080},
                            "periodSeconds": 5,
                            "failureThreshold": 6,
                        },
                        "securityContext": hardened_container_security_context(),
                        "resources": {
                            "requests": {"cpu": "25m", "memory": "48Mi"},
                            "limits": {"cpu": "250m", "memory": "192Mi"},
                        },
                        "volumeMounts": [
                            {
                                "name": "proxy-code",
                                "mountPath": "/app/variant_kserve_proxy.py",
                                "subPath": "variant_kserve_proxy.py",
                                "readOnly": True,
                            },
                            {"name": "runtime-tmp", "mountPath": "/tmp"},
                        ],
                    }
                ],
                "volumes": [
                    {"name": "proxy-code", "configMap": {"name": component.config_map_name}},
                    {"name": "runtime-tmp", "emptyDir": {}},
                ],
            }
        },
    }


def route_document(
    component: ObservedKServeComponent,
    stage: RolloutStage,
) -> dict[str, Any]:
    stable = {"name": f"{component.stable_service_name}-predictor", "port": 80}
    candidate = {"name": f"{component.candidate_service_name}-predictor", "port": 80}
    if stage == "SHADOW":
        rule: dict[str, Any] = {
            "backendRefs": [{**stable, "weight": 100}],
            "filters": [
                {"type": "RequestMirror", "requestMirror": {"backendRef": candidate}}
            ],
        }
    elif stage in {"CANARY_5", "CANARY_25"}:
        candidate_weight = 5 if stage == "CANARY_5" else 25
        rule = {
            "backendRefs": [
                {**stable, "weight": 100 - candidate_weight},
                {**candidate, "weight": candidate_weight},
            ]
        }
    elif stage == "ROLLED_BACK":
        rule = {"backendRefs": [{**stable, "weight": 100}]}
    else:  # pragma: no cover - closed literal
        raise ValueError("observed_kserve_stage_is_invalid")
    rule["matches"] = [
        {"path": {"type": "PathPrefix", "value": component.route_path_prefix}}
    ]
    rule["timeouts"] = {
        "request": f"{component.request_timeout_seconds}s",
        "backendRequest": f"{component.request_timeout_seconds}s",
    }
    return {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "HTTPRoute",
        "metadata": {
            "name": component.route_name,
            "namespace": component.namespace,
            "annotations": {
                "ioap.openai.com/stage": stage,
                "ioap.openai.com/component": component.component.lower(),
            },
            "labels": _labels(component),
        },
        "spec": {
            "parentRefs": [{"name": component.gateway_name}],
            "hostnames": [component.hostname],
            "rules": [rule],
        },
    }


def route_rules_semantically_equal(
    observed: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    return _normalize_route_rules(observed) == _normalize_route_rules(expected)


def route_condition_is_true(document: dict[str, Any], condition_type: str) -> bool:
    parents = document.get("status", {}).get("parents", [])
    if not isinstance(parents, list):
        return False
    return any(
        isinstance(condition, dict)
        and condition.get("type") == condition_type
        and condition.get("status") == "True"
        for parent in parents
        if isinstance(parent, dict)
        for condition in (
            parent.get("conditions", []) if isinstance(parent.get("conditions"), list) else []
        )
    )


def resource_ready(document: dict[str, Any]) -> bool:
    conditions = document.get("status", {}).get("conditions", [])
    return bool(
        isinstance(conditions, list)
        and any(
            isinstance(condition, dict)
            and condition.get("type") == "Ready"
            and condition.get("status") == "True"
            for condition in conditions
        )
    )


def _mismatch_reason(
    target: ProviderTarget,
    component: ObservedKServeComponent,
    stage: RolloutStage,
    desired_route: dict[str, Any],
    stable: dict[str, Any],
    candidate: dict[str, Any],
    route: dict[str, Any],
) -> str | None:
    if (
        target.namespace != component.namespace
        or target.route_name != component.route_name
        or target.stable_service_name != component.stable_service_name
    ):
        return "observed_kserve_target_coordinates_mismatch"
    if not _component_present(target.desired_spec, component.component):
        return "observed_kserve_desired_component_missing"
    if not resource_ready(stable) or not resource_ready(candidate):
        return "observed_kserve_inference_service_not_ready"
    if not route_condition_is_true(route, "Accepted"):
        return "observed_kserve_route_not_accepted"
    if not route_condition_is_true(route, "ResolvedRefs"):
        return "observed_kserve_route_refs_not_resolved"
    annotations = route.get("metadata", {}).get("annotations", {})
    if annotations.get("ioap.openai.com/stage") != stage:
        return "observed_kserve_route_stage_mismatch"
    if not route_rules_semantically_equal(route, desired_route):
        return "observed_kserve_route_rules_mismatch"
    for document, variant in ((stable, "stable"), (candidate, "candidate")):
        metadata = document.get("metadata", {})
        labels = metadata.get("labels", {}) if isinstance(metadata, dict) else {}
        service_annotations = (
            metadata.get("annotations", {}) if isinstance(metadata, dict) else {}
        )
        if (
            labels.get("ioap.openai.com/model-variant") != variant
            or labels.get("ioap.openai.com/release-hash") != target.manifest_hash[:63]
            or service_annotations.get("ioap.openai.com/desired-spec-sha256")
            != target.desired_spec_hash
        ):
            return "observed_kserve_release_binding_mismatch"
    return None


def _component_present(spec: dict[str, Any], component: Component) -> bool:
    return {
        "LLM": True,
        "TTS": "tts" in spec,
        "EMBEDDING": "embedding" in spec,
    }[component]


def _normalize_route_rules(document: dict[str, Any]) -> object | None:
    rules = document.get("spec", {}).get("rules")
    if not isinstance(rules, list):
        return None
    return _without_gateway_defaults(rules)


def _without_gateway_defaults(value: object) -> object:
    if isinstance(value, list):
        return [_without_gateway_defaults(item) for item in value]
    if isinstance(value, dict):
        normalized = {
            key: _without_gateway_defaults(item)
            for key, item in value.items()
            if not (key in {"group", "kind"} and item in {"", "Service"})
        }
        if normalized.get("weight") == 1:
            normalized.pop("weight")
        return normalized
    return value


def _labels(component: ObservedKServeComponent) -> dict[str, str]:
    return {
        "app.kubernetes.io/managed-by": "industrial-ops-agent",
        "ioap.openai.com/acceptance": "enterprise-candidate-kserve-v1",
        "ioap.openai.com/component": component.component.lower(),
        "ioap.openai.com/runtime-image-digest": component.proxy_image_digest.removeprefix(
            "sha256:"
        )[:63],
    }


def _stage(value: str) -> RolloutStage:
    if value not in {"SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"}:
        raise DeploymentProviderUnavailable("observed_kserve_stage_is_not_supported")
    return cast(RolloutStage, value)


def _resource_version(document: dict[str, Any]) -> str:
    value = document.get("metadata", {}).get("resourceVersion")
    return str(value) if value is not None else "missing"


def _digest(document: object) -> str:
    return sha256(
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
