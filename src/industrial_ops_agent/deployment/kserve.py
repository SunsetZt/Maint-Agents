"""KServe Standard-mode and Gateway API deployment provider."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import quote

import httpx

from industrial_ops_agent.deployment.service import (
    KEDA_AUTOSCALING_POLICY_VERSION,
    KEDA_PROMETHEUS_SERVER_ADDRESS,
    KEDA_SCALE_DOWN_MAX_PODS_PER_TWO_MINUTES,
    KEDA_SCALE_DOWN_STABILIZATION_SECONDS,
    KEDA_SCALE_UP_MAX_PODS_PER_MINUTE,
    KEDA_SCALE_UP_STABILIZATION_SECONDS,
    KEDA_TARGET_TYPE,
    KEDA_VLLM_METRIC_NAME,
    DeploymentProviderUnavailable,
    ProviderResult,
    ProviderTarget,
    autoscaling_metric_query,
)
from industrial_ops_agent.training.gpu_promotion_security import (
    hardened_container_security_context,
)


class KubernetesApiError(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class KubernetesResourceApi(Protocol):
    def apply(
        self,
        *,
        api_version: str,
        plural: str,
        namespace: str,
        name: str,
        document: dict[str, Any],
    ) -> dict[str, Any]: ...

    def get(
        self,
        *,
        api_version: str,
        plural: str,
        namespace: str,
        name: str,
    ) -> dict[str, Any]: ...

class KubernetesHttpApi:
    """Minimal server-side-apply client using an in-cluster service account."""

    def __init__(
        self,
        api_url: str,
        bearer_token: str,
        *,
        ca_file: str | bool = True,
        timeout_seconds: float = 10.0,
        field_manager: str = "industrial-ops-release-controller",
    ) -> None:
        if not api_url.startswith("https://") or not bearer_token:
            raise ValueError("a TLS Kubernetes API URL and bearer token are required")
        self._api_url = api_url.rstrip("/")
        self._field_manager = field_manager
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {bearer_token}"},
            verify=ca_file,
            timeout=httpx.Timeout(timeout_seconds),
        )

    def apply(
        self,
        *,
        api_version: str,
        plural: str,
        namespace: str,
        name: str,
        document: dict[str, Any],
    ) -> dict[str, Any]:
        response = self._request(
            "PATCH",
            self._resource_path(api_version, plural, namespace, name),
            params={"fieldManager": self._field_manager, "force": "true"},
            content=json.dumps(document, separators=(",", ":")),
            headers={"Content-Type": "application/apply-patch+yaml"},
        )
        return _json_object(response)

    def get(
        self,
        *,
        api_version: str,
        plural: str,
        namespace: str,
        name: str,
    ) -> dict[str, Any]:
        response = self._request(
            "GET",
            self._resource_path(api_version, plural, namespace, name),
        )
        return _json_object(response)

    def create(
        self,
        *,
        api_version: str,
        plural: str,
        namespace: str,
        document: dict[str, Any],
    ) -> dict[str, Any]:
        response = self._request(
            "POST",
            self._collection_path(api_version, plural, namespace),
            content=json.dumps(document, separators=(",", ":")),
            headers={"Content-Type": "application/json"},
        )
        return _json_object(response)

    def replace(
        self,
        *,
        api_version: str,
        plural: str,
        namespace: str,
        name: str,
        document: dict[str, Any],
    ) -> dict[str, Any]:
        response = self._request(
            "PUT",
            self._resource_path(api_version, plural, namespace, name),
            content=json.dumps(document, separators=(",", ":")),
            headers={"Content-Type": "application/json"},
        )
        return _json_object(response)

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, f"{self._api_url}{path}", **kwargs)
            response.raise_for_status()
            return response
        except httpx.TimeoutException as exc:
            raise KubernetesApiError("kubernetes_api_timeout") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            reason = (
                "kubernetes_api_forbidden"
                if status in {401, 403}
                else "kubernetes_api_not_found"
                if status == 404
                else "kubernetes_api_conflict"
                if status == 409
                else "kubernetes_api_rejected_resource"
                if status in {400, 422}
                else "kubernetes_api_unavailable"
            )
            raise KubernetesApiError(reason) from exc
        except httpx.HTTPError as exc:
            raise KubernetesApiError("kubernetes_api_unavailable") from exc

    @staticmethod
    def _resource_path(
        api_version: str,
        plural: str,
        namespace: str,
        name: str,
    ) -> str:
        if "/" not in api_version:
            raise ValueError("custom resource api_version must include a group")
        group, version = api_version.split("/", 1)
        values = [group, version, namespace, plural, name]
        if any(not value for value in values):
            raise ValueError("Kubernetes resource coordinates are incomplete")
        encoded = [quote(value, safe=".-") for value in values]
        return f"/apis/{encoded[0]}/{encoded[1]}/namespaces/{encoded[2]}/{encoded[3]}/{encoded[4]}"

    @staticmethod
    def _collection_path(api_version: str, plural: str, namespace: str) -> str:
        if "/" not in api_version:
            raise ValueError("custom resource api_version must include a group")
        group, version = api_version.split("/", 1)
        values = [group, version, namespace, plural]
        if any(not value for value in values):
            raise ValueError("Kubernetes resource coordinates are incomplete")
        encoded = [quote(value, safe=".-") for value in values]
        return f"/apis/{encoded[0]}/{encoded[1]}/namespaces/{encoded[2]}/{encoded[3]}"


@dataclass(slots=True)
class KubectlResourceApi:
    """Server-side-apply Kubernetes adapter for controlled local acceptance runs."""

    context: str
    working_directory: Path
    field_manager: str = "industrial-ops-release-controller"
    apply_timeout_seconds: int = 60
    get_timeout_seconds: int = 30

    def apply(
        self,
        *,
        api_version: str,
        plural: str,
        namespace: str,
        name: str,
        document: dict[str, Any],
    ) -> dict[str, Any]:
        del api_version, plural, namespace, name
        payload = json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._run(
            [
                "kubectl",
                "--context",
                self.context,
                "apply",
                "--server-side",
                f"--field-manager={self.field_manager}",
                "--force-conflicts",
                "--filename=-",
            ],
            timeout_seconds=self.apply_timeout_seconds,
            input_text=payload,
            reason_code="kubectl_apply_failed",
        )
        return cast(dict[str, Any], json.loads(payload))

    def get(
        self,
        *,
        api_version: str,
        plural: str,
        namespace: str,
        name: str,
    ) -> dict[str, Any]:
        del api_version
        output = self._run(
            [
                "kubectl",
                "--context",
                self.context,
                "--namespace",
                namespace,
                "get",
                plural,
                name,
                "--output=json",
            ],
            timeout_seconds=self.get_timeout_seconds,
            input_text=None,
            reason_code="kubectl_get_failed",
        )
        try:
            value = json.loads(output)
        except json.JSONDecodeError as exc:
            raise KubernetesApiError("kubectl_get_json_invalid") from exc
        if not isinstance(value, dict):
            raise KubernetesApiError("kubectl_get_json_invalid")
        return cast(dict[str, Any], value)

    def _run(
        self,
        argv: list[str],
        *,
        timeout_seconds: int,
        input_text: str | None,
        reason_code: str,
    ) -> str:
        try:
            result = subprocess.run(
                argv,
                cwd=self.working_directory,
                input=input_text,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise KubernetesApiError(f"{reason_code}_timeout") from exc
        except OSError as exc:
            raise KubernetesApiError("kubectl_unavailable") from exc
        if result.returncode != 0:
            raise KubernetesApiError(reason_code)
        return result.stdout


@dataclass(slots=True)
class KServeGatewayProvider:
    api: KubernetesResourceApi
    service_port: int = 80

    def reconcile(self, target: ProviderTarget) -> ProviderResult:
        _validated_autoscaling(target)
        observed_pool: dict[str, Any] | None = None
        try:
            runtime = _serving_runtime(target)
            self.api.apply(
                api_version="serving.kserve.io/v1alpha1",
                plural="servingruntimes",
                namespace=target.namespace,
                name=runtime["metadata"]["name"],
                document=runtime,
            )
            inference_service = _inference_service(target)
            self.api.apply(
                api_version="serving.kserve.io/v1beta1",
                plural="inferenceservices",
                namespace=target.namespace,
                name=target.service_name,
                document=inference_service,
            )
            observed_service = self.api.get(
                api_version="serving.kserve.io/v1beta1",
                plural="inferenceservices",
                namespace=target.namespace,
                name=target.service_name,
            )
            if not _resource_ready(observed_service, ("Ready",)):
                return _pending_result(target, observed_service, "kserve_inference_not_ready")

            observed_components: list[dict[str, Any]] = []
            if "embedding" in target.desired_spec:
                embedding_runtime = _embedding_serving_runtime(target)
                self.api.apply(
                    api_version="serving.kserve.io/v1alpha1",
                    plural="servingruntimes",
                    namespace=target.namespace,
                    name=embedding_runtime["metadata"]["name"],
                    document=embedding_runtime,
                )
                embedding_service = _embedding_inference_service(target)
                embedding_name = str(target.desired_spec["embedding"]["service_name"])
                self.api.apply(
                    api_version="serving.kserve.io/v1beta1",
                    plural="inferenceservices",
                    namespace=target.namespace,
                    name=embedding_name,
                    document=embedding_service,
                )
                observed_embedding = self.api.get(
                    api_version="serving.kserve.io/v1beta1",
                    plural="inferenceservices",
                    namespace=target.namespace,
                    name=embedding_name,
                )
                if not _resource_ready(observed_embedding, ("Ready",)):
                    return _pending_result(
                        target,
                        observed_embedding,
                        "kserve_embedding_inference_not_ready",
                    )
                observed_components.append(observed_embedding)

            if "reranker" in target.desired_spec:
                reranker_runtime = _reranker_serving_runtime(target)
                self.api.apply(
                    api_version="serving.kserve.io/v1alpha1",
                    plural="servingruntimes",
                    namespace=target.namespace,
                    name=reranker_runtime["metadata"]["name"],
                    document=reranker_runtime,
                )
                reranker_service = _reranker_inference_service(target)
                reranker_name = str(target.desired_spec["reranker"]["service_name"])
                self.api.apply(
                    api_version="serving.kserve.io/v1beta1",
                    plural="inferenceservices",
                    namespace=target.namespace,
                    name=reranker_name,
                    document=reranker_service,
                )
                observed_reranker = self.api.get(
                    api_version="serving.kserve.io/v1beta1",
                    plural="inferenceservices",
                    namespace=target.namespace,
                    name=reranker_name,
                )
                if not _resource_ready(observed_reranker, ("Ready",)):
                    return _pending_result(
                        target,
                        observed_reranker,
                        "kserve_reranker_inference_not_ready",
                    )
                observed_components.append(observed_reranker)

            if "tts" in target.desired_spec:
                tts_runtime = _tts_serving_runtime(target)
                self.api.apply(
                    api_version="serving.kserve.io/v1alpha1",
                    plural="servingruntimes",
                    namespace=target.namespace,
                    name=tts_runtime["metadata"]["name"],
                    document=tts_runtime,
                )
                tts_service = _tts_inference_service(target)
                tts_name = str(target.desired_spec["tts"]["service_name"])
                self.api.apply(
                    api_version="serving.kserve.io/v1beta1",
                    plural="inferenceservices",
                    namespace=target.namespace,
                    name=tts_name,
                    document=tts_service,
                )
                observed_tts = self.api.get(
                    api_version="serving.kserve.io/v1beta1",
                    plural="inferenceservices",
                    namespace=target.namespace,
                    name=tts_name,
                )
                if not _resource_ready(observed_tts, ("Ready",)):
                    return _pending_result(
                        target,
                        observed_tts,
                        "kserve_tts_inference_not_ready",
                    )
                observed_components.append(observed_tts)

            if "timeseries" in target.desired_spec:
                timeseries_runtime = _timeseries_serving_runtime(target)
                self.api.apply(
                    api_version="serving.kserve.io/v1alpha1",
                    plural="servingruntimes",
                    namespace=target.namespace,
                    name=timeseries_runtime["metadata"]["name"],
                    document=timeseries_runtime,
                )
                timeseries_service = _timeseries_inference_service(target)
                timeseries_name = str(target.desired_spec["timeseries"]["service_name"])
                self.api.apply(
                    api_version="serving.kserve.io/v1beta1",
                    plural="inferenceservices",
                    namespace=target.namespace,
                    name=timeseries_name,
                    document=timeseries_service,
                )
                observed_timeseries = self.api.get(
                    api_version="serving.kserve.io/v1beta1",
                    plural="inferenceservices",
                    namespace=target.namespace,
                    name=timeseries_name,
                )
                if not _resource_ready(observed_timeseries, ("Ready",)):
                    return _pending_result(
                        target,
                        observed_timeseries,
                        "kserve_timeseries_inference_not_ready",
                    )
                observed_components.append(observed_timeseries)

            if "rul" in target.desired_spec:
                rul_runtime = _rul_serving_runtime(target)
                self.api.apply(
                    api_version="serving.kserve.io/v1alpha1",
                    plural="servingruntimes",
                    namespace=target.namespace,
                    name=rul_runtime["metadata"]["name"],
                    document=rul_runtime,
                )
                rul_service = _rul_inference_service(target)
                rul_name = str(target.desired_spec["rul"]["service_name"])
                self.api.apply(
                    api_version="serving.kserve.io/v1beta1",
                    plural="inferenceservices",
                    namespace=target.namespace,
                    name=rul_name,
                    document=rul_service,
                )
                observed_rul = self.api.get(
                    api_version="serving.kserve.io/v1beta1",
                    plural="inferenceservices",
                    namespace=target.namespace,
                    name=rul_name,
                )
                if not _resource_ready(observed_rul, ("Ready",)):
                    return _pending_result(
                        target,
                        observed_rul,
                        "kserve_rul_inference_not_ready",
                    )
                observed_components.append(observed_rul)

            pool_routing = _inference_pool_routing(target)
            if pool_routing is not None and target.desired_stage != "ROLLED_BACK":
                pool = _inference_pool(target, pool_routing)
                pool_name = str(pool_routing["inference_pool_name"])
                self.api.apply(
                    api_version="inference.networking.k8s.io/v1",
                    plural="inferencepools",
                    namespace=target.namespace,
                    name=pool_name,
                    document=pool,
                )
                observed_pool = self.api.get(
                    api_version="inference.networking.k8s.io/v1",
                    plural="inferencepools",
                    namespace=target.namespace,
                    name=pool_name,
                )
                if not _gateway_parent_ready(
                    observed_pool,
                    str(target.desired_spec["route"]["gateway_name"]),
                ):
                    return _pending_result(
                        target,
                        observed_pool,
                        "inference_pool_not_ready",
                    )

            route = _http_route(target, self.service_port)
            self.api.apply(
                api_version="gateway.networking.k8s.io/v1",
                plural="httproutes",
                namespace=target.namespace,
                name=target.route_name,
                document=route,
            )
            observed_route = self.api.get(
                api_version="gateway.networking.k8s.io/v1",
                plural="httproutes",
                namespace=target.namespace,
                name=target.route_name,
            )
            gateway_name = (
                str(target.desired_spec["route"]["gateway_name"])
                if pool_routing is not None
                else None
            )
            if not _route_ready(observed_route, gateway_name):
                return _pending_result(target, observed_route, "gateway_route_not_ready")
        except KubernetesApiError as exc:
            raise DeploymentProviderUnavailable(exc.reason_code) from exc

        # Consumers must use the stable Gateway API route. KServe's status URL points
        # at the revision service and would bypass the reviewed traffic split/rollback.
        endpoint = f"https://{target.desired_spec['route']['hostname']}"
        revision_documents = [observed_service, *observed_components]
        if observed_pool is not None:
            revision_documents.append(observed_pool)
        revision_documents.append(observed_route)
        revision = _provider_revision(*revision_documents)
        return ProviderResult(
            ready=True,
            observed_stage=target.desired_stage,
            observed_traffic_percent=target.desired_traffic_percent,
            applied_spec_hash=target.desired_spec_hash,
            provider_revision=revision,
            endpoint_url=endpoint,
        )


def _serving_runtime(target: ProviderTarget) -> dict[str, Any]:
    spec = target.desired_spec
    inference = spec["inference_config"]
    if inference.get("engine") == "llama.cpp":
        return _llama_cpp_serving_runtime(target)
    gpu_count = spec["hardware_profile"]["gpu_count"]
    args = [
        "--model",
        spec["base_model_id"],
        "--served-model-name",
        target.release_id,
        "--enable-lora",
        "--lora-modules",
        f"{target.release_id}=/mnt/models/adapter",
        "--tensor-parallel-size",
        str(inference["tensor_parallel_size"]),
        "--max-model-len",
        str(inference["max_model_len"]),
        "--dtype",
        str(inference["dtype"]),
        "--max-num-seqs",
        str(inference["max_num_seqs"]),
        "--max-lora-rank",
        str(inference["max_lora_rank"]),
    ]
    return {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": spec["serving_runtime_name"],
            "namespace": target.namespace,
            "labels": _labels(target),
        },
        "spec": {
            "supportedModelFormats": [{"name": "huggingface", "version": "1", "autoSelect": False}],
            "protocolVersions": ["v1", "v2"],
            "containers": [
                {
                    "name": "kserve-container",
                    "image": (f"{spec['runtime_image_repository']}@{spec['runtime_image_digest']}"),
                    "args": args,
                    "ports": [
                        {"name": "http", "containerPort": 8080, "protocol": "TCP"}
                    ],
                    "resources": {
                        "limits": {"nvidia.com/gpu": str(gpu_count)},
                        "requests": {"nvidia.com/gpu": str(gpu_count)},
                    },
                }
            ],
        },
    }


def _inference_service(target: ProviderTarget) -> dict[str, Any]:
    spec = target.desired_spec
    if spec["inference_config"].get("engine") == "llama.cpp":
        return _llama_cpp_inference_service(target)
    autoscaling = _validated_autoscaling(target)
    annotations = {"serving.kserve.io/deploymentMode": "Standard"}
    # KServe v0.19 defines storageUris on PredictorSpec, alongside ModelSpec.
    predictor: dict[str, Any] = {
        "serviceAccountName": spec["service_account_name"],
        "storageUris": [
            {"uri": spec["adapter_uri"], "mountPath": "/mnt/models/adapter"}
        ],
        "model": {
            "modelFormat": {"name": "huggingface", "version": "1"},
            "runtime": spec["serving_runtime_name"],
        },
    }
    if autoscaling is not None:
        metric = autoscaling["metric"]
        behavior = autoscaling["behavior"]
        annotations.update(
            {
                "serving.kserve.io/autoscalerClass": "keda",
                "serving.kserve.io/enable-prometheus-scraping": "true",
                "prometheus.io/scrape": "true",
                "prometheus.io/path": "/metrics",
                "prometheus.io/port": "8080",
                "prometheus.io/scheme": "http",
            }
        )
        predictor.update(
            {
                "minReplicas": autoscaling["min_replicas"],
                "maxReplicas": autoscaling["max_replicas"],
                "autoScaling": {
                    "metrics": [
                        {
                            "type": "External",
                            "external": {
                                "metric": {
                                    "backend": metric["backend"],
                                    "serverAddress": metric["server_address"],
                                    "query": metric["query"],
                                },
                                "target": {
                                    "type": metric["target_type"],
                                    "value": str(metric["target_value"]),
                                },
                            },
                        }
                    ],
                    "behavior": {
                        "scaleUp": {
                            "stabilizationWindowSeconds": behavior[
                                "scale_up_stabilization_seconds"
                            ],
                            "selectPolicy": "Max",
                            "policies": [
                                {
                                    "type": "Pods",
                                    "value": behavior["scale_up_max_pods_per_minute"],
                                    "periodSeconds": 60,
                                }
                            ],
                        },
                        "scaleDown": {
                            "stabilizationWindowSeconds": behavior[
                                "scale_down_stabilization_seconds"
                            ],
                            "selectPolicy": "Min",
                            "policies": [
                                {
                                    "type": "Pods",
                                    "value": behavior[
                                        "scale_down_max_pods_per_two_minutes"
                                    ],
                                    "periodSeconds": 120,
                                }
                            ],
                        },
                    },
                },
            }
        )
    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": target.service_name,
            "namespace": target.namespace,
            "labels": _primary_labels(target),
            "annotations": annotations,
        },
        "spec": {"predictor": predictor},
    }


def _llama_cpp_serving_runtime(target: ProviderTarget) -> dict[str, Any]:
    spec = target.desired_spec
    inference = spec["inference_config"]
    model_file = str(spec["model_file"])
    model_mount_subpath = spec.get("model_mount_subpath")
    if model_mount_subpath is None:
        model_path = f"/mnt/models/gguf/{model_file}"
    elif model_mount_subpath in {"stable", "candidate"}:
        model_path = f"/mnt/models/gguf/{model_mount_subpath}/{model_file}"
    else:
        raise ValueError("model_mount_subpath must identify stable or candidate")
    args = [
        "--model",
        model_path,
        "--alias",
        target.release_id,
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--ctx-size",
        str(inference["context_size"]),
        "--threads",
        str(inference["threads"]),
        "--batch-size",
        str(inference["batch_size"]),
        "--metrics",
    ]
    if inference["mlock"]:
        args.append("--mlock")
    if not inference["mmap"]:
        args.append("--no-mmap")
    hardware = spec["hardware_profile"]
    resources = {
        "requests": {
            "cpu": str(hardware["cpu_cores"]),
            "memory": f"{hardware['memory_gb']}Gi",
        },
        "limits": {
            "cpu": str(hardware["cpu_cores"]),
            "memory": f"{hardware['memory_gb']}Gi",
        },
    }
    local_image_reference = spec.get("runtime_image_reference")
    if local_image_reference is None:
        image = f"{spec['runtime_image_repository']}@{spec['runtime_image_digest']}"
        image_pull_policy = "IfNotPresent"
    elif isinstance(local_image_reference, str) and local_image_reference:
        image = local_image_reference
        image_pull_policy = "Never"
    else:
        raise ValueError("runtime_image_reference must be a non-empty string")
    return {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": spec["serving_runtime_name"],
            "namespace": target.namespace,
            "labels": _labels(target),
        },
        "spec": {
            "supportedModelFormats": [{"name": "gguf", "version": "1", "autoSelect": False}],
            "protocolVersions": ["v1"],
            "containers": [
                {
                    "name": "kserve-container",
                    "image": image,
                    "imagePullPolicy": image_pull_policy,
                    "args": args,
                    "ports": [
                        {
                            "name": "http1",
                            "containerPort": 8080,
                            "protocol": "TCP",
                        }
                    ],
                    "resources": resources,
                    "securityContext": hardened_container_security_context(),
                    "volumeMounts": [
                        {"name": "runtime-tmp", "mountPath": "/tmp"},
                    ],
                    "readinessProbe": {
                        "httpGet": {"path": "/health", "port": 8080},
                        "initialDelaySeconds": 5,
                        "periodSeconds": 10,
                        "failureThreshold": 30,
                    },
                }
            ],
            "volumes": [{"name": "runtime-tmp", "emptyDir": {}}],
        },
    }


def _llama_cpp_inference_service(target: ProviderTarget) -> dict[str, Any]:
    spec = target.desired_spec
    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": target.service_name,
            "namespace": target.namespace,
            "labels": _primary_labels(target),
            "annotations": {"serving.kserve.io/deploymentMode": "Standard"},
        },
        "spec": {
            "predictor": {
                "serviceAccountName": spec["service_account_name"],
                "storageUris": [
                    {"uri": spec["model_uri"], "mountPath": "/mnt/models/gguf"}
                ],
                "model": {
                    "modelFormat": {"name": "gguf", "version": "1"},
                    "runtime": spec["serving_runtime_name"],
                },
            }
        },
    }


def _timeseries_serving_runtime(target: ProviderTarget) -> dict[str, Any]:
    spec = target.desired_spec["timeseries"]
    inference = spec["inference_config"]
    gpu_count = spec["hardware_profile"]["gpu_count"]
    args = [
        "--model-dir",
        "/mnt/models/timeseries",
        "--release-id",
        target.release_id,
        "--manifest-hash",
        target.manifest_hash,
        "--component-model-id",
        str(spec["component_model_id"]),
        "--artifact-content-hash",
        str(spec["artifact_content_hash"]),
        "--precision",
        str(inference["precision"]),
        "--anomaly-threshold",
        str(inference["anomaly_threshold"]),
        "--port",
        "8080",
    ]
    return {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": spec["serving_runtime_name"],
            "namespace": target.namespace,
            "labels": _timeseries_labels(target),
        },
        "spec": {
            "supportedModelFormats": [
                {"name": "industrial-timeseries", "version": "1", "autoSelect": False}
            ],
            "protocolVersions": ["v1"],
            "containers": [
                {
                    "name": "kserve-container",
                    "image": (f"{spec['runtime_image_repository']}@{spec['runtime_image_digest']}"),
                    "args": args,
                    "ports": [{"name": "http", "containerPort": 8080, "protocol": "TCP"}],
                    "resources": {
                        "limits": {"nvidia.com/gpu": str(gpu_count)},
                        "requests": {"nvidia.com/gpu": str(gpu_count)},
                    },
                }
            ],
        },
    }


def _rul_serving_runtime(target: ProviderTarget) -> dict[str, Any]:
    spec = target.desired_spec["rul"]
    inference = spec["inference_config"]
    gpu_count = spec["hardware_profile"]["gpu_count"]
    args = [
        "--model-dir",
        "/mnt/models/rul",
        "--release-id",
        target.release_id,
        "--manifest-hash",
        target.manifest_hash,
        "--component-model-id",
        str(spec["component_model_id"]),
        "--artifact-content-hash",
        str(spec["artifact_content_hash"]),
        "--precision",
        str(inference["precision"]),
        "--port",
        "8080",
    ]
    return {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": spec["serving_runtime_name"],
            "namespace": target.namespace,
            "labels": _rul_labels(target),
        },
        "spec": {
            "supportedModelFormats": [
                {"name": "industrial-rul", "version": "1", "autoSelect": False}
            ],
            "protocolVersions": ["v1"],
            "containers": [
                {
                    "name": "kserve-container",
                    "image": (f"{spec['runtime_image_repository']}@{spec['runtime_image_digest']}"),
                    "args": args,
                    "ports": [{"name": "http", "containerPort": 8080, "protocol": "TCP"}],
                    "resources": {
                        "limits": {"nvidia.com/gpu": str(gpu_count)},
                        "requests": {"nvidia.com/gpu": str(gpu_count)},
                    },
                }
            ],
        },
    }


def _reranker_serving_runtime(target: ProviderTarget) -> dict[str, Any]:
    spec = target.desired_spec["reranker"]
    inference = spec["inference_config"]
    gpu_count = spec["hardware_profile"]["gpu_count"]
    args = [
        "--model-dir",
        "/mnt/models/reranker",
        "--release-id",
        target.release_id,
        "--manifest-hash",
        target.manifest_hash,
        "--component-model-id",
        str(spec["component_model_id"]),
        "--artifact-content-hash",
        str(spec["artifact_content_hash"]),
        "--precision",
        str(inference["precision"]),
        "--batch-size",
        str(inference["batch_size"]),
        "--max-sequence-length",
        str(inference["max_sequence_length"]),
        "--port",
        "8080",
    ]
    return {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": spec["serving_runtime_name"],
            "namespace": target.namespace,
            "labels": _reranker_labels(target),
        },
        "spec": {
            "supportedModelFormats": [
                {"name": "industrial-reranker", "version": "1", "autoSelect": False}
            ],
            "protocolVersions": ["v1"],
            "containers": [
                {
                    "name": "kserve-container",
                    "image": f"{spec['runtime_image_repository']}@{spec['runtime_image_digest']}",
                    "args": args,
                    "ports": [{"name": "http", "containerPort": 8080, "protocol": "TCP"}],
                    "securityContext": hardened_container_security_context(),
                    "resources": {
                        "limits": {
                            "cpu": "2",
                            "memory": "4Gi",
                            "nvidia.com/gpu": str(gpu_count),
                        },
                        "requests": {
                            "cpu": "250m",
                            "memory": "512Mi",
                            "nvidia.com/gpu": str(gpu_count),
                        },
                    },
                    "readinessProbe": {
                        "httpGet": {"path": "/health/ready", "port": 8080},
                        "initialDelaySeconds": 10,
                        "periodSeconds": 10,
                        "failureThreshold": 30,
                    },
                }
            ],
        },
    }


def _embedding_serving_runtime(target: ProviderTarget) -> dict[str, Any]:
    spec = target.desired_spec["embedding"]
    inference = spec["inference_config"]
    gpu_count = spec["hardware_profile"]["gpu_count"]
    args = [
        "--model-dir",
        "/mnt/models/embedding",
        "--release-id",
        target.release_id,
        "--manifest-hash",
        target.manifest_hash,
        "--component-model-id",
        str(spec["component_model_id"]),
        "--artifact-content-hash",
        str(spec["artifact_content_hash"]),
        "--precision",
        str(inference["precision"]),
        "--batch-size",
        str(inference["batch_size"]),
        "--port",
        "8080",
    ]
    return {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": spec["serving_runtime_name"],
            "namespace": target.namespace,
            "labels": _embedding_labels(target),
        },
        "spec": {
            "supportedModelFormats": [
                {"name": "industrial-embedding", "version": "1", "autoSelect": False}
            ],
            "protocolVersions": ["v1"],
            "containers": [
                {
                    "name": "kserve-container",
                    "image": f"{spec['runtime_image_repository']}@{spec['runtime_image_digest']}",
                    "args": args,
                    "ports": [{"name": "http", "containerPort": 8080, "protocol": "TCP"}],
                    "resources": {
                        "limits": {"nvidia.com/gpu": str(gpu_count)},
                        "requests": {"nvidia.com/gpu": str(gpu_count)},
                    },
                    "readinessProbe": {
                        "httpGet": {"path": "/health/ready", "port": 8080},
                        "initialDelaySeconds": 10,
                        "periodSeconds": 10,
                        "failureThreshold": 30,
                    },
                }
            ],
        },
    }


def _tts_serving_runtime(target: ProviderTarget) -> dict[str, Any]:
    spec = target.desired_spec["tts"]
    inference = spec["inference_config"]
    gpu_count = spec["hardware_profile"]["gpu_count"]
    args = [
        "--model-dir",
        "/mnt/models/tts",
        "--candidate-dir",
        "/mnt/models/tts",
        "--vocoder-dir",
        str(inference["vocoder_path"]),
        "--speaker-embedding-path",
        "/mnt/models/tts/speaker_embedding.npy",
        "--model-revision",
        str(inference["model_revision"]),
        "--served-model-name",
        str(spec["component_model_id"]),
        "--allowed-voice",
        str(inference["voice_profile_id"]),
        "--fixed-variant",
        "candidate",
        "--port",
        "8080",
    ]
    return {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": spec["serving_runtime_name"],
            "namespace": target.namespace,
            "labels": _tts_labels(target),
        },
        "spec": {
            "supportedModelFormats": [
                {"name": "industrial-tts", "version": "1", "autoSelect": False}
            ],
            "protocolVersions": ["v1"],
            "containers": [
                {
                    "name": "kserve-container",
                    "image": f"{spec['runtime_image_repository']}@{spec['runtime_image_digest']}",
                    "command": [
                        "python",
                        "-m",
                        "industrial_ops_agent.multimodal.tts_runtime",
                    ],
                    "args": args,
                    "ports": [{"name": "http", "containerPort": 8080, "protocol": "TCP"}],
                    "securityContext": hardened_container_security_context(),
                    "resources": {
                        "limits": {
                            "cpu": "2",
                            "memory": "4Gi",
                            "nvidia.com/gpu": str(gpu_count),
                        },
                        "requests": {
                            "cpu": "250m",
                            "memory": "512Mi",
                            "nvidia.com/gpu": str(gpu_count),
                        },
                    },
                    "readinessProbe": {
                        "httpGet": {"path": "/health/ready", "port": 8080},
                        "initialDelaySeconds": 10,
                        "periodSeconds": 10,
                        "failureThreshold": 30,
                    },
                }
            ],
        },
    }


def _embedding_inference_service(target: ProviderTarget) -> dict[str, Any]:
    deployment = target.desired_spec
    spec = deployment["embedding"]
    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": spec["service_name"],
            "namespace": target.namespace,
            "labels": _embedding_labels(target),
            "annotations": {"serving.kserve.io/deploymentMode": "Standard"},
        },
        "spec": {
            "predictor": {
                "serviceAccountName": deployment["service_account_name"],
                "storageUris": [
                    {"uri": spec["artifact_uri"], "mountPath": "/mnt/models/embedding"}
                ],
                "model": {
                    "modelFormat": {"name": "industrial-embedding", "version": "1"},
                    "runtime": spec["serving_runtime_name"],
                },
            }
        },
    }


def _tts_inference_service(target: ProviderTarget) -> dict[str, Any]:
    deployment = target.desired_spec
    spec = deployment["tts"]
    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": spec["service_name"],
            "namespace": target.namespace,
            "labels": _tts_labels(target),
            "annotations": {"serving.kserve.io/deploymentMode": "Standard"},
        },
        "spec": {
            "predictor": {
                "serviceAccountName": deployment["service_account_name"],
                "storageUris": [
                    {"uri": spec["artifact_uri"], "mountPath": "/mnt/models/tts"}
                ],
                "model": {
                    "modelFormat": {"name": "industrial-tts", "version": "1"},
                    "runtime": spec["serving_runtime_name"],
                },
            }
        },
    }


def _reranker_inference_service(target: ProviderTarget) -> dict[str, Any]:
    deployment = target.desired_spec
    spec = deployment["reranker"]
    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": spec["service_name"],
            "namespace": target.namespace,
            "labels": _reranker_labels(target),
            "annotations": {"serving.kserve.io/deploymentMode": "Standard"},
        },
        "spec": {
            "predictor": {
                "serviceAccountName": deployment["service_account_name"],
                "storageUris": [
                    {"uri": spec["artifact_uri"], "mountPath": "/mnt/models/reranker"}
                ],
                "model": {
                    "modelFormat": {"name": "industrial-reranker", "version": "1"},
                    "runtime": spec["serving_runtime_name"],
                },
            }
        },
    }


def _timeseries_inference_service(target: ProviderTarget) -> dict[str, Any]:
    deployment = target.desired_spec
    spec = deployment["timeseries"]
    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": spec["service_name"],
            "namespace": target.namespace,
            "labels": _timeseries_labels(target),
            "annotations": {"serving.kserve.io/deploymentMode": "Standard"},
        },
        "spec": {
            "predictor": {
                "serviceAccountName": deployment["service_account_name"],
                "storageUris": [
                    {"uri": spec["artifact_uri"], "mountPath": "/mnt/models/timeseries"}
                ],
                "model": {
                    "modelFormat": {"name": "industrial-timeseries", "version": "1"},
                    "runtime": spec["serving_runtime_name"],
                },
            }
        },
    }


def _rul_inference_service(target: ProviderTarget) -> dict[str, Any]:
    deployment = target.desired_spec
    spec = deployment["rul"]
    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": spec["service_name"],
            "namespace": target.namespace,
            "labels": _rul_labels(target),
            "annotations": {"serving.kserve.io/deploymentMode": "Standard"},
        },
        "spec": {
            "predictor": {
                "serviceAccountName": deployment["service_account_name"],
                "storageUris": [
                    {"uri": spec["artifact_uri"], "mountPath": "/mnt/models/rul"}
                ],
                "model": {
                    "modelFormat": {"name": "industrial-rul", "version": "1"},
                    "runtime": spec["serving_runtime_name"],
                },
            }
        },
    }


def _validated_autoscaling(target: ProviderTarget) -> dict[str, Any] | None:
    """Validate persisted v3 autoscaling state before any cluster mutation."""

    desired = target.desired_spec
    autoscaling = desired.get("autoscaling")
    if autoscaling is None:
        if desired.get("schema_version") == "ioap-kserve-rollout/v3":
            raise DeploymentProviderUnavailable("invalid_autoscaling_contract")
        return None
    if desired.get("schema_version") != "ioap-kserve-rollout/v3" or not isinstance(
        autoscaling, dict
    ):
        raise DeploymentProviderUnavailable("invalid_autoscaling_contract")
    metric = autoscaling.get("metric")
    behavior = autoscaling.get("behavior")
    inference = desired.get("inference_config")
    hardware = desired.get("hardware_profile")
    if (
        set(autoscaling)
        != {
            "mode",
            "policy_version",
            "min_replicas",
            "max_replicas",
            "metric",
            "behavior",
        }
        or not isinstance(metric, dict)
        or set(metric)
        != {
            "backend",
            "name",
            "server_address",
            "query",
            "target_type",
            "target_value",
        }
        or not isinstance(behavior, dict)
        or set(behavior)
        != {
            "scale_up_stabilization_seconds",
            "scale_down_stabilization_seconds",
            "scale_up_max_pods_per_minute",
            "scale_down_max_pods_per_two_minutes",
        }
        or not isinstance(inference, dict)
        or inference.get("engine") != "vllm"
        or not isinstance(hardware, dict)
    ):
        raise DeploymentProviderUnavailable("invalid_autoscaling_contract")

    minimum = autoscaling.get("min_replicas")
    maximum = autoscaling.get("max_replicas")
    target_value = metric.get("target_value")
    gpu_count = hardware.get("gpu_count")
    integers = (minimum, maximum, target_value, gpu_count)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in integers):
        raise DeploymentProviderUnavailable("invalid_autoscaling_contract")
    minimum_value = cast(int, minimum)
    maximum_value = cast(int, maximum)
    target_metric_value = cast(int, target_value)
    requested_gpu_count = cast(int, gpu_count)
    if not (
        2 <= minimum_value <= 8
        and 2 <= maximum_value <= 32
        and minimum_value <= maximum_value
    ):
        raise DeploymentProviderUnavailable("invalid_autoscaling_contract")
    if not 1 <= target_metric_value <= 32 or requested_gpu_count <= 0:
        raise DeploymentProviderUnavailable("invalid_autoscaling_contract")

    try:
        expected_query = autoscaling_metric_query(target.namespace, target.service_name)
    except ValueError as exc:
        raise DeploymentProviderUnavailable("invalid_autoscaling_contract") from exc
    expected_behavior = {
        "scale_up_stabilization_seconds": KEDA_SCALE_UP_STABILIZATION_SECONDS,
        "scale_down_stabilization_seconds": KEDA_SCALE_DOWN_STABILIZATION_SECONDS,
        "scale_up_max_pods_per_minute": KEDA_SCALE_UP_MAX_PODS_PER_MINUTE,
        "scale_down_max_pods_per_two_minutes": (
            KEDA_SCALE_DOWN_MAX_PODS_PER_TWO_MINUTES
        ),
    }
    if (
        autoscaling.get("mode") != "KEDA_VLLM"
        or autoscaling.get("policy_version") != KEDA_AUTOSCALING_POLICY_VERSION
        or metric.get("backend") != "prometheus"
        or metric.get("name") != KEDA_VLLM_METRIC_NAME
        or metric.get("server_address") != KEDA_PROMETHEUS_SERVER_ADDRESS
        or metric.get("query") != expected_query
        or metric.get("target_type") != KEDA_TARGET_TYPE
        or behavior != expected_behavior
    ):
        raise DeploymentProviderUnavailable("invalid_autoscaling_contract")
    return autoscaling


def _inference_pool_routing(target: ProviderTarget) -> dict[str, Any] | None:
    routing = target.desired_spec.get("routing")
    if routing is None:
        return None
    if not isinstance(routing, dict):
        raise DeploymentProviderUnavailable("invalid_inference_pool_desired_state")
    mode = routing.get("mode")
    if mode == "STANDARD":
        return None
    if mode != "INFERENCE_POOL" or target.desired_spec.get("inference_config", {}).get(
        "engine"
    ) != "vllm":
        raise DeploymentProviderUnavailable("invalid_inference_pool_desired_state")
    pool_name = routing.get("inference_pool_name")
    target_port = routing.get("target_port")
    app_protocol = routing.get("app_protocol")
    selector = routing.get("selector")
    endpoint = routing.get("endpoint_picker_ref")
    if (
        not isinstance(pool_name, str)
        or not pool_name
        or target_port != 8080
        or app_protocol != "http"
        or not isinstance(selector, dict)
        or not isinstance(selector.get("matchLabels"), dict)
        or not isinstance(endpoint, dict)
        or not isinstance(endpoint.get("name"), str)
        or not endpoint.get("name")
        or not isinstance(endpoint.get("port"), int)
        or not 1 <= endpoint["port"] <= 65535
        or endpoint.get("failure_mode") != "FailClose"
    ):
        raise DeploymentProviderUnavailable("invalid_inference_pool_desired_state")
    match_labels = selector["matchLabels"]
    expected_labels = {
        "industrial-ops.ai/inference-pool": pool_name,
        "industrial-ops.ai/release-id": target.release_id[:63],
    }
    if match_labels != expected_labels:
        raise DeploymentProviderUnavailable("invalid_inference_pool_desired_state")
    return routing


def _inference_pool(target: ProviderTarget, routing: dict[str, Any]) -> dict[str, Any]:
    endpoint = routing["endpoint_picker_ref"]
    return {
        "apiVersion": "inference.networking.k8s.io/v1",
        "kind": "InferencePool",
        "metadata": {
            "name": routing["inference_pool_name"],
            "namespace": target.namespace,
            "labels": _labels(target),
        },
        "spec": {
            "selector": routing["selector"],
            "targetPorts": [
                {
                    "number": routing["target_port"],
                }
            ],
            "appProtocol": routing["app_protocol"],
            "endpointPickerRef": {
                "name": endpoint["name"],
                "port": {"number": endpoint["port"]},
                "failureMode": endpoint["failure_mode"],
            },
        },
    }


def _main_candidate_backend(target: ProviderTarget, service_port: int) -> dict[str, Any]:
    routing = _inference_pool_routing(target)
    if routing is None:
        return {"name": f"{target.service_name}-predictor", "port": service_port}
    return {
        "group": "inference.networking.k8s.io",
        "kind": "InferencePool",
        "name": routing["inference_pool_name"],
        "port": routing["target_port"],
    }


def _http_route(target: ProviderTarget, port: int) -> dict[str, Any]:
    route = target.desired_spec["route"]
    rules: list[dict[str, Any]] = []
    embedding = target.desired_spec.get("embedding")
    if isinstance(embedding, dict):
        embedding_rule = _traffic_rule(
            target,
            stable_name=str(embedding["stable_service_name"]),
            candidate_name=f"{embedding['service_name']}-predictor",
            port=port,
        )
        embedding_rule["matches"] = [
            {"path": {"type": "PathPrefix", "value": embedding["route_path"]}}
        ]
        rules.append(embedding_rule)
    reranker = target.desired_spec.get("reranker")
    if isinstance(reranker, dict):
        reranker_rule = _traffic_rule(
            target,
            stable_name=str(reranker["stable_service_name"]),
            candidate_name=f"{reranker['service_name']}-predictor",
            port=port,
        )
        reranker_rule["matches"] = [
            {"path": {"type": "PathPrefix", "value": reranker["route_path"]}}
        ]
        rules.append(reranker_rule)
    tts = target.desired_spec.get("tts")
    if isinstance(tts, dict):
        tts_rule = _traffic_rule(
            target,
            stable_name=str(tts["stable_service_name"]),
            candidate_name=f"{tts['service_name']}-predictor",
            port=port,
        )
        tts_rule["matches"] = [
            {"path": {"type": "PathPrefix", "value": tts["route_path"]}}
        ]
        rules.append(tts_rule)
    timeseries = target.desired_spec.get("timeseries")
    if isinstance(timeseries, dict):
        timeseries_rule = _traffic_rule(
            target,
            stable_name=str(timeseries["stable_service_name"]),
            candidate_name=f"{timeseries['service_name']}-predictor",
            port=port,
        )
        timeseries_rule["matches"] = [
            {"path": {"type": "PathPrefix", "value": timeseries["route_path"]}}
        ]
        rules.append(timeseries_rule)
    rul = target.desired_spec.get("rul")
    if isinstance(rul, dict):
        rul_rule = _traffic_rule(
            target,
            stable_name=str(rul["stable_service_name"]),
            candidate_name=f"{rul['service_name']}-predictor",
            port=port,
        )
        rul_rule["matches"] = [
            {"path": {"type": "PathPrefix", "value": rul["route_path"]}}
        ]
        rules.append(rul_rule)
    rules.append(
        _traffic_rule(
            target,
            stable_name=target.stable_service_name,
            candidate_name=f"{target.service_name}-predictor",
            port=port,
            candidate_backend=_main_candidate_backend(target, port),
        )
    )
    return {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "HTTPRoute",
        "metadata": {
            "name": target.route_name,
            "namespace": target.namespace,
            "labels": _labels(target),
        },
        "spec": {
            "parentRefs": [{"name": route["gateway_name"]}],
            "hostnames": [route["hostname"]],
            "rules": rules,
        },
    }


def _traffic_rule(
    target: ProviderTarget,
    *,
    stable_name: str,
    candidate_name: str,
    port: int,
    candidate_backend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stable = {"name": stable_name, "port": port}
    candidate = candidate_backend or {"name": candidate_name, "port": port}
    if target.desired_stage == "SHADOW":
        rule = {
            "backendRefs": [{**stable, "weight": 100}],
            "filters": [
                {
                    "type": "RequestMirror",
                    "requestMirror": {"backendRef": candidate},
                }
            ],
        }
    elif target.desired_stage in {"CANARY_5", "CANARY_25"}:
        candidate_weight = int(target.desired_traffic_percent)
        rule = {
            "backendRefs": [
                {**stable, "weight": 100 - candidate_weight},
                {**candidate, "weight": candidate_weight},
            ]
        }
    elif target.desired_stage == "PRODUCTION":
        rule = {"backendRefs": [{**candidate, "weight": 100}]}
    elif target.desired_stage == "ROLLED_BACK":
        rule = {"backendRefs": [{**stable, "weight": 100}]}
    else:
        raise DeploymentProviderUnavailable("unsupported_deployment_stage")
    return rule


def _labels(target: ProviderTarget) -> dict[str, str]:
    return {
        "app.kubernetes.io/managed-by": "industrial-ops-release-controller",
        "industrial-ops.ai/release-id": target.release_id[:63],
        "industrial-ops.ai/manifest-hash": target.manifest_hash[:63],
    }


def _primary_labels(target: ProviderTarget) -> dict[str, str]:
    labels = {**_labels(target), "industrial-ops.ai/component": "llm"}
    routing = _inference_pool_routing(target)
    if routing is None:
        return labels
    selector = routing["selector"]
    labels.update(selector["matchLabels"])
    return labels


def _timeseries_labels(target: ProviderTarget) -> dict[str, str]:
    return {**_labels(target), "industrial-ops.ai/component": "timeseries"}


def _rul_labels(target: ProviderTarget) -> dict[str, str]:
    return {**_labels(target), "industrial-ops.ai/component": "rul"}


def _reranker_labels(target: ProviderTarget) -> dict[str, str]:
    return {**_labels(target), "industrial-ops.ai/component": "reranker"}


def _embedding_labels(target: ProviderTarget) -> dict[str, str]:
    return {**_labels(target), "industrial-ops.ai/component": "embedding"}


def _tts_labels(target: ProviderTarget) -> dict[str, str]:
    return {**_labels(target), "industrial-ops.ai/component": "tts"}


def _resource_ready(document: dict[str, Any], required_types: tuple[str, ...]) -> bool:
    generation = document.get("metadata", {}).get("generation")
    conditions = document.get("status", {}).get("conditions", [])
    if not isinstance(conditions, list):
        return False
    by_type = {
        item.get("type"): item
        for item in conditions
        if isinstance(item, dict) and isinstance(item.get("type"), str)
    }
    for condition_type in required_types:
        condition = by_type.get(condition_type)
        if condition is None or condition.get("status") != "True":
            return False
        observed = condition.get("observedGeneration")
        if isinstance(generation, int) and isinstance(observed, int) and observed < generation:
            return False
    return True


def _route_ready(document: dict[str, Any], gateway_name: str | None = None) -> bool:
    parents = document.get("status", {}).get("parents", [])
    if not isinstance(parents, list) or not parents:
        return False
    if gateway_name is not None:
        return _gateway_parent_ready(document, gateway_name)
    return any(
        isinstance(parent, dict)
        and _conditions_pass(parent.get("conditions"), ("Accepted", "ResolvedRefs"))
        for parent in parents
    )


def _gateway_parent_ready(document: dict[str, Any], gateway_name: str) -> bool:
    parents = document.get("status", {}).get("parents", [])
    if not isinstance(parents, list):
        return False
    for parent in parents:
        if not isinstance(parent, dict):
            continue
        parent_ref = parent.get("parentRef")
        if not isinstance(parent_ref, dict) or parent_ref.get("name") != gateway_name:
            continue
        if parent_ref.get("group", "gateway.networking.k8s.io") != (
            "gateway.networking.k8s.io"
        ):
            continue
        if parent_ref.get("kind", "Gateway") != "Gateway":
            continue
        if _conditions_pass(parent.get("conditions"), ("Accepted", "ResolvedRefs")):
            return True
    return False


def _conditions_pass(conditions: Any, required_types: tuple[str, ...]) -> bool:
    if not isinstance(conditions, list):
        return False
    passing = {
        item.get("type")
        for item in conditions
        if isinstance(item, dict) and item.get("status") == "True"
    }
    return set(required_types).issubset(passing)


def _pending_result(
    target: ProviderTarget,
    document: dict[str, Any],
    reason_code: str,
) -> ProviderResult:
    return ProviderResult(
        ready=False,
        observed_stage=target.desired_stage,
        observed_traffic_percent=0.0,
        applied_spec_hash=None,
        provider_revision=str(document.get("metadata", {}).get("resourceVersion") or "") or None,
        endpoint_url=None,
        reason_code=reason_code,
    )


def _provider_revision(*documents: dict[str, Any]) -> str:
    versions = [str(item.get("metadata", {}).get("resourceVersion") or "") for item in documents]
    payload = "|".join(versions).encode()
    return f"sha256:{sha256(payload).hexdigest()}"


def _json_object(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise KubernetesApiError("kubernetes_api_invalid_json") from exc
    if not isinstance(payload, dict):
        raise KubernetesApiError("kubernetes_api_invalid_json")
    return payload
