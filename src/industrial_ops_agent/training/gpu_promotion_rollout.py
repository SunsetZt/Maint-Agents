"""Contracts and Kubernetes documents for the local GPU promotion rollout."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

from industrial_ops_agent.training.gpu_promotion_lab import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    GpuPromotionLabContractError,
)
from industrial_ops_agent.training.gpu_promotion_security import (
    hardened_container_security_context,
)

_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_EXPECTED_HARD_GATES = {
    "cross_tenant_isolation",
    "data_governance",
    "high_risk_approval",
    "no_blocking_regressions",
    "replay_reproducibility",
    "t3_control_execution",
    "tool_schema_success",
    "unauthorized_tool_execution",
    "valid_citations",
}

RolloutStage = Literal["SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"]

NAMESPACE = "ioap-gpu-promotion-lab"
GATEWAY_CLASS = "ioap-gpu-lab-envoy"
GATEWAY_NAME = "ioap-gpu-lab-gateway"
ENVOY_PROXY_NAME = "ioap-gpu-lab-proxy"
ROUTE_NAME = "ioap-gpu-lab-route"
HOSTNAME = "gpu-lab.local"
STABLE_SERVICE = "ioap-qwen-stable"
CANDIDATE_SERVICE = "ioap-qwen-candidate"
ENVOY_GATEWAY_CONTROLLER = "gateway.envoyproxy.io/gatewayclass-controller"


@dataclass(frozen=True, slots=True)
class PromotionEvidence:
    git_commit: str
    container_digest: str
    stable_experiment_id: str
    stable_method: str
    stable_adapter_artifact_id: str
    stable_adapter_object_key: str
    stable_adapter_content_hash: str
    stable_evaluation_id: str
    experiment_id: str
    method: str
    adapter_artifact_id: str
    adapter_object_key: str
    adapter_content_hash: str
    evaluation_id: str
    replay_evaluation_id: str
    evidence_chain_sha256: str


def load_promotion_evidence(path: Path) -> PromotionEvidence:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GpuPromotionLabContractError("promotion_evidence_is_invalid") from exc
    if not isinstance(document, dict):
        raise GpuPromotionLabContractError("promotion_evidence_is_invalid")
    chain = document.get("evidence_chain_sha256")
    unsigned = dict(document)
    unsigned.pop("evidence_chain_sha256", None)
    observed_chain = sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    runtime = document.get("runtime")
    datasets = document.get("datasets")
    selection = document.get("selection")
    experiments = document.get("controlled_training_experiments")
    evaluations = document.get("independent_gold_evaluations")
    reviews = document.get("dual_domain_expert_reviews")
    separation = document.get("separation_of_duties")
    if (
        document.get("schema_version") != "local-gpu-promotion-evidence/v1"
        or document.get("classification") != "LOCAL_STAGING_PROJECT_AUTHORIZED"
        or document.get("data_classification") != "SYNTHETIC_DATA"
        or document.get("authorization_classification")
        != "PROJECT_OWNED_SYNTHETIC"
        or document.get("actual_gpu_execution") is not True
        or document.get("model_training_simulated") is not False
        or document.get("enterprise_production_data") is not False
        or document.get("production_claim") is not False
        or document.get("role_attestation") != "LOCAL_ROLE_SIMULATION"
        or chain != observed_chain
        or not isinstance(runtime, dict)
        or not _GIT_SHA.fullmatch(str(runtime.get("git_commit", "")))
        or not _IMAGE_DIGEST.fullmatch(str(runtime.get("container_digest", "")))
        or runtime.get("base_model_id") != BASE_MODEL_ID
        or runtime.get("base_model_revision") != BASE_MODEL_REVISION
        or not _datasets_are_isolated(datasets)
        or not isinstance(selection, dict)
        or selection.get("status") != "SELECTED_FOR_LOCAL_STAGING_PROMOTION"
        or selection.get("method") not in {"LORA", "QLORA"}
        or selection.get("all_hard_gates_passed") is not True
        or selection.get("critical_slice_no_regression") is not True
        or not isinstance(experiments, dict)
        or set(experiments) != {"LORA", "QLORA"}
        or not isinstance(evaluations, dict)
        or set(evaluations) != {"LORA", "QLORA"}
    ):
        raise GpuPromotionLabContractError("promotion_evidence_binding_failed")
    typed_experiments = cast(dict[str, Any], experiments)
    typed_evaluations = cast(dict[str, Any], evaluations)
    method = cast(str, selection["method"])
    for training_method in ("LORA", "QLORA"):
        _require_actual_gpu_experiment(
            typed_experiments.get(training_method), method=training_method
        )
        _require_independent_gold_evaluation(
            typed_evaluations.get(training_method),
            method=training_method,
            experiment=typed_experiments[training_method],
            selected=training_method == method,
        )
    selected = typed_experiments.get(method) if isinstance(method, str) else None
    selected_evaluation = (
        typed_evaluations.get(method) if isinstance(method, str) else None
    )
    replay = selection.get("replay")
    if (
        method not in {"LORA", "QLORA"}
        or not isinstance(selected, dict)
        or not isinstance(selected_evaluation, dict)
        or selected.get("experiment_id") != selection.get("experiment_id")
        or selected_evaluation.get("evaluation_id") != selection.get("evaluation_id")
        or not _finite(selection.get("observed_quality_delta"))
        or float(selection["observed_quality_delta"])
        != float(selected_evaluation["quality_delta"])
        or selection.get("quality_lift_min") != 0.02
        or not isinstance(replay, dict)
        or replay.get("status") != "STABLE"
        or replay.get("candidate_score_drift") != 0.0
        or replay.get("baseline_score_drift") != 0.0
        or replay.get("candidate_score") != selected_evaluation.get("candidate_score")
        or replay.get("baseline_score") != selected_evaluation.get("baseline_score")
        or not isinstance(replay.get("evaluation_id"), str)
        or not _SHA256.fullmatch(str(replay.get("report_hash", "")))
        or not _case_evidence_is_bound(replay.get("case_evidence"), method=method)
    ):
        raise GpuPromotionLabContractError("promotion_candidate_is_not_replay_stable")
    ranked_methods = sorted(
        (
            item
            for item in ("LORA", "QLORA")
            if typed_evaluations[item]["decision"] == "SMOKE_PASSED"
        ),
        key=lambda item: (
            float(typed_evaluations[item]["candidate_score"]),
            float(typed_evaluations[item]["quality_delta"]),
            item,
        ),
    )
    if not ranked_methods or method != ranked_methods[-1]:
        raise GpuPromotionLabContractError("promotion_candidate_is_not_gold_winner")
    stable_method = next(item for item in ("LORA", "QLORA") if item != method)
    stable = typed_experiments[stable_method]
    _require_dual_reviews(
        reviews,
        separation=separation,
        selected=selected,
        selected_evaluation=selected_evaluation,
        replay=cast(dict[str, Any], replay),
        datasets=datasets,
    )
    values = {
        "git_commit": runtime.get("git_commit"),
        "container_digest": runtime.get("container_digest"),
        "stable_experiment_id": stable.get("experiment_id"),
        "stable_method": stable_method,
        "stable_adapter_artifact_id": stable.get("artifact_id"),
        "stable_adapter_object_key": stable.get("artifact_object_key"),
        "stable_adapter_content_hash": stable.get("artifact_content_hash"),
        "stable_evaluation_id": typed_evaluations[stable_method].get("evaluation_id"),
        "experiment_id": selection.get("experiment_id"),
        "method": method,
        "adapter_artifact_id": selected.get("artifact_id"),
        "adapter_object_key": selected.get("artifact_object_key"),
        "adapter_content_hash": selected.get("artifact_content_hash"),
        "evaluation_id": selection.get("evaluation_id"),
        "replay_evaluation_id": replay.get("evaluation_id"),
        "evidence_chain_sha256": chain,
    }
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise GpuPromotionLabContractError("promotion_evidence_identity_is_incomplete")
    return PromotionEvidence(**cast(dict[str, str], values))


def _datasets_are_isolated(value: object) -> bool:
    if not isinstance(value, dict) or value.get("split_isolation") != "VERIFIED":
        return False
    training = value.get("training")
    gold = value.get("independent_gold")
    if not isinstance(training, dict) or not isinstance(gold, dict):
        return False
    return (
        training.get("classification") == "LOCAL_STAGING_PROJECT_AUTHORIZED"
        and training.get("data_classification") == "SYNTHETIC_DATA"
        and training.get("authorization_classification")
        == "PROJECT_OWNED_SYNTHETIC"
        and training.get("production_claim") is False
        and gold.get("classification") == "LOCAL_STAGING_PROJECT_AUTHORIZED"
        and gold.get("data_classification") == "SYNTHETIC_DATA"
        and gold.get("production_claim") is False
        and training.get("training_snapshot_id") != gold.get("snapshot_id")
        and _SHA256.fullmatch(str(training.get("authorization_digest", "")))
        is not None
        and _SHA256.fullmatch(str(training.get("dataset_digest", ""))) is not None
        and _SHA256.fullmatch(str(gold.get("dataset_digest", ""))) is not None
        and _IMAGE_DIGEST.fullmatch(str(training.get("training_manifest_hash", "")))
        is not None
        and _IMAGE_DIGEST.fullmatch(str(gold.get("manifest_hash", ""))) is not None
    )


def _require_actual_gpu_experiment(value: object, *, method: str) -> None:
    if not isinstance(value, dict):
        raise GpuPromotionLabContractError("promotion_training_evidence_is_incomplete")
    runtime = value.get("runtime")
    metrics = value.get("metrics")
    if (
        value.get("method") != method
        or not isinstance(value.get("experiment_id"), str)
        or not isinstance(value.get("mlflow_run_id"), str)
        or not isinstance(value.get("artifact_id"), str)
        or not isinstance(value.get("artifact_object_key"), str)
        or not _SHA256.fullmatch(str(value.get("artifact_content_hash", "")))
        or not _positive_int(value.get("optimizer_steps"))
        or not isinstance(metrics, dict)
        or not metrics
        or any(not _finite(item) for item in metrics.values())
        or not isinstance(runtime, dict)
        or not _positive_int(runtime.get("gpu_count"))
        or not _positive_int(runtime.get("gpu_total_memory_bytes"))
        or not isinstance(runtime.get("gpu_name"), str)
        or not str(runtime.get("gpu_name", "")).strip()
        or not isinstance(runtime.get("cuda_version"), str)
        or not isinstance(runtime.get("torch_version"), str)
        or runtime.get("resolved_base_model_revision") != BASE_MODEL_REVISION
        or runtime.get("resumed_from_checkpoint") is not False
        or not _positive_int(runtime.get("train_rows"))
        or not _positive_int(runtime.get("validation_rows"))
    ):
        raise GpuPromotionLabContractError("promotion_training_evidence_is_incomplete")


def _require_independent_gold_evaluation(
    value: object,
    *,
    method: str,
    experiment: dict[str, Any],
    selected: bool,
) -> None:
    if not isinstance(value, dict):
        raise GpuPromotionLabContractError("promotion_gold_evidence_is_incomplete")
    hard_gates = value.get("hard_gate_results")
    if (
        value.get("candidate_experiment_id") != experiment.get("experiment_id")
        or not isinstance(value.get("evaluation_id"), str)
        or not isinstance(value.get("baseline_experiment_id"), str)
        or not all(
            _finite(value.get(key))
            for key in ("candidate_score", "baseline_score", "quality_delta", "ci_low", "ci_high")
        )
        or not isinstance(hard_gates, dict)
        or set(hard_gates) != _EXPECTED_HARD_GATES
        or not all(isinstance(item, bool) for item in hard_gates.values())
        or not _SHA256.fullmatch(str(value.get("report_hash", "")))
        or not _case_evidence_is_bound(value.get("case_evidence"), method=method)
    ):
        raise GpuPromotionLabContractError("promotion_gold_evidence_is_incomplete")
    passes_promotion_gates = (
        float(value["quality_delta"]) >= 0.02
        and float(value["candidate_score"]) > float(value["baseline_score"])
        and float(value["ci_low"]) > 0.0
        and all(item is True for item in hard_gates.values())
    )
    decision = value.get("decision")
    if (
        decision not in {"SMOKE_PASSED", "REJECTED"}
        or (decision == "SMOKE_PASSED") != passes_promotion_gates
        or (selected and not passes_promotion_gates)
    ):
        raise GpuPromotionLabContractError("promotion_gold_evidence_is_incomplete")


def _case_evidence_is_bound(value: object, *, method: str) -> bool:
    if not isinstance(value, dict):
        return False
    metadata = value.get("metadata")
    return (
        isinstance(value.get("artifact_id"), str)
        and isinstance(value.get("object_key"), str)
        and _SHA256.fullmatch(str(value.get("content_hash", ""))) is not None
        and _positive_int(value.get("size_bytes"))
        and isinstance(metadata, dict)
        and metadata.get("candidate_method") == method
        and _SHA256.fullmatch(str(metadata.get("candidate_artifact_hash", "")))
        is not None
        and _SHA256.fullmatch(str(metadata.get("suite_manifest_hash", "")))
        is not None
    )


def _require_dual_reviews(
    value: object,
    *,
    separation: object,
    selected: dict[str, Any],
    selected_evaluation: dict[str, Any],
    replay: dict[str, Any],
    datasets: object,
) -> None:
    if not isinstance(separation, dict) or not isinstance(datasets, dict):
        raise GpuPromotionLabContractError("promotion_review_evidence_is_incomplete")
    training = datasets.get("training")
    if not isinstance(training, dict):
        raise GpuPromotionLabContractError("promotion_review_evidence_is_incomplete")
    reviewers = separation.get("reviewer_subjects")
    duty_subjects = (
        separation.get("trainer_subject"),
        separation.get("evaluator_subject"),
        separation.get("promotion_subject"),
    )
    expected_binding = {
        "evaluation_id": selected_evaluation.get("evaluation_id"),
        "evaluation_report_hash": selected_evaluation.get("report_hash"),
        "replay_evaluation_id": replay.get("evaluation_id"),
        "replay_report_hash": replay.get("report_hash"),
        "candidate_experiment_id": selected.get("experiment_id"),
        "adapter_content_hash": selected.get("artifact_content_hash"),
        "authorization_digest": training.get("authorization_digest"),
    }
    expected_binding_hash = _digest(expected_binding)
    if (
        separation.get("status") != "PASSED"
        or not isinstance(reviewers, list)
        or len(reviewers) != 2
        or any(not isinstance(item, str) or not item for item in reviewers)
        or any(not isinstance(item, str) or not item for item in duty_subjects)
        or len(set(cast(list[str], reviewers))) != 2
        or set(cast(list[str], reviewers)) & set(cast(tuple[str, str, str], duty_subjects))
        or not isinstance(value, list)
        or len(value) != 2
    ):
        raise GpuPromotionLabContractError("promotion_review_evidence_is_incomplete")
    observed_reviewers: set[str] = set()
    for receipt in value:
        if not isinstance(receipt, dict):
            raise GpuPromotionLabContractError("promotion_review_evidence_is_incomplete")
        unsigned = dict(receipt)
        receipt_hash = unsigned.pop("receipt_hash", None)
        reviewer = receipt.get("reviewer_subject_id")
        if (
            reviewer not in reviewers
            or reviewer in observed_reviewers
            or receipt.get("reviewer_role") != "domain_expert"
            or receipt.get("decision") != "APPROVED"
            or receipt.get("role_attestation") != "LOCAL_ROLE_SIMULATION"
            or receipt.get("production_claim") is not False
            or receipt.get("binding") != expected_binding
            or receipt.get("binding_hash") != expected_binding_hash
            or receipt_hash != _digest(unsigned)
        ):
            raise GpuPromotionLabContractError("promotion_review_evidence_is_incomplete")
        observed_reviewers.add(cast(str, reviewer))
    if observed_reviewers != set(cast(list[str], reviewers)):
        raise GpuPromotionLabContractError("promotion_review_evidence_is_incomplete")


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def namespace_document() -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {
            "name": NAMESPACE,
            "labels": {"ioap.openai.com/environment": "local-staging"},
        },
    }


def gateway_class_document() -> dict[str, Any]:
    return {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "GatewayClass",
        "metadata": {"name": GATEWAY_CLASS},
        "spec": {"controllerName": ENVOY_GATEWAY_CONTROLLER},
    }


def envoy_proxy_document() -> dict[str, Any]:
    return {
        "apiVersion": "gateway.envoyproxy.io/v1alpha1",
        "kind": "EnvoyProxy",
        "metadata": {"name": ENVOY_PROXY_NAME, "namespace": NAMESPACE},
        "spec": {
            "provider": {
                "type": "Kubernetes",
                "kubernetes": {"envoyService": {"type": "ClusterIP"}},
            }
        },
    }


def gateway_document() -> dict[str, Any]:
    return {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "Gateway",
        "metadata": {"name": GATEWAY_NAME, "namespace": NAMESPACE},
        "spec": {
            "gatewayClassName": GATEWAY_CLASS,
            "infrastructure": {
                "parametersRef": {
                    "group": "gateway.envoyproxy.io",
                    "kind": "EnvoyProxy",
                    "name": ENVOY_PROXY_NAME,
                }
            },
            "listeners": [
                {
                    "name": "http",
                    "protocol": "HTTP",
                    "port": 80,
                    "hostname": HOSTNAME,
                    "allowedRoutes": {"namespaces": {"from": "Same"}},
                }
            ],
        },
    }


def inference_service_document(
    *, name: str, image: str, image_digest: str, variant: Literal["stable", "candidate"]
) -> dict[str, Any]:
    if name not in {STABLE_SERVICE, CANDIDATE_SERVICE}:
        raise GpuPromotionLabContractError("rollout_service_name_is_invalid")
    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "annotations": {"serving.kserve.io/deploymentMode": "Standard"},
            "labels": {
                "ioap.openai.com/promotion-lab": "true",
                "ioap.openai.com/model-variant": variant,
                "ioap.openai.com/runtime-image-digest": image_digest.removeprefix(
                    "sha256:"
                )[:63],
            },
        },
        "spec": {
            "predictor": {
                "minReplicas": 1,
                "maxReplicas": 1,
                "containers": [
                    {
                        "name": "kserve-container",
                        "image": image,
                        "imagePullPolicy": "IfNotPresent",
                        "securityContext": hardened_container_security_context(),
                        "env": [
                            {
                                "name": "IOAP_GPU_LAB_USE_ADAPTER",
                                "value": "1" if variant == "candidate" else "0",
                            },
                            {"name": "IOAP_GPU_LAB_CPU_THREADS", "value": "4"},
                        ],
                        "ports": [{"name": "http1", "containerPort": 8080}],
                        "startupProbe": {
                            "httpGet": {"path": "/health/ready", "port": 8080},
                            "periodSeconds": 5,
                            "failureThreshold": 120,
                        },
                        "readinessProbe": {
                            "httpGet": {"path": "/health/ready", "port": 8080},
                            "periodSeconds": 5,
                            "failureThreshold": 6,
                        },
                        "resources": {
                            "requests": {"cpu": "2", "memory": "3Gi"},
                            "limits": {"cpu": "6", "memory": "6Gi"},
                        },
                    }
                ],
            }
        },
    }


def route_document(stage: RolloutStage) -> dict[str, Any]:
    stable = {"name": f"{STABLE_SERVICE}-predictor", "port": 80}
    candidate = {"name": f"{CANDIDATE_SERVICE}-predictor", "port": 80}
    if stage == "SHADOW":
        rule: dict[str, Any] = {
            "backendRefs": [{**stable, "weight": 100}],
            "filters": [
                {
                    "type": "RequestMirror",
                    "requestMirror": {"backendRef": candidate},
                }
            ],
        }
    elif stage in {"CANARY_5", "CANARY_25"}:
        weight = 5 if stage == "CANARY_5" else 25
        rule = {
            "backendRefs": [
                {**stable, "weight": 100 - weight},
                {**candidate, "weight": weight},
            ]
        }
    elif stage == "ROLLED_BACK":
        rule = {"backendRefs": [{**stable, "weight": 100}]}
    else:
        raise GpuPromotionLabContractError("rollout_stage_is_invalid")
    rule["timeouts"] = {"request": "300s", "backendRequest": "300s"}
    return {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "HTTPRoute",
        "metadata": {
            "name": ROUTE_NAME,
            "namespace": NAMESPACE,
            "annotations": {"ioap.openai.com/stage": stage},
        },
        "spec": {
            "parentRefs": [{"name": GATEWAY_NAME}],
            "hostnames": [HOSTNAME],
            "rules": [rule],
        },
    }


def http_route_parent_condition_is_true(
    document: dict[str, Any], *, condition: str
) -> bool:
    metadata = document.get("metadata", {})
    generation = metadata.get("generation")
    parents = document.get("status", {}).get("parents", [])
    if not isinstance(generation, int) or not isinstance(parents, list):
        return False
    for parent in parents:
        if not isinstance(parent, dict):
            continue
        parent_ref = parent.get("parentRef", {})
        if (
            parent.get("controllerName") != ENVOY_GATEWAY_CONTROLLER
            or not isinstance(parent_ref, dict)
            or parent_ref.get("name") != GATEWAY_NAME
        ):
            continue
        conditions = parent.get("conditions", [])
        if not isinstance(conditions, list):
            continue
        if any(
            isinstance(item, dict)
            and item.get("type") == condition
            and item.get("status") == "True"
            and item.get("observedGeneration") == generation
            for item in conditions
        ):
            return True
    return False
