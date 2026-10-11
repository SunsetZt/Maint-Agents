"""Deterministic non-production GPU promotion lab for the RUL component."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from industrial_ops_agent.evaluation.backend import EvaluationRuntimeConfig
from industrial_ops_agent.evaluation.dataset import EvaluationCase, RulEvaluationContract
from industrial_ops_agent.evaluation.metrics import score_rul_paired_observations
from industrial_ops_agent.persistence.models import TrainingExperimentRecord
from industrial_ops_agent.predictive_maintenance.evaluation_backend import (
    RulComponentEvaluationBackend,
)
from industrial_ops_agent.predictive_maintenance.rul_dataset_contract import (
    CONTRACT_VERSION,
    SIGNAL_ORDER,
)
from industrial_ops_agent.predictive_maintenance.rul_training_dataset import (
    RulSequenceSample,
    RulTrainingDatasetBundle,
)
from industrial_ops_agent.predictive_maintenance.rul_transformer_backend import (
    train_rul_transformer,
)
from industrial_ops_agent.training.config import CompiledRulTransformerConfig
from industrial_ops_agent.training.gpu_promotion_security import (
    hardened_container_security_context,
)

CLASSIFICATION = "SIMULATED_NON_PRODUCTION"
SCHEMA_VERSION = "local-rul-promotion-evidence/v1"
DATASET_GENERATOR_VERSION = "project-rul-degradation-simulator/v1"
EXPERIMENT_ID = "rul-transformer-local-promotion-v1"
BASELINE_ID = "rul-empirical-local-baseline-v1"
K8S_NAMESPACE = "ioap-gpu-promotion-lab"
K8S_GATEWAY = "ioap-gpu-lab-gateway"
K8S_ROUTE = "ioap-rul-lab-route"
K8S_STABLE_SERVICE = "ioap-rul-stable"
K8S_CANDIDATE_SERVICE = "ioap-rul-candidate"
K8S_HOSTNAME = "gpu-lab.local"

RulRolloutStage = Literal["SHADOW", "CANARY_25", "ROLLED_BACK"]


class RulPromotionLabError(RuntimeError):
    pass


def run_train_evaluate(output_root: Path) -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_directory = output_root / run_id
    model_directory = run_directory / "model"
    run_directory.mkdir(parents=True, exist_ok=False)

    train = _samples("train", asset_count=12, samples_per_asset=20, offset=0)
    validation = _samples("validation", asset_count=4, samples_per_asset=15, offset=10_000)
    gold_samples = _samples("gold", asset_count=6, samples_per_asset=15, offset=20_000)
    _require_disjoint_assets(train, validation, gold_samples)

    training_snapshot = {
        "schema_version": "local-rul-frozen-snapshot/v1",
        "classification": CLASSIFICATION,
        "generator_version": DATASET_GENERATOR_VERSION,
        "contract_version": CONTRACT_VERSION,
        "signal_order": list(SIGNAL_ORDER),
        "splits": {
            "train": [_sample_document(item) for item in train],
            "validation": [_sample_document(item) for item in validation],
        },
    }
    gold_snapshot = {
        "schema_version": "local-rul-frozen-gold/v1",
        "classification": CLASSIFICATION,
        "generator_version": DATASET_GENERATOR_VERSION,
        "contract_version": CONTRACT_VERSION,
        "signal_order": list(SIGNAL_ORDER),
        "cases": [_sample_document(item) for item in gold_samples],
    }
    training_payload = _canonical_bytes(training_snapshot)
    gold_payload = _canonical_bytes(gold_snapshot)
    training_path = run_directory / "training-snapshot.json"
    gold_path = run_directory / "gold-snapshot.json"
    training_path.write_bytes(training_payload)
    gold_path.write_bytes(gold_payload)
    training_hash = "sha256:" + sha256(training_payload).hexdigest()
    gold_hash = "sha256:" + sha256(gold_payload).hexdigest()

    config = _training_config()
    candidate = TrainingExperimentRecord(
        experiment_id=EXPERIMENT_ID,
        method="RUL_TRANSFORMER",
        base_model_digest="architecture:rul-transformer-v1",
        training_config={"base_model_revision": "rul-transformer-v1"},
    )
    dataset = RulTrainingDatasetBundle(
        method="RUL_TRANSFORMER",
        snapshot_id="local-rul-training-v1",
        manifest_hash=training_hash,
        contract_version=CONTRACT_VERSION,
        signal_order=SIGNAL_ORDER,
        train=train,
        validation=validation,
        split_hashes={
            "train": _sample_hash(train),
            "validation": _sample_hash(validation),
        },
    )
    outcome = train_rul_transformer(candidate, config, dataset, model_directory)

    baseline_quantiles = _quantiles(
        tuple(item.lead_time_minutes for item in train), (0.1, 0.5, 0.9)
    )
    baseline = TrainingExperimentRecord(
        experiment_id=BASELINE_ID,
        method="RUL_EMPIRICAL_BASELINE",
        training_config={"empirical_quantiles_minutes": list(baseline_quantiles)},
    )
    cases = tuple(_evaluation_case(item) for item in gold_samples)
    backend = RulComponentEvaluationBackend()
    runtime = EvaluationRuntimeConfig(
        max_new_tokens=None,
        precision="bfloat16",
        gpu_hourly_cost_usd=0.0,
    )
    candidate_observations = backend.evaluate(
        experiment=candidate,
        cases=cases,
        config=runtime,
        adapter_directory=model_directory,
    )
    baseline_observations = backend.evaluate(
        experiment=baseline,
        cases=cases,
        config=runtime,
        adapter_directory=None,
    )
    thresholds = {
        "median_absolute_error_minutes_max": 120.0,
        "interval_coverage_min": 0.70,
        "interval_coverage_max": 0.99,
        "pinball_loss_max": 60.0,
    }
    evaluation = score_rul_paired_observations(
        cases,
        candidate_observations,
        baseline_observations,
        **thresholds,
    )
    hard_gates = dict(evaluation.evidence.hard_gate_results)
    selected = all(hard_gates.values())
    model_files = {
        name: "sha256:" + sha256((model_directory / name).read_bytes()).hexdigest()
        for name in ("model.safetensors", "model-config.json", "normalizer.json")
    }
    artifact_content_hash = "sha256:" + sha256(_canonical_bytes(model_files)).hexdigest()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "production_claim": False,
        "actual_gpu_execution": True,
        "model_training_simulated": False,
        "data_source": "PROJECT_GENERATED_SYNTHETIC_INDUSTRIAL_FAILURES",
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "candidate": {
            "experiment_id": EXPERIMENT_ID,
            "method": "RUL_TRANSFORMER",
            "artifact_directory": str(model_directory),
            "artifact_content_hash": artifact_content_hash,
            "files": model_files,
            "training_metrics": outcome.metrics,
            "runtime": outcome.runtime_metadata,
            "compiled_config": config.as_dict(),
            "compiled_config_digest": config.digest,
        },
        "baseline": {
            "experiment_id": BASELINE_ID,
            "method": "RUL_EMPIRICAL_BASELINE",
            "empirical_quantiles_minutes": list(baseline_quantiles),
            "artifact": None,
        },
        "frozen_data": {
            "training_snapshot": str(training_path),
            "training_snapshot_sha256": training_hash,
            "gold_snapshot": str(gold_path),
            "gold_snapshot_sha256": gold_hash,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "gold_rows": len(gold_samples),
            "asset_groups_disjoint": True,
        },
        "independent_gold_evaluation": {
            "backend": "RulComponentEvaluationBackend",
            "thresholds": thresholds,
            "hard_gate_results": hard_gates,
            "candidate": evaluation.aggregate_metrics["candidate"],
            "baseline": evaluation.aggregate_metrics["baseline"],
            "candidate_p95_latency_ms": evaluation.evidence.candidate_p95_latency_ms,
            "baseline_p95_latency_ms": evaluation.evidence.baseline_p95_latency_ms,
            "sample_count": len(cases),
        },
        "selection": {
            "status": (
                "SELECTED_FOR_LOCAL_KSERVE_PROMOTION" if selected else "REJECTED_BY_GOLD_EVALUATION"
            ),
            "all_hard_gates_passed": selected,
        },
    }
    report["evidence_chain_sha256"] = sha256(_canonical_bytes(report)).hexdigest()
    report_path = run_directory / "promotion-evidence.json"
    report_path.write_bytes(_canonical_bytes(report) + b"\n")
    latest = {
        "schema_version": "local-rul-promotion-latest/v1",
        "classification": CLASSIFICATION,
        "run_id": run_id,
        "report": str(report_path),
        "report_sha256": "sha256:" + sha256(report_path.read_bytes()).hexdigest(),
        "status": report["selection"]["status"],
    }
    (output_root / "latest.json").write_bytes(_canonical_bytes(latest) + b"\n")
    if not selected:
        failed = sorted(name for name, passed in hard_gates.items() if not passed)
        raise RulPromotionLabError("rul_gold_evaluation_failed:" + ",".join(failed))
    return report_path


def render_kserve_manifest(
    report_path: Path,
    *,
    image: str,
    stage: RulRolloutStage,
    output_path: Path,
) -> Path:
    report = _promotion_report(report_path)
    candidate = report["candidate"]
    baseline = report["baseline"]
    frozen = report["frozen_data"]
    image_id = image.strip()
    if not image_id or ":" not in image_id or "@" in image_id:
        raise RulPromotionLabError("rul_kserve_image_tag_is_invalid")
    identity = {
        "release_id": "rul-local-release-v1",
        "manifest_hash": str(frozen["gold_snapshot_sha256"]),
        "component_model_id": str(candidate["experiment_id"]),
        "artifact_content_hash": str(candidate["artifact_content_hash"]),
    }
    common_args = [
        "--model-dir",
        "/mnt/models/rul",
        "--release-id",
        identity["release_id"],
        "--manifest-hash",
        identity["manifest_hash"],
        "--component-model-id",
        identity["component_model_id"],
        "--artifact-content-hash",
        identity["artifact_content_hash"],
        "--precision",
        "float32",
        "--device",
        "cpu",
        "--port",
        "8080",
    ]
    stable_args = [
        *common_args,
        "--runtime-mode",
        "empirical",
        "--empirical-p10",
        str(baseline["empirical_quantiles_minutes"][0]),
        "--empirical-p50",
        str(baseline["empirical_quantiles_minutes"][1]),
        "--empirical-p90",
        str(baseline["empirical_quantiles_minutes"][2]),
    ]
    candidate_args = [*common_args, "--runtime-mode", "transformer"]
    document = {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            _local_inference_service(
                name=K8S_STABLE_SERVICE,
                image=image_id,
                args=stable_args,
                variant="stable-empirical",
            ),
            _local_inference_service(
                name=K8S_CANDIDATE_SERVICE,
                image=image_id,
                args=candidate_args,
                variant="candidate-transformer",
            ),
            _local_rul_route(stage),
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_bytes(document) + b"\n")
    return output_path


def record_kserve_rollout(
    report_path: Path,
    *,
    image: str,
    probe_results_path: Path,
    output_path: Path,
) -> Path:
    promotion = _promotion_report(report_path)
    probes = _live_probe_results(probe_results_path)
    route = _capture_json(
        [
            "kubectl",
            "-n",
            K8S_NAMESPACE,
            "get",
            "httproute",
            K8S_ROUTE,
            "-o",
            "json",
        ]
    )
    services = {
        variant: _capture_json(
            [
                "kubectl",
                "-n",
                K8S_NAMESPACE,
                "get",
                "inferenceservice",
                name,
                "-o",
                "json",
            ]
        )
        for variant, name in (
            ("stable", K8S_STABLE_SERVICE),
            ("candidate", K8S_CANDIDATE_SERVICE),
        )
    }
    deployments = {
        variant: _capture_json(
            [
                "kubectl",
                "-n",
                K8S_NAMESPACE,
                "get",
                "deployment",
                f"{name}-predictor",
                "-o",
                "json",
            ]
        )
        for variant, name in (
            ("stable", K8S_STABLE_SERVICE),
            ("candidate", K8S_CANDIDATE_SERVICE),
        )
    }
    context = _capture_text(["kubectl", "config", "current-context"])
    image_document = _capture_json(["docker", "image", "inspect", image])
    if not isinstance(image_document, list) or len(image_document) != 1:
        raise RulPromotionLabError("rul_kserve_image_inspection_failed")
    image_metadata = image_document[0]
    route_metadata = route.get("metadata", {})
    route_status = route.get("status", {})
    generation = route_metadata.get("generation")
    conditions = route_status.get("parents", [{}])[0].get("conditions", [])
    route_ready = all(
        any(
            item.get("type") == condition
            and item.get("status") == "True"
            and item.get("observedGeneration") == generation
            for item in conditions
        )
        for condition in ("Accepted", "ResolvedRefs")
    )
    expected_security = hardened_container_security_context()
    workload_projection: dict[str, Any] = {}
    workloads_ready = True
    for variant, deployment in deployments.items():
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        available = int(deployment.get("status", {}).get("availableReplicas", 0))
        projected = {
            "deployment": deployment["metadata"]["name"],
            "image": container.get("image"),
            "security_context": container.get("securityContext"),
            "available_replicas": available,
        }
        workload_projection[variant] = projected
        workloads_ready = workloads_ready and (
            projected["image"] == image
            and projected["security_context"] == expected_security
            and available == 1
            and _resource_condition_true(services[variant], "Ready")
        )
    final_rule = route.get("spec", {}).get("rules", [{}])[0]
    rollback_route_valid = (
        route_metadata.get("annotations", {}).get("ioap.openai.com/stage")
        == "ROLLED_BACK"
        and final_rule.get("backendRefs")
        == [
            {
                "group": "",
                "kind": "Service",
                "name": "ioap-rul-stable-predictor",
                "port": 80,
                "weight": 100,
            }
        ]
        and not final_rule.get("filters")
    )
    shadow = probes["shadow"]
    canary = probes["canary_25"]
    rollback = probes["rollback"]
    canary_total = int(canary["request_count"])
    canary_fraction = float(canary["candidate_responses"]) / canary_total
    stage_results: dict[str, dict[str, int | float | bool]] = {
        "shadow": {
            **shadow,
            "passed": (
                int(shadow["request_count"]) == 1
                and int(shadow["stable_responses"]) == 1
                and int(shadow["candidate_responses"]) == 0
                and int(shadow["candidate_counter_delta"]) == 1
            ),
        },
        "canary_25": {
            **canary,
            "observed_candidate_fraction": canary_fraction,
            "passed": (
                canary_total >= 40
                and int(canary["stable_responses"])
                + int(canary["candidate_responses"])
                == canary_total
                and 0.15 <= canary_fraction <= 0.35
            ),
        },
        "rollback": {
            **rollback,
            "passed": (
                int(rollback["request_count"]) >= 10
                and int(rollback["stable_responses"])
                == int(rollback["request_count"])
                and int(rollback["candidate_responses"]) == 0
                and int(rollback["candidate_counter_delta"]) == 0
            ),
        },
    }
    gates = {
        "promotion_evidence_bound": True,
        "fixed_kind_context": context == "kind-ioap-gpu-promotion-lab",
        "inference_services_ready": workloads_ready,
        "route_accepted_and_resolved": route_ready,
        "shadow_mirror_observed": bool(stage_results["shadow"]["passed"]),
        "canary_25_distribution_observed": bool(stage_results["canary_25"]["passed"]),
        "rollback_stable_only_observed": bool(stage_results["rollback"]["passed"]),
        "rollback_route_is_stable_only": rollback_route_valid,
    }
    if not all(gates.values()):
        failed = sorted(name for name, passed in gates.items() if not passed)
        raise RulPromotionLabError("rul_kserve_rollout_failed:" + ",".join(failed))
    manifests = {
        stage: {
            "path": str(path),
            "sha256": "sha256:" + sha256(path.read_bytes()).hexdigest(),
        }
        for stage, path in (
            ("shadow", output_path.parent / "kserve-shadow.json"),
            ("canary_25", output_path.parent / "kserve-canary-25.json"),
            ("rolled_back", output_path.parent / "kserve-rolled-back.json"),
        )
    }
    evidence: dict[str, Any] = {
        "schema_version": "local-rul-kserve-rollout-evidence/v1",
        "classification": CLASSIFICATION,
        "production_claim": False,
        "training_and_gold_evaluation_gpu_execution": True,
        "kserve_inference_device": "cpu",
        "gpu_serving_claim": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "promotion_evidence": {
            "path": str(report_path),
            "evidence_chain_sha256": promotion["evidence_chain_sha256"],
            "artifact_content_hash": promotion["candidate"]["artifact_content_hash"],
            "gold_snapshot_sha256": promotion["frozen_data"]["gold_snapshot_sha256"],
        },
        "cluster": {
            "context": context,
            "namespace": K8S_NAMESPACE,
            "kserve_mode": "Standard",
            "route_generation": generation,
        },
        "runtime_image": {
            "tag": image,
            "id": image_metadata.get("Id"),
            "repo_digests": image_metadata.get("RepoDigests", []),
        },
        "live_probes": {
            "path": str(probe_results_path),
            "sha256": "sha256:" + sha256(probe_results_path.read_bytes()).hexdigest(),
            "source": probes["source"],
        },
        "workloads": workload_projection,
        "route": {
            "name": K8S_ROUTE,
            "stage": "ROLLED_BACK",
            "accepted": route_ready,
            "stable_only": rollback_route_valid,
        },
        "stages": stage_results,
        "manifests": manifests,
        "hard_gate_results": gates,
        "status": "RUL_LOCAL_KSERVE_ROLLOUT_PASSED",
    }
    evidence["evidence_chain_sha256"] = sha256(_canonical_bytes(evidence)).hexdigest()
    output_path.write_bytes(_canonical_bytes(evidence) + b"\n")
    return output_path


def _live_probe_results(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RulPromotionLabError("rul_kserve_probe_results_are_invalid") from exc
    if not isinstance(value, dict):
        raise RulPromotionLabError("rul_kserve_probe_results_are_invalid")
    unsigned = dict(value)
    chain = unsigned.pop("evidence_chain_sha256", None)
    required_fields = {
        "shadow": {
            "request_count",
            "stable_responses",
            "candidate_responses",
            "candidate_counter_delta",
        },
        "canary_25": {"request_count", "stable_responses", "candidate_responses"},
        "rollback": {
            "request_count",
            "stable_responses",
            "candidate_responses",
            "candidate_counter_delta",
        },
    }
    stages_valid = all(
        isinstance(value.get(stage), dict)
        and required.issubset(value[stage])
        and all(
            isinstance(value[stage][name], int) and value[stage][name] >= 0
            for name in required
        )
        for stage, required in required_fields.items()
    )
    if (
        value.get("schema_version") != "local-rul-kserve-live-probes/v1"
        or value.get("classification") != CLASSIFICATION
        or value.get("source") != "LIVE_HTTP_GATEWAY_AND_PROMETHEUS"
        or value.get("production_claim") is not False
        or chain != sha256(_canonical_bytes(unsigned)).hexdigest()
        or not stages_valid
    ):
        raise RulPromotionLabError("rul_kserve_probe_results_binding_failed")
    return value


def _capture_text(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RulPromotionLabError("rul_kserve_local_command_failed") from exc
    return result.stdout.strip()


def _capture_json(command: list[str]) -> Any:
    try:
        return json.loads(_capture_text(command))
    except json.JSONDecodeError as exc:
        raise RulPromotionLabError("rul_kserve_local_command_returned_invalid_json") from exc


def _resource_condition_true(document: dict[str, Any], condition: str) -> bool:
    generation = document.get("metadata", {}).get("generation")
    status = document.get("status", {})
    if status.get("observedGeneration") != generation:
        return False
    return any(
        item.get("type") == condition
        and item.get("status") == "True"
        and item.get("observedGeneration", generation) == generation
        for item in status.get("conditions", [])
    )


def _promotion_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RulPromotionLabError("rul_promotion_evidence_is_invalid") from exc
    if not isinstance(value, dict):
        raise RulPromotionLabError("rul_promotion_evidence_is_invalid")
    unsigned = dict(value)
    chain = unsigned.pop("evidence_chain_sha256", None)
    selection = value.get("selection")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("classification") != CLASSIFICATION
        or value.get("actual_gpu_execution") is not True
        or value.get("model_training_simulated") is not False
        or value.get("production_claim") is not False
        or chain != sha256(_canonical_bytes(unsigned)).hexdigest()
        or not isinstance(selection, dict)
        or selection.get("status") != "SELECTED_FOR_LOCAL_KSERVE_PROMOTION"
        or selection.get("all_hard_gates_passed") is not True
    ):
        raise RulPromotionLabError("rul_promotion_evidence_binding_failed")
    return value


def _local_inference_service(
    *,
    name: str,
    image: str,
    args: list[str],
    variant: str,
) -> dict[str, Any]:
    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": name,
            "namespace": K8S_NAMESPACE,
            "annotations": {"serving.kserve.io/deploymentMode": "Standard"},
            "labels": {
                "ioap.openai.com/rul-promotion-lab": "true",
                "ioap.openai.com/model-variant": variant,
                "ioap.openai.com/classification": "simulated-non-production",
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
                        "imagePullPolicy": "Never",
                        "args": args,
                        "env": [
                            {"name": "OMP_NUM_THREADS", "value": "2"},
                            {"name": "MKL_NUM_THREADS", "value": "2"},
                        ],
                        "ports": [{"name": "http1", "containerPort": 8080}],
                        "securityContext": hardened_container_security_context(),
                        "startupProbe": {
                            "httpGet": {"path": "/health/ready", "port": 8080},
                            "periodSeconds": 2,
                            "failureThreshold": 90,
                        },
                        "readinessProbe": {
                            "httpGet": {"path": "/health/ready", "port": 8080},
                            "periodSeconds": 3,
                            "failureThreshold": 10,
                        },
                        "resources": {
                            "requests": {"cpu": "250m", "memory": "512Mi"},
                            "limits": {"cpu": "2", "memory": "2Gi"},
                        },
                    }
                ],
            }
        },
    }


def _local_rul_route(stage: RulRolloutStage) -> dict[str, Any]:
    stable = {"name": f"{K8S_STABLE_SERVICE}-predictor", "port": 80}
    candidate = {"name": f"{K8S_CANDIDATE_SERVICE}-predictor", "port": 80}
    if stage == "SHADOW":
        rule: dict[str, Any] = {
            "matches": [{"path": {"type": "PathPrefix", "value": "/v1/rul"}}],
            "backendRefs": [{**stable, "weight": 100}],
            "filters": [
                {"type": "RequestMirror", "requestMirror": {"backendRef": candidate}}
            ],
        }
    elif stage == "CANARY_25":
        rule = {
            "matches": [{"path": {"type": "PathPrefix", "value": "/v1/rul"}}],
            "backendRefs": [
                {**stable, "weight": 75},
                {**candidate, "weight": 25},
            ],
        }
    elif stage == "ROLLED_BACK":
        rule = {
            "matches": [{"path": {"type": "PathPrefix", "value": "/v1/rul"}}],
            "backendRefs": [{**stable, "weight": 100}],
        }
    else:
        raise RulPromotionLabError("rul_rollout_stage_is_invalid")
    rule["timeouts"] = {"request": "30s", "backendRequest": "30s"}
    return {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "HTTPRoute",
        "metadata": {
            "name": K8S_ROUTE,
            "namespace": K8S_NAMESPACE,
            "annotations": {"ioap.openai.com/stage": stage},
        },
        "spec": {
            "parentRefs": [{"name": K8S_GATEWAY}],
            "hostnames": [K8S_HOSTNAME],
            "rules": [rule],
        },
    }


def _training_config() -> CompiledRulTransformerConfig:
    return CompiledRulTransformerConfig(
        method="RUL_TRANSFORMER",
        base_model_revision="rul-transformer-v1",
        max_steps=1_500,
        effective_batch_size=32,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        evaluation_interval=250,
        save_steps=1_500,
        logging_steps=50,
        learning_rate=2e-3,
        weight_decay=0.01,
        precision="bfloat16",
        seed=42,
        data_seed=84,
        gpu_hourly_cost_usd=0.0,
        sequence_contract_version="industrial-rul-sequence-v1",
        signal_order=SIGNAL_ORDER,
        max_sequence_length=16,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.05,
        objective="quantile_regression",
        quantiles=(0.1, 0.5, 0.9),
        target_transform="log1p_minutes",
    )


def _samples(
    split: str,
    *,
    asset_count: int,
    samples_per_asset: int,
    offset: int,
) -> tuple[RulSequenceSample, ...]:
    samples: list[RulSequenceSample] = []
    noise_values = (-60.0, -35.0, -15.0, -5.0, 5.0, 15.0, 35.0, 60.0)
    for asset_index in range(asset_count):
        asset_id = f"rul-{split}-asset-{asset_index:03d}"
        asset_bias = (asset_index % 5 - 2) * 0.006
        for sample_index in range(samples_per_asset):
            global_index = offset + asset_index * samples_per_asset + sample_index
            risk = 0.05 + 0.90 * (
                ((sample_index * 7 + asset_index * 11) % samples_per_asset)
                / max(samples_per_asset - 1, 1)
            )
            risk = min(0.98, max(0.02, risk + asset_bias))
            noise = noise_values[global_index % len(noise_values)]
            target = max(15.0, 75.0 + (1.0 - risk) * 900.0 + noise)
            sequence: list[tuple[float, ...]] = []
            for step in range(12):
                progress = step / 11
                sequence.append(
                    (
                        2.0 + 8.0 * risk + 0.35 * progress,
                        44.0 + 38.0 * risk + 1.5 * progress,
                        7.0 + 5.0 * risk + 0.2 * progress,
                        1_520.0 - 120.0 * risk - 3.0 * progress,
                    )
                )
            samples.append(
                RulSequenceSample(
                    window_id=f"rul-{split}-window-{global_index:05d}",
                    asset_id=asset_id,
                    sequence=tuple(sequence),
                    mask=tuple((True, True, True, True) for _ in sequence),
                    lead_time_minutes=target,
                )
            )
    return tuple(samples)


def _evaluation_case(sample: RulSequenceSample) -> EvaluationCase:
    return EvaluationCase(
        case_id=sample.window_id,
        prompt=(),
        expected_output={"lead_time_minutes": sample.lead_time_minutes},
        slices={"asset_id": sample.asset_id, "risk": "HIGH"},
        required_gates=(),
        allowed_citations=(),
        forbidden_substrings=(),
        expected_decision=None,
        risk="HIGH",
        contexts=(),
        rul=RulEvaluationContract(
            asset_id=sample.asset_id,
            schema_version="rotating-equipment.telemetry.v1",
            sequence=sample.sequence,
            mask=sample.mask,
            lead_time_minutes=sample.lead_time_minutes,
            outcome_evidence_digest="sha256:"
            + sha256(f"{sample.window_id}:{sample.lead_time_minutes}".encode()).hexdigest(),
        ),
    )


def _sample_document(sample: RulSequenceSample) -> dict[str, Any]:
    return {
        "window_id": sample.window_id,
        "asset_id": sample.asset_id,
        "sequence": [list(row) for row in sample.sequence],
        "mask": [list(row) for row in sample.mask],
        "lead_time_minutes": sample.lead_time_minutes,
    }


def _sample_hash(samples: tuple[RulSequenceSample, ...]) -> str:
    return (
        "sha256:"
        + sha256(_canonical_bytes([_sample_document(item) for item in samples])).hexdigest()
    )


def _require_disjoint_assets(*splits: tuple[RulSequenceSample, ...]) -> None:
    groups = [{item.asset_id for item in split} for split in splits]
    for index, current in enumerate(groups):
        if any(current & other for other in groups[index + 1 :]):
            raise RulPromotionLabError("rul_lab_asset_group_leakage")


def _quantiles(
    values: tuple[float, ...], probabilities: tuple[float, float, float]
) -> tuple[float, float, float]:
    ordered = sorted(values)
    if not ordered:
        raise RulPromotionLabError("rul_lab_baseline_is_empty")
    calculated: list[float] = []
    for probability in probabilities:
        position = (len(ordered) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            calculated.append(ordered[lower])
        else:
            fraction = position - lower
            calculated.append(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)
    return calculated[0], calculated[1], calculated[2]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local RUL GPU promotion lab")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train-evaluate")
    training.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/rul-promotion-lab"),
    )
    render = commands.add_parser("render-kserve")
    render.add_argument("--report", type=Path, required=True)
    render.add_argument("--image", required=True)
    render.add_argument(
        "--stage",
        choices=("SHADOW", "CANARY_25", "ROLLED_BACK"),
        required=True,
    )
    render.add_argument("--output", type=Path, required=True)
    record = commands.add_parser("record-kserve")
    record.add_argument("--report", type=Path, required=True)
    record.add_argument("--image", required=True)
    record.add_argument("--probe-results", type=Path, required=True)
    record.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "train-evaluate":
        report = run_train_evaluate(args.output_root.resolve())
        print("RUL_LOCAL_GPU_GOLD_EVALUATION_PASSED")
        print(report)
        return 0
    if args.command == "render-kserve":
        output = render_kserve_manifest(
            args.report.resolve(),
            image=args.image,
            stage=args.stage,
            output_path=args.output.resolve(),
        )
        print(output)
        return 0
    if args.command == "record-kserve":
        output = record_kserve_rollout(
            args.report.resolve(),
            image=args.image,
            probe_results_path=args.probe_results.resolve(),
            output_path=args.output.resolve(),
        )
        print("RUL_LOCAL_KSERVE_ROLLOUT_PASSED")
        print(output)
        return 0
    raise AssertionError("unreachable")


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
