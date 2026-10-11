"""Fail-closed evidence contract for the dual-GGUF GPU promotion stage."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

from industrial_ops_agent.evaluation.edge_fidelity import (
    stable_edge_evaluation_preserves_source,
)
from industrial_ops_agent.experiments.service import MANDATORY_HARD_GATES
from industrial_ops_agent.training.gpu_promotion_lab import (
    GpuPromotionLabContractError,
)
from industrial_ops_agent.training.gpu_promotion_rollout import (
    load_promotion_evidence,
)

SCHEMA_VERSION = "local-gpu-promotion-gguf/v2"
STATUS = "GPU_MODEL_DUAL_GGUF_QUANTIZATION_PASSED"
CLASSIFICATION = "LOCAL_STAGING_PROJECT_AUTHORIZED"
DATA_CLASSIFICATION = "SYNTHETIC_DATA"
AUTHORIZATION_CLASSIFICATION = "PROJECT_OWNED_SYNTHETIC"
ROLE_ATTESTATION = "LOCAL_ROLE_SIMULATION"
LLAMA_CPP_COMMIT = "7ba604f1cb61cd14898138e9abc0b4ff2601f180"
QUANTIZATION_PROFILE_ID = "gguf-q4_k_m-llama_cpp"
QUANTIZATION_SCHEME = "Q4_K_M"
TARGET_RUNTIME = "llama.cpp"

_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_VARIANTS = ("stable", "candidate")


@dataclass(frozen=True, slots=True)
class GgufEdgeEvaluationEvidence:
    job_id: str
    evaluation_id: str
    suite_id: str
    policy_id: str
    candidate_experiment_id: str
    baseline_experiment_id: str
    sample_count: int
    candidate_score: float
    baseline_score: float
    quality_delta: float
    latency_improvement: float
    cost_improvement: float
    decision: str
    hard_gate_results: dict[str, bool]
    slice_metrics: dict[str, Any]
    report_hash: str
    case_evidence_content_hash: str


@dataclass(frozen=True, slots=True)
class GgufVariantEvidence:
    role: Literal["stable", "candidate"]
    source_method: str
    source_experiment_id: str
    source_adapter_artifact_id: str
    source_adapter_object_key: str
    source_adapter_content_hash: str
    source_evaluation_id: str
    quantization_experiment_id: str
    mlflow_run_id: str
    quantized_artifact_id: str
    quantized_artifact_object_key: str
    quantized_artifact_content_hash: str
    quantized_artifact_size_bytes: int
    local_model_path: str
    model_file: str
    model_file_sha256: str
    model_file_size_bytes: int
    quantization_manifest_path: str
    quantization_manifest_sha256: str
    smoke_output_sha256: str
    llama_cpp_binary_sha256: str
    edge_evaluation: GgufEdgeEvaluationEvidence


@dataclass(frozen=True, slots=True)
class GgufPromotionEvidence:
    git_commit: str
    quantization_image: str
    quantization_image_digest: str
    training_evidence_chain_sha256: str
    stable: GgufVariantEvidence
    candidate: GgufVariantEvidence
    evidence_chain_sha256: str


def load_gguf_promotion_evidence(
    path: Path,
    *,
    repo_root: Path,
    training_evidence_path: Path,
) -> GgufPromotionEvidence:
    """Verify the receipt, both GGUF files and their immutable training sources."""

    document = _read_document(path, "GGUF_evidence_is_invalid")
    chain = document.get("evidence_chain_sha256")
    unsigned = dict(document)
    unsigned.pop("evidence_chain_sha256", None)
    if chain != _digest(unsigned):
        raise GpuPromotionLabContractError("GGUF_evidence_chain_mismatch")
    expected_top_level = {
        "schema_version",
        "classification",
        "data_classification",
        "authorization_classification",
        "role_attestation",
        "generated_at",
        "production_claim",
        "enterprise_production_data",
        "actual_quantization_execution",
        "quantization_simulated",
        "source_training",
        "runtime",
        "variants",
        "status",
        "evidence_chain_sha256",
    }
    if (
        set(document) != expected_top_level
        or document.get("schema_version") != SCHEMA_VERSION
        or document.get("classification") != CLASSIFICATION
        or document.get("data_classification") != DATA_CLASSIFICATION
        or document.get("authorization_classification")
        != AUTHORIZATION_CLASSIFICATION
        or document.get("role_attestation") != ROLE_ATTESTATION
        or document.get("production_claim") is not False
        or document.get("enterprise_production_data") is not False
        or document.get("actual_quantization_execution") is not True
        or document.get("quantization_simulated") is not False
        or document.get("status") != STATUS
    ):
        raise GpuPromotionLabContractError("GGUF_evidence_contract_changed")
    training = load_promotion_evidence(training_evidence_path)
    source = document.get("source_training")
    runtime = document.get("runtime")
    variants = document.get("variants")
    if (
        not isinstance(source, dict)
        or source.get("evidence_chain_sha256") != training.evidence_chain_sha256
        or source.get("git_commit") != training.git_commit
        or not isinstance(source.get("path"), str)
        or not isinstance(runtime, dict)
        or not _GIT_SHA.fullmatch(str(runtime.get("git_commit", "")))
        or not isinstance(runtime.get("quantization_image"), str)
        or not _IMAGE_DIGEST.fullmatch(
            str(runtime.get("quantization_image_digest", ""))
        )
        or runtime.get("llama_cpp_commit") != LLAMA_CPP_COMMIT
        or runtime.get("quantization_profile_id") != QUANTIZATION_PROFILE_ID
        or runtime.get("scheme") != QUANTIZATION_SCHEME
        or runtime.get("target_runtime") != TARGET_RUNTIME
        or not isinstance(variants, dict)
        or set(variants) != set(_VARIANTS)
    ):
        raise GpuPromotionLabContractError("GGUF_evidence_binding_failed")
    stable = _load_variant(
        cast(dict[str, Any], variants)["stable"],
        role="stable",
        repo_root=repo_root,
        expected_source={
            "method": training.stable_method,
            "experiment_id": training.stable_experiment_id,
            "adapter_artifact_id": training.stable_adapter_artifact_id,
            "adapter_object_key": training.stable_adapter_object_key,
            "adapter_content_hash": training.stable_adapter_content_hash,
            "evaluation_id": training.stable_evaluation_id,
        },
    )
    candidate = _load_variant(
        cast(dict[str, Any], variants)["candidate"],
        role="candidate",
        repo_root=repo_root,
        expected_source={
            "method": training.method,
            "experiment_id": training.experiment_id,
            "adapter_artifact_id": training.adapter_artifact_id,
            "adapter_object_key": training.adapter_object_key,
            "adapter_content_hash": training.adapter_content_hash,
            "evaluation_id": training.evaluation_id,
        },
    )
    if stable.quantization_experiment_id == candidate.quantization_experiment_id:
        raise GpuPromotionLabContractError("GGUF_quantization_experiments_are_not_distinct")
    return GgufPromotionEvidence(
        git_commit=cast(str, runtime["git_commit"]),
        quantization_image=cast(str, runtime["quantization_image"]),
        quantization_image_digest=cast(str, runtime["quantization_image_digest"]),
        training_evidence_chain_sha256=training.evidence_chain_sha256,
        stable=stable,
        candidate=candidate,
        evidence_chain_sha256=cast(str, chain),
    )


def finalize_gguf_evidence(document: dict[str, Any]) -> dict[str, Any]:
    """Attach a canonical digest and validate structure before persistence."""

    unsigned = dict(document)
    unsigned.pop("evidence_chain_sha256", None)
    return {**unsigned, "evidence_chain_sha256": _digest(unsigned)}


def write_gguf_evidence(path: Path, document: dict[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def relative_file_binding(repo_root: Path, path: Path) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    target = path.resolve(strict=True)
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise GpuPromotionLabContractError("GGUF_artifact_path_escaped_repository") from exc
    if not target.is_file():
        raise GpuPromotionLabContractError("GGUF_artifact_is_not_a_file")
    content_hash = _file_sha256(target)
    return {
        "path": relative.as_posix(),
        "size_bytes": target.stat().st_size,
        "sha256": content_hash,
    }


def _load_variant(
    value: object,
    *,
    role: Literal["stable", "candidate"],
    repo_root: Path,
    expected_source: dict[str, str],
) -> GgufVariantEvidence:
    if not isinstance(value, dict):
        raise GpuPromotionLabContractError("GGUF_variant_evidence_is_invalid")
    expected_fields = {
        "role",
        "source_method",
        "source_experiment_id",
        "source_adapter_artifact_id",
        "source_adapter_object_key",
        "source_adapter_content_hash",
        "source_evaluation_id",
        "quantization_experiment_id",
        "mlflow_run_id",
        "mlflow_status",
        "quantized_artifact_id",
        "quantized_artifact_object_key",
        "quantized_artifact_content_hash",
        "quantized_artifact_size_bytes",
        "local_model",
        "quantization_manifest",
        "runtime",
        "llama_cpp_smoke",
        "edge_evaluation",
    }
    local_model = value.get("local_model")
    manifest = value.get("quantization_manifest")
    runtime = value.get("runtime")
    smoke = value.get("llama_cpp_smoke")
    edge_evaluation = value.get("edge_evaluation")
    if (
        set(value) != expected_fields
        or value.get("role") != role
        or value.get("source_method") != expected_source["method"]
        or value.get("source_experiment_id") != expected_source["experiment_id"]
        or value.get("source_adapter_artifact_id")
        != expected_source["adapter_artifact_id"]
        or value.get("source_adapter_object_key")
        != expected_source["adapter_object_key"]
        or value.get("source_adapter_content_hash")
        != expected_source["adapter_content_hash"]
        or value.get("source_evaluation_id") != expected_source["evaluation_id"]
        or not isinstance(value.get("quantization_experiment_id"), str)
        or not isinstance(value.get("mlflow_run_id"), str)
        or value.get("mlflow_status") != "FINISHED"
        or not isinstance(value.get("quantized_artifact_id"), str)
        or not isinstance(value.get("quantized_artifact_object_key"), str)
        or not _SHA256.fullmatch(
            str(value.get("quantized_artifact_content_hash", ""))
        )
        or not _positive_int(value.get("quantized_artifact_size_bytes"))
        or not isinstance(local_model, dict)
        or not isinstance(manifest, dict)
        or not isinstance(runtime, dict)
        or runtime
        != {
            "engine": TARGET_RUNTIME,
            "gguf_type": QUANTIZATION_SCHEME,
            "model_file": "model-q4_k_m.gguf",
            "target_runtime": TARGET_RUNTIME,
            "version": LLAMA_CPP_COMMIT,
        }
        or not isinstance(smoke, dict)
        or smoke.get("actual_llama_cpp_execution") is not True
        or smoke.get("exit_code") != 0
        or not _positive_int(smoke.get("generated_token_count"))
        or not _SHA256.fullmatch(str(smoke.get("prompt_sha256", "")))
        or not _SHA256.fullmatch(str(smoke.get("output_sha256", "")))
        or not _SHA256.fullmatch(str(smoke.get("binary_sha256", "")))
    ):
        raise GpuPromotionLabContractError("GGUF_variant_evidence_is_invalid")
    model_path = _bound_file(repo_root, local_model, expected_magic=b"GGUF")
    manifest_path = _bound_file(repo_root, manifest)
    verified_evaluation = _load_edge_evaluation(
        edge_evaluation,
        role=role,
        candidate_experiment_id=cast(str, value["quantization_experiment_id"]),
        baseline_experiment_id=expected_source["experiment_id"],
    )
    manifest_document = _read_document(
        manifest_path, "GGUF_quantization_manifest_is_invalid"
    )
    configuration = manifest_document.get("configuration")
    if (
        manifest_document.get("schema_version")
        != "industrial-ops-quantized-model/v1"
        or manifest_document.get("experiment_id")
        != value.get("quantization_experiment_id")
        or manifest_document.get("approval_inheritance") is not False
        or not isinstance(configuration, dict)
        or configuration.get("algorithm") != "GGUF"
        or configuration.get("scheme") != QUANTIZATION_SCHEME
        or configuration.get("target_runtime") != "LLAMA_CPP"
        or manifest_document.get("runtime") != runtime
        or manifest_document.get("manifest_sha256")
        != _digest(
            {
                key: item
                for key, item in manifest_document.items()
                if key != "manifest_sha256"
            }
        )
    ):
        raise GpuPromotionLabContractError("GGUF_quantization_manifest_binding_failed")
    return GgufVariantEvidence(
        role=role,
        source_method=cast(str, value["source_method"]),
        source_experiment_id=cast(str, value["source_experiment_id"]),
        source_adapter_artifact_id=cast(str, value["source_adapter_artifact_id"]),
        source_adapter_object_key=cast(str, value["source_adapter_object_key"]),
        source_adapter_content_hash=cast(str, value["source_adapter_content_hash"]),
        source_evaluation_id=cast(str, value["source_evaluation_id"]),
        quantization_experiment_id=cast(str, value["quantization_experiment_id"]),
        mlflow_run_id=cast(str, value["mlflow_run_id"]),
        quantized_artifact_id=cast(str, value["quantized_artifact_id"]),
        quantized_artifact_object_key=cast(str, value["quantized_artifact_object_key"]),
        quantized_artifact_content_hash=cast(
            str, value["quantized_artifact_content_hash"]
        ),
        quantized_artifact_size_bytes=cast(int, value["quantized_artifact_size_bytes"]),
        local_model_path=cast(str, local_model["path"]),
        model_file=Path(model_path).name,
        model_file_sha256=cast(str, local_model["sha256"]),
        model_file_size_bytes=cast(int, local_model["size_bytes"]),
        quantization_manifest_path=cast(str, manifest["path"]),
        quantization_manifest_sha256=cast(str, manifest["sha256"]),
        smoke_output_sha256=cast(str, smoke["output_sha256"]),
        llama_cpp_binary_sha256=cast(str, smoke["binary_sha256"]),
        edge_evaluation=verified_evaluation,
    )


def _load_edge_evaluation(
    value: object,
    *,
    role: Literal["stable", "candidate"],
    candidate_experiment_id: str,
    baseline_experiment_id: str,
) -> GgufEdgeEvaluationEvidence:
    if not isinstance(value, dict):
        raise GpuPromotionLabContractError("GGUF_edge_evaluation_is_invalid")
    expected_fields = {
        "job_id",
        "candidate_experiment_id",
        "baseline_experiment_id",
        "suite_id",
        "suite_tier",
        "sample_count",
        "policy_id",
        "target_profile",
        "evaluation_id",
        "decision",
        "candidate_score",
        "baseline_score",
        "quality_delta",
        "latency_improvement",
        "cost_improvement",
        "hard_gate_results",
        "slice_metrics",
        "report_hash",
        "case_evidence",
    }
    identity_fields = (
        "job_id",
        "suite_id",
        "policy_id",
        "evaluation_id",
    )
    scores = (
        "candidate_score",
        "baseline_score",
        "quality_delta",
        "latency_improvement",
        "cost_improvement",
    )
    hard_gates = value.get("hard_gate_results")
    slice_metrics = value.get("slice_metrics")
    case_evidence = value.get("case_evidence")
    if (
        set(value) != expected_fields
        or any(
            not isinstance(value.get(field), str) or not str(value[field]).strip()
            for field in identity_fields
        )
        or value.get("candidate_experiment_id") != candidate_experiment_id
        or value.get("baseline_experiment_id") != baseline_experiment_id
        or value.get("suite_tier") != "PROJECT_AUTHORIZED"
        or not _positive_int(value.get("sample_count"))
        or value.get("target_profile") != "EDGE_MODEL_COMPONENT"
        or value.get("decision") not in {"CANDIDATE", "REJECTED"}
        or any(not _finite(value.get(field)) for field in scores)
        or not isinstance(hard_gates, dict)
        or set(hard_gates) != set(MANDATORY_HARD_GATES)
        or not all(isinstance(item, bool) for item in hard_gates.values())
        or not isinstance(slice_metrics, dict)
        or not _SHA256.fullmatch(str(value.get("report_hash", "")))
        or not _case_evidence_is_valid(case_evidence)
    ):
        raise GpuPromotionLabContractError("GGUF_edge_evaluation_is_invalid")
    if role == "candidate":
        accepted = value.get("decision") == "CANDIDATE" and all(
            item is True for item in hard_gates.values()
        )
    else:
        accepted = stable_edge_evaluation_preserves_source(
            decision=value.get("decision"),
            candidate_score=value.get("candidate_score"),
            baseline_score=value.get("baseline_score"),
            quality_delta=value.get("quality_delta"),
            cost_improvement=value.get("cost_improvement"),
            hard_gate_results=hard_gates,
            slice_metrics=slice_metrics,
        )
    if not accepted:
        raise GpuPromotionLabContractError("GGUF_edge_evaluation_is_invalid")
    typed_case_evidence = cast(dict[str, Any], case_evidence)
    return GgufEdgeEvaluationEvidence(
        job_id=cast(str, value["job_id"]),
        evaluation_id=cast(str, value["evaluation_id"]),
        suite_id=cast(str, value["suite_id"]),
        policy_id=cast(str, value["policy_id"]),
        candidate_experiment_id=candidate_experiment_id,
        baseline_experiment_id=baseline_experiment_id,
        sample_count=cast(int, value["sample_count"]),
        candidate_score=float(value["candidate_score"]),
        baseline_score=float(value["baseline_score"]),
        quality_delta=float(value["quality_delta"]),
        latency_improvement=float(value["latency_improvement"]),
        cost_improvement=float(value["cost_improvement"]),
        decision=cast(str, value["decision"]),
        hard_gate_results=cast(dict[str, bool], hard_gates),
        slice_metrics=cast(dict[str, Any], slice_metrics),
        report_hash=cast(str, value["report_hash"]),
        case_evidence_content_hash=cast(str, typed_case_evidence["content_hash"]),
    )


def _case_evidence_is_valid(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    metadata = value.get("metadata")
    return bool(
        set(value) == {"artifact_id", "object_key", "content_hash", "size_bytes", "metadata"}
        and isinstance(value.get("artifact_id"), str)
        and value.get("artifact_id")
        and isinstance(value.get("object_key"), str)
        and value.get("object_key")
        and _SHA256.fullmatch(str(value.get("content_hash", "")))
        and _positive_int(value.get("size_bytes"))
        and isinstance(metadata, dict)
        and metadata.get("candidate_method") == "QUANTIZATION"
        and _SHA256.fullmatch(str(metadata.get("candidate_artifact_hash", "")))
        and _SHA256.fullmatch(str(metadata.get("suite_manifest_hash", "")))
    )


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _bound_file(
    repo_root: Path,
    binding: dict[str, Any],
    *,
    expected_magic: bytes | None = None,
) -> Path:
    if set(binding) != {"path", "size_bytes", "sha256"}:
        raise GpuPromotionLabContractError("GGUF_local_file_binding_is_invalid")
    relative = binding.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or not _positive_int(binding.get("size_bytes"))
        or not _SHA256.fullmatch(str(binding.get("sha256", "")))
    ):
        raise GpuPromotionLabContractError("GGUF_local_file_binding_is_invalid")
    root = repo_root.resolve(strict=True)
    try:
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as exc:
        raise GpuPromotionLabContractError("GGUF_local_file_is_not_visible") from exc
    if (
        not path.is_file()
        or path.stat().st_size != binding["size_bytes"]
        or _file_sha256(path) != binding["sha256"]
    ):
        raise GpuPromotionLabContractError("GGUF_local_file_integrity_failed")
    if expected_magic is not None:
        with path.open("rb") as stream:
            if stream.read(len(expected_magic)) != expected_magic:
                raise GpuPromotionLabContractError("GGUF_model_magic_is_invalid")
    return path


def _read_document(path: Path, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GpuPromotionLabContractError(reason) from exc
    if not isinstance(value, dict):
        raise GpuPromotionLabContractError(reason)
    return cast(dict[str, Any], value)


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
