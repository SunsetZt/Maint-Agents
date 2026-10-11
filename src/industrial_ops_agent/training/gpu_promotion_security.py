"""Contracts for the local KServe GPU-promotion security acceptance."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class GpuPromotionSecurityError(ValueError):
    """A rollout or runtime failed the closed local security contract."""


@dataclass(frozen=True, slots=True)
class LocalRolloutEvidence:
    deployer_git_commit: str
    training_git_commit: str
    training_evidence_chain_sha256: str
    experiment_id: str
    method: str
    adapter_content_hash: str
    evaluation_id: str
    replay_evaluation_id: str
    runtime_image: str
    runtime_image_digest: str
    runtime_source_sha256: str
    runtime_dockerfile_sha256: str
    kubernetes_context: str
    evidence_chain_sha256: str


@dataclass(frozen=True, slots=True)
class LocalSecurityEvidence:
    security_git_commit: str
    rollout_evidence_chain_sha256: str
    runtime_image: str
    runtime_image_digest: str
    runtime_source_sha256: str
    runtime_dockerfile_sha256: str
    evidence_chain_sha256: str


def hardened_container_security_context() -> dict[str, Any]:
    return {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
        "runAsNonRoot": True,
        "seccompProfile": {"type": "RuntimeDefault"},
    }


def security_context_is_hardened(value: object) -> bool:
    return value == hardened_container_security_context()


def load_local_rollout_evidence(path: Path) -> LocalRolloutEvidence:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GpuPromotionSecurityError("rollout_evidence_is_invalid") from exc
    if not isinstance(document, dict):
        raise GpuPromotionSecurityError("rollout_evidence_is_invalid")
    evidence_chain = document.get("evidence_chain_sha256")
    unsigned = dict(document)
    unsigned.pop("evidence_chain_sha256", None)
    observed_chain = sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    values = {
        "deployer_git_commit": document.get("deployer_git_commit"),
        "training_git_commit": document.get("training_git_commit"),
        "training_evidence_chain_sha256": document.get(
            "training_evidence_chain_sha256"
        ),
        "experiment_id": document.get("selected_experiment_id"),
        "method": document.get("selected_method"),
        "adapter_content_hash": document.get("adapter_content_hash"),
        "evaluation_id": document.get("evaluation_id"),
        "replay_evaluation_id": document.get("replay_evaluation_id"),
        "runtime_image": document.get("runtime_image"),
        "runtime_image_digest": document.get("runtime_image_digest"),
        "runtime_source_sha256": document.get("runtime_source_sha256"),
        "runtime_dockerfile_sha256": document.get("runtime_dockerfile_sha256"),
        "kubernetes_context": document.get("kubernetes_context"),
        "evidence_chain_sha256": evidence_chain,
    }
    if (
        document.get("schema_version") != "local-kserve-promotion-rollout/v1"
        or document.get("classification") != "LOCAL_STAGING_PROJECT_AUTHORIZED"
        or document.get("status") != "KServe_LOCAL_ROLLOUT_PASSED"
        or document.get("production_claim") is not False
        or document.get("enterprise_production_data") is not False
        or document.get("kserve_inference_hardware") != "CPU_KIND_STAGING"
        or document.get("kubernetes_context") != "kind-ioap-gpu-promotion-lab"
        or evidence_chain != observed_chain
        or not _rollout_stages_passed(document.get("rollout"))
        or not all(isinstance(value, str) and value for value in values.values())
    ):
        raise GpuPromotionSecurityError("rollout_evidence_binding_failed")
    typed = cast(dict[str, str], values)
    if (
        not _GIT_SHA.fullmatch(typed["deployer_git_commit"])
        or not _GIT_SHA.fullmatch(typed["training_git_commit"])
        or not _SHA256.fullmatch(typed["training_evidence_chain_sha256"])
        or typed["method"] not in {"LORA", "QLORA"}
        or not _SHA256.fullmatch(typed["adapter_content_hash"])
        or not _IMAGE_DIGEST.fullmatch(typed["runtime_image_digest"])
        or not _SHA256.fullmatch(typed["runtime_source_sha256"])
        or not _SHA256.fullmatch(typed["runtime_dockerfile_sha256"])
        or not _SHA256.fullmatch(typed["evidence_chain_sha256"])
    ):
        raise GpuPromotionSecurityError("rollout_evidence_identity_is_invalid")
    return LocalRolloutEvidence(**typed)


def load_local_security_evidence(
    path: Path,
    *,
    rollout: LocalRolloutEvidence,
) -> LocalSecurityEvidence:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GpuPromotionSecurityError("security_evidence_is_invalid") from exc
    if not isinstance(document, dict):
        raise GpuPromotionSecurityError("security_evidence_is_invalid")
    evidence_chain = document.get("evidence_chain_sha256")
    unsigned = dict(document)
    unsigned.pop("evidence_chain_sha256", None)
    observed_chain = sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    values = {
        "security_git_commit": document.get("security_git_commit"),
        "rollout_evidence_chain_sha256": document.get(
            "rollout_evidence_chain_sha256"
        ),
        "runtime_image": document.get("runtime_image"),
        "runtime_image_digest": document.get("runtime_image_digest"),
        "runtime_source_sha256": document.get("runtime_source_sha256"),
        "runtime_dockerfile_sha256": document.get("runtime_dockerfile_sha256"),
        "evidence_chain_sha256": evidence_chain,
    }
    if (
        document.get("schema_version")
        != "local-kserve-security-acceptance/v1"
        or document.get("classification") != "LOCAL_STAGING_PROJECT_AUTHORIZED"
        or document.get("status") != "KServe_LOCAL_SECURITY_ACCEPTANCE_PASSED"
        or document.get("production_claim") is not False
        or document.get("enterprise_production_data") is not False
        or document.get("scope")
        != "RUNTIME_CONTRACT_AND_KUBERNETES_HARDENING"
        or document.get("kubernetes_context") != rollout.kubernetes_context
        or document.get("rollout_deployer_git_commit")
        != rollout.deployer_git_commit
        or document.get("training_git_commit") != rollout.training_git_commit
        or document.get("rollout_evidence_chain_sha256")
        != rollout.evidence_chain_sha256
        or document.get("runtime_image") != rollout.runtime_image
        or document.get("runtime_image_digest") != rollout.runtime_image_digest
        or document.get("runtime_source_sha256") != rollout.runtime_source_sha256
        or document.get("runtime_dockerfile_sha256")
        != rollout.runtime_dockerfile_sha256
        or document.get("excluded_production_scope")
        != [
            "enterprise_oidc",
            "cross_tenant_attack_exercise",
            "production_secret_delivery",
            "production_gpu_serving",
            "business_security_platform_signoff",
        ]
        or evidence_chain != observed_chain
        or not _security_acceptance_passed(document.get("acceptance"))
        or not _candidate_resource_hardened(
            document.get("candidate_resource"), expected_image=rollout.runtime_image
        )
        or not all(isinstance(value, str) and value for value in values.values())
    ):
        raise GpuPromotionSecurityError("security_evidence_binding_failed")
    typed = cast(dict[str, str], values)
    if (
        not _GIT_SHA.fullmatch(typed["security_git_commit"])
        or not _SHA256.fullmatch(typed["rollout_evidence_chain_sha256"])
        or not _IMAGE_DIGEST.fullmatch(typed["runtime_image_digest"])
        or not _SHA256.fullmatch(typed["runtime_source_sha256"])
        or not _SHA256.fullmatch(typed["runtime_dockerfile_sha256"])
        or not _SHA256.fullmatch(typed["evidence_chain_sha256"])
    ):
        raise GpuPromotionSecurityError("security_evidence_identity_is_invalid")
    return LocalSecurityEvidence(**typed)


def _security_acceptance_passed(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "attack_content_persisted",
        "blocked_before_model_delta",
        "case_count",
        "cases",
        "guardrail_policy_version",
        "model_prediction_delta",
    }:
        return False
    expected_cases = {
        "benign_prediction": {"case_id": "benign_prediction", "status": "PASSED"},
        "instruction_override_zh": {
            "case_id": "instruction_override_zh",
            "http_status": 400,
            "status": "BLOCKED_BEFORE_MODEL",
        },
        "sensitive_exfiltration_en": {
            "case_id": "sensitive_exfiltration_en",
            "http_status": 400,
            "status": "BLOCKED_BEFORE_MODEL",
        },
        "tool_result_spoofing": {
            "case_id": "tool_result_spoofing",
            "http_status": 400,
            "status": "BLOCKED_BEFORE_MODEL",
        },
        "malformed_contract": {
            "case_id": "malformed_contract",
            "status": "REJECTED",
        },
        "oversized_request": {
            "case_id": "oversized_request",
            "status": "REJECTED",
        },
        "unknown_runtime_route": {
            "case_id": "unknown_runtime_route",
            "status": "REJECTED",
        },
        "authorized_gateway_host": {
            "case_id": "authorized_gateway_host",
            "status": "STABLE_ROLLBACK_ROUTE_PASSED",
        },
        "unknown_gateway_host": {
            "case_id": "unknown_gateway_host",
            "status": "REJECTED",
        },
    }
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != len(expected_cases):
        return False
    observed_cases: dict[str, dict[str, Any]] = {}
    for item in cases:
        if not isinstance(item, dict) or not isinstance(item.get("case_id"), str):
            return False
        case_id = str(item["case_id"])
        if case_id in observed_cases:
            return False
        observed_cases[case_id] = item
    return (
        observed_cases == expected_cases
        and value.get("attack_content_persisted") is False
        and value.get("blocked_before_model_delta") == 3
        and value.get("case_count") == len(expected_cases)
        and value.get("guardrail_policy_version") == "m6-prompt-injection-v1"
        and value.get("model_prediction_delta") == 1
    )


def _candidate_resource_hardened(value: object, *, expected_image: str) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "available_replicas",
        "deployment_generation",
        "deployment_uid",
        "image",
        "inference_service_generation",
        "inference_service_uid",
        "name",
        "security_context",
    }:
        return False
    return (
        value.get("name") == "ioap-qwen-candidate"
        and value.get("image") == expected_image
        and security_context_is_hardened(value.get("security_context"))
        and _positive_int(value.get("available_replicas"))
        and _positive_int(value.get("deployment_generation"))
        and _positive_int(value.get("inference_service_generation"))
        and isinstance(value.get("deployment_uid"), str)
        and bool(value.get("deployment_uid"))
        and isinstance(value.get("inference_service_uid"), str)
        and bool(value.get("inference_service_uid"))
    )


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _rollout_stages_passed(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "SHADOW",
        "CANARY_5",
        "CANARY_25",
        "ROLLED_BACK",
    }:
        return False
    shadow = value.get("SHADOW")
    canary_5 = value.get("CANARY_5")
    canary_25 = value.get("CANARY_25")
    rollback = value.get("ROLLED_BACK")
    if not all(isinstance(item, dict) for item in (shadow, canary_5, canary_25, rollback)):
        return False
    assert isinstance(shadow, dict)
    assert isinstance(canary_5, dict)
    assert isinstance(canary_25, dict)
    assert isinstance(rollback, dict)
    try:
        mirror_delta = int(shadow.get("candidate_mirror_delta", 0))
        rollback_requests = int(rollback.get("request_count", 0))
    except (TypeError, ValueError):
        return False
    return (
        shadow.get("stable_response_variant") == "stable"
        and mirror_delta >= 1
        and bool(shadow.get("candidate_root_cause_code"))
        and _canary_passed(canary_5, lower=0.01, upper=0.15)
        and _canary_passed(canary_25, lower=0.15, upper=0.35)
        and rollback.get("request_count") == rollback.get("stable_count")
        and rollback.get("candidate_count") == 0
        and rollback_requests > 0
    )


def _canary_passed(value: dict[str, Any], *, lower: float, upper: float) -> bool:
    try:
        request_count = int(value["request_count"])
        stable_count = int(value["stable_count"])
        candidate_count = int(value["candidate_count"])
        ratio = float(value["candidate_ratio"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        request_count > 0
        and stable_count + candidate_count == request_count
        and abs(candidate_count / request_count - ratio) <= 1e-12
        and lower <= ratio <= upper
    )
