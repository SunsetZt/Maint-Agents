"""Truthful local STAGING provenance for an imported, evaluated Qwen Adapter."""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.persistence.models import (
    ModelReleaseRecord,
    TrainingArtifactRecord,
    TrainingExperimentRecord,
)


def local_adapter_binding(
    session: Session,
    tenant_id: str,
    *,
    candidate_id: str,
    admission: dict[str, Any],
    runtime: dict[str, Any],
    environment: str,
) -> dict[str, Any]:
    if (
        environment != "STAGING"
        or admission.get("target_environment") != "STAGING"
        or admission.get("mode") != "STAGING_DIAGNOSIS_SMOKE_14"
        or not admission.get("transfer")
        or admission.get("tenant_id") != tenant_id
    ):
        raise ValueError("local_adapter_requires_imported_staging_smoke")
    candidate = session.scalar(
        select(TrainingExperimentRecord).where(
            TrainingExperimentRecord.tenant_id == tenant_id,
            TrainingExperimentRecord.experiment_id == candidate_id,
        )
    )
    adapter = session.scalar(
        select(TrainingArtifactRecord).where(
            TrainingArtifactRecord.tenant_id == tenant_id,
            TrainingArtifactRecord.artifact_id == admission["adapter_artifact_id"],
        )
    )
    if (
        candidate is None
        or adapter is None
        or candidate.status != "COMPLETED"
        or candidate.task_type != "DIAGNOSIS"
        or candidate.method != "LORA"
        or candidate.base_model_id != "Qwen/Qwen3-0.6B"
        or adapter.experiment_id != candidate_id
        or adapter.size_bytes <= 0
        or adapter.content_hash != admission["adapter_content_hash"]
        or adapter.metadata_json.get("format") != "tar.gz"
        or adapter.metadata_json.get("serialization") != "safetensors"
        or candidate_id != admission["candidate_experiment_id"]
        or not re.fullmatch(r"[a-f0-9]{64}", adapter.content_hash.removeprefix("sha256:"))
    ):
        raise ValueError("local_adapter_binding_invalid")
    if (
        not isinstance(runtime.get("image_repository"), str)
        or not re.fullmatch(r"[a-z0-9][a-z0-9./:_-]*", runtime["image_repository"])
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", runtime.get("image_digest", ""))
        or runtime.get("profile_id") != "vllm-peft-v1"
        or runtime.get("inference_config", {}).get("engine") != "vllm"
        or runtime.get("inference_config", {}).get("enable_lora") is not True
    ):
        raise ValueError("local_adapter_runtime_binding_invalid")
    return {
        "kind": "LOCAL_PINNED",
        "tenant_id": tenant_id,
        "target_environment": "STAGING",
        "deployment_provider": "COMPOSE_VLLM_GATEWAY",
        "deployment_use": "INTEGRATION_ONLY",
        "signature_status": "NOT_VERIFIED",
        "scan_status": "NOT_ATTESTED",
        "quality_status": "LIMITED_REGRESSION_ONLY",
        "production_eligible": False,
        "evidence_id": None,
        "adapter_artifact_id": adapter.artifact_id,
        "adapter_content_hash": adapter.content_hash,
        "adapter_size_bytes": adapter.size_bytes,
        "source_revision": candidate.git_commit,
        "base_model_id": candidate.base_model_id,
        "base_model_digest": candidate.base_model_digest,
        "runtime_image_repository": runtime["image_repository"],
        "runtime_image_digest": runtime["image_digest"],
        "import_id": admission["transfer"]["import_id"],
        "import_hash": admission["transfer"]["import_hash"],
    }


def require_local_adapter_current(
    session: Session,
    release: ModelReleaseRecord,
    admission: dict[str, Any],
) -> None:
    manifest = release.manifest_json
    chain = manifest.get("supply_chain", {})
    if not manifest.get("local_staging_adapter") and chain.get("kind") != "LOCAL_PINNED":
        return
    if (
        manifest.get("local_staging_adapter") is not True
        or release.target_environment != "STAGING"
        or manifest.get("target_environment") != "STAGING"
    ):
        raise ValueError("local_adapter_is_staging_only")
    expected = local_adapter_binding(
        session,
        release.tenant_id,
        candidate_id=release.candidate_experiment_id,
        admission=admission,
        runtime=manifest["runtime"],
        environment=release.target_environment,
    )
    if (
        chain != expected
        or manifest.get("adapter", {}).get("content_hash") != expected["adapter_content_hash"]
    ):
        raise ValueError("local_adapter_provenance_changed")
