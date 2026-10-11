"""Server-owned AIReleaseManifest construction and approval state machine."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import structlog
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from industrial_ops_agent.agent.contracts import DIAGNOSIS_AGENT_PROMPT_ID, DIAGNOSIS_GRAPH_VERSION
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.data_pipeline.contracts import (
    production_release_blocker_for_dataset_contract,
)
from industrial_ops_agent.domain import as_utc as _as_utc
from industrial_ops_agent.domain.json import strict_document_digest as _digest_json
from industrial_ops_agent.evaluation.edge_fidelity import (
    stable_edge_evaluation_preserves_source,
)
from industrial_ops_agent.experiments.comparison import (
    SYNTHETIC_DATASET_ABLATION,
    SyntheticAblationContractError,
    synthetic_ablation_context,
)
from industrial_ops_agent.experiments.service import (
    ASR_HARD_GATES,
    ASR_PRIMARY_METRICS,
    RETRIEVAL_HARD_GATES,
    RETRIEVAL_PRIMARY_METRICS,
    RUL_HARD_GATES,
    RUL_PRIMARY_METRICS,
    TIMESERIES_HARD_GATES,
    TIMESERIES_PRIMARY_METRICS,
    TTS_HARD_GATES,
    TTS_PRIMARY_METRICS,
    VLM_HARD_GATES,
    VLM_PRIMARY_METRICS,
)
from industrial_ops_agent.model_methods import PRODUCTION_RELEASE_METHODS
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.model_import_integrity import (
    enterprise_model_import_hash,
)
from industrial_ops_agent.persistence.models import (
    DatasetSnapshotRecord,
    EnterpriseModelAssetImportRecord,
    EvaluationArtifactRecord,
    EvaluationPolicyRecord,
    EvaluationSuiteRecord,
    IndexReleaseRecord,
    KnowledgeChunkRecord,
    KnowledgeDocumentVersionRecord,
    ModelEvaluationJobRecord,
    ModelEvaluationRunRecord,
    ModelReleaseApprovalRecord,
    ModelReleaseRecord,
    ModelReleaseTransitionRecord,
    PromptBundleRecord,
    SupplyChainEvidenceRecord,
    TrainingArtifactRecord,
    TrainingExperimentRecord,
)
from industrial_ops_agent.predictive_maintenance.dataset_contract import SIGNAL_ORDER
from industrial_ops_agent.predictive_maintenance.rul_dataset_contract import (
    CONTRACT_VERSION as RUL_CONTRACT_VERSION,
)
from industrial_ops_agent.predictive_maintenance.rul_dataset_contract import (
    SIGNAL_ORDER as RUL_SIGNAL_ORDER,
)
from industrial_ops_agent.prompting import DEFAULT_PROMPT_BUNDLE_ID, default_prompt_registry
from industrial_ops_agent.prompting.registry import PromptBundleNotDeployed
from industrial_ops_agent.releases.local_adapter import local_adapter_binding
from industrial_ops_agent.releases.staging_smoke import (
    STAGING_DIAGNOSIS_SMOKE_14,
    build_staging_smoke_admission,
    require_publishable_staging_admission,
    require_staging_smoke_admission_current,
)
from industrial_ops_agent.supply_chain.service import (
    component_evidence_matches,
    component_supply_chain_identity,
    media_release_supply_chain_status,
    record_verification_hash,
)
from industrial_ops_agent.supply_chain.service import (
    supply_chain_manifest as _supply_chain_manifest,
)
from industrial_ops_agent.tools.registry import default_tool_registry

ReleaseEnvironment = Literal["STAGING", "PRODUCTION"]
ApprovalDecision = Literal["APPROVED", "REJECTED"]

PROMPT_BUNDLE_ID = DEFAULT_PROMPT_BUNDLE_ID
REQUIRED_TOOL_IDS = frozenset(
    {
        "asset.get",
        "warranty.get",
        "parts.availability",
        "schedule.availability",
        "work_orders.history",
        "work_order.draft",
        "parts.reserve",
        "customer.notify",
        "purchase.request",
        "refund.request",
        "work_order.assign",
        "work_order.close",
        "equipment.control.*",
    }
)
SPECIALIZED_COMPONENT_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "embedding": {
        "method": "EMBEDDING",
        "profile": "RETRIEVAL_COMPONENT",
        "artifact_kind": "embedding_model_bundle",
        "primary_metrics": RETRIEVAL_PRIMARY_METRICS,
        "hard_gates": RETRIEVAL_HARD_GATES,
    },
    "reranker": {
        "method": "RERANKER",
        "profile": "RETRIEVAL_COMPONENT",
        "artifact_kind": "reranker_model_bundle",
        "primary_metrics": RETRIEVAL_PRIMARY_METRICS,
        "hard_gates": RETRIEVAL_HARD_GATES,
    },
    "vlm": {
        "method": "VLM",
        "profile": "VLM_COMPONENT",
        "artifact_kind": "vlm_adapter_bundle",
        "primary_metrics": VLM_PRIMARY_METRICS,
        "hard_gates": VLM_HARD_GATES,
    },
    "asr": {
        "method": "ASR",
        "profile": "ASR_COMPONENT",
        "artifact_kind": "asr_adapter_bundle",
        "primary_metrics": ASR_PRIMARY_METRICS,
        "hard_gates": ASR_HARD_GATES,
    },
    "tts": {
        "method": "TTS",
        "profile": "TTS_COMPONENT",
        "artifact_kind": "tts_model_bundle",
        "primary_metrics": TTS_PRIMARY_METRICS,
        "hard_gates": TTS_HARD_GATES,
    },
    "timeseries": {
        "method": "TIMESERIES_TRANSFORMER",
        "profile": "TIMESERIES_COMPONENT",
        "artifact_kind": "timeseries_transformer_bundle",
        "primary_metrics": TIMESERIES_PRIMARY_METRICS,
        "hard_gates": TIMESERIES_HARD_GATES,
    },
    "rul": {
        "method": "RUL_TRANSFORMER",
        "profile": "RUL_COMPONENT",
        "artifact_kind": "rul_transformer_bundle",
        "primary_metrics": RUL_PRIMARY_METRICS,
        "hard_gates": RUL_HARD_GATES,
        "minimum_suite_samples": 30,
    },
}
CORE_SPECIALIZED_COMPONENTS = frozenset({"embedding", "reranker", "vlm", "asr"})
TIMESERIES_RUNTIME_PROFILE = "native-pytorch-timeseries-v1"
RUL_RUNTIME_PROFILE = "native-pytorch-rul-v1"
RERANKER_RUNTIME_PROFILE = "sentence-transformers-cross-encoder-v1"
EMBEDDING_RUNTIME_PROFILE = "sentence-transformers-embedding-v1"
TTS_RUNTIME_PROFILE = "speecht5-tts-v1"
PROJECT_AUTHORIZED_LOCAL_RELEASE_SCHEMA = "project-authorized-local-staging-release/v1"
PROJECT_AUTHORIZED_LOCAL_CLASSIFICATION = "LOCAL_STAGING_PROJECT_AUTHORIZED"


class ReleaseNotVisible(Exception):
    pass


class ReleaseConflict(Exception):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ReleaseManifestPlan:
    evaluation_id: str
    component_evaluation_ids: dict[str, str]
    target_environment: ReleaseEnvironment
    quantization_profile_id: str
    runtime_profile_id: str
    runtime_image_repository: str
    runtime_image_digest: str
    prompt_bundle_id: str
    agent_graph_id: str
    tool_versions: dict[str, str]
    mcp_server_versions: dict[str, str]
    index_release_id: str
    embedding_model_id: str
    reranker_model_id: str
    multimodal_model_ids: dict[str, str]
    inference_config: dict[str, Any]
    hardware_profile: dict[str, Any]
    supply_chain_evidence_id: str | None
    evaluation_admission: str = "STANDARD"
    prompt_evaluation_only: bool = False
    local_staging_adapter: bool = False
    rollback_release_id: str | None = None
    timeseries_model_id: str | None = None
    timeseries_supply_chain_evidence_id: str | None = None
    rul_model_id: str | None = None
    rul_supply_chain_evidence_id: str | None = None
    reranker_supply_chain_evidence_id: str | None = None
    embedding_supply_chain_evidence_id: str | None = None
    tts_supply_chain_evidence_id: str | None = None
    vlm_supply_chain_evidence_id: str | None = None
    asr_supply_chain_evidence_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProjectAuthorizedLocalReleasePlan:
    """Closed staging-only plan for an independently evaluated project GGUF."""

    evaluation_id: str
    quantization_profile_id: str
    runtime_profile_id: str
    runtime_git_commit: str
    runtime_image_repository: str
    runtime_image_reference: str
    runtime_image_digest: str
    inference_config: dict[str, Any]
    hardware_profile: dict[str, Any]
    source_evidence_chain_sha256: str
    model_file_sha256: str
    model_storage_subpath: str


@dataclass(frozen=True, slots=True)
class ReleaseAggregate:
    release: ModelReleaseRecord
    approval: ModelReleaseApprovalRecord | None
    transitions: tuple[ModelReleaseTransitionRecord, ...]


class ModelReleaseService:
    """Build immutable release manifests from authoritative records.

    This service intentionally stops at an approved, deployment-ready request.
    Shadow and Canary are only entered by the deployment controller after it has
    real routing and observation evidence; an approval alone never claims traffic.
    """

    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def create(
        self,
        identity: IdentityContext,
        plan: ReleaseManifestPlan,
        *,
        idempotency_key: str,
        request_id: str,
    ) -> ReleaseAggregate:
        self._require(identity, Action.CREATE_MODEL_RELEASE, "model-releases", request_id)
        _validate_plan(plan)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            manifest, candidate_id = _build_manifest(session, identity.tenant_id, plan)
            manifest_hash = _digest_json(manifest)
            existing = session.scalar(
                select(ModelReleaseRecord).where(
                    ModelReleaseRecord.tenant_id == identity.tenant_id,
                    ModelReleaseRecord.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                if existing.manifest_hash != manifest_hash:
                    raise ReleaseConflict("idempotency_key_reused", existing.version)
                return _aggregate(session, identity.tenant_id, existing)
            duplicate = session.scalar(
                select(ModelReleaseRecord).where(
                    ModelReleaseRecord.tenant_id == identity.tenant_id,
                    ModelReleaseRecord.manifest_hash == manifest_hash,
                )
            )
            if duplicate is not None:
                raise ReleaseConflict("manifest_already_registered", duplicate.version)

            release = ModelReleaseRecord(
                release_id=f"model-release-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                idempotency_key=idempotency_key,
                evaluation_id=plan.evaluation_id,
                candidate_experiment_id=candidate_id,
                status="DRAFT",
                target_environment=plan.target_environment,
                manifest_json=manifest,
                manifest_hash=manifest_hash,
                rollback_release_id=plan.rollback_release_id,
                traffic_percent=0.0,
                failure_reason=None,
                created_by_subject_id=identity.subject_id,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(release)
            session.flush()
            _append_transition(
                session,
                release,
                sequence=1,
                from_status="NONE",
                to_status="DRAFT",
                actor_subject_id=identity.subject_id,
                reason_code="manifest_registered",
                evidence={
                    "evaluation_id": plan.evaluation_id,
                    "manifest_hash": manifest_hash,
                    "target_environment": plan.target_environment,
                },
                occurred_at=now,
            )
            session.flush()
            return _aggregate(session, identity.tenant_id, release)

    def create_project_authorized_local_staging(
        self,
        identity: IdentityContext,
        plan: ProjectAuthorizedLocalReleasePlan,
        *,
        idempotency_key: str,
        request_id: str,
    ) -> ReleaseAggregate:
        """Register a closed, non-production GGUF release from governed local evidence."""

        self._require(identity, Action.CREATE_MODEL_RELEASE, "model-releases", request_id)
        _validate_project_authorized_local_plan(plan)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            manifest, candidate_id = _build_project_authorized_local_manifest(
                session,
                identity.tenant_id,
                plan,
            )
            manifest_hash = _digest_json(manifest)
            existing = session.scalar(
                select(ModelReleaseRecord).where(
                    ModelReleaseRecord.tenant_id == identity.tenant_id,
                    ModelReleaseRecord.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                if existing.manifest_hash != manifest_hash:
                    raise ReleaseConflict("idempotency_key_reused", existing.version)
                return _aggregate(session, identity.tenant_id, existing)
            duplicate = session.scalar(
                select(ModelReleaseRecord).where(
                    ModelReleaseRecord.tenant_id == identity.tenant_id,
                    ModelReleaseRecord.manifest_hash == manifest_hash,
                )
            )
            if duplicate is not None:
                raise ReleaseConflict("manifest_already_registered", duplicate.version)
            release = ModelReleaseRecord(
                release_id=f"model-release-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                idempotency_key=idempotency_key,
                evaluation_id=plan.evaluation_id,
                candidate_experiment_id=candidate_id,
                status="DRAFT",
                target_environment="STAGING",
                manifest_json=manifest,
                manifest_hash=manifest_hash,
                rollback_release_id=None,
                traffic_percent=0.0,
                failure_reason=None,
                created_by_subject_id=identity.subject_id,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(release)
            session.flush()
            _append_transition(
                session,
                release,
                sequence=1,
                from_status="NONE",
                to_status="DRAFT",
                actor_subject_id=identity.subject_id,
                reason_code="project_authorized_local_manifest_registered",
                evidence={
                    "evaluation_id": plan.evaluation_id,
                    "manifest_hash": manifest_hash,
                    "classification": PROJECT_AUTHORIZED_LOCAL_CLASSIFICATION,
                    "production_claim": False,
                },
                occurred_at=now,
            )
            session.flush()
            return _aggregate(session, identity.tenant_id, release)

    def create_supply_chain_successor(
        self,
        identity: IdentityContext,
        release_id: str,
        *,
        vlm_supply_chain_evidence_id: str | None,
        asr_supply_chain_evidence_id: str | None,
        expected_version: int,
        idempotency_key: str,
        request_id: str,
    ) -> ReleaseAggregate:
        """Clone immutable model intent, never its approval or deployment authority."""
        self._require(identity, Action.CREATE_MODEL_RELEASE, release_id, request_id)
        supplied = {"vlm": vlm_supply_chain_evidence_id, "asr": asr_supply_chain_evidence_id}
        operation_key = "supply-chain-successor:" + _digest_json(
            {
                "tenant": identity.tenant_id,
                "subject": identity.subject_id,
                "parent": release_id,
                "key": idempotency_key,
            }
        ).removeprefix("sha256:")
        request_hash = _digest_json(
            {"evidence_ids": supplied, "expected_version": expected_version}
        )
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            # Serialize retries on a durable parent row, including the first child insert.
            parent = _release_or_hidden(session, identity.tenant_id, release_id, lock=True)
            _require_manifest_integrity(parent)
            existing = session.scalar(
                select(ModelReleaseRecord).where(
                    ModelReleaseRecord.tenant_id == identity.tenant_id,
                    ModelReleaseRecord.idempotency_key == operation_key,
                )
            )
            if existing is not None:
                _require_manifest_integrity(existing)
                provenance = existing.manifest_json.get("supply_chain_successor", {})
                if (
                    provenance.get("request_hash") != request_hash
                    or provenance.get("parent_manifest_hash") != parent.manifest_hash
                ):
                    raise ReleaseConflict("idempotency_key_reused", existing.version)
                return _aggregate(session, identity.tenant_id, existing)
            _require_version(parent, expected_version)
            if parent.manifest_json.get("schema_version") not in {
                "ai-release-manifest/v4",
                "ai-release-manifest/v5",
            }:
                raise ReleaseConflict("supply_chain_successor_schema_unsupported")
            manifest = deepcopy(parent.manifest_json)
            components = manifest.get("specialized_components")
            if not isinstance(components, dict):
                raise ReleaseConflict("supply_chain_successor_components_required")
            for name, evidence_id in supplied.items():
                if name not in components:
                    if evidence_id is not None or name in manifest.get("multimodal_model_ids", {}):
                        raise ReleaseConflict(f"{name}_successor_component_missing")
                    continue
                original = components[name]
                if not isinstance(original, dict):
                    raise ReleaseConflict(f"{name}_successor_component_invalid")
                previous_evidence = original.get("supply_chain", {}).get("evidence_id")
                attached = _attach_media_supply_chain(
                    session,
                    identity.tenant_id,
                    original,
                    evidence_id=evidence_id or previous_evidence,
                )
                previous_source = original.get("source")
                if previous_source is not None and previous_source != attached["source"]:
                    raise ReleaseConflict(f"{name}_successor_source_changed")
                previous_runtime = original.get("runtime")
                if isinstance(previous_runtime, dict) and any(
                    previous_runtime.get(key) != attached["runtime"][key]
                    for key in ("image_repository", "image_digest")
                ):
                    raise ReleaseConflict(f"{name}_successor_runtime_changed")
                components[name] = attached
            manifest["schema_version"] = "ai-release-manifest/v5"
            manifest["supply_chain_successor"] = {
                "parent_release_id": parent.release_id,
                "parent_manifest_hash": parent.manifest_hash,
                "request_hash": request_hash,
                "evidence_hashes": {
                    name: components[name]["supply_chain"]["verification_hash"]
                    for name in ("vlm", "asr")
                    if name in components
                },
            }
            successor = ModelReleaseRecord(
                release_id=f"model-release-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                idempotency_key=operation_key,
                evaluation_id=parent.evaluation_id,
                candidate_experiment_id=parent.candidate_experiment_id,
                status="DRAFT",
                target_environment=parent.target_environment,
                manifest_json=manifest,
                manifest_hash=_digest_json(manifest),
                rollback_release_id=parent.rollback_release_id,
                traffic_percent=0.0,
                failure_reason=None,
                created_by_subject_id=identity.subject_id,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(successor)
            session.flush()
            _append_transition(
                session,
                successor,
                sequence=1,
                from_status="NONE",
                to_status="DRAFT",
                actor_subject_id=identity.subject_id,
                reason_code="supply_chain_successor_registered",
                evidence=manifest["supply_chain_successor"],
                occurred_at=now,
            )
            session.flush()
            return _aggregate(session, identity.tenant_id, successor)

    def get(
        self,
        identity: IdentityContext,
        release_id: str,
        *,
        request_id: str,
    ) -> ReleaseAggregate:
        self._require(identity, Action.READ_MODEL_RELEASE, release_id, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            release = _release_or_hidden(session, identity.tenant_id, release_id)
            return _aggregate(session, identity.tenant_id, release)

    def list(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
    ) -> list[ReleaseAggregate]:
        self._require(identity, Action.READ_MODEL_RELEASE, "model-releases", request_id)
        with self._database.transaction(identity.tenant_context) as session:
            releases = list(
                session.scalars(
                    select(ModelReleaseRecord)
                    .where(ModelReleaseRecord.tenant_id == identity.tenant_id)
                    .order_by(ModelReleaseRecord.created_at.desc())
                )
            )
            return [_aggregate(session, identity.tenant_id, release) for release in releases]

    def validate(
        self,
        identity: IdentityContext,
        release_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> ReleaseAggregate:
        self._require(identity, Action.VALIDATE_MODEL_RELEASE, release_id, request_id)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            release = _release_or_hidden(session, identity.tenant_id, release_id)
            _require_version(release, expected_version)
            if release.status != "DRAFT":
                raise ReleaseConflict("release_not_validatable", release.version)
            _require_manifest_integrity(release)
            if release.manifest_json.get("prompt_evaluation_only"):
                raise ReleaseConflict("evaluation_only_not_publishable", release.version)

            release.status = "VALIDATING"
            release.version += 1
            _append_transition(
                session,
                release,
                sequence=_next_sequence(session, identity.tenant_id, release_id),
                from_status="DRAFT",
                to_status="VALIDATING",
                actor_subject_id=identity.subject_id,
                reason_code="release_validation_started",
                evidence={"manifest_hash": release.manifest_hash},
                occurred_at=now,
            )

            gate_results = (
                _evaluate_project_authorized_local_release_gates(
                    session,
                    identity.tenant_id,
                    release,
                )
                if release.manifest_json.get("schema_version")
                == PROJECT_AUTHORIZED_LOCAL_RELEASE_SCHEMA
                else _evaluate_release_gates(session, identity.tenant_id, release)
            )
            failed = sorted(name for name, passed in gate_results.items() if not passed)
            destination = "REJECTED" if failed else "CANDIDATE"
            reason_code = "release_validation_failed" if failed else "release_validation_passed"
            release.status = destination
            release.failure_reason = ",".join(failed) if failed else None
            release.version += 1
            _append_transition(
                session,
                release,
                sequence=_next_sequence(session, identity.tenant_id, release_id),
                from_status="VALIDATING",
                to_status=destination,
                actor_subject_id=identity.subject_id,
                reason_code=reason_code,
                evidence={
                    "failed_gates": failed,
                    "gate_results": gate_results,
                    "manifest_hash": release.manifest_hash,
                },
                occurred_at=now,
            )
            session.flush()
            return _aggregate(session, identity.tenant_id, release)

    def submit_for_approval(
        self,
        identity: IdentityContext,
        release_id: str,
        *,
        expected_version: int,
        request_id: str,
        approval_ttl_hours: int = 24,
    ) -> ReleaseAggregate:
        self._require(
            identity,
            Action.SUBMIT_MODEL_RELEASE_APPROVAL,
            release_id,
            request_id,
        )
        if approval_ttl_hours < 1 or approval_ttl_hours > 168:
            raise ValueError("approval_ttl_hours must be between 1 and 168")
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            release = _release_or_hidden(session, identity.tenant_id, release_id)
            _require_version(release, expected_version)
            _require_manifest_integrity(release)
            _require_media_supply_chain_current(session, release)
            _require_staging_publication(session, release)
            if release.status != "CANDIDATE":
                raise ReleaseConflict("release_not_approval_ready", release.version)
            existing = _approval_for_release(session, identity.tenant_id, release_id)
            if existing is not None:
                raise ReleaseConflict("approval_already_requested", release.version)
            if release.target_environment == "PRODUCTION" and release.rollback_release_id is None:
                raise ReleaseConflict("production_rollback_release_required", release.version)

            approval = ModelReleaseApprovalRecord(
                approval_id=f"release-approval-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                release_id=release_id,
                manifest_hash=release.manifest_hash,
                status="PENDING",
                requested_by_subject_id=identity.subject_id,
                decided_by_subject_id=None,
                decision_reason=None,
                expires_at=now + timedelta(hours=approval_ttl_hours),
                decided_at=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(approval)
            release.status = "APPROVAL_PENDING"
            release.version += 1
            _append_transition(
                session,
                release,
                sequence=_next_sequence(session, identity.tenant_id, release_id),
                from_status="CANDIDATE",
                to_status="APPROVAL_PENDING",
                actor_subject_id=identity.subject_id,
                reason_code="release_approval_requested",
                evidence={
                    "approval_id": approval.approval_id,
                    "expires_at": approval.expires_at.isoformat(),
                    "manifest_hash": release.manifest_hash,
                    "rollback_release_id": release.rollback_release_id,
                },
                occurred_at=now,
            )
            session.flush()
            return _aggregate(session, identity.tenant_id, release)

    def decide_approval(
        self,
        identity: IdentityContext,
        release_id: str,
        *,
        expected_version: int,
        expected_approval_version: int,
        decision: ApprovalDecision,
        reason: str,
        request_id: str,
    ) -> ReleaseAggregate:
        self._require(
            identity,
            Action.DECIDE_MODEL_RELEASE_APPROVAL,
            release_id,
            request_id,
        )
        normalized_reason = reason.strip()
        if not normalized_reason or len(normalized_reason) > 1024:
            raise ValueError("approval decision reason is invalid")
        if decision not in {"APPROVED", "REJECTED"}:
            raise ValueError("approval decision is invalid")
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            release = _release_or_hidden(session, identity.tenant_id, release_id)
            _require_version(release, expected_version)
            _require_manifest_integrity(release)
            approval = _approval_for_release(session, identity.tenant_id, release_id)
            if approval is None:
                raise ReleaseConflict("approval_not_found", release.version)
            if release.status != "APPROVAL_PENDING" or approval.status != "PENDING":
                raise ReleaseConflict("approval_not_pending", release.version)
            if approval.version != expected_approval_version:
                raise ReleaseConflict("approval_version_mismatch", release.version)
            if approval.manifest_hash != release.manifest_hash:
                raise ReleaseConflict("approval_manifest_mismatch", release.version)
            if approval.requested_by_subject_id == identity.subject_id:
                raise ReleaseConflict("separation_of_duties_required", release.version)
            if _as_utc(approval.expires_at) <= now:
                approval.status = "EXPIRED"
                approval.version += 1
                release.status = "REJECTED"
                release.failure_reason = "release_approval_expired"
                release.version += 1
                _append_transition(
                    session,
                    release,
                    sequence=_next_sequence(session, identity.tenant_id, release_id),
                    from_status="APPROVAL_PENDING",
                    to_status="REJECTED",
                    actor_subject_id=identity.subject_id,
                    reason_code="release_approval_expired",
                    evidence={"approval_id": approval.approval_id},
                    occurred_at=now,
                )
                session.flush()
                return _aggregate(session, identity.tenant_id, release)

            if decision == "APPROVED":
                _require_media_supply_chain_current(session, release)
                _require_staging_publication(session, release)

            approval.status = decision
            approval.decided_by_subject_id = identity.subject_id
            approval.decision_reason = normalized_reason
            approval.decided_at = now
            approval.version += 1
            release.version += 1
            if decision == "REJECTED":
                release.status = "REJECTED"
                release.failure_reason = "release_approval_rejected"
                _append_transition(
                    session,
                    release,
                    sequence=_next_sequence(session, identity.tenant_id, release_id),
                    from_status="APPROVAL_PENDING",
                    to_status="REJECTED",
                    actor_subject_id=identity.subject_id,
                    reason_code="release_approval_rejected",
                    evidence={
                        "approval_id": approval.approval_id,
                        "decision_reason_hash": sha256(normalized_reason.encode()).hexdigest(),
                    },
                    occurred_at=now,
                )
            session.flush()
            return _aggregate(session, identity.tenant_id, release)

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        resource_id: str,
        request_id: str,
    ) -> None:
        self._authorizer.require(
            identity,
            action,
            ResourceContext(tenant_id=identity.tenant_id, resource_id=resource_id),
            request_id=request_id,
        )


def _validate_project_authorized_local_plan(
    plan: ProjectAuthorizedLocalReleasePlan,
) -> None:
    identity_fields = (
        plan.evaluation_id,
        plan.quantization_profile_id,
        plan.runtime_profile_id,
        plan.runtime_image_repository,
    )
    if any(not value.strip() or len(value) > 255 for value in identity_fields):
        raise ValueError("project authorized local release identity is invalid")
    _validate_image_repository(plan.runtime_image_repository)
    _validate_sha256_digest(plan.runtime_image_digest, "runtime_image_digest")
    if len(plan.runtime_git_commit) != 40 or any(
        character not in "0123456789abcdef" for character in plan.runtime_git_commit
    ):
        raise ValueError("runtime_git_commit must be a full lowercase Git commit")
    if plan.runtime_image_reference != (
        f"{plan.runtime_image_repository}:{plan.runtime_git_commit[:12]}"
    ):
        raise ValueError("runtime_image_reference must bind the repository to the Git commit tag")
    for value, name in (
        (plan.source_evidence_chain_sha256, "source_evidence_chain_sha256"),
        (plan.model_file_sha256, "model_file_sha256"),
    ):
        normalized = value.removeprefix("sha256:")
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError(f"{name} must be sha256")
    if (
        not plan.model_storage_subpath
        or len(plan.model_storage_subpath) > 128
        or Path(plan.model_storage_subpath).is_absolute()
        or ".." in Path(plan.model_storage_subpath).parts
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-_/"
            for character in plan.model_storage_subpath
        )
    ):
        raise ValueError("model_storage_subpath is invalid")
    if plan.model_storage_subpath not in {"stable", "candidate"}:
        raise ValueError("model_storage_subpath must identify stable or candidate")
    _validate_inference_config(plan.inference_config)
    _validate_hardware_profile(plan.hardware_profile)
    if (
        plan.inference_config.get("engine") != "llama.cpp"
        or plan.hardware_profile.get("accelerator") != "CPU"
    ):
        raise ValueError("project authorized local release must use llama.cpp on CPU")
    for document in (plan.inference_config, plan.hardware_profile):
        _digest_json(document)


def _build_project_authorized_local_manifest(
    session: Session,
    tenant_id: str,
    plan: ProjectAuthorizedLocalReleasePlan,
) -> tuple[dict[str, Any], str]:
    evaluation = _tenant_record(
        session,
        ModelEvaluationRunRecord,
        tenant_id,
        ModelEvaluationRunRecord.evaluation_id,
        plan.evaluation_id,
    )
    candidate = _tenant_record(
        session,
        TrainingExperimentRecord,
        tenant_id,
        TrainingExperimentRecord.experiment_id,
        evaluation.candidate_experiment_id,
    )
    source_id = str(candidate.training_config.get("source_experiment_id", ""))
    source = _tenant_record(
        session,
        TrainingExperimentRecord,
        tenant_id,
        TrainingExperimentRecord.experiment_id,
        source_id,
    )
    suite = _tenant_record(
        session,
        EvaluationSuiteRecord,
        tenant_id,
        EvaluationSuiteRecord.suite_id,
        evaluation.suite_id,
    )
    policy = _tenant_record(
        session,
        EvaluationPolicyRecord,
        tenant_id,
        EvaluationPolicyRecord.policy_id,
        evaluation.policy_id,
    )
    jobs = list(
        session.scalars(
            select(ModelEvaluationJobRecord).where(
                ModelEvaluationJobRecord.tenant_id == tenant_id,
                ModelEvaluationJobRecord.result_evaluation_id == evaluation.evaluation_id,
            )
        )
    )
    quantized_artifacts = list(
        session.scalars(
            select(TrainingArtifactRecord).where(
                TrainingArtifactRecord.tenant_id == tenant_id,
                TrainingArtifactRecord.experiment_id == candidate.experiment_id,
                TrainingArtifactRecord.kind == "quantized_model_bundle",
            )
        )
    )
    adapters = list(
        session.scalars(
            select(TrainingArtifactRecord).where(
                TrainingArtifactRecord.tenant_id == tenant_id,
                TrainingArtifactRecord.experiment_id == source.experiment_id,
                TrainingArtifactRecord.kind == "adapter_bundle",
            )
        )
    )
    evaluation_artifacts = list(
        session.scalars(
            select(EvaluationArtifactRecord)
            .where(
                EvaluationArtifactRecord.tenant_id == tenant_id,
                EvaluationArtifactRecord.evaluation_id == evaluation.evaluation_id,
            )
            .order_by(
                EvaluationArtifactRecord.kind,
                EvaluationArtifactRecord.artifact_id,
            )
        )
    )
    if len(jobs) != 1 or len(quantized_artifacts) != 1 or len(adapters) != 1:
        raise ReleaseConflict("project_authorized_local_release_records_are_incomplete")
    job = jobs[0]
    quantized = quantized_artifacts[0]
    adapter = adapters[0]
    quantized_runtime = quantized.metadata_json.get("runtime")
    if (
        candidate.status != "COMPLETED"
        or candidate.method != "QUANTIZATION"
        or candidate.training_config.get("target_runtime") != "LLAMA_CPP"
        or candidate.training_config.get("quantization_profile_id") != plan.quantization_profile_id
        or source.status != "COMPLETED"
        or source.method not in {"LORA", "QLORA"}
        or candidate.training_config.get("source_artifact_id") != adapter.artifact_id
        or str(candidate.training_config.get("source_artifact_hash", "")).removeprefix("sha256:")
        != adapter.content_hash
        or quantized.metadata_json.get("serialization") != "gguf"
        or quantized.metadata_json.get("approval_inheritance") is not False
        or quantized.metadata_json.get("quantization_profile_id") != plan.quantization_profile_id
        or not isinstance(quantized_runtime, dict)
        or quantized_runtime.get("target_runtime") != "llama.cpp"
        or not _valid_artifact_filename(quantized_runtime.get("model_file"))
        or evaluation.status != "COMPLETED"
        or not _project_authorized_local_evaluation_accepted(
            evaluation,
            role=plan.model_storage_subpath,
        )
        or suite.tier != "PROJECT_AUTHORIZED"
        or suite.purpose != "EVALUATION_ONLY"
        or suite.status != "FROZEN"
        or suite.sample_count <= 0
        or suite.source_snapshot_id == candidate.dataset_snapshot_id
        or policy.status != "ACTIVE"
        or job.status != "COMPLETED"
        or job.target_profile != "EDGE_MODEL_COMPONENT"
        or job.candidate_experiment_id != candidate.experiment_id
        or job.baseline_experiment_id != source.experiment_id
        or job.result_evaluation_id != evaluation.evaluation_id
        or not any(artifact.kind == "case_evidence" for artifact in evaluation_artifacts)
    ):
        raise ReleaseConflict("project_authorized_local_release_gate_failed")

    def artifact_binding(artifact: TrainingArtifactRecord) -> dict[str, Any]:
        return {
            "artifact_id": artifact.artifact_id,
            "content_hash": artifact.content_hash,
            "metadata": artifact.metadata_json,
            "object_key": artifact.object_key,
            "size_bytes": artifact.size_bytes,
        }

    manifest: dict[str, Any] = {
        "schema_version": PROJECT_AUTHORIZED_LOCAL_RELEASE_SCHEMA,
        "classification": PROJECT_AUTHORIZED_LOCAL_CLASSIFICATION,
        "data_classification": "SYNTHETIC_DATA",
        "authorization_classification": "PROJECT_OWNED_SYNTHETIC",
        "role_attestation": "LOCAL_ROLE_SIMULATION",
        "production_claim": False,
        "enterprise_production_data": False,
        "target_environment": "STAGING",
        "source_evidence_chain_sha256": plan.source_evidence_chain_sha256.removeprefix("sha256:"),
        "model_file_sha256": plan.model_file_sha256.removeprefix("sha256:"),
        "model_storage_subpath": plan.model_storage_subpath,
        "base_model": {
            "id": candidate.base_model_id,
            "digest": candidate.base_model_digest,
        },
        "adapter": artifact_binding(adapter),
        "quantization": {
            "experiment_id": candidate.experiment_id,
            **artifact_binding(quantized),
        },
        "tokenizer_digest": candidate.tokenizer_digest,
        "chat_template_digest": candidate.chat_template_digest,
        "quantization_profile_id": plan.quantization_profile_id,
        "training": {
            "experiment_id": candidate.experiment_id,
            "method": candidate.method,
            "config_hash": candidate.config_hash,
            "container_digest": candidate.container_digest,
            "git_commit": candidate.git_commit,
            "dataset_snapshot_id": candidate.dataset_snapshot_id,
            "dataset_manifest_hash": candidate.dataset_manifest_hash,
        },
        "source_training": {
            "experiment_id": source.experiment_id,
            "method": source.method,
            "config_hash": source.config_hash,
            "container_digest": source.container_digest,
            "git_commit": source.git_commit,
        },
        "evaluation": {
            "evaluation_id": evaluation.evaluation_id,
            "decision": evaluation.decision,
            "report_hash": evaluation.report_hash,
            "hard_gate_results": evaluation.hard_gate_results,
            "suite": {
                "id": suite.suite_id,
                "version": suite.version,
                "tier": suite.tier,
                "purpose": suite.purpose,
                "status": suite.status,
                "sample_count": suite.sample_count,
                "source_snapshot_id": suite.source_snapshot_id,
                "manifest_hash": suite.manifest_hash,
            },
            "policy": {
                "id": policy.policy_id,
                "version": policy.version,
                "status": policy.status,
                "hash": policy.policy_hash,
            },
            "runner": {
                "job_id": job.job_id,
                "status": job.status,
                "result_evaluation_id": job.result_evaluation_id,
                "target_profile": job.target_profile,
                "git_commit": job.runner_git_commit,
                "container_digest": job.container_digest,
                "config_hash": job.config_hash,
            },
            "artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "kind": artifact.kind,
                    "object_key": artifact.object_key,
                    "content_hash": artifact.content_hash,
                    "size_bytes": artifact.size_bytes,
                    "metadata": artifact.metadata_json,
                }
                for artifact in evaluation_artifacts
            ],
        },
        "runtime": {
            "profile_id": plan.runtime_profile_id,
            "git_commit": plan.runtime_git_commit,
            "image_repository": plan.runtime_image_repository,
            "image_reference": plan.runtime_image_reference,
            "image_digest": plan.runtime_image_digest,
            "inference_config": plan.inference_config,
            "hardware_profile": plan.hardware_profile,
        },
    }
    _digest_json(manifest)
    return manifest, candidate.experiment_id


def _project_authorized_local_evaluation_accepted(
    evaluation: ModelEvaluationRunRecord,
    *,
    role: str,
) -> bool:
    if role == "candidate":
        return (
            evaluation.decision == "CANDIDATE"
            and bool(evaluation.hard_gate_results)
            and all(evaluation.hard_gate_results.values())
        )
    if role == "stable":
        return stable_edge_evaluation_preserves_source(
            decision=evaluation.decision,
            candidate_score=evaluation.candidate_score,
            baseline_score=evaluation.baseline_score,
            quality_delta=evaluation.quality_delta,
            cost_improvement=evaluation.cost_improvement,
            hard_gate_results=evaluation.hard_gate_results,
            slice_metrics=evaluation.slice_metrics,
        )
    return False


def _evaluate_project_authorized_local_release_gates(
    session: Session,
    tenant_id: str,
    release: ModelReleaseRecord,
) -> dict[str, bool]:
    manifest = release.manifest_json
    runtime = manifest.get("runtime")
    try:
        if not isinstance(runtime, dict):
            raise ValueError("runtime is missing")
        plan = ProjectAuthorizedLocalReleasePlan(
            evaluation_id=release.evaluation_id,
            quantization_profile_id=str(manifest["quantization_profile_id"]),
            runtime_profile_id=str(runtime["profile_id"]),
            runtime_git_commit=str(runtime["git_commit"]),
            runtime_image_repository=str(runtime["image_repository"]),
            runtime_image_reference=str(runtime["image_reference"]),
            runtime_image_digest=str(runtime["image_digest"]),
            inference_config=dict(runtime["inference_config"]),
            hardware_profile=dict(runtime["hardware_profile"]),
            source_evidence_chain_sha256=str(manifest["source_evidence_chain_sha256"]),
            model_file_sha256=str(manifest["model_file_sha256"]),
            model_storage_subpath=str(manifest["model_storage_subpath"]),
        )
        _validate_project_authorized_local_plan(plan)
        expected, candidate_id = _build_project_authorized_local_manifest(
            session,
            tenant_id,
            plan,
        )
    except (KeyError, TypeError, ValueError, ReleaseConflict, ReleaseNotVisible) as exc:
        structlog.get_logger(__name__).warning(
            "release_gate_evidence_unreadable",
            release_id=release.release_id,
            stage="project_authorized_local",
            error_type=type(exc).__name__,
        )
        return {
            "project_authorized_contract_closed": False,
            "governed_quantization_current": False,
            "independent_edge_evaluation_passed": False,
            "staging_only_classification": False,
            "manifest_integrity": False,
        }
    return {
        "project_authorized_contract_closed": set(manifest) == set(expected),
        "governed_quantization_current": (
            candidate_id == release.candidate_experiment_id
            and manifest.get("quantization") == expected.get("quantization")
            and manifest.get("adapter") == expected.get("adapter")
        ),
        "independent_edge_evaluation_passed": (
            manifest.get("evaluation") == expected.get("evaluation")
        ),
        "staging_only_classification": (
            release.target_environment == "STAGING"
            and manifest.get("classification") == PROJECT_AUTHORIZED_LOCAL_CLASSIFICATION
            and manifest.get("data_classification") == "SYNTHETIC_DATA"
            and manifest.get("authorization_classification") == "PROJECT_OWNED_SYNTHETIC"
            and manifest.get("role_attestation") == "LOCAL_ROLE_SIMULATION"
            and manifest.get("production_claim") is False
            and manifest.get("enterprise_production_data") is False
        ),
        "manifest_integrity": expected == manifest,
    }


def _validate_plan(plan: ReleaseManifestPlan) -> None:
    smoke = plan.evaluation_admission == STAGING_DIAGNOSIS_SMOKE_14
    if plan.evaluation_admission not in {"STANDARD", STAGING_DIAGNOSIS_SMOKE_14}:
        raise ValueError("evaluation_admission_invalid")
    if smoke and plan.target_environment != "STAGING":
        raise ValueError("staging_smoke_is_staging_only")
    if (plan.prompt_evaluation_only or plan.local_staging_adapter) and not smoke:
        raise ValueError("staging_flags_require_smoke_admission")
    if not plan.local_staging_adapter and not plan.supply_chain_evidence_id:
        raise ValueError("supply_chain_evidence_required")
    if plan.local_staging_adapter and plan.supply_chain_evidence_id:
        raise ValueError("local_adapter_cannot_claim_signed_evidence")
    required_strings = (
        plan.evaluation_id,
        plan.quantization_profile_id,
        plan.runtime_profile_id,
        plan.embedding_model_id,
        plan.reranker_model_id,
        plan.supply_chain_evidence_id or "local-pinned",
    )
    if any(not value.strip() or len(value) > 255 for value in required_strings):
        raise ValueError("release manifest identifiers are invalid")
    if (
        not _valid_specialized_component_set(set(plan.component_evaluation_ids))
        or any(
            not isinstance(value, str) or not value.strip() or len(value) > 128
            for value in plan.component_evaluation_ids.values()
        )
        or len(set(plan.component_evaluation_ids.values())) != len(plan.component_evaluation_ids)
        or plan.evaluation_id in plan.component_evaluation_ids.values()
    ):
        raise ValueError(
            "component_evaluation_ids must bind distinct embedding, reranker, "
            "vlm and asr evaluations, with optional tts, timeseries and rul evaluations"
        )
    has_timeseries = "timeseries" in plan.component_evaluation_ids
    if has_timeseries != bool(plan.timeseries_model_id) or has_timeseries != bool(
        plan.timeseries_supply_chain_evidence_id
    ):
        raise ValueError(
            "timeseries evaluation, model and supply-chain evidence must be provided together"
        )
    for value in (plan.timeseries_model_id, plan.timeseries_supply_chain_evidence_id):
        if value is not None and (not value.strip() or len(value) > 255):
            raise ValueError("timeseries release identifiers are invalid")
    has_rul = "rul" in plan.component_evaluation_ids
    if has_rul != bool(plan.rul_model_id) or has_rul != bool(plan.rul_supply_chain_evidence_id):
        raise ValueError(
            "rul evaluation, model and supply-chain evidence must be provided together"
        )
    for value in (plan.rul_model_id, plan.rul_supply_chain_evidence_id):
        if value is not None and (not value.strip() or len(value) > 255):
            raise ValueError("rul release identifiers are invalid")
    if plan.reranker_supply_chain_evidence_id is not None and (
        not plan.reranker_supply_chain_evidence_id.strip()
        or len(plan.reranker_supply_chain_evidence_id) > 255
    ):
        raise ValueError("reranker supply-chain evidence identifier is invalid")
    if plan.embedding_supply_chain_evidence_id is not None and (
        not plan.embedding_supply_chain_evidence_id.strip()
        or len(plan.embedding_supply_chain_evidence_id) > 255
    ):
        raise ValueError("embedding supply-chain evidence identifier is invalid")
    has_tts = "tts" in plan.component_evaluation_ids
    if has_tts != bool(plan.tts_supply_chain_evidence_id):
        raise ValueError("tts evaluation and supply-chain evidence must be provided together")
    if plan.tts_supply_chain_evidence_id is not None and (
        not plan.tts_supply_chain_evidence_id.strip()
        or len(plan.tts_supply_chain_evidence_id) > 255
    ):
        raise ValueError("tts supply-chain evidence identifier is invalid")
    if plan.target_environment not in {"STAGING", "PRODUCTION"}:
        raise ValueError("target_environment is invalid")
    try:
        default_prompt_registry().get(plan.prompt_bundle_id)
    except PromptBundleNotDeployed as exc:
        raise ValueError("prompt_bundle_id is not deployed by this platform version") from exc
    if (
        plan.agent_graph_id == DIAGNOSIS_GRAPH_VERSION
        and plan.prompt_bundle_id != DIAGNOSIS_AGENT_PROMPT_ID
    ):
        raise ValueError("diagnosis graph v2 requires its matching agent prompt bundle")
    if plan.agent_graph_id != DIAGNOSIS_GRAPH_VERSION:
        raise ValueError("agent_graph_id is not deployed by this platform version")
    _validate_sha256_digest(plan.runtime_image_digest, "runtime_image_digest")
    _validate_image_repository(plan.runtime_image_repository)
    if set(plan.tool_versions) != REQUIRED_TOOL_IDS:
        raise ValueError("tool_versions must bind the complete production registry")
    registry = default_tool_registry()
    for tool_id, version in plan.tool_versions.items():
        registry.get(tool_id, version)
    if any(not key or not value for key, value in plan.mcp_server_versions.items()):
        raise ValueError("mcp_server_versions contains an empty identifier")
    required_multimodal = {"ocr", "vlm", "asr", "tts"}
    if set(plan.multimodal_model_ids) != required_multimodal or any(
        not value for value in plan.multimodal_model_ids.values()
    ):
        raise ValueError("multimodal_model_ids must bind ocr, vlm, asr and tts")
    _validate_inference_config(plan.inference_config)
    _validate_hardware_profile(plan.hardware_profile)
    if (plan.inference_config.get("engine") == "llama.cpp") != (
        plan.hardware_profile.get("accelerator") == "CPU"
    ):
        raise ValueError("inference_config and hardware_profile target different runtimes")
    for document in (
        plan.mcp_server_versions,
        plan.multimodal_model_ids,
        plan.inference_config,
        plan.hardware_profile,
    ):
        _digest_json(document)


def _project_authorized_import_binding(
    session: Session,
    tenant_id: str,
    *,
    component: str,
    candidate: TrainingExperimentRecord,
    evaluation: ModelEvaluationRunRecord,
    suite: EvaluationSuiteRecord,
    target_environment: ReleaseEnvironment,
) -> dict[str, Any] | None:
    if target_environment != "STAGING" or component not in {
        "LLM",
        "VLM",
        "ASR",
        "TTS",
        "RUL",
        "EMBEDDING",
        "RERANKER",
    }:
        return None
    record = session.scalar(
        select(EnterpriseModelAssetImportRecord).where(
            EnterpriseModelAssetImportRecord.tenant_id == tenant_id,
            EnterpriseModelAssetImportRecord.component == component,
            EnterpriseModelAssetImportRecord.imported_candidate_experiment_id
            == candidate.experiment_id,
            EnterpriseModelAssetImportRecord.imported_evaluation_id == evaluation.evaluation_id,
            EnterpriseModelAssetImportRecord.imported_suite_id == suite.suite_id,
            EnterpriseModelAssetImportRecord.status == "ACTIVE",
        )
    )
    if record is None:
        return None
    source_paths = list(record.source_paths_json)
    source_hashes = dict(record.source_file_hashes_json)
    imported_records = dict(record.imported_records_json)
    digests = [record.source_evidence_chain_sha256, *source_hashes.values()]
    if (
        record.import_hash != enterprise_model_import_hash(record)
        or record.release_scope != "STAGING_ONLY"
        or record.operational_classification != "LOCAL_STAGING_PROJECT_AUTHORIZED"
        or record.actual_sample_count != suite.sample_count
        or suite.tier != "PROJECT_AUTHORIZED"
        or suite.purpose != "EVALUATION_ONLY"
        or suite.status != "FROZEN"
        or suite.source_snapshot_id == candidate.dataset_snapshot_id
        or not source_paths
        or set(source_paths) != set(source_hashes)
        or any(
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
            for digest in digests
        )
        or imported_records.get("candidate") != candidate.experiment_id
        or imported_records.get("evaluation") != evaluation.evaluation_id
        or imported_records.get("suite") != suite.suite_id
    ):
        return None
    return {
        "schema_version": "enterprise-model-asset-import/v1",
        "import_id": record.import_id,
        "import_hash": record.import_hash,
        "component": record.component,
        "source_candidate_experiment_id": record.source_candidate_experiment_id,
        "source_evidence_chain_sha256": record.source_evidence_chain_sha256,
        "source_classification": record.source_classification,
        "source_decision": record.source_decision,
        "source_paths": source_paths,
        "source_file_hashes": dict(sorted(source_hashes.items())),
        "operational_classification": record.operational_classification,
        "actual_sample_count": record.actual_sample_count,
        "source_artifact_size_recorded": record.source_artifact_size_recorded,
        "release_scope": record.release_scope,
    }


def _evaluation_suite_is_release_governed(
    suite: EvaluationSuiteRecord,
    candidate: TrainingExperimentRecord,
    *,
    minimum_samples: int,
    project_authorized_import: dict[str, Any] | None,
) -> bool:
    common = (
        suite.purpose == "EVALUATION_ONLY"
        and suite.status == "FROZEN"
        and suite.source_snapshot_id != candidate.dataset_snapshot_id
    )
    standard_gold = suite.tier == "GOLD" and suite.sample_count >= minimum_samples
    project_authorized = (
        project_authorized_import is not None
        and suite.tier == "PROJECT_AUTHORIZED"
        and suite.sample_count == project_authorized_import.get("actual_sample_count")
    )
    return common and (standard_gold or project_authorized)


def _release_artifact_is_complete(
    *,
    size_bytes: Any,
    content_hash: Any,
    metadata: Any,
    project_authorized_import: dict[str, Any] | None,
) -> bool:
    if not isinstance(metadata, dict) or not isinstance(size_bytes, int):
        return False
    standard = (
        metadata.get("format") == "tar.gz"
        and metadata.get("serialization") == "safetensors"
        and size_bytes > 0
        and bool(content_hash)
    )
    if standard or project_authorized_import is None:
        return standard
    size_recorded = project_authorized_import.get("source_artifact_size_recorded")
    project_size_valid = (size_recorded is True and size_bytes > 0) or (
        size_recorded is False and size_bytes == 0
    )
    return bool(
        metadata.get("format") in {"tar.gz", "directory"}
        and metadata.get("serialization") == "safetensors"
        and project_size_valid
        and content_hash
    )


def _project_import_binding_is_current(
    current: dict[str, Any] | None,
    manifest_binding: Any,
) -> bool:
    if current is None:
        return manifest_binding is None
    return isinstance(manifest_binding, dict) and current == manifest_binding


def _build_manifest(
    session: Session,
    tenant_id: str,
    plan: ReleaseManifestPlan,
) -> tuple[dict[str, Any], str]:
    admission = (
        build_staging_smoke_admission(
            session,
            tenant_id,
            evaluation_id=plan.evaluation_id,
            prompt_bundle_id=plan.prompt_bundle_id,
            target_environment=plan.target_environment,
        )
        if plan.evaluation_admission == STAGING_DIAGNOSIS_SMOKE_14
        else None
    )
    evaluation = session.scalar(
        select(ModelEvaluationRunRecord).where(
            ModelEvaluationRunRecord.tenant_id == tenant_id,
            ModelEvaluationRunRecord.evaluation_id == plan.evaluation_id,
        )
    )
    if evaluation is None:
        raise ReleaseNotVisible
    candidate = _tenant_record(
        session,
        TrainingExperimentRecord,
        tenant_id,
        TrainingExperimentRecord.experiment_id,
        evaluation.candidate_experiment_id,
    )
    suite = _tenant_record(
        session,
        EvaluationSuiteRecord,
        tenant_id,
        EvaluationSuiteRecord.suite_id,
        evaluation.suite_id,
    )
    policy = _tenant_record(
        session,
        EvaluationPolicyRecord,
        tenant_id,
        EvaluationPolicyRecord.policy_id,
        evaluation.policy_id,
    )
    snapshot = _tenant_record(
        session,
        DatasetSnapshotRecord,
        tenant_id,
        DatasetSnapshotRecord.snapshot_id,
        candidate.dataset_snapshot_id,
    )
    project_authorized_import = _project_authorized_import_binding(
        session,
        tenant_id,
        component="LLM",
        candidate=candidate,
        evaluation=evaluation,
        suite=suite,
        target_environment=plan.target_environment,
    )
    dataset_contract_blocker = production_release_blocker_for_dataset_contract(
        snapshot.contract_version
    )
    if (
        dataset_contract_blocker is not None
        and project_authorized_import is None
        and admission is None
    ):
        raise ReleaseConflict(dataset_contract_blocker)
    index = _tenant_record(
        session,
        IndexReleaseRecord,
        tenant_id,
        IndexReleaseRecord.release_id,
        plan.index_release_id,
    )
    jobs = list(
        session.scalars(
            select(ModelEvaluationJobRecord).where(
                ModelEvaluationJobRecord.tenant_id == tenant_id,
                ModelEvaluationJobRecord.result_evaluation_id == evaluation.evaluation_id,
            )
        )
    )
    if len(jobs) != 1:
        raise ReleaseConflict("evaluation_must_have_one_independent_job")
    job = jobs[0]
    source_candidate = candidate
    quantized_artifact: TrainingArtifactRecord | None = None
    if candidate.method == "QUANTIZATION":
        source_candidate = _tenant_record(
            session,
            TrainingExperimentRecord,
            tenant_id,
            TrainingExperimentRecord.experiment_id,
            str(candidate.training_config.get("source_experiment_id", "")),
        )
        quantized_artifacts = list(
            session.scalars(
                select(TrainingArtifactRecord).where(
                    TrainingArtifactRecord.tenant_id == tenant_id,
                    TrainingArtifactRecord.experiment_id == candidate.experiment_id,
                    TrainingArtifactRecord.kind == "quantized_model_bundle",
                )
            )
        )
        if len(quantized_artifacts) != 1:
            raise ReleaseConflict("quantization_candidate_must_have_one_model_bundle")
        quantized_artifact = quantized_artifacts[0]
        target_runtime = candidate.training_config.get("target_runtime")
        serialization = quantized_artifact.metadata_json.get("serialization")
        expected_serialization = "gguf" if target_runtime == "LLAMA_CPP" else "compressed-tensors"
        if (
            target_runtime not in {"VLLM", "LLAMA_CPP"}
            or serialization != expected_serialization
            or candidate.training_config.get("quantization_profile_id")
            != plan.quantization_profile_id
            or quantized_artifact.metadata_json.get("quantization_profile_id")
            != plan.quantization_profile_id
            or quantized_artifact.metadata_json.get("approval_inheritance") is not False
        ):
            raise ReleaseConflict("quantization_release_profile_binding_invalid")
        if target_runtime == "VLLM":
            if (
                plan.inference_config.get("engine") != "vllm"
                or plan.inference_config.get("enable_lora") is not False
                or plan.inference_config.get("max_lora_rank") != 0
            ):
                raise ReleaseConflict("quantized_runtime_must_load_the_merged_model_bundle")
        elif (
            plan.inference_config.get("engine") != "llama.cpp"
            or quantized_artifact.metadata_json.get("runtime", {}).get("target_runtime")
            != "llama.cpp"
            or not _valid_artifact_filename(
                quantized_artifact.metadata_json.get("runtime", {}).get("model_file")
            )
        ):
            raise ReleaseConflict("GGUF_runtime_must_load_the_reviewed_model_bundle")
    elif (
        plan.inference_config.get("engine") != "vllm"
        or plan.inference_config.get("enable_lora") is not True
    ):
        raise ReleaseConflict("PEFT_runtime_must_enable_the_adapter")
    adapters = list(
        session.scalars(
            select(TrainingArtifactRecord).where(
                TrainingArtifactRecord.tenant_id == tenant_id,
                TrainingArtifactRecord.experiment_id == source_candidate.experiment_id,
                TrainingArtifactRecord.kind == "adapter_bundle",
            )
        )
    )
    if len(adapters) != 1:
        raise ReleaseConflict("candidate_must_have_one_adapter_bundle")
    adapter = adapters[0]
    if candidate.method == "QUANTIZATION" and (
        candidate.training_config.get("source_artifact_id") != adapter.artifact_id
        or str(candidate.training_config.get("source_artifact_hash", "")).removeprefix("sha256:")
        != adapter.content_hash
    ):
        raise ReleaseConflict("quantization_source_adapter_binding_changed")
    runtime = {
        "profile_id": plan.runtime_profile_id,
        "image_repository": plan.runtime_image_repository,
        "image_digest": plan.runtime_image_digest,
        "inference_config": plan.inference_config,
        "hardware_profile": plan.hardware_profile,
    }
    if plan.local_staging_adapter:
        if admission is None:
            raise ReleaseConflict("local_adapter_requires_imported_staging_smoke")
        supply_chain = local_adapter_binding(
            session,
            tenant_id,
            candidate_id=candidate.experiment_id,
            admission=admission,
            runtime=runtime,
            environment=plan.target_environment,
        )
    else:
        if plan.supply_chain_evidence_id is None:
            raise ReleaseConflict("supply_chain_evidence_required")
        supply_chain_evidence = _tenant_record(
            session,
            SupplyChainEvidenceRecord,
            tenant_id,
            SupplyChainEvidenceRecord.evidence_id,
            plan.supply_chain_evidence_id,
        )
        _require_supply_chain_binding(
            plan, candidate, quantized_artifact or adapter, supply_chain_evidence
        )
        supply_chain = _supply_chain_manifest(supply_chain_evidence)
    evidence = list(
        session.scalars(
            select(EvaluationArtifactRecord)
            .where(
                EvaluationArtifactRecord.tenant_id == tenant_id,
                EvaluationArtifactRecord.evaluation_id == evaluation.evaluation_id,
            )
            .order_by(EvaluationArtifactRecord.kind, EvaluationArtifactRecord.artifact_id)
        )
    )
    if not any(artifact.kind == "case_evidence" for artifact in evidence):
        raise ReleaseConflict("evaluation_case_evidence_required")
    component_runtime_model_ids = {
        "embedding": plan.embedding_model_id,
        "reranker": plan.reranker_model_id,
        "vlm": plan.multimodal_model_ids["vlm"],
        "asr": plan.multimodal_model_ids["asr"],
    }
    tts_runtime_id = plan.multimodal_model_ids["tts"]
    trained_tts = session.scalar(
        select(TrainingExperimentRecord).where(
            TrainingExperimentRecord.tenant_id == tenant_id,
            TrainingExperimentRecord.experiment_id == tts_runtime_id,
            TrainingExperimentRecord.method == "TTS",
        )
    )
    if trained_tts is not None and "tts" not in plan.component_evaluation_ids:
        raise ReleaseConflict("platform_tts_runtime_requires_component_evaluation")
    if "tts" in plan.component_evaluation_ids:
        component_runtime_model_ids["tts"] = tts_runtime_id
    if plan.timeseries_model_id is not None:
        component_runtime_model_ids["timeseries"] = plan.timeseries_model_id
    if plan.rul_model_id is not None:
        component_runtime_model_ids["rul"] = plan.rul_model_id
    specialized_components = _build_specialized_components(
        session,
        tenant_id,
        plan.component_evaluation_ids,
        component_runtime_model_ids,
        target_environment=plan.target_environment,
    )
    for component in ("vlm", "asr"):
        specialized_components[component] = _attach_media_supply_chain(
            session,
            tenant_id,
            specialized_components[component],
            evidence_id=getattr(plan, f"{component}_supply_chain_evidence_id"),
        )
    if plan.embedding_supply_chain_evidence_id is not None:
        specialized_components["embedding"] = _attach_embedding_runtime(
            session,
            tenant_id,
            specialized_components["embedding"],
            evidence_id=plan.embedding_supply_chain_evidence_id,
        )
    if plan.reranker_supply_chain_evidence_id is not None:
        specialized_components["reranker"] = _attach_reranker_runtime(
            session,
            tenant_id,
            specialized_components["reranker"],
            evidence_id=plan.reranker_supply_chain_evidence_id,
        )
    if "tts" in specialized_components:
        evidence_id = plan.tts_supply_chain_evidence_id
        if evidence_id is None:  # guarded by _validate_plan; keeps narrowing explicit.
            raise ReleaseConflict("tts_supply_chain_evidence_required")
        specialized_components["tts"] = _attach_tts_runtime(
            session,
            tenant_id,
            specialized_components["tts"],
            evidence_id=evidence_id,
        )
    if "timeseries" in specialized_components:
        evidence_id = plan.timeseries_supply_chain_evidence_id
        if evidence_id is None:  # guarded by _validate_plan; keeps narrowing explicit.
            raise ReleaseConflict("timeseries_supply_chain_evidence_required")
        specialized_components["timeseries"] = _attach_timeseries_runtime(
            session,
            tenant_id,
            specialized_components["timeseries"],
            evidence_id=evidence_id,
        )
    if "rul" in specialized_components:
        evidence_id = plan.rul_supply_chain_evidence_id
        if evidence_id is None:  # guarded by _validate_plan; keeps narrowing explicit.
            raise ReleaseConflict("rul_supply_chain_evidence_required")
        specialized_components["rul"] = _attach_rul_runtime(
            session,
            tenant_id,
            specialized_components["rul"],
            evidence_id=evidence_id,
        )
    corpus = _corpus_descriptor(session, tenant_id, index)
    rollback_manifest_hash = None
    if plan.rollback_release_id is not None:
        rollback = _release_or_hidden(session, tenant_id, plan.rollback_release_id)
        rollback_manifest_hash = rollback.manifest_hash
    prompt_bundle = _release_prompt_bundle_binding(
        session,
        tenant_id,
        plan.prompt_bundle_id,
        evaluation_only=plan.prompt_evaluation_only,
    )

    manifest: dict[str, Any] = {
        "schema_version": "ai-release-manifest/v5",
        "base_model": {
            "id": candidate.base_model_id,
            "digest": candidate.base_model_digest,
        },
        "adapter": {
            "artifact_id": adapter.artifact_id,
            "content_hash": adapter.content_hash,
            "metadata": adapter.metadata_json,
            "object_key": adapter.object_key,
            "size_bytes": adapter.size_bytes,
        },
        "quantization": (
            {
                "experiment_id": candidate.experiment_id,
                "artifact_id": quantized_artifact.artifact_id,
                "content_hash": quantized_artifact.content_hash,
                "metadata": quantized_artifact.metadata_json,
                "object_key": quantized_artifact.object_key,
                "size_bytes": quantized_artifact.size_bytes,
            }
            if quantized_artifact is not None
            else None
        ),
        "tokenizer_digest": candidate.tokenizer_digest,
        "chat_template_digest": candidate.chat_template_digest,
        "quantization_profile_id": plan.quantization_profile_id,
        "training": {
            "experiment_id": candidate.experiment_id,
            "method": candidate.method,
            "config_hash": candidate.config_hash,
            "container_digest": candidate.container_digest,
            "git_commit": candidate.git_commit,
            "dataset_snapshot_id": candidate.dataset_snapshot_id,
            "dataset_manifest_hash": candidate.dataset_manifest_hash,
        },
        "source_training": (
            {
                "experiment_id": source_candidate.experiment_id,
                "method": source_candidate.method,
                "config_hash": source_candidate.config_hash,
                "container_digest": source_candidate.container_digest,
                "git_commit": source_candidate.git_commit,
            }
            if source_candidate.experiment_id != candidate.experiment_id
            else None
        ),
        "dataset_snapshot": {
            "id": snapshot.snapshot_id,
            "contract_version": snapshot.contract_version,
            "manifest_hash": snapshot.manifest_hash,
            "lineage_status": snapshot.lineage_status,
        },
        "evaluation": {
            "evaluation_id": evaluation.evaluation_id,
            "decision": evaluation.decision,
            "report_hash": evaluation.report_hash,
            "suite": {
                "id": suite.suite_id,
                "version": suite.version,
                "tier": suite.tier,
                "purpose": suite.purpose,
                "status": suite.status,
                "sample_count": suite.sample_count,
                "source_snapshot_id": suite.source_snapshot_id,
                "manifest_hash": suite.manifest_hash,
            },
            "policy": {
                "id": policy.policy_id,
                "version": policy.version,
                "status": policy.status,
                "hash": policy.policy_hash,
            },
            "runner": {
                "job_id": job.job_id,
                "status": job.status,
                "result_evaluation_id": job.result_evaluation_id,
                "target_profile": job.target_profile,
                "git_commit": job.runner_git_commit,
                "container_digest": job.container_digest,
                "config_hash": job.config_hash,
            },
            "artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "kind": artifact.kind,
                    "object_key": artifact.object_key,
                    "content_hash": artifact.content_hash,
                    "size_bytes": artifact.size_bytes,
                    "metadata": artifact.metadata_json,
                }
                for artifact in evidence
            ],
        },
        "runtime": runtime,
        "prompt_bundle_id": plan.prompt_bundle_id,
        "prompt_bundle": prompt_bundle,
        "agent_graph_id": plan.agent_graph_id,
        "tool_registry": dict(sorted(plan.tool_versions.items())),
        "mcp_server_versions": dict(sorted(plan.mcp_server_versions.items())),
        "retrieval": {
            "index_release_id": index.release_id,
            "index_content_checksum": index.content_checksum,
            "corpus_snapshot_id": corpus["snapshot_id"],
            "corpus_snapshot_hash": corpus["snapshot_hash"],
            "parser_versions": corpus["parser_versions"],
            "chunker_versions": corpus["chunker_versions"],
            "embedding_model_id": plan.embedding_model_id,
            "reranker_model_id": plan.reranker_model_id,
        },
        "specialized_components": specialized_components,
        "multimodal_model_ids": dict(sorted(plan.multimodal_model_ids.items())),
        "project_authorized_import": project_authorized_import,
        "supply_chain": supply_chain,
        "target_environment": plan.target_environment,
        "rollback": {
            "release_id": plan.rollback_release_id,
            "manifest_hash": rollback_manifest_hash,
        },
    }
    if admission is not None:
        manifest.update(
            evaluation_admission=admission,
            prompt_evaluation_only=plan.prompt_evaluation_only,
            local_staging_adapter=plan.local_staging_adapter,
        )
    _digest_json(manifest)
    return manifest, candidate.experiment_id


def _build_specialized_components(
    session: Session,
    tenant_id: str,
    evaluation_ids: dict[str, str],
    runtime_model_ids: dict[str, str],
    *,
    target_environment: ReleaseEnvironment,
) -> dict[str, dict[str, Any]]:
    if not _valid_specialized_component_set(set(evaluation_ids)) or set(runtime_model_ids) != set(
        evaluation_ids
    ):
        raise ReleaseConflict("specialized_component_set_is_incomplete")
    return {
        component: _build_specialized_component(
            session,
            tenant_id,
            component=component,
            evaluation_id=evaluation_ids[component],
            runtime_model_id=runtime_model_ids[component],
            target_environment=target_environment,
        )
        for component in sorted(evaluation_ids)
    }


def _valid_specialized_component_set(components: set[str]) -> bool:
    return CORE_SPECIALIZED_COMPONENTS.issubset(components) and components.issubset(
        SPECIALIZED_COMPONENT_REQUIREMENTS
    )


def _build_specialized_component(
    session: Session,
    tenant_id: str,
    *,
    component: str,
    evaluation_id: str,
    runtime_model_id: str,
    target_environment: ReleaseEnvironment,
) -> dict[str, Any]:
    requirement = SPECIALIZED_COMPONENT_REQUIREMENTS.get(component)
    if requirement is None:
        raise ReleaseConflict("specialized_component_is_unsupported")
    evaluation = _tenant_record(
        session,
        ModelEvaluationRunRecord,
        tenant_id,
        ModelEvaluationRunRecord.evaluation_id,
        evaluation_id,
    )
    candidate = _tenant_record(
        session,
        TrainingExperimentRecord,
        tenant_id,
        TrainingExperimentRecord.experiment_id,
        evaluation.candidate_experiment_id,
    )
    suite = _tenant_record(
        session,
        EvaluationSuiteRecord,
        tenant_id,
        EvaluationSuiteRecord.suite_id,
        evaluation.suite_id,
    )
    policy = _tenant_record(
        session,
        EvaluationPolicyRecord,
        tenant_id,
        EvaluationPolicyRecord.policy_id,
        evaluation.policy_id,
    )
    jobs = list(
        session.scalars(
            select(ModelEvaluationJobRecord).where(
                ModelEvaluationJobRecord.tenant_id == tenant_id,
                ModelEvaluationJobRecord.result_evaluation_id == evaluation.evaluation_id,
            )
        )
    )
    artifacts = list(
        session.scalars(
            select(TrainingArtifactRecord).where(
                TrainingArtifactRecord.tenant_id == tenant_id,
                TrainingArtifactRecord.experiment_id == candidate.experiment_id,
                TrainingArtifactRecord.kind == requirement["artifact_kind"],
            )
        )
    )
    evidence = list(
        session.scalars(
            select(EvaluationArtifactRecord)
            .where(
                EvaluationArtifactRecord.tenant_id == tenant_id,
                EvaluationArtifactRecord.evaluation_id == evaluation.evaluation_id,
            )
            .order_by(EvaluationArtifactRecord.kind, EvaluationArtifactRecord.artifact_id)
        )
    )
    expected_gates = set(requirement["hard_gates"])
    project_authorized_import = _project_authorized_import_binding(
        session,
        tenant_id,
        component=component.upper(),
        candidate=candidate,
        evaluation=evaluation,
        suite=suite,
        target_environment=target_environment,
    )
    if candidate.method != requirement["method"]:
        raise ReleaseConflict(f"{component}_evaluation_candidate_method_mismatch")
    if runtime_model_id != candidate.experiment_id:
        raise ReleaseConflict(f"{component}_runtime_model_must_bind_candidate_experiment")
    if candidate.status != "COMPLETED" or candidate.license_status != "APPROVED":
        raise ReleaseConflict(f"{component}_candidate_is_not_release_eligible")
    if (
        evaluation.status != "COMPLETED"
        or evaluation.decision != "CANDIDATE"
        or evaluation.primary_metric not in requirement["primary_metrics"]
        or evaluation.primary_metric != policy.primary_metric
        or set(evaluation.hard_gate_results) != expected_gates
        or not all(evaluation.hard_gate_results.values())
    ):
        raise ReleaseConflict(f"{component}_evaluation_is_not_an_approved_candidate")
    if (
        not _evaluation_suite_is_release_governed(
            suite,
            candidate,
            minimum_samples=int(requirement.get("minimum_suite_samples", 500)),
            project_authorized_import=project_authorized_import,
        )
        or policy.status != "ACTIVE"
    ):
        raise ReleaseConflict(f"{component}_evaluation_governance_is_incomplete")
    if (
        len(jobs) != 1
        or jobs[0].status != "COMPLETED"
        or jobs[0].result_evaluation_id != evaluation.evaluation_id
        or jobs[0].target_profile != requirement["profile"]
        or not jobs[0].config_hash
    ):
        raise ReleaseConflict(f"{component}_independent_evaluation_job_is_incomplete")
    if len(artifacts) != 1:
        raise ReleaseConflict(f"{component}_candidate_requires_one_training_artifact")
    artifact = artifacts[0]
    artifact_metadata = artifact.metadata_json
    if component == "tts":
        profile = artifact_metadata.get("voice_profile")
        if not isinstance(profile, dict):
            raise ReleaseConflict("tts_training_artifact_voice_profile_is_missing")
        artifact_metadata = {
            key: value for key, value in artifact_metadata.items() if key != "voice_profile"
        }
        artifact_metadata["voice_profile"] = {
            key: profile.get(key)
            for key in (
                "voice_profile_id",
                "language",
                "sampling_rate",
                "speaker_embedding_dimension",
                "speech_contract_version",
            )
        }
    if not _release_artifact_is_complete(
        size_bytes=artifact.size_bytes,
        content_hash=artifact.content_hash,
        metadata=artifact.metadata_json,
        project_authorized_import=project_authorized_import,
    ):
        raise ReleaseConflict(f"{component}_training_artifact_is_incomplete")
    case_evidence = [item for item in evidence if item.kind == "case_evidence"]
    if not case_evidence or any(
        item.size_bytes <= 0 or not item.content_hash for item in case_evidence
    ):
        raise ReleaseConflict(f"{component}_case_evidence_is_incomplete")
    job = jobs[0]
    if component == "rul":
        _require_rul_evaluation_binding(
            session,
            tenant_id,
            candidate=candidate,
            evaluation=evaluation,
            job=job,
        )
    component_manifest: dict[str, Any] = {
        "component": component,
        "runtime_model_id": candidate.experiment_id,
        "training": {
            "experiment_id": candidate.experiment_id,
            "method": candidate.method,
            "base_model_id": candidate.base_model_id,
            "base_model_digest": candidate.base_model_digest,
            "tokenizer_digest": candidate.tokenizer_digest,
            "config_hash": candidate.config_hash,
            "dataset_snapshot_id": candidate.dataset_snapshot_id,
            "dataset_manifest_hash": candidate.dataset_manifest_hash,
        },
        "artifact": {
            "artifact_id": artifact.artifact_id,
            "kind": artifact.kind,
            "object_key": artifact.object_key,
            "content_hash": artifact.content_hash,
            "size_bytes": artifact.size_bytes,
            "metadata": artifact_metadata,
        },
        "evaluation": {
            "evaluation_id": evaluation.evaluation_id,
            "decision": evaluation.decision,
            "primary_metric": evaluation.primary_metric,
            "report_hash": evaluation.report_hash,
            "hard_gate_results": evaluation.hard_gate_results,
            "suite_id": suite.suite_id,
            "suite_manifest_hash": suite.manifest_hash,
            "suite_sample_count": suite.sample_count,
            "policy_id": policy.policy_id,
            "policy_hash": policy.policy_hash,
            "job_id": job.job_id,
            "target_profile": job.target_profile,
            "runner_config_hash": job.config_hash,
            "case_evidence": [
                {
                    "artifact_id": item.artifact_id,
                    "content_hash": item.content_hash,
                    "size_bytes": item.size_bytes,
                }
                for item in case_evidence
            ],
        },
        "project_authorized_import": project_authorized_import,
    }
    if component == "vlm":
        component_manifest["dataset_ablation"] = _vlm_dataset_ablation_evidence(
            session,
            tenant_id,
            evaluation=evaluation,
            candidate=candidate,
            suite=suite,
            policy=policy,
        )
    return component_manifest


def _require_rul_evaluation_binding(
    session: Session,
    tenant_id: str,
    *,
    candidate: TrainingExperimentRecord,
    evaluation: ModelEvaluationRunRecord,
    job: ModelEvaluationJobRecord,
) -> None:
    baseline = _tenant_record(
        session,
        TrainingExperimentRecord,
        tenant_id,
        TrainingExperimentRecord.experiment_id,
        evaluation.baseline_experiment_id,
    )
    quantiles = baseline.training_config.get("empirical_quantiles_minutes")
    if (
        baseline.method != "RUL_EMPIRICAL_BASELINE"
        or baseline.status != "PLANNED"
        or baseline.dataset_snapshot_id != candidate.dataset_snapshot_id
        or baseline.training_config.get("source_manifest_hash") != candidate.dataset_manifest_hash
        or job.candidate_experiment_id != candidate.experiment_id
        or job.baseline_experiment_id != baseline.experiment_id
        or not isinstance(quantiles, list)
        or len(quantiles) != 3
    ):
        raise ReleaseConflict("rul_empirical_baseline_binding_is_invalid")
    try:
        values = tuple(float(item) for item in quantiles)
    except (TypeError, ValueError) as exc:
        raise ReleaseConflict("rul_empirical_baseline_binding_is_invalid") from exc
    if (
        not all(math.isfinite(item) and item > 0 for item in values)
        or not values[0] <= values[1] <= values[2]
    ):
        raise ReleaseConflict("rul_empirical_baseline_binding_is_invalid")


def _vlm_dataset_ablation_evidence(
    session: Session,
    tenant_id: str,
    *,
    evaluation: ModelEvaluationRunRecord,
    candidate: TrainingExperimentRecord,
    suite: EvaluationSuiteRecord,
    policy: EvaluationPolicyRecord,
) -> dict[str, Any] | None:
    candidate_snapshot = _tenant_record(
        session,
        DatasetSnapshotRecord,
        tenant_id,
        DatasetSnapshotRecord.snapshot_id,
        candidate.dataset_snapshot_id,
    )
    if candidate_snapshot.base_snapshot_id is None:
        return None
    baseline = _tenant_record(
        session,
        TrainingExperimentRecord,
        tenant_id,
        TrainingExperimentRecord.experiment_id,
        evaluation.baseline_experiment_id,
    )
    try:
        context = synthetic_ablation_context(
            session,
            tenant_id,
            candidate,
            baseline,
        )
    except SyntheticAblationContractError as exc:
        raise ReleaseConflict("vlm_synthetic_ablation_contract_is_invalid") from exc
    quality_lift_min = policy.thresholds.get("quality_lift_min")
    if (
        context is None
        or evaluation.comparison_kind != SYNTHETIC_DATASET_ABLATION
        or evaluation.comparison_context != context
        or baseline.status != "COMPLETED"
        or suite.source_snapshot_id == baseline.dataset_snapshot_id
        or isinstance(quality_lift_min, bool)
        or not isinstance(quality_lift_min, (int, float))
        or evaluation.quality_delta < float(quality_lift_min)
        or evaluation.ci_low <= 0
        or evaluation.candidate_score <= evaluation.baseline_score
    ):
        raise ReleaseConflict("vlm_synthetic_ablation_real_quality_gain_required")
    return {
        "comparison_kind": SYNTHETIC_DATASET_ABLATION,
        **context,
        "evaluation_suite_id": suite.suite_id,
        "evaluation_suite_manifest_hash": suite.manifest_hash,
        "candidate_score": evaluation.candidate_score,
        "baseline_score": evaluation.baseline_score,
        "quality_delta": evaluation.quality_delta,
        "ci_low": evaluation.ci_low,
        "ci_high": evaluation.ci_high,
        "quality_lift_min": float(quality_lift_min),
    }


def _attach_media_supply_chain(
    session: Session, tenant_id: str, component: dict[str, Any], *, evidence_id: str | None
) -> dict[str, Any]:
    name = component["component"]
    if not evidence_id:
        raise ReleaseConflict(f"{name}_supply_chain_evidence_required")
    try:
        binding = component_supply_chain_identity(
            session, tenant_id, name.upper(), component["runtime_model_id"]
        )
    except ValueError as exc:
        raise ReleaseConflict(f"{name}_{str(exc).lower()}") from exc
    evidence = session.scalar(
        select(SupplyChainEvidenceRecord).where(
            SupplyChainEvidenceRecord.tenant_id == tenant_id,
            SupplyChainEvidenceRecord.evidence_id == evidence_id,
        )
    )
    if evidence is None or not component_evidence_matches(evidence, tenant_id, binding):
        raise ReleaseConflict(f"{name}_supply_chain_evidence_mismatch")
    if _artifact_digest(component["artifact"]["content_hash"]) != binding.model_artifact_digest:
        raise ReleaseConflict(f"{name}_supply_chain_artifact_mismatch")
    return {
        **component,
        "runtime": {
            **component.get("runtime", {}),
            "image_repository": binding.image_repository,
            "image_digest": binding.image_digest,
        },
        "source": {"repository": binding.source_repository, "revision": binding.source_revision},
        "supply_chain": _supply_chain_manifest(evidence),
    }


def _attach_timeseries_runtime(
    session: Session,
    tenant_id: str,
    component: dict[str, Any],
    *,
    evidence_id: str,
) -> dict[str, Any]:
    candidate = _tenant_record(
        session,
        TrainingExperimentRecord,
        tenant_id,
        TrainingExperimentRecord.experiment_id,
        str(component["training"]["experiment_id"]),
    )
    artifact = _tenant_record(
        session,
        TrainingArtifactRecord,
        tenant_id,
        TrainingArtifactRecord.artifact_id,
        str(component["artifact"]["artifact_id"]),
    )
    job = _tenant_record(
        session,
        ModelEvaluationJobRecord,
        tenant_id,
        ModelEvaluationJobRecord.job_id,
        str(component["evaluation"]["job_id"]),
    )
    evidence = _tenant_record(
        session,
        SupplyChainEvidenceRecord,
        tenant_id,
        SupplyChainEvidenceRecord.evidence_id,
        evidence_id,
    )
    _require_artifact_supply_chain_binding(candidate, artifact, evidence)
    config = candidate.training_config
    max_sequence_length = config.get("max_sequence_length")
    gpu_count = candidate.hardware_topology.get("count")
    if (
        isinstance(max_sequence_length, bool)
        or not isinstance(max_sequence_length, int)
        or not 3 <= max_sequence_length <= 4096
        or config.get("signal_order") != list(SIGNAL_ORDER)
        or candidate.hardware_topology.get("accelerator") != "NVIDIA_GPU"
        or gpu_count != 1
    ):
        raise ReleaseConflict("timeseries_runtime_hardware_or_sequence_contract_is_invalid")
    runner = job.runner_config
    threshold = runner.get("anomaly_threshold")
    precision = runner.get("precision")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or float(threshold) <= 0
        or precision not in {"bfloat16", "float16"}
    ):
        raise ReleaseConflict("timeseries_evaluation_runtime_config_is_invalid")
    result = dict(component)
    result["runtime"] = {
        "profile_id": TIMESERIES_RUNTIME_PROFILE,
        "image_repository": evidence.image_repository,
        "image_digest": evidence.image_digest,
        "inference_config": {
            "engine": "native-pytorch-transformer-reconstruction",
            "precision": precision,
            "anomaly_threshold": float(threshold),
            "max_sequence_length": max_sequence_length,
            "signal_order": list(SIGNAL_ORDER),
        },
        "hardware_profile": {
            "accelerator": "NVIDIA_GPU",
            "gpu_count": 1,
        },
    }
    result["supply_chain"] = _supply_chain_manifest(evidence)
    return result


def _attach_rul_runtime(
    session: Session,
    tenant_id: str,
    component: dict[str, Any],
    *,
    evidence_id: str,
) -> dict[str, Any]:
    candidate = _tenant_record(
        session,
        TrainingExperimentRecord,
        tenant_id,
        TrainingExperimentRecord.experiment_id,
        str(component["training"]["experiment_id"]),
    )
    artifact = _tenant_record(
        session,
        TrainingArtifactRecord,
        tenant_id,
        TrainingArtifactRecord.artifact_id,
        str(component["artifact"]["artifact_id"]),
    )
    job = _tenant_record(
        session,
        ModelEvaluationJobRecord,
        tenant_id,
        ModelEvaluationJobRecord.job_id,
        str(component["evaluation"]["job_id"]),
    )
    evidence = _tenant_record(
        session,
        SupplyChainEvidenceRecord,
        tenant_id,
        SupplyChainEvidenceRecord.evidence_id,
        evidence_id,
    )
    _require_artifact_supply_chain_binding(candidate, artifact, evidence)
    config = candidate.training_config
    max_sequence_length = config.get("max_sequence_length")
    gpu_count = candidate.hardware_topology.get("count")
    quantiles = config.get("quantiles")
    if (
        candidate.method != "RUL_TRANSFORMER"
        or candidate.base_model_digest != "architecture:rul-transformer-v1"
        or config.get("base_model_revision") != "rul-transformer-v1"
        or config.get("sequence_contract_version") != RUL_CONTRACT_VERSION
        or config.get("signal_order") != list(RUL_SIGNAL_ORDER)
        or quantiles != [0.1, 0.5, 0.9]
        or isinstance(max_sequence_length, bool)
        or not isinstance(max_sequence_length, int)
        or not 3 <= max_sequence_length <= 4096
        or candidate.hardware_topology.get("accelerator") != "NVIDIA_GPU"
        or gpu_count != 1
    ):
        raise ReleaseConflict("rul_runtime_model_or_hardware_contract_is_invalid")
    precision = job.runner_config.get("precision")
    if precision not in {"bfloat16", "float16"}:
        raise ReleaseConflict("rul_evaluation_runtime_config_is_invalid")
    result = dict(component)
    result["runtime"] = {
        "profile_id": RUL_RUNTIME_PROFILE,
        "image_repository": evidence.image_repository,
        "image_digest": evidence.image_digest,
        "inference_config": {
            "engine": "native-pytorch-rul-quantile-regression",
            "precision": precision,
            "quantiles": [0.1, 0.5, 0.9],
            "max_sequence_length": max_sequence_length,
            "signal_order": list(RUL_SIGNAL_ORDER),
            "sequence_contract_version": RUL_CONTRACT_VERSION,
        },
        "hardware_profile": {
            "accelerator": "NVIDIA_GPU",
            "gpu_count": 1,
        },
    }
    result["supply_chain"] = _supply_chain_manifest(evidence)
    return result


def _attach_reranker_runtime(
    session: Session,
    tenant_id: str,
    component: dict[str, Any],
    *,
    evidence_id: str,
) -> dict[str, Any]:
    candidate = _tenant_record(
        session,
        TrainingExperimentRecord,
        tenant_id,
        TrainingExperimentRecord.experiment_id,
        str(component["training"]["experiment_id"]),
    )
    artifact = _tenant_record(
        session,
        TrainingArtifactRecord,
        tenant_id,
        TrainingArtifactRecord.artifact_id,
        str(component["artifact"]["artifact_id"]),
    )
    evidence = _tenant_record(
        session,
        SupplyChainEvidenceRecord,
        tenant_id,
        SupplyChainEvidenceRecord.evidence_id,
        evidence_id,
    )
    _require_artifact_supply_chain_binding(candidate, artifact, evidence)
    config = candidate.training_config
    max_sequence_length = config.get("max_sequence_length")
    precision = config.get("precision")
    batch_size = config.get("per_device_eval_batch_size")
    calibration_profile_id = config.get("calibration_profile_id")
    calibration = config.get("calibration")
    if calibration_profile_id is None and isinstance(calibration, dict):
        calibration_profile_id = calibration.get("profile_id")
    if (
        candidate.method != "RERANKER"
        or isinstance(max_sequence_length, bool)
        or not isinstance(max_sequence_length, int)
        or not 128 <= max_sequence_length <= 8192
        or precision not in {"bfloat16", "float16"}
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= 128
        or calibration_profile_id not in {None, "industrial-reranker-evidence-calibration-v1"}
    ):
        raise ReleaseConflict("reranker_runtime_config_is_invalid")
    result = dict(component)
    result["runtime"] = {
        "profile_id": RERANKER_RUNTIME_PROFILE,
        "image_repository": evidence.image_repository,
        "image_digest": evidence.image_digest,
        "inference_config": {
            "engine": "sentence-transformers-cross-encoder",
            "precision": precision,
            "batch_size": batch_size,
            "max_sequence_length": max_sequence_length,
            "calibration_profile_id": (calibration_profile_id or "AUTO_FROM_IMMUTABLE_BUNDLE"),
        },
        "hardware_profile": {
            "accelerator": "NVIDIA_GPU",
            "gpu_count": 1,
        },
    }
    result["supply_chain"] = _supply_chain_manifest(evidence)
    return result


def _attach_tts_runtime(
    session: Session,
    tenant_id: str,
    component: dict[str, Any],
    *,
    evidence_id: str,
) -> dict[str, Any]:
    candidate = _tenant_record(
        session,
        TrainingExperimentRecord,
        tenant_id,
        TrainingExperimentRecord.experiment_id,
        str(component["training"]["experiment_id"]),
    )
    artifact = _tenant_record(
        session,
        TrainingArtifactRecord,
        tenant_id,
        TrainingArtifactRecord.artifact_id,
        str(component["artifact"]["artifact_id"]),
    )
    evidence = _tenant_record(
        session,
        SupplyChainEvidenceRecord,
        tenant_id,
        SupplyChainEvidenceRecord.evidence_id,
        evidence_id,
    )
    _require_artifact_supply_chain_binding(candidate, artifact, evidence)
    config = candidate.training_config
    metadata = artifact.metadata_json
    voice_profile = metadata.get("voice_profile")
    sampling_rate = config.get("sampling_rate")
    precision = config.get("precision")
    voice_profile_id = config.get("voice_profile_id")
    if (
        candidate.method != "TTS"
        or not isinstance(voice_profile, dict)
        or isinstance(sampling_rate, bool)
        or not isinstance(sampling_rate, int)
        or not 8_000 <= sampling_rate <= 48_000
        or precision not in {"bfloat16", "float16", "float32"}
        or not isinstance(voice_profile_id, str)
        or not voice_profile_id
        or voice_profile.get("voice_profile_id") != voice_profile_id
        or voice_profile.get("sampling_rate") != sampling_rate
        or voice_profile.get("speech_contract_version") != "industrial-tts-speech-v1"
    ):
        raise ReleaseConflict("tts_runtime_voice_or_model_contract_is_invalid")
    result = dict(component)
    result["runtime"] = {
        "profile_id": TTS_RUNTIME_PROFILE,
        "image_repository": evidence.image_repository,
        "image_digest": evidence.image_digest,
        "inference_config": {
            "engine": "speecht5-tts",
            "model_revision": candidate.git_commit,
            "precision": precision,
            "sampling_rate": sampling_rate,
            "voice_profile_id": voice_profile_id,
            "speech_contract_version": "industrial-tts-speech-v1",
            "batch_size": 1,
            "max_input_characters": 6_000,
            "vocoder_path": "/opt/industrial-ops/models/speecht5_hifigan",
        },
        "hardware_profile": {
            "accelerator": "NVIDIA_GPU",
            "gpu_count": 1,
        },
    }
    result["supply_chain"] = _supply_chain_manifest(evidence)
    return result


def _attach_embedding_runtime(
    session: Session,
    tenant_id: str,
    component: dict[str, Any],
    *,
    evidence_id: str,
) -> dict[str, Any]:
    candidate = _tenant_record(
        session,
        TrainingExperimentRecord,
        tenant_id,
        TrainingExperimentRecord.experiment_id,
        str(component["training"]["experiment_id"]),
    )
    artifact = _tenant_record(
        session,
        TrainingArtifactRecord,
        tenant_id,
        TrainingArtifactRecord.artifact_id,
        str(component["artifact"]["artifact_id"]),
    )
    evidence = _tenant_record(
        session,
        SupplyChainEvidenceRecord,
        tenant_id,
        SupplyChainEvidenceRecord.evidence_id,
        evidence_id,
    )
    _require_artifact_supply_chain_binding(candidate, artifact, evidence)
    config = candidate.training_config
    max_sequence_length = config.get("max_sequence_length")
    precision = config.get("precision")
    batch_size = config.get("per_device_eval_batch_size")
    if (
        candidate.method != "EMBEDDING"
        or isinstance(max_sequence_length, bool)
        or not isinstance(max_sequence_length, int)
        or not 128 <= max_sequence_length <= 8192
        or precision not in {"bfloat16", "float16"}
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= 128
    ):
        raise ReleaseConflict("embedding_runtime_config_is_invalid")
    result = dict(component)
    result["runtime"] = {
        "profile_id": EMBEDDING_RUNTIME_PROFILE,
        "image_repository": evidence.image_repository,
        "image_digest": evidence.image_digest,
        "inference_config": {
            "engine": "sentence-transformers-embedding",
            "precision": precision,
            "batch_size": batch_size,
            "max_sequence_length": max_sequence_length,
            "normalize_embeddings": True,
        },
        "hardware_profile": {
            "accelerator": "NVIDIA_GPU",
            "gpu_count": 1,
        },
    }
    result["supply_chain"] = _supply_chain_manifest(evidence)
    return result


def _corpus_descriptor(
    session: Session,
    tenant_id: str,
    index: IndexReleaseRecord,
) -> dict[str, Any]:
    versions = list(
        session.scalars(
            select(KnowledgeDocumentVersionRecord)
            .join(
                KnowledgeChunkRecord,
                KnowledgeChunkRecord.document_version_id
                == KnowledgeDocumentVersionRecord.document_version_id,
            )
            .where(
                KnowledgeDocumentVersionRecord.tenant_id == tenant_id,
                KnowledgeChunkRecord.tenant_id == tenant_id,
                KnowledgeChunkRecord.release_id == index.release_id,
            )
            .distinct()
            .order_by(KnowledgeDocumentVersionRecord.document_version_id)
        )
    )
    documents = [
        {
            "document_version_id": item.document_version_id,
            "content_checksum": item.content_checksum,
            "parser_version": item.parser_version,
            "chunker_version": item.chunker_version,
        }
        for item in versions
    ]
    snapshot_hash = _digest_json(
        {"index_content_checksum": index.content_checksum, "documents": documents}
    )
    return {
        "snapshot_id": f"corpus-{snapshot_hash[:32]}",
        "snapshot_hash": snapshot_hash,
        "parser_versions": sorted({item.parser_version for item in versions}),
        "chunker_versions": sorted({item.chunker_version for item in versions}),
    }


def _require_supply_chain_binding(
    plan: ReleaseManifestPlan,
    candidate: TrainingExperimentRecord,
    adapter: TrainingArtifactRecord,
    evidence: SupplyChainEvidenceRecord,
) -> None:
    _require_artifact_supply_chain_binding(candidate, adapter, evidence)
    if (
        evidence.image_repository != plan.runtime_image_repository
        or evidence.image_digest != plan.runtime_image_digest
    ):
        raise ReleaseConflict("supply_chain_evidence_runtime_image_mismatch")


def _require_artifact_supply_chain_binding(
    candidate: TrainingExperimentRecord,
    artifact: TrainingArtifactRecord,
    evidence: SupplyChainEvidenceRecord,
) -> None:
    if (
        evidence.verification_status != "VERIFIED"
        or evidence.verification_hash != record_verification_hash(evidence)
    ):
        raise ReleaseConflict("supply_chain_evidence_is_not_verified")
    if evidence.model_artifact_digest != _artifact_digest(artifact.content_hash):
        raise ReleaseConflict("supply_chain_evidence_model_artifact_mismatch")
    if evidence.source_revision != candidate.git_commit:
        raise ReleaseConflict("supply_chain_evidence_source_revision_mismatch")


def _supply_chain_evidence_is_current(
    session: Session,
    tenant_id: str,
    manifest_supply_chain: Any,
    candidate: TrainingExperimentRecord,
    adapter: dict[str, Any],
    runtime: dict[str, Any],
) -> bool:
    if not isinstance(manifest_supply_chain, dict):
        return False
    evidence_id = manifest_supply_chain.get("evidence_id")
    if not isinstance(evidence_id, str):
        return False
    evidence = session.scalar(
        select(SupplyChainEvidenceRecord).where(
            SupplyChainEvidenceRecord.tenant_id == tenant_id,
            SupplyChainEvidenceRecord.evidence_id == evidence_id,
        )
    )
    if evidence is None:
        return False
    try:
        current_manifest = _supply_chain_manifest(evidence)
        content_hash = str(adapter["content_hash"])
        image_repository = str(runtime["image_repository"])
        image_digest = str(runtime["image_digest"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        evidence.verification_status == "VERIFIED"
        and evidence.verification_hash == record_verification_hash(evidence)
        and _digest_json(current_manifest) == _digest_json(manifest_supply_chain)
        and evidence.image_repository == image_repository
        and evidence.image_digest == image_digest
        and evidence.model_artifact_digest == _artifact_digest(content_hash)
        and evidence.source_revision == candidate.git_commit
        and evidence.vulnerability_scan_status == "PASSED"
        and evidence.license_status == "APPROVED"
    )


def _specialized_component_supply_chain_is_current(
    session: Session,
    tenant_id: str,
    specialized: Any,
) -> bool:
    if not isinstance(specialized, dict):
        return False
    for name in ("timeseries", "rul", "tts", "embedding", "reranker", "vlm", "asr"):
        if name not in specialized:
            continue
        component = specialized[name]
        if not isinstance(component, dict):
            return False
        if (
            name in {"embedding", "reranker"}
            and "runtime" not in component
            and "supply_chain" not in component
        ):
            continue
        try:
            if (
                name in {"vlm", "asr"}
                and _attach_media_supply_chain(
                    session,
                    tenant_id,
                    component,
                    evidence_id=component.get("supply_chain", {}).get("evidence_id"),
                )
                != component
            ):
                return False
            training = component["training"]
            artifact = component["artifact"]
            runtime = component["runtime"]
            if not all(isinstance(value, dict) for value in (training, artifact, runtime)):
                return False
            if not all(
                isinstance(value, str) and value
                for value in (
                    training.get("experiment_id"),
                    artifact.get("content_hash"),
                    runtime.get("image_repository"),
                    runtime.get("image_digest"),
                )
            ):
                return False
            candidate = _tenant_record(
                session,
                TrainingExperimentRecord,
                tenant_id,
                TrainingExperimentRecord.experiment_id,
                training["experiment_id"],
            )
            if not _supply_chain_evidence_is_current(
                session,
                tenant_id,
                component["supply_chain"],
                candidate,
                artifact,
                runtime,
            ):
                return False
        except (KeyError, TypeError, ValueError, ReleaseConflict, ReleaseNotVisible):
            return False
    return True


def _require_staging_publication(session: Session, release: ModelReleaseRecord) -> None:
    try:
        require_publishable_staging_admission(session, release)
    except ValueError as exc:
        raise ReleaseConflict(str(exc), release.version) from exc


def _require_media_supply_chain_current(session: Session, release: ModelReleaseRecord) -> None:
    try:
        status = media_release_supply_chain_status(session, release)
    except ValueError as exc:
        raise ReleaseConflict(str(exc)) from exc
    if status != "VERIFIED":
        raise ReleaseConflict("legacy_supply_chain_evidence_required")


def _artifact_digest(content_hash: str) -> str:
    value = content_hash.removeprefix("sha256:")
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ReleaseConflict("adapter_content_hash_is_not_sha256")
    return f"sha256:{value}"


def _release_prompt_bundle_binding(
    session: Session,
    tenant_id: str,
    prompt_bundle_id: str,
    *,
    evaluation_only: bool = False,
) -> dict[str, str]:
    try:
        definition = default_prompt_registry().get(prompt_bundle_id)
    except PromptBundleNotDeployed as exc:
        raise ReleaseConflict("prompt_bundle_not_deployed") from exc
    record = session.scalar(
        select(PromptBundleRecord).where(
            PromptBundleRecord.tenant_id == tenant_id,
            PromptBundleRecord.prompt_bundle_id == prompt_bundle_id,
        )
    )
    if record is None:
        if evaluation_only or prompt_bundle_id != DEFAULT_PROMPT_BUNDLE_ID:
            raise ReleaseConflict("prompt_bundle_governance_record_required")
        governance_status = "LEGACY_BUILTIN"
    else:
        allowed = (
            {"DRAFT", "REJECTED", "REVIEW_PENDING", "APPROVED"} if evaluation_only else {"APPROVED"}
        )
        if record.status not in allowed:
            raise ReleaseConflict("prompt_bundle_governance_approval_required")
        if record.content_hash != definition.content_hash:
            raise ReleaseConflict("prompt_bundle_runtime_hash_mismatch")
        governance_status = record.status
    return {
        "id": definition.prompt_bundle_id,
        "content_hash": definition.content_hash,
        "governance_status": governance_status,
    }


def _prompt_bundle_binding_is_current(
    session: Session,
    tenant_id: str,
    manifest_binding: Any,
) -> bool:
    if not isinstance(manifest_binding, dict):
        return False
    prompt_bundle_id = manifest_binding.get("id")
    if not isinstance(prompt_bundle_id, str):
        return False
    try:
        current = _release_prompt_bundle_binding(session, tenant_id, prompt_bundle_id)
    except ReleaseConflict:
        return False
    return current == manifest_binding


def _evaluate_release_gates(
    session: Session,
    tenant_id: str,
    release: ModelReleaseRecord,
) -> dict[str, bool]:
    evaluation = _tenant_record(
        session,
        ModelEvaluationRunRecord,
        tenant_id,
        ModelEvaluationRunRecord.evaluation_id,
        release.evaluation_id,
    )
    candidate = _tenant_record(
        session,
        TrainingExperimentRecord,
        tenant_id,
        TrainingExperimentRecord.experiment_id,
        release.candidate_experiment_id,
    )
    manifest = release.manifest_json
    evaluation_artifacts = manifest["evaluation"]["artifacts"]
    job = manifest["evaluation"]["runner"]
    adapter = manifest["adapter"]
    quantization = manifest.get("quantization")
    release_artifact = quantization if isinstance(quantization, dict) else adapter
    retrieval = manifest["retrieval"]
    supply_chain = manifest["supply_chain"]
    suite = manifest["evaluation"]["suite"]
    policy = manifest["evaluation"]["policy"]
    dataset = manifest["dataset_snapshot"]
    specialized = manifest.get("specialized_components")
    specialized_set_complete = isinstance(specialized, dict) and _valid_specialized_component_set(
        set(specialized)
    )
    current_specialized: dict[str, dict[str, Any]] | None = None
    if specialized_set_complete and isinstance(specialized, dict):
        try:
            components = set(specialized)
            current_specialized = _build_specialized_components(
                session,
                tenant_id,
                {
                    component: str(specialized[component]["evaluation"]["evaluation_id"])
                    for component in components
                },
                {
                    component: str(specialized[component]["runtime_model_id"])
                    for component in components
                },
                target_environment=cast(ReleaseEnvironment, release.target_environment),
            )
            for name in ("vlm", "asr"):
                current_specialized[name] = _attach_media_supply_chain(
                    session,
                    tenant_id,
                    current_specialized[name],
                    evidence_id=specialized[name].get("supply_chain", {}).get("evidence_id"),
                )
            reranker_manifest = specialized.get("reranker")
            if isinstance(reranker_manifest, dict) and isinstance(
                reranker_manifest.get("runtime"), dict
            ):
                current_specialized["reranker"] = _attach_reranker_runtime(
                    session,
                    tenant_id,
                    current_specialized["reranker"],
                    evidence_id=str(reranker_manifest["supply_chain"]["evidence_id"]),
                )
            embedding_manifest = specialized.get("embedding")
            if isinstance(embedding_manifest, dict) and isinstance(
                embedding_manifest.get("runtime"), dict
            ):
                current_specialized["embedding"] = _attach_embedding_runtime(
                    session,
                    tenant_id,
                    current_specialized["embedding"],
                    evidence_id=str(embedding_manifest["supply_chain"]["evidence_id"]),
                )
            tts_manifest = specialized.get("tts")
            if isinstance(tts_manifest, dict) and isinstance(tts_manifest.get("runtime"), dict):
                current_specialized["tts"] = _attach_tts_runtime(
                    session,
                    tenant_id,
                    current_specialized["tts"],
                    evidence_id=str(tts_manifest["supply_chain"]["evidence_id"]),
                )
            if "timeseries" in components:
                current_specialized["timeseries"] = _attach_timeseries_runtime(
                    session,
                    tenant_id,
                    current_specialized["timeseries"],
                    evidence_id=str(specialized["timeseries"]["supply_chain"]["evidence_id"]),
                )
            if "rul" in components:
                current_specialized["rul"] = _attach_rul_runtime(
                    session,
                    tenant_id,
                    current_specialized["rul"],
                    evidence_id=str(specialized["rul"]["supply_chain"]["evidence_id"]),
                )
        except (KeyError, TypeError, ValueError, ReleaseConflict, ReleaseNotVisible) as exc:
            structlog.get_logger(__name__).warning(
                "release_gate_evidence_unreadable",
                release_id=release.release_id,
                stage="specialized_components",
                error_type=type(exc).__name__,
            )
            current_specialized = None
    rollback_valid = True
    if release.rollback_release_id is not None:
        rollback = _release_or_hidden(session, tenant_id, release.rollback_release_id)
        rollback_valid = (
            rollback.status in {"PRODUCTION", "RETIRED"}
            and manifest["rollback"]["manifest_hash"] == rollback.manifest_hash
        )
    suite_record = _tenant_record(
        session,
        EvaluationSuiteRecord,
        tenant_id,
        EvaluationSuiteRecord.suite_id,
        evaluation.suite_id,
    )
    current_project_import = _project_authorized_import_binding(
        session,
        tenant_id,
        component="LLM",
        candidate=candidate,
        evaluation=evaluation,
        suite=suite_record,
        target_environment=cast(ReleaseEnvironment, release.target_environment),
    )
    project_import_current = _project_import_binding_is_current(
        current_project_import,
        manifest.get("project_authorized_import"),
    )
    gates = {
        "adapter_bundle_complete": _release_artifact_is_complete(
            size_bytes=adapter["size_bytes"],
            content_hash=adapter["content_hash"],
            metadata=adapter["metadata"],
            project_authorized_import=current_project_import,
        ),
        "candidate_training_complete": candidate.status == "COMPLETED",
        "quantization_artifact_complete": (
            quantization is None
            or (
                isinstance(quantization, dict)
                and quantization.get("metadata", {}).get("format") == "tar.gz"
                and (
                    (
                        manifest.get("runtime", {}).get("inference_config", {}).get("engine")
                        == "vllm"
                        and quantization.get("metadata", {}).get("serialization")
                        == "compressed-tensors"
                    )
                    or (
                        manifest.get("runtime", {}).get("inference_config", {}).get("engine")
                        == "llama.cpp"
                        and quantization.get("metadata", {}).get("serialization") == "gguf"
                        and quantization.get("metadata", {})
                        .get("runtime", {})
                        .get("target_runtime")
                        == "llama.cpp"
                        and _valid_artifact_filename(
                            quantization.get("metadata", {}).get("runtime", {}).get("model_file")
                        )
                    )
                )
                and quantization.get("metadata", {}).get("approval_inheritance") is False
                and quantization.get("metadata", {}).get("quantization_profile_id")
                == manifest.get("quantization_profile_id")
                and quantization.get("size_bytes", 0) > 0
                and bool(quantization.get("content_hash"))
            )
        ),
        "dataset_snapshot_governed": (
            dataset["id"] == candidate.dataset_snapshot_id
            and dataset["manifest_hash"] == candidate.dataset_manifest_hash
            and dataset["lineage_status"] == "CONFIRMED"
            and _snapshot_is_training_eligible(session, tenant_id, dataset["id"])
        ),
        "evaluation_artifacts_complete": any(
            artifact["kind"] == "case_evidence" and artifact["size_bytes"] > 0
            for artifact in evaluation_artifacts
        ),
        "evaluation_candidate_decision": (
            evaluation.status == "COMPLETED" and evaluation.decision == "CANDIDATE"
        ),
        "evaluation_hard_gates": (
            bool(evaluation.hard_gate_results) and all(evaluation.hard_gate_results.values())
        ),
        "evaluation_gold_suite": (
            _evaluation_suite_is_release_governed(
                suite_record,
                candidate,
                minimum_samples=500,
                project_authorized_import=current_project_import,
            )
            and suite["id"] == suite_record.suite_id
            and suite["tier"] == suite_record.tier
            and suite["sample_count"] == suite_record.sample_count
        ),
        "evaluation_policy_active": policy["status"] == "ACTIVE",
        "evaluation_report_immutable": evaluation.report_hash
        == manifest["evaluation"]["report_hash"],
        "evaluation_runner_complete": (
            job["status"] == "COMPLETED"
            and job["result_evaluation_id"] == evaluation.evaluation_id
            and job["target_profile"] == "AGENT_RUNTIME"
            and bool(job["config_hash"])
        ),
        "prompt_bundle_governed": _prompt_bundle_binding_is_current(
            session,
            tenant_id,
            manifest.get("prompt_bundle"),
        ),
        "specialized_component_set_complete": specialized_set_complete,
        "specialized_component_evaluations_approved": current_specialized is not None,
        "specialized_component_bindings_immutable": (
            current_specialized is not None
            and _digest_json(current_specialized) == _digest_json(specialized)
        ),
        "specialized_component_supply_chain_approved": (
            _specialized_component_supply_chain_is_current(session, tenant_id, specialized)
        ),
        "knowledge_index_published": _index_is_published(
            session, tenant_id, retrieval["index_release_id"], retrieval["index_content_checksum"]
        ),
        "license_approved": candidate.license_status == "APPROVED",
        "manifest_integrity": _digest_json(manifest) == release.manifest_hash,
        "peft_release_method": candidate.method in PRODUCTION_RELEASE_METHODS,
        "project_authorized_import_current": project_import_current,
        "rollback_target_valid": rollback_valid,
        "supply_chain_approved": _supply_chain_evidence_is_current(
            session,
            tenant_id,
            supply_chain,
            candidate,
            release_artifact,
            manifest["runtime"],
        ),
    }

    if (
        "evaluation_admission" in manifest
        or manifest.get("prompt_evaluation_only")
        or manifest.get("local_staging_adapter")
    ):
        try:
            require_staging_smoke_admission_current(session, release)
            admission_current = True
        except (ValueError, KeyError, TypeError) as exc:
            structlog.get_logger(__name__).warning(
                "staging_release_admission_invalid",
                release_id=release.release_id,
                error_type=type(exc).__name__,
            )
            admission_current = False
        # These are different qualifications, not an assertion of GOLD/agent success.
        for name in (
            "evaluation_candidate_decision",
            "evaluation_gold_suite",
            "evaluation_runner_complete",
        ):
            gates.pop(name)
        gates["staging_smoke_admission_current"] = admission_current
        gates["evaluation_only_not_publishable"] = manifest.get("prompt_evaluation_only") is False
        if manifest.get("local_staging_adapter"):
            gates.pop("supply_chain_approved")
            gates["local_adapter_provenance_current"] = admission_current
    return gates


def _snapshot_is_training_eligible(
    session: Session,
    tenant_id: str,
    snapshot_id: str,
) -> bool:
    snapshot = session.scalar(
        select(DatasetSnapshotRecord).where(
            DatasetSnapshotRecord.tenant_id == tenant_id,
            DatasetSnapshotRecord.snapshot_id == snapshot_id,
        )
    )
    return bool(
        snapshot is not None
        and snapshot.status == "CANDIDATE"
        and snapshot.training_eligible
        and snapshot.lineage_status == "CONFIRMED"
        and snapshot.manifest_hash
    )


def _index_is_published(
    session: Session,
    tenant_id: str,
    release_id: str,
    expected_checksum: str,
) -> bool:
    index = session.scalar(
        select(IndexReleaseRecord).where(
            IndexReleaseRecord.tenant_id == tenant_id,
            IndexReleaseRecord.release_id == release_id,
        )
    )
    return bool(
        index is not None
        and index.status in {"PUBLISHED", "ACTIVE"}
        and index.content_checksum == expected_checksum
    )


def _validate_inference_config(config: dict[str, Any]) -> None:
    if config.get("engine") == "llama.cpp":
        _validate_llama_cpp_inference_config(config)
        return
    required = {
        "engine",
        "tensor_parallel_size",
        "max_model_len",
        "dtype",
        "max_num_seqs",
        "enable_lora",
        "max_lora_rank",
    }
    if set(config) != required:
        raise ValueError("inference_config fields are incomplete")
    if config["engine"] != "vllm":
        raise ValueError("inference_config.engine must be vllm")
    if config["dtype"] not in {"bfloat16", "float16"}:
        raise ValueError("inference_config.dtype is invalid")
    for key, minimum, maximum in (
        ("tensor_parallel_size", 1, 64),
        ("max_model_len", 1024, 1_048_576),
        ("max_num_seqs", 1, 65_536),
        ("max_lora_rank", 0, 1024),
    ):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"inference_config.{key} is invalid")
    if not isinstance(config["enable_lora"], bool):
        raise ValueError("inference_config.enable_lora must be a boolean")
    if (config["enable_lora"] and config["max_lora_rank"] < 1) or (
        not config["enable_lora"] and config["max_lora_rank"] != 0
    ):
        raise ValueError("inference_config.max_lora_rank conflicts with enable_lora")


def _validate_hardware_profile(profile: dict[str, Any]) -> None:
    if profile.get("accelerator") == "CPU":
        _validate_cpu_hardware_profile(profile)
        return
    required = {"accelerator", "gpu_count", "gpu_model", "gpu_memory_gb"}
    if set(profile) != required or profile["accelerator"] != "NVIDIA_GPU":
        raise ValueError("hardware_profile fields are invalid")
    if not isinstance(profile["gpu_model"], str) or not profile["gpu_model"].strip():
        raise ValueError("hardware_profile.gpu_model is invalid")
    for key in ("gpu_count", "gpu_memory_gb"):
        value = profile[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"hardware_profile.{key} is invalid")


def _validate_llama_cpp_inference_config(config: dict[str, Any]) -> None:
    required = {"engine", "context_size", "threads", "batch_size", "mmap", "mlock"}
    if set(config) != required:
        raise ValueError("llama.cpp inference_config fields are incomplete")
    for key, minimum, maximum in (
        ("context_size", 1024, 131_072),
        ("threads", 1, 256),
        ("batch_size", 1, 4096),
    ):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"inference_config.{key} is invalid")
    if not isinstance(config["mmap"], bool) or not isinstance(config["mlock"], bool):
        raise ValueError("llama.cpp memory settings must be booleans")


def _validate_cpu_hardware_profile(profile: dict[str, Any]) -> None:
    required = {
        "accelerator",
        "cpu_architecture",
        "cpu_model",
        "cpu_cores",
        "memory_gb",
        "instruction_set",
    }
    if set(profile) != required:
        raise ValueError("CPU hardware_profile fields are invalid")
    if profile["cpu_architecture"] not in {"x86_64", "aarch64"}:
        raise ValueError("hardware_profile.cpu_architecture is invalid")
    for key in ("cpu_model", "instruction_set"):
        value = profile[key]
        if not isinstance(value, str) or not value.strip() or len(value) > 128:
            raise ValueError(f"hardware_profile.{key} is invalid")
    for key, maximum in (("cpu_cores", 512), ("memory_gb", 4096)):
        value = profile[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f"hardware_profile.{key} is invalid")


def _valid_artifact_filename(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and value.endswith(".gguf")
        and len(value) <= 255
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
    )


def _validate_sha256_digest(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field} must be a sha256 digest")


def _validate_image_repository(value: str) -> None:
    pattern = re.compile(
        r"^[a-z0-9](?:[a-z0-9.-]*)(?::[0-9]{1,5})?"
        r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+$"
    )
    if len(value) > 512 or pattern.fullmatch(value) is None:
        raise ValueError("runtime_image_repository must be an OCI repository without tag")


def _require_manifest_integrity(release: ModelReleaseRecord) -> None:
    if _digest_json(release.manifest_json) != release.manifest_hash:
        raise ReleaseConflict("release_manifest_integrity_failed", release.version)


def _require_version(release: ModelReleaseRecord, expected_version: int) -> None:
    if release.version != expected_version:
        raise ReleaseConflict("version_mismatch", release.version)


def _release_or_hidden(
    session: Session, tenant_id: str, release_id: str, *, lock: bool = False
) -> ModelReleaseRecord:
    query = select(ModelReleaseRecord).where(
        ModelReleaseRecord.tenant_id == tenant_id,
        ModelReleaseRecord.release_id == release_id,
    )
    release = session.scalar(query.with_for_update() if lock else query)
    if release is None:
        raise ReleaseNotVisible
    return release


def _tenant_record(
    session: Session,
    model: type[Any],
    tenant_id: str,
    identifier_column: Any,
    identifier: str,
) -> Any:
    record = session.scalar(
        select(model).where(model.tenant_id == tenant_id, identifier_column == identifier)
    )
    if record is None:
        raise ReleaseNotVisible
    return record


def _approval_for_release(
    session: Session,
    tenant_id: str,
    release_id: str,
) -> ModelReleaseApprovalRecord | None:
    return session.scalar(
        select(ModelReleaseApprovalRecord).where(
            ModelReleaseApprovalRecord.tenant_id == tenant_id,
            ModelReleaseApprovalRecord.release_id == release_id,
        )
    )


def _aggregate(
    session: Session,
    tenant_id: str,
    release: ModelReleaseRecord,
) -> ReleaseAggregate:
    approval = _approval_for_release(session, tenant_id, release.release_id)
    transitions = tuple(
        session.scalars(
            select(ModelReleaseTransitionRecord)
            .where(
                ModelReleaseTransitionRecord.tenant_id == tenant_id,
                ModelReleaseTransitionRecord.release_id == release.release_id,
            )
            .order_by(ModelReleaseTransitionRecord.sequence)
        )
    )
    return ReleaseAggregate(release=release, approval=approval, transitions=transitions)


def _next_sequence(session: Session, tenant_id: str, release_id: str) -> int:
    current = session.scalar(
        select(func.max(ModelReleaseTransitionRecord.sequence)).where(
            ModelReleaseTransitionRecord.tenant_id == tenant_id,
            ModelReleaseTransitionRecord.release_id == release_id,
        )
    )
    return int(current or 0) + 1


def _append_transition(
    session: Session,
    release: ModelReleaseRecord,
    *,
    sequence: int,
    from_status: str,
    to_status: str,
    actor_subject_id: str,
    reason_code: str,
    evidence: dict[str, Any],
    occurred_at: datetime,
) -> None:
    session.add(
        ModelReleaseTransitionRecord(
            transition_id=f"release-transition-{uuid4().hex}",
            tenant_id=release.tenant_id,
            release_id=release.release_id,
            sequence=sequence,
            from_status=from_status,
            to_status=to_status,
            actor_subject_id=actor_subject_id,
            reason_code=reason_code,
            evidence_json=evidence,
            evidence_hash=_digest_json(evidence),
            occurred_at=occurred_at,
            created_at=occurred_at,
            updated_at=occurred_at,
        )
    )
    session.flush()
