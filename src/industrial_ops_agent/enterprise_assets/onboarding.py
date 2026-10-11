"""Governed adoption of project-authorized candidates into ModelRelease."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.deployment.service import (
    DeploymentConflict,
    DeploymentNotVisible,
    DeploymentPlan,
    ModelDeploymentService,
    validate_deployment_plan,
)
from industrial_ops_agent.enterprise_assets.importer import (
    enterprise_model_asset_import_projection,
)
from industrial_ops_agent.enterprise_assets.models import (
    EnterpriseModelComponent,
    EnterpriseReleaseAutomationStatus,
    EnterpriseReleaseDraftBaseline,
    EnterpriseReleaseDraftPreview,
    EnterpriseReleaseOnboarding,
    EnterpriseRuntimeEvidence,
)
from industrial_ops_agent.enterprise_assets.service import EnterpriseProjectAdoptionReader
from industrial_ops_agent.model_methods import PRODUCTION_RELEASE_METHODS
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.model_import_integrity import (
    enterprise_model_import_hash,
)
from industrial_ops_agent.persistence.models import (
    EnterpriseModelAssetImportRecord,
    EnterpriseModelReleaseOnboardingRecord,
    ModelDeploymentRecord,
    ModelEvaluationRunRecord,
    ModelReleaseApprovalRecord,
    ModelReleaseRecord,
    SupplyChainEvidenceRecord,
    TrainingArtifactRecord,
    TrainingExperimentRecord,
)
from industrial_ops_agent.releases.service import ModelReleaseService, ReleaseManifestPlan
from industrial_ops_agent.supply_chain.service import (
    matching_component_evidence,
    record_verification_hash,
)

_BASELINE_STATUSES = frozenset(
    {"APPROVAL_PENDING", "SHADOW", "CANARY", "PRODUCTION", "ROLLED_BACK"}
)
_COMPONENT_METHOD = {
    "VLM": "VLM",
    "ASR": "ASR",
    "TTS": "TTS",
    "RUL": "RUL_TRANSFORMER",
    "EMBEDDING": "EMBEDDING",
    "RERANKER": "RERANKER",
}
_TERMINAL_RELEASE_STATUSES = frozenset({"REJECTED", "ROLLED_BACK"})


class EnterpriseReleaseOnboardingConflict(Exception):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class AutoShadowDispatchReport:
    scanned: int
    waiting_approval: int
    requested: int
    already_requested: int
    cancelled: int
    blocked: int


@dataclass(frozen=True, slots=True)
class _ResolvedDraft:
    model_import: EnterpriseModelAssetImportRecord | None
    candidate: TrainingExperimentRecord | None
    evaluation: ModelEvaluationRunRecord | None
    supply_chain: SupplyChainEvidenceRecord | None
    baselines: tuple[ModelReleaseRecord, ...]
    blockers: tuple[str, ...]


class EnterpriseReleaseOnboardingService:
    """Clone a governed baseline and replace only one authoritative candidate."""

    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        adoption_reader: EnterpriseProjectAdoptionReader,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._adoption_reader = adoption_reader

    def preview(
        self,
        identity: IdentityContext,
        component: EnterpriseModelComponent,
        *,
        request_id: str,
    ) -> EnterpriseReleaseDraftPreview:
        self._require(identity, Action.READ_MODEL_RELEASE, "model-releases", request_id)
        evidence = _evidence_for_component(self._adoption_reader, component)
        with self._database.transaction(identity.tenant_context) as session:
            resolved = _resolve_draft(session, identity.tenant_id, evidence)
            baselines = tuple(_baseline_projection(item) for item in resolved.baselines)
            evaluation_id = (
                resolved.evaluation.evaluation_id if resolved.evaluation is not None else None
            )
            supply_chain_id = (
                resolved.supply_chain.evidence_id if resolved.supply_chain is not None else None
            )
        return EnterpriseReleaseDraftPreview(
            evidence=evidence,
            import_state=("IMPORTED" if resolved.model_import is not None else "NOT_IMPORTED"),
            model_import=(
                enterprise_model_asset_import_projection(resolved.model_import)
                if resolved.model_import is not None
                else None
            ),
            eligible=not resolved.blockers,
            blockers=resolved.blockers,
            evaluation_id=evaluation_id,
            suggested_supply_chain_evidence_id=supply_chain_id,
            baselines=baselines,
        )

    def create_draft(
        self,
        identity: IdentityContext,
        component: EnterpriseModelComponent,
        *,
        baseline_release_id: str,
        auto_shadow_enabled: bool,
        deployment_plan: DeploymentPlan | None,
        active_mcp_server_versions: dict[str, str],
        idempotency_key: str,
        request_id: str,
    ) -> EnterpriseReleaseOnboarding:
        self._require(identity, Action.CREATE_MODEL_RELEASE, "model-releases", request_id)
        if auto_shadow_enabled and deployment_plan is None:
            raise ValueError("deployment_plan is required when auto_shadow_enabled is true")
        if deployment_plan is not None:
            validate_deployment_plan(deployment_plan)
        deployment_document = asdict(deployment_plan) if deployment_plan is not None else {}
        existing = self._existing_replay(
            identity,
            component=component,
            baseline_release_id=baseline_release_id,
            auto_shadow_enabled=auto_shadow_enabled,
            deployment_document=deployment_document,
            active_mcp_server_versions=active_mcp_server_versions,
            idempotency_key=idempotency_key,
        )
        if existing is not None:
            return existing
        evidence = _evidence_for_component(self._adoption_reader, component)
        with self._database.transaction(identity.tenant_context) as session:
            resolved = _resolve_draft(session, identity.tenant_id, evidence)
            if resolved.blockers:
                raise EnterpriseReleaseOnboardingConflict(resolved.blockers[0])
            baseline = next(
                (item for item in resolved.baselines if item.release_id == baseline_release_id),
                None,
            )
            if baseline is None:
                raise EnterpriseReleaseOnboardingConflict("COMPATIBLE_BASELINE_RELEASE_REQUIRED")
            candidate = resolved.candidate
            evaluation = resolved.evaluation
            if candidate is None or evaluation is None:
                raise EnterpriseReleaseOnboardingConflict("CANDIDATE_EVALUATION_NOT_IMPORTED")
            evaluation_id = evaluation.evaluation_id
            plan = _plan_from_baseline(
                baseline.manifest_json,
                evidence=evidence,
                candidate=candidate,
                evaluation=evaluation,
                supply_chain=resolved.supply_chain,
                active_mcp_server_versions=active_mcp_server_versions,
            )

        request_document = {
            "component": component,
            "source_candidate_experiment_id": evidence.candidate_experiment_id,
            "candidate_experiment_id": candidate.experiment_id,
            "evidence_chain_sha256": evidence.evidence_chain_sha256,
            "baseline_release_id": baseline_release_id,
            "release_plan": asdict(plan),
            "auto_shadow_enabled": auto_shadow_enabled,
            "deployment_plan": deployment_document,
        }
        request_hash = _digest_json(request_document)
        existing = self._existing_idempotent(
            identity,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
        )
        if existing is not None:
            return existing

        release = ModelReleaseService(self._database, self._authorizer).create(
            identity,
            plan,
            idempotency_key=(
                "enterprise-onboarding:" + sha256(idempotency_key.encode()).hexdigest()
            ),
            request_id=request_id,
        )
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            record = session.scalar(
                select(EnterpriseModelReleaseOnboardingRecord).where(
                    EnterpriseModelReleaseOnboardingRecord.tenant_id == identity.tenant_id,
                    EnterpriseModelReleaseOnboardingRecord.release_id == release.release.release_id,
                )
            )
            if record is None:
                record = EnterpriseModelReleaseOnboardingRecord(
                    onboarding_id=f"enterprise-release-onboarding-{uuid4().hex}",
                    tenant_id=identity.tenant_id,
                    release_id=release.release.release_id,
                    baseline_release_id=baseline_release_id,
                    component=component,
                    candidate_experiment_id=candidate.experiment_id,
                    evaluation_id=evaluation_id,
                    evidence_chain_sha256=evidence.evidence_chain_sha256,
                    idempotency_key=idempotency_key,
                    request_hash=request_hash,
                    auto_shadow_enabled=auto_shadow_enabled,
                    deployment_plan_json=deployment_document,
                    deployment_plan_hash=_digest_json(deployment_document),
                    automation_status=("WAITING_APPROVAL" if auto_shadow_enabled else "DISABLED"),
                    attempt_count=0,
                    last_error=None,
                    last_attempt_at=None,
                    requested_by_subject_id=identity.subject_id,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
                session.add(record)
                session.flush()
            return _onboarding_projection(
                record,
                release.release.status,
                release.release.version,
            )

    def _existing_replay(
        self,
        identity: IdentityContext,
        *,
        component: EnterpriseModelComponent,
        baseline_release_id: str,
        auto_shadow_enabled: bool,
        deployment_document: dict[str, Any],
        active_mcp_server_versions: dict[str, str],
        idempotency_key: str,
    ) -> EnterpriseReleaseOnboarding | None:
        """Replay an immutable request before candidate-registration preflight."""

        with self._database.transaction(identity.tenant_context) as session:
            record = session.scalar(
                select(EnterpriseModelReleaseOnboardingRecord).where(
                    EnterpriseModelReleaseOnboardingRecord.tenant_id == identity.tenant_id,
                    EnterpriseModelReleaseOnboardingRecord.requested_by_subject_id
                    == identity.subject_id,
                    EnterpriseModelReleaseOnboardingRecord.idempotency_key == idempotency_key,
                )
            )
            if record is None:
                return None
            release = _release(session, identity.tenant_id, record.release_id)
            manifest_mcp_versions = release.manifest_json.get("mcp_server_versions")
            if (
                record.component != component
                or record.baseline_release_id != baseline_release_id
                or record.auto_shadow_enabled != auto_shadow_enabled
                or record.deployment_plan_json != deployment_document
                or manifest_mcp_versions != active_mcp_server_versions
            ):
                raise EnterpriseReleaseOnboardingConflict(
                    "idempotency_key_reused",
                    record.version,
                )
            return _onboarding_projection(record, release.status, release.version)

    def _existing_idempotent(
        self,
        identity: IdentityContext,
        *,
        idempotency_key: str,
        request_hash: str,
    ) -> EnterpriseReleaseOnboarding | None:
        with self._database.transaction(identity.tenant_context) as session:
            record = session.scalar(
                select(EnterpriseModelReleaseOnboardingRecord).where(
                    EnterpriseModelReleaseOnboardingRecord.tenant_id == identity.tenant_id,
                    EnterpriseModelReleaseOnboardingRecord.requested_by_subject_id
                    == identity.subject_id,
                    EnterpriseModelReleaseOnboardingRecord.idempotency_key == idempotency_key,
                )
            )
            if record is None:
                return None
            if record.request_hash != request_hash:
                raise EnterpriseReleaseOnboardingConflict(
                    "idempotency_key_reused",
                    record.version,
                )
            release = _release(session, identity.tenant_id, record.release_id)
            return _onboarding_projection(record, release.status, release.version)

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


class EnterpriseAutoShadowDispatcher:
    """Turn approved intent into a real deployment request exactly once."""

    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def run_once(self, identity: IdentityContext) -> AutoShadowDispatchReport:
        with self._database.transaction(identity.tenant_context) as session:
            records = list(
                session.scalars(
                    select(EnterpriseModelReleaseOnboardingRecord)
                    .where(
                        EnterpriseModelReleaseOnboardingRecord.tenant_id == identity.tenant_id,
                        EnterpriseModelReleaseOnboardingRecord.auto_shadow_enabled.is_(True),
                        EnterpriseModelReleaseOnboardingRecord.automation_status
                        == "WAITING_APPROVAL",
                    )
                    .order_by(EnterpriseModelReleaseOnboardingRecord.created_at)
                )
            )
            pending = tuple(
                (
                    item.onboarding_id,
                    item.release_id,
                    dict(item.deployment_plan_json),
                    item.deployment_plan_hash,
                    item.attempt_count,
                )
                for item in records
            )

        waiting = requested = existing = cancelled = blocked = 0
        for onboarding_id, release_id, plan_json, plan_hash, attempt_count in pending:
            state = self._release_state(identity, release_id)
            if state is None:
                self._mark(
                    identity,
                    onboarding_id,
                    status="BLOCKED",
                    error="release_not_found_or_not_visible",
                    attempted=True,
                )
                blocked += 1
                continue
            release, approval, deployment_exists = state
            if deployment_exists:
                self._mark(
                    identity,
                    onboarding_id,
                    status="SHADOW_REQUESTED",
                    error=None,
                    attempted=False,
                )
                existing += 1
                continue
            if release.status in _TERMINAL_RELEASE_STATUSES:
                self._mark(
                    identity,
                    onboarding_id,
                    status="CANCELLED",
                    error=f"release_{release.status.lower()}",
                    attempted=False,
                )
                cancelled += 1
                continue
            if (
                release.status != "APPROVAL_PENDING"
                or approval is None
                or approval.status != "APPROVED"
            ):
                waiting += 1
                continue
            if _digest_json(plan_json) != plan_hash:
                self._mark(
                    identity,
                    onboarding_id,
                    status="BLOCKED",
                    error="deployment_plan_integrity_mismatch",
                    attempted=True,
                )
                blocked += 1
                continue
            try:
                plan = DeploymentPlan(**plan_json)
                validate_deployment_plan(plan)
                ModelDeploymentService(self._database, self._authorizer).request_shadow(
                    identity,
                    release_id,
                    plan,
                    expected_release_version=release.version,
                    request_id=f"enterprise-auto-shadow:{onboarding_id}:{uuid4().hex}",
                )
            except DeploymentConflict as exc:
                if exc.reason == "model_deployment_already_requested":
                    self._mark(
                        identity,
                        onboarding_id,
                        status="SHADOW_REQUESTED",
                        error=None,
                        attempted=True,
                    )
                    existing += 1
                    continue
                next_status = "BLOCKED" if attempt_count >= 4 else "WAITING_APPROVAL"
                self._mark(
                    identity,
                    onboarding_id,
                    status=next_status,
                    error=exc.reason,
                    attempted=True,
                )
                if next_status == "BLOCKED":
                    blocked += 1
                else:
                    waiting += 1
                continue
            except (AuthorizationDenied, DeploymentNotVisible, TypeError, ValueError) as exc:
                self._mark(
                    identity,
                    onboarding_id,
                    status="BLOCKED",
                    error=_safe_error(exc),
                    attempted=True,
                )
                blocked += 1
                continue
            self._mark(
                identity,
                onboarding_id,
                status="SHADOW_REQUESTED",
                error=None,
                attempted=True,
            )
            requested += 1

        return AutoShadowDispatchReport(
            scanned=len(pending),
            waiting_approval=waiting,
            requested=requested,
            already_requested=existing,
            cancelled=cancelled,
            blocked=blocked,
        )

    def _release_state(
        self,
        identity: IdentityContext,
        release_id: str,
    ) -> tuple[ModelReleaseRecord, ModelReleaseApprovalRecord | None, bool] | None:
        with self._database.transaction(identity.tenant_context) as session:
            release = session.scalar(
                select(ModelReleaseRecord).where(
                    ModelReleaseRecord.tenant_id == identity.tenant_id,
                    ModelReleaseRecord.release_id == release_id,
                )
            )
            if release is None:
                return None
            approval = session.scalar(
                select(ModelReleaseApprovalRecord).where(
                    ModelReleaseApprovalRecord.tenant_id == identity.tenant_id,
                    ModelReleaseApprovalRecord.release_id == release_id,
                )
            )
            deployment = session.scalar(
                select(ModelDeploymentRecord.deployment_id).where(
                    ModelDeploymentRecord.tenant_id == identity.tenant_id,
                    ModelDeploymentRecord.release_id == release_id,
                )
            )
            session.expunge(release)
            if approval is not None:
                session.expunge(approval)
            return release, approval, deployment is not None

    def _mark(
        self,
        identity: IdentityContext,
        onboarding_id: str,
        *,
        status: str,
        error: str | None,
        attempted: bool,
    ) -> None:
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            record = session.scalar(
                select(EnterpriseModelReleaseOnboardingRecord).where(
                    EnterpriseModelReleaseOnboardingRecord.tenant_id == identity.tenant_id,
                    EnterpriseModelReleaseOnboardingRecord.onboarding_id == onboarding_id,
                )
            )
            if record is None:
                return
            record.automation_status = status
            record.last_error = error[:255] if error is not None else None
            if attempted:
                record.attempt_count += 1
                record.last_attempt_at = now
            record.version += 1
            record.updated_at = now
            session.flush()


def _resolve_draft(
    session: Session,
    tenant_id: str,
    evidence: EnterpriseRuntimeEvidence,
) -> _ResolvedDraft:
    blockers: list[str] = []
    model_import = session.scalar(
        select(EnterpriseModelAssetImportRecord).where(
            EnterpriseModelAssetImportRecord.tenant_id == tenant_id,
            EnterpriseModelAssetImportRecord.component == evidence.component,
            EnterpriseModelAssetImportRecord.source_candidate_experiment_id
            == evidence.candidate_experiment_id,
            EnterpriseModelAssetImportRecord.status == "ACTIVE",
        )
    )
    import_integrity_failed = bool(
        model_import is not None
        and model_import.import_hash != enterprise_model_import_hash(model_import)
    )
    if import_integrity_failed:
        blockers.append("ENTERPRISE_MODEL_IMPORT_INTEGRITY_FAILED")
        model_import = None
    elif model_import is None:
        blockers.append("ENTERPRISE_MODEL_ASSET_NOT_IMPORTED")

    candidate = (
        session.scalar(
            select(TrainingExperimentRecord).where(
                TrainingExperimentRecord.tenant_id == tenant_id,
                TrainingExperimentRecord.experiment_id
                == model_import.imported_candidate_experiment_id,
            )
        )
        if model_import is not None
        else None
    )
    if candidate is None:
        if model_import is not None:
            blockers.append("CANDIDATE_EXPERIMENT_NOT_IMPORTED")
    elif not _candidate_method_matches(evidence.component, candidate.method):
        blockers.append("CANDIDATE_METHOD_MISMATCH")
    elif candidate.status != "COMPLETED" or candidate.license_status != "APPROVED":
        blockers.append("CANDIDATE_NOT_RELEASE_ELIGIBLE")
    elif evidence.component == "LLM" and candidate.method == "QUANTIZATION":
        blockers.append("LLM_QUANTIZATION_REQUIRES_MANUAL_RELEASE_PLAN")

    evaluations = (
        _candidate_evaluations(
            session,
            tenant_id,
            evidence,
            candidate_experiment_id=candidate.experiment_id,
            imported_evaluation_id=(
                model_import.imported_evaluation_id if model_import is not None else None
            ),
        )
        if candidate is not None
        else []
    )
    evaluation = evaluations[0] if len(evaluations) == 1 else None
    if candidate is not None and not evaluations:
        blockers.append("CANDIDATE_EVALUATION_NOT_IMPORTED")
    elif len(evaluations) > 1:
        blockers.append("CANDIDATE_EVALUATION_AMBIGUOUS")

    releases = list(
        session.scalars(
            select(ModelReleaseRecord)
            .where(ModelReleaseRecord.tenant_id == tenant_id)
            .order_by(ModelReleaseRecord.created_at.desc())
        )
    )
    runtime_candidate_id = (
        model_import.imported_candidate_experiment_id
        if model_import is not None
        else evidence.candidate_experiment_id
    )
    if any(
        _release_matches_evidence(
            item,
            evidence,
            runtime_candidate_id=runtime_candidate_id,
        )
        for item in releases
    ):
        blockers.append("CANDIDATE_ALREADY_REGISTERED")
    approved_release_ids = set(
        session.scalars(
            select(ModelReleaseApprovalRecord.release_id).where(
                ModelReleaseApprovalRecord.tenant_id == tenant_id,
                ModelReleaseApprovalRecord.status == "APPROVED",
            )
        )
    )
    baselines = tuple(
        item
        for item in releases
        if item.release_id in approved_release_ids
        and item.status in _BASELINE_STATUSES
        and not _release_matches_evidence(
            item,
            evidence,
            runtime_candidate_id=runtime_candidate_id,
        )
        and _is_cloneable_manifest(item.manifest_json)
        and item.manifest_hash == _digest_json(item.manifest_json)
    )
    if not baselines:
        blockers.append("COMPATIBLE_BASELINE_RELEASE_REQUIRED")

    supply_chain = None
    if (
        candidate is not None
        and evidence.component in {"LLM", "RUL", "EMBEDDING", "RERANKER", "VLM", "ASR"}
        and candidate.method != "QUANTIZATION"
    ):
        supply_chain = _matching_supply_chain(
            session,
            tenant_id,
            candidate,
            component=evidence.component,
        )
        if supply_chain is None:
            blockers.append(f"{evidence.component}_SUPPLY_CHAIN_EVIDENCE_REQUIRED")

    return _ResolvedDraft(
        model_import=model_import,
        candidate=candidate,
        evaluation=evaluation,
        supply_chain=supply_chain,
        baselines=baselines,
        blockers=tuple(dict.fromkeys(blockers)),
    )


def _candidate_evaluations(
    session: Session,
    tenant_id: str,
    evidence: EnterpriseRuntimeEvidence,
    *,
    candidate_experiment_id: str,
    imported_evaluation_id: str | None,
) -> list[ModelEvaluationRunRecord]:
    evaluations = list(
        session.scalars(
            select(ModelEvaluationRunRecord)
            .where(
                ModelEvaluationRunRecord.tenant_id == tenant_id,
                ModelEvaluationRunRecord.candidate_experiment_id == candidate_experiment_id,
                ModelEvaluationRunRecord.status == "COMPLETED",
                ModelEvaluationRunRecord.decision == "CANDIDATE",
            )
            .order_by(ModelEvaluationRunRecord.completed_at.desc())
        )
    )
    if imported_evaluation_id is not None:
        return [item for item in evaluations if item.evaluation_id == imported_evaluation_id]
    if evidence.component == "LLM" and evidence.evaluation_reference is not None:
        return [item for item in evaluations if item.evaluation_id == evidence.evaluation_reference]
    if evidence.component == "VLM" and evidence.evaluation_reference is not None:
        return [item for item in evaluations if item.suite_id == evidence.evaluation_reference]
    if (
        evidence.component in {"ASR", "TTS", "EMBEDDING", "RERANKER"}
        and evidence.evaluation_reference is not None
    ):
        return [item for item in evaluations if item.evaluation_id == evidence.evaluation_reference]
    return evaluations


def _matching_supply_chain(
    session: Session,
    tenant_id: str,
    candidate: TrainingExperimentRecord,
    *,
    component: EnterpriseModelComponent,
) -> SupplyChainEvidenceRecord | None:
    if component in {"VLM", "ASR"}:
        matches, _ = matching_component_evidence(
            session, tenant_id, component, candidate.experiment_id
        )
        return matches[0] if matches else None
    artifact_kinds = {
        "LLM": "adapter_bundle",
        "RUL": "rul_transformer_bundle",
        "EMBEDDING": "embedding_model_bundle",
        "RERANKER": "reranker_model_bundle",
        "TTS": "tts_model_bundle",
    }
    artifact_kind = artifact_kinds.get(component)
    if artifact_kind is None:
        return None
    artifacts = list(
        session.scalars(
            select(TrainingArtifactRecord).where(
                TrainingArtifactRecord.tenant_id == tenant_id,
                TrainingArtifactRecord.experiment_id == candidate.experiment_id,
                TrainingArtifactRecord.kind == artifact_kind,
            )
        )
    )
    if len(artifacts) != 1:
        return None
    artifact_digest = _artifact_digest(artifacts[0].content_hash)
    matches = list(
        session.scalars(
            select(SupplyChainEvidenceRecord)
            .where(
                SupplyChainEvidenceRecord.tenant_id == tenant_id,
                SupplyChainEvidenceRecord.model_artifact_digest == artifact_digest,
                SupplyChainEvidenceRecord.source_revision == candidate.git_commit,
                SupplyChainEvidenceRecord.verification_status == "VERIFIED",
                SupplyChainEvidenceRecord.vulnerability_scan_status == "PASSED",
                SupplyChainEvidenceRecord.license_status == "APPROVED",
            )
            .order_by(SupplyChainEvidenceRecord.verified_at.desc())
        )
    )
    return next(
        (item for item in matches if item.verification_hash == record_verification_hash(item)),
        None,
    )


def _plan_from_baseline(
    manifest: dict[str, Any],
    *,
    evidence: EnterpriseRuntimeEvidence,
    candidate: TrainingExperimentRecord,
    evaluation: ModelEvaluationRunRecord,
    supply_chain: SupplyChainEvidenceRecord | None,
    active_mcp_server_versions: dict[str, str],
) -> ReleaseManifestPlan:
    runtime = _object(manifest, "runtime")
    retrieval = _object(manifest, "retrieval")
    specialized = _object(manifest, "specialized_components")
    component_evaluations: dict[str, str] = {}
    for name, value in specialized.items():
        if not isinstance(value, dict):
            raise ValueError("specialized component must be an object")
        component_evaluations[name] = _text(_object(value, "evaluation")["evaluation_id"])
    multimodal = {
        key: _text(value) for key, value in _object(manifest, "multimodal_model_ids").items()
    }
    if evidence.component == "VLM":
        component_evaluations["vlm"] = evaluation.evaluation_id
        multimodal["vlm"] = candidate.experiment_id
    if evidence.component == "ASR":
        component_evaluations["asr"] = evaluation.evaluation_id
        multimodal["asr"] = candidate.experiment_id
    if evidence.component == "TTS":
        component_evaluations["tts"] = evaluation.evaluation_id
        multimodal["tts"] = candidate.experiment_id
    if evidence.component == "RUL":
        component_evaluations["rul"] = evaluation.evaluation_id
    if evidence.component == "EMBEDDING":
        component_evaluations["embedding"] = evaluation.evaluation_id
    if evidence.component == "RERANKER":
        component_evaluations["reranker"] = evaluation.evaluation_id
    main_evaluation_id = (
        evaluation.evaluation_id
        if evidence.component == "LLM"
        else _text(_object(manifest, "evaluation")["evaluation_id"])
    )
    primary_supply_chain = _text(_object(manifest, "supply_chain")["evidence_id"])
    runtime_image_repository = _text(runtime["image_repository"])
    runtime_image_digest = _text(runtime["image_digest"])
    if evidence.component == "LLM":
        if supply_chain is None:
            raise EnterpriseReleaseOnboardingConflict("LLM_SUPPLY_CHAIN_EVIDENCE_REQUIRED")
        primary_supply_chain = supply_chain.evidence_id
        runtime_image_repository = supply_chain.image_repository
        runtime_image_digest = supply_chain.image_digest

    return ReleaseManifestPlan(
        evaluation_id=main_evaluation_id,
        component_evaluation_ids=component_evaluations,
        target_environment="STAGING",
        quantization_profile_id=_text(manifest["quantization_profile_id"]),
        runtime_profile_id=_text(runtime["profile_id"]),
        runtime_image_repository=runtime_image_repository,
        runtime_image_digest=runtime_image_digest,
        prompt_bundle_id=_text(manifest["prompt_bundle_id"]),
        agent_graph_id=_text(manifest["agent_graph_id"]),
        tool_versions={
            key: _text(value) for key, value in _object(manifest, "tool_registry").items()
        },
        mcp_server_versions=dict(active_mcp_server_versions),
        index_release_id=_text(retrieval["index_release_id"]),
        embedding_model_id=(
            candidate.experiment_id
            if evidence.component == "EMBEDDING"
            else _text(retrieval["embedding_model_id"])
        ),
        reranker_model_id=(
            candidate.experiment_id
            if evidence.component == "RERANKER"
            else _text(retrieval["reranker_model_id"])
        ),
        multimodal_model_ids=multimodal,
        inference_config=dict(_object(runtime, "inference_config")),
        hardware_profile=dict(_object(runtime, "hardware_profile")),
        supply_chain_evidence_id=primary_supply_chain,
        rollback_release_id=None,
        timeseries_model_id=_component_runtime_id(specialized, "timeseries"),
        timeseries_supply_chain_evidence_id=_component_supply_chain_id(
            specialized,
            "timeseries",
        ),
        rul_model_id=(
            candidate.experiment_id
            if evidence.component == "RUL"
            else _component_runtime_id(specialized, "rul")
        ),
        rul_supply_chain_evidence_id=(
            supply_chain.evidence_id
            if evidence.component == "RUL" and supply_chain is not None
            else _component_supply_chain_id(specialized, "rul")
        ),
        reranker_supply_chain_evidence_id=(
            supply_chain.evidence_id
            if evidence.component == "RERANKER" and supply_chain is not None
            else _component_supply_chain_id(specialized, "reranker")
        ),
        embedding_supply_chain_evidence_id=(
            supply_chain.evidence_id
            if evidence.component == "EMBEDDING" and supply_chain is not None
            else _component_supply_chain_id(specialized, "embedding")
        ),
        vlm_supply_chain_evidence_id=(
            supply_chain.evidence_id
            if evidence.component == "VLM" and supply_chain is not None
            else _component_supply_chain_id(specialized, "vlm")
        ),
        asr_supply_chain_evidence_id=(
            supply_chain.evidence_id
            if evidence.component == "ASR" and supply_chain is not None
            else _component_supply_chain_id(specialized, "asr")
        ),
        tts_supply_chain_evidence_id=(
            supply_chain.evidence_id
            if evidence.component == "TTS" and supply_chain is not None
            else _component_supply_chain_id(specialized, "tts")
        ),
    )


def _is_cloneable_manifest(manifest: dict[str, Any]) -> bool:
    try:
        if manifest.get("schema_version") not in {
            "ai-release-manifest/v4", "ai-release-manifest/v5",
        }:
            return False
        runtime = _object(manifest, "runtime")
        retrieval = _object(manifest, "retrieval")
        specialized = _object(manifest, "specialized_components")
        multimodal = _object(manifest, "multimodal_model_ids")
        if not {"embedding", "reranker", "vlm", "asr"}.issubset(specialized):
            return False
        if set(multimodal) != {"ocr", "vlm", "asr", "tts"}:
            return False
        for key in (
            "quantization_profile_id",
            "prompt_bundle_id",
            "agent_graph_id",
        ):
            _text(manifest[key])
        for key in ("profile_id", "image_repository", "image_digest"):
            _text(runtime[key])
        for key in ("index_release_id", "embedding_model_id", "reranker_model_id"):
            _text(retrieval[key])
        _object(runtime, "inference_config")
        _object(runtime, "hardware_profile")
        _object(manifest, "tool_registry")
        _object(manifest, "mcp_server_versions")
        _text(_object(manifest, "evaluation")["evaluation_id"])
        _text(_object(manifest, "supply_chain")["evidence_id"])
        for value in specialized.values():
            if not isinstance(value, dict):
                return False
            _text(_object(value, "evaluation")["evaluation_id"])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _component_runtime_id(
    specialized: dict[str, Any],
    component: str,
) -> str | None:
    value = specialized.get(component)
    if not isinstance(value, dict):
        return None
    runtime_id = value.get("runtime_model_id")
    return runtime_id if isinstance(runtime_id, str) and runtime_id else None


def _component_supply_chain_id(
    specialized: dict[str, Any],
    component: str,
) -> str | None:
    value = specialized.get(component)
    if not isinstance(value, dict):
        return None
    supply_chain = value.get("supply_chain")
    if not isinstance(supply_chain, dict):
        return None
    evidence_id = supply_chain.get("evidence_id")
    return evidence_id if isinstance(evidence_id, str) and evidence_id else None


def _candidate_method_matches(component: EnterpriseModelComponent, method: str) -> bool:
    if component == "LLM":
        return method in PRODUCTION_RELEASE_METHODS
    return method == _COMPONENT_METHOD[component]


def _release_matches_evidence(
    release: ModelReleaseRecord,
    evidence: EnterpriseRuntimeEvidence,
    *,
    runtime_candidate_id: str,
) -> bool:
    if evidence.component == "LLM":
        return release.candidate_experiment_id == runtime_candidate_id
    specialized = release.manifest_json.get("specialized_components")
    if not isinstance(specialized, dict):
        return False
    component = specialized.get(evidence.component.lower())
    return bool(
        isinstance(component, dict) and component.get("runtime_model_id") == runtime_candidate_id
    )


def _baseline_projection(record: ModelReleaseRecord) -> EnterpriseReleaseDraftBaseline:
    return EnterpriseReleaseDraftBaseline(
        release_id=record.release_id,
        status=record.status,
        target_environment=record.target_environment,
        candidate_experiment_id=record.candidate_experiment_id,
        manifest_hash=record.manifest_hash,
        version=record.version,
        created_at=record.created_at,
    )


def _onboarding_projection(
    record: EnterpriseModelReleaseOnboardingRecord,
    release_status: str,
    release_version: int,
) -> EnterpriseReleaseOnboarding:
    return EnterpriseReleaseOnboarding(
        onboarding_id=record.onboarding_id,
        release_id=record.release_id,
        release_status=release_status,
        release_version=release_version,
        baseline_release_id=record.baseline_release_id,
        component=cast(EnterpriseModelComponent, record.component),
        candidate_experiment_id=record.candidate_experiment_id,
        evaluation_id=record.evaluation_id,
        evidence_chain_sha256=record.evidence_chain_sha256,
        auto_shadow_enabled=record.auto_shadow_enabled,
        automation_status=cast(
            EnterpriseReleaseAutomationStatus,
            record.automation_status,
        ),
        attempt_count=record.attempt_count,
        last_error=record.last_error,
        last_attempt_at=record.last_attempt_at,
        version=record.version,
        created_at=record.created_at,
    )


def _evidence_for_component(
    reader: EnterpriseProjectAdoptionReader,
    component: EnterpriseModelComponent,
) -> EnterpriseRuntimeEvidence:
    matches = [item for item in reader.runtime_evidence() if item.component == component]
    if len(matches) != 1:
        raise EnterpriseReleaseOnboardingConflict("ENTERPRISE_RUNTIME_EVIDENCE_NOT_FOUND")
    return matches[0]


def _release(session: Session, tenant_id: str, release_id: str) -> ModelReleaseRecord:
    record = session.scalar(
        select(ModelReleaseRecord).where(
            ModelReleaseRecord.tenant_id == tenant_id,
            ModelReleaseRecord.release_id == release_id,
        )
    )
    if record is None:
        raise EnterpriseReleaseOnboardingConflict("release_not_found_or_not_visible")
    return record


def _object(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document[key]
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("manifest identifier is invalid")
    return value


def _artifact_digest(content_hash: str) -> str:
    return f"sha256:{content_hash.removeprefix('sha256:')}"


def _digest_json(document: object) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(payload).hexdigest()


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, AuthorizationDenied):
        return "authorization_denied"
    if isinstance(exc, DeploymentNotVisible):
        return "deployment_not_visible"
    return str(exc) or type(exc).__name__
