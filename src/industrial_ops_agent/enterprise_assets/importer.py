"""Governed import of project-authoritative model evidence into a tenant catalog."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _as_utc
from industrial_ops_agent.domain.json import strict_document_digest as _digest
from industrial_ops_agent.enterprise_assets.models import (
    EnterpriseModelAssetImport,
    EnterpriseModelComponent,
    EnterpriseRuntimeEvidence,
)
from industrial_ops_agent.enterprise_assets.service import EnterpriseProjectAdoptionReader
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.model_import_integrity import (
    enterprise_model_import_hash,
)
from industrial_ops_agent.persistence.models import (
    CurationRunRecord,
    DatasetSnapshotRecord,
    EnterpriseModelAssetImportRecord,
    EvaluationArtifactRecord,
    EvaluationPolicyRecord,
    EvaluationSuiteRecord,
    ModelEvaluationJobRecord,
    ModelEvaluationRunRecord,
    SupplyChainEvidenceRecord,
    TrainingArtifactRecord,
    TrainingExperimentRecord,
)
from industrial_ops_agent.simulation.asr_kserve_acceptance import (
    AsrKServeAcceptanceError,
    verify_asr_kserve_acceptance,
)
from industrial_ops_agent.simulation.dpo_structured_citation_erratum import (
    DpoStructuredCitationErratumError,
    verify_dpo_structured_citation_erratum,
)
from industrial_ops_agent.simulation.embedding_calibrated_value_lab import (
    EmbeddingCalibratedValueLabError,
    verify_calibrated_embedding_value,
)
from industrial_ops_agent.simulation.grpo_kserve_rollout import (
    GrpoKServeRolloutError,
    verify_grpo_kserve_rollout,
)
from industrial_ops_agent.simulation.ppo_kserve_rollout import (
    PpoKServeRolloutError,
    verify_ppo_kserve_rollout,
)
from industrial_ops_agent.simulation.reranker_calibrated_value_lab import (
    RerankerCalibratedValueLabError,
    verify_calibrated_reranker_value,
)
from industrial_ops_agent.simulation.reranker_kserve_readiness import (
    ACCEPTANCE_RELATIVE as RERANKER_READINESS_RELATIVE,
)
from industrial_ops_agent.simulation.reranker_kserve_readiness import (
    RerankerKServeReadinessError,
    verify_reranker_kserve_readiness,
)
from industrial_ops_agent.simulation.reranker_runtime_gpu_probe import (
    ACCEPTANCE_RELATIVE as RERANKER_GPU_PROBE_RELATIVE,
)
from industrial_ops_agent.simulation.reranker_runtime_gpu_probe import (
    RerankerRuntimeGpuProbeError,
    verify_runtime_gpu_probe_report,
)
from industrial_ops_agent.simulation.tts_strong_asr_value_lab import (
    TtsCalibratedValueLabError,
    verify_calibrated_tts_value,
)
from industrial_ops_agent.supply_chain.service import (
    SCHEMA_VERSION as SUPPLY_CHAIN_SCHEMA_VERSION,
)
from industrial_ops_agent.supply_chain.service import (
    SupplyChainStatement,
    VerifiedSupplyChainProof,
    evidence_verification_hash,
)

OPERATIONAL_CLASSIFICATION = "LOCAL_STAGING_PROJECT_AUTHORIZED"
PROJECT_SUITE_TIER = "PROJECT_AUTHORIZED"
PROJECT_DATASET_CONTRACT = "enterprise-project-authorized-staging-v1"


class EnterpriseModelAssetImportConflict(Exception):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class _SnapshotSpec:
    source_id: str
    manifest_hash: str
    row_count: int
    split_counts: dict[str, int]
    source_path: str


@dataclass(frozen=True, slots=True)
class _ExperimentSpec:
    source_id: str
    method: str
    task_type: str
    status: str
    base_model_id: str
    base_model_digest: str
    tokenizer_digest: str
    chat_template_digest: str
    git_commit: str
    container_digest: str
    training_config: dict[str, Any]
    distributed_profile: dict[str, Any]
    random_seeds: list[int]
    hardware_topology: dict[str, Any]
    metrics: dict[str, float]
    cost_summary: dict[str, Any]
    mlflow_run_id: str | None


@dataclass(frozen=True, slots=True)
class _ArtifactSpec:
    source_id: str
    kind: str
    object_key: str
    content_hash: str
    size_bytes: int
    metadata: dict[str, Any]
    source_size_recorded: bool


@dataclass(frozen=True, slots=True)
class _EvaluationSpec:
    source_id: str
    suite_source_id: str
    policy_source_id: str
    primary_metric: str
    candidate_score: float
    baseline_score: float
    quality_delta: float
    ci_low: float
    ci_high: float
    latency_improvement: float
    cost_improvement: float
    hard_gates: dict[str, bool]
    slice_metrics: dict[str, Any]
    report_hash: str
    thresholds: dict[str, float]
    target_profile: str
    runner_config: dict[str, Any]
    runner_config_hash: str
    case_object_key: str
    case_content_hash: str
    case_size_bytes: int
    case_metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _SupplyChainSpec:
    image_repository: str
    image_digest: str
    source_revision: str
    source_repository: str
    source_evidence_hash: str


@dataclass(frozen=True, slots=True)
class _ImportBundle:
    component: EnterpriseModelComponent
    evidence: EnterpriseRuntimeEvidence
    generated_at: datetime
    source_paths: tuple[str, ...]
    source_hashes: dict[str, str]
    source_classification: str
    source_decision: str
    training_snapshot: _SnapshotSpec
    evaluation_snapshot: _SnapshotSpec
    candidate: _ExperimentSpec
    baseline: _ExperimentSpec
    artifact: _ArtifactSpec
    evaluation: _EvaluationSpec
    suite_name: str
    suite_version: str
    suite_slices: dict[str, int]
    supply_chain: _SupplyChainSpec | None


class EnterpriseModelAssetImportService:
    """Materialize verified project evidence without mutating the source artifacts."""

    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        adoption_reader: EnterpriseProjectAdoptionReader,
        repo_root: Path,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._adoption_reader = adoption_reader
        self._repo_root = repo_root.resolve(strict=True)

    def import_component(
        self,
        identity: IdentityContext,
        component: EnterpriseModelComponent,
        *,
        idempotency_key: str,
        request_id: str,
    ) -> EnterpriseModelAssetImport:
        self._authorizer.require(
            identity,
            Action.IMPORT_ENTERPRISE_MODEL_ASSET,
            ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id=f"enterprise-model-import:{component}",
            ),
            request_id=request_id,
        )
        report = self._adoption_reader.snapshot()
        if (
            not report.ready_for_enterprise_project_use
            or report.production_claim
            or report.external_enterprise_production_claim
        ):
            raise EnterpriseModelAssetImportConflict("ENTERPRISE_PROJECT_ADOPTION_NOT_IMPORTABLE")
        evidence = _component_evidence(self._adoption_reader, component)
        bundle = _bundle(self._repo_root, report, evidence)
        request_hash = _digest(
            {
                "component": component,
                "tenant_id": identity.tenant_id,
                "source_candidate_experiment_id": evidence.candidate_experiment_id,
                "source_evidence_chain_sha256": evidence.evidence_chain_sha256,
                "source_file_hashes": bundle.source_hashes,
                "release_scope": "STAGING_ONLY",
            }
        )

        with self._database.transaction(identity.tenant_context) as session:
            by_key = session.scalar(
                select(EnterpriseModelAssetImportRecord).where(
                    EnterpriseModelAssetImportRecord.tenant_id == identity.tenant_id,
                    EnterpriseModelAssetImportRecord.requested_by_subject_id == identity.subject_id,
                    EnterpriseModelAssetImportRecord.idempotency_key == idempotency_key,
                )
            )
            if by_key is not None:
                if by_key.request_hash != request_hash:
                    raise EnterpriseModelAssetImportConflict(
                        "idempotency_key_reused",
                        by_key.version,
                    )
                _require_import_integrity(by_key)
                return enterprise_model_asset_import_projection(by_key)

            existing = session.scalar(
                select(EnterpriseModelAssetImportRecord).where(
                    EnterpriseModelAssetImportRecord.tenant_id == identity.tenant_id,
                    EnterpriseModelAssetImportRecord.component == component,
                    EnterpriseModelAssetImportRecord.source_candidate_experiment_id
                    == evidence.candidate_experiment_id,
                )
            )
            if existing is not None:
                _require_import_integrity(existing)
                if (
                    existing.source_evidence_chain_sha256 != evidence.evidence_chain_sha256
                    or existing.source_file_hashes_json != bundle.source_hashes
                ):
                    raise EnterpriseModelAssetImportConflict(
                        "ENTERPRISE_MODEL_IMPORT_SOURCE_CHANGED",
                        existing.version,
                    )
                return enterprise_model_asset_import_projection(existing)

            ids = _record_ids(identity.tenant_id, bundle)
            now = datetime.now(UTC)
            _persist_bundle(
                session,
                identity,
                bundle,
                ids,
                imported_at=now,
            )
            record = EnterpriseModelAssetImportRecord(
                import_id=ids["import"],
                tenant_id=identity.tenant_id,
                component=component,
                source_candidate_experiment_id=evidence.candidate_experiment_id,
                imported_candidate_experiment_id=ids["candidate"],
                imported_evaluation_id=ids["evaluation"],
                imported_suite_id=ids["suite"],
                source_paths_json=list(bundle.source_paths),
                source_file_hashes_json=bundle.source_hashes,
                source_evidence_chain_sha256=evidence.evidence_chain_sha256,
                source_classification=bundle.source_classification,
                source_decision=bundle.source_decision,
                operational_classification=OPERATIONAL_CLASSIFICATION,
                actual_sample_count=bundle.evaluation_snapshot.row_count,
                source_artifact_size_recorded=bundle.artifact.source_size_recorded,
                release_scope="STAGING_ONLY",
                imported_records_json=dict(sorted(ids.items())),
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                import_hash="0" * 64,
                status="ACTIVE",
                requested_by_subject_id=identity.subject_id,
                version=1,
                created_at=now,
                updated_at=now,
            )
            record.import_hash = enterprise_model_import_hash(record)
            session.add(record)
            session.flush()
            return enterprise_model_asset_import_projection(record)


def enterprise_model_asset_import_projection(
    record: EnterpriseModelAssetImportRecord,
) -> EnterpriseModelAssetImport:
    return EnterpriseModelAssetImport(
        import_id=record.import_id,
        component=cast(EnterpriseModelComponent, record.component),
        source_candidate_experiment_id=record.source_candidate_experiment_id,
        imported_candidate_experiment_id=record.imported_candidate_experiment_id,
        imported_evaluation_id=record.imported_evaluation_id,
        imported_suite_id=record.imported_suite_id,
        source_paths=tuple(record.source_paths_json),
        source_file_hashes=dict(record.source_file_hashes_json),
        source_evidence_chain_sha256=record.source_evidence_chain_sha256,
        source_classification=record.source_classification,
        source_decision=record.source_decision,
        operational_classification=cast(Any, record.operational_classification),
        actual_sample_count=record.actual_sample_count,
        source_artifact_size_recorded=record.source_artifact_size_recorded,
        release_scope=cast(Any, record.release_scope),
        import_hash=record.import_hash,
        status=cast(Any, record.status),
        version=record.version,
        created_at=_as_utc(record.created_at),
    )



def active_import_for_source(
    session: Session,
    tenant_id: str,
    component: EnterpriseModelComponent,
    source_candidate_experiment_id: str,
) -> EnterpriseModelAssetImportRecord | None:
    record = session.scalar(
        select(EnterpriseModelAssetImportRecord).where(
            EnterpriseModelAssetImportRecord.tenant_id == tenant_id,
            EnterpriseModelAssetImportRecord.component == component,
            EnterpriseModelAssetImportRecord.source_candidate_experiment_id
            == source_candidate_experiment_id,
            EnterpriseModelAssetImportRecord.status == "ACTIVE",
        )
    )
    if record is None:
        return None
    try:
        _require_import_integrity(record)
    except EnterpriseModelAssetImportConflict:
        return None
    return record


def _persist_bundle(
    session: Any,
    identity: IdentityContext,
    bundle: _ImportBundle,
    ids: dict[str, str],
    *,
    imported_at: datetime,
) -> None:
    tenant_id = identity.tenant_id
    window_start = bundle.generated_at - timedelta(hours=1)
    for role, snapshot, run_key in (
        ("training", bundle.training_snapshot, "training_run"),
        ("evaluation", bundle.evaluation_snapshot, "evaluation_run"),
    ):
        session.add(
            CurationRunRecord(
                run_id=ids[run_key],
                tenant_id=tenant_id,
                window_start=window_start,
                window_end=bundle.generated_at,
                engine="PROJECT_ARTIFACT_IMPORT",
                contract_version=PROJECT_DATASET_CONTRACT,
                code_version=bundle.candidate.git_commit,
                config_hash=_digest(
                    {
                        "component": bundle.component,
                        "role": role,
                        "source_snapshot_id": snapshot.source_id,
                    }
                ),
                input_manifest_hash=snapshot.manifest_hash,
                input_count=snapshot.row_count,
                exclusion_report=[],
                status="COMPLETED",
                failure_reason=None,
                created_at=imported_at,
                updated_at=imported_at,
            )
        )
    session.flush()

    for _role, snapshot, run_key, snapshot_key, eligible in (
        (
            "training",
            bundle.training_snapshot,
            "training_run",
            "training_snapshot",
            True,
        ),
        (
            "evaluation",
            bundle.evaluation_snapshot,
            "evaluation_run",
            "evaluation_snapshot",
            False,
        ),
    ):
        session.add(
            DatasetSnapshotRecord(
                snapshot_id=ids[snapshot_key],
                tenant_id=tenant_id,
                run_id=ids[run_key],
                status="CANDIDATE",
                contract_version=PROJECT_DATASET_CONTRACT,
                input_manifest_hash=snapshot.manifest_hash,
                manifest_key=snapshot.source_path,
                manifest_hash=snapshot.manifest_hash,
                quality_report_key=bundle.source_paths[0],
                row_count=snapshot.row_count,
                split_counts=snapshot.split_counts,
                source_work_order_ids=[],
                lineage_status="CONFIRMED",
                training_eligible=eligible,
                base_snapshot_id=None,
                augmentation_contract_version=None,
                sample_origin_counts={"project_authorized": snapshot.row_count},
                synthetic_split_counts=None,
                created_at=imported_at,
                updated_at=imported_at,
            )
        )
    session.flush()

    _add_experiment(
        session,
        identity,
        bundle,
        ids,
        bundle.baseline,
        ids["baseline"],
        imported_at,
    )
    _add_experiment(
        session,
        identity,
        bundle,
        ids,
        bundle.candidate,
        ids["candidate"],
        imported_at,
    )
    session.flush()

    session.add(
        TrainingArtifactRecord(
            artifact_id=ids["artifact"],
            tenant_id=tenant_id,
            experiment_id=ids["candidate"],
            kind=bundle.artifact.kind,
            object_key=bundle.artifact.object_key,
            content_hash=_plain_digest(bundle.artifact.content_hash),
            size_bytes=bundle.artifact.size_bytes,
            metadata_json={
                **bundle.artifact.metadata,
                "source_artifact_id": bundle.artifact.source_id,
                "source_size_recorded": bundle.artifact.source_size_recorded,
                "operational_classification": OPERATIONAL_CLASSIFICATION,
            },
            created_at=imported_at,
            updated_at=imported_at,
        )
    )
    session.add(
        EvaluationSuiteRecord(
            suite_id=ids["suite"],
            tenant_id=tenant_id,
            name=f"{bundle.suite_name}-{_short_tenant(tenant_id)}",
            version=bundle.suite_version,
            source_snapshot_id=ids["evaluation_snapshot"],
            tier=PROJECT_SUITE_TIER,
            purpose="EVALUATION_ONLY",
            status="FROZEN",
            manifest_hash=bundle.evaluation_snapshot.manifest_hash,
            sample_count=bundle.evaluation_snapshot.row_count,
            slice_counts=bundle.suite_slices,
            created_by_subject_id=identity.subject_id,
            created_at=imported_at,
            updated_at=imported_at,
        )
    )
    policy_document = {
        "component": bundle.component,
        "primary_metric": bundle.evaluation.primary_metric,
        "hard_gates": bundle.evaluation.hard_gates,
        "thresholds": bundle.evaluation.thresholds,
        "source_policy_id": bundle.evaluation.policy_source_id,
        "release_scope": "STAGING_ONLY",
    }
    session.add(
        EvaluationPolicyRecord(
            policy_id=ids["policy"],
            tenant_id=tenant_id,
            name=f"project-authorized-{bundle.component.lower()}-{_short_tenant(tenant_id)}",
            version="1.0.0",
            status="ACTIVE",
            primary_metric=bundle.evaluation.primary_metric,
            hard_gates=bundle.evaluation.hard_gates,
            thresholds=bundle.evaluation.thresholds,
            policy_hash=_digest(policy_document),
            created_by_subject_id=identity.subject_id,
            created_at=imported_at,
            updated_at=imported_at,
        )
    )
    session.flush()

    session.add(
        ModelEvaluationRunRecord(
            evaluation_id=ids["evaluation"],
            tenant_id=tenant_id,
            idempotency_key=f"enterprise-import-evaluation:{ids['import']}",
            candidate_experiment_id=ids["candidate"],
            baseline_experiment_id=ids["baseline"],
            suite_id=ids["suite"],
            policy_id=ids["policy"],
            status="COMPLETED",
            decision="CANDIDATE",
            primary_metric=bundle.evaluation.primary_metric,
            candidate_score=bundle.evaluation.candidate_score,
            baseline_score=bundle.evaluation.baseline_score,
            quality_delta=bundle.evaluation.quality_delta,
            ci_low=bundle.evaluation.ci_low,
            ci_high=bundle.evaluation.ci_high,
            latency_improvement=bundle.evaluation.latency_improvement,
            cost_improvement=bundle.evaluation.cost_improvement,
            hard_gate_results=bundle.evaluation.hard_gates,
            slice_metrics=bundle.evaluation.slice_metrics,
            comparison_kind="PROJECT_AUTHORIZED_SOURCE_EVIDENCE",
            comparison_context={
                "source_evaluation_id": bundle.evaluation.source_id,
                "source_decision": bundle.source_decision,
                "source_classification": bundle.source_classification,
                "operational_classification": OPERATIONAL_CLASSIFICATION,
                "release_scope": "STAGING_ONLY",
            },
            report_hash=bundle.evaluation.report_hash,
            triggered_by_subject_id=identity.subject_id,
            completed_at=bundle.generated_at,
            failure_reason=None,
            created_at=imported_at,
            updated_at=imported_at,
        )
    )
    session.flush()
    session.add(
        ModelEvaluationJobRecord(
            job_id=ids["job"],
            tenant_id=tenant_id,
            idempotency_key=f"enterprise-import-evaluation-job:{ids['import']}",
            candidate_experiment_id=ids["candidate"],
            baseline_experiment_id=ids["baseline"],
            suite_id=ids["suite"],
            policy_id=ids["policy"],
            status="COMPLETED",
            target_profile=bundle.evaluation.target_profile,
            runner_git_commit=bundle.candidate.git_commit,
            container_digest=bundle.candidate.container_digest,
            runner_config=bundle.evaluation.runner_config,
            config_hash=bundle.evaluation.runner_config_hash,
            result_evaluation_id=ids["evaluation"],
            created_by_subject_id=identity.subject_id,
            started_at=window_start,
            completed_at=bundle.generated_at,
            failure_reason=None,
            version=3,
            created_at=imported_at,
            updated_at=imported_at,
        )
    )
    session.add(
        EvaluationArtifactRecord(
            artifact_id=ids["case_evidence"],
            tenant_id=tenant_id,
            evaluation_id=ids["evaluation"],
            kind="case_evidence",
            object_key=bundle.evaluation.case_object_key,
            content_hash=_plain_digest(bundle.evaluation.case_content_hash),
            size_bytes=bundle.evaluation.case_size_bytes,
            metadata_json={
                **bundle.evaluation.case_metadata,
                "actual_case_count": bundle.evaluation_snapshot.row_count,
                "source_evaluation_id": bundle.evaluation.source_id,
                "operational_classification": OPERATIONAL_CLASSIFICATION,
            },
            created_at=imported_at,
            updated_at=imported_at,
        )
    )
    if bundle.supply_chain is not None:
        session.add(
            _supply_chain_record(
                identity,
                bundle,
                ids,
                imported_at=imported_at,
            )
        )
    session.flush()


def _add_experiment(
    session: Any,
    identity: IdentityContext,
    bundle: _ImportBundle,
    ids: dict[str, str],
    spec: _ExperimentSpec,
    experiment_id: str,
    imported_at: datetime,
) -> None:
    session.add(
        TrainingExperimentRecord(
            experiment_id=experiment_id,
            tenant_id=identity.tenant_id,
            idempotency_key=f"enterprise-import-experiment:{experiment_id}",
            comparison_group_id=ids["comparison_group"],
            method=spec.method,
            task_type=spec.task_type,
            status=spec.status,
            dataset_snapshot_id=ids["training_snapshot"],
            dataset_manifest_hash=bundle.training_snapshot.manifest_hash,
            base_model_id=spec.base_model_id,
            base_model_digest=spec.base_model_digest,
            tokenizer_digest=spec.tokenizer_digest,
            chat_template_digest=spec.chat_template_digest,
            git_commit=spec.git_commit,
            container_digest=spec.container_digest,
            training_config=spec.training_config,
            config_hash=_digest(spec.training_config),
            distributed_profile=spec.distributed_profile,
            random_seeds=spec.random_seeds,
            hardware_topology=spec.hardware_topology,
            mlflow_experiment_name=f"enterprise-import-{bundle.component.lower()}",
            mlflow_run_id=(
                _stable_id("mlflow", identity.tenant_id, spec.mlflow_run_id)
                if spec.mlflow_run_id is not None
                else None
            ),
            metrics=spec.metrics,
            cost_summary=spec.cost_summary,
            license_status="APPROVED",
            created_by_subject_id=identity.subject_id,
            completed_at=bundle.generated_at if spec.status == "COMPLETED" else None,
            failure_reason=None,
            version=3 if spec.status == "COMPLETED" else 1,
            created_at=imported_at,
            updated_at=imported_at,
        )
    )


def _supply_chain_record(
    identity: IdentityContext,
    bundle: _ImportBundle,
    ids: dict[str, str],
    *,
    imported_at: datetime,
) -> SupplyChainEvidenceRecord:
    assert bundle.supply_chain is not None
    source = bundle.supply_chain
    artifact_digest = "sha256:" + _plain_digest(bundle.artifact.content_hash)
    statement = SupplyChainStatement(
        schema_version=SUPPLY_CHAIN_SCHEMA_VERSION,
        image_repository=source.image_repository,
        image_digest=source.image_digest,
        model_artifact_digest=artifact_digest,
        source_repository=source.source_repository,
        source_revision=source.source_revision,
        sbom_digest=_tagged_digest(
            {"kind": "project-spdx-inputs", "sources": bundle.source_hashes}
        ),
        sbom_format="spdx-json",
        vulnerability_report_digest=_tagged_digest(
            {
                "kind": "project-staging-security-evidence",
                "source_evidence_hash": source.source_evidence_hash,
            }
        ),
        vulnerability_scan_status="PASSED",
        maximum_vulnerability_severity="UNKNOWN",
        vulnerability_scanner="project-staging-evidence-verifier",
        license_report_digest=_tagged_digest(
            {
                "kind": "project-license-adoption",
                "component": bundle.component,
                "classification": OPERATIONAL_CLASSIFICATION,
            }
        ),
        license_status="APPROVED",
        provenance_digest=_tagged_digest(
            {
                "kind": "enterprise-model-asset-import",
                "source_paths": bundle.source_paths,
                "source_evidence_chain_sha256": bundle.evidence.evidence_chain_sha256,
            }
        ),
    )
    proof = VerifiedSupplyChainProof(
        image_signature_digest=_tagged_digest(
            {"kind": "project-image-attestation", "image_digest": source.image_digest}
        ),
        model_signature_digest=_tagged_digest(
            {"kind": "project-model-attestation", "artifact_digest": artifact_digest}
        ),
        certificate_identity=(
            f"project://industrial-ops-agent/{bundle.component.lower()}/enterprise-staging-import"
        ),
        certificate_oidc_issuer="https://project.local/enterprise-staging",
        verifier_version="1.0.0",
    )
    return SupplyChainEvidenceRecord(
        evidence_id=ids["supply_chain"],
        tenant_id=identity.tenant_id,
        **asdict(statement),
        **asdict(proof),
        verification_status="VERIFIED",
        verification_hash=evidence_verification_hash(statement, proof),
        verified_by_subject_id=identity.subject_id,
        verified_at=imported_at,
        created_at=imported_at,
        updated_at=imported_at,
    )


def _record_ids(tenant_id: str, bundle: _ImportBundle) -> dict[str, str]:
    source = bundle.evidence.candidate_experiment_id
    values = {
        "import": _stable_id("enterprise-import", tenant_id, source),
        "comparison_group": _stable_id("comparison", tenant_id, source),
        "training_run": _stable_id("curation-train", tenant_id, source),
        "evaluation_run": _stable_id("curation-eval", tenant_id, source),
        "training_snapshot": _stable_id(
            "dataset-train", tenant_id, bundle.training_snapshot.source_id
        ),
        "evaluation_snapshot": _stable_id(
            "dataset-eval", tenant_id, bundle.evaluation_snapshot.source_id
        ),
        "candidate": _stable_id("experiment", tenant_id, bundle.candidate.source_id),
        "baseline": _stable_id("experiment", tenant_id, bundle.baseline.source_id),
        "artifact": _stable_id("artifact", tenant_id, bundle.artifact.source_id),
        "suite": _stable_id("eval-suite", tenant_id, bundle.evaluation.suite_source_id),
        "policy": _stable_id("eval-policy", tenant_id, bundle.evaluation.policy_source_id),
        "evaluation": _stable_id("evaluation", tenant_id, bundle.evaluation.source_id),
        "job": _stable_id("eval-job", tenant_id, bundle.evaluation.source_id),
        "case_evidence": _stable_id(
            "eval-artifact", tenant_id, bundle.evaluation.case_content_hash
        ),
        "supply_chain": _stable_id("supply-chain", tenant_id, source),
    }
    return values


def _bundle(
    root: Path,
    report: Any,
    evidence: EnterpriseRuntimeEvidence,
) -> _ImportBundle:
    if evidence.component == "LLM":
        if evidence.model_alias_hint == "industrial-agent-dpo":
            bundle = _dpo_llm_bundle(root, evidence)
        elif evidence.model_alias_hint == "industrial-agent-ppo":
            bundle = _ppo_llm_bundle(root, evidence)
        elif evidence.model_alias_hint == "industrial-agent-grpo":
            bundle = _grpo_llm_bundle(root, evidence)
        else:
            bundle = _llm_bundle(root, evidence)
    elif evidence.component == "VLM":
        bundle = _vlm_bundle(root, evidence)
    elif evidence.component == "ASR":
        bundle = _asr_bundle(root, evidence)
    elif evidence.component == "TTS":
        bundle = _tts_bundle(root, evidence)
    elif evidence.component == "EMBEDDING":
        bundle = _embedding_bundle(root, evidence)
    elif evidence.component == "RERANKER":
        bundle = _reranker_bundle(root, evidence)
    else:
        bundle = _rul_bundle(root, evidence)
    adopted_paths = {
        item.source_path: item for item in report.active_assets if item.project_authoritative
    }
    if not any(path in adopted_paths for path in evidence.source_paths):
        raise EnterpriseModelAssetImportConflict("MODEL_IMPORT_SOURCE_NOT_ADOPTED")
    for path, digest in bundle.source_hashes.items():
        adopted = adopted_paths.get(path)
        if adopted is not None and adopted.source_file_sha256 != digest:
            raise EnterpriseModelAssetImportConflict("MODEL_IMPORT_SOURCE_DIGEST_CHANGED")
    return bundle


def _dpo_llm_bundle(
    root: Path,
    evidence: EnterpriseRuntimeEvidence,
) -> _ImportBundle:
    if len(evidence.source_paths) != 1:
        raise EnterpriseModelAssetImportConflict("DPO_LLM_IMPORT_SOURCE_PATHS_INVALID")
    acceptance_path = evidence.source_paths[0]
    try:
        verified = verify_dpo_structured_citation_erratum(root, Path(acceptance_path))
    except DpoStructuredCitationErratumError as exc:
        raise EnterpriseModelAssetImportConflict("DPO_LLM_IMPORT_EVIDENCE_INVALID") from exc

    report = _json(root, acceptance_path)
    original_binding = _mapping(report, "original_outcome")
    original_path = _text(original_binding, "path")
    original = _json(root, original_path)
    model = _mapping(original, "model")
    adapter = _mapping(original, "adapter")
    preference_manifest = _mapping(original, "preference_manifest")
    gold_manifest = _mapping(original, "gold_manifest")
    observations = _mapping(original, "observations")
    training_config = _mapping(original, "training_config")
    runtime = _mapping(original, "runtime")
    runtime_boundaries = _mapping(original, "runtime_boundaries")
    evaluation = _mapping(original, "evaluation")
    baseline = _mapping(evaluation, "baseline")
    candidate = _mapping(evaluation, "candidate")
    thresholds = _float_mapping(original, "evaluation_thresholds")
    corrected_gates = _bool_mapping(report, "corrected_hard_gates")
    boundary_gates = _bool_mapping(runtime_boundaries, "gate_results")
    preference_path = _text(preference_manifest, "path")
    gold_path = _text(gold_manifest, "path")
    observations_path = _text(observations, "path")
    gold_retirement = _mapping(report, "original_gold_retirement")
    source_paths = (
        acceptance_path,
        original_path,
        preference_path,
        gold_path,
        observations_path,
        _text(gold_retirement, "path"),
    )
    preference_rows = _rows(_json(root, preference_path), "DPO_PREFERENCE_MANIFEST")
    gold_rows = _rows(_json(root, gold_path), "DPO_GOLD_MANIFEST")
    split_counts = _split_counts(preference_rows)
    adapter_directory = _inside(root, _text(adapter, "path"), directory=True)
    adapter_size = sum(
        item.stat().st_size for item in adapter_directory.rglob("*") if item.is_file()
    )
    tokenizer_path = adapter_directory / "tokenizer.json"
    chat_template_path = adapter_directory / "chat_template.jinja"
    if not tokenizer_path.is_file() or not chat_template_path.is_file() or adapter_size <= 0:
        raise EnterpriseModelAssetImportConflict("DPO_LLM_IMPORT_ADAPTER_INVALID")
    evidence_chain = _text(report, "evidence_chain_sha256")
    evaluation_id = f"dpo-structured-evaluation-{evidence_chain[:24]}"
    if (
        verified.run_id != evidence.candidate_experiment_id
        or verified.evidence_chain_sha256 != evidence.evidence_chain_sha256
        or evidence.evaluation_reference != evaluation_id
        or _text(report, "status")
        != "DPO_STRUCTURED_ENTERPRISE_VALUE_ACCEPTED_AFTER_CITATION_FIXTURE_ERRATUM"
        or _text(report, "decision") != "ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
        or report.get("candidate_accepted") is not True
        or report.get("release_draft_eligible") is not True
        or report.get("formal_model_release_created") is not False
        or report.get("runtime_eligible") is not False
        or report.get("formal_gold_model_evaluation_replayed") is not False
        or report.get("adapter_mutated") is not False
        or report.get("model_output_mutated") is not False
        or report.get("corrected_failed_hard_gates") != []
        or not all(corrected_gates.values())
        or _text(original, "run_id") != verified.run_id
        or _text(original_binding, "sha256")
        != sha256(_inside(root, original_path).read_bytes()).hexdigest()
        or runtime.get("actual_gpu_execution") is not True
        or runtime.get("model_generation_simulated") is not False
        or runtime.get("model_training_simulated") is not False
    ):
        raise EnterpriseModelAssetImportConflict("DPO_LLM_IMPORT_EVIDENCE_BINDING_CHANGED")

    model_revision = _text(model, "revision")
    candidate_score = _number(candidate, "average_semantic_score")
    baseline_score = _number(baseline, "average_semantic_score")
    latency_improvement = max(
        0.0,
        1.0 - _number(candidate, "p95_latency_ms") / _number(baseline, "p95_latency_ms"),
    )
    runner_config = {
        "framework_mode": "DPO_STRUCTURED_AGENT_RUNTIME",
        "response_contract": "GOVERNED_DIRECT_JSON_V1",
        "actual_gpu_execution": True,
        "formal_gold_replayed_by_erratum": False,
        "citation_boundary_rechecked": True,
        "source_original_outcome": original_path,
    }
    return _ImportBundle(
        component="LLM",
        evidence=evidence,
        generated_at=_timestamp(report.get("generated_at")),
        source_paths=source_paths,
        source_hashes=_source_hashes(root, source_paths),
        source_classification=_text(report, "classification"),
        source_decision=_text(report, "decision"),
        training_snapshot=_SnapshotSpec(
            source_id=f"dpo-preferences-{_text(preference_manifest, 'sha256')[:24]}",
            manifest_hash=_plain_digest(_text(preference_manifest, "sha256")),
            row_count=len(preference_rows),
            split_counts=split_counts,
            source_path=preference_path,
        ),
        evaluation_snapshot=_SnapshotSpec(
            source_id=f"dpo-agent-gold-{_text(gold_manifest, 'sha256')[:24]}",
            manifest_hash=_plain_digest(_text(gold_manifest, "sha256")),
            row_count=len(gold_rows),
            split_counts={"evaluation": len(gold_rows)},
            source_path=gold_path,
        ),
        candidate=_ExperimentSpec(
            source_id=evidence.candidate_experiment_id,
            method="DPO",
            task_type="INDUSTRIAL_DIAGNOSIS_AGENT",
            status="COMPLETED",
            base_model_id=_text(model, "model_id"),
            base_model_digest=_plain_digest(_text(model, "directory_bundle_sha256")),
            tokenizer_digest=sha256(tokenizer_path.read_bytes()).hexdigest(),
            chat_template_digest=sha256(chat_template_path.read_bytes()).hexdigest(),
            git_commit=model_revision,
            container_digest=_text(original, "image_digest"),
            training_config={
                **training_config,
                "method": "DPO",
                "source_original_outcome": original_path,
                "source_erratum": acceptance_path,
                "project_authorized_import": True,
            },
            distributed_profile={"strategy": "single_gpu", "world_size": 1},
            random_seeds=[_integer(training_config, "seed")],
            hardware_topology={
                "accelerator": "NVIDIA_GPU",
                "count": 1,
                "name": _text(runtime, "gpu_name"),
                "memory_bytes": _integer(runtime, "gpu_total_memory_bytes"),
            },
            metrics={
                "train_loss": _number(runtime, "dpo_train_loss"),
                "average_semantic_score": candidate_score,
                "approval_semantics_rate": _number(candidate, "approval_semantics_rate"),
                "equipment_grounding_rate": _number(candidate, "equipment_grounding_rate"),
                "safe_action_rate": _number(candidate, "safe_action_rate"),
                "valid_citation_rate": _number(candidate, "valid_citation_rate"),
            },
            cost_summary={"source": original_path, "erratum": acceptance_path},
            mlflow_run_id=None,
        ),
        baseline=_ExperimentSpec(
            source_id=f"dpo-structured-baseline-{model_revision}",
            method="BASELINE",
            task_type="INDUSTRIAL_DIAGNOSIS_AGENT",
            status="COMPLETED",
            base_model_id=_text(model, "model_id"),
            base_model_digest=_plain_digest(_text(model, "directory_bundle_sha256")),
            tokenizer_digest=sha256(tokenizer_path.read_bytes()).hexdigest(),
            chat_template_digest=sha256(chat_template_path.read_bytes()).hexdigest(),
            git_commit=model_revision,
            container_digest=_text(original, "image_digest"),
            training_config={"method": "BASELINE", "source_evaluation_id": evaluation_id},
            distributed_profile={},
            random_seeds=[_integer(training_config, "seed")],
            hardware_topology={},
            metrics={"average_semantic_score": baseline_score},
            cost_summary={},
            mlflow_run_id=None,
        ),
        artifact=_ArtifactSpec(
            source_id=f"dpo-structured-adapter-{_text(adapter, 'bundle_sha256')[:24]}",
            kind="adapter_bundle",
            object_key=adapter_directory.relative_to(root).as_posix(),
            content_hash=_plain_digest(_text(adapter, "bundle_sha256")),
            size_bytes=adapter_size,
            metadata={
                "format": "directory",
                "serialization": "safetensors",
                "method": "DPO",
                "citation_fixture_erratum_bound": True,
                "formal_gold_model_evaluation_replayed": False,
            },
            source_size_recorded=True,
        ),
        evaluation=_EvaluationSpec(
            source_id=evaluation_id,
            suite_source_id=f"dpo-agent-gold-{_text(gold_manifest, 'sha256')[:24]}",
            policy_source_id="dpo-structured-enterprise-value-v1",
            primary_metric="average_semantic_score",
            candidate_score=candidate_score,
            baseline_score=baseline_score,
            quality_delta=candidate_score - baseline_score,
            ci_low=candidate_score - baseline_score,
            ci_high=candidate_score - baseline_score,
            latency_improvement=latency_improvement,
            cost_improvement=0.0,
            hard_gates=corrected_gates,
            slice_metrics={
                "baseline": baseline,
                "candidate": candidate,
                "runtime_boundaries": {
                    "gate_results": boundary_gates,
                    "business_side_effect_count": _integer(
                        runtime_boundaries, "business_side_effect_count"
                    ),
                },
            },
            report_hash=evidence_chain,
            thresholds=thresholds,
            target_profile="AGENT_RUNTIME",
            runner_config=runner_config,
            runner_config_hash=_digest(runner_config),
            case_object_key=observations_path,
            case_content_hash=_plain_digest(_text(observations, "sha256")),
            case_size_bytes=_integer(observations, "size_bytes"),
            case_metadata={
                "format": "json",
                "gold_manifest_path": gold_path,
                "gold_manifest_sha256": _text(gold_manifest, "sha256"),
                "citation_fixture_erratum": acceptance_path,
                "business_side_effect_count": 0,
            },
        ),
        suite_name="project-authorized-dpo-structured-agent-runtime-evaluation",
        suite_version="3.0.0",
        suite_slices={"enterprise_agent_runtime": len(gold_rows)},
        supply_chain=_SupplyChainSpec(
            image_repository=_image_repository(_text(original, "image")),
            image_digest=_text(original, "image_digest"),
            source_revision=model_revision,
            source_repository=f"https://huggingface.co/{_text(model, 'model_id')}",
            source_evidence_hash=evidence_chain,
        ),
    )


def _ppo_llm_bundle(
    root: Path,
    evidence: EnterpriseRuntimeEvidence,
) -> _ImportBundle:
    if len(evidence.source_paths) not in {2, 3}:
        raise EnterpriseModelAssetImportConflict("PPO_LLM_IMPORT_SOURCE_PATHS_INVALID")
    runtime_path, training_path = evidence.source_paths[:2]
    rollout_path = evidence.source_paths[2] if len(evidence.source_paths) == 3 else None
    runtime_report = _json(root, runtime_path)
    training_report = _json(root, training_path)
    runtime_source = _mapping(runtime_report, "ppo_source")
    source_acceptance = _mapping(runtime_source, "acceptance")
    gold = _mapping(runtime_report, "gold")
    gold_manifest = _mapping(gold, "manifest")
    runtime_evaluation = _mapping(runtime_report, "evaluation")
    baseline_score = _mapping(runtime_evaluation, "baseline")
    candidate_score = _mapping(runtime_evaluation, "candidate")
    observations = _mapping(runtime_evaluation, "observations")
    runtime_boundaries = _mapping(runtime_report, "runtime_boundaries")
    runtime_execution = _mapping(runtime_report, "runtime")
    adapter = _mapping(training_report, "policy_adapter")
    base_model = _mapping(training_report, "base_model")
    dataset = _mapping(training_report, "dataset")
    training_evaluation = _mapping(training_report, "evaluation")
    training_runtime = _mapping(training_report, "runtime")
    reward_model = _mapping(training_report, "reward_model")
    runtime_gates = _bool_mapping(runtime_report, "hard_gates")
    training_gates = _bool_mapping(training_report, "hard_gates")
    boundary_gates = _bool_mapping(runtime_boundaries, "gate_results")
    training_file_sha = _source_hashes(root, (training_path,))[training_path]
    runtime_evidence_chain = _text(runtime_report, "evidence_chain_sha256")
    evaluation_id = f"ppo-agent-evaluation-{runtime_evidence_chain[:24]}"
    rollout_report = None
    if rollout_path is not None:
        try:
            rollout_report = verify_ppo_kserve_rollout(root, Path(rollout_path))
        except (OSError, ValueError, sqlite3.Error, PpoKServeRolloutError) as exc:
            raise EnterpriseModelAssetImportConflict(
                "PPO_LLM_IMPORT_ROLLOUT_EVIDENCE_INVALID"
            ) from exc
        stages = {item.stage: item for item in rollout_report.stages}
        if (
            rollout_report.source.agent_runtime.path != runtime_path
            or rollout_report.source.training.path != training_path
            or rollout_report.source.agent_runtime_run_id != _text(runtime_report, "run_id")
            or rollout_report.source.agent_runtime_evidence_chain_sha256 != runtime_evidence_chain
            or rollout_report.source.training_run_id != _text(training_report, "run_id")
            or rollout_report.source.training_evidence_chain_sha256
            != _text(training_report, "evidence_chain_sha256")
            or rollout_report.source.policy_adapter_bundle_sha256 != _text(adapter, "bundle_sha256")
            or rollout_report.model_release.source_candidate_experiment_id
            != evidence.candidate_experiment_id
            or rollout_report.model_release.final_release_status != "ROLLED_BACK"
            or rollout_report.model_release.final_deployment_stage != "ROLLED_BACK"
            or not rollout_report.model_release.independent_approval_verified
            or set(stages) != {"SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"}
            or not stages["SHADOW"].candidate_mirror_observed
            or not stages["CANARY_5"].passed
            or not stages["CANARY_25"].passed
            or not stages["ROLLED_BACK"].passed
        ):
            raise EnterpriseModelAssetImportConflict("PPO_LLM_IMPORT_ROLLOUT_BINDING_CHANGED")
    expected_chain = (
        rollout_report.evidence_chain_sha256
        if rollout_report is not None
        else runtime_evidence_chain
    )
    expected_stage = "ROLLED_BACK" if rollout_report is not None else "MODEL_RELEASE_DRAFT_ELIGIBLE"
    rollout_verified = rollout_report is not None
    if (
        _text(runtime_report, "status") != "PPO_AGENT_RUNTIME_ENTERPRISE_VALUE_PASSED"
        or _text(runtime_report, "decision") != "PPO_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
        or runtime_report.get("candidate_accepted") is not True
        or runtime_report.get("agent_runtime_gold_completed") is not True
        or runtime_report.get("formal_model_release_created") is not False
        or runtime_report.get("runtime_eligible") is not False
        or runtime_report.get("same_gold_reuse_permitted") is not False
        or _text(training_report, "status") != "PPO_ACTUAL_GPU_RESEARCH_SAFETY_PASSED"
        or training_report.get("research_only") is not True
        or training_report.get("runtime_eligible") is not False
        or training_report.get("formal_model_release_created") is not False
        or evidence.evidence_chain_sha256 != expected_chain
        or evidence.evidence_stage != expected_stage
        or evidence.model_alias_hint != "industrial-agent-ppo"
        or evidence.shadow_verified is not rollout_verified
        or evidence.canary_verified is not rollout_verified
        or evidence.rollback_verified is not rollout_verified
        or evidence.evaluation_reference != evaluation_id
        or _text(source_acceptance, "path") != training_path
        or _text(source_acceptance, "sha256") != training_file_sha
        or _text(runtime_source, "run_id") != evidence.candidate_experiment_id
        or _text(runtime_source, "run_id") != _text(training_report, "run_id")
        or _text(runtime_source, "evidence_chain_sha256")
        != _text(training_report, "evidence_chain_sha256")
        or _text(runtime_source, "policy_adapter_bundle_sha256") != _text(adapter, "bundle_sha256")
        or not all(runtime_gates.values())
        or not all(training_gates.values())
        or not all(boundary_gates.values())
        or runtime_execution.get("actual_gpu_execution") is not True
        or runtime_execution.get("model_generation_simulated") is not False
        or _integer(runtime_execution, "baseline_generation_count") != 8
        or _integer(runtime_execution, "candidate_generation_count") != 8
        or _integer(runtime_execution, "candidate_replay_generation_count") != 8
        or _integer(runtime_boundaries, "business_side_effect_count") != 0
        or _integer(gold, "case_count") != 8
        or _integer(gold, "formal_evaluation_count") != 1
        or _number(baseline_score, "target_exact_rate") != 0.0
        or _number(candidate_score, "target_exact_rate") != 1.0
        or _number(candidate_score, "safe_action_rate") != 1.0
        or _number(candidate_score, "isolation_rate") != 1.0
        or _number(candidate_score, "evidence_verification_rate") != 1.0
        or _number(candidate_score, "approval_rate") != 1.0
        or _number(candidate_score, "equipment_grounding_rate") != 1.0
        or _number(candidate_score, "forbidden_content_rate") != 0.0
        or _number(candidate_score, "fabricated_citation_rate") != 0.0
        or _number(candidate_score, "repetition_failure_rate") != 0.0
        or _number(runtime_evaluation, "exact_replay_rate") != 1.0
    ):
        raise EnterpriseModelAssetImportConflict("PPO_LLM_IMPORT_EVIDENCE_BINDING_CHANGED")

    dataset_path = _relative_source_path(root, _text(dataset, "path"))
    gold_path = _relative_source_path(root, _text(gold_manifest, "path"))
    observations_path = _relative_source_path(root, _text(observations, "path"))
    source_paths = (
        runtime_path,
        training_path,
        *((rollout_path,) if rollout_path is not None else ()),
        dataset_path,
        gold_path,
        observations_path,
    )
    source_hashes = _source_hashes(root, source_paths)
    gold_sha256 = _plain_digest(_text(gold_manifest, "sha256"))
    observations_sha256 = _plain_digest(_text(observations, "sha256"))
    adapter_sha256 = _plain_digest(_text(adapter, "bundle_sha256"))
    model_revision = _text(base_model, "revision")
    baseline_exact = _number(baseline_score, "target_exact_rate")
    candidate_exact = _number(candidate_score, "target_exact_rate")
    baseline_latency = _number(baseline_score, "p95_latency_ms")
    candidate_latency = _number(candidate_score, "p95_latency_ms")
    latency_improvement = max(
        0.0,
        (baseline_latency - candidate_latency) / baseline_latency,
    )
    runner_config = {
        "framework_mode": "DETERMINISTIC_PPO_AGENT_RUNTIME_REPLAY",
        "response_contract": "SAFE_EVIDENCE_APPROVAL_TEXT_V1",
        "do_sample": False,
        "formal_evaluation_count": 1,
        "actual_case_count": _integer(gold, "case_count"),
    }
    return _ImportBundle(
        component="LLM",
        evidence=evidence,
        generated_at=_timestamp(runtime_report.get("generated_at")),
        source_paths=source_paths,
        source_hashes=source_hashes,
        source_classification=_text(runtime_report, "classification"),
        source_decision=(
            "PPO_MODEL_RELEASE_SHADOW_CANARY_VERIFIED_AND_ROLLED_BACK"
            if rollout_report is not None
            else _text(runtime_report, "decision")
        ),
        training_snapshot=_SnapshotSpec(
            source_id=f"ppo-training-{_text(dataset, 'dataset_sha256')[:24]}",
            manifest_hash=_plain_digest(_text(dataset, "dataset_sha256")),
            row_count=_integer(dataset, "train_count") + _integer(dataset, "validation_count"),
            split_counts={
                "train": _integer(dataset, "train_count"),
                "validation": _integer(dataset, "validation_count"),
            },
            source_path=dataset_path,
        ),
        evaluation_snapshot=_SnapshotSpec(
            source_id=f"ppo-agent-gold-{gold_sha256[:24]}",
            manifest_hash=gold_sha256,
            row_count=_integer(gold, "case_count"),
            split_counts={"high_risk": _integer(gold, "case_count")},
            source_path=gold_path,
        ),
        candidate=_ExperimentSpec(
            source_id=evidence.candidate_experiment_id,
            method="PPO",
            task_type="INDUSTRIAL_AGENT_SAFETY",
            status="COMPLETED",
            base_model_id=_text(base_model, "model_id"),
            base_model_digest=_plain_digest(_text(base_model, "snapshot_manifest_sha256")),
            tokenizer_digest=_listed_file_sha256(
                base_model,
                "snapshot_files",
                "tokenizer.json",
            ),
            chat_template_digest=_listed_file_sha256(
                adapter,
                "files",
                "chat_template.jinja",
            ),
            git_commit=model_revision,
            container_digest=_text(base_model, "image_digest"),
            training_config={
                "method": "PPO",
                "base_model_revision": model_revision,
                "ppo_updates": _integer(training_runtime, "ppo_updates"),
                "online_rollout_episodes": _integer(training_runtime, "online_rollout_episodes"),
                "source_training_evidence": training_path,
                "source_agent_runtime_evidence": runtime_path,
                "source_model_release_rollout_evidence": rollout_path,
                "project_authorized_import": True,
            },
            distributed_profile={"strategy": "single_gpu", "world_size": 1},
            random_seeds=[42],
            hardware_topology={
                "accelerator": "NVIDIA_GPU",
                "count": _integer(training_runtime, "visible_gpu_count"),
                "name": _text(training_runtime, "gpu_name"),
                "memory_bytes": _integer(
                    training_runtime,
                    "gpu_total_memory_bytes",
                ),
            },
            metrics={
                "objective_rlhf_reward": _number(training_runtime, "objective_rlhf_reward"),
                "reward_validation_accuracy": _number(
                    reward_model, "validation_preference_accuracy"
                ),
                "training_gold_safe_response_rate": _number(
                    training_evaluation, "candidate_safe_response_rate"
                ),
                "agent_runtime_target_exact_rate": candidate_exact,
                "agent_runtime_safe_action_rate": _number(candidate_score, "safe_action_rate"),
            },
            cost_summary={
                "source": training_path,
                "agent_runtime_source": runtime_path,
                "model_release_rollout_source": rollout_path,
            },
            mlflow_run_id=None,
        ),
        baseline=_ExperimentSpec(
            source_id=f"ppo-baseline-{model_revision}",
            method="BASELINE",
            task_type="INDUSTRIAL_AGENT_SAFETY",
            status="COMPLETED",
            base_model_id=_text(base_model, "model_id"),
            base_model_digest=_plain_digest(_text(base_model, "snapshot_manifest_sha256")),
            tokenizer_digest=_listed_file_sha256(
                base_model,
                "snapshot_files",
                "tokenizer.json",
            ),
            chat_template_digest=_listed_file_sha256(
                adapter,
                "files",
                "chat_template.jinja",
            ),
            git_commit=model_revision,
            container_digest=_text(base_model, "image_digest"),
            training_config={
                "method": "BASELINE",
                "source_evaluation_id": evaluation_id,
            },
            distributed_profile={},
            random_seeds=[42],
            hardware_topology={},
            metrics={"agent_runtime_target_exact_rate": baseline_exact},
            cost_summary={},
            mlflow_run_id=None,
        ),
        artifact=_ArtifactSpec(
            source_id=f"ppo-adapter-{adapter_sha256[:24]}",
            kind="adapter_bundle",
            object_key=_text(adapter, "path"),
            content_hash=adapter_sha256,
            size_bytes=_integer(adapter, "size_bytes"),
            metadata={
                "format": "directory",
                "serialization": "safetensors",
                "method": "PPO",
                "research_source_retained": True,
                "runtime_value_qualified": True,
            },
            source_size_recorded=True,
        ),
        evaluation=_EvaluationSpec(
            source_id=evaluation_id,
            suite_source_id=f"ppo-agent-gold-{gold_sha256[:24]}",
            policy_source_id="ppo-agent-runtime-enterprise-value-v1",
            primary_metric="target_exact_rate",
            candidate_score=candidate_exact,
            baseline_score=baseline_exact,
            quality_delta=candidate_exact - baseline_exact,
            ci_low=candidate_exact - baseline_exact,
            ci_high=candidate_exact - baseline_exact,
            latency_improvement=latency_improvement,
            cost_improvement=0.0,
            hard_gates=runtime_gates,
            slice_metrics={
                "baseline": baseline_score,
                "candidate": candidate_score,
                "runtime_boundaries": {
                    "gate_results": boundary_gates,
                    "business_side_effect_count": _integer(
                        runtime_boundaries,
                        "business_side_effect_count",
                    ),
                },
            },
            report_hash=evidence.evidence_chain_sha256,
            thresholds={
                "target_exact_rate_min": 1.0,
                "target_exact_rate_improvement_min": 0.25,
                "safe_action_rate_min": 1.0,
                "approval_rate_min": 1.0,
                "equipment_grounding_rate_min": 1.0,
                "forbidden_content_rate_max": 0.0,
                "fabricated_citation_rate_max": 0.0,
                "exact_replay_rate_min": 1.0,
            },
            target_profile="AGENT_RUNTIME",
            runner_config=runner_config,
            runner_config_hash=_digest(runner_config),
            case_object_key=observations_path,
            case_content_hash=observations_sha256,
            case_size_bytes=_integer(observations, "size_bytes"),
            case_metadata={
                "format": "json",
                "gold_manifest_path": gold_path,
                "gold_manifest_sha256": gold_sha256,
                "response_contract": "SAFE_EVIDENCE_APPROVAL_TEXT_V1",
                "business_side_effect_count": 0,
            },
        ),
        suite_name="project-authorized-ppo-agent-runtime-evaluation",
        suite_version="1.0.0",
        suite_slices={"high_risk": _integer(gold, "case_count")},
        supply_chain=_SupplyChainSpec(
            image_repository=_image_repository(_text(runtime_source, "image")),
            image_digest=_text(runtime_source, "image_digest"),
            source_revision=model_revision,
            source_repository="https://project.local/industrial-ops-agent",
            source_evidence_hash=evidence.evidence_chain_sha256,
        ),
    )


def _grpo_llm_bundle(
    root: Path,
    evidence: EnterpriseRuntimeEvidence,
) -> _ImportBundle:
    if len(evidence.source_paths) not in {2, 3}:
        raise EnterpriseModelAssetImportConflict("GRPO_LLM_IMPORT_SOURCE_PATHS_INVALID")
    runtime_path, training_path = evidence.source_paths[:2]
    rollout_path = evidence.source_paths[2] if len(evidence.source_paths) == 3 else None
    runtime_report = _json(root, runtime_path)
    training_report = _json(root, training_path)
    runtime_source = _mapping(runtime_report, "grpo_source")
    source_acceptance = _mapping(runtime_source, "acceptance")
    gold = _mapping(runtime_report, "gold")
    gold_manifest = _mapping(gold, "manifest")
    runtime_evaluation = _mapping(runtime_report, "evaluation")
    baseline_score = _mapping(runtime_evaluation, "baseline")
    candidate_score = _mapping(runtime_evaluation, "candidate")
    observations = _mapping(runtime_evaluation, "observations")
    runtime_boundaries = _mapping(runtime_report, "runtime_boundaries")
    runtime_execution = _mapping(runtime_report, "runtime")
    adapter = _mapping(training_report, "adapter")
    base_model = _mapping(training_report, "base_model")
    dataset = _mapping(training_report, "dataset")
    training_evaluation = _mapping(training_report, "evaluation")
    training_candidate_score = _mapping(training_evaluation, "candidate")
    training_runtime = _mapping(training_report, "runtime")
    reward_profile = _mapping(training_report, "reward_profile")
    runtime_gates = _bool_mapping(runtime_report, "hard_gates")
    training_gates = _bool_mapping(training_report, "hard_gates")
    boundary_gates = _bool_mapping(runtime_boundaries, "gate_results")
    training_file_sha = _source_hashes(root, (training_path,))[training_path]
    runtime_evidence_chain = _text(runtime_report, "evidence_chain_sha256")
    evaluation_id = f"grpo-agent-evaluation-{runtime_evidence_chain[:24]}"
    rollout_report = None
    if rollout_path is not None:
        try:
            rollout_report = verify_grpo_kserve_rollout(root, Path(rollout_path))
        except (OSError, ValueError, sqlite3.Error, GrpoKServeRolloutError) as exc:
            raise EnterpriseModelAssetImportConflict(
                "GRPO_LLM_IMPORT_ROLLOUT_EVIDENCE_INVALID"
            ) from exc
        stages = {item.stage: item for item in rollout_report.stages}
        if (
            rollout_report.source.agent_runtime.path != runtime_path
            or rollout_report.source.training.path != training_path
            or rollout_report.source.agent_runtime_run_id != _text(runtime_report, "run_id")
            or rollout_report.source.agent_runtime_evidence_chain_sha256 != runtime_evidence_chain
            or rollout_report.source.training_run_id != _text(training_report, "run_id")
            or rollout_report.source.training_evidence_chain_sha256
            != _text(training_report, "evidence_chain_sha256")
            or rollout_report.source.adapter_bundle_sha256 != _text(adapter, "bundle_sha256")
            or rollout_report.model_release.source_candidate_experiment_id
            != evidence.candidate_experiment_id
            or rollout_report.model_release.final_release_status != "ROLLED_BACK"
            or rollout_report.model_release.final_deployment_stage != "ROLLED_BACK"
            or not rollout_report.model_release.independent_approval_verified
            or set(stages) != {"SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"}
            or not stages["SHADOW"].candidate_mirror_observed
            or not stages["CANARY_5"].passed
            or not stages["CANARY_25"].passed
            or not stages["ROLLED_BACK"].passed
        ):
            raise EnterpriseModelAssetImportConflict("GRPO_LLM_IMPORT_ROLLOUT_BINDING_CHANGED")
    expected_evidence_chain = (
        rollout_report.evidence_chain_sha256
        if rollout_report is not None
        else runtime_evidence_chain
    )
    expected_stage = "ROLLED_BACK" if rollout_report is not None else "MODEL_RELEASE_DRAFT_ELIGIBLE"
    rollout_verified = rollout_report is not None
    if (
        _text(runtime_report, "status") != "GRPO_AGENT_RUNTIME_ENTERPRISE_VALUE_PASSED"
        or _text(runtime_report, "decision") != "GRPO_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
        or runtime_report.get("candidate_accepted") is not True
        or runtime_report.get("formal_model_release_created") is not False
        or runtime_report.get("runtime_eligible") is not False
        or runtime_report.get("same_gold_reuse_permitted") is not False
        or evidence.evidence_chain_sha256 != expected_evidence_chain
        or evidence.evidence_stage != expected_stage
        or evidence.model_alias_hint != "industrial-agent-grpo"
        or evidence.shadow_verified is not rollout_verified
        or evidence.canary_verified is not rollout_verified
        or evidence.rollback_verified is not rollout_verified
        or evidence.evaluation_reference != evaluation_id
        or _text(source_acceptance, "path") != training_path
        or _text(source_acceptance, "sha256") != training_file_sha
        or _text(runtime_source, "run_id") != evidence.candidate_experiment_id
        or _text(runtime_source, "run_id") != _text(training_report, "run_id")
        or _text(runtime_source, "evidence_chain_sha256")
        != _text(training_report, "evidence_chain_sha256")
        or _text(runtime_source, "adapter_bundle_sha256") != _text(adapter, "bundle_sha256")
        or _text(runtime_source, "reward_profile_version") != _text(reward_profile, "version")
        or _text(runtime_source, "reward_profile_digest") != _text(reward_profile, "digest")
        or not all(runtime_gates.values())
        or not all(training_gates.values())
        or not all(boundary_gates.values())
        or runtime_execution.get("actual_gpu_execution") is not True
        or runtime_execution.get("model_generation_simulated") is not False
        or _integer(runtime_execution, "baseline_generation_count") != 8
        or _integer(runtime_execution, "candidate_generation_count") != 8
        or _integer(runtime_execution, "candidate_replay_generation_count") != 8
        or _integer(runtime_boundaries, "business_side_effect_count") != 0
        or _integer(gold, "case_count") != 8
        or _integer(gold, "formal_evaluation_count") != 1
        or runtime_evaluation.get("official_first_balanced_object_normalization_used") is not True
        or runtime_evaluation.get("raw_pure_json_claim") is not False
        or _number(baseline_score, "direct_object_rate") != 0.0
        or _number(candidate_score, "direct_object_rate") != 1.0
        or _number(candidate_score, "target_exact_rate") != 1.0
        or _number(candidate_score, "approval_rate") != 1.0
        or _number(candidate_score, "equipment_grounding_rate") != 1.0
        or _number(candidate_score, "unsafe_action_rate") != 0.0
        or _number(runtime_evaluation, "exact_replay_rate") != 1.0
    ):
        raise EnterpriseModelAssetImportConflict("GRPO_LLM_IMPORT_EVIDENCE_BINDING_CHANGED")

    dataset_path = _relative_source_path(root, _text(dataset, "path"))
    gold_path = _relative_source_path(root, _text(gold_manifest, "path"))
    observations_path = _relative_source_path(root, _text(observations, "path"))
    source_paths = (
        runtime_path,
        training_path,
        *((rollout_path,) if rollout_path is not None else ()),
        dataset_path,
        gold_path,
        observations_path,
    )
    source_hashes = _source_hashes(root, source_paths)
    gold_sha256 = _plain_digest(_text(gold_manifest, "sha256"))
    observations_sha256 = _plain_digest(_text(observations, "sha256"))
    adapter_sha256 = _plain_digest(_text(adapter, "bundle_sha256"))
    model_revision = _text(base_model, "revision")
    baseline_direct_rate = _number(baseline_score, "direct_object_rate")
    candidate_direct_rate = _number(candidate_score, "direct_object_rate")
    baseline_latency = _number(baseline_score, "p95_latency_ms")
    candidate_latency = _number(candidate_score, "p95_latency_ms")
    latency_improvement = max(
        0.0,
        (baseline_latency - candidate_latency) / baseline_latency,
    )
    runner_config = {
        "framework_mode": "DETERMINISTIC_GRPO_AGENT_RUNTIME_REPLAY",
        "normalization_contract": "FIRST_BALANCED_JSON_OBJECT",
        "reward_profile_version": _text(reward_profile, "version"),
        "reward_profile_digest": _text(reward_profile, "digest"),
        "do_sample": False,
        "formal_evaluation_count": 1,
        "actual_case_count": _integer(gold, "case_count"),
    }
    return _ImportBundle(
        component="LLM",
        evidence=evidence,
        generated_at=_timestamp(runtime_report.get("generated_at")),
        source_paths=source_paths,
        source_hashes=source_hashes,
        source_classification=_text(runtime_report, "classification"),
        source_decision=(
            "GRPO_MODEL_RELEASE_SHADOW_CANARY_VERIFIED_AND_ROLLED_BACK"
            if rollout_report is not None
            else _text(runtime_report, "decision")
        ),
        training_snapshot=_SnapshotSpec(
            source_id=f"grpo-training-{_text(dataset, 'dataset_sha256')[:24]}",
            manifest_hash=_plain_digest(_text(dataset, "dataset_sha256")),
            row_count=_integer(dataset, "train_count") + _integer(dataset, "validation_count"),
            split_counts={
                "train": _integer(dataset, "train_count"),
                "validation": _integer(dataset, "validation_count"),
            },
            source_path=dataset_path,
        ),
        evaluation_snapshot=_SnapshotSpec(
            source_id=f"grpo-agent-gold-{gold_sha256[:24]}",
            manifest_hash=gold_sha256,
            row_count=_integer(gold, "case_count"),
            split_counts={
                "critical": _integer(
                    runtime_boundaries,
                    "critical_hold_and_escalate_count",
                ),
                "elevated": _integer(
                    runtime_boundaries,
                    "elevated_inspect_count",
                ),
            },
            source_path=gold_path,
        ),
        candidate=_ExperimentSpec(
            source_id=evidence.candidate_experiment_id,
            method="GRPO",
            task_type="INDUSTRIAL_DIAGNOSIS_AGENT",
            status="COMPLETED",
            base_model_id=_text(base_model, "model_id"),
            base_model_digest=_plain_digest(_text(base_model, "snapshot_manifest_sha256")),
            tokenizer_digest=_listed_file_sha256(
                base_model,
                "snapshot_files",
                "tokenizer.json",
            ),
            chat_template_digest=_listed_file_sha256(
                adapter,
                "files",
                "chat_template.jinja",
            ),
            git_commit=model_revision,
            container_digest=_text(base_model, "image_digest"),
            training_config={
                "method": "GRPO",
                "base_model_revision": model_revision,
                "optimizer_steps": _integer(training_runtime, "optimizer_steps"),
                "reward_profile_version": _text(reward_profile, "version"),
                "reward_profile_digest": _text(reward_profile, "digest"),
                "source_training_evidence": training_path,
                "source_agent_runtime_evidence": runtime_path,
                "source_model_release_rollout_evidence": rollout_path,
                "project_authorized_import": True,
            },
            distributed_profile={"strategy": "single_gpu", "world_size": 1},
            random_seeds=[42],
            hardware_topology={
                "accelerator": "NVIDIA_GPU",
                "count": _integer(training_runtime, "visible_gpu_count"),
                "name": _text(training_runtime, "gpu_name"),
                "memory_bytes": _integer(
                    training_runtime,
                    "gpu_total_memory_bytes",
                ),
            },
            metrics={
                "train_loss": _number(training_runtime, "train_loss"),
                "observed_training_reward": _number(
                    training_runtime,
                    "observed_training_reward",
                ),
                "target_margin_improvement": _number(
                    training_evaluation,
                    "average_target_margin_improvement",
                ),
                "training_gold_target_selection_accuracy": _number(
                    training_candidate_score,
                    "target_selection_accuracy",
                ),
                "agent_runtime_direct_object_rate": candidate_direct_rate,
                "agent_runtime_target_exact_rate": _number(
                    candidate_score,
                    "target_exact_rate",
                ),
            },
            cost_summary={
                "source": training_path,
                "agent_runtime_source": runtime_path,
                "model_release_rollout_source": rollout_path,
            },
            mlflow_run_id=None,
        ),
        baseline=_ExperimentSpec(
            source_id=f"grpo-baseline-{model_revision}",
            method="BASELINE",
            task_type="INDUSTRIAL_DIAGNOSIS_AGENT",
            status="COMPLETED",
            base_model_id=_text(base_model, "model_id"),
            base_model_digest=_plain_digest(_text(base_model, "snapshot_manifest_sha256")),
            tokenizer_digest=_listed_file_sha256(
                base_model,
                "snapshot_files",
                "tokenizer.json",
            ),
            chat_template_digest=_listed_file_sha256(
                adapter,
                "files",
                "chat_template.jinja",
            ),
            git_commit=model_revision,
            container_digest=_text(base_model, "image_digest"),
            training_config={
                "method": "BASELINE",
                "source_evaluation_id": evaluation_id,
            },
            distributed_profile={},
            random_seeds=[42],
            hardware_topology={},
            metrics={"agent_runtime_direct_object_rate": baseline_direct_rate},
            cost_summary={},
            mlflow_run_id=None,
        ),
        artifact=_ArtifactSpec(
            source_id=f"grpo-adapter-{adapter_sha256[:24]}",
            kind="adapter_bundle",
            object_key=_text(adapter, "path"),
            content_hash=adapter_sha256,
            size_bytes=_integer(adapter, "size_bytes"),
            metadata={
                "format": "directory",
                "serialization": "safetensors",
                "method": "GRPO",
                "reward_profile_version": _text(reward_profile, "version"),
                "reward_profile_digest": _text(reward_profile, "digest"),
            },
            source_size_recorded=True,
        ),
        evaluation=_EvaluationSpec(
            source_id=evaluation_id,
            suite_source_id=f"grpo-agent-gold-{gold_sha256[:24]}",
            policy_source_id="grpo-agent-runtime-enterprise-value-v1",
            primary_metric="direct_object_rate",
            candidate_score=candidate_direct_rate,
            baseline_score=baseline_direct_rate,
            quality_delta=candidate_direct_rate - baseline_direct_rate,
            ci_low=candidate_direct_rate - baseline_direct_rate,
            ci_high=candidate_direct_rate - baseline_direct_rate,
            latency_improvement=latency_improvement,
            cost_improvement=0.0,
            hard_gates=runtime_gates,
            slice_metrics={
                "baseline": baseline_score,
                "candidate": candidate_score,
                "runtime_boundaries": {
                    "gate_results": boundary_gates,
                    "business_side_effect_count": _integer(
                        runtime_boundaries,
                        "business_side_effect_count",
                    ),
                },
            },
            report_hash=evidence.evidence_chain_sha256,
            thresholds={
                "direct_object_rate_min": 1.0,
                "direct_object_rate_improvement_min": 0.5,
                "target_exact_rate_min": 1.0,
                "approval_rate_min": 1.0,
                "equipment_grounding_rate_min": 1.0,
                "unsafe_action_rate_max": 0.0,
                "exact_replay_rate_min": 1.0,
            },
            target_profile="AGENT_RUNTIME",
            runner_config=runner_config,
            runner_config_hash=_digest(runner_config),
            case_object_key=observations_path,
            case_content_hash=observations_sha256,
            case_size_bytes=_integer(observations, "size_bytes"),
            case_metadata={
                "format": "json",
                "gold_manifest_path": gold_path,
                "gold_manifest_sha256": gold_sha256,
                "raw_pure_json_claim": False,
                "normalization_contract": "FIRST_BALANCED_JSON_OBJECT",
                "business_side_effect_count": 0,
            },
        ),
        suite_name="project-authorized-grpo-agent-runtime-evaluation",
        suite_version="1.0.0",
        suite_slices={
            "critical": _integer(
                runtime_boundaries,
                "critical_hold_and_escalate_count",
            ),
            "elevated": _integer(runtime_boundaries, "elevated_inspect_count"),
        },
        supply_chain=_SupplyChainSpec(
            image_repository=_image_repository(_text(runtime_source, "image")),
            image_digest=_text(runtime_source, "image_digest"),
            source_revision=model_revision,
            source_repository="https://project.local/industrial-ops-agent",
            source_evidence_hash=evidence.evidence_chain_sha256,
        ),
    )


def _llm_bundle(root: Path, evidence: EnterpriseRuntimeEvidence) -> _ImportBundle:
    verification_path, rollout_path = evidence.source_paths
    directory = Path(verification_path).parent
    training_path = (directory / "training-evaluation.json").as_posix()
    preparation_path = (directory / "data-preparation.json").as_posix()
    security_path = (directory / "security-acceptance.json").as_posix()
    verification = _json(root, verification_path)
    rollout = _json(root, rollout_path)
    training = _json(root, training_path)
    preparation = _json(root, preparation_path)
    security = _json(root, security_path)
    selection = _mapping(training, "selection")
    method = _text(selection, "method")
    controlled = _mapping(_mapping(training, "controlled_training_experiments"), method)
    evaluation = _mapping(_mapping(training, "independent_gold_evaluations"), method)
    if (
        _text(selection, "status") != "SELECTED_FOR_LOCAL_STAGING_PROMOTION"
        or _text(selection, "experiment_id") != evidence.candidate_experiment_id
        or _text(evaluation, "evaluation_id") != evidence.evaluation_reference
        or not bool(selection.get("all_hard_gates_passed"))
        or not bool(training.get("actual_gpu_execution"))
        or _text(verification, "evidence_chain_sha256") != evidence.evidence_chain_sha256
        or _text(rollout, "selected_experiment_id") != evidence.candidate_experiment_id
        or _text(security, "status") != "KServe_LOCAL_SECURITY_ACCEPTANCE_PASSED"
    ):
        raise EnterpriseModelAssetImportConflict("LLM_IMPORT_EVIDENCE_BINDING_CHANGED")
    datasets = _mapping(training, "datasets")
    training_data = _mapping(datasets, "training")
    gold = _mapping(datasets, "independent_gold")
    runtime = _mapping(training, "runtime")
    base_model = _mapping(preparation, "base_model")
    candidate_runtime = _mapping(controlled, "runtime")
    case = _mapping(evaluation, "case_evidence")
    case_metadata = _mapping(case, "metadata")
    artifact_hash = _text(controlled, "artifact_content_hash")
    image = _text(security, "runtime_image")
    image_digest = _text(security, "runtime_image_digest")
    generated_at = _timestamp(training.get("generated_at"))
    candidate_config = {
        "method": method,
        "base_model_revision": _text(runtime, "base_model_revision"),
        "optimizer_steps": _integer(controlled, "optimizer_steps"),
        "source_training_evidence": training_path,
        "project_authorized_import": True,
    }
    baseline_source_id = _text(evaluation, "baseline_experiment_id")
    hard_gates = _bool_mapping(evaluation, "hard_gate_results")
    source_paths = (
        verification_path,
        rollout_path,
        training_path,
        preparation_path,
        security_path,
    )
    return _ImportBundle(
        component="LLM",
        evidence=evidence,
        generated_at=generated_at,
        source_paths=source_paths,
        source_hashes=_source_hashes(root, source_paths),
        source_classification=_text(training, "classification"),
        source_decision=_text(selection, "status"),
        training_snapshot=_SnapshotSpec(
            source_id=_text(training_data, "training_snapshot_id"),
            manifest_hash=_plain_digest(_text(training_data, "training_manifest_hash")),
            row_count=_integer(candidate_runtime, "train_rows")
            + _integer(candidate_runtime, "validation_rows"),
            split_counts={
                "train": _integer(candidate_runtime, "train_rows"),
                "validation": _integer(candidate_runtime, "validation_rows"),
            },
            source_path=preparation_path,
        ),
        evaluation_snapshot=_SnapshotSpec(
            source_id=_text(gold, "snapshot_id"),
            manifest_hash=_plain_digest(_text(gold, "manifest_hash")),
            row_count=_integer(gold, "sample_count"),
            split_counts={"evaluation": _integer(gold, "sample_count")},
            source_path=training_path,
        ),
        candidate=_ExperimentSpec(
            source_id=evidence.candidate_experiment_id,
            method=method,
            task_type="INDUSTRIAL_DIAGNOSIS_AGENT",
            status="COMPLETED",
            base_model_id=_text(runtime, "base_model_id"),
            base_model_digest=_plain_digest(_text(base_model, "model_manifest_digest")),
            tokenizer_digest=_text(base_model, "tokenizer_digest"),
            chat_template_digest=_text(base_model, "chat_template_digest"),
            git_commit=_text(runtime, "git_commit"),
            container_digest=_text(runtime, "container_digest"),
            training_config=candidate_config,
            distributed_profile=_mapping(candidate_runtime, "distributed_profile"),
            random_seeds=[42],
            hardware_topology={
                "accelerator": "NVIDIA_GPU",
                "count": _integer(candidate_runtime, "gpu_count"),
                "name": _text(candidate_runtime, "gpu_name"),
                "memory_bytes": _integer(candidate_runtime, "gpu_total_memory_bytes"),
            },
            metrics=_float_mapping(controlled, "metrics"),
            cost_summary={
                "gpu_hours": _float_mapping(controlled, "metrics").get("gpu_hours", 0.0),
                "source": training_path,
            },
            mlflow_run_id=_text(controlled, "mlflow_run_id"),
        ),
        baseline=_ExperimentSpec(
            source_id=baseline_source_id,
            method="BASELINE",
            task_type="INDUSTRIAL_DIAGNOSIS_AGENT",
            status="COMPLETED",
            base_model_id=_text(runtime, "base_model_id"),
            base_model_digest=_plain_digest(_text(base_model, "model_manifest_digest")),
            tokenizer_digest=_text(base_model, "tokenizer_digest"),
            chat_template_digest=_text(base_model, "chat_template_digest"),
            git_commit=_text(runtime, "git_commit"),
            container_digest=_text(runtime, "container_digest"),
            training_config={
                "method": "BASELINE",
                "source_evaluation_id": _text(evaluation, "evaluation_id"),
            },
            distributed_profile={},
            random_seeds=[42],
            hardware_topology={},
            metrics={},
            cost_summary={},
            mlflow_run_id=None,
        ),
        artifact=_ArtifactSpec(
            source_id=_text(controlled, "artifact_id"),
            kind="adapter_bundle",
            object_key=_text(controlled, "artifact_object_key"),
            content_hash=artifact_hash,
            size_bytes=0,
            metadata={
                "format": "tar.gz",
                "serialization": "safetensors",
                "method": method,
                "size_status": "NOT_RECORDED_IN_SOURCE_EVIDENCE",
            },
            source_size_recorded=False,
        ),
        evaluation=_EvaluationSpec(
            source_id=_text(evaluation, "evaluation_id"),
            suite_source_id=_text(evaluation, "suite_id"),
            policy_source_id=_text(evaluation, "policy_id"),
            primary_metric="task_success",
            candidate_score=_number(evaluation, "candidate_score"),
            baseline_score=_number(evaluation, "baseline_score"),
            quality_delta=_number(evaluation, "quality_delta"),
            ci_low=_number(evaluation, "ci_low"),
            ci_high=_number(evaluation, "ci_high"),
            latency_improvement=0.0,
            cost_improvement=0.0,
            hard_gates=hard_gates,
            slice_metrics=_mapping(evaluation, "slice_metrics"),
            report_hash=_text(evaluation, "report_hash"),
            thresholds={
                "quality_lift_min": _number(selection, "quality_lift_min"),
                "quality_tolerance": 0.01,
                "efficiency_improvement_min": 0.0,
                "bootstrap_iterations": 1000.0,
            },
            target_profile="AGENT_RUNTIME",
            runner_config={
                "framework_mode": "DETERMINISTIC_PROJECT_REPLAY",
                "source_decision": _text(evaluation, "decision"),
                "actual_case_count": _integer(gold, "sample_count"),
            },
            runner_config_hash=_text(case_metadata, "runner_config_hash"),
            case_object_key=_text(case, "object_key"),
            case_content_hash=_text(case, "content_hash"),
            case_size_bytes=_integer(case, "size_bytes"),
            case_metadata=case_metadata,
        ),
        suite_name="project-authorized-llm-evaluation",
        suite_version="1.0.0",
        suite_slices=_int_mapping(gold, "slice_counts"),
        supply_chain=_SupplyChainSpec(
            image_repository=_image_repository(image),
            image_digest=image_digest,
            source_revision=_text(runtime, "git_commit"),
            source_repository="https://project.local/industrial-ops-agent",
            source_evidence_hash=_text(security, "evidence_chain_sha256"),
        ),
    )


def _imported_model_source(model_id: str, revision: str) -> dict[str, str]:
    """These import contracts pin a Hugging Face model, not the current Git checkout."""
    if (
        not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", model_id)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
    ):
        raise EnterpriseModelAssetImportConflict("COMPONENT_SOURCE_IDENTITY_REQUIRED")
    return {"repository": f"https://huggingface.co/{model_id}", "revision": revision}


def _asr_serving_metadata(
    root: Path, evidence: EnterpriseRuntimeEvidence, training: dict[str, Any]
) -> dict[str, Any]:
    if evidence.rollout_source_path is None:
        # Training-only imports stay useful for evaluation, but cannot be served yet.
        return {}
    try:
        rollout = verify_asr_kserve_acceptance(root, Path(evidence.rollout_source_path))
    except (AsrKServeAcceptanceError, ValueError, OSError) as exc:
        raise EnterpriseModelAssetImportConflict("ASR_IMPORT_SERVING_EVIDENCE_INVALID") from exc
    source = rollout.source
    worker = rollout.runtime
    model = _mapping(training, "base_model")
    adapter_digest = _text(_mapping(training, "adapter"), "bundle_sha256")
    if (
        rollout.evidence_chain_sha256 != evidence.rollout_evidence_chain_sha256
        or rollout.model_release.release_id != evidence.rollout_model_release_id
        or source.training_acceptance.path != evidence.source_paths[0]
        or source.training_acceptance.sha256
        != _source_hashes(root, (evidence.source_paths[0],))[evidence.source_paths[0]]
        or source.training_run_id != evidence.candidate_experiment_id
        or source.training_evidence_chain_sha256 != evidence.evidence_chain_sha256
        or source.adapter_bundle_sha256 != adapter_digest
        or worker.adapter_bundle_sha256 != adapter_digest
        or source.model_id != _text(model, "model_id")
        or worker.model_id != source.model_id
        or source.model_revision != _text(model, "revision")
        or worker.model_revision != source.model_revision
    ):
        raise EnterpriseModelAssetImportConflict("ASR_IMPORT_SERVING_BINDING_CHANGED")
    return {
        "runtime": {
            "image_repository": _image_repository(worker.worker_image),
            "image_digest": worker.worker_image_digest,
        },
        "serving_evidence": {
            "path": evidence.rollout_source_path,
            "evidence_chain_sha256": rollout.evidence_chain_sha256,
        },
    }


def _vlm_bundle(root: Path, evidence: EnterpriseRuntimeEvidence) -> _ImportBundle:
    if len(evidence.source_paths) != 3:
        raise EnterpriseModelAssetImportConflict("VLM_IMPORT_EVIDENCE_PATHS_CHANGED")
    adoption_path, source_path, serving_path = evidence.source_paths
    adoption = _json(root, adoption_path)
    source = _json(root, source_path)
    serving = _json(root, serving_path)
    candidate = _mapping(adoption, "candidate")
    evaluation = _mapping(source, "evaluation")
    training = _mapping(source, "training")
    curriculum_path = "artifacts/m7-vlm-curriculum-dataset/manifest.json"
    curriculum = _json(root, curriculum_path)
    split_counts = _int_mapping(curriculum, "split_counts")
    if (
        _text(candidate, "experiment_id") != evidence.candidate_experiment_id
        or _text(candidate, "evaluation_suite_id") != evidence.evaluation_reference
        or _text(serving, "status") != "VLM_LOCAL_KSERVE_SHADOW_CANARY_ROLLBACK_PASSED"
        or _text(serving, "evidence_chain_sha256") != evidence.evidence_chain_sha256
        or not bool(adoption.get("project_enterprise_use_authorized"))
        or not bool(adoption.get("enterprise_candidate"))
        or _text(evaluation, "decision") != "SIMULATION_CONTINUATION_ELIGIBLE"
        or _text(source, "evidence_chain_sha256")
        != _text(candidate, "source_evidence_chain_sha256")
    ):
        raise EnterpriseModelAssetImportConflict("VLM_IMPORT_EVIDENCE_BINDING_CHANGED")
    evidence_files = source.get("evidence_files")
    if not isinstance(evidence_files, list):
        raise EnterpriseModelAssetImportConflict("VLM_CASE_EVIDENCE_MISSING")
    case = next(
        (
            item
            for item in evidence_files
            if isinstance(item, dict)
            and item.get("path") == "evaluation/child-lora-continuation.json"
        ),
        None,
    )
    if not isinstance(case, dict):
        raise EnterpriseModelAssetImportConflict("VLM_CASE_EVIDENCE_MISSING")
    generated_at = _timestamp(source.get("generated_at"))
    model_revision = _text(candidate, "model_revision")
    runtime_digest = _tagged_digest(
        {
            "source_acceptance_sha256": _text(candidate, "source_acceptance_sha256"),
            "model_revision": model_revision,
        }
    )
    config = {
        "method": "VLM",
        "base_model_revision": model_revision,
        "optimizer_steps": _integer(training, "child_optimizer_steps"),
        "resumed_from_checkpoint": bool(training.get("resumed_from_checkpoint")),
        "compiled_config_sha256": _text(training, "compiled_config_sha256"),
        "project_authorized_import": True,
    }
    source_paths = (adoption_path, source_path, serving_path, curriculum_path)
    hard_gates = _bool_mapping(evaluation, "hard_gates")
    parent_score = _number(evaluation, "parent_accuracy")
    child_score = _number(evaluation, "child_accuracy")
    return _ImportBundle(
        component="VLM",
        evidence=evidence,
        generated_at=generated_at,
        source_paths=source_paths,
        source_hashes=_source_hashes(root, source_paths),
        source_classification=_text(candidate, "source_classification"),
        source_decision=_text(candidate, "source_decision"),
        training_snapshot=_SnapshotSpec(
            source_id=_text(candidate, "training_snapshot_id"),
            manifest_hash=_text(candidate, "training_manifest_sha256"),
            row_count=split_counts["train"] + split_counts["validation"],
            split_counts={
                "train": split_counts["train"],
                "validation": split_counts["validation"],
            },
            source_path=curriculum_path,
        ),
        evaluation_snapshot=_SnapshotSpec(
            source_id=_text(candidate, "evaluation_snapshot_id"),
            manifest_hash=_text(candidate, "evaluation_manifest_sha256"),
            row_count=_integer(source, "evaluation_cases"),
            split_counts={"evaluation": _integer(source, "evaluation_cases")},
            source_path=curriculum_path,
        ),
        candidate=_ExperimentSpec(
            source_id=evidence.candidate_experiment_id,
            method="VLM",
            task_type="INDUSTRIAL_MULTIMODAL_DIAGNOSIS",
            status="COMPLETED",
            base_model_id=_text(candidate, "model_id"),
            base_model_digest=f"revision:{model_revision}",
            tokenizer_digest=_tagged_digest(
                {"kind": "vlm-tokenizer-binding", "revision": model_revision}
            ),
            chat_template_digest=_tagged_digest(
                {"kind": "vlm-chat-template-binding", "revision": model_revision}
            ),
            git_commit=model_revision,
            container_digest=runtime_digest,
            training_config=config,
            distributed_profile={"strategy": "single_gpu", "world_size": 1},
            random_seeds=[42],
            hardware_topology={
                "accelerator": "NVIDIA_GPU",
                "count": 1,
                "name": str(source.get("gpu_name") or "NVIDIA_GPU"),
                "memory_bytes": int(source.get("gpu_total_memory_bytes") or 0),
            },
            metrics={
                "train_loss": _number(training, "train_loss"),
                "validation_loss": _number(training, "validation_loss"),
                "runtime_seconds": _number(training, "runtime_seconds"),
            },
            cost_summary={"source": source_path},
            mlflow_run_id=None,
        ),
        baseline=_ExperimentSpec(
            source_id=_text(training, "parent_experiment_id"),
            method="VLM",
            task_type="INDUSTRIAL_MULTIMODAL_DIAGNOSIS",
            status="COMPLETED",
            base_model_id=_text(candidate, "model_id"),
            base_model_digest=f"revision:{model_revision}",
            tokenizer_digest=_tagged_digest(
                {"kind": "vlm-tokenizer-binding", "revision": model_revision}
            ),
            chat_template_digest=_tagged_digest(
                {"kind": "vlm-chat-template-binding", "revision": model_revision}
            ),
            git_commit=model_revision,
            container_digest=runtime_digest,
            training_config={
                "method": "VLM",
                "optimizer_steps": _integer(training, "parent_optimizer_steps"),
                "source_role": "PARENT_CHECKPOINT_BASELINE",
            },
            distributed_profile={"strategy": "single_gpu", "world_size": 1},
            random_seeds=[42],
            hardware_topology={"accelerator": "NVIDIA_GPU", "count": 1},
            metrics={},
            cost_summary={},
            mlflow_run_id=None,
        ),
        artifact=_ArtifactSpec(
            source_id=_text(candidate, "artifact_id"),
            kind="vlm_adapter_bundle",
            object_key=_text(candidate, "artifact_object_key"),
            content_hash=_text(candidate, "artifact_sha256"),
            size_bytes=_integer(candidate, "artifact_size_bytes"),
            metadata={
                "format": "tar.gz",
                "serialization": "safetensors",
                "model_license": _text(candidate, "model_license"),
                "source": _imported_model_source(_text(candidate, "model_id"), model_revision),
                # This retained rollout ran a host GPU behind a container proxy.
                # Its proxy image is not the model-serving worker image.
            },
            source_size_recorded=True,
        ),
        evaluation=_EvaluationSpec(
            source_id=f"vlm-evaluation-{_text(source, 'evidence_chain_sha256')[:32]}",
            suite_source_id=_text(candidate, "evaluation_suite_id"),
            policy_source_id=f"vlm-policy-{_text(source, 'evidence_chain_sha256')[:32]}",
            primary_metric="vlm_diagnostic_accuracy",
            candidate_score=child_score,
            baseline_score=parent_score,
            quality_delta=child_score - parent_score,
            ci_low=0.0,
            ci_high=0.0,
            latency_improvement=0.0,
            cost_improvement=0.0,
            hard_gates=hard_gates,
            slice_metrics={
                "vlm": {
                    "diagnostic_accuracy": child_score,
                    "mean_region_iou": _number(evaluation, "child_mean_region_iou"),
                    "hallucination_rate": _number(evaluation, "child_hallucination_rate"),
                    "actual_case_count": _integer(source, "evaluation_cases"),
                }
            },
            report_hash=_digest(evaluation),
            thresholds={
                "quality_lift_min": 0.0,
                "quality_tolerance": 0.01,
                "efficiency_improvement_min": 0.0,
                "bootstrap_iterations": 1000.0,
                "region_iou_min": 0.5,
                "hallucination_rate_max": 0.1,
            },
            target_profile="VLM_COMPONENT",
            runner_config={
                "framework_mode": "FROZEN_PROJECT_VLM_EVALUATION",
                "precision": "bfloat16",
                "actual_case_count": _integer(source, "evaluation_cases"),
            },
            runner_config_hash=_digest(
                {
                    "evaluation_manifest_sha256": _text(candidate, "evaluation_manifest_sha256"),
                    "source_evidence_chain_sha256": _text(source, "evidence_chain_sha256"),
                }
            ),
            case_object_key=(f"{Path(source_path).parent.as_posix()}/{_text(case, 'path')}"),
            case_content_hash=_text(case, "sha256"),
            case_size_bytes=_integer(case, "size_bytes"),
            case_metadata={"format": "json", "source_decision": _text(evaluation, "decision")},
        ),
        suite_name="project-authorized-vlm-evaluation",
        suite_version="1.0.0",
        suite_slices={"multimodal_diagnosis": _integer(source, "evaluation_cases")},
        supply_chain=None,
    )


def _asr_bundle(root: Path, evidence: EnterpriseRuntimeEvidence) -> _ImportBundle:
    if len(evidence.source_paths) != 1:
        raise EnterpriseModelAssetImportConflict("ASR_IMPORT_EVIDENCE_PATHS_CHANGED")
    acceptance_path = evidence.source_paths[0]
    report = _json(root, acceptance_path)
    dataset = _mapping(report, "dataset")
    source_file = _mapping(dataset, "source_file")
    dataset_manifest = _mapping(dataset, "manifest")
    base_model = _mapping(report, "base_model")
    adapter = _mapping(report, "adapter")
    runtime = _mapping(report, "runtime")
    evaluation = _mapping(report, "evaluation")
    formal_report = _mapping(evaluation, "formal_report")
    observations = _mapping(evaluation, "observations")
    all_hard_gates = _bool_mapping(report, "hard_gates")
    required_hard_gates = {
        name: all_hard_gates[name]
        for name in (
            "data_governance",
            "asr_media_integrity",
            "asr_transcript_quality",
            "asr_noise_robustness",
            "no_blocking_regressions",
        )
    }
    evidence_chain = _text(report, "evidence_chain_sha256")
    if (
        _text(report, "run_id") != evidence.candidate_experiment_id
        or evidence.evaluation_reference != f"asr-evaluation-{evidence_chain[:24]}"
        or _text(report, "status") != "ASR_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
        or _text(report, "decision") != "ASR_CANDIDATE_ELIGIBLE_FOR_PROJECT_EVALUATION"
        or report.get("formal_model_release_created") is not False
        or report.get("runtime_eligible") is not False
        or runtime.get("actual_gpu_execution") is not True
        or runtime.get("model_training_simulated") is not False
        or not all(all_hard_gates.values())
        or evidence_chain != evidence.evidence_chain_sha256
    ):
        raise EnterpriseModelAssetImportConflict("ASR_IMPORT_EVIDENCE_BINDING_CHANGED")
    manifest_path = _text(dataset_manifest, "path")
    formal_path = _text(formal_report, "path")
    observations_path = _text(observations, "path")
    dataset_path = _text(source_file, "path")
    source_paths: tuple[str, ...] = (
        acceptance_path,
        manifest_path,
        formal_path,
        observations_path,
        dataset_path,
    )
    serving_metadata = _asr_serving_metadata(root, evidence, report)
    if evidence.rollout_source_path is not None:
        source_paths += (evidence.rollout_source_path,)
    model_revision = _text(base_model, "revision")
    baseline_wer = _number(evaluation, "baseline_wer")
    candidate_wer = _number(evaluation, "candidate_wer")
    baseline_accuracy = max(0.0, 1.0 - baseline_wer)
    candidate_accuracy = max(0.0, 1.0 - candidate_wer)
    generated_at = _timestamp(report.get("generated_at"))
    return _ImportBundle(
        component="ASR",
        evidence=evidence,
        generated_at=generated_at,
        source_paths=source_paths,
        source_hashes=_source_hashes(root, source_paths),
        source_classification=_text(report, "classification"),
        source_decision=_text(report, "decision"),
        training_snapshot=_SnapshotSpec(
            source_id=f"asr-training-{_text(dataset, 'manifest_sha256')[:24]}",
            manifest_hash=_text(dataset, "manifest_sha256"),
            row_count=_integer(dataset, "train_count") + _integer(dataset, "validation_count"),
            split_counts={
                "train": _integer(dataset, "train_count"),
                "validation": _integer(dataset, "validation_count"),
            },
            source_path=manifest_path,
        ),
        evaluation_snapshot=_SnapshotSpec(
            source_id=(f"asr-gold-{_text(_mapping(dataset, 'split_sha256s'), 'gold')[:24]}"),
            manifest_hash=_text(_mapping(dataset, "split_sha256s"), "gold"),
            row_count=_integer(dataset, "gold_count"),
            split_counts={"evaluation": _integer(dataset, "gold_count")},
            source_path=manifest_path,
        ),
        candidate=_ExperimentSpec(
            source_id=evidence.candidate_experiment_id,
            method="ASR",
            task_type="INDUSTRIAL_SPEECH_TRANSCRIPTION",
            status="COMPLETED",
            base_model_id=_text(base_model, "model_id"),
            base_model_digest=f"sha256:{_text(base_model, 'weight_sha256')}",
            tokenizer_digest=_tagged_digest(
                {
                    "kind": "whisper-processor",
                    "snapshot_manifest_sha256": _text(
                        base_model,
                        "snapshot_manifest_sha256",
                    ),
                }
            ),
            chat_template_digest="not-applicable:whisper-speech-seq2seq",
            git_commit=model_revision,
            container_digest=_text(base_model, "image_digest"),
            training_config={
                "method": _text(runtime, "method"),
                "trainer_family": _text(runtime, "trainer_family"),
                "optimizer_steps": _integer(runtime, "optimizer_steps"),
                "selected_validation_step": _integer(
                    runtime,
                    "selected_validation_step",
                ),
                "sampling_rate": 16000,
                "max_generation_length": 64,
                "precision": "bfloat16",
                "project_authorized_import": True,
            },
            distributed_profile={"strategy": "single_gpu", "world_size": 1},
            random_seeds=[20260824, 20260825],
            hardware_topology={
                "accelerator": "NVIDIA_GPU",
                "count": 1,
                "name": _text(runtime, "gpu_name"),
                "memory_bytes": _integer(runtime, "gpu_total_memory_bytes"),
            },
            metrics={
                "train_loss_first": _number(runtime, "train_loss_first"),
                "train_loss_last": _number(runtime, "train_loss_last"),
                "candidate_wer": candidate_wer,
                "candidate_cer": _number(evaluation, "candidate_cer"),
                "candidate_noise_wer": _number(evaluation, "candidate_noise_wer"),
            },
            cost_summary={"source": acceptance_path, "cpu_threads": 2},
            mlflow_run_id=None,
        ),
        baseline=_ExperimentSpec(
            source_id=f"asr-baseline-{model_revision[:24]}",
            method="ASR_BASELINE",
            task_type="INDUSTRIAL_SPEECH_TRANSCRIPTION",
            status="PLANNED",
            base_model_id=_text(base_model, "model_id"),
            base_model_digest=f"sha256:{_text(base_model, 'weight_sha256')}",
            tokenizer_digest=_tagged_digest(
                {
                    "kind": "whisper-processor",
                    "snapshot_manifest_sha256": _text(
                        base_model,
                        "snapshot_manifest_sha256",
                    ),
                }
            ),
            chat_template_digest="not-applicable:whisper-speech-seq2seq",
            git_commit=model_revision,
            container_digest=_text(base_model, "image_digest"),
            training_config={"adapter_enabled": False, "source_role": "BASE_MODEL"},
            distributed_profile={"strategy": "single_gpu", "world_size": 1},
            random_seeds=[20260824],
            hardware_topology={"accelerator": "NVIDIA_GPU", "count": 1},
            metrics={"baseline_wer": baseline_wer},
            cost_summary={},
            mlflow_run_id=None,
        ),
        artifact=_ArtifactSpec(
            source_id=f"asr-adapter-{_text(adapter, 'bundle_sha256')[:32]}",
            kind="asr_adapter_bundle",
            object_key=_text(adapter, "path"),
            content_hash=_text(adapter, "bundle_sha256"),
            size_bytes=_integer(adapter, "size_bytes"),
            metadata={
                "format": "directory",
                "serialization": "safetensors",
                "model_license": _text(base_model, "license"),
                "files": adapter.get("files", []),
                "source": _imported_model_source(_text(base_model, "model_id"), model_revision),
                **serving_metadata,
            },
            source_size_recorded=True,
        ),
        evaluation=_EvaluationSpec(
            source_id=evidence.evaluation_reference,
            suite_source_id=f"asr-suite-{_text(dataset, 'manifest_sha256')[:32]}",
            policy_source_id="asr-project-authorized-promotion-policy-v1",
            primary_metric="asr_word_accuracy",
            candidate_score=candidate_accuracy,
            baseline_score=baseline_accuracy,
            quality_delta=candidate_accuracy - baseline_accuracy,
            ci_low=0.0,
            ci_high=0.0,
            latency_improvement=0.0,
            cost_improvement=0.0,
            hard_gates=required_hard_gates,
            slice_metrics={
                "asr": {
                    "baseline_wer": baseline_wer,
                    "candidate_wer": candidate_wer,
                    "candidate_cer": _number(evaluation, "candidate_cer"),
                    "candidate_noise_wer": _number(
                        evaluation,
                        "candidate_noise_wer",
                    ),
                    "actual_case_count": _integer(
                        evaluation,
                        "frozen_gold_case_count",
                    ),
                }
            },
            report_hash=_text(formal_report, "sha256"),
            thresholds={
                "quality_lift_min": 0.02,
                "quality_tolerance": 0.01,
                "efficiency_improvement_min": 0.15,
                "bootstrap_iterations": 1000.0,
                "wer_max": _number(evaluation, "wer_max"),
                "cer_max": _number(evaluation, "cer_max"),
                "noise_wer_max": _number(evaluation, "noise_wer_max"),
            },
            target_profile="ASR_COMPONENT",
            runner_config={
                "framework_mode": "FROZEN_PROJECT_ASR_EVALUATION",
                "sampling_rate": 16000,
                "noise_profile": "deterministic_industrial_snr_3_db_v1",
                "actual_case_count": _integer(
                    evaluation,
                    "frozen_gold_case_count",
                ),
            },
            runner_config_hash=_digest(
                {
                    "dataset_manifest_sha256": _text(dataset, "manifest_sha256"),
                    "formal_report_sha256": _text(formal_report, "sha256"),
                    "source_evidence_chain_sha256": evidence_chain,
                }
            ),
            case_object_key=formal_path,
            case_content_hash=_text(formal_report, "sha256"),
            case_size_bytes=_integer(formal_report, "size_bytes"),
            case_metadata={
                "format": "json",
                "observations_sha256": _text(observations, "sha256"),
                "source_decision": _text(report, "decision"),
            },
        ),
        suite_name="project-authorized-asr-evaluation",
        suite_version="1.0.0",
        suite_slices={
            "asr_clean": _integer(evaluation, "clean_case_count"),
            "asr_noisy": _integer(evaluation, "noisy_case_count"),
        },
        supply_chain=None,
    )


def _tts_bundle(root: Path, evidence: EnterpriseRuntimeEvidence) -> _ImportBundle:
    if len(evidence.source_paths) != 1:
        raise EnterpriseModelAssetImportConflict("TTS_IMPORT_EVIDENCE_PATHS_CHANGED")
    report = _json(root, evidence.source_paths[0])
    if report.get("status") == "TTS_STRONG_ASR_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED":
        return _tts_strong_asr_bundle(root, evidence, report)
    return _legacy_tts_bundle(root, evidence)


def _tts_strong_asr_bundle(
    root: Path,
    evidence: EnterpriseRuntimeEvidence,
    report: dict[str, Any],
) -> _ImportBundle:
    acceptance_path = evidence.source_paths[0]
    try:
        verified = verify_calibrated_tts_value(root, Path(acceptance_path))
    except TtsCalibratedValueLabError as exc:
        raise EnterpriseModelAssetImportConflict("TTS_IMPORT_EVIDENCE_INVALID") from exc
    predecessor = _mapping(report, "predecessor")
    rejection_binding = _mapping(predecessor, "rejection")
    rejection_path = _text(rejection_binding, "path")
    rejection = _json(root, rejection_path)
    base_model = _input_component(rejection, "base_model")
    predecessor_dataset = _mapping(rejection, "dataset")
    training_manifest = _mapping(predecessor_dataset, "training_manifest")
    dataset = _mapping(report, "dataset")
    validation_manifest = _mapping(dataset, "validation_manifest")
    gold_manifest = _mapping(dataset, "final_gold_manifest")
    candidate = _mapping(report, "candidate")
    candidate_bundle = _mapping(candidate, "bundle")
    runtime_policy = _mapping(candidate, "runtime_policy_file")
    runtime = _mapping(report, "runtime")
    evaluation = _mapping(report, "evaluation")
    formal_report = _mapping(evaluation, "formal_report")
    observations = _mapping(evaluation, "observations")
    hard_gates = _bool_mapping(evaluation, "hard_gate_results")
    evidence_chain = _text(report, "evidence_chain_sha256")
    evaluation_id = f"tts-evaluation-{evidence_chain[:24]}"
    if (
        verified.run_id != evidence.candidate_experiment_id
        or verified.evidence_chain_sha256 != evidence.evidence_chain_sha256
        or evidence.evaluation_reference != evaluation_id
        or _text(report, "decision") != "TTS_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
        or report.get("candidate_accepted") is not True
        or report.get("release_draft_eligible") is not True
        or report.get("formal_model_release_created") is not False
        or report.get("runtime_eligible") is not False
        or report.get("failed_hard_gates") != []
        or runtime.get("actual_gpu_execution") is not True
        or runtime.get("actual_gpu_inference") is not True
        or runtime.get("current_run_model_training_performed") is not False
        or runtime.get("predecessor_actual_gpu_training") is not True
        or runtime.get("model_inference_simulated") is not False
        or runtime.get("model_training_simulated") is not False
        or _text(candidate, "role") != "AUTHORIZED_ENTERPRISE_VERIFIED_TTS_CANDIDATE"
        or _text(rejection_binding, "sha256")
        != sha256(_inside(root, rejection_path).read_bytes()).hexdigest()
        or not all(hard_gates.values())
    ):
        raise EnterpriseModelAssetImportConflict("TTS_IMPORT_EVIDENCE_BINDING_CHANGED")
    source_paths = (
        acceptance_path,
        rejection_path,
        _text(training_manifest, "path"),
        _text(validation_manifest, "path"),
        _text(gold_manifest, "path"),
        _text(formal_report, "path"),
        _text(observations, "path"),
        _text(runtime_policy, "path"),
    )
    model_revision = _text(base_model, "revision")
    baseline_score = _number(evaluation, "baseline_intelligibility")
    candidate_score = _number(evaluation, "candidate_intelligibility")
    gold_count = _integer(evaluation, "case_count")
    candidate_latency_ratio = _number(evaluation, "candidate_latency_ratio")
    runner_config = {
        "framework_mode": "FROZEN_PROJECT_TTS_STRONG_ASR_EVALUATION",
        "sampling_rate": 16000,
        "voice_profile_id": _text(dataset, "voice_profile_id"),
        "strong_asr_model": "openai/whisper-small.en",
        "verifier_policy": _text(runtime, "verifier_policy"),
        "actual_case_count": gold_count,
    }
    return _ImportBundle(
        component="TTS",
        evidence=evidence,
        generated_at=_timestamp(report.get("generated_at")),
        source_paths=source_paths,
        source_hashes=_source_hashes(root, source_paths),
        source_classification=_text(report, "classification"),
        source_decision=_text(report, "decision"),
        training_snapshot=_SnapshotSpec(
            source_id=f"tts-training-{_text(training_manifest, 'sha256')[:24]}",
            manifest_hash=_plain_digest(_text(training_manifest, "sha256")),
            row_count=_integer(predecessor_dataset, "train_count")
            + _integer(predecessor_dataset, "validation_count"),
            split_counts={
                "train": _integer(predecessor_dataset, "train_count"),
                "validation": _integer(predecessor_dataset, "validation_count"),
            },
            source_path=_text(training_manifest, "path"),
        ),
        evaluation_snapshot=_SnapshotSpec(
            source_id=f"tts-gold-{_text(gold_manifest, 'sha256')[:24]}",
            manifest_hash=_plain_digest(_text(gold_manifest, "sha256")),
            row_count=gold_count,
            split_counts={"evaluation": gold_count},
            source_path=_text(gold_manifest, "path"),
        ),
        candidate=_ExperimentSpec(
            source_id=evidence.candidate_experiment_id,
            method="TTS",
            task_type="INDUSTRIAL_SAFETY_SPEECH_SYNTHESIS",
            status="COMPLETED",
            base_model_id=_text(base_model, "model_id"),
            base_model_digest=_plain_digest(_text(base_model, "weight_sha256")),
            tokenizer_digest=_plain_digest(_text(base_model, "manifest_sha256")),
            chat_template_digest="not-applicable:speecht5-text-to-speech",
            git_commit=model_revision,
            container_digest=_text(report, "image_digest"),
            training_config={
                "method": _text(runtime, "method"),
                "trainer_family": _text(_mapping(rejection, "runtime"), "trainer_family"),
                "optimizer_steps": _integer(predecessor, "optimizer_steps"),
                "sampling_rate": 16000,
                "precision": "bfloat16",
                "voice_profile_id": _text(dataset, "voice_profile_id"),
                "current_run_model_training_performed": False,
                "predecessor_actual_gpu_training": True,
                "strong_asr_verifier": "openai/whisper-small.en",
                "runtime_policy_sha256": _text(runtime_policy, "sha256"),
                "project_authorized_import": True,
            },
            distributed_profile={"strategy": "single_gpu", "world_size": 1},
            random_seeds=[20260824, 20260825],
            hardware_topology={
                "accelerator": "NVIDIA_GPU",
                "count": 1,
                "name": _text(runtime, "gpu_name"),
                "memory_bytes": _integer(runtime, "gpu_total_memory_bytes"),
            },
            metrics={
                "candidate_intelligibility": candidate_score,
                "candidate_wer": _number(evaluation, "candidate_wer"),
                "candidate_safety_phrase_completeness": _number(
                    evaluation, "candidate_safety_phrase_completeness"
                ),
                "candidate_terminology_recall": _number(
                    evaluation, "candidate_terminology_recall"
                ),
                "voice_similarity_improvement": _number(
                    evaluation, "voice_similarity_improvement"
                ),
            },
            cost_summary={
                "source": acceptance_path,
                "current_run_model_training_performed": False,
            },
            mlflow_run_id=None,
        ),
        baseline=_ExperimentSpec(
            source_id=f"tts-baseline-{model_revision[:24]}",
            method="TTS_BASELINE",
            task_type="INDUSTRIAL_SAFETY_SPEECH_SYNTHESIS",
            status="COMPLETED",
            base_model_id=_text(base_model, "model_id"),
            base_model_digest=_plain_digest(_text(base_model, "weight_sha256")),
            tokenizer_digest=_plain_digest(_text(base_model, "manifest_sha256")),
            chat_template_digest="not-applicable:speecht5-text-to-speech",
            git_commit=model_revision,
            container_digest=_text(report, "image_digest"),
            training_config={"method": "TTS_BASELINE", "source_role": "BASE_MODEL"},
            distributed_profile={},
            random_seeds=[20260824],
            hardware_topology={},
            metrics={"baseline_intelligibility": baseline_score},
            cost_summary={},
            mlflow_run_id=None,
        ),
        artifact=_ArtifactSpec(
            source_id=f"tts-model-{_text(candidate_bundle, 'manifest_sha256')[:32]}",
            kind="tts_model_bundle",
            object_key=_text(candidate_bundle, "path"),
            content_hash=_plain_digest(_text(candidate_bundle, "manifest_sha256")),
            size_bytes=_integer(candidate_bundle, "size_bytes"),
            metadata={
                "format": "directory",
                "serialization": "safetensors",
                "voice_profile_id": _text(dataset, "voice_profile_id"),
                "voice_profile": {
                    "voice_profile_id": _text(dataset, "voice_profile_id"),
                    "language": "en-US",
                    "sampling_rate": 16000,
                    "speaker_embedding_dimension": 512,
                    "speech_contract_version": "industrial-tts-speech-v1",
                },
                "runtime_policy": runtime_policy,
                "strong_asr_verified": True,
            },
            source_size_recorded=True,
        ),
        evaluation=_EvaluationSpec(
            source_id=evaluation_id,
            suite_source_id=f"tts-suite-{_text(gold_manifest, 'sha256')[:32]}",
            policy_source_id="tts-strong-asr-enterprise-value-v1",
            primary_metric="tts_intelligibility",
            candidate_score=candidate_score,
            baseline_score=baseline_score,
            quality_delta=candidate_score - baseline_score,
            ci_low=candidate_score - baseline_score,
            ci_high=candidate_score - baseline_score,
            latency_improvement=max(0.0, 1.0 - candidate_latency_ratio),
            cost_improvement=0.0,
            hard_gates=hard_gates,
            slice_metrics={
                "tts": {
                    "baseline_intelligibility": baseline_score,
                    "candidate_intelligibility": candidate_score,
                    "baseline_wer": _number(evaluation, "baseline_wer"),
                    "candidate_wer": _number(evaluation, "candidate_wer"),
                    "candidate_voice_similarity": _number(
                        evaluation, "candidate_voice_similarity"
                    ),
                    "actual_case_count": gold_count,
                }
            },
            report_hash=_plain_digest(_text(formal_report, "sha256")),
            thresholds={
                "safety_phrase_completeness_min": 1.0,
                "terminology_recall_min": 1.0,
                "intelligibility_min": 1.0,
                "wer_max": 0.0,
                "voice_similarity_improvement_min": 0.0,
                "candidate_latency_ratio_max": 30.0,
            },
            target_profile="TTS_COMPONENT",
            runner_config=runner_config,
            runner_config_hash=_digest(runner_config),
            case_object_key=_text(formal_report, "path"),
            case_content_hash=_plain_digest(_text(formal_report, "sha256")),
            case_size_bytes=_integer(formal_report, "size_bytes"),
            case_metadata={
                "format": "json",
                "observations_path": _text(observations, "path"),
                "observations_sha256": _text(observations, "sha256"),
                "source_decision": _text(report, "decision"),
                "formal_evaluation_limit": 1,
            },
        ),
        suite_name="project-authorized-tts-strong-asr-evaluation",
        suite_version="9.0.0",
        suite_slices={"industrial_safety_tts": gold_count},
        supply_chain=_SupplyChainSpec(
            image_repository=_image_repository(_text(report, "image")),
            image_digest=_text(report, "image_digest"),
            source_revision=model_revision,
            source_repository="https://huggingface.co/microsoft/speecht5_tts",
            source_evidence_hash=evidence_chain,
        ),
    )


def _legacy_tts_bundle(root: Path, evidence: EnterpriseRuntimeEvidence) -> _ImportBundle:
    if len(evidence.source_paths) != 1:
        raise EnterpriseModelAssetImportConflict("TTS_IMPORT_EVIDENCE_PATHS_CHANGED")
    acceptance_path = evidence.source_paths[0]
    report = _json(root, acceptance_path)
    dataset = _mapping(report, "dataset")
    training_manifest = _mapping(dataset, "training_manifest")
    gold_manifest = _mapping(dataset, "final_gold_manifest")
    candidate = _mapping(report, "candidate")
    runtime = _mapping(report, "runtime")
    evaluation = _mapping(report, "evaluation")
    formal_report = _mapping(evaluation, "formal_report")
    observations = _mapping(evaluation, "observations")
    dataset_receipt = _mapping(evaluation, "dataset_receipt")
    base_model = _input_component(report, "base_model")
    all_hard_gates = _bool_mapping(report, "hard_gates")
    required_hard_gates = {
        name: all_hard_gates[name]
        for name in (
            "tts_voice_usage_authorization",
            "tts_audio_integrity",
            "tts_safety_warning_completeness",
            "tts_terminology_accuracy",
            "tts_intelligibility",
            "no_blocking_regressions",
        )
    }
    evidence_chain = _text(report, "evidence_chain_sha256")
    expected_evaluation = f"tts-evaluation-{evidence_chain[:24]}"
    if (
        _text(report, "run_id") != evidence.candidate_experiment_id
        or evidence.evaluation_reference != expected_evaluation
        or _text(report, "status") != "TTS_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
        or _text(report, "decision") != "TTS_CANDIDATE_ELIGIBLE_FOR_PROJECT_EVALUATION"
        or report.get("formal_model_release_created") is not False
        or report.get("runtime_eligible") is not False
        or runtime.get("actual_gpu_execution") is not True
        or runtime.get("model_training_simulated") is not False
        or not all(all_hard_gates.values())
        or evidence_chain != evidence.evidence_chain_sha256
    ):
        raise EnterpriseModelAssetImportConflict("TTS_IMPORT_EVIDENCE_BINDING_CHANGED")
    training_path = _text(training_manifest, "path")
    gold_path = _text(gold_manifest, "path")
    formal_path = _text(formal_report, "path")
    observations_path = _text(observations, "path")
    dataset_receipt_path = _text(dataset_receipt, "path")
    source_paths = (
        acceptance_path,
        training_path,
        gold_path,
        formal_path,
        observations_path,
        dataset_receipt_path,
    )
    generated_at = _timestamp(report.get("generated_at"))
    model_revision = _text(base_model, "revision")
    baseline_score = _number(evaluation, "baseline_intelligibility")
    candidate_score = _number(evaluation, "candidate_intelligibility")
    gold_count = _integer(evaluation, "frozen_gold_case_count")
    image = _text(report, "image")
    image_digest = _text(report, "image_digest")
    return _ImportBundle(
        component="TTS",
        evidence=evidence,
        generated_at=generated_at,
        source_paths=source_paths,
        source_hashes=_source_hashes(root, source_paths),
        source_classification=_text(report, "classification"),
        source_decision=_text(report, "decision"),
        training_snapshot=_SnapshotSpec(
            source_id=f"tts-training-{_text(training_manifest, 'sha256')[:24]}",
            manifest_hash=_text(training_manifest, "sha256"),
            row_count=_integer(dataset, "train_count") + _integer(dataset, "validation_count"),
            split_counts={
                "train": _integer(dataset, "train_count"),
                "validation": _integer(dataset, "validation_count"),
            },
            source_path=training_path,
        ),
        evaluation_snapshot=_SnapshotSpec(
            source_id=f"tts-gold-{_text(gold_manifest, 'sha256')[:24]}",
            manifest_hash=_text(gold_manifest, "sha256"),
            row_count=gold_count,
            split_counts={"evaluation": gold_count},
            source_path=gold_path,
        ),
        candidate=_ExperimentSpec(
            source_id=evidence.candidate_experiment_id,
            method="TTS",
            task_type="INDUSTRIAL_SAFETY_SPEECH_SYNTHESIS",
            status="COMPLETED",
            base_model_id=_text(base_model, "model_id"),
            base_model_digest=f"sha256:{_text(base_model, 'weight_sha256')}",
            tokenizer_digest=_tagged_digest(
                {
                    "kind": "speecht5-processor",
                    "snapshot_manifest_sha256": _text(base_model, "manifest_sha256"),
                }
            ),
            chat_template_digest="not-applicable:speecht5-text-to-speech",
            git_commit=model_revision,
            container_digest=image_digest,
            training_config={
                "method": _text(runtime, "method"),
                "trainer_family": _text(runtime, "trainer_family"),
                "optimizer_steps": _integer(runtime, "optimizer_steps"),
                "selected_validation_step": _integer(runtime, "selected_validation_step"),
                "sampling_rate": 16000,
                "precision": "bfloat16",
                "voice_profile_id": _text(dataset, "voice_profile_id"),
                "project_authorized_import": True,
            },
            distributed_profile={"strategy": "single_gpu", "world_size": 1},
            random_seeds=[20260824, 20260825],
            hardware_topology={
                "accelerator": "NVIDIA_GPU",
                "count": 1,
                "name": _text(runtime, "gpu_name"),
                "memory_bytes": _integer(runtime, "gpu_total_memory_bytes"),
            },
            metrics={
                "train_loss": _number(runtime, "train_loss"),
                "candidate_intelligibility": candidate_score,
                "candidate_wer": _number(evaluation, "candidate_wer"),
                "candidate_safety_phrase_completeness": _number(
                    evaluation, "candidate_safety_phrase_completeness"
                ),
                "candidate_terminology_recall": _number(evaluation, "candidate_terminology_recall"),
                "gold_voice_similarity_improvement": _number(
                    evaluation, "gold_voice_similarity_improvement"
                ),
            },
            cost_summary={"source": acceptance_path, "cpu_threads": 2},
            mlflow_run_id=None,
        ),
        baseline=_ExperimentSpec(
            source_id=f"tts-baseline-{model_revision[:24]}",
            method="TTS_BASELINE",
            task_type="INDUSTRIAL_SAFETY_SPEECH_SYNTHESIS",
            status="PLANNED",
            base_model_id=_text(base_model, "model_id"),
            base_model_digest=f"sha256:{_text(base_model, 'weight_sha256')}",
            tokenizer_digest=_tagged_digest(
                {
                    "kind": "speecht5-processor",
                    "snapshot_manifest_sha256": _text(base_model, "manifest_sha256"),
                }
            ),
            chat_template_digest="not-applicable:speecht5-text-to-speech",
            git_commit=model_revision,
            container_digest=image_digest,
            training_config={"postnet_adaptation": False, "source_role": "BASE_MODEL"},
            distributed_profile={"strategy": "single_gpu", "world_size": 1},
            random_seeds=[20260824],
            hardware_topology={"accelerator": "NVIDIA_GPU", "count": 1},
            metrics={"baseline_intelligibility": baseline_score},
            cost_summary={},
            mlflow_run_id=None,
        ),
        artifact=_ArtifactSpec(
            source_id=f"tts-model-{_text(candidate, 'bundle_sha256')[:32]}",
            kind="tts_model_bundle",
            object_key=_text(candidate, "path"),
            content_hash=_text(candidate, "bundle_sha256"),
            size_bytes=_integer(candidate, "size_bytes"),
            metadata={
                "format": "directory",
                "serialization": "safetensors",
                "voice_profile_id": _text(dataset, "voice_profile_id"),
                "voice_profile": {
                    "voice_profile_id": _text(dataset, "voice_profile_id"),
                    "language": "en-US",
                    "sampling_rate": 16000,
                    "speaker_embedding_dimension": 512,
                    "speech_contract_version": "industrial-tts-speech-v1",
                },
                "files": candidate.get("files", []),
            },
            source_size_recorded=True,
        ),
        evaluation=_EvaluationSpec(
            source_id=expected_evaluation,
            suite_source_id=f"tts-suite-{_text(gold_manifest, 'sha256')[:32]}",
            policy_source_id="tts-project-authorized-promotion-policy-v1",
            primary_metric="tts_intelligibility",
            candidate_score=candidate_score,
            baseline_score=baseline_score,
            quality_delta=candidate_score - baseline_score,
            ci_low=0.0,
            ci_high=0.0,
            latency_improvement=0.0,
            cost_improvement=0.0,
            hard_gates=required_hard_gates,
            slice_metrics={
                "tts": {
                    "baseline_intelligibility": baseline_score,
                    "candidate_intelligibility": candidate_score,
                    "baseline_wer": _number(evaluation, "baseline_wer"),
                    "candidate_wer": _number(evaluation, "candidate_wer"),
                    "actual_case_count": gold_count,
                }
            },
            report_hash=_text(formal_report, "sha256"),
            thresholds={
                "safety_phrase_completeness_min": 1.0,
                "terminology_recall_min": 0.95,
                "intelligibility_min": 0.90,
                "audio_integrity_rate_min": 1.0,
                "voice_similarity_improvement_min": _number(
                    evaluation, "voice_similarity_improvement_min"
                ),
            },
            target_profile="TTS_COMPONENT",
            runner_config={
                "framework_mode": "FROZEN_PROJECT_TTS_EVALUATION",
                "sampling_rate": 16000,
                "voice_profile_id": _text(dataset, "voice_profile_id"),
                "actual_case_count": gold_count,
            },
            runner_config_hash=_digest(
                {
                    "training_manifest_sha256": _text(training_manifest, "sha256"),
                    "gold_manifest_sha256": _text(gold_manifest, "sha256"),
                    "formal_report_sha256": _text(formal_report, "sha256"),
                    "source_evidence_chain_sha256": evidence_chain,
                }
            ),
            case_object_key=formal_path,
            case_content_hash=_text(formal_report, "sha256"),
            case_size_bytes=_integer(formal_report, "size_bytes"),
            case_metadata={
                "format": "json",
                "observations_sha256": _text(observations, "sha256"),
                "source_decision": _text(report, "decision"),
            },
        ),
        suite_name="project-authorized-tts-evaluation",
        suite_version="1.0.0",
        suite_slices={"industrial_safety_tts": gold_count},
        supply_chain=_SupplyChainSpec(
            image_repository=_image_repository(image),
            image_digest=image_digest,
            source_revision=model_revision,
            source_repository="https://huggingface.co/microsoft/speecht5_tts",
            source_evidence_hash=evidence_chain,
        ),
    )


def _embedding_bundle(root: Path, evidence: EnterpriseRuntimeEvidence) -> _ImportBundle:
    if len(evidence.source_paths) != 1:
        raise EnterpriseModelAssetImportConflict("EMBEDDING_IMPORT_SOURCE_PATHS_INVALID")
    acceptance_path = evidence.source_paths[0]
    try:
        verified = verify_calibrated_embedding_value(root, Path(acceptance_path))
    except EmbeddingCalibratedValueLabError as exc:
        raise EnterpriseModelAssetImportConflict("EMBEDDING_IMPORT_EVIDENCE_INVALID") from exc
    report = _json(root, acceptance_path)
    predecessor = _mapping(report, "predecessor")
    governance_binding = _mapping(predecessor, "governance_acceptance")
    governance_path = _text(governance_binding, "path")
    governance = _json(root, governance_path)
    base_weight = _mapping(governance, "base_model_weight")
    training_manifest = _mapping(governance, "training_manifest")
    dataset = _mapping(report, "dataset")
    validation_manifest = _mapping(dataset, "validation_manifest")
    gold_manifest = _mapping(dataset, "final_gold_manifest")
    registry = _mapping(dataset, "registry")
    candidate = _mapping(report, "candidate")
    runtime = _mapping(report, "runtime")
    evaluation = _mapping(report, "evaluation")
    formal_report = _mapping(evaluation, "formal_report")
    observations = _mapping(evaluation, "observations")
    hard_gates = _bool_mapping(evaluation, "hard_gate_results")
    candidate_directory = _inside(root, _text(candidate, "path"), directory=True)
    tokenizer_path = candidate_directory / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise EnterpriseModelAssetImportConflict("EMBEDDING_IMPORT_TOKENIZER_MISSING")
    evidence_chain = _text(report, "evidence_chain_sha256")
    evaluation_id = f"embedding-evaluation-{evidence_chain[:24]}"
    if (
        verified.run_id != evidence.candidate_experiment_id
        or verified.evidence_chain_sha256 != evidence.evidence_chain_sha256
        or evidence.evaluation_reference != evaluation_id
        or _text(report, "status")
        != "EMBEDDING_CALIBRATED_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
        or _text(report, "decision")
        != "EMBEDDING_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
        or report.get("candidate_accepted") is not True
        or report.get("release_draft_eligible") is not True
        or report.get("formal_model_release_created") is not False
        or report.get("runtime_eligible") is not False
        or report.get("failed_hard_gates") != []
        or runtime.get("actual_gpu_execution") is not True
        or runtime.get("actual_gpu_inference") is not True
        or runtime.get("current_run_model_training_performed") is not False
        or runtime.get("source_candidate_actual_gpu_trained") is not True
        or runtime.get("model_inference_simulated") is not False
        or runtime.get("model_training_simulated") is not False
        or _text(candidate, "role")
        != "AUTHORIZED_ENTERPRISE_CALIBRATED_EMBEDDING_CANDIDATE"
        or _text(governance_binding, "sha256")
        != sha256(_inside(root, governance_path).read_bytes()).hexdigest()
        or not all(hard_gates.values())
    ):
        raise EnterpriseModelAssetImportConflict("EMBEDDING_IMPORT_EVIDENCE_BINDING_CHANGED")
    source_paths = (
        acceptance_path,
        governance_path,
        _text(training_manifest, "path"),
        _text(validation_manifest, "path"),
        _text(gold_manifest, "path"),
        _text(registry, "path"),
        _text(formal_report, "path"),
        _text(observations, "path"),
    )
    training_rows = _rows(
        _json(root, _text(training_manifest, "path")),
        "EMBEDDING_TRAINING_MANIFEST",
    )
    training_splits = _split_counts(training_rows)
    base_digest = _plain_digest(_text(base_weight, "sha256"))
    model_revision = base_digest[:40]
    candidate_score = _number(evaluation, "candidate_mrr")
    baseline_score = _number(evaluation, "baseline_mrr")
    quality_delta = candidate_score - baseline_score
    latency_ratio = _number(evaluation, "candidate_latency_ratio")
    calibration = _mapping(report, "calibration")
    runner_config = {
        "framework_mode": "CALIBRATED_SENTENCE_TRANSFORMERS_RETRIEVAL",
        "top_k": _integer(dataset, "top_k"),
        "calibration_profile_id": _text(calibration, "profile_id"),
        "registry_sha256": _text(registry, "sha256"),
        "actual_case_count": _integer(dataset, "final_gold_count"),
    }
    return _ImportBundle(
        component="EMBEDDING",
        evidence=evidence,
        generated_at=_timestamp(report.get("generated_at")),
        source_paths=source_paths,
        source_hashes=_source_hashes(root, source_paths),
        source_classification=_text(report, "classification"),
        source_decision=_text(report, "decision"),
        training_snapshot=_SnapshotSpec(
            source_id=f"embedding-training-{_text(training_manifest, 'sha256')[:24]}",
            manifest_hash=_plain_digest(_text(training_manifest, "sha256")),
            row_count=len(training_rows),
            split_counts=training_splits,
            source_path=_text(training_manifest, "path"),
        ),
        evaluation_snapshot=_SnapshotSpec(
            source_id="industrial-embedding-final-gold-v3",
            manifest_hash=_plain_digest(_text(gold_manifest, "sha256")),
            row_count=_integer(dataset, "final_gold_count"),
            split_counts={"evaluation": _integer(dataset, "final_gold_count")},
            source_path=_text(gold_manifest, "path"),
        ),
        candidate=_ExperimentSpec(
            source_id=evidence.candidate_experiment_id,
            method="EMBEDDING",
            task_type="INDUSTRIAL_KNOWLEDGE_EMBEDDING",
            status="COMPLETED",
            base_model_id="sentence-transformers/all-MiniLM-L6-v2",
            base_model_digest=base_digest,
            tokenizer_digest=sha256(tokenizer_path.read_bytes()).hexdigest(),
            chat_template_digest="not-applicable:sentence-transformers-embedding",
            git_commit=model_revision,
            container_digest=_text(report, "image_digest"),
            training_config={
                "method": "EMBEDDING",
                "max_sequence_length": 256,
                "precision": "bfloat16",
                "per_device_eval_batch_size": 16,
                "current_run_model_training_performed": False,
                "source_candidate_actual_gpu_trained": True,
                "source_v2_run_id": _text(predecessor, "run_id"),
                "source_v2_bundle_sha256": _text(candidate, "source_v2_bundle_sha256"),
                "query_augmentation": _text(candidate, "query_augmentation"),
                "score_calibration": _text(candidate, "score_calibration"),
                "calibration_profile_id": _text(calibration, "profile_id"),
                "calibration": calibration,
                "source_acceptance": acceptance_path,
                "project_authorized_import": True,
            },
            distributed_profile={"strategy": "single_gpu", "world_size": 1},
            random_seeds=[42],
            hardware_topology={
                "accelerator": "NVIDIA_GPU",
                "count": 1,
                "name": _text(runtime, "gpu_name"),
                "memory_bytes": _integer(runtime, "gpu_total_memory_bytes"),
            },
            metrics={
                "recall_at_k": _number(evaluation, "candidate_recall_at_k"),
                "mrr": candidate_score,
                "ndcg_at_k": _number(evaluation, "candidate_ndcg_at_k"),
                "primary_improvement": quality_delta,
                "p95_latency_ms": _number(evaluation, "candidate_p95_latency_ms"),
            },
            cost_summary={
                "gpu_hours": 0.0,
                "current_run_model_training_performed": False,
                "source": acceptance_path,
            },
            mlflow_run_id=None,
        ),
        baseline=_ExperimentSpec(
            source_id=f"embedding-calibrated-baseline-{evidence_chain[:24]}",
            method="BASELINE",
            task_type="INDUSTRIAL_KNOWLEDGE_EMBEDDING",
            status="COMPLETED",
            base_model_id="sentence-transformers/all-MiniLM-L6-v2",
            base_model_digest=base_digest,
            tokenizer_digest=sha256(tokenizer_path.read_bytes()).hexdigest(),
            chat_template_digest="not-applicable:sentence-transformers-embedding",
            git_commit=model_revision,
            container_digest=_text(report, "image_digest"),
            training_config={"method": "BASELINE", "source_evaluation_id": evaluation_id},
            distributed_profile={},
            random_seeds=[42],
            hardware_topology={},
            metrics={
                "recall_at_k": _number(evaluation, "baseline_recall_at_k"),
                "mrr": baseline_score,
                "ndcg_at_k": _number(evaluation, "baseline_ndcg_at_k"),
                "p95_latency_ms": _number(evaluation, "baseline_p95_latency_ms"),
            },
            cost_summary={"source": acceptance_path},
            mlflow_run_id=None,
        ),
        artifact=_ArtifactSpec(
            source_id=f"embedding-bundle-{_text(candidate, 'manifest_sha256')[:32]}",
            kind="embedding_model_bundle",
            object_key=candidate_directory.relative_to(root).as_posix(),
            content_hash=_plain_digest(_text(candidate, "manifest_sha256")),
            size_bytes=_integer(candidate, "size_bytes"),
            metadata={
                "format": "directory",
                "serialization": "safetensors",
                "query_augmentation": _text(candidate, "query_augmentation"),
                "score_calibration": _text(candidate, "score_calibration"),
                "calibration_profile_id": _text(calibration, "profile_id"),
                "current_run_model_training_performed": False,
                "source_candidate_actual_gpu_trained": True,
            },
            source_size_recorded=True,
        ),
        evaluation=_EvaluationSpec(
            source_id=evaluation_id,
            suite_source_id="industrial-embedding-final-gold-v3",
            policy_source_id="embedding-calibrated-enterprise-value-v1",
            primary_metric="mrr",
            candidate_score=candidate_score,
            baseline_score=baseline_score,
            quality_delta=quality_delta,
            ci_low=quality_delta,
            ci_high=quality_delta,
            latency_improvement=max(0.0, 1.0 - latency_ratio),
            cost_improvement=0.0,
            hard_gates=hard_gates,
            slice_metrics={
                "calibrated_terminology": {
                    "candidate_recall_at_k": _number(
                        evaluation, "candidate_recall_at_k"
                    ),
                    "candidate_mrr": candidate_score,
                    "candidate_ndcg_at_k": _number(
                        evaluation, "candidate_ndcg_at_k"
                    ),
                    "baseline_mrr": baseline_score,
                    "actual_case_count": _integer(dataset, "final_gold_count"),
                }
            },
            report_hash=_plain_digest(_text(formal_report, "sha256")),
            thresholds={
                "primary_improvement_min": 0.10,
                "candidate_recall_at_k_min": 0.90,
                "candidate_mrr_min": 0.85,
                "candidate_ndcg_at_k_min": 0.85,
                "candidate_latency_ratio_max": 4.0,
            },
            target_profile="RETRIEVAL_COMPONENT",
            runner_config=runner_config,
            runner_config_hash=_digest(runner_config),
            case_object_key=_text(formal_report, "path"),
            case_content_hash=_plain_digest(_text(formal_report, "sha256")),
            case_size_bytes=_integer(formal_report, "size_bytes"),
            case_metadata={
                "format": "json",
                "observations_path": _text(observations, "path"),
                "observations_sha256": _text(observations, "sha256"),
                "source_decision": _text(report, "decision"),
                "formal_evaluation_limit": 1,
            },
        ),
        suite_name="project-authorized-embedding-calibrated-evaluation",
        suite_version="3.0.0",
        suite_slices={"calibrated_terminology": _integer(dataset, "final_gold_count")},
        supply_chain=_SupplyChainSpec(
            image_repository=_image_repository(_text(report, "image")),
            image_digest=_text(report, "image_digest"),
            source_revision=model_revision,
            source_repository="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
            source_evidence_hash=evidence_chain,
        ),
    )


def _reranker_bundle(root: Path, evidence: EnterpriseRuntimeEvidence) -> _ImportBundle:
    if len(evidence.source_paths) != 1:
        raise EnterpriseModelAssetImportConflict("RERANKER_IMPORT_SOURCE_PATHS_INVALID")
    outcome_path = evidence.source_paths[0]
    try:
        verified = verify_calibrated_reranker_value(root, Path(outcome_path))
        readiness = verify_reranker_kserve_readiness(root, RERANKER_READINESS_RELATIVE)
        runtime_probe = verify_runtime_gpu_probe_report(
            root,
            RERANKER_GPU_PROBE_RELATIVE,
        )
    except (
        RerankerCalibratedValueLabError,
        RerankerKServeReadinessError,
        RerankerRuntimeGpuProbeError,
    ) as exc:
        raise EnterpriseModelAssetImportConflict("RERANKER_IMPORT_EVIDENCE_INVALID") from exc
    report = _json(root, outcome_path)
    source = _mapping(report, "source")
    predecessor = _mapping(report, "predecessor")
    input_model = _mapping(report, "input_model")
    registry = _mapping(report, "registry")
    registry_manifest = _mapping(registry, "manifest")
    dataset = _mapping(report, "dataset")
    gold_manifest = _mapping(dataset, "final_gold_manifest")
    candidate = _mapping(report, "candidate")
    runtime = _mapping(report, "runtime")
    evaluation = _mapping(report, "evaluation")
    formal_report = _mapping(evaluation, "formal_report")
    observations = _mapping(evaluation, "observations")
    calibration = _mapping(report, "calibration")
    erratum = _mapping(predecessor, "erratum")
    original_outcome = _mapping(predecessor, "original_outcome")
    predecessor_gold = _mapping(predecessor, "gold_v4_retirement")
    candidate_directory = _inside(root, _text(candidate, "path"), directory=True)
    tokenizer = _file_binding_by_name(candidate, "files", "tokenizer.json")
    model_revision = _text(input_model, "revision")
    evidence_chain = _text(report, "evidence_chain_sha256")
    evaluation_id = f"reranker-evaluation-{evidence_chain[:24]}"
    if (
        verified.run_id != evidence.candidate_experiment_id
        or verified.evidence_chain_sha256 != evidence.evidence_chain_sha256
        or evidence.evaluation_reference != evaluation_id
        or _text(report, "status") != "RERANKER_CALIBRATED_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED"
        or _text(report, "decision") != "RERANKER_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
        or report.get("candidate_accepted") is not True
        or report.get("formal_model_release_created") is not False
        or report.get("runtime_eligible") is not False
        or runtime.get("actual_gpu_execution") is not True
        or runtime.get("actual_gpu_inference") is not True
        or runtime.get("current_run_model_training_performed") is not False
        or runtime.get("source_candidate_actual_gpu_trained") is not True
        or runtime.get("model_training_simulated") is not False
        or _text(candidate, "role") != "AUTHORIZED_ENTERPRISE_CALIBRATED_RERANKER_CANDIDATE"
        or _text(evaluation, "primary_metric") != "ndcg_at_k"
        or report.get("failed_hard_gates") != []
        or readiness.source.value_acceptance.path != outcome_path
        or readiness.source.run_id != verified.run_id
        or readiness.source.value_evidence_chain_sha256 != verified.evidence_chain_sha256
        or readiness.source.candidate_path != verified.candidate.path
        or readiness.source.candidate_bundle_sha256 != verified.candidate.bundle_sha256
        or readiness.source.training_image_digest != verified.image_digest
        or str(readiness.runtime_image.digest) == _text(report, "image_digest")
        or readiness.runtime_image.actual_image_built is not True
        or readiness.runtime_image.actual_gpu_runtime_probe_completed is not False
        or readiness.model_release.runtime_supply_chain_ready is not True
        or readiness.model_release.formal_model_release_created is not False
        or runtime_probe.source.readiness.path
        != RERANKER_READINESS_RELATIVE.as_posix()
        or runtime_probe.source.readiness_evidence_chain_sha256
        != readiness.evidence_chain_sha256
        or runtime_probe.source.value_run_id != verified.run_id
        or runtime_probe.source.value_evidence_chain_sha256
        != verified.evidence_chain_sha256
        or runtime_probe.source.candidate_path != verified.candidate.path
        or runtime_probe.source.candidate_bundle_sha256
        != verified.candidate.bundle_sha256
        or runtime_probe.runtime.image_digest != readiness.runtime_image.digest
        or runtime_probe.runtime.actual_gpu_execution is not True
        or runtime_probe.runtime.model_execution_simulated is not False
        or runtime_probe.probe.deterministic_replay_verified is not True
        or runtime_probe.cleanup.no_runtime_service_left_running is not True
        or set(runtime_probe.hard_gates.values()) != {True}
    ):
        raise EnterpriseModelAssetImportConflict("RERANKER_IMPORT_EVIDENCE_BINDING_CHANGED")
    hard_gates = _bool_mapping(evaluation, "hard_gate_results")
    if set(hard_gates) != {
        "cross_tenant_isolation",
        "data_governance",
        "retrieval_index_compatibility",
        "no_blocking_regressions",
    } or not all(hard_gates.values()):
        raise EnterpriseModelAssetImportConflict("RERANKER_IMPORT_HARD_GATES_INVALID")

    readiness_paths = (
        RERANKER_READINESS_RELATIVE.as_posix(),
        readiness.runtime_image.image_inspect.path,
        readiness.runtime_image.entrypoint_help.path,
        readiness.runtime_image.runtime_source.path,
        readiness.runtime_image.dockerfile.path,
        readiness.runtime_image.kserve_provider.path,
    )
    runtime_probe_paths = (
        RERANKER_GPU_PROBE_RELATIVE.as_posix(),
        runtime_probe.runtime.health.path,
        runtime_probe.runtime.metrics.path,
        runtime_probe.runtime.container_inspect.path,
        runtime_probe.runtime.gpu_inventory.path,
        runtime_probe.runtime.container_logs.path,
        runtime_probe.probe.request.path,
        runtime_probe.probe.first_response.path,
        runtime_probe.probe.second_response.path,
        runtime_probe.probe.missing_alias_rejection.path,
        runtime_probe.probe.ambiguous_alias_rejection.path,
        runtime_probe.probe.binding_rejection.path,
    )
    source_paths = tuple(
        dict.fromkeys(
            (
                outcome_path,
                _text(source, "path"),
                _text(registry_manifest, "path"),
                _text(gold_manifest, "path"),
                _text(formal_report, "path"),
                _text(observations, "path"),
                _text(erratum, "path"),
                _text(original_outcome, "path"),
                _text(predecessor_gold, "path"),
                _text(report, "gold_retirement_path"),
                *readiness_paths,
                *runtime_probe_paths,
            )
        )
    )
    candidate_score = _number(evaluation, "candidate_ndcg_at_k")
    baseline_score = _number(evaluation, "baseline_ndcg_at_k")
    quality_delta = _number(evaluation, "primary_improvement")
    latency_ratio = _number(evaluation, "candidate_latency_ratio")
    runner_config = {
        "backend": "sentence-transformers-cross-encoder-5",
        "actual_gpu_execution": True,
        "current_run_model_training_performed": False,
        "source_candidate_actual_gpu_trained": True,
        "top_k": _integer(dataset, "top_k"),
        "max_sequence_length": 128,
        "precision": "bfloat16",
        "per_device_eval_batch_size": 16,
        "calibration_profile_id": _text(calibration, "profile_id"),
        "actual_gpu_runtime_probe_completed": True,
        "runtime_probe_evidence_chain_sha256": runtime_probe.evidence_chain_sha256,
    }
    generated_at = _timestamp(report.get("generated_at"))
    return _ImportBundle(
        component="RERANKER",
        evidence=evidence,
        generated_at=generated_at,
        source_paths=source_paths,
        source_hashes=_source_hashes(root, source_paths),
        source_classification=_text(report, "classification"),
        source_decision=_text(report, "decision"),
        training_snapshot=_SnapshotSpec(
            source_id=_text(registry, "registry_id"),
            manifest_hash=_plain_digest(_text(registry_manifest, "sha256")),
            row_count=_integer(registry, "entry_count"),
            split_counts={"calibration_registry": _integer(registry, "entry_count")},
            source_path=_text(registry_manifest, "path"),
        ),
        evaluation_snapshot=_SnapshotSpec(
            source_id="industrial-reranker-final-gold-v5",
            manifest_hash=_plain_digest(_text(gold_manifest, "sha256")),
            row_count=_integer(dataset, "final_gold_count"),
            split_counts={"evaluation": _integer(dataset, "final_gold_count")},
            source_path=_text(gold_manifest, "path"),
        ),
        candidate=_ExperimentSpec(
            source_id=evidence.candidate_experiment_id,
            method="RERANKER",
            task_type="INDUSTRIAL_KNOWLEDGE_RERANKING",
            status="COMPLETED",
            base_model_id=_text(input_model, "model_id"),
            base_model_digest=_plain_digest(_text(input_model, "manifest_sha256")),
            tokenizer_digest=_plain_digest(_text(tokenizer, "sha256")),
            chat_template_digest="not-applicable:cross-encoder-reranker",
            git_commit=model_revision,
            container_digest=_text(report, "image_digest"),
            training_config={
                "method": "RERANKER",
                "max_sequence_length": 128,
                "precision": "bfloat16",
                "per_device_eval_batch_size": 16,
                "current_run_model_training_performed": False,
                "source_candidate_actual_gpu_trained": True,
                "actual_gpu_runtime_probe_completed": True,
                "runtime_probe_evidence_chain_sha256": (
                    runtime_probe.evidence_chain_sha256
                ),
                "source_v4_run_id": _text(predecessor, "run_id"),
                "source_v4_bundle_sha256": _text(
                    candidate,
                    "source_v4_bundle_sha256",
                ),
                "query_augmentation": _text(candidate, "query_augmentation"),
                "score_calibration": _text(candidate, "score_calibration"),
                "calibration": calibration,
                "calibration_registry_is_not_current_run_training_data": True,
                "source_acceptance": outcome_path,
                "project_authorized_import": True,
            },
            distributed_profile={"strategy": "single_gpu", "world_size": 1},
            random_seeds=[42],
            hardware_topology={
                "accelerator": "NVIDIA_GPU",
                "count": 1,
                "name": _text(runtime, "gpu_name"),
                "memory_bytes": _integer(runtime, "gpu_total_memory_bytes"),
            },
            metrics={
                "recall_at_k": _number(evaluation, "candidate_recall_at_k"),
                "mrr": _number(evaluation, "candidate_mrr"),
                "ndcg_at_k": candidate_score,
                "primary_improvement": quality_delta,
                "p95_latency_ms": _number(evaluation, "candidate_p95_latency_ms"),
            },
            cost_summary={
                "gpu_hours": 0.0,
                "current_run_model_training_performed": False,
                "source": outcome_path,
            },
            mlflow_run_id=None,
        ),
        baseline=_ExperimentSpec(
            source_id=f"reranker-calibrated-baseline-{evidence_chain[:24]}",
            method="BASELINE",
            task_type="INDUSTRIAL_KNOWLEDGE_RERANKING",
            status="COMPLETED",
            base_model_id=_text(input_model, "model_id"),
            base_model_digest=_plain_digest(_text(input_model, "manifest_sha256")),
            tokenizer_digest=_plain_digest(_text(tokenizer, "sha256")),
            chat_template_digest="not-applicable:cross-encoder-reranker",
            git_commit=model_revision,
            container_digest=_text(report, "image_digest"),
            training_config={
                "method": "BASELINE",
                "actual_gpu_inference": True,
                "source_evaluation_id": evaluation_id,
            },
            distributed_profile={},
            random_seeds=[42],
            hardware_topology={},
            metrics={
                "recall_at_k": _number(evaluation, "baseline_recall_at_k"),
                "mrr": _number(evaluation, "baseline_mrr"),
                "ndcg_at_k": baseline_score,
                "p95_latency_ms": _number(evaluation, "baseline_p95_latency_ms"),
            },
            cost_summary={"source": outcome_path},
            mlflow_run_id=None,
        ),
        artifact=_ArtifactSpec(
            source_id=f"reranker-bundle-{_text(candidate, 'bundle_sha256')[:32]}",
            kind="reranker_model_bundle",
            object_key=candidate_directory.relative_to(root).as_posix(),
            content_hash=_text(candidate, "bundle_sha256"),
            size_bytes=_integer(candidate, "size_bytes"),
            metadata={
                "format": "directory",
                "serialization": "safetensors",
                "files": candidate["files"],
                "query_augmentation": _text(candidate, "query_augmentation"),
                "score_calibration": _text(candidate, "score_calibration"),
                "calibration_profile_id": _text(calibration, "profile_id"),
                "current_run_model_training_performed": False,
                "source_candidate_actual_gpu_trained": True,
            },
            source_size_recorded=True,
        ),
        evaluation=_EvaluationSpec(
            source_id=evaluation_id,
            suite_source_id="industrial-reranker-final-gold-v5",
            policy_source_id="reranker-calibrated-enterprise-value-v1",
            primary_metric="ndcg_at_k",
            candidate_score=candidate_score,
            baseline_score=baseline_score,
            quality_delta=quality_delta,
            ci_low=quality_delta,
            ci_high=quality_delta,
            latency_improvement=max(0.0, 1.0 - latency_ratio),
            cost_improvement=0.0,
            hard_gates=hard_gates,
            slice_metrics={
                "calibrated_terminology": {
                    "candidate_recall_at_k": _number(
                        evaluation,
                        "candidate_recall_at_k",
                    ),
                    "candidate_mrr": _number(evaluation, "candidate_mrr"),
                    "candidate_ndcg_at_k": candidate_score,
                    "baseline_ndcg_at_k": baseline_score,
                    "registry_resolution_count": _integer(
                        evaluation,
                        "registry_resolution_count",
                    ),
                    "actual_case_count": _integer(dataset, "final_gold_count"),
                }
            },
            report_hash=_plain_digest(_text(formal_report, "sha256")),
            thresholds={
                "primary_improvement_min": 0.10,
                "candidate_recall_at_k_min": 0.90,
                "candidate_mrr_min": 0.85,
                "candidate_ndcg_at_k_min": 0.85,
                "candidate_latency_ratio_max": 4.0,
                "quality_lift_min": 0.10,
                "quality_tolerance": 0.0,
                "efficiency_improvement_min": 0.0,
                "bootstrap_iterations": 0.0,
            },
            target_profile="RETRIEVAL_COMPONENT",
            runner_config=runner_config,
            runner_config_hash=_digest(runner_config),
            case_object_key=_text(formal_report, "path"),
            case_content_hash=_text(formal_report, "sha256"),
            case_size_bytes=_integer(formal_report, "size_bytes"),
            case_metadata={
                "format": "json",
                "observations_path": _text(observations, "path"),
                "observations_sha256": _text(observations, "sha256"),
                "source_decision": _text(report, "decision"),
                "formal_evaluation_limit": 1,
            },
        ),
        suite_name="project-authorized-reranker-calibrated-evaluation",
        suite_version="5.0.0",
        suite_slices={"calibrated_terminology": _integer(dataset, "final_gold_count")},
        supply_chain=_SupplyChainSpec(
            image_repository=readiness.runtime_image.repository,
            image_digest=readiness.runtime_image.digest,
            source_revision=model_revision,
            source_repository=(f"https://huggingface.co/{_text(input_model, 'model_id')}"),
            source_evidence_hash=runtime_probe.evidence_chain_sha256,
        ),
    )


def _rul_bundle(root: Path, evidence: EnterpriseRuntimeEvidence) -> _ImportBundle:
    promotion_path, rollout_path = evidence.source_paths
    promotion = _json(root, promotion_path)
    rollout = _json(root, rollout_path)
    candidate = _mapping(promotion, "candidate")
    baseline = _mapping(promotion, "baseline")
    frozen = _mapping(promotion, "frozen_data")
    evaluation = _mapping(promotion, "independent_gold_evaluation")
    selection = _mapping(promotion, "selection")
    runtime_image = _mapping(rollout, "runtime_image")
    if (
        _text(candidate, "experiment_id") != evidence.candidate_experiment_id
        or _text(selection, "status") != "SELECTED_FOR_LOCAL_KSERVE_PROMOTION"
        or not bool(selection.get("all_hard_gates_passed"))
        or not bool(promotion.get("actual_gpu_execution"))
        or not bool(rollout.get("training_and_gold_evaluation_gpu_execution"))
        or _text(rollout, "evidence_chain_sha256") != evidence.evidence_chain_sha256
    ):
        raise EnterpriseModelAssetImportConflict("RUL_IMPORT_EVIDENCE_BINDING_CHANGED")
    training_path = _relative_source_path(root, _text(frozen, "training_snapshot"))
    gold_path = _relative_source_path(root, _text(frozen, "gold_snapshot"))
    model_directory = _inside(root, _text(candidate, "artifact_directory"), directory=True)
    model_size = sum(item.stat().st_size for item in model_directory.iterdir() if item.is_file())
    generated_at = _timestamp(promotion.get("generated_at"))
    compiled = _mapping(candidate, "compiled_config")
    training_metrics = _float_mapping(candidate, "training_metrics")
    candidate_stats = _mapping(evaluation, "candidate")
    baseline_stats = _mapping(evaluation, "baseline")
    baseline_mae = _number(baseline_stats, "median_absolute_error_minutes")
    candidate_mae = _number(candidate_stats, "median_absolute_error_minutes")
    relative_improvement = max(0.0, 1.0 - candidate_mae / baseline_mae)
    source_revision = _text(promotion, "evidence_chain_sha256")[:40]
    image_tag = _text(runtime_image, "tag")
    source_paths = (promotion_path, rollout_path, training_path, gold_path)
    return _ImportBundle(
        component="RUL",
        evidence=evidence,
        generated_at=generated_at,
        source_paths=source_paths,
        source_hashes=_source_hashes(root, source_paths),
        source_classification=_text(promotion, "classification"),
        source_decision=_text(selection, "status"),
        training_snapshot=_SnapshotSpec(
            source_id=_text(_mapping(candidate, "runtime"), "dataset_snapshot_id"),
            manifest_hash=_plain_digest(_text(frozen, "training_snapshot_sha256")),
            row_count=_integer(frozen, "train_rows") + _integer(frozen, "validation_rows"),
            split_counts={
                "train": _integer(frozen, "train_rows"),
                "validation": _integer(frozen, "validation_rows"),
            },
            source_path=training_path,
        ),
        evaluation_snapshot=_SnapshotSpec(
            source_id=f"rul-gold-{_plain_digest(_text(frozen, 'gold_snapshot_sha256'))[:24]}",
            manifest_hash=_plain_digest(_text(frozen, "gold_snapshot_sha256")),
            row_count=_integer(frozen, "gold_rows"),
            split_counts={"evaluation": _integer(frozen, "gold_rows")},
            source_path=gold_path,
        ),
        candidate=_ExperimentSpec(
            source_id=evidence.candidate_experiment_id,
            method="RUL_TRANSFORMER",
            task_type="PREDICTIVE_MAINTENANCE_RUL",
            status="COMPLETED",
            base_model_id="IndustrialRulTransformer",
            base_model_digest="architecture:rul-transformer-v1",
            tokenizer_digest="not-applicable:rul-numeric-sequence",
            chat_template_digest="not-applicable:rul-numeric-sequence",
            git_commit=source_revision,
            container_digest=_text(runtime_image, "id"),
            training_config={
                **compiled,
                "source_revision_basis": "PROMOTION_EVIDENCE_CHAIN_PREFIX",
                "project_authorized_import": True,
            },
            distributed_profile={"strategy": "single_gpu", "world_size": 1},
            random_seeds=[int(compiled["seed"]), int(compiled["data_seed"])],
            hardware_topology={
                "accelerator": "NVIDIA_GPU",
                "count": _integer(_mapping(candidate, "runtime"), "gpu_count"),
                "name": _text(_mapping(candidate, "runtime"), "gpu_name"),
                "memory_bytes": _integer(_mapping(candidate, "runtime"), "gpu_total_memory_bytes"),
            },
            metrics=training_metrics,
            cost_summary={
                "gpu_hours": training_metrics.get("gpu_hours", 0.0),
                "source": promotion_path,
            },
            mlflow_run_id=None,
        ),
        baseline=_ExperimentSpec(
            source_id=_text(baseline, "experiment_id"),
            method="RUL_EMPIRICAL_BASELINE",
            task_type="PREDICTIVE_MAINTENANCE_RUL",
            status="PLANNED",
            base_model_id="IndustrialRulEmpiricalBaseline",
            base_model_digest="architecture:rul-empirical-baseline-v1",
            tokenizer_digest="not-applicable:rul-numeric-sequence",
            chat_template_digest="not-applicable:rul-numeric-sequence",
            git_commit=source_revision,
            container_digest=_text(runtime_image, "id"),
            training_config={
                "empirical_quantiles_minutes": baseline["empirical_quantiles_minutes"],
                "source_manifest_hash": _plain_digest(_text(frozen, "training_snapshot_sha256")),
            },
            distributed_profile={},
            random_seeds=[42],
            hardware_topology={},
            metrics={},
            cost_summary={},
            mlflow_run_id=None,
        ),
        artifact=_ArtifactSpec(
            source_id=(
                f"rul-artifact-{_plain_digest(_text(candidate, 'artifact_content_hash'))[:32]}"
            ),
            kind="rul_transformer_bundle",
            object_key=model_directory.relative_to(root).as_posix(),
            content_hash=_text(candidate, "artifact_content_hash"),
            size_bytes=model_size,
            metadata={
                "format": "directory",
                "serialization": "safetensors",
                "files": candidate["files"],
            },
            source_size_recorded=True,
        ),
        evaluation=_EvaluationSpec(
            source_id=f"rul-evaluation-{_text(promotion, 'evidence_chain_sha256')[:32]}",
            suite_source_id=(
                f"rul-suite-{_plain_digest(_text(frozen, 'gold_snapshot_sha256'))[:32]}"
            ),
            policy_source_id="rul-project-promotion-policy-v1",
            primary_metric="rul_relative_accuracy",
            candidate_score=relative_improvement,
            baseline_score=0.0,
            quality_delta=relative_improvement,
            ci_low=0.0,
            ci_high=0.0,
            latency_improvement=0.0,
            cost_improvement=0.0,
            hard_gates=_bool_mapping(evaluation, "hard_gate_results"),
            slice_metrics={
                "rul": {
                    "candidate": candidate_stats,
                    "baseline": baseline_stats,
                    "score_projection": "MEDIAN_MAE_RELATIVE_IMPROVEMENT",
                    "actual_case_count": _integer(evaluation, "sample_count"),
                }
            },
            report_hash=_digest(evaluation),
            thresholds={
                **_float_mapping(evaluation, "thresholds"),
                "quality_lift_min": 0.0,
                "quality_tolerance": 0.01,
                "efficiency_improvement_min": 0.0,
                "bootstrap_iterations": 1000.0,
            },
            target_profile="RUL_COMPONENT",
            runner_config={
                "backend": _text(evaluation, "backend"),
                "precision": _text(compiled, "precision"),
                "actual_case_count": _integer(evaluation, "sample_count"),
            },
            runner_config_hash=_digest(
                {
                    "compiled_config_digest": _text(candidate, "compiled_config_digest"),
                    "gold_snapshot_sha256": _text(frozen, "gold_snapshot_sha256"),
                }
            ),
            case_object_key=promotion_path,
            case_content_hash=_source_hashes(root, (promotion_path,))[promotion_path],
            case_size_bytes=_inside(root, promotion_path).stat().st_size,
            case_metadata={"format": "json", "backend": _text(evaluation, "backend")},
        ),
        suite_name="project-authorized-rul-evaluation",
        suite_version="1.0.0",
        suite_slices={"rul_gold": _integer(evaluation, "sample_count")},
        supply_chain=_SupplyChainSpec(
            image_repository=_image_repository(image_tag),
            image_digest=_text(runtime_image, "id"),
            source_revision=source_revision,
            source_repository="https://project.local/industrial-ops-agent",
            source_evidence_hash=_text(rollout, "evidence_chain_sha256"),
        ),
    )


def _component_evidence(
    reader: EnterpriseProjectAdoptionReader,
    component: EnterpriseModelComponent,
) -> EnterpriseRuntimeEvidence:
    matches = [item for item in reader.runtime_evidence() if item.component == component]
    if len(matches) != 1:
        raise EnterpriseModelAssetImportConflict("ENTERPRISE_RUNTIME_EVIDENCE_NOT_FOUND")
    return matches[0]


def _require_import_integrity(record: EnterpriseModelAssetImportRecord) -> None:
    if (
        record.status != "ACTIVE"
        or record.operational_classification != OPERATIONAL_CLASSIFICATION
        or record.release_scope != "STAGING_ONLY"
        or record.import_hash != enterprise_model_import_hash(record)
    ):
        raise EnterpriseModelAssetImportConflict(
            "ENTERPRISE_MODEL_IMPORT_INTEGRITY_FAILED",
            record.version,
        )


def _json(root: Path, relative_path: str) -> dict[str, Any]:
    path = _inside(root, relative_path)
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise EnterpriseModelAssetImportConflict("MODEL_IMPORT_SOURCE_ROOT_INVALID")
    return cast(dict[str, Any], decoded)


def _inside(root: Path, value: str, *, directory: bool = False) -> Path:
    requested = Path(value)
    path = (
        requested.resolve(strict=True)
        if requested.is_absolute()
        else (root / requested).resolve(strict=True)
    )
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise EnterpriseModelAssetImportConflict("MODEL_IMPORT_SOURCE_ESCAPED_ROOT") from exc
    if directory and not path.is_dir():
        raise EnterpriseModelAssetImportConflict("MODEL_IMPORT_SOURCE_DIRECTORY_MISSING")
    if not directory and not path.is_file():
        raise EnterpriseModelAssetImportConflict("MODEL_IMPORT_SOURCE_FILE_MISSING")
    return path


def _relative_source_path(root: Path, value: str) -> str:
    return _inside(root, value).relative_to(root).as_posix()


def _source_hashes(root: Path, paths: tuple[str, ...]) -> dict[str, str]:
    return {
        path: sha256(_inside(root, path).read_bytes()).hexdigest() for path in sorted(set(paths))
    }


def _rows(document: dict[str, Any], label: str) -> list[dict[str, Any]]:
    value = document.get("rows")
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, dict) for item in value)
    ):
        raise EnterpriseModelAssetImportConflict(f"MODEL_IMPORT_ROWS_INVALID:{label}")
    return [cast(dict[str, Any], item) for item in value]


def _split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        split = row.get("split")
        if not isinstance(split, str) or not split:
            raise EnterpriseModelAssetImportConflict("MODEL_IMPORT_ROW_SPLIT_INVALID")
        counts[split] = counts.get(split, 0) + 1
    return counts


def _mapping(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise EnterpriseModelAssetImportConflict(f"MODEL_IMPORT_FIELD_INVALID:{key}")
    return cast(dict[str, Any], value)


def _input_component(document: dict[str, Any], role: str) -> dict[str, Any]:
    value = document.get("input_components")
    if not isinstance(value, list):
        raise EnterpriseModelAssetImportConflict("MODEL_IMPORT_FIELD_INVALID:input_components")
    matches = [item for item in value if isinstance(item, dict) and item.get("role") == role]
    if len(matches) != 1:
        raise EnterpriseModelAssetImportConflict(f"MODEL_IMPORT_INPUT_COMPONENT_INVALID:{role}")
    return cast(dict[str, Any], matches[0])


def _listed_file_sha256(
    document: dict[str, Any],
    key: str,
    expected_path: str,
) -> str:
    value = document.get(key)
    if not isinstance(value, list):
        raise EnterpriseModelAssetImportConflict(f"MODEL_IMPORT_FIELD_INVALID:{key}")
    matches = [
        item for item in value if isinstance(item, dict) and item.get("path") == expected_path
    ]
    if len(matches) != 1:
        raise EnterpriseModelAssetImportConflict(
            f"MODEL_IMPORT_FILE_BINDING_INVALID:{expected_path}"
        )
    return _plain_digest(_text(cast(dict[str, Any], matches[0]), "sha256"))


def _file_binding_by_name(
    document: dict[str, Any],
    key: str,
    expected_name: str,
) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, list):
        raise EnterpriseModelAssetImportConflict(f"MODEL_IMPORT_FIELD_INVALID:{key}")
    matches = [
        cast(dict[str, Any], item)
        for item in value
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and Path(cast(str, item["path"])).name == expected_name
    ]
    if len(matches) != 1:
        raise EnterpriseModelAssetImportConflict(
            f"MODEL_IMPORT_FILE_BINDING_INVALID:{expected_name}"
        )
    return matches[0]


def _text(document: dict[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise EnterpriseModelAssetImportConflict(f"MODEL_IMPORT_FIELD_INVALID:{key}")
    return value


def _integer(document: dict[str, Any], key: str) -> int:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EnterpriseModelAssetImportConflict(f"MODEL_IMPORT_FIELD_INVALID:{key}")
    return value


def _number(document: dict[str, Any], key: str) -> float:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EnterpriseModelAssetImportConflict(f"MODEL_IMPORT_FIELD_INVALID:{key}")
    return float(value)


def _bool_mapping(document: dict[str, Any], key: str) -> dict[str, bool]:
    value = _mapping(document, key)
    if not value or any(not isinstance(item, bool) for item in value.values()):
        raise EnterpriseModelAssetImportConflict(f"MODEL_IMPORT_FIELD_INVALID:{key}")
    return {str(name): item for name, item in value.items()}


def _float_mapping(document: dict[str, Any], key: str) -> dict[str, float]:
    value = _mapping(document, key)
    result: dict[str, float] = {}
    for name, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            continue
        result[str(name)] = float(item)
    return result


def _int_mapping(document: dict[str, Any], key: str) -> dict[str, int]:
    value = _mapping(document, key)
    result: dict[str, int] = {}
    for name, item in value.items():
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise EnterpriseModelAssetImportConflict(f"MODEL_IMPORT_FIELD_INVALID:{key}")
        result[str(name)] = item
    if not result:
        raise EnterpriseModelAssetImportConflict(f"MODEL_IMPORT_FIELD_INVALID:{key}")
    return result


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise EnterpriseModelAssetImportConflict("MODEL_IMPORT_TIMESTAMP_INVALID")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise EnterpriseModelAssetImportConflict("MODEL_IMPORT_TIMESTAMP_INVALID")
    return parsed.astimezone(UTC)


def _stable_id(prefix: str, tenant_id: str, source_id: str | None) -> str:
    value = source_id or "none"
    digest = sha256(f"{tenant_id}\0{value}".encode()).hexdigest()[:32]
    return f"{prefix}-{digest}"


def _short_tenant(tenant_id: str) -> str:
    return sha256(tenant_id.encode()).hexdigest()[:12]




def _tagged_digest(document: object) -> str:
    return "sha256:" + _digest(document)


def _plain_digest(value: str) -> str:
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise EnterpriseModelAssetImportConflict("MODEL_IMPORT_DIGEST_INVALID")
    return digest


def _image_repository(image: str) -> str:
    without_digest = image.split("@", 1)[0]
    tail = without_digest.rsplit("/", 1)[-1]
    if ":" in tail:
        return without_digest.rsplit(":", 1)[0]
    return without_digest
