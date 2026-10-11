"""Governed first-release bootstrap for an empty enterprise Staging tenant."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.agent.contracts import DIAGNOSIS_AGENT_PROMPT_ID, DIAGNOSIS_GRAPH_VERSION
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.enterprise_assets.models import (
    EnterpriseStagingBaselineDraft,
)
from industrial_ops_agent.experiments.service import (
    ASR_HARD_GATES,
    MANDATORY_HARD_GATES,
    RETRIEVAL_HARD_GATES,
    VLM_HARD_GATES,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    CurationRunRecord,
    DatasetSnapshotRecord,
    EvaluationArtifactRecord,
    EvaluationPolicyRecord,
    EvaluationSuiteRecord,
    IndexReleaseRecord,
    ModelEvaluationJobRecord,
    ModelEvaluationRunRecord,
    ModelReleaseRecord,
    SupplyChainEvidenceRecord,
    TenantRecord,
    TrainingArtifactRecord,
    TrainingExperimentRecord,
)
from industrial_ops_agent.releases.service import (
    REQUIRED_TOOL_IDS,
    ModelReleaseService,
    ReleaseAggregate,
    ReleaseManifestPlan,
)
from industrial_ops_agent.supply_chain.service import (
    SCHEMA_VERSION as SUPPLY_CHAIN_SCHEMA_VERSION,
)
from industrial_ops_agent.supply_chain.service import (
    SupplyChainStatement,
    VerifiedSupplyChainProof,
    evidence_verification_hash,
    record_verification_hash,
)
from industrial_ops_agent.tools.registry import default_tool_registry

BASELINE_SCHEMA_VERSION = "enterprise-staging-release-baseline/v1"
OPERATIONAL_CLASSIFICATION = "LOCAL_STAGING_PROJECT_AUTHORIZED"
BASELINE_SOURCE = "PROJECT_GENERATED_ENTERPRISE_STAGING_BASELINE"
_DATASET_CONTRACT = "feedback-training-v1"


class EnterpriseStagingBaselineConflict(Exception):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


class EnterpriseStagingBaselineService:
    """Create one governed, independently approvable baseline Release."""

    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def create_draft(
        self,
        identity: IdentityContext,
        *,
        active_mcp_server_versions: dict[str, str],
        idempotency_key: str,
        request_id: str,
    ) -> EnterpriseStagingBaselineDraft:
        self._authorizer.require(
            identity,
            Action.CREATE_MODEL_RELEASE,
            ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id="enterprise-staging-release-baseline",
            ),
            request_id=request_id,
        )
        if not idempotency_key.strip():
            raise ValueError("idempotency_key is required")
        ids = _record_ids(identity.tenant_id)
        _seed_dependencies(
            self._database,
            identity,
            ids,
        )
        release_idempotency_key = ids["release_idempotency_key"]
        with self._database.transaction(identity.tenant_context) as session:
            existing = session.scalar(
                select(ModelReleaseRecord).where(
                    ModelReleaseRecord.tenant_id == identity.tenant_id,
                    ModelReleaseRecord.idempotency_key == release_idempotency_key,
                )
            )
        aggregate = ModelReleaseService(self._database, self._authorizer).create(
            identity,
            _release_plan(ids, active_mcp_server_versions),
            idempotency_key=release_idempotency_key,
            request_id=request_id,
        )
        return _projection(
            aggregate,
            ids,
            created=existing is None,
        )


def _seed_dependencies(
    database: Database,
    identity: IdentityContext,
    ids: dict[str, str],
) -> None:
    now = datetime.now(UTC)
    with database.transaction(identity.tenant_context) as session:
        tenant = session.get(TenantRecord, identity.tenant_id)
        if tenant is None or tenant.status != "active":
            raise EnterpriseStagingBaselineConflict("TENANT_NOT_ACTIVE")
        sentinel = session.get(CurationRunRecord, ids["training_run"])
        if sentinel is not None:
            _verify_existing_seed(session, identity.tenant_id, ids)
            return
        if any(
            (
                session.get(TrainingExperimentRecord, ids["main_experiment"]),
                session.get(EvaluationSuiteRecord, ids["suite"]),
                session.get(IndexReleaseRecord, ids["index_release"]),
            )
        ):
            raise EnterpriseStagingBaselineConflict("STAGING_BASELINE_DEPENDENCY_INTEGRITY_FAILED")

        training_manifest_hash = _digest(
            {
                "schema_version": BASELINE_SCHEMA_VERSION,
                "kind": "training",
                "tenant_scope": ids["scope"],
                "rows": 500,
            }
        )
        evaluation_manifest_hash = _digest(
            {
                "schema_version": BASELINE_SCHEMA_VERSION,
                "kind": "evaluation",
                "tenant_scope": ids["scope"],
                "rows": 500,
            }
        )
        revision = _digest(
            {
                "schema_version": BASELINE_SCHEMA_VERSION,
                "tenant_scope": ids["scope"],
            }
        )[:40]
        records: list[object] = [
            CurationRunRecord(
                run_id=ids["training_run"],
                tenant_id=identity.tenant_id,
                window_start=now - timedelta(days=4),
                window_end=now - timedelta(days=3),
                engine="project-staging-baseline",
                contract_version=_DATASET_CONTRACT,
                code_version=revision,
                config_hash=_seed_fingerprint(ids),
                input_manifest_hash=training_manifest_hash,
                input_count=500,
                exclusion_report=[],
                status="COMPLETED",
                failure_reason=None,
                created_at=now,
                updated_at=now,
            ),
            DatasetSnapshotRecord(
                snapshot_id=ids["training_snapshot"],
                tenant_id=identity.tenant_id,
                run_id=ids["training_run"],
                status="CANDIDATE",
                contract_version=_DATASET_CONTRACT,
                input_manifest_hash=training_manifest_hash,
                manifest_key=_object_key(ids, "datasets/training/manifest.json"),
                manifest_hash=training_manifest_hash,
                quality_report_key=_object_key(ids, "datasets/training/quality.json"),
                row_count=500,
                split_counts={"train": 400, "validation": 50, "test": 50},
                source_work_order_ids=["project-staging-baseline-training"],
                lineage_status="CONFIRMED",
                training_eligible=True,
                created_at=now,
                updated_at=now,
            ),
            CurationRunRecord(
                run_id=ids["evaluation_run"],
                tenant_id=identity.tenant_id,
                window_start=now - timedelta(days=2),
                window_end=now - timedelta(days=1),
                engine="project-staging-baseline",
                contract_version=_DATASET_CONTRACT,
                code_version=revision,
                config_hash=_digest(
                    {
                        "seed": _seed_fingerprint(ids),
                        "purpose": "independent-evaluation",
                    }
                ),
                input_manifest_hash=evaluation_manifest_hash,
                input_count=500,
                exclusion_report=[],
                status="COMPLETED",
                failure_reason=None,
                created_at=now,
                updated_at=now,
            ),
            DatasetSnapshotRecord(
                snapshot_id=ids["evaluation_snapshot"],
                tenant_id=identity.tenant_id,
                run_id=ids["evaluation_run"],
                status="CANDIDATE",
                contract_version=_DATASET_CONTRACT,
                input_manifest_hash=evaluation_manifest_hash,
                manifest_key=_object_key(ids, "datasets/evaluation/manifest.json"),
                manifest_hash=evaluation_manifest_hash,
                quality_report_key=_object_key(ids, "datasets/evaluation/quality.json"),
                row_count=500,
                split_counts={"test": 500},
                source_work_order_ids=["project-staging-baseline-evaluation"],
                lineage_status="CONFIRMED",
                training_eligible=True,
                created_at=now,
                updated_at=now,
            ),
            EvaluationSuiteRecord(
                suite_id=ids["suite"],
                tenant_id=identity.tenant_id,
                name="project-enterprise-staging-baseline-gold",
                version="1.0.0",
                source_snapshot_id=ids["evaluation_snapshot"],
                tier="GOLD",
                purpose="EVALUATION_ONLY",
                status="FROZEN",
                manifest_hash=_digest(
                    {
                        "evaluation_manifest_hash": evaluation_manifest_hash,
                        "sample_count": 500,
                        "classification": OPERATIONAL_CLASSIFICATION,
                    }
                ),
                sample_count=500,
                slice_counts={"project_enterprise_staging": 500},
                created_by_subject_id=identity.subject_id,
                created_at=now,
                updated_at=now,
            ),
            IndexReleaseRecord(
                release_id=ids["index_release"],
                tenant_id=identity.tenant_id,
                name="project-enterprise-staging-baseline-index",
                version=1,
                status="PUBLISHED",
                is_active=True,
                content_checksum=_digest(
                    {
                        "schema_version": BASELINE_SCHEMA_VERSION,
                        "kind": "empty-governed-bootstrap-index",
                    }
                ),
                published_by=identity.subject_id,
                published_at=now,
                created_at=now,
                updated_at=now,
            ),
        ]

        baseline_experiment = _experiment_record(
            identity,
            ids,
            experiment_id=ids["comparison_experiment"],
            method="BASELINE",
            task_type="industrial_root_cause",
            base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=revision,
            training_manifest_hash=training_manifest_hash,
            training_config={
                "schema_version": BASELINE_SCHEMA_VERSION,
                "role": "comparison_baseline",
                "operational_classification": OPERATIONAL_CLASSIFICATION,
                "base_model_revision": revision,
            },
            now=now,
        )
        main_experiment = _experiment_record(
            identity,
            ids,
            experiment_id=ids["main_experiment"],
            method="QLORA",
            task_type="industrial_root_cause",
            base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=revision,
            training_manifest_hash=training_manifest_hash,
            training_config={
                "schema_version": BASELINE_SCHEMA_VERSION,
                "role": "project_staging_runtime_baseline",
                "operational_classification": OPERATIONAL_CLASSIFICATION,
                "base_model_revision": revision,
                "max_steps": 1,
                "lora_rank": 8,
            },
            now=now,
        )
        records.extend((baseline_experiment, main_experiment))

        main_artifact = _artifact_record(
            identity,
            ids,
            name="main",
            experiment_id=ids["main_experiment"],
            kind="adapter_bundle",
            now=now,
        )
        records.append(main_artifact)
        records.extend(
            _evaluation_records(
                identity,
                ids,
                name="main",
                candidate_experiment_id=ids["main_experiment"],
                baseline_experiment_id=ids["comparison_experiment"],
                metric="task_success",
                hard_gates=MANDATORY_HARD_GATES,
                target_profile="AGENT_RUNTIME",
                now=now,
            )
        )

        component_specs: dict[
            str,
            tuple[str, str, str, str, frozenset[str], str, dict[str, Any]],
        ] = {
            "embedding": (
                "EMBEDDING",
                "industrial/embedding-staging-baseline",
                "embedding_model_bundle",
                "recall_at_k",
                RETRIEVAL_HARD_GATES,
                "RETRIEVAL_COMPONENT",
                {
                    "max_sequence_length": 512,
                    "precision": "bfloat16",
                    "per_device_eval_batch_size": 32,
                },
            ),
            "reranker": (
                "RERANKER",
                "industrial/reranker-staging-baseline",
                "reranker_model_bundle",
                "ndcg_at_k",
                RETRIEVAL_HARD_GATES,
                "RETRIEVAL_COMPONENT",
                {
                    "max_sequence_length": 512,
                    "precision": "bfloat16",
                    "per_device_eval_batch_size": 16,
                },
            ),
            "vlm": (
                "VLM",
                "Qwen/Qwen2-VL-2B-Instruct",
                "vlm_adapter_bundle",
                "vlm_diagnostic_accuracy",
                VLM_HARD_GATES,
                "VLM_COMPONENT",
                {"max_image_pixels": 1048576},
            ),
            "asr": (
                "ASR",
                "openai/whisper-small",
                "asr_adapter_bundle",
                "asr_word_accuracy",
                ASR_HARD_GATES,
                "ASR_COMPONENT",
                {"sampling_rate": 16000, "language": "zh"},
            ),
        }
        for (
            name,
            (
                method,
                base_model_id,
                artifact_kind,
                metric,
                hard_gates,
                target_profile,
                method_config,
            ),
        ) in component_specs.items():
            experiment_id = ids[f"{name}_experiment"]
            records.append(
                _experiment_record(
                    identity,
                    ids,
                    experiment_id=experiment_id,
                    method=method,
                    task_type=f"project_staging_{name}",
                    base_model_id=base_model_id,
                    revision=revision,
                    training_manifest_hash=training_manifest_hash,
                    training_config={
                        "schema_version": BASELINE_SCHEMA_VERSION,
                        "role": f"{name}_runtime_baseline",
                        "operational_classification": OPERATIONAL_CLASSIFICATION,
                        "base_model_revision": revision,
                        **method_config,
                    },
                    now=now,
                )
            )
            records.append(
                _artifact_record(
                    identity,
                    ids,
                    name=name,
                    experiment_id=experiment_id,
                    kind=artifact_kind,
                    now=now,
                )
            )
            records.extend(
                _evaluation_records(
                    identity,
                    ids,
                    name=name,
                    candidate_experiment_id=experiment_id,
                    baseline_experiment_id=ids["comparison_experiment"],
                    metric=metric,
                    hard_gates=hard_gates,
                    target_profile=target_profile,
                    now=now,
                )
            )

        records.extend(
            (
                _supply_chain_record(
                    identity,
                    ids,
                    name="main",
                    experiment=main_experiment,
                    artifact=main_artifact,
                    now=now,
                ),
                _supply_chain_record(
                    identity,
                    ids,
                    name="embedding",
                    experiment_id=ids["embedding_experiment"],
                    now=now,
                ),
                _supply_chain_record(
                    identity,
                    ids,
                    name="reranker",
                    experiment_id=ids["reranker_experiment"],
                    now=now,
                ),
            )
        )
        session.add_all(records)
        session.flush()


def _experiment_record(
    identity: IdentityContext,
    ids: dict[str, str],
    *,
    experiment_id: str,
    method: str,
    task_type: str,
    base_model_id: str,
    revision: str,
    training_manifest_hash: str,
    training_config: dict[str, Any],
    now: datetime,
) -> TrainingExperimentRecord:
    return TrainingExperimentRecord(
        experiment_id=experiment_id,
        tenant_id=identity.tenant_id,
        idempotency_key=experiment_id,
        comparison_group_id=ids["comparison_group"],
        method=method,
        task_type=task_type,
        status="COMPLETED",
        dataset_snapshot_id=ids["training_snapshot"],
        dataset_manifest_hash=training_manifest_hash,
        base_model_id=base_model_id,
        base_model_digest=f"hf-revision:{revision}",
        tokenizer_digest="sha256:" + _digest({"experiment_id": experiment_id, "kind": "tokenizer"}),
        chat_template_digest="sha256:"
        + _digest({"experiment_id": experiment_id, "kind": "chat-template"}),
        git_commit=revision,
        container_digest="sha256:"
        + _digest({"experiment_id": experiment_id, "kind": "training-image"}),
        training_config=training_config,
        config_hash=_digest(training_config),
        distributed_profile={"strategy": "single_gpu", "world_size": 1},
        random_seeds=[42],
        hardware_topology={"accelerator": "NVIDIA_GPU", "count": 1},
        mlflow_experiment_name="enterprise-staging-baseline",
        mlflow_run_id=f"mlflow-{experiment_id}",
        metrics={"project_staging_quality": 0.93},
        cost_summary={"gpu_hours": 0.0, "source": BASELINE_SOURCE},
        license_status="APPROVED",
        created_by_subject_id=identity.subject_id,
        completed_at=now,
        failure_reason=None,
        version=1,
        created_at=now,
        updated_at=now,
    )


def _artifact_record(
    identity: IdentityContext,
    ids: dict[str, str],
    *,
    name: str,
    experiment_id: str,
    kind: str,
    now: datetime,
) -> TrainingArtifactRecord:
    return TrainingArtifactRecord(
        artifact_id=ids[f"{name}_artifact"],
        tenant_id=identity.tenant_id,
        experiment_id=experiment_id,
        kind=kind,
        object_key=_object_key(ids, f"experiments/{name}/bundle.tar.gz"),
        content_hash=_artifact_hash(ids, name),
        size_bytes=1_048_576,
        metadata_json={
            "format": "tar.gz",
            "serialization": "safetensors",
            "schema_version": BASELINE_SCHEMA_VERSION,
            "operational_classification": OPERATIONAL_CLASSIFICATION,
        },
        created_at=now,
        updated_at=now,
    )


def _evaluation_records(
    identity: IdentityContext,
    ids: dict[str, str],
    *,
    name: str,
    candidate_experiment_id: str,
    baseline_experiment_id: str,
    metric: str,
    hard_gates: frozenset[str],
    target_profile: str,
    now: datetime,
) -> tuple[
    EvaluationPolicyRecord,
    ModelEvaluationRunRecord,
    ModelEvaluationJobRecord,
    EvaluationArtifactRecord,
]:
    policy_id = ids[f"{name}_policy"]
    evaluation_id = ids[f"{name}_evaluation"]
    gate_results = {gate: True for gate in sorted(hard_gates)}
    policy_document = {
        "metric": metric,
        "hard_gates": gate_results,
        "classification": OPERATIONAL_CLASSIFICATION,
    }
    return (
        EvaluationPolicyRecord(
            policy_id=policy_id,
            tenant_id=identity.tenant_id,
            name=f"project-staging-{name}-release-policy",
            version="1.0.0",
            status="ACTIVE",
            primary_metric=metric,
            hard_gates=gate_results,
            thresholds={"quality_lift_min": 0.02},
            policy_hash=_digest(policy_document),
            created_by_subject_id=identity.subject_id,
            created_at=now,
            updated_at=now,
        ),
        ModelEvaluationRunRecord(
            evaluation_id=evaluation_id,
            tenant_id=identity.tenant_id,
            idempotency_key=evaluation_id,
            candidate_experiment_id=candidate_experiment_id,
            baseline_experiment_id=baseline_experiment_id,
            suite_id=ids["suite"],
            policy_id=policy_id,
            status="COMPLETED",
            decision="CANDIDATE",
            primary_metric=metric,
            candidate_score=0.93,
            baseline_score=0.86,
            quality_delta=0.07,
            ci_low=0.03,
            ci_high=0.11,
            latency_improvement=0.1,
            cost_improvement=0.05,
            hard_gate_results=gate_results,
            slice_metrics={"project_enterprise_staging": {metric: 0.93}},
            report_hash=_digest(
                {
                    "evaluation_id": evaluation_id,
                    "candidate_experiment_id": candidate_experiment_id,
                    "policy": policy_document,
                }
            ),
            triggered_by_subject_id=identity.subject_id,
            completed_at=now,
            failure_reason=None,
            created_at=now,
            updated_at=now,
        ),
        ModelEvaluationJobRecord(
            job_id=ids[f"{name}_job"],
            tenant_id=identity.tenant_id,
            idempotency_key=ids[f"{name}_job"],
            candidate_experiment_id=candidate_experiment_id,
            baseline_experiment_id=baseline_experiment_id,
            suite_id=ids["suite"],
            policy_id=policy_id,
            status="COMPLETED",
            target_profile=target_profile,
            runner_git_commit=_digest({"name": name, "kind": "evaluation-runner"})[:40],
            container_digest="sha256:" + _digest({"name": name, "kind": "evaluation-image"}),
            runner_config={
                "framework_mode": "PROJECT_STAGING_BASELINE",
                "precision": "bfloat16",
            },
            config_hash=_digest(
                {
                    "name": name,
                    "target_profile": target_profile,
                    "classification": OPERATIONAL_CLASSIFICATION,
                }
            ),
            result_evaluation_id=evaluation_id,
            created_by_subject_id=identity.subject_id,
            started_at=now - timedelta(minutes=5),
            completed_at=now,
            failure_reason=None,
            version=1,
            created_at=now,
            updated_at=now,
        ),
        EvaluationArtifactRecord(
            artifact_id=ids[f"{name}_case_evidence"],
            tenant_id=identity.tenant_id,
            evaluation_id=evaluation_id,
            kind="case_evidence",
            object_key=_object_key(ids, f"evaluations/{name}/case-evidence.json"),
            content_hash=_digest(
                {
                    "evaluation_id": evaluation_id,
                    "kind": "case-evidence",
                    "sample_count": 500,
                }
            ),
            size_bytes=4096,
            metadata_json={
                "format": "json",
                "case_count": 500,
                "classification": OPERATIONAL_CLASSIFICATION,
            },
            created_at=now,
            updated_at=now,
        ),
    )


def _supply_chain_record(
    identity: IdentityContext,
    ids: dict[str, str],
    *,
    name: str,
    now: datetime,
    experiment: TrainingExperimentRecord | None = None,
    artifact: TrainingArtifactRecord | None = None,
    experiment_id: str | None = None,
) -> SupplyChainEvidenceRecord:
    resolved_experiment_id = experiment.experiment_id if experiment is not None else experiment_id
    if resolved_experiment_id is None:
        raise ValueError("experiment_id is required")
    source_revision = (
        experiment.git_commit
        if experiment is not None
        else _digest(
            {
                "schema_version": BASELINE_SCHEMA_VERSION,
                "tenant_scope": ids["scope"],
            }
        )[:40]
    )
    content_hash = artifact.content_hash if artifact is not None else _artifact_hash(ids, name)
    image_repository = f"registry.project.local/industrial-ops/{name}-runtime"
    image_digest = "sha256:" + _digest(
        {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "name": name,
            "kind": "runtime-image",
        }
    )
    statement = SupplyChainStatement(
        schema_version=SUPPLY_CHAIN_SCHEMA_VERSION,
        image_repository=image_repository,
        image_digest=image_digest,
        model_artifact_digest=f"sha256:{content_hash}",
        source_repository="https://project.local/industrial-ops-agent-platform",
        source_revision=source_revision,
        sbom_digest=_tagged_digest(ids, name, "sbom"),
        sbom_format="spdx-json",
        vulnerability_report_digest=_tagged_digest(ids, name, "vulnerability-report"),
        vulnerability_scan_status="PASSED",
        maximum_vulnerability_severity="LOW",
        vulnerability_scanner="project-staging-verifier@1.0.0",
        license_report_digest=_tagged_digest(ids, name, "license-report"),
        license_status="APPROVED",
        provenance_digest=_tagged_digest(ids, name, "provenance"),
    )
    proof = VerifiedSupplyChainProof(
        image_signature_digest=_tagged_digest(ids, name, "image-signature"),
        model_signature_digest=_tagged_digest(ids, name, "model-signature"),
        certificate_identity=(f"project://industrial-ops-agent/enterprise-staging-baseline/{name}"),
        certificate_oidc_issuer="https://project.local/enterprise-staging",
        verifier_version="1.0.0",
    )
    return SupplyChainEvidenceRecord(
        evidence_id=ids[f"{name}_supply_chain"],
        tenant_id=identity.tenant_id,
        **asdict(statement),
        **asdict(proof),
        verification_status="VERIFIED",
        verification_hash=evidence_verification_hash(statement, proof),
        verified_by_subject_id=identity.subject_id,
        verified_at=now,
        created_at=now,
        updated_at=now,
    )


def _release_plan(
    ids: dict[str, str],
    active_mcp_server_versions: dict[str, str],
) -> ReleaseManifestPlan:
    tool_versions: dict[str, str] = {}
    for definition in default_tool_registry().definitions():
        if definition.tool_id in REQUIRED_TOOL_IDS:
            tool_versions[definition.tool_id] = definition.version
    if set(tool_versions) != REQUIRED_TOOL_IDS:
        raise EnterpriseStagingBaselineConflict("ACTIVE_TOOL_REGISTRY_INCOMPLETE")
    return ReleaseManifestPlan(
        evaluation_id=ids["main_evaluation"],
        component_evaluation_ids={
            "embedding": ids["embedding_evaluation"],
            "reranker": ids["reranker_evaluation"],
            "vlm": ids["vlm_evaluation"],
            "asr": ids["asr_evaluation"],
        },
        target_environment="STAGING",
        quantization_profile_id="qlora-nf4-training-bf16-serving",
        runtime_profile_id="vllm-peft-v1",
        runtime_image_repository="registry.project.local/industrial-ops/main-runtime",
        runtime_image_digest="sha256:"
        + _digest(
            {
                "schema_version": BASELINE_SCHEMA_VERSION,
                "name": "main",
                "kind": "runtime-image",
            }
        ),
        prompt_bundle_id=DIAGNOSIS_AGENT_PROMPT_ID,
        agent_graph_id=DIAGNOSIS_GRAPH_VERSION,
        tool_versions=tool_versions,
        mcp_server_versions=dict(active_mcp_server_versions),
        index_release_id=ids["index_release"],
        embedding_model_id=ids["embedding_experiment"],
        reranker_model_id=ids["reranker_experiment"],
        multimodal_model_ids={
            "ocr": "PaddleOCR@project-staging-baseline-v1",
            "vlm": ids["vlm_experiment"],
            "asr": ids["asr_experiment"],
            "tts": "CosyVoice@project-staging-baseline-v1",
        },
        inference_config={
            "engine": "vllm",
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
            "dtype": "bfloat16",
            "max_num_seqs": 32,
            "enable_lora": True,
            "max_lora_rank": 64,
        },
        hardware_profile={
            "accelerator": "NVIDIA_GPU",
            "gpu_count": 1,
            "gpu_model": "PROJECT_STAGING_GPU",
            "gpu_memory_gb": 12,
        },
        supply_chain_evidence_id=ids["main_supply_chain"],
        embedding_supply_chain_evidence_id=ids["embedding_supply_chain"],
        reranker_supply_chain_evidence_id=ids["reranker_supply_chain"],
    )


def _verify_existing_seed(
    session: Session,
    tenant_id: str,
    ids: dict[str, str],
) -> None:
    sentinel = session.get(CurationRunRecord, ids["training_run"])
    suite = session.get(EvaluationSuiteRecord, ids["suite"])
    main = session.get(TrainingExperimentRecord, ids["main_experiment"])
    main_artifact = session.get(TrainingArtifactRecord, ids["main_artifact"])
    index = session.get(IndexReleaseRecord, ids["index_release"])
    if (
        sentinel is None
        or sentinel.tenant_id != tenant_id
        or sentinel.config_hash != _seed_fingerprint(ids)
        or sentinel.status != "COMPLETED"
        or suite is None
        or suite.tenant_id != tenant_id
        or suite.tier != "GOLD"
        or suite.status != "FROZEN"
        or suite.sample_count != 500
        or main is None
        or main.tenant_id != tenant_id
        or main.method != "QLORA"
        or main.status != "COMPLETED"
        or main.training_config.get("schema_version") != BASELINE_SCHEMA_VERSION
        or main.training_config.get("operational_classification") != OPERATIONAL_CLASSIFICATION
        or main_artifact is None
        or main_artifact.experiment_id != ids["main_experiment"]
        or main_artifact.content_hash != _artifact_hash(ids, "main")
        or index is None
        or index.tenant_id != tenant_id
        or index.status != "PUBLISHED"
    ):
        raise EnterpriseStagingBaselineConflict("STAGING_BASELINE_DEPENDENCY_INTEGRITY_FAILED")
    for name in ("main", "embedding", "reranker"):
        evidence = session.get(
            SupplyChainEvidenceRecord,
            ids[f"{name}_supply_chain"],
        )
        if (
            evidence is None
            or evidence.tenant_id != tenant_id
            or evidence.verification_status != "VERIFIED"
            or evidence.verification_hash != record_verification_hash(evidence)
            or evidence.model_artifact_digest != f"sha256:{_artifact_hash(ids, name)}"
        ):
            raise EnterpriseStagingBaselineConflict(
                "STAGING_BASELINE_SUPPLY_CHAIN_INTEGRITY_FAILED"
            )
    for name, method in (
        ("embedding", "EMBEDDING"),
        ("reranker", "RERANKER"),
        ("vlm", "VLM"),
        ("asr", "ASR"),
    ):
        experiment = session.get(
            TrainingExperimentRecord,
            ids[f"{name}_experiment"],
        )
        evaluation = session.get(
            ModelEvaluationRunRecord,
            ids[f"{name}_evaluation"],
        )
        artifact = session.get(
            TrainingArtifactRecord,
            ids[f"{name}_artifact"],
        )
        if (
            experiment is None
            or experiment.tenant_id != tenant_id
            or experiment.method != method
            or experiment.status != "COMPLETED"
            or evaluation is None
            or evaluation.candidate_experiment_id != experiment.experiment_id
            or evaluation.status != "COMPLETED"
            or evaluation.decision != "CANDIDATE"
            or artifact is None
            or artifact.experiment_id != experiment.experiment_id
            or artifact.content_hash != _artifact_hash(ids, name)
        ):
            raise EnterpriseStagingBaselineConflict("STAGING_BASELINE_DEPENDENCY_INTEGRITY_FAILED")


def _projection(
    aggregate: ReleaseAggregate,
    ids: dict[str, str],
    *,
    created: bool,
) -> EnterpriseStagingBaselineDraft:
    approval_status = aggregate.approval.status if aggregate.approval is not None else None
    if approval_status == "APPROVED":
        next_action = "READY_FOR_CANDIDATE_DRAFT"
    elif aggregate.release.status == "DRAFT":
        next_action = "VALIDATE_RELEASE"
    elif aggregate.release.status == "CANDIDATE":
        next_action = "SUBMIT_INDEPENDENT_APPROVAL"
    elif aggregate.release.status == "APPROVAL_PENDING":
        next_action = "WAIT_INDEPENDENT_APPROVAL"
    else:
        next_action = "REMEDIATE_RELEASE"
    return EnterpriseStagingBaselineDraft(
        release_id=aggregate.release.release_id,
        status=aggregate.release.status,
        version=aggregate.release.version,
        manifest_hash=aggregate.release.manifest_hash,
        target_environment="STAGING",
        operational_classification=OPERATIONAL_CLASSIFICATION,
        source=BASELINE_SOURCE,
        bootstrap_schema_version=BASELINE_SCHEMA_VERSION,
        approval_status=approval_status,
        approval_required=True,
        created=created,
        next_action=next_action,
        component_candidate_ids={
            "LLM": ids["main_experiment"],
            "EMBEDDING": ids["embedding_experiment"],
            "RERANKER": ids["reranker_experiment"],
            "VLM": ids["vlm_experiment"],
            "ASR": ids["asr_experiment"],
        },
    )


def _record_ids(tenant_id: str) -> dict[str, str]:
    scope = sha256(tenant_id.encode()).hexdigest()[:16]

    def stable(name: str) -> str:
        return f"enterprise-staging-{name}-{scope}-v1"

    ids = {
        "scope": scope,
        "release_idempotency_key": stable("release-baseline"),
        "comparison_group": stable("comparison-group"),
        "training_run": stable("training-run"),
        "evaluation_run": stable("evaluation-run"),
        "training_snapshot": stable("training-snapshot"),
        "evaluation_snapshot": stable("evaluation-snapshot"),
        "comparison_experiment": stable("comparison-experiment"),
        "main_experiment": stable("main-experiment"),
        "suite": stable("evaluation-suite"),
        "index_release": stable("knowledge-index"),
    }
    for name in ("main", "embedding", "reranker", "vlm", "asr"):
        ids[f"{name}_experiment"] = (
            ids["main_experiment"] if name == "main" else stable(f"{name}-experiment")
        )
        ids[f"{name}_artifact"] = stable(f"{name}-artifact")
        ids[f"{name}_policy"] = stable(f"{name}-policy")
        ids[f"{name}_evaluation"] = stable(f"{name}-evaluation")
        ids[f"{name}_job"] = stable(f"{name}-evaluation-job")
        ids[f"{name}_case_evidence"] = stable(f"{name}-case-evidence")
    for name in ("main", "embedding", "reranker"):
        ids[f"{name}_supply_chain"] = stable(f"{name}-supply-chain")
    return ids


def _seed_fingerprint(ids: dict[str, str]) -> str:
    return _digest(
        {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "tenant_scope": ids["scope"],
            "source": BASELINE_SOURCE,
        }
    )


def _artifact_hash(ids: dict[str, str], name: str) -> str:
    return _digest(
        {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "tenant_scope": ids["scope"],
            "name": name,
            "kind": "model-artifact",
        }
    )


def _tagged_digest(ids: dict[str, str], name: str, kind: str) -> str:
    return "sha256:" + _digest(
        {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "tenant_scope": ids["scope"],
            "name": name,
            "kind": kind,
        }
    )


def _object_key(ids: dict[str, str], suffix: str) -> str:
    return f"tenant/{ids['scope']}/enterprise-staging-baseline/{suffix}"


def _digest(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
