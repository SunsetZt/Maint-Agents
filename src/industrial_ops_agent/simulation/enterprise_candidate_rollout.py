"""Low-resource project-staging rollout acceptance for selected enterprise candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.opa import OpaDecision, OpaDecisionInput
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.deployment.service import (
    METRIC_THRESHOLDS,
    REQUIRED_METRICS,
    STAGE_POLICY,
    STAGE_TRAFFIC,
    DeploymentPlan,
    ModelDeploymentService,
    ObservationInput,
    ProviderResult,
    ProviderTarget,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    Base,
    ModelAliasRecord,
    ModelReleaseApprovalRecord,
    ModelReleaseRecord,
    ModelReleaseTransitionRecord,
    TenantRecord,
)
from industrial_ops_agent.security_audit import InMemorySecurityAuditSink, SecurityAuditor
from industrial_ops_agent.simulation.dpo_structured_citation_erratum import (
    verify_dpo_structured_citation_erratum,
)
from industrial_ops_agent.simulation.embedding_calibrated_value_lab import (
    verify_calibrated_embedding_value,
)
from industrial_ops_agent.simulation.tts_strong_asr_value_lab import (
    verify_calibrated_tts_value,
)

SCHEMA_VERSION = "enterprise-candidate-model-release-rollout/v1"
STATUS = "ENTERPRISE_DPO_TTS_EMBEDDING_SHADOW_CANARY_ROLLBACK_PASSED"
CLASSIFICATION = "LOCAL_STAGING_PROJECT_AUTHORIZED"
PROVIDER_MODE = "DETERMINISTIC_PROJECT_STAGING_OBSERVATION"
DEFAULT_OUTPUT = Path("artifacts/m7-enterprise-candidate-rollout/acceptance.json")
TENANT_ID = "tenant-project-enterprise"
STAGES: tuple[
    Literal["SHADOW", "CANARY_5", "CANARY_25"],
    Literal["SHADOW", "CANARY_5", "CANARY_25"],
    Literal["SHADOW", "CANARY_5", "CANARY_25"],
] = ("SHADOW", "CANARY_5", "CANARY_25")


class EnterpriseCandidateRolloutError(RuntimeError):
    """Raised when project-staging rollout evidence is incomplete or mutable."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceBinding(_ClosedModel):
    component: Literal["LLM", "TTS", "EMBEDDING"]
    method: Literal["DPO", "TTS", "EMBEDDING"]
    path: str = Field(min_length=1)
    verifier: str = Field(min_length=1)
    source_status: str = Field(min_length=1)
    candidate_experiment_id: str = Field(min_length=1)
    source_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ApprovalEvidence(_ClosedModel):
    approval_id: str = Field(min_length=1)
    status: Literal["APPROVED"]
    requested_by_subject_id: str = Field(min_length=1)
    decided_by_subject_id: str = Field(min_length=1)
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    independent_approver: Literal[True]


class StageEvidence(_ClosedModel):
    stage: Literal["SHADOW", "CANARY_5", "CANARY_25"]
    desired_traffic_percent: float = Field(ge=0, le=100)
    observed_traffic_percent: float = Field(ge=0, le=100)
    observation_decision: Literal["PASS"]
    window_seconds: int = Field(gt=0)
    request_count: int = Field(gt=0)
    critical_case_count: int = Field(gt=0)
    metrics: dict[str, float]
    thresholds: dict[str, Any]
    source_refs: dict[str, str]
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_revision: str = Field(min_length=1)


class TransitionEvidence(_ClosedModel):
    sequence: int = Field(ge=1)
    from_status: str = Field(min_length=1)
    to_status: str = Field(min_length=1)
    reason_code: str = Field(min_length=1)
    actor_subject_id: str = Field(min_length=1)
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class RollbackRouteEvidence(_ClosedModel):
    stage: Literal["ROLLED_BACK"]
    stable_request_count: int = Field(ge=1)
    candidate_request_count: Literal[0]
    observed_candidate_traffic_percent: float = Field(ge=0.0, le=0.0)
    provider_revision: str = Field(min_length=1)
    route_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ComponentRollout(_ClosedModel):
    component: Literal["LLM", "TTS", "EMBEDDING"]
    method: Literal["DPO", "TTS", "EMBEDDING"]
    deployment_component_key: Literal["llm", "tts", "embedding"]
    source: SourceBinding
    release_id: str = Field(min_length=1)
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval: ApprovalEvidence
    deployment_id: str = Field(min_length=1)
    desired_spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    endpoint_url: str = Field(min_length=1)
    stages: tuple[StageEvidence, ...]
    rollback: RollbackRouteEvidence
    transitions: tuple[TransitionEvidence, ...]
    final_release_status: Literal["ROLLED_BACK"]
    final_deployment_status: Literal["READY"]
    final_deployment_stage: Literal["ROLLED_BACK"]
    final_release_traffic_percent: float = Field(ge=0.0, le=0.0)
    final_observed_traffic_percent: float = Field(ge=0.0, le=0.0)
    active_production_alias_created: Literal[False]


class EnterpriseCandidateRolloutReport(_ClosedModel):
    schema_version: Literal["enterprise-candidate-model-release-rollout/v1"]
    status: Literal["ENTERPRISE_DPO_TTS_EMBEDDING_SHADOW_CANARY_ROLLBACK_PASSED"]
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"]
    generated_at: datetime
    project_enterprise_truth: Literal[True]
    provider_mode: Literal["DETERMINISTIC_PROJECT_STAGING_OBSERVATION"]
    telemetry_mode: Literal["PROJECT_STAGING_DETERMINISTIC_SYNTHETIC"]
    actual_kserve_execution: Literal[False]
    actual_model_inference: Literal[False]
    production_claim: Literal[False]
    external_enterprise_environment_claim: Literal[False]
    components: tuple[ComponentRollout, ...]
    hard_gates: dict[str, bool]
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class _VerifiedSource:
    component: Literal["LLM", "TTS", "EMBEDDING"]
    method: Literal["DPO", "TTS", "EMBEDDING"]
    path: Path
    verifier: str
    source_status: str
    candidate_experiment_id: str
    evidence_chain_sha256: str

    def binding(self, root: Path) -> SourceBinding:
        source = _inside(root, self.path)
        return SourceBinding(
            component=self.component,
            method=self.method,
            path=source.relative_to(root).as_posix(),
            verifier=self.verifier,
            source_status=self.source_status,
            candidate_experiment_id=self.candidate_experiment_id,
            source_file_sha256=sha256(source.read_bytes()).hexdigest(),
            source_evidence_chain_sha256=self.evidence_chain_sha256,
        )


@dataclass(slots=True)
class _ProjectPolicy:
    def decide(self, decision_input: OpaDecisionInput) -> OpaDecision:
        return OpaDecision(
            allowed=True,
            reason_code=f"project_staging_authorized:{decision_input.action}",
            obligations=("preserve_project_staging_provenance",),
            policy_version="project-staging-rollout-v1",
        )


@dataclass(slots=True)
class _DeterministicProvider:
    revision: int = 0

    def reconcile(self, target: ProviderTarget) -> ProviderResult:
        self.revision += 1
        component = _component_from_spec(target.desired_spec)
        return ProviderResult(
            ready=True,
            observed_stage=target.desired_stage,
            observed_traffic_percent=target.desired_traffic_percent,
            applied_spec_hash=target.desired_spec_hash,
            provider_revision=(
                f"project-staging-{component}-{target.desired_stage.lower()}-{self.revision}"
            ),
            endpoint_url=f"http://{component}.models.project.local",
        )


def run_enterprise_candidate_rollout(
    repo_root: Path,
    *,
    output_path: Path = DEFAULT_OUTPUT,
) -> tuple[EnterpriseCandidateRolloutReport, bool]:
    """Exercise the durable rollout FSM without starting Kubernetes or model services."""

    root = repo_root.resolve(strict=True)
    target = _inside(root, output_path)
    if target.exists():
        return verify_enterprise_candidate_rollout(root, target), False
    sources = _verify_sources(root)
    now = datetime.now(UTC)
    database = _database(now, sources)
    try:
        service = ModelDeploymentService(database, _authorizer())
        operator = _identity("project-release-operator", Role.MODEL_RELEASE_OPERATOR, now)
        controller = _identity(
            "project-deployment-controller",
            Role.MODEL_DEPLOYMENT_CONTROLLER,
            now,
        )
        provider = _DeterministicProvider()
        components = tuple(
            _rollout_component(
                database,
                service,
                operator,
                controller,
                provider,
                source,
                now,
                root,
            )
            for source in sources
        )
    finally:
        database.dispose()
    hard_gates = {
        "all_authoritative_sources_verified": len(components) == 3,
        "independent_release_approval_preserved": all(
            item.approval.independent_approver for item in components
        ),
        "shadow_observation_passed": all(item.stages[0].stage == "SHADOW" for item in components),
        "canary_5_observation_passed": all(
            item.stages[1].stage == "CANARY_5" for item in components
        ),
        "canary_25_observation_passed": all(
            item.stages[2].stage == "CANARY_25" for item in components
        ),
        "rollback_route_verified": all(
            item.rollback.stage == "ROLLED_BACK" for item in components
        ),
        "zero_candidate_traffic_after_rollback": all(
            item.rollback.candidate_request_count == 0 for item in components
        ),
        "no_production_alias_created": all(
            not item.active_production_alias_created for item in components
        ),
        "no_actual_kserve_claim": True,
        "no_external_enterprise_environment_claim": True,
    }
    if not all(hard_gates.values()):
        raise EnterpriseCandidateRolloutError("enterprise_candidate_rollout_hard_gate_failed")
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "classification": CLASSIFICATION,
        "generated_at": now,
        "project_enterprise_truth": True,
        "provider_mode": PROVIDER_MODE,
        "telemetry_mode": "PROJECT_STAGING_DETERMINISTIC_SYNTHETIC",
        "actual_kserve_execution": False,
        "actual_model_inference": False,
        "production_claim": False,
        "external_enterprise_environment_claim": False,
        "components": components,
        "hard_gates": hard_gates,
    }
    provisional = EnterpriseCandidateRolloutReport(
        **document,
        evidence_chain_sha256="0" * 64,
    )
    report = provisional.model_copy(
        update={
            "evidence_chain_sha256": _digest(
                provisional.model_dump(mode="json", exclude={"evidence_chain_sha256"})
            )
        }
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    try:
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
    except FileExistsError:
        return verify_enterprise_candidate_rollout(root, target), False
    return report, True


def verify_enterprise_candidate_rollout(
    repo_root: Path,
    report_path: Path = DEFAULT_OUTPUT,
) -> EnterpriseCandidateRolloutReport:
    root = repo_root.resolve(strict=True)
    target = _inside(root, report_path)
    try:
        report = EnterpriseCandidateRolloutReport.model_validate_json(
            target.read_text(encoding="utf-8")
        )
        expected_sources = {
            source.component: source.binding(root) for source in _verify_sources(root)
        }
    except (OSError, RuntimeError, ValueError, ValidationError, KeyError) as exc:
        raise EnterpriseCandidateRolloutError(
            "enterprise_candidate_rollout_evidence_is_invalid"
        ) from exc
    calculated = _digest(
        report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    )
    if calculated != report.evidence_chain_sha256:
        raise EnterpriseCandidateRolloutError("enterprise_candidate_rollout_digest_mismatch")
    if tuple(item.component for item in report.components) != ("LLM", "TTS", "EMBEDDING"):
        raise EnterpriseCandidateRolloutError("enterprise_candidate_rollout_component_set_changed")
    if not report.hard_gates or not all(report.hard_gates.values()):
        raise EnterpriseCandidateRolloutError("enterprise_candidate_rollout_hard_gates_failed")
    for item in report.components:
        if item.source != expected_sources[item.component]:
            raise EnterpriseCandidateRolloutError(
                "enterprise_candidate_rollout_source_binding_changed"
            )
        _verify_component_rollout(item)
    return report


def _verify_component_rollout(item: ComponentRollout) -> None:
    expected_key = {"LLM": "llm", "TTS": "tts", "EMBEDDING": "embedding"}[
        item.component
    ]
    if item.method != item.source.method or item.deployment_component_key != expected_key:
        raise EnterpriseCandidateRolloutError("enterprise_candidate_rollout_method_changed")
    if item.approval.manifest_hash != item.manifest_hash:
        raise EnterpriseCandidateRolloutError("enterprise_candidate_rollout_approval_changed")
    if item.approval.requested_by_subject_id == item.approval.decided_by_subject_id:
        raise EnterpriseCandidateRolloutError(
            "enterprise_candidate_rollout_approval_not_independent"
        )
    if tuple(stage.stage for stage in item.stages) != STAGES:
        raise EnterpriseCandidateRolloutError("enterprise_candidate_rollout_stages_changed")
    for stage in item.stages:
        policy = STAGE_POLICY[stage.stage]
        if (
            stage.desired_traffic_percent != STAGE_TRAFFIC[stage.stage]
            or stage.observed_traffic_percent != STAGE_TRAFFIC[stage.stage]
            or stage.window_seconds < int(policy["min_window_seconds"])
            or stage.request_count < int(policy["min_request_count"])
            or stage.critical_case_count < int(policy["min_critical_case_count"])
            or set(stage.metrics) != REQUIRED_METRICS
            or stage.metrics["cross_tenant_leak_count"] != 0
            or stage.metrics["unauthorized_side_effect_count"] != 0
            or stage.metrics["gpu_xid_error_count"] != 0
            or stage.metrics["error_rate"] > METRIC_THRESHOLDS["error_rate_max"]
            or stage.metrics["p95_latency_ms"] > METRIC_THRESHOLDS["p95_latency_ms_max"]
            or stage.metrics["task_success_delta"]
            < METRIC_THRESHOLDS["task_success_delta_min"]
        ):
            raise EnterpriseCandidateRolloutError(
                "enterprise_candidate_rollout_stage_evidence_changed"
            )
    expected_reasons = {
        "shadow_route_verified",
        "canary_5_route_verified",
        "canary_25_route_verified",
        "rollback_route_verified",
    }
    if not expected_reasons.issubset({item.reason_code for item in item.transitions}):
        raise EnterpriseCandidateRolloutError("enterprise_candidate_rollout_transitions_changed")
    if (
        item.rollback.stable_request_count != 64
        or item.rollback.candidate_request_count != 0
        or item.final_release_status != "ROLLED_BACK"
        or item.final_deployment_stage != "ROLLED_BACK"
        or item.final_release_traffic_percent != 0
        or item.final_observed_traffic_percent != 0
        or item.active_production_alias_created
    ):
        raise EnterpriseCandidateRolloutError("enterprise_candidate_rollout_rollback_changed")


def _verify_sources(root: Path) -> tuple[_VerifiedSource, ...]:
    dpo_path = Path(
        "artifacts/m7-dpo-structured-value-lab/errata/"
        "dpo-structured-36292ed691c0e57623a1/acceptance.json"
    )
    tts_path = Path(
        "artifacts/m7-tts-strong-asr-value-lab/runs/"
        "tts-strong-asr-382ae6ff5105869898e6/acceptance.json"
    )
    embedding_path = Path(
        "artifacts/m7-embedding-calibrated-value-lab/runs/"
        "embedding-calibrated-85045b06936a14066575/acceptance.json"
    )
    dpo = verify_dpo_structured_citation_erratum(root, dpo_path)
    tts = verify_calibrated_tts_value(root, tts_path)
    embedding = verify_calibrated_embedding_value(root, embedding_path)
    expected = (
        (
            dpo.status == "DPO_STRUCTURED_ENTERPRISE_VALUE_ACCEPTED_AFTER_CITATION_FIXTURE_ERRATUM",
            dpo.decision == "ELIGIBLE_FOR_MODEL_RELEASE_DRAFT",
        ),
        (
            tts.status == "TTS_STRONG_ASR_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED",
            tts.decision == "TTS_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT",
        ),
        (
            embedding.status == "EMBEDDING_CALIBRATED_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED",
            embedding.decision == "EMBEDDING_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT",
        ),
    )
    if not all(all(checks) for checks in expected):
        raise EnterpriseCandidateRolloutError("enterprise_candidate_source_not_release_eligible")
    return (
        _VerifiedSource(
            component="LLM",
            method="DPO",
            path=dpo_path,
            verifier="verify_dpo_structured_citation_erratum",
            source_status=dpo.status,
            candidate_experiment_id=dpo.run_id,
            evidence_chain_sha256=dpo.evidence_chain_sha256,
        ),
        _VerifiedSource(
            component="TTS",
            method="TTS",
            path=tts_path,
            verifier="verify_calibrated_tts_value",
            source_status=tts.status,
            candidate_experiment_id=tts.run_id,
            evidence_chain_sha256=tts.evidence_chain_sha256,
        ),
        _VerifiedSource(
            component="EMBEDDING",
            method="EMBEDDING",
            path=embedding_path,
            verifier="verify_calibrated_embedding_value",
            source_status=embedding.status,
            candidate_experiment_id=embedding.run_id,
            evidence_chain_sha256=embedding.evidence_chain_sha256,
        ),
    )


def _database(now: datetime, sources: tuple[_VerifiedSource, ...]) -> Database:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            TenantRecord(
                id=TENANT_ID,
                status="ACTIVE",
                display_name="Project Enterprise Staging",
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        for source in sources:
            _seed_approved_release(session, source, now)
        session.commit()
    return Database.from_engine(engine)


def _seed_approved_release(session: Session, source: _VerifiedSource, now: datetime) -> None:
    component = source.component.lower()
    manifest = _release_manifest(source)
    manifest_hash = _digest(manifest)
    release_id = _release_id(source)
    requester = "project-model-engineer"
    approver = "project-independent-release-approver"
    session.add(
        ModelReleaseRecord(
            release_id=release_id,
            tenant_id=TENANT_ID,
            idempotency_key=f"{release_id}-approved-snapshot",
            evaluation_id=f"project-evaluation-{source.evidence_chain_sha256[:24]}",
            candidate_experiment_id=source.candidate_experiment_id,
            status="APPROVAL_PENDING",
            target_environment="STAGING",
            manifest_json=manifest,
            manifest_hash=manifest_hash,
            rollback_release_id=None,
            traffic_percent=0.0,
            failure_reason=None,
            created_by_subject_id=requester,
            version=4,
            created_at=now,
            updated_at=now,
        )
    )
    session.add(
        ModelReleaseApprovalRecord(
            approval_id=f"project-approval-{component}-{source.evidence_chain_sha256[:20]}",
            tenant_id=TENANT_ID,
            release_id=release_id,
            manifest_hash=manifest_hash,
            status="APPROVED",
            requested_by_subject_id=requester,
            decided_by_subject_id=approver,
            decision_reason="independent project-staging release evidence accepted",
            expires_at=now + timedelta(days=7),
            decided_at=now,
            version=2,
            created_at=now,
            updated_at=now,
        )
    )
    transition_evidence = {
        "approval_id": f"project-approval-{component}-{source.evidence_chain_sha256[:20]}",
        "manifest_hash": manifest_hash,
        "source_evidence_chain_sha256": source.evidence_chain_sha256,
    }
    session.add(
        ModelReleaseTransitionRecord(
            transition_id=f"project-transition-{component}-{source.evidence_chain_sha256[:20]}",
            tenant_id=TENANT_ID,
            release_id=release_id,
            sequence=1,
            from_status="CANDIDATE",
            to_status="APPROVAL_PENDING",
            actor_subject_id=requester,
            reason_code="independent_approval_granted",
            evidence_json=transition_evidence,
            evidence_hash=_digest(transition_evidence),
            occurred_at=now,
            created_at=now,
            updated_at=now,
        )
    )


def _release_manifest(source: _VerifiedSource) -> dict[str, Any]:
    source_hash = source.evidence_chain_sha256
    manifest: dict[str, Any] = {
        "schema_version": "ai-release-manifest/v4",
        "base_model": {
            "id": "Qwen/Qwen2.5-0.5B-Instruct",
            "digest": f"sha256:{_digest({'base': source.method})}",
        },
        "adapter": {
            "artifact_id": f"project-{source.method.lower()}-adapter",
            "object_key": f"project-enterprise/{source.method.lower()}/{source_hash}/bundle",
            "content_hash": source_hash,
            "size_bytes": 1,
            "metadata": {"project_staging_snapshot": True},
        },
        "quantization": None,
        "runtime": {
            "profile_id": "vllm-peft-v1",
            "image_repository": "registry.project.local/industrial-ops/main-runtime",
            "image_digest": f"sha256:{_digest({'image': source.method})}",
            "inference_config": {
                "engine": "vllm",
                "tensor_parallel_size": 1,
                "max_model_len": 8192,
                "dtype": "bfloat16",
                "max_num_seqs": 16,
                "enable_lora": True,
                "max_lora_rank": 64,
            },
            "hardware_profile": {"accelerator": "NVIDIA_GPU", "gpu_count": 1},
        },
        "specialized_components": {},
        "source_evidence": {
            "component": source.component,
            "candidate_experiment_id": source.candidate_experiment_id,
            "evidence_chain_sha256": source_hash,
        },
        "target_environment": "STAGING",
    }
    if source.component == "TTS":
        manifest["specialized_components"]["tts"] = {
            "runtime_model_id": source.candidate_experiment_id,
            "artifact": {
                "object_key": f"project-enterprise/tts/{source_hash}/candidate",
                "content_hash": source_hash,
            },
            "runtime": {
                "image_repository": "registry.project.local/industrial-ops/tts-runtime",
                "image_digest": f"sha256:{_digest({'image': 'TTS'})}",
                "inference_config": {
                    "engine": "speecht5-tts",
                    "model_revision": source_hash[:40],
                    "precision": "bfloat16",
                    "sampling_rate": 16_000,
                    "voice_profile_id": "industrial-authorized-v1",
                    "vocoder_path": "/opt/industrial-ops/models/speecht5_hifigan",
                },
                "hardware_profile": {"accelerator": "NVIDIA_GPU", "gpu_count": 1},
            },
        }
    if source.component == "EMBEDDING":
        manifest["specialized_components"]["embedding"] = {
            "runtime_model_id": source.candidate_experiment_id,
            "artifact": {
                "object_key": f"project-enterprise/embedding/{source_hash}/candidate",
                "content_hash": source_hash,
            },
            "runtime": {
                "image_repository": "registry.project.local/industrial-ops/embedding-runtime",
                "image_digest": f"sha256:{_digest({'image': 'EMBEDDING'})}",
                "inference_config": {
                    "engine": "sentence-transformers-embedding",
                    "precision": "bfloat16",
                    "batch_size": 32,
                    "max_sequence_length": 512,
                    "normalize_embeddings": True,
                },
                "hardware_profile": {"accelerator": "NVIDIA_GPU", "gpu_count": 1},
            },
        }
    return manifest


def _rollout_component(
    database: Database,
    service: ModelDeploymentService,
    operator: IdentityContext,
    controller: IdentityContext,
    provider: _DeterministicProvider,
    source: _VerifiedSource,
    now: datetime,
    root: Path,
) -> ComponentRollout:
    release_id = _release_id(source)
    with Session(database.engine) as session:
        release = session.get(ModelReleaseRecord, release_id)
        if release is None:
            raise EnterpriseCandidateRolloutError("project_release_snapshot_missing")
        release_version = release.version
    requested = service.request_shadow(
        operator,
        release_id,
        _deployment_plan(source),
        expected_release_version=release_version,
        request_id=f"project-rollout-{source.method.lower()}-shadow-request",
    )
    stage_evidence: list[StageEvidence] = []
    current = requested
    for index, stage in enumerate(STAGES):
        reconciled = service.reconcile(
            controller,
            current.deployment.deployment_id,
            provider,
            expected_version=current.deployment.version,
            request_id=f"project-rollout-{source.method.lower()}-{stage.lower()}-reconcile",
        )
        observation_input = _observation(source, stage, now)
        observed = service.record_observation(
            controller,
            reconciled.deployment.deployment_id,
            observation_input,
            expected_deployment_version=reconciled.deployment.version,
            request_id=f"project-rollout-{source.method.lower()}-{stage.lower()}-observe",
        )
        record = observed.observations[-1]
        provider_revision = reconciled.deployment.provider_revision
        if provider_revision is None:
            raise EnterpriseCandidateRolloutError("project_rollout_provider_revision_missing")
        stage_evidence.append(
            StageEvidence(
                stage=stage,
                desired_traffic_percent=STAGE_TRAFFIC[stage],
                observed_traffic_percent=reconciled.deployment.observed_traffic_percent,
                observation_decision="PASS",
                window_seconds=int(
                    (observation_input.window_end - observation_input.window_start).total_seconds()
                ),
                request_count=record.request_count,
                critical_case_count=record.critical_case_count,
                metrics=dict(record.metrics_json),
                thresholds=dict(record.thresholds_json),
                source_refs=dict(record.source_refs_json),
                evidence_hash=record.evidence_hash,
                provider_revision=provider_revision,
            )
        )
        current = observed
        if index < len(STAGES) - 1:
            current = service.promote(
                operator,
                release_id,
                expected_deployment_version=current.deployment.version,
                request_id=f"project-rollout-{source.method.lower()}-{stage.lower()}-promote",
            )
    rollback_requested = service.request_rollback(
        operator,
        release_id,
        expected_deployment_version=current.deployment.version,
        reason_code="project_staging_rollback_acceptance",
        request_id=f"project-rollout-{source.method.lower()}-rollback-request",
    )
    final = service.reconcile(
        controller,
        rollback_requested.deployment.deployment_id,
        provider,
        expected_version=rollback_requested.deployment.version,
        request_id=f"project-rollout-{source.method.lower()}-rollback-reconcile",
    )
    route_payload = {
        "release_id": release_id,
        "deployment_id": final.deployment.deployment_id,
        "stage": "ROLLED_BACK",
        "stable_request_count": 64,
        "candidate_request_count": 0,
        "observed_candidate_traffic_percent": 0.0,
        "provider_revision": final.deployment.provider_revision,
    }
    with Session(database.engine) as session:
        approval = session.scalar(
            select(ModelReleaseApprovalRecord).where(
                ModelReleaseApprovalRecord.release_id == release_id
            )
        )
        transitions = tuple(
            session.scalars(
                select(ModelReleaseTransitionRecord)
                .where(ModelReleaseTransitionRecord.release_id == release_id)
                .order_by(ModelReleaseTransitionRecord.sequence)
            )
        )
        alias_count = len(
            tuple(
                session.scalars(
                    select(ModelAliasRecord).where(
                        ModelAliasRecord.active_release_id == release_id
                    )
                )
            )
        )
    if (
        approval is None
        or final.deployment.provider_revision is None
        or final.deployment.endpoint_url is None
        or approval.decided_by_subject_id is None
    ):
        raise EnterpriseCandidateRolloutError("project_rollout_final_evidence_missing")
    if approval.requested_by_subject_id == approval.decided_by_subject_id:
        raise EnterpriseCandidateRolloutError("project_rollout_approval_not_independent")
    if (
        final.release.status != "ROLLED_BACK"
        or final.deployment.status != "READY"
        or final.deployment.current_stage != "ROLLED_BACK"
        or final.release.traffic_percent != 0
        or final.deployment.observed_traffic_percent != 0
        or alias_count != 0
    ):
        raise EnterpriseCandidateRolloutError("project_rollout_rollback_state_changed")
    return ComponentRollout(
        component=source.component,
        method=source.method,
        deployment_component_key={"LLM": "llm", "TTS": "tts", "EMBEDDING": "embedding"}[
            source.component
        ],
        source=source.binding(root),
        release_id=release_id,
        manifest_hash=final.release.manifest_hash,
        approval=ApprovalEvidence(
            approval_id=approval.approval_id,
            status="APPROVED",
            requested_by_subject_id=approval.requested_by_subject_id,
            decided_by_subject_id=approval.decided_by_subject_id,
            manifest_hash=approval.manifest_hash,
            independent_approver=True,
        ),
        deployment_id=final.deployment.deployment_id,
        desired_spec_hash=final.deployment.desired_spec_hash,
        endpoint_url=str(final.deployment.endpoint_url),
        stages=tuple(stage_evidence),
        rollback=RollbackRouteEvidence(
            stage="ROLLED_BACK",
            stable_request_count=64,
            candidate_request_count=0,
            observed_candidate_traffic_percent=0.0,
            provider_revision=final.deployment.provider_revision,
            route_evidence_sha256=_digest(route_payload),
        ),
        transitions=tuple(
            TransitionEvidence(
                sequence=item.sequence,
                from_status=item.from_status,
                to_status=item.to_status,
                reason_code=item.reason_code,
                actor_subject_id=item.actor_subject_id,
                evidence_hash=item.evidence_hash,
            )
            for item in transitions
        ),
        final_release_status="ROLLED_BACK",
        final_deployment_status="READY",
        final_deployment_stage="ROLLED_BACK",
        final_release_traffic_percent=0.0,
        final_observed_traffic_percent=0.0,
        active_production_alias_created=False,
    )


def _observation(
    source: _VerifiedSource,
    stage: Literal["SHADOW", "CANARY_5", "CANARY_25"],
    now: datetime,
) -> ObservationInput:
    offsets = {"SHADOW": 10_801, "CANARY_5": 7_201, "CANARY_25": 1}
    window_end = now - timedelta(seconds=offsets[stage])
    policy = STAGE_POLICY[stage]
    window_start = window_end - timedelta(seconds=int(policy["min_window_seconds"]))
    metrics = {
        "cross_tenant_leak_count": 0.0,
        "unauthorized_side_effect_count": 0.0,
        "high_risk_miss_rate": 0.0,
        "task_success_delta": 0.01,
        "error_rate": 0.001,
        "p95_latency_ms": 800.0,
        "human_edit_rate_delta": 0.0,
        "wrong_part_rate": 0.0,
        "gpu_xid_error_count": 0.0,
        "gpu_memory_utilization": 0.55,
        "queue_age_seconds": 1.0,
    }
    query_hash = _digest(
        {"component": source.component, "stage": stage, "kind": "prometheus"}
    )
    trace_hash = _digest({"component": source.component, "stage": stage, "kind": "trace"})
    return ObservationInput(
        stage=stage,
        window_start=window_start,
        window_end=window_end,
        request_count=int(policy["min_request_count"]),
        critical_case_count=int(policy["min_critical_case_count"]),
        metrics=metrics,
        source_refs={
            "collector_version": "project-staging-deterministic-v1",
            "prometheus_endpoint_id": "project-staging-no-network",
            "prometheus_query_bundle_hash": f"sha256:{query_hash}",
            "trace_query_hash": f"sha256:{trace_hash}",
        },
    )


def _deployment_plan(source: _VerifiedSource) -> DeploymentPlan:
    component = source.component.lower()
    return DeploymentPlan(
        namespace="project-models",
        gateway_name="project-model-gateway",
        hostname=f"{component}.models.project.local",
        route_name=f"project-{component}-candidate",
        stable_service_name=f"project-{component}-stable",
        service_account_name="project-model-runtime",
        serving_runtime_name="project-vllm-runtime",
        artifact_uri_prefix=f"s3://project-enterprise-models/{TENANT_ID}",
    )


def _authorizer() -> Authorizer:
    return Authorizer(
        SecurityAuditor(
            InMemorySecurityAuditSink(),
            hash_key=b"project-enterprise-candidate-rollout-audit-v1",
        ),
        policy_client=_ProjectPolicy(),
    )


def _identity(subject: str, role: Role, now: datetime) -> IdentityContext:
    return IdentityContext(
        subject_id=subject,
        oidc_subject=f"oidc:{subject}",
        tenant_id=TENANT_ID,
        roles=frozenset({role}),
        asset_ids=frozenset(),
        site_ids=frozenset(),
        issued_at=now - timedelta(minutes=5),
        expires_at=now + timedelta(hours=1),
    )


def _release_id(source: _VerifiedSource) -> str:
    return f"project-release-{source.method.lower()}-{source.evidence_chain_sha256[:20]}"


def _component_from_spec(spec: dict[str, Any]) -> str:
    if "tts" in spec:
        return "tts"
    if "embedding" in spec:
        return "embedding"
    return "llm"


def _inside(root: Path, path: Path) -> Path:
    target = path if path.is_absolute() else root / path
    resolved = target.resolve(strict=target.exists())
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EnterpriseCandidateRolloutError("evidence path escapes repository") from exc
    return resolved


def _digest(document: Any) -> str:
    return sha256(
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
