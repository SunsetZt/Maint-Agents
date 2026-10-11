"""Adopt every project-generated simulation asset as enterprise-project staging truth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.enterprise_assets.acceptance import (
    COMPONENTS,
    verify_enterprise_model_import_acceptance,
)
from industrial_ops_agent.simulation.asr_enterprise_value_lab import (
    verify_asr_enterprise_value,
)
from industrial_ops_agent.simulation.asr_kserve_acceptance import (
    verify_asr_kserve_acceptance,
)
from industrial_ops_agent.simulation.dpo_agent_runtime_value_erratum import (
    verify_dpo_agent_runtime_erratum,
)
from industrial_ops_agent.simulation.dpo_post_training_lab import (
    verify_dpo_post_training,
)
from industrial_ops_agent.simulation.dpo_recovery_value_lab import (
    verify_dpo_recovery_value,
)
from industrial_ops_agent.simulation.dpo_structured_citation_erratum import (
    verify_dpo_structured_citation_erratum,
)
from industrial_ops_agent.simulation.enterprise_candidate_kserve_acceptance import (
    verify_enterprise_candidate_kserve_acceptance,
)
from industrial_ops_agent.simulation.grpo_agent_runtime_value_lab import (
    verify_grpo_agent_runtime_value,
)
from industrial_ops_agent.simulation.grpo_kserve_rollout import (
    verify_grpo_kserve_rollout,
)
from industrial_ops_agent.simulation.grpo_post_training_lab import (
    verify_grpo_post_training,
)
from industrial_ops_agent.simulation.maintenance_planning_value_lab import (
    verify_maintenance_planning_value_lab,
)
from industrial_ops_agent.simulation.model_experiment_outcomes import (
    AcceptedEmbeddingOutcome,
    AcceptedRerankerOutcome,
    AcceptedTtsOutcome,
    verify_model_experiment_outcomes,
)
from industrial_ops_agent.simulation.multimodal_fault_dataset import (
    verify_multimodal_fault_dataset,
)
from industrial_ops_agent.simulation.multimodal_platform_lab import (
    verify_simulated_multimodal_platform_lab,
)
from industrial_ops_agent.simulation.ppo_agent_runtime_value_lab import (
    verify_ppo_agent_runtime_value,
)
from industrial_ops_agent.simulation.ppo_kserve_rollout import (
    verify_ppo_kserve_rollout,
)
from industrial_ops_agent.simulation.ppo_post_training_lab import (
    verify_ppo_research_safety,
)
from industrial_ops_agent.simulation.project_assurance_lab import (
    verify_project_assurance_lab,
)
from industrial_ops_agent.simulation.reranker_calibrated_value_lab import (
    verify_calibrated_reranker_value,
)
from industrial_ops_agent.simulation.reranker_kserve_acceptance import (
    verify_reranker_kserve_acceptance,
)
from industrial_ops_agent.simulation.supplier_a2a_lab import verify_supplier_a2a_lab
from industrial_ops_agent.simulation.vlm_enterprise_staging import (
    verify_enterprise_vlm_staging_adoption,
)
from industrial_ops_agent.simulation.vlm_kserve_shadow import (
    verify_vlm_kserve_rollout,
)

CLASSIFICATION: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = "LOCAL_STAGING_PROJECT_AUTHORIZED"
SCHEMA_VERSION: Literal["enterprise-project-authoritative-adoption/v1"] = (
    "enterprise-project-authoritative-adoption/v1"
)
STATUS: Literal["ALL_PROJECT_ASSETS_ENTERPRISE_STAGING_ADOPTED"] = (
    "ALL_PROJECT_ASSETS_ENTERPRISE_STAGING_ADOPTED"
)
SOURCE_CLASSIFICATION: Literal["SIMULATED_NON_PRODUCTION"] = "SIMULATED_NON_PRODUCTION"

_CLOSURE_PATH = Path("artifacts/simulated-enterprise-closure/acceptance.json")
_DATASET_DIRS = (
    Path("artifacts/m7-multimodal-fault-dataset"),
    Path("artifacts/m7-vlm-curriculum-dataset"),
)
_PLATFORM_DIRS = (
    Path("artifacts/m7-multimodal-platform-lab"),
    Path("artifacts/m7-vlm-curriculum-platform-lab"),
)
_MODEL_SANDBOX_ROOT = Path("artifacts/model-training-sandbox")
_OUTPUT_PATH = Path("artifacts/enterprise-project-adoption/acceptance.json")


class EnterpriseProjectAdoptionError(RuntimeError):
    """The project-wide enterprise adoption contract is incomplete or changed."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnterpriseAdoptionPolicy(_ClosedModel):
    source_marker_interpretation: Literal["PROVENANCE_ONLY"] = "PROVENANCE_ONLY"
    project_operational_classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    passing_asset_policy: Literal["AUTHORITATIVE_FOR_PROJECT_ENTERPRISE_STAGING"] = (
        "AUTHORITATIVE_FOR_PROJECT_ENTERPRISE_STAGING"
    )
    failed_experiment_policy: Literal["ENTERPRISE_AUDIT_HISTORY_ONLY"] = (
        "ENTERPRISE_AUDIT_HISTORY_ONLY"
    )
    generated_data_policy: Literal["PROJECT_ENTERPRISE_DATA_AUTHORIZED"] = (
        "PROJECT_ENTERPRISE_DATA_AUTHORIZED"
    )
    manifest_descendants_inherit_adoption: Literal[True] = True
    external_enterprise_identity_claimed: Literal[False] = False


class AdoptedEnterpriseAsset(_ClosedModel):
    domain: str = Field(min_length=1)
    asset_kind: str = Field(min_length=1)
    role: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_schema_version: str = Field(min_length=1)
    source_classification: str = Field(min_length=1)
    source_status: str = Field(min_length=1)
    source_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_evidence_chain_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    operational_classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    project_authoritative: Literal[True] = True
    runtime_eligible: bool
    capabilities: tuple[str, ...]


class EnterpriseExperimentHistory(_ClosedModel):
    source_path: str = Field(min_length=1)
    source_schema_version: str = Field(min_length=1)
    source_classification: Literal[
        "SIMULATED_NON_PRODUCTION",
        "LOCAL_STAGING_PROJECT_AUTHORIZED",
    ] = SOURCE_CLASSIFICATION
    source_status: str = Field(min_length=1)
    source_decision: str | None = None
    source_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    disposition: Literal[
        "REJECTED_ENTERPRISE_EXPERIMENT_HISTORY",
        "SUPERSEDED_ENTERPRISE_EXPERIMENT_HISTORY",
    ]
    operational_classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    project_authoritative_history: Literal[True] = True
    runtime_eligible: Literal[False] = False


class EnterpriseProjectAdoptionReport(_ClosedModel):
    schema_version: Literal["enterprise-project-authoritative-adoption/v1"] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = CLASSIFICATION
    status: Literal["ALL_PROJECT_ASSETS_ENTERPRISE_STAGING_ADOPTED"] = STATUS
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    authorization_basis: Literal["PROJECT_OWNER_EXPLICIT_GLOBAL_ADOPTION"] = (
        "PROJECT_OWNER_EXPLICIT_GLOBAL_ADOPTION"
    )
    project_enterprise_truth: Literal[True] = True
    authoritative_for_project: Literal[True] = True
    project_generated_sources_accepted: Literal[True] = True
    ready_for_enterprise_project_use: Literal[True] = True
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    source_provenance_preserved: Literal[True] = True
    adopted_at: datetime
    policy: EnterpriseAdoptionPolicy
    active_assets: tuple[AdoptedEnterpriseAsset, ...]
    experiment_history: tuple[EnterpriseExperimentHistory, ...]
    coverage: dict[str, Literal["PASSED"]]
    source_marker_counts: dict[str, int]
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class _CollectedState:
    active_assets: tuple[AdoptedEnterpriseAsset, ...]
    experiment_history: tuple[EnterpriseExperimentHistory, ...]
    coverage: dict[str, Literal["PASSED"]]
    source_marker_counts: dict[str, int]


def adopt_all_project_assets_as_enterprise(
    repo_root: Path,
    *,
    output_path: Path = _OUTPUT_PATH,
) -> tuple[EnterpriseProjectAdoptionReport, bool]:
    """Create or refresh the global project-enterprise authority receipt."""

    root = repo_root.resolve(strict=True)
    state = _collect_current_state(root)
    target = _output_file(root, output_path)
    if target.exists():
        existing = _load_report(target)
        if (
            existing.active_assets == state.active_assets
            and existing.experiment_history == state.experiment_history
            and existing.coverage == state.coverage
            and existing.source_marker_counts == state.source_marker_counts
        ):
            return existing, False

    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "status": STATUS,
        "enterprise_scope": "PROJECT_INTERNAL",
        "authorization_basis": "PROJECT_OWNER_EXPLICIT_GLOBAL_ADOPTION",
        "project_enterprise_truth": True,
        "authoritative_for_project": True,
        "project_generated_sources_accepted": True,
        "ready_for_enterprise_project_use": True,
        "production_claim": False,
        "external_enterprise_production_claim": False,
        "source_provenance_preserved": True,
        "adopted_at": datetime.now(UTC).isoformat(),
        "policy": EnterpriseAdoptionPolicy().model_dump(mode="json"),
        "active_assets": [item.model_dump(mode="json") for item in state.active_assets],
        "experiment_history": [item.model_dump(mode="json") for item in state.experiment_history],
        "coverage": state.coverage,
        "source_marker_counts": state.source_marker_counts,
    }
    draft = EnterpriseProjectAdoptionReport.model_validate(
        {**unsigned, "evidence_chain_sha256": "0" * 64}
    )
    report = draft.model_copy(update={"evidence_chain_sha256": _report_digest(draft)})
    _write_atomic(report, target)
    return verify_enterprise_project_adoption(root, target), True


def verify_enterprise_project_adoption(
    repo_root: Path,
    acceptance_path: Path = _OUTPUT_PATH,
) -> EnterpriseProjectAdoptionReport:
    """Re-read every canonical source and reject stale or incomplete global adoption."""

    root = repo_root.resolve(strict=True)
    path = _inside_file(root, acceptance_path)
    report = _load_report(path)
    state = _collect_current_state(root)
    if (
        report.active_assets != state.active_assets
        or report.experiment_history != state.experiment_history
        or report.coverage != state.coverage
        or report.source_marker_counts != state.source_marker_counts
    ):
        raise EnterpriseProjectAdoptionError("enterprise_project_adoption_sources_changed")
    return report


def load_enterprise_project_adoption_envelope(
    repo_root: Path,
    acceptance_path: Path = _OUTPUT_PATH,
) -> EnterpriseProjectAdoptionReport:
    """Verify only the immutable adoption envelope for a scoped source migration.

    Callers must independently verify every source they consume. Normal runtime
    reads must continue to use :func:`verify_enterprise_project_adoption`.
    """

    root = repo_root.resolve(strict=True)
    return _load_report(_inside_file(root, acceptance_path))


def _collect_current_state(root: Path) -> _CollectedState:
    closure_path = _inside_file(root, _CLOSURE_PATH)
    closure = _load_chained(
        closure_path,
        schema="simulated-enterprise-data-model-closure/v1",
        status="SIMULATED_ENTERPRISE_DATA_MODEL_CLOSURE_PASSED",
        classification=SOURCE_CLASSIFICATION,
    )
    raw_coverage = closure.get("coverage")
    required_closure_coverage = {
        "enterprise_systems",
        "supplier_a2a_collaboration",
        "maintenance_planning_value_evaluation",
        "project_assurance_lab",
        "llm_training_and_kserve",
        "vlm_enterprise_staging_adoption",
        "vlm_kserve_rollout",
        "dpo_post_training",
        "dpo_agent_runtime_evaluation",
        "dpo_enterprise_value",
        "grpo_post_training",
        "grpo_agent_runtime_value",
        "grpo_kserve_rollout",
        "ppo_research_safety",
        "ppo_agent_runtime_value",
        "ppo_kserve_rollout",
        "asr_enterprise_value",
        "asr_kserve_rollout",
        "enterprise_model_import_preflight",
        "rejected_model_experiment_governance",
        "tts_enterprise_value",
        "embedding_enterprise_value",
        "reranker_kserve_rollout",
        "enterprise_candidate_release_rollout",
        "predictive_maintenance_training",
        "predictive_maintenance_kserve",
    }
    if (
        not isinstance(raw_coverage, dict)
        or not required_closure_coverage.issubset(raw_coverage)
        or any(raw_coverage.get(name) != "PASSED" for name in required_closure_coverage)
    ):
        raise EnterpriseProjectAdoptionError("enterprise_closure_coverage_is_incomplete")
    required_projection_domains = required_closure_coverage - {
        "tts_enterprise_value",
        "embedding_enterprise_value",
    }
    raw_sources = closure.get("source_evidence")
    projected_domains = (
        {
            item.get("domain")
            for item in raw_sources
            if isinstance(item, dict) and isinstance(item.get("domain"), str)
        }
        if isinstance(raw_sources, list)
        else set()
    )
    if (
        not isinstance(raw_sources, list)
        or len(raw_sources) < len(required_projection_domains)
        or not required_projection_domains.issubset(projected_domains)
    ):
        raise EnterpriseProjectAdoptionError("enterprise_closure_sources_are_incomplete")

    active: list[AdoptedEnterpriseAsset] = [
        _asset(
            root,
            closure_path,
            closure,
            domain="enterprise_project_closure",
            asset_kind="aggregate_acceptance",
            role="ENTERPRISE_PROJECT_EVIDENCE_ROOT",
            runtime_eligible=False,
            capabilities=tuple(sorted(required_closure_coverage)),
        )
    ]
    selected_paths = {closure_path}
    vlm_source_path: Path | None = None
    tts_source_path: Path | None = None
    embedding_source_path: Path | None = None
    reranker_source_path: Path | None = None

    for raw_projection in raw_sources:
        if not isinstance(raw_projection, dict):
            raise EnterpriseProjectAdoptionError("enterprise_closure_source_projection_is_invalid")
        source_value = raw_projection.get("source_path")
        capabilities_value = raw_projection.get("capabilities")
        if (
            not isinstance(source_value, str)
            or not source_value
            or not isinstance(capabilities_value, list)
            or not capabilities_value
            or not all(isinstance(item, str) and item for item in capabilities_value)
        ):
            raise EnterpriseProjectAdoptionError("enterprise_closure_source_projection_is_invalid")
        source_path = _inside_file(root, Path(source_value))
        source_document = _load_object(source_path)
        source_chain = source_document.get("evidence_chain_sha256")
        if (
            sha256(source_path.read_bytes()).hexdigest() != raw_projection.get("file_sha256")
            or source_document.get("schema_version") != raw_projection.get("schema_version")
            or (source_document.get("classification") or source_document.get("marker"))
            != raw_projection.get("source_classification")
            or source_chain != raw_projection.get("evidence_chain_sha256")
            or not isinstance(source_chain, str)
            or source_chain != _document_digest(source_document)
        ):
            raise EnterpriseProjectAdoptionError("enterprise_closure_source_projection_changed")
        domain = raw_projection.get("domain")
        source_status = raw_projection.get("status")
        if not isinstance(domain, str) or not isinstance(source_status, str):
            raise EnterpriseProjectAdoptionError("enterprise_closure_source_projection_is_invalid")
        if domain == "vlm_enterprise_staging_adoption":
            vlm_adoption = verify_enterprise_vlm_staging_adoption(root, source_path)
            vlm_source_path = _inside_file(
                root,
                Path(vlm_adoption.candidate.source_acceptance_path),
            )
        elif domain == "vlm_kserve_rollout":
            verify_vlm_kserve_rollout(root, source_path)
        elif domain == "dpo_post_training":
            verify_dpo_post_training(root, source_path)
        elif domain == "dpo_agent_runtime_evaluation":
            if source_document.get("schema_version") == "enterprise-dpo-recovery-value-lab/v2":
                verify_dpo_recovery_value(root, source_path)
            else:
                verify_dpo_agent_runtime_erratum(root, source_path)
        elif domain == "dpo_enterprise_value":
            verify_dpo_structured_citation_erratum(root, source_path)
        elif domain == "grpo_post_training":
            verify_grpo_post_training(root, source_path)
        elif domain == "grpo_agent_runtime_value":
            verify_grpo_agent_runtime_value(root, source_path)
        elif domain == "grpo_kserve_rollout":
            verify_grpo_kserve_rollout(root, source_path)
        elif domain == "ppo_research_safety":
            verify_ppo_research_safety(root, source_path)
        elif domain == "ppo_agent_runtime_value":
            verify_ppo_agent_runtime_value(root, source_path)
        elif domain == "ppo_kserve_rollout":
            verify_ppo_kserve_rollout(root, source_path)
        elif domain == "asr_enterprise_value":
            verify_asr_enterprise_value(root, source_path)
        elif domain == "asr_kserve_rollout":
            verify_asr_kserve_acceptance(root, source_path)
        elif domain == "enterprise_model_import_preflight":
            import_acceptance = verify_enterprise_model_import_acceptance(source_path)
            baseline_sets = {
                item.compatible_baseline_release_ids for item in import_acceptance.components
            }
            if (
                import_acceptance.execution_mode != "EPHEMERAL_LOOPBACK_STAGING"
                or tuple(item.component for item in import_acceptance.components) != COMPONENTS
                or any(
                    not item.release_draft_eligible or item.release_draft_blockers
                    for item in import_acceptance.components
                )
                or len(baseline_sets) != 1
            ):
                raise EnterpriseProjectAdoptionError("enterprise_model_import_preflight_is_invalid")
        elif domain == "rejected_model_experiment_governance":
            model_outcomes = verify_model_experiment_outcomes(root, source_path)
            tts_outcome = next(
                (item for item in model_outcomes.outcomes if item.method == "TTS"),
                None,
            )
            embedding_outcome = next(
                (item for item in model_outcomes.outcomes if item.method == "EMBEDDING"),
                None,
            )
            reranker_outcome = next(
                (item for item in model_outcomes.outcomes if item.method == "RERANKER"),
                None,
            )
            if (
                not isinstance(tts_outcome, AcceptedTtsOutcome)
                or not isinstance(embedding_outcome, AcceptedEmbeddingOutcome)
                or not isinstance(reranker_outcome, AcceptedRerankerOutcome)
                or model_outcomes.rejected_method_count != 0
                or model_outcomes.active_candidate_count != 3
                or model_outcomes.intentionally_unverified_methods
            ):
                raise EnterpriseProjectAdoptionError(
                    "accepted_enterprise_model_candidates_are_missing"
                )
            tts_source_path = _inside_file(
                root,
                Path(tts_outcome.governance_acceptance.path),
            )
            embedding_source_path = _inside_file(
                root,
                Path(embedding_outcome.governance_acceptance.path),
            )
            reranker_source_path = _inside_file(
                root,
                Path(reranker_outcome.governance_acceptance.path),
            )
        elif domain == "reranker_kserve_rollout":
            verify_reranker_kserve_acceptance(root, source_path)
        elif domain == "enterprise_candidate_release_rollout":
            verify_enterprise_candidate_kserve_acceptance(root, source_path)
        elif domain == "project_assurance_lab":
            verify_project_assurance_lab(root, source_path)
        elif domain == "supplier_a2a_collaboration":
            verify_supplier_a2a_lab(root, source_path)
        elif domain == "maintenance_planning_value_evaluation":
            verify_maintenance_planning_value_lab(root, source_path)
        active.append(
            _asset(
                root,
                source_path,
                source_document,
                domain=domain,
                asset_kind="verified_evidence",
                role=_closure_role(domain),
                runtime_eligible=domain
                not in {
                    "dpo_post_training",
                    "dpo_agent_runtime_evaluation",
                    "dpo_enterprise_value",
                    "grpo_post_training",
                    "grpo_agent_runtime_value",
                    "grpo_kserve_rollout",
                    "ppo_research_safety",
                    "ppo_agent_runtime_value",
                    "ppo_kserve_rollout",
                    "asr_enterprise_value",
                    "asr_kserve_rollout",
                    "enterprise_model_import_preflight",
                    "rejected_model_experiment_governance",
                    "reranker_kserve_rollout",
                    "enterprise_candidate_release_rollout",
                    "project_assurance_lab",
                    "supplier_a2a_collaboration",
                    "maintenance_planning_value_evaluation",
                },
                capabilities=tuple(str(item) for item in capabilities_value),
                source_status=source_status,
            )
        )
        selected_paths.add(source_path)

    if vlm_source_path is None:
        raise EnterpriseProjectAdoptionError("enterprise_vlm_source_is_missing")
    vlm_source = _load_chained(
        vlm_source_path,
        schema="m7-vlm-checkpoint-continuation-lab/v1",
        status=None,
        classification=SOURCE_CLASSIFICATION,
    )
    active.append(
        _asset(
            root,
            vlm_source_path,
            vlm_source,
            domain="vlm_training_candidate",
            asset_kind="model_acceptance",
            role="ACTIVE_ENTERPRISE_VLM_MODEL_SOURCE",
            runtime_eligible=True,
            capabilities=(
                "VLM LoRA checkpoint",
                "frozen multimodal evaluation",
                "region grounding",
            ),
        )
    )
    selected_paths.add(vlm_source_path)

    if tts_source_path is None:
        raise EnterpriseProjectAdoptionError("enterprise_tts_value_source_is_missing")
    tts = _load_chained(
        tts_source_path,
        schema="enterprise-tts-strong-asr-value-lab/v8",
        status="TTS_STRONG_ASR_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED",
        classification=CLASSIFICATION,
    )
    if (
        tts.get("decision") != "TTS_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
        or tts.get("candidate_accepted") is not True
        or tts.get("release_draft_eligible") is not True
        or tts.get("formal_model_release_created") is not False
        or tts.get("runtime_eligible") is not False
        or tts.get("same_gold_reuse_permitted") is not False
        or tts.get("failed_hard_gates") != []
    ):
        raise EnterpriseProjectAdoptionError("enterprise_tts_value_source_is_invalid")
    active.append(
        _asset(
            root,
            tts_source_path,
            tts,
            domain="tts_enterprise_value_candidate",
            asset_kind="model_acceptance",
            role="AUTHORIZED_ENTERPRISE_TTS_MODEL_RELEASE_CANDIDATE",
            runtime_eligible=False,
            capabilities=(
                "actual GPU trained TTS predecessor and calibrated inference",
                "independent Whisper Small strong-ASR verification",
                "safety phrase terminology intelligibility and WER gates",
                "fresh single-use frozen Gold v9 evaluation",
                "ModelRelease draft eligibility without runtime activation",
            ),
        )
    )
    selected_paths.add(tts_source_path)

    if embedding_source_path is None:
        raise EnterpriseProjectAdoptionError("enterprise_embedding_value_source_is_missing")
    embedding = _load_chained(
        embedding_source_path,
        schema="enterprise-embedding-calibrated-value-lab/v3",
        status="EMBEDDING_CALIBRATED_ACTUAL_GPU_ENTERPRISE_VALUE_PASSED",
        classification=CLASSIFICATION,
    )
    if (
        embedding.get("decision") != "EMBEDDING_CANDIDATE_ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
        or embedding.get("candidate_accepted") is not True
        or embedding.get("release_draft_eligible") is not True
        or embedding.get("formal_model_release_created") is not False
        or embedding.get("runtime_eligible") is not False
        or embedding.get("same_gold_reuse_permitted") is not False
        or embedding.get("failed_hard_gates") != []
    ):
        raise EnterpriseProjectAdoptionError("enterprise_embedding_value_source_is_invalid")
    active.append(
        _asset(
            root,
            embedding_source_path,
            embedding,
            domain="embedding_enterprise_value_candidate",
            asset_kind="model_acceptance",
            role="AUTHORIZED_ENTERPRISE_EMBEDDING_MODEL_RELEASE_CANDIDATE",
            runtime_eligible=False,
            capabilities=(
                "actual GPU trained embedding predecessor and calibrated inference",
                "governed industrial terminology and fault retrieval",
                "fresh single-use frozen Gold v3 evaluation",
                "formal Recall MRR and nDCG quality gates",
                "ModelRelease draft eligibility without runtime activation",
            ),
        )
    )
    selected_paths.add(embedding_source_path)

    if reranker_source_path is None:
        raise EnterpriseProjectAdoptionError("enterprise_reranker_value_source_is_missing")
    reranker = verify_calibrated_reranker_value(root, reranker_source_path)
    if (
        reranker.candidate_accepted is not True
        or reranker.formal_model_release_created is not False
        or reranker.runtime_eligible is not False
        or reranker.failed_hard_gates
    ):
        raise EnterpriseProjectAdoptionError("enterprise_reranker_value_source_is_invalid")
    active.append(
        _asset(
            root,
            reranker_source_path,
            reranker.model_dump(mode="json"),
            domain="reranker_enterprise_value_candidate",
            asset_kind="model_acceptance",
            role="AUTHORIZED_ENTERPRISE_RERANKER_MODEL_RELEASE_CANDIDATE",
            runtime_eligible=False,
            capabilities=(
                "actual GPU trained predecessor and calibrated inference",
                "governed industrial terminology resolution",
                "fresh single-use frozen Gold v5 evaluation",
                "formal Recall MRR and nDCG quality gates",
                "ModelRelease draft eligibility without runtime activation",
            ),
        )
    )
    selected_paths.add(reranker_source_path)

    for dataset_directory in _DATASET_DIRS:
        directory = _inside_directory(root, dataset_directory)
        result = verify_multimodal_fault_dataset(directory)
        manifest_path = _inside_file(root, directory / "manifest.json")
        active.append(
            _asset(
                root,
                manifest_path,
                _load_object(manifest_path),
                domain="multimodal_enterprise_data",
                asset_kind="dataset_manifest",
                role="AUTHORITATIVE_ENTERPRISE_PROJECT_DATA",
                runtime_eligible=True,
                capabilities=(
                    "image fault evidence",
                    "acoustic evidence",
                    "telemetry",
                    "engineer notes",
                    "frozen evaluation split",
                ),
                source_status=result.status,
            )
        )
        selected_paths.add(manifest_path)

    for platform_directory in _PLATFORM_DIRS:
        directory = _inside_directory(root, platform_directory)
        report = verify_simulated_multimodal_platform_lab(directory)
        acceptance_path = _inside_file(root, directory / "acceptance.json")
        active.append(
            _asset(
                root,
                acceptance_path,
                report.model_dump(mode="json"),
                domain="multimodal_enterprise_pipeline",
                asset_kind="platform_import_acceptance",
                role="AUTHORITATIVE_ENTERPRISE_PROJECT_PIPELINE",
                runtime_eligible=True,
                capabilities=(
                    "governed training snapshot",
                    "frozen evaluation snapshot",
                    "DLP and review projections",
                    "OpenLineage evidence",
                ),
            )
        )
        selected_paths.add(acceptance_path)

    sandbox_assets = _model_sandbox_assets(root)
    active.extend(sandbox_assets)
    selected_paths.update(_inside_file(root, Path(item.source_path)) for item in sandbox_assets)
    history = _experiment_history(root, selected_paths)
    active_tuple = tuple(sorted(active, key=lambda item: (item.domain, item.source_path)))
    markers: dict[str, int] = {}
    for marker in [
        *(item.source_classification for item in active_tuple),
        *(item.source_classification for item in history),
    ]:
        markers[marker] = markers.get(marker, 0) + 1

    return _CollectedState(
        active_assets=active_tuple,
        experiment_history=history,
        coverage={
            "enterprise_systems": "PASSED",
            "multimodal_enterprise_data": "PASSED",
            "multimodal_enterprise_pipeline": "PASSED",
            "llm_training_and_kserve": "PASSED",
            "vlm_training_and_enterprise_adoption": "PASSED",
            "vlm_kserve_release": "PASSED",
            "dpo_actual_gpu_post_training": "PASSED",
            "dpo_agent_runtime_evaluation": "PASSED",
            "dpo_actual_gpu_enterprise_value": "PASSED",
            "grpo_actual_gpu_post_training": "PASSED",
            "grpo_agent_runtime_enterprise_value": "PASSED",
            "ppo_actual_gpu_research_safety": "PASSED",
            "ppo_agent_runtime_enterprise_value": "PASSED",
            "ppo_kserve_release": "PASSED",
            "asr_actual_gpu_enterprise_value": "PASSED",
            "asr_kserve_release": "PASSED",
            "enterprise_model_import_preflight": "PASSED",
            "rejected_actual_gpu_model_experiments": "PASSED",
            "tts_actual_gpu_enterprise_value": "PASSED",
            "embedding_actual_gpu_enterprise_value": "PASSED",
            "reranker_actual_gpu_enterprise_value": "PASSED",
            "reranker_kserve_rollout": "PASSED",
            "enterprise_candidate_release_rollout": "PASSED",
            "predictive_maintenance_and_kserve": "PASSED",
            "project_training_reference": "PASSED",
            "enterprise_experiment_history": "PASSED",
        },
        source_marker_counts=dict(sorted(markers.items())),
    )


def _model_sandbox_assets(root: Path) -> tuple[AdoptedEnterpriseAsset, ...]:
    sandbox_root = _inside_directory(root, _MODEL_SANDBOX_ROOT)
    receipt_candidates = sorted(sandbox_root.glob("*/promotion-receipt.json"))
    if not receipt_candidates:
        raise EnterpriseProjectAdoptionError("model_training_promotion_receipt_is_missing")
    receipt_path = _inside_file(root, receipt_candidates[-1])
    report_path = _inside_file(root, receipt_path.parent / "sandbox-report.json")
    receipt = _load_object(receipt_path)
    report = _load_object(report_path)
    receipt_gates = receipt.get("gates")
    report_gates = report.get("gates")
    selected_digest = receipt.get("selected_adapter_manifest_digest")
    if (
        receipt.get("schema_version") != "model-training-promotion-receipt/v1"
        or report.get("schema_version") != "model-training-sandbox-report/v1"
        or receipt.get("marker") != SOURCE_CLASSIFICATION
        or report.get("marker") != SOURCE_CLASSIFICATION
        or receipt.get("decision") != "CANDIDATE_ELIGIBLE"
        or report.get("decision") != "CANDIDATE_ELIGIBLE"
        or receipt.get("formal_release_created") is not False
        or report.get("formal_release_created") is not False
        or not _is_sha256(selected_digest)
        or selected_digest != report.get("selected_adapter_manifest_digest")
        or not isinstance(receipt_gates, dict)
        or not receipt_gates
        or receipt_gates != report_gates
        or not all(value is True for value in receipt_gates.values())
    ):
        raise EnterpriseProjectAdoptionError("model_training_promotion_receipt_is_invalid")
    combined = dict(receipt)
    combined["related_report_path"] = report_path.relative_to(root).as_posix()
    combined["related_report_sha256"] = sha256(report_path.read_bytes()).hexdigest()
    return (
        _asset(
            root,
            receipt_path,
            combined,
            domain="project_model_training_reference",
            asset_kind="training_promotion_receipt",
            role="AUTHORIZED_ENTERPRISE_PROJECT_TRAINING_REFERENCE",
            runtime_eligible=False,
            capabilities=(
                "LoRA and QLoRA comparison",
                "frozen Gold isolation",
                "adapter integrity",
                "reproducible optimizer execution",
            ),
            source_status="CANDIDATE_ELIGIBLE",
        ),
    )


def _experiment_history(
    root: Path,
    selected_paths: set[Path],
) -> tuple[EnterpriseExperimentHistory, ...]:
    artifacts_root = _inside_directory(root, Path("artifacts"))
    history: list[EnterpriseExperimentHistory] = []
    candidate_paths = {
        *artifacts_root.glob("*/acceptance.json"),
        *artifacts_root.glob("*/rejections/*/rejection.json"),
    }
    for candidate in sorted(candidate_paths):
        path = _inside_file(root, candidate)
        if path in selected_paths:
            continue
        document = _load_object(path)
        classification = document.get("classification") or document.get("marker")
        is_rejection = path.name == "rejection.json" and "rejections" in path.parts
        if classification != SOURCE_CLASSIFICATION and not (
            is_rejection and classification == CLASSIFICATION
        ):
            continue
        schema = document.get("schema_version")
        decision_value = document.get("decision")
        status_value = document.get("status")
        outcome_value = document.get("outcome")
        decision = decision_value if isinstance(decision_value, str) else None
        status = (
            status_value
            if isinstance(status_value, str)
            else decision
            or (outcome_value if isinstance(outcome_value, str) else None)
            or "VERIFIED_EXPERIMENT"
        )
        if not isinstance(schema, str) or not schema:
            raise EnterpriseProjectAdoptionError("enterprise_experiment_history_schema_is_invalid")
        disposition_source = f"{status} {decision or ''} {outcome_value or ''}".upper()
        disposition: Literal[
            "REJECTED_ENTERPRISE_EXPERIMENT_HISTORY",
            "SUPERSEDED_ENTERPRISE_EXPERIMENT_HISTORY",
        ]
        if "REJECTED" in disposition_source or "NO_GAIN" in disposition_source:
            disposition = "REJECTED_ENTERPRISE_EXPERIMENT_HISTORY"
        else:
            disposition = "SUPERSEDED_ENTERPRISE_EXPERIMENT_HISTORY"
        history_classification: Literal[
            "SIMULATED_NON_PRODUCTION",
            "LOCAL_STAGING_PROJECT_AUTHORIZED",
        ] = CLASSIFICATION if classification == CLASSIFICATION else SOURCE_CLASSIFICATION
        history.append(
            EnterpriseExperimentHistory(
                source_path=path.relative_to(root).as_posix(),
                source_schema_version=schema,
                source_status=status,
                source_decision=decision,
                source_file_sha256=sha256(path.read_bytes()).hexdigest(),
                source_classification=history_classification,
                disposition=disposition,
            )
        )
    return tuple(history)


def _asset(
    root: Path,
    path: Path,
    document: dict[str, Any],
    *,
    domain: str,
    asset_kind: str,
    role: str,
    runtime_eligible: bool,
    capabilities: tuple[str, ...],
    source_status: str | None = None,
) -> AdoptedEnterpriseAsset:
    schema = document.get("schema_version")
    classification = document.get("classification") or document.get("marker")
    observed_status = source_status or document.get("status") or document.get("decision")
    chain_value = document.get("evidence_chain_sha256")
    if chain_value is None:
        chain_value = document.get("manifest_sha256")
    if (
        not isinstance(schema, str)
        or not schema
        or not isinstance(classification, str)
        or not classification
        or not isinstance(observed_status, str)
        or not observed_status
        or not capabilities
        or (chain_value is not None and not _is_sha256(chain_value))
    ):
        raise EnterpriseProjectAdoptionError("enterprise_adoption_asset_is_invalid")
    return AdoptedEnterpriseAsset(
        domain=domain,
        asset_kind=asset_kind,
        role=role,
        source_path=path.relative_to(root).as_posix(),
        source_schema_version=schema,
        source_classification=classification,
        source_status=observed_status,
        source_file_sha256=sha256(path.read_bytes()).hexdigest(),
        source_evidence_chain_sha256=chain_value,
        runtime_eligible=runtime_eligible,
        capabilities=capabilities,
    )


def _closure_role(domain: str) -> str:
    return {
        "enterprise_systems": "AUTHORITATIVE_ENTERPRISE_PROJECT_SYSTEMS",
        "supplier_a2a_collaboration": "SUPPLIER_A2A_SANDBOX_ACCEPTANCE_VERIFIED",
        "maintenance_planning_value_evaluation": (
            "MAINTENANCE_PLANNING_PROJECT_CANDIDATE_ACCEPTANCE_VERIFIED"
        ),
        "project_assurance_lab": "PROJECT_ENTERPRISE_ASSURANCE_ACCEPTANCE_VERIFIED",
        "llm_training_and_kserve": "ACTIVE_ENTERPRISE_LLM_MODEL_AND_SERVING",
        "vlm_enterprise_staging_adoption": "ACTIVE_ENTERPRISE_VLM_MODEL_AND_SERVING",
        "vlm_kserve_rollout": "ACTIVE_ENTERPRISE_VLM_KSERVE",
        "dpo_post_training": "DPO_POST_TRAINING_SOURCE_REJECTED_AT_AGENT_RUNTIME",
        "dpo_agent_runtime_evaluation": "GOVERNED_REJECTED_DPO_AGENT_RUNTIME_OUTCOME",
        "dpo_enterprise_value": ("AUTHORIZED_ENTERPRISE_DPO_MODEL_RELEASE_CANDIDATE"),
        "grpo_post_training": "GRPO_POST_TRAINING_SOURCE_VALIDATED_AT_AGENT_RUNTIME",
        "grpo_agent_runtime_value": ("AUTHORIZED_ENTERPRISE_GRPO_MODEL_RELEASE_CANDIDATE"),
        "grpo_kserve_rollout": "GRPO_KSERVE_RELEASE_ACCEPTANCE_VERIFIED",
        "ppo_research_safety": "PPO_POST_TRAINING_SOURCE_VALIDATED_AT_AGENT_RUNTIME",
        "ppo_agent_runtime_value": "AUTHORIZED_ENTERPRISE_PPO_MODEL_RELEASE_CANDIDATE",
        "ppo_kserve_rollout": "PPO_KSERVE_RELEASE_ACCEPTANCE_VERIFIED",
        "asr_enterprise_value": "AUTHORIZED_ENTERPRISE_ASR_EVALUATION_CANDIDATE",
        "asr_kserve_rollout": "ASR_KSERVE_RELEASE_ACCEPTANCE_VERIFIED",
        "enterprise_model_import_preflight": (
            "ENTERPRISE_MODEL_IMPORT_PREFLIGHT_ACCEPTANCE_VERIFIED"
        ),
        "rejected_model_experiment_governance": ("GOVERNED_MODEL_EXPERIMENT_OUTCOMES"),
        "reranker_kserve_rollout": "RERANKER_KSERVE_RELEASE_ACCEPTANCE_VERIFIED",
        "enterprise_candidate_release_rollout": ("DPO_TTS_EMBEDDING_RELEASE_ROLLOUT_VERIFIED"),
        "predictive_maintenance_training": "ACTIVE_ENTERPRISE_RUL_MODEL",
        "predictive_maintenance_kserve": "ACTIVE_ENTERPRISE_RUL_KSERVE",
    }.get(domain, "AUTHORITATIVE_ENTERPRISE_PROJECT_CAPABILITY")


def _load_chained(
    path: Path,
    *,
    schema: str,
    status: str | None,
    classification: str,
) -> dict[str, Any]:
    document = _load_object(path)
    chain = document.get("evidence_chain_sha256")
    if (
        document.get("schema_version") != schema
        or document.get("classification") != classification
        or document.get("production_claim") is not False
        or (status is not None and document.get("status") != status)
        or not _is_sha256(chain)
        or chain != _document_digest(document)
    ):
        raise EnterpriseProjectAdoptionError(
            f"enterprise_adoption_source_contract_failed:{path.name}"
        )
    return document


def _load_report(path: Path) -> EnterpriseProjectAdoptionReport:
    try:
        report = EnterpriseProjectAdoptionReport.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise EnterpriseProjectAdoptionError(
            "enterprise_project_adoption_report_is_invalid"
        ) from exc
    if report.evidence_chain_sha256 != _report_digest(report):
        raise EnterpriseProjectAdoptionError("enterprise_project_adoption_report_digest_mismatch")
    return report


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EnterpriseProjectAdoptionError(
            f"enterprise_adoption_json_is_invalid:{path.name}"
        ) from exc
    if not isinstance(value, dict):
        raise EnterpriseProjectAdoptionError(f"enterprise_adoption_json_is_invalid:{path.name}")
    return value


def _inside_file(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise EnterpriseProjectAdoptionError(
            "enterprise_adoption_path_is_outside_repository"
        ) from exc
    if not resolved.is_file():
        raise EnterpriseProjectAdoptionError("enterprise_adoption_path_is_not_a_file")
    return resolved


def _inside_directory(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise EnterpriseProjectAdoptionError(
            "enterprise_adoption_path_is_outside_repository"
        ) from exc
    if not resolved.is_dir():
        raise EnterpriseProjectAdoptionError("enterprise_adoption_path_is_not_a_directory")
    return resolved


def _output_file(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise EnterpriseProjectAdoptionError(
            "enterprise_adoption_output_is_outside_repository"
        ) from exc
    if resolved.exists() and not resolved.is_file():
        raise EnterpriseProjectAdoptionError("enterprise_adoption_output_is_not_a_file")
    return resolved


def _write_atomic(report: EnterpriseProjectAdoptionReport, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_bytes(_canonical(report.model_dump(mode="json")) + b"\n")
    temporary.replace(target)


def _report_digest(report: EnterpriseProjectAdoptionReport) -> str:
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    return sha256(_canonical(unsigned)).hexdigest()


def _document_digest(document: dict[str, Any]) -> str:
    unsigned = dict(document)
    unsigned.pop("evidence_chain_sha256", None)
    return sha256(_canonical(unsigned)).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
