"""Run and verify the local M7 simulated multimodal platform import lab."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.data_pipeline.contracts import (
    production_release_blocker_for_dataset_contract,
)
from industrial_ops_agent.data_pipeline.lineage import RecordingLineageEmitter
from industrial_ops_agent.data_pipeline.storage import FilesystemDatasetStore
from industrial_ops_agent.evaluation.dataset import FrozenEvaluationDatasetLoader
from industrial_ops_agent.experiments.service import EvaluationGovernanceService
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    Base,
    DatasetSnapshotRecord,
    EvaluationSuiteRecord,
    TenantRecord,
    TrainingExperimentRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.security_audit import InMemorySecurityAuditSink, SecurityAuditor
from industrial_ops_agent.simulation.multimodal_snapshot import (
    SIMULATION_TASK_TYPE,
    SimulatedMultimodalSnapshotService,
)
from industrial_ops_agent.training.dataset import GovernedSftDatasetLoader

SCHEMA_VERSION = "simulated-multimodal-platform-import/v1"
STATUS = "SIMULATED_MULTIMODAL_PLATFORM_IMPORT_PASSED"
CLASSIFICATION = "SIMULATED_NON_PRODUCTION"
TENANT_ID = "tenant-simulation-lab"
TRAINING_EXPERIMENT_ID = "experiment-m7-simulated-vlm-dataset"
SUITE_NAME = "m7-simulated-multimodal-reference"
SUITE_VERSION = "1.0.0"


class SimulatedMultimodalPlatformLabError(RuntimeError):
    """The local import lab is missing, inconsistent, or unsafe."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SnapshotEvidence(_ClosedModel):
    snapshot_id: str
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(gt=0)
    split_counts: dict[str, int]
    purpose: str


class SimulatedMultimodalPlatformReport(_ClosedModel):
    schema_version: Literal["simulated-multimodal-platform-import/v1"] = (
        "simulated-multimodal-platform-import/v1"
    )
    classification: Literal["SIMULATED_NON_PRODUCTION"] = "SIMULATED_NON_PRODUCTION"
    production_claim: Literal[False] = False
    production_release_eligible: Literal[False] = False
    status: Literal["SIMULATED_MULTIMODAL_PLATFORM_IMPORT_PASSED"] = (
        "SIMULATED_MULTIMODAL_PLATFORM_IMPORT_PASSED"
    )
    source_dataset_id: str
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_snapshot: SnapshotEvidence
    evaluation_snapshot: SnapshotEvidence
    training_experiment_id: str
    training_task_type: Literal["SIMULATED_MULTIMODAL_VLM"] = "SIMULATED_MULTIMODAL_VLM"
    training_records: int = Field(gt=0)
    validation_records: int = Field(gt=0)
    simulation_reference_suite_id: str
    simulation_reference_cases: int = Field(gt=0)
    simulation_reference_tier: Literal["SIMULATION_REFERENCE"] = "SIMULATION_REFERENCE"
    feedback_candidate_projection_count: int = Field(gt=0)
    label_studio_review_projection_count: int = Field(gt=0)
    label_studio_external_service_called: Literal[False] = False
    presidio_dlp_status: Literal["PASSED"] = "PASSED"
    openlineage_events: int = Field(ge=4)
    release_gate: Literal["DENIED_SIMULATION_CONTRACT"] = "DENIED_SIMULATION_CONTRACT"
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_simulated_multimodal_platform_lab(
    repo_root: Path,
    *,
    source_dir: Path,
    output_dir: Path,
) -> tuple[SimulatedMultimodalPlatformReport, bool]:
    root = repo_root.resolve(strict=True)
    source = _inside(root, source_dir)
    output = output_dir if output_dir.is_absolute() else root / output_dir
    output = output.resolve()
    if output.exists():
        return verify_simulated_multimodal_platform_lab(output), False
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        report = _build_into(source=source, output=temporary)
        _write_new(temporary / "acceptance.json", _json_bytes(report.model_dump(mode="json")))
        verify_simulated_multimodal_platform_lab(temporary)
        temporary.rename(output)
    except Exception:
        _safe_remove_temporary(temporary, output.parent)
        raise
    return verify_simulated_multimodal_platform_lab(output), True


def verify_simulated_multimodal_platform_lab(
    output_dir: Path,
) -> SimulatedMultimodalPlatformReport:
    output = output_dir.resolve(strict=True)
    report_path = output / "acceptance.json"
    try:
        report = SimulatedMultimodalPlatformReport.model_validate_json(report_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise SimulatedMultimodalPlatformLabError(
            "multimodal_platform_acceptance_is_invalid"
        ) from exc
    if report.evidence_chain_sha256 != _report_hash(report):
        raise SimulatedMultimodalPlatformLabError("multimodal_platform_acceptance_digest_mismatch")
    database_path = output / "catalog.sqlite3"
    object_store_path = output / "object-store"
    if not database_path.is_file() or not object_store_path.is_dir():
        raise SimulatedMultimodalPlatformLabError("multimodal_platform_catalog_is_missing")
    database = Database(f"sqlite+pysqlite:///{database_path}")
    store = FilesystemDatasetStore(object_store_path)
    context = TenantContext(TENANT_ID, "simulation-lab-verifier")
    try:
        with database.transaction(context) as session:
            experiment = session.get(TrainingExperimentRecord, report.training_experiment_id)
            suite = session.get(
                EvaluationSuiteRecord,
                report.simulation_reference_suite_id,
            )
            training_snapshot = session.get(
                DatasetSnapshotRecord,
                report.training_snapshot.snapshot_id,
            )
        if experiment is None or suite is None or training_snapshot is None:
            raise SimulatedMultimodalPlatformLabError(
                "multimodal_platform_catalog_binding_is_missing"
            )
        if (
            experiment.task_type != SIMULATION_TASK_TYPE
            or experiment.method != "VLM"
            or experiment.dataset_snapshot_id != report.training_snapshot.snapshot_id
            or experiment.dataset_manifest_hash != report.training_snapshot.manifest_hash
            or suite.tier != "SIMULATION_REFERENCE"
            or suite.source_snapshot_id != report.evaluation_snapshot.snapshot_id
            or suite.manifest_hash != report.evaluation_snapshot.manifest_hash
            or production_release_blocker_for_dataset_contract(training_snapshot.contract_version)
            != "simulated_multimodal_snapshot_is_not_production_releasable"
        ):
            raise SimulatedMultimodalPlatformLabError("multimodal_platform_catalog_binding_changed")
        training = GovernedSftDatasetLoader(database, store).load(context, experiment)
        evaluation = FrozenEvaluationDatasetLoader(database, store).load(context, suite)
        if (
            len(training.train) != report.training_records
            or len(training.validation) != report.validation_records
            or len(evaluation.cases) != report.simulation_reference_cases
            or training.sample_origin_counts
            != {
                "real": 0,
                "synthetic": report.training_records + report.validation_records,
            }
            or any(not sample.synthetic for sample in (*training.train, *training.validation))
            or any(case.vlm is None or case.media_content is None for case in evaluation.cases)
        ):
            raise SimulatedMultimodalPlatformLabError("multimodal_platform_loader_evidence_changed")
    finally:
        database.dispose()
    return report


def _build_into(
    *,
    source: Path,
    output: Path,
) -> SimulatedMultimodalPlatformReport:
    database_path = output / "catalog.sqlite3"
    store_path = output / "object-store"
    database = Database(f"sqlite+pysqlite:///{database_path}")
    Base.metadata.create_all(database.engine)
    now = datetime(2026, 8, 22, 12, tzinfo=UTC)
    context = TenantContext(TENANT_ID, "simulation-platform-importer")
    with database.transaction(context) as session:
        session.add(
            TenantRecord(
                id=TENANT_ID,
                status="ACTIVE",
                display_name="M7 Simulation Lab",
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
    lineage = RecordingLineageEmitter(
        namespace="industrial-ops-simulation",
        job_name="import_m7_simulated_multimodal_dataset",
    )
    store = FilesystemDatasetStore(store_path)
    try:
        publication = SimulatedMultimodalSnapshotService(
            database,
            store,
            lineage,
            code_version="m7-simulated-platform-import-v1",
        ).publish(
            context,
            source_dir=source,
            reviewed_by_subject_id="simulation-domain-reviewer",
            occurred_at=now,
        )
        experiment = _create_training_experiment(
            database,
            context,
            snapshot_id=publication.training.snapshot_id,
            manifest_hash=publication.training.manifest_hash,
            now=now + timedelta(seconds=2),
        )
        training = GovernedSftDatasetLoader(database, store).load(context, experiment)
        identity = _evaluator_identity(now)
        suite = EvaluationGovernanceService(
            database,
            Authorizer(
                SecurityAuditor(
                    InMemorySecurityAuditSink(),
                    hash_key=b"m7-simulation-reference-audit",
                )
            ),
        ).register_suite(
            identity,
            name=SUITE_NAME,
            version=SUITE_VERSION,
            tier="SIMULATION_REFERENCE",
            source_snapshot_id=publication.evaluation.snapshot_id,
            manifest_hash=publication.evaluation.manifest_hash,
            sample_count=publication.evaluation.row_count,
            slice_counts=_evaluation_slice_counts(source),
            request_id="m7-simulation-reference-register",
        )
        evaluation = FrozenEvaluationDatasetLoader(database, store).load(context, suite)
        feedback_count, labeling_count = _projection_counts(
            store,
            publication.training.manifest_key,
            publication.evaluation.manifest_key,
        )
        report = SimulatedMultimodalPlatformReport(
            source_dataset_id=publication.source_dataset_id,
            source_manifest_sha256=publication.source_manifest_sha256,
            training_snapshot=SnapshotEvidence(
                snapshot_id=publication.training.snapshot_id,
                manifest_hash=publication.training.manifest_hash,
                row_count=publication.training.row_count,
                split_counts=publication.training.split_counts,
                purpose=publication.training.purpose,
            ),
            evaluation_snapshot=SnapshotEvidence(
                snapshot_id=publication.evaluation.snapshot_id,
                manifest_hash=publication.evaluation.manifest_hash,
                row_count=publication.evaluation.row_count,
                split_counts=publication.evaluation.split_counts,
                purpose=publication.evaluation.purpose,
            ),
            training_experiment_id=experiment.experiment_id,
            training_records=len(training.train),
            validation_records=len(training.validation),
            simulation_reference_suite_id=suite.suite_id,
            simulation_reference_cases=len(evaluation.cases),
            feedback_candidate_projection_count=feedback_count,
            label_studio_review_projection_count=labeling_count,
            openlineage_events=len(lineage.events),
            evidence_chain_sha256="0" * 64,
        )
        return report.model_copy(update={"evidence_chain_sha256": _report_hash(report)})
    finally:
        database.dispose()


def _create_training_experiment(
    database: Database,
    context: TenantContext,
    *,
    snapshot_id: str,
    manifest_hash: str,
    now: datetime,
) -> TrainingExperimentRecord:
    config = {
        "vision_contract_version": "industrial-vision-instruction-v1",
        "simulation_only": True,
    }
    experiment = TrainingExperimentRecord(
        experiment_id=TRAINING_EXPERIMENT_ID,
        tenant_id=context.tenant_id,
        idempotency_key=TRAINING_EXPERIMENT_ID,
        comparison_group_id="m7-simulated-vlm-comparison",
        method="VLM",
        task_type=SIMULATION_TASK_TYPE,
        status="PLANNED",
        dataset_snapshot_id=snapshot_id,
        dataset_manifest_hash=manifest_hash,
        base_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        base_model_digest="sha256:" + "1" * 64,
        tokenizer_digest="sha256:" + "2" * 64,
        chat_template_digest="sha256:" + "3" * 64,
        git_commit="simulation-local",
        container_digest="sha256:" + "4" * 64,
        training_config=config,
        config_hash="sha256:" + sha256(_json_text(config).encode()).hexdigest(),
        distributed_profile={},
        random_seeds=[20260822],
        hardware_topology={"profile": "LOCAL_SIMULATION", "gpu_required_for_training": True},
        mlflow_experiment_name="m7-simulated-multimodal-vlm",
        mlflow_run_id=None,
        metrics={},
        cost_summary={},
        license_status="SIMULATION_ONLY",
        created_by_subject_id="simulation-model-engineer",
        completed_at=None,
        failure_reason=None,
        version=1,
        created_at=now,
        updated_at=now,
    )
    with database.transaction(context) as session:
        session.add(experiment)
        session.flush()
    return experiment


def _evaluator_identity(now: datetime) -> IdentityContext:
    return IdentityContext(
        subject_id="simulation-model-evaluator",
        oidc_subject="oidc-simulation-model-evaluator",
        tenant_id=TENANT_ID,
        roles=frozenset({Role.MODEL_EVALUATOR}),
        asset_ids=frozenset(),
        site_ids=frozenset(),
        issued_at=now,
        expires_at=now + timedelta(hours=1),
    )


def _evaluation_slice_counts(source: Path) -> dict[str, int]:
    faults: Counter[str] = Counter()
    for line in (source / "cases/simulation_evaluation.jsonl").read_text().splitlines():
        if line.strip():
            value = json.loads(line)
            faults[str(value["fault_code"])] += 1
    if not faults:
        raise SimulatedMultimodalPlatformLabError("simulation_reference_source_is_empty")
    return {f"fault:{key}": count for key, count in sorted(faults.items())}


def _projection_counts(
    store: FilesystemDatasetStore,
    training_manifest_key: str,
    evaluation_manifest_key: str,
) -> tuple[int, int]:
    feedback = 0
    labeling = 0
    for manifest_key in (training_manifest_key, evaluation_manifest_key):
        manifest = json.loads(store.get_bytes(manifest_key))
        for artifact in manifest["artifacts"]:
            if artifact["kind"] == "simulation_feedback":
                feedback += int(artifact["row_count"])
            if artifact["kind"] == "simulation_labeling":
                labeling += int(artifact["row_count"])
    return feedback, labeling


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root) or not resolved.exists():
        raise SimulatedMultimodalPlatformLabError("platform_lab_path_is_outside_repo")
    return resolved


def _report_hash(report: SimulatedMultimodalPlatformReport) -> str:
    document = report.model_dump(mode="json")
    document.pop("evidence_chain_sha256", None)
    return sha256(_json_text(document).encode()).hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"


def _write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(content)


def _safe_remove_temporary(temporary: Path, parent: Path) -> None:
    resolved = temporary.resolve()
    if resolved.parent != parent.resolve() or not resolved.name.startswith("."):
        raise SimulatedMultimodalPlatformLabError("temporary_platform_lab_cleanup_target_is_unsafe")
    shutil.rmtree(resolved, ignore_errors=True)
