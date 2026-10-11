"""Checksum-verified loading of frozen evaluation-only snapshots."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from io import BytesIO
from typing import Any

import pyarrow.parquet as parquet  # type: ignore[import-untyped]
from sqlalchemy import select

from industrial_ops_agent.data_pipeline.contracts import (
    CONTRACT_VERSION,
    DATASET_COLUMNS,
    DATASET_SCHEMA,
    SIMULATED_MULTIMODAL_CONTRACT,
)
from industrial_ops_agent.data_pipeline.quality import (
    DatasetQualityReportError,
    verify_dataset_quality_report,
)
from industrial_ops_agent.data_pipeline.storage import DatasetStore
from industrial_ops_agent.media.validation import MediaValidationFailure, inspect_media
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    DatasetArtifactRecord,
    DatasetSnapshotRecord,
    EvaluationSuiteRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.predictive_maintenance.dataset_contract import (
    CONTRACT_VERSION as TELEMETRY_CONTRACT_VERSION,
)
from industrial_ops_agent.predictive_maintenance.dataset_contract import (
    DATASET_COLUMNS as TELEMETRY_DATASET_COLUMNS,
)
from industrial_ops_agent.predictive_maintenance.dataset_contract import (
    DATASET_SCHEMA as TELEMETRY_DATASET_SCHEMA,
)
from industrial_ops_agent.predictive_maintenance.dataset_contract import (
    SIGNAL_ORDER,
)
from industrial_ops_agent.predictive_maintenance.rul_dataset_contract import (
    CONTRACT_VERSION as RUL_CONTRACT_VERSION,
)
from industrial_ops_agent.predictive_maintenance.rul_dataset_contract import (
    DATASET_COLUMNS as RUL_DATASET_COLUMNS,
)
from industrial_ops_agent.predictive_maintenance.rul_dataset_contract import (
    DATASET_SCHEMA as RUL_DATASET_SCHEMA,
)
from industrial_ops_agent.predictive_maintenance.rul_dataset_contract import (
    SIGNAL_ORDER as RUL_SIGNAL_ORDER,
)
from industrial_ops_agent.training.dataset import (
    TrainingDatasetError,
    classification_system_instruction,
)


class EvaluationDatasetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RetrievalDocument:
    document_id: str
    text: str
    relevance: int
    access: str


@dataclass(frozen=True, slots=True)
class RetrievalContract:
    query: str
    documents: tuple[RetrievalDocument, ...]

    @property
    def allowed_document_ids(self) -> frozenset[str]:
        return frozenset(
            document.document_id for document in self.documents if document.access == "ALLOWED"
        )

    @property
    def forbidden_document_ids(self) -> frozenset[str]:
        return frozenset(
            document.document_id for document in self.documents if document.access == "FORBIDDEN"
        )

    @property
    def relevant_document_ids(self) -> frozenset[str]:
        return frozenset(
            document.document_id
            for document in self.documents
            if document.access == "ALLOWED" and document.relevance > 0
        )


@dataclass(frozen=True, slots=True)
class NormalizedRegion:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class VlmExpectedFinding:
    label: str
    region: NormalizedRegion


@dataclass(frozen=True, slots=True)
class VlmEvaluationContract:
    instruction: str
    findings: tuple[VlmExpectedFinding, ...]
    forbidden_labels: frozenset[str]


@dataclass(frozen=True, slots=True)
class AsrEvaluationContract:
    transcript: str
    language: str
    noise_condition: str


@dataclass(frozen=True, slots=True)
class TtsEvaluationContract:
    target_text: str
    language: str
    voice_profile_id: str
    required_safety_phrases: tuple[str, ...]
    industrial_terms: tuple[str, ...]
    risk: str


@dataclass(frozen=True, slots=True)
class PpoSafetyEvaluationContract:
    preferred_response: str
    adversarial_response: str
    attack_category: str
    forbidden_patterns: tuple[str, ...]
    minimum_reward_margin: float


@dataclass(frozen=True, slots=True)
class TelemetryEvaluationContract:
    asset_id: str
    sequence: tuple[tuple[float, ...], ...]
    mask: tuple[tuple[bool, ...], ...]
    label: int
    duration_minutes: float


@dataclass(frozen=True, slots=True)
class RulEvaluationContract:
    asset_id: str
    schema_version: str
    sequence: tuple[tuple[float, ...], ...]
    mask: tuple[tuple[bool, ...], ...]
    lead_time_minutes: float
    outcome_evidence_digest: str


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    prompt: tuple[dict[str, str], ...]
    expected_output: dict[str, Any]
    slices: dict[str, str]
    required_gates: tuple[str, ...]
    allowed_citations: tuple[str, ...]
    forbidden_substrings: tuple[str, ...]
    expected_decision: str | None
    risk: str
    contexts: tuple[str, ...]
    retrieval: RetrievalContract | None = None
    vlm: VlmEvaluationContract | None = None
    asr: AsrEvaluationContract | None = None
    tts: TtsEvaluationContract | None = None
    ppo_safety: PpoSafetyEvaluationContract | None = None
    telemetry: TelemetryEvaluationContract | None = None
    rul: RulEvaluationContract | None = None
    media_id: str | None = None
    media_content: bytes | None = None
    media_mime_type: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationDatasetBundle:
    suite_id: str
    snapshot_id: str
    manifest_hash: str
    cases: tuple[EvaluationCase, ...]
    split_hashes: dict[str, str]


@dataclass(frozen=True, slots=True)
class _Artifact:
    split: str
    object_key: str
    content_hash: str
    size_bytes: int
    row_count: int
    schema_hash: str


@dataclass(frozen=True, slots=True)
class _MediaArtifact:
    source_media_id: str
    split: str
    object_key: str
    content_hash: str
    size_bytes: int
    schema_hash: str
    mime_type: str
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _GovernedMedia:
    source_media_id: str
    split: str
    mime_type: str
    content: bytes
    candidate_ids: tuple[str, ...]


class FrozenEvaluationDatasetLoader:
    def __init__(self, database: Database, store: DatasetStore) -> None:
        self._database = database
        self._store = store

    def load(
        self,
        context: TenantContext,
        suite: EvaluationSuiteRecord,
    ) -> EvaluationDatasetBundle:
        if suite.tenant_id != context.tenant_id:
            raise EvaluationDatasetError("evaluation_suite_tenant_mismatch")
        if suite.status != "FROZEN" or suite.purpose != "EVALUATION_ONLY":
            raise EvaluationDatasetError("evaluation_suite_is_not_frozen")
        with self._database.transaction(context) as session:
            snapshot = session.scalar(
                select(DatasetSnapshotRecord).where(
                    DatasetSnapshotRecord.tenant_id == context.tenant_id,
                    DatasetSnapshotRecord.snapshot_id == suite.source_snapshot_id,
                )
            )
            if snapshot is None:
                raise EvaluationDatasetError("evaluation_snapshot_not_visible")
            if (
                snapshot.status != "CANDIDATE"
                or not snapshot.training_eligible
                or snapshot.lineage_status != "CONFIRMED"
                or not snapshot.manifest_key
                or not snapshot.manifest_hash
            ):
                raise EvaluationDatasetError("evaluation_snapshot_is_not_governed")
            if (
                snapshot.manifest_hash != suite.manifest_hash
                or snapshot.row_count != suite.sample_count
            ):
                raise EvaluationDatasetError("evaluation_suite_snapshot_binding_changed")
            catalog = list(
                session.scalars(
                    select(DatasetArtifactRecord).where(
                        DatasetArtifactRecord.tenant_id == context.tenant_id,
                        DatasetArtifactRecord.snapshot_id == snapshot.snapshot_id,
                    )
                )
            )
            if any(item.kind == "synthetic_provenance" for item in catalog):
                raise EvaluationDatasetError(
                    "synthetic_training_snapshot_is_not_evaluation_eligible"
                )
            manifest_key = snapshot.manifest_key
            snapshot_id = snapshot.snapshot_id
            split_counts = dict(snapshot.split_counts)
            snapshot_contract = str(snapshot.contract_version)
            quality_artifacts = [item for item in catalog if item.kind == "quality_report"]
            quality_key = snapshot.quality_report_key

        manifest_bytes = self._store.get_bytes(manifest_key)
        manifest_hash = _digest(manifest_bytes)
        if manifest_hash != suite.manifest_hash.removeprefix("sha256:"):
            raise EvaluationDatasetError("evaluation_manifest_hash_mismatch")
        manifest = _json_dict(manifest_bytes, "evaluation_manifest_is_invalid")
        if snapshot_contract in {TELEMETRY_CONTRACT_VERSION, RUL_CONTRACT_VERSION}:
            if quality_key is None or len(quality_artifacts) != 1:
                raise EvaluationDatasetError("evaluation_quality_report_is_missing")
            try:
                verify_dataset_quality_report(
                    snapshot,
                    quality_artifacts[0],
                    self._store.get_bytes(quality_key),
                )
            except (FileNotFoundError, DatasetQualityReportError) as exc:
                raise EvaluationDatasetError("evaluation_quality_report_integrity_failed") from exc
            if snapshot_contract == TELEMETRY_CONTRACT_VERSION:
                return _telemetry_bundle(
                    store=self._store,
                    context=context,
                    suite=suite,
                    snapshot_id=snapshot_id,
                    manifest_hash=suite.manifest_hash,
                    split_counts=split_counts,
                    manifest=manifest,
                    catalog=catalog,
                )
            return _rul_bundle(
                store=self._store,
                context=context,
                suite=suite,
                snapshot_id=snapshot_id,
                manifest_hash=suite.manifest_hash,
                split_counts=split_counts,
                manifest=manifest,
                catalog=catalog,
            )
        is_simulation_reference = snapshot_contract == SIMULATED_MULTIMODAL_CONTRACT
        if is_simulation_reference:
            if suite.tier != "SIMULATION_REFERENCE":
                raise EvaluationDatasetError(
                    "simulated_multimodal_snapshot_requires_simulation_reference_suite"
                )
            if quality_key is None or len(quality_artifacts) != 1:
                raise EvaluationDatasetError("evaluation_quality_report_is_missing")
            try:
                verify_dataset_quality_report(
                    snapshot,
                    quality_artifacts[0],
                    self._store.get_bytes(quality_key),
                )
            except (FileNotFoundError, DatasetQualityReportError) as exc:
                raise EvaluationDatasetError("evaluation_quality_report_integrity_failed") from exc
        elif suite.tier == "SIMULATION_REFERENCE":
            raise EvaluationDatasetError(
                "simulation_reference_suite_requires_simulated_multimodal_snapshot"
            )
        if (
            manifest.get("tenant_id") != context.tenant_id
            or manifest.get("snapshot_id") != snapshot_id
            or manifest.get("contract_version")
            != (SIMULATED_MULTIMODAL_CONTRACT if is_simulation_reference else CONTRACT_VERSION)
            or manifest.get("lineage_status") != "CONFIRMED"
            or manifest.get("row_count") != suite.sample_count
            or manifest.get("split_counts") != split_counts
        ):
            raise EvaluationDatasetError("evaluation_manifest_binding_mismatch")
        if is_simulation_reference:
            profile = manifest.get("simulation_profile")
            if (
                manifest.get("classification") != "SIMULATED_NON_PRODUCTION"
                or manifest.get("production_claim") is not False
                or manifest.get("production_release_eligible") is not False
                or not isinstance(profile, dict)
                or profile.get("purpose") != "SIMULATION_REFERENCE_EVALUATION"
                or profile.get("production_gold_eligible") is not False
            ):
                raise EvaluationDatasetError("simulation_reference_governance_contract_is_invalid")

        artifacts = _manifest_artifacts(manifest)
        media_artifacts = _manifest_media_artifacts(manifest)
        _verify_catalog(artifacts, media_artifacts, catalog)
        governed_media = _load_governed_media(self._store, media_artifacts)
        cases: list[EvaluationCase] = []
        split_hashes: dict[str, str] = {}
        seen_case_ids: set[str] = set()
        for split in ("train", "validation", "test"):
            artifact = artifacts[split]
            content = self._store.get_bytes(artifact.object_key)
            if len(content) != artifact.size_bytes or _digest(content) != artifact.content_hash:
                raise EvaluationDatasetError(f"evaluation_{split}_integrity_failed")
            frame = parquet.read_table(BytesIO(content)).to_pandas()
            frame = frame.loc[:, list(DATASET_COLUMNS)]
            DATASET_SCHEMA.validate(frame, lazy=True)
            observed_splits = set(frame["split"].astype(str))
            expected_splits = set() if frame.empty else {split}
            if len(frame) != artifact.row_count or observed_splits != expected_splits:
                raise EvaluationDatasetError(f"evaluation_{split}_row_contract_failed")
            if not frame.empty and set(frame["tenant_id"].astype(str)) != {context.tenant_id}:
                raise EvaluationDatasetError(f"evaluation_{split}_tenant_mismatch")
            split_hashes[split] = artifact.content_hash
            for row in frame.to_dict("records"):
                case = _case(row, governed_media)
                if case.case_id in seen_case_ids:
                    raise EvaluationDatasetError("evaluation_case_id_is_not_unique")
                seen_case_ids.add(case.case_id)
                cases.append(case)
        cases.sort(key=lambda item: item.case_id)
        if len(cases) != suite.sample_count:
            raise EvaluationDatasetError("evaluation_case_count_mismatch")
        if not cases:
            raise EvaluationDatasetError("evaluation_suite_is_empty")
        return EvaluationDatasetBundle(
            suite_id=suite.suite_id,
            snapshot_id=snapshot_id,
            manifest_hash=manifest_hash,
            cases=tuple(cases),
            split_hashes=split_hashes,
        )


def _telemetry_bundle(
    *,
    store: DatasetStore,
    context: TenantContext,
    suite: EvaluationSuiteRecord,
    snapshot_id: str,
    manifest_hash: str,
    split_counts: dict[str, int],
    manifest: dict[str, Any],
    catalog: list[DatasetArtifactRecord],
) -> EvaluationDatasetBundle:
    expected_manifest = {
        "tenant_id": context.tenant_id,
        "snapshot_id": snapshot_id,
        "contract_version": TELEMETRY_CONTRACT_VERSION,
        "lineage_status": "CONFIRMED",
        "row_count": suite.sample_count,
        "split_counts": split_counts,
        "signal_order": list(SIGNAL_ORDER),
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise EvaluationDatasetError("telemetry_evaluation_manifest_binding_mismatch")
    artifacts = _manifest_artifacts(manifest)
    _verify_telemetry_catalog(artifacts, catalog)
    cases: list[EvaluationCase] = []
    split_hashes: dict[str, str] = {}
    group_keys: dict[str, set[str]] = {}
    labels: set[int] = set()
    case_ids: set[str] = set()
    for split in ("train", "validation", "test"):
        artifact = artifacts[split]
        content = store.get_bytes(artifact.object_key)
        if len(content) != artifact.size_bytes or _digest(
            content
        ) != artifact.content_hash.removeprefix("sha256:"):
            raise EvaluationDatasetError(f"telemetry_evaluation_{split}_integrity_failed")
        frame = parquet.read_table(BytesIO(content)).to_pandas()
        frame = frame.loc[:, list(TELEMETRY_DATASET_COLUMNS)]
        TELEMETRY_DATASET_SCHEMA.validate(frame, lazy=True)
        observed_splits = set(frame["split"].astype(str))
        if (
            len(frame) != artifact.row_count
            or observed_splits != (set() if frame.empty else {split})
            or (not frame.empty and set(frame["tenant_id"].astype(str)) != {context.tenant_id})
        ):
            raise EvaluationDatasetError(f"telemetry_evaluation_{split}_row_contract_failed")
        group_keys[split] = set(frame["group_key"].astype(str))
        split_hashes[split] = artifact.content_hash
        for row in frame.to_dict("records"):
            case = _telemetry_case(row)
            if case.case_id in case_ids:
                raise EvaluationDatasetError("telemetry_evaluation_case_id_is_not_unique")
            case_ids.add(case.case_id)
            assert case.telemetry is not None
            labels.add(case.telemetry.label)
            cases.append(case)
    if (
        group_keys["train"] & group_keys["validation"]
        or group_keys["train"] & group_keys["test"]
        or group_keys["validation"] & group_keys["test"]
    ):
        raise EvaluationDatasetError("telemetry_evaluation_equipment_split_leakage")
    if len(cases) != suite.sample_count or labels != {0, 1}:
        raise EvaluationDatasetError("telemetry_evaluation_label_coverage_is_incomplete")
    cases.sort(key=lambda item: item.case_id)
    return EvaluationDatasetBundle(
        suite_id=suite.suite_id,
        snapshot_id=snapshot_id,
        manifest_hash=manifest_hash,
        cases=tuple(cases),
        split_hashes=split_hashes,
    )


def _rul_bundle(
    *,
    store: DatasetStore,
    context: TenantContext,
    suite: EvaluationSuiteRecord,
    snapshot_id: str,
    manifest_hash: str,
    split_counts: dict[str, int],
    manifest: dict[str, Any],
    catalog: list[DatasetArtifactRecord],
) -> EvaluationDatasetBundle:
    expected_manifest = {
        "tenant_id": context.tenant_id,
        "snapshot_id": snapshot_id,
        "contract_version": RUL_CONTRACT_VERSION,
        "lineage_status": "CONFIRMED",
        "row_count": suite.sample_count,
        "split_counts": split_counts,
        "signal_order": list(RUL_SIGNAL_ORDER),
        "target": "lead_time_minutes_from_alert_detection",
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise EvaluationDatasetError("rul_evaluation_manifest_binding_mismatch")
    artifacts = _manifest_artifacts(manifest)
    _verify_rul_catalog(artifacts, catalog)
    cases: list[EvaluationCase] = []
    split_hashes: dict[str, str] = {}
    group_keys: dict[str, set[str]] = {}
    case_ids: set[str] = set()
    for split in ("train", "validation", "test"):
        artifact = artifacts[split]
        content = store.get_bytes(artifact.object_key)
        if len(content) != artifact.size_bytes or _digest(
            content
        ) != artifact.content_hash.removeprefix("sha256:"):
            raise EvaluationDatasetError(f"rul_evaluation_{split}_integrity_failed")
        frame = parquet.read_table(BytesIO(content)).to_pandas()
        frame = frame.loc[:, list(RUL_DATASET_COLUMNS)]
        RUL_DATASET_SCHEMA.validate(frame, lazy=True)
        observed_splits = set(frame["split"].astype(str))
        if (
            len(frame) != artifact.row_count
            or observed_splits != (set() if frame.empty else {split})
            or (not frame.empty and set(frame["tenant_id"].astype(str)) != {context.tenant_id})
        ):
            raise EvaluationDatasetError(f"rul_evaluation_{split}_row_contract_failed")
        group_keys[split] = set(frame["group_key"].astype(str))
        split_hashes[split] = artifact.content_hash
        for row in frame.to_dict("records"):
            case = _rul_case(row)
            if case.case_id in case_ids:
                raise EvaluationDatasetError("rul_evaluation_case_id_is_not_unique")
            case_ids.add(case.case_id)
            cases.append(case)
    if (
        group_keys["train"] & group_keys["validation"]
        or group_keys["train"] & group_keys["test"]
        or group_keys["validation"] & group_keys["test"]
    ):
        raise EvaluationDatasetError("rul_evaluation_equipment_split_leakage")
    if len(cases) != suite.sample_count or not cases:
        raise EvaluationDatasetError("rul_evaluation_target_coverage_is_incomplete")
    cases.sort(key=lambda item: item.case_id)
    return EvaluationDatasetBundle(
        suite_id=suite.suite_id,
        snapshot_id=snapshot_id,
        manifest_hash=manifest_hash,
        cases=tuple(cases),
        split_hashes=split_hashes,
    )


def _rul_case(row: dict[str, Any]) -> EvaluationCase:
    try:
        sequence_raw = json.loads(str(row["sequence_json"]))
        mask_raw = json.loads(str(row["mask_json"]))
        target = float(row["lead_time_minutes"])
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise EvaluationDatasetError("rul_evaluation_sequence_is_invalid") from exc
    if (
        not isinstance(sequence_raw, list)
        or not isinstance(mask_raw, list)
        or len(sequence_raw) != len(mask_raw)
        or len(sequence_raw) != int(row["sample_count"])
        or len(sequence_raw) < 3
        or not math.isfinite(target)
        or target <= 0
    ):
        raise EvaluationDatasetError("rul_evaluation_sequence_shape_is_invalid")
    sequence: list[tuple[float, ...]] = []
    mask: list[tuple[bool, ...]] = []
    for values_raw, present_raw in zip(sequence_raw, mask_raw, strict=True):
        if (
            not isinstance(values_raw, list)
            or not isinstance(present_raw, list)
            or len(values_raw) != len(RUL_SIGNAL_ORDER)
            or len(present_raw) != len(RUL_SIGNAL_ORDER)
        ):
            raise EvaluationDatasetError("rul_evaluation_signal_shape_is_invalid")
        values = tuple(float(value) for value in values_raw)
        if not all(math.isfinite(value) for value in values):
            raise EvaluationDatasetError("rul_evaluation_signal_is_non_finite")
        if any(value not in {0, 1} for value in present_raw):
            raise EvaluationDatasetError("rul_evaluation_mask_is_invalid")
        sequence.append(values)
        mask.append(tuple(bool(value) for value in present_raw))
    evidence_digest = str(row["outcome_evidence_digest"])
    contract = RulEvaluationContract(
        asset_id=str(row["asset_id"]),
        schema_version=str(row["schema_version"]),
        sequence=tuple(sequence),
        mask=tuple(mask),
        lead_time_minutes=target,
        outcome_evidence_digest=evidence_digest,
    )
    lead_time_slice = "lt_4h" if target < 240 else "4h_24h" if target <= 1440 else "gt_24h"
    return EvaluationCase(
        case_id=str(row["window_id"]),
        prompt=(),
        expected_output={"lead_time_minutes": target},
        slices={
            "asset_id": contract.asset_id,
            "schema_version": contract.schema_version,
            "split_origin": str(row["split"]),
            "lead_time": lead_time_slice,
        },
        required_gates=(),
        allowed_citations=(),
        forbidden_substrings=(),
        expected_decision=None,
        risk="HIGH",
        contexts=(),
        rul=contract,
    )


def _telemetry_case(row: dict[str, Any]) -> EvaluationCase:
    try:
        sequence_raw = json.loads(str(row["sequence_json"]))
        mask_raw = json.loads(str(row["mask_json"]))
        started = datetime.fromisoformat(str(row["window_start"]).replace("Z", "+00:00"))
        ended = datetime.fromisoformat(str(row["window_end"]).replace("Z", "+00:00"))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise EvaluationDatasetError("telemetry_evaluation_sequence_is_invalid") from exc
    if (
        not isinstance(sequence_raw, list)
        or not isinstance(mask_raw, list)
        or len(sequence_raw) != len(mask_raw)
        or len(sequence_raw) != int(row["sample_count"])
        or len(sequence_raw) < 3
    ):
        raise EvaluationDatasetError("telemetry_evaluation_sequence_shape_is_invalid")
    sequence: list[tuple[float, ...]] = []
    mask: list[tuple[bool, ...]] = []
    for values_raw, present_raw in zip(sequence_raw, mask_raw, strict=True):
        if (
            not isinstance(values_raw, list)
            or not isinstance(present_raw, list)
            or len(values_raw) != len(SIGNAL_ORDER)
            or len(present_raw) != len(SIGNAL_ORDER)
        ):
            raise EvaluationDatasetError("telemetry_evaluation_signal_shape_is_invalid")
        values = tuple(float(value) for value in values_raw)
        if not all(math.isfinite(value) for value in values):
            raise EvaluationDatasetError("telemetry_evaluation_signal_is_non_finite")
        if any(value not in {0, 1} for value in present_raw):
            raise EvaluationDatasetError("telemetry_evaluation_mask_is_invalid")
        sequence.append(values)
        mask.append(tuple(bool(value) for value in present_raw))
    label = int(row["label"])
    duration_minutes = (ended - started).total_seconds() / 60
    if label not in {0, 1} or duration_minutes <= 0:
        raise EvaluationDatasetError("telemetry_evaluation_label_or_window_is_invalid")
    contract = TelemetryEvaluationContract(
        asset_id=str(row["asset_id"]),
        sequence=tuple(sequence),
        mask=tuple(mask),
        label=label,
        duration_minutes=duration_minutes,
    )
    return EvaluationCase(
        case_id=str(row["window_id"]),
        prompt=(),
        expected_output={"label": label},
        slices={
            "asset_id": contract.asset_id,
            "split_origin": str(row["split"]),
            "label": "anomaly" if label else "normal",
        },
        required_gates=(),
        allowed_citations=(),
        forbidden_substrings=(),
        expected_decision=None,
        risk="HIGH" if label else "NORMAL",
        contexts=(),
        telemetry=contract,
    )


def _verify_telemetry_catalog(
    manifest: dict[str, _Artifact], catalog_records: list[DatasetArtifactRecord]
) -> None:
    catalog = {
        str(record.split): record
        for record in catalog_records
        if record.kind == "parquet" and record.split is not None
    }
    expected_schema_hash = (
        "sha256:" + sha256("\0".join(TELEMETRY_DATASET_COLUMNS).encode()).hexdigest()
    )
    if set(catalog) != set(manifest):
        raise EvaluationDatasetError("telemetry_evaluation_catalog_and_manifest_differ")
    for split, artifact in manifest.items():
        record = catalog[split]
        if (
            artifact.object_key != record.object_key
            or artifact.content_hash != record.content_hash
            or artifact.size_bytes != record.size_bytes
            or artifact.row_count != record.row_count
            or artifact.schema_hash != record.schema_hash
            or artifact.schema_hash != expected_schema_hash
        ):
            raise EvaluationDatasetError("telemetry_evaluation_catalog_and_manifest_differ")


def _verify_rul_catalog(
    manifest: dict[str, _Artifact], catalog_records: list[DatasetArtifactRecord]
) -> None:
    catalog = {
        str(record.split): record
        for record in catalog_records
        if record.kind == "parquet" and record.split is not None
    }
    expected_schema_hash = "sha256:" + sha256("\0".join(RUL_DATASET_COLUMNS).encode()).hexdigest()
    if set(catalog) != set(manifest):
        raise EvaluationDatasetError("rul_evaluation_catalog_and_manifest_differ")
    for split, artifact in manifest.items():
        record = catalog[split]
        if (
            artifact.object_key != record.object_key
            or artifact.content_hash != record.content_hash
            or artifact.size_bytes != record.size_bytes
            or artifact.row_count != record.row_count
            or artifact.schema_hash != record.schema_hash
            or artifact.schema_hash != expected_schema_hash
        ):
            raise EvaluationDatasetError("rul_evaluation_catalog_and_manifest_differ")


def _case(row: dict[str, Any], governed_media: dict[str, _GovernedMedia]) -> EvaluationCase:
    evidence = _parsed_json(row["redacted_content_json"], "evaluation_evidence_is_invalid")
    labels = _parsed_json(row["labels_json"], "evaluation_labels_are_invalid")
    classification_raw = labels.pop("_classification_contract", None)
    try:
        system_instruction = classification_system_instruction(classification_raw)
    except TrainingDatasetError as exc:
        raise EvaluationDatasetError("classification_contract_is_invalid") from exc
    contract_raw = labels.pop("_evaluation", {})
    retrieval_raw = labels.pop("_retrieval_evaluation", None)
    vlm_raw = labels.pop("_vlm_evaluation", None)
    asr_raw = labels.pop("_asr_evaluation", None)
    tts_raw = labels.pop("_tts_evaluation", None)
    ppo_safety_raw = labels.pop("_ppo_safety_evaluation", None)
    if sum(value is not None for value in (vlm_raw, asr_raw, tts_raw)) > 1:
        raise EvaluationDatasetError("multimodal_evaluation_contract_is_ambiguous")
    if not isinstance(contract_raw, dict):
        raise EvaluationDatasetError("evaluation_case_contract_is_invalid")
    slices_raw = contract_raw.get("slices", {})
    if not isinstance(slices_raw, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in slices_raw.items()
    ):
        raise EvaluationDatasetError("evaluation_case_slices_are_invalid")
    slices = {
        "device_id": str(row["device_id"]),
        "split_origin": str(row["split"]),
        **slices_raw,
    }
    risk = str(contract_raw.get("risk", slices.get("risk", labels.get("severity", "NORMAL"))))
    required_gates = _string_tuple(contract_raw.get("required_gates", []), "required_gates")
    allowed_citations = _string_tuple(
        contract_raw.get("allowed_citations", []), "allowed_citations"
    )
    forbidden = _string_tuple(contract_raw.get("forbidden_substrings", []), "forbidden_substrings")
    contexts = _string_tuple(contract_raw.get("contexts", []), "contexts")
    expected_decision_raw = contract_raw.get("expected_decision")
    if expected_decision_raw is not None and not isinstance(expected_decision_raw, str):
        raise EvaluationDatasetError("expected_decision_is_invalid")
    canonical_evidence = json.dumps(
        evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    vlm_contract, media = _vlm_contract(
        vlm_raw,
        candidate_id=str(row["candidate_id"]),
        split=str(row["split"]),
        governed_media=governed_media,
    )
    asr_contract, asr_media = _asr_contract(
        asr_raw,
        candidate_id=str(row["candidate_id"]),
        split=str(row["split"]),
        governed_media=governed_media,
    )
    media = media or asr_media
    tts_contract = _tts_contract(tts_raw)
    if asr_contract is not None:
        slices["noise_condition"] = asr_contract.noise_condition
        slices["language"] = asr_contract.language
    if tts_contract is not None:
        slices["language"] = tts_contract.language
        slices["voice_profile_id"] = tts_contract.voice_profile_id
        risk = tts_contract.risk
    return EvaluationCase(
        case_id=str(row["candidate_id"]),
        prompt=(
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": "请根据以下脱敏工业工单证据完成根因分类。\n"
                f"evidence={canonical_evidence}",
            },
        ),
        expected_output=labels,
        slices=slices,
        required_gates=required_gates,
        allowed_citations=allowed_citations,
        forbidden_substrings=forbidden,
        expected_decision=expected_decision_raw,
        risk=risk.upper(),
        contexts=contexts,
        retrieval=_retrieval_contract(retrieval_raw),
        vlm=vlm_contract,
        asr=asr_contract,
        tts=tts_contract,
        ppo_safety=_ppo_safety_contract(ppo_safety_raw),
        media_id=media.source_media_id if media is not None else None,
        media_content=media.content if media is not None else None,
        media_mime_type=media.mime_type if media is not None else None,
    )


_PPO_SAFETY_ATTACK_CATEGORIES = frozenset(
    {
        "LENGTH_PADDING",
        "KEYWORD_STUFFING",
        "FABRICATED_CITATION",
        "UNAUTHORIZED_TOOL_CLAIM",
        "UNSAFE_RECOMMENDATION",
    }
)


def _tts_contract(value: Any) -> TtsEvaluationContract | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "review_status",
        "target_text",
        "language",
        "voice_profile_id",
        "required_safety_phrases",
        "industrial_terms",
        "risk",
    }:
        raise EvaluationDatasetError("tts_evaluation_contract_is_invalid")
    target_text = value["target_text"]
    language = value["language"]
    voice_profile_id = value["voice_profile_id"]
    risk = value["risk"]
    phrases_raw = value["required_safety_phrases"]
    terms_raw = value["industrial_terms"]
    if (
        value["review_status"] != "APPROVED"
        or not isinstance(target_text, str)
        or not target_text.strip()
        or len(target_text) > 4096
        or not isinstance(language, str)
        or not language.strip()
        or len(language) > 32
        or not isinstance(voice_profile_id, str)
        or not voice_profile_id.strip()
        or len(voice_profile_id) > 128
        or risk not in {"NORMAL", "HIGH"}
        or not _bounded_unique_text_list(phrases_raw, maximum=32, item_maximum=256)
        or not _bounded_unique_text_list(terms_raw, maximum=128, item_maximum=256)
        or (risk == "HIGH" and not phrases_raw)
    ):
        raise EvaluationDatasetError("tts_evaluation_contract_is_invalid")
    return TtsEvaluationContract(
        target_text=target_text.strip(),
        language=language.strip(),
        voice_profile_id=voice_profile_id.strip(),
        required_safety_phrases=tuple(item.strip() for item in phrases_raw),
        industrial_terms=tuple(item.strip() for item in terms_raw),
        risk=str(risk),
    )


def _bounded_unique_text_list(value: Any, *, maximum: int, item_maximum: int) -> bool:
    if (
        not isinstance(value, list)
        or len(value) > maximum
        or any(
            not isinstance(item, str) or not item.strip() or len(item) > item_maximum
            for item in value
        )
    ):
        return False
    normalized = [item.strip() for item in value]
    return len(normalized) == len(set(normalized))


def _ppo_safety_contract(value: Any) -> PpoSafetyEvaluationContract | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "review_status",
        "preferred_response",
        "adversarial_response",
        "attack_category",
        "forbidden_patterns",
        "minimum_reward_margin",
    }:
        raise EvaluationDatasetError("ppo_safety_evaluation_contract_is_invalid")
    preferred = value["preferred_response"]
    adversarial = value["adversarial_response"]
    category = value["attack_category"]
    patterns_raw = value["forbidden_patterns"]
    margin = value["minimum_reward_margin"]
    if (
        value["review_status"] != "APPROVED"
        or not isinstance(preferred, str)
        or not preferred.strip()
        or len(preferred) > 32_768
        or not isinstance(adversarial, str)
        or not adversarial.strip()
        or len(adversarial) > 32_768
        or category not in _PPO_SAFETY_ATTACK_CATEGORIES
        or not isinstance(patterns_raw, list)
        or not 1 <= len(patterns_raw) <= 100
        or not all(
            isinstance(pattern, str) and pattern.strip() and len(pattern) <= 512
            for pattern in patterns_raw
        )
        or isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(float(margin))
        or not 0 <= float(margin) <= 1000
    ):
        raise EvaluationDatasetError("ppo_safety_evaluation_contract_is_invalid")
    patterns = tuple(pattern.strip() for pattern in patterns_raw)
    if len(set(patterns)) != len(patterns):
        raise EvaluationDatasetError("ppo_safety_evaluation_contract_is_invalid")
    return PpoSafetyEvaluationContract(
        preferred_response=preferred.strip(),
        adversarial_response=adversarial.strip(),
        attack_category=str(category),
        forbidden_patterns=patterns,
        minimum_reward_margin=float(margin),
    )


def _asr_contract(
    value: Any,
    *,
    candidate_id: str,
    split: str,
    governed_media: dict[str, _GovernedMedia],
) -> tuple[AsrEvaluationContract | None, _GovernedMedia | None]:
    if value is None:
        return None, None
    if not isinstance(value, dict) or set(value) != {
        "review_status",
        "modality",
        "media_id",
        "transcript",
        "language",
        "noise_condition",
    }:
        raise EvaluationDatasetError("asr_evaluation_contract_is_invalid")
    media_id = value["media_id"]
    transcript = value["transcript"]
    language = value["language"]
    noise_condition = value["noise_condition"]
    if (
        value["review_status"] != "APPROVED"
        or value["modality"] != "audio"
        or not isinstance(media_id, str)
        or not media_id.strip()
        or not isinstance(transcript, str)
        or not transcript.strip()
        or len(transcript) > 32_768
        or not isinstance(language, str)
        or not language.strip()
        or len(language) > 32
        or noise_condition not in {"CLEAN", "LOW", "MEDIUM", "HIGH"}
    ):
        raise EvaluationDatasetError("asr_evaluation_contract_is_invalid")
    media = governed_media.get(media_id.strip())
    if (
        media is None
        or media.split != split
        or candidate_id not in media.candidate_ids
        or media.mime_type not in {"audio/wav", "audio/flac"}
    ):
        raise EvaluationDatasetError("asr_evaluation_media_binding_is_invalid")
    return (
        AsrEvaluationContract(
            transcript=transcript.strip(),
            language=language.strip(),
            noise_condition=noise_condition,
        ),
        media,
    )


def _retrieval_contract(value: Any) -> RetrievalContract | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"query", "documents", "review_status"}:
        raise EvaluationDatasetError("retrieval_evaluation_contract_is_invalid")
    query = value["query"]
    documents_raw = value["documents"]
    if (
        value["review_status"] != "APPROVED"
        or not isinstance(query, str)
        or not query.strip()
        or len(query) > 4_096
        or not isinstance(documents_raw, list)
        or not 2 <= len(documents_raw) <= 500
    ):
        raise EvaluationDatasetError("retrieval_evaluation_contract_is_invalid")
    documents: list[RetrievalDocument] = []
    document_ids: set[str] = set()
    for item in documents_raw:
        if not isinstance(item, dict) or set(item) != {
            "document_id",
            "text",
            "relevance",
            "access",
        }:
            raise EvaluationDatasetError("retrieval_evaluation_document_is_invalid")
        document_id = item["document_id"]
        text = item["text"]
        relevance = item["relevance"]
        access = item["access"]
        if (
            not isinstance(document_id, str)
            or not document_id
            or len(document_id) > 255
            or document_id in document_ids
            or not isinstance(text, str)
            or not text.strip()
            or len(text) > 32_768
            or isinstance(relevance, bool)
            or not isinstance(relevance, int)
            or not 0 <= relevance <= 3
            or access not in {"ALLOWED", "FORBIDDEN"}
        ):
            raise EvaluationDatasetError("retrieval_evaluation_document_is_invalid")
        document_ids.add(document_id)
        documents.append(
            RetrievalDocument(
                document_id=document_id,
                text=text,
                relevance=relevance,
                access=access,
            )
        )
    contract = RetrievalContract(query=query.strip(), documents=tuple(documents))
    if not contract.relevant_document_ids or not contract.forbidden_document_ids:
        raise EvaluationDatasetError("retrieval_evaluation_coverage_is_incomplete")
    return contract


def _vlm_contract(
    value: Any,
    *,
    candidate_id: str,
    split: str,
    governed_media: dict[str, _GovernedMedia],
) -> tuple[VlmEvaluationContract | None, _GovernedMedia | None]:
    if value is None:
        return None, None
    if not isinstance(value, dict) or set(value) != {
        "review_status",
        "modality",
        "media_id",
        "instruction",
        "expected_findings",
        "forbidden_labels",
    }:
        raise EvaluationDatasetError("vlm_evaluation_contract_is_invalid")
    media_id = value["media_id"]
    instruction = value["instruction"]
    findings_raw = value["expected_findings"]
    forbidden_raw = value["forbidden_labels"]
    if (
        value["review_status"] != "APPROVED"
        or value["modality"] != "image"
        or not isinstance(media_id, str)
        or not media_id.strip()
        or not isinstance(instruction, str)
        or not instruction.strip()
        or len(instruction) > 4_096
        or not isinstance(findings_raw, list)
        or not 1 <= len(findings_raw) <= 100
        or not isinstance(forbidden_raw, list)
        or not all(isinstance(label, str) and label.strip() for label in forbidden_raw)
    ):
        raise EvaluationDatasetError("vlm_evaluation_contract_is_invalid")
    findings: list[VlmExpectedFinding] = []
    identities: set[tuple[str, float, float, float, float]] = set()
    for item in findings_raw:
        if not isinstance(item, dict) or set(item) != {"label", "region"}:
            raise EvaluationDatasetError("vlm_evaluation_finding_is_invalid")
        label = item["label"]
        if not isinstance(label, str) or not label.strip() or len(label) > 128:
            raise EvaluationDatasetError("vlm_evaluation_finding_is_invalid")
        region = _normalized_region(item["region"], "vlm_evaluation_finding_is_invalid")
        identity = (label.strip(), region.x, region.y, region.width, region.height)
        if identity in identities:
            raise EvaluationDatasetError("vlm_evaluation_finding_is_invalid")
        identities.add(identity)
        findings.append(VlmExpectedFinding(label=label.strip(), region=region))
    forbidden = frozenset(label.strip() for label in forbidden_raw)
    if len(forbidden) != len(forbidden_raw) or forbidden & {finding.label for finding in findings}:
        raise EvaluationDatasetError("vlm_evaluation_forbidden_labels_are_invalid")
    media = governed_media.get(media_id.strip())
    if (
        media is None
        or media.split != split
        or candidate_id not in media.candidate_ids
        or media.mime_type not in {"image/png", "image/jpeg"}
    ):
        raise EvaluationDatasetError("vlm_evaluation_media_binding_is_invalid")
    return (
        VlmEvaluationContract(
            instruction=instruction.strip(),
            findings=tuple(findings),
            forbidden_labels=forbidden,
        ),
        media,
    )


def _normalized_region(value: Any, error: str) -> NormalizedRegion:
    if not isinstance(value, dict) or set(value) != {"x", "y", "width", "height"}:
        raise EvaluationDatasetError(error)
    coordinates: list[float] = []
    for key in ("x", "y", "width", "height"):
        coordinate = value[key]
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise EvaluationDatasetError(error)
        coordinates.append(float(coordinate))
    region = NormalizedRegion(*coordinates)
    if (
        not 0 <= region.x < 1
        or not 0 <= region.y < 1
        or not 0 < region.width <= 1
        or not 0 < region.height <= 1
        or region.x + region.width > 1
        or region.y + region.height > 1
    ):
        raise EvaluationDatasetError(error)
    return region


def _manifest_artifacts(manifest: dict[str, Any]) -> dict[str, _Artifact]:
    values = manifest.get("artifacts")
    if not isinstance(values, list):
        raise EvaluationDatasetError("evaluation_manifest_artifacts_are_invalid")
    result: dict[str, _Artifact] = {}
    for value in values:
        if not isinstance(value, dict) or value.get("kind") != "parquet":
            continue
        split = value.get("split")
        if split not in {"train", "validation", "test"} or split in result:
            raise EvaluationDatasetError("evaluation_manifest_splits_are_invalid")
        try:
            result[split] = _Artifact(
                split=split,
                object_key=str(value["object_key"]),
                content_hash=str(value["content_hash"]),
                size_bytes=int(value["size_bytes"]),
                row_count=int(value["row_count"]),
                schema_hash=str(value["schema_hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationDatasetError("evaluation_manifest_artifact_is_invalid") from exc
    if set(result) != {"train", "validation", "test"}:
        raise EvaluationDatasetError("evaluation_manifest_requires_all_splits")
    return result


def _manifest_media_artifacts(manifest: dict[str, Any]) -> dict[str, _MediaArtifact]:
    values = manifest.get("artifacts")
    if not isinstance(values, list):
        raise EvaluationDatasetError("evaluation_manifest_artifacts_are_invalid")
    result: dict[str, _MediaArtifact] = {}
    expected_schema_hash = sha256(b"industrial-governed-media-v1").hexdigest()
    for value in values:
        if not isinstance(value, dict) or value.get("kind") not in {
            "media_image",
            "media_audio",
        }:
            continue
        try:
            source_media_id = str(value["source_media_id"])
            candidate_ids_raw = value["candidate_ids"]
            if not isinstance(candidate_ids_raw, list):
                raise TypeError
            artifact = _MediaArtifact(
                source_media_id=source_media_id,
                split=str(value["split"]),
                object_key=str(value["object_key"]),
                content_hash=str(value["content_hash"]),
                size_bytes=int(value["size_bytes"]),
                schema_hash=str(value["schema_hash"]),
                mime_type=str(value["mime_type"]),
                candidate_ids=tuple(str(item) for item in candidate_ids_raw),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationDatasetError("evaluation_media_artifact_is_invalid") from exc
        if (
            not source_media_id
            or source_media_id in result
            or artifact.split not in {"train", "validation", "test"}
            or artifact.size_bytes <= 0
            or artifact.schema_hash != expected_schema_hash
            or artifact.mime_type not in {"image/png", "image/jpeg", "audio/wav", "audio/flac"}
            or not artifact.candidate_ids
            or len(set(artifact.candidate_ids)) != len(artifact.candidate_ids)
        ):
            raise EvaluationDatasetError("evaluation_media_artifact_is_invalid")
        result[source_media_id] = artifact
    return result


def _verify_catalog(
    manifest: dict[str, _Artifact],
    media_manifest: dict[str, _MediaArtifact],
    catalog_records: list[DatasetArtifactRecord],
) -> None:
    catalog = {
        str(record.split): record
        for record in catalog_records
        if record.kind == "parquet" and record.split is not None
    }
    schema_hash = sha256("\0".join(DATASET_COLUMNS).encode()).hexdigest()
    if set(catalog) != set(manifest):
        raise EvaluationDatasetError("evaluation_catalog_and_manifest_differ")
    for split, artifact in manifest.items():
        record = catalog[split]
        if (
            artifact.object_key != record.object_key
            or artifact.content_hash != record.content_hash
            or artifact.size_bytes != record.size_bytes
            or artifact.row_count != record.row_count
            or artifact.schema_hash != record.schema_hash
            or artifact.schema_hash.removeprefix("sha256:") != schema_hash
        ):
            raise EvaluationDatasetError("evaluation_catalog_and_manifest_differ")
    media_catalog = {
        record.object_key: record
        for record in catalog_records
        if record.kind in {"media_image", "media_audio"}
    }
    if set(media_catalog) != {artifact.object_key for artifact in media_manifest.values()}:
        raise EvaluationDatasetError("evaluation_media_catalog_and_manifest_differ")
    for media_artifact in media_manifest.values():
        record = media_catalog[media_artifact.object_key]
        expected_kind = (
            "media_image" if media_artifact.mime_type.startswith("image/") else "media_audio"
        )
        if (
            record.kind != expected_kind
            or record.split != media_artifact.split
            or record.content_hash != media_artifact.content_hash
            or record.size_bytes != media_artifact.size_bytes
            or record.row_count != len(media_artifact.candidate_ids)
            or record.schema_hash != media_artifact.schema_hash
        ):
            raise EvaluationDatasetError("evaluation_media_catalog_and_manifest_differ")


def _load_governed_media(
    store: DatasetStore, artifacts: dict[str, _MediaArtifact]
) -> dict[str, _GovernedMedia]:
    result: dict[str, _GovernedMedia] = {}
    for media_id, artifact in artifacts.items():
        content = store.get_bytes(artifact.object_key)
        expected_hash = artifact.content_hash.removeprefix("sha256:")
        if len(content) != artifact.size_bytes or _digest(content) != expected_hash:
            raise EvaluationDatasetError("evaluation_media_integrity_failed")
        try:
            inspection = inspect_media(
                content,
                declared_mime=artifact.mime_type,
                max_bytes=artifact.size_bytes,
            )
        except MediaValidationFailure as exc:
            raise EvaluationDatasetError("evaluation_media_signature_is_invalid") from exc
        if inspection.content_hash != artifact.content_hash.removeprefix("sha256:"):
            raise EvaluationDatasetError("evaluation_media_integrity_failed")
        result[media_id] = _GovernedMedia(
            source_media_id=media_id,
            split=artifact.split,
            mime_type=artifact.mime_type,
            content=content,
            candidate_ids=artifact.candidate_ids,
        )
    return result


def _parsed_json(value: Any, error: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise EvaluationDatasetError(error)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise EvaluationDatasetError(error) from exc
    if not isinstance(parsed, dict):
        raise EvaluationDatasetError(error)
    return parsed


def _json_dict(content: bytes, error: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationDatasetError(error) from exc
    if not isinstance(parsed, dict):
        raise EvaluationDatasetError(error)
    return parsed


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise EvaluationDatasetError(f"evaluation_{field}_is_invalid")
    return tuple(dict.fromkeys(value))


def _digest(content: bytes) -> str:
    return sha256(content).hexdigest()
