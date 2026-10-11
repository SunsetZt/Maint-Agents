"""Publish M7 project data as isolated simulation training and evaluation snapshots."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as arrow  # type: ignore[import-untyped]
import pyarrow.parquet as parquet  # type: ignore[import-untyped]
from sqlalchemy import select

from industrial_ops_agent.data_governance.dlp import POLICY_VERSION, PresidioDlpProcessor
from industrial_ops_agent.data_pipeline.contracts import (
    DATASET_COLUMNS,
    DATASET_SCHEMA,
    SIMULATED_MULTIMODAL_CONTRACT,
)
from industrial_ops_agent.data_pipeline.lineage import LineageEmitter, event_now
from industrial_ops_agent.data_pipeline.storage import DatasetStore
from industrial_ops_agent.multimodal.vlm_output import (
    VLM_FINDINGS_PROMPT_CONTRACT,
    build_vlm_findings_instruction,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    CurationRunRecord,
    DataLineageRunRecord,
    DatasetArtifactRecord,
    DatasetSnapshotRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.simulation.multimodal_fault_dataset import (
    CLASSIFICATION,
    MultimodalFaultCase,
    verify_multimodal_fault_dataset,
)

SIMULATION_TASK_TYPE = "SIMULATED_MULTIMODAL_VLM"
SIMULATION_REVIEW_SCHEMA_VERSION = "simulated-label-studio-consensus/v1"


class SimulatedMultimodalSnapshotError(RuntimeError):
    """The source data cannot be safely projected into platform snapshots."""


@dataclass(frozen=True, slots=True)
class PublishedSimulationSnapshot:
    snapshot_id: str
    run_id: str
    purpose: str
    manifest_key: str
    manifest_hash: str
    row_count: int
    split_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class SimulatedMultimodalPublication:
    source_dataset_id: str
    source_manifest_sha256: str
    training: PublishedSimulationSnapshot
    evaluation: PublishedSimulationSnapshot
    reviewer_subject_id: str
    lineage_event_count: int


@dataclass(slots=True)
class _Artifact:
    kind: str
    split: str | None
    object_key: str
    content: bytes
    row_count: int
    schema_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return _digest(self.content)

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "split": self.split,
            "object_key": self.object_key,
            "content_hash": self.content_hash,
            "size_bytes": len(self.content),
            "row_count": self.row_count,
            "schema_hash": self.schema_hash,
            **self.metadata,
        }


class SimulatedMultimodalSnapshotService:
    """Use the real snapshot catalog while preserving a non-production contract."""

    def __init__(
        self,
        database: Database,
        store: DatasetStore,
        lineage: LineageEmitter,
        *,
        code_version: str,
    ) -> None:
        self._database = database
        self._store = store
        self._lineage = lineage
        self._code_version = code_version

    def publish(
        self,
        context: TenantContext,
        *,
        source_dir: Path,
        generated_by_subject_id: str = "simulation-data-generator",
        reviewed_by_subject_id: str,
        occurred_at: datetime | None = None,
    ) -> SimulatedMultimodalPublication:
        generated_by = generated_by_subject_id.strip()
        reviewer = reviewed_by_subject_id.strip()
        if not generated_by or not reviewer or generated_by == reviewer:
            raise SimulatedMultimodalSnapshotError(
                "simulation_generation_and_review_separation_required"
            )
        source = source_dir.resolve(strict=True)
        source_result = verify_multimodal_fault_dataset(source)
        source_manifest = _object(
            (source / "manifest.json").read_bytes(),
            "source_multimodal_manifest_is_invalid",
        )
        source_freeze = _object(
            (source / "evaluation-freeze.json").read_bytes(),
            "source_multimodal_freeze_is_invalid",
        )
        cases = _load_source_cases(source)
        _require_source_split_isolation(cases)
        controlled_labels = _controlled_visual_labels(cases)
        now = (occurred_at or datetime.now(UTC)).astimezone(UTC)
        dlp_processor = PresidioDlpProcessor()
        training = self._publish_snapshot(
            context,
            source=source,
            source_manifest=source_manifest,
            source_freeze=source_freeze,
            source_cases=[*cases["train"], *cases["validation"]],
            purpose="LOCAL_SIMULATION_VLM_TRAINING",
            generated_by=generated_by,
            reviewer=reviewer,
            dlp_processor=dlp_processor,
            now=now,
            controlled_labels=controlled_labels,
        )
        evaluation = self._publish_snapshot(
            context,
            source=source,
            source_manifest=source_manifest,
            source_freeze=source_freeze,
            source_cases=cases["simulation_evaluation"],
            purpose="SIMULATION_REFERENCE_EVALUATION",
            generated_by=generated_by,
            reviewer=reviewer,
            dlp_processor=dlp_processor,
            now=now + timedelta(seconds=1),
            controlled_labels=controlled_labels,
        )
        event_count = len(getattr(self._lineage, "events", ()))
        return SimulatedMultimodalPublication(
            source_dataset_id=source_result.dataset_id,
            source_manifest_sha256=source_result.manifest_sha256,
            training=training,
            evaluation=evaluation,
            reviewer_subject_id=reviewer,
            lineage_event_count=event_count,
        )

    def _publish_snapshot(
        self,
        context: TenantContext,
        *,
        source: Path,
        source_manifest: dict[str, Any],
        source_freeze: dict[str, Any],
        source_cases: list[MultimodalFaultCase],
        purpose: Literal[
            "LOCAL_SIMULATION_VLM_TRAINING",
            "SIMULATION_REFERENCE_EVALUATION",
        ],
        generated_by: str,
        reviewer: str,
        dlp_processor: PresidioDlpProcessor,
        now: datetime,
        controlled_labels: tuple[str, ...],
    ) -> PublishedSimulationSnapshot:
        identity_digest = sha256(
            (f"{context.tenant_id}\0{source_manifest['manifest_sha256']}\0{purpose}").encode()
        ).hexdigest()[:24]
        scope = "train" if purpose == "LOCAL_SIMULATION_VLM_TRAINING" else "eval"
        run_id = f"run-sim-m7-{scope}-{identity_digest}"
        snapshot_id = f"dataset-sim-m7-{scope}-{identity_digest}"
        existing = self._existing_snapshot(context, snapshot_id)
        if existing is not None:
            return existing

        rows_by_split: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        projected_cases: list[tuple[MultimodalFaultCase, str, dict[str, Any]]] = []
        for index, case in enumerate(source_cases):
            split = case.split if purpose == "LOCAL_SIMULATION_VLM_TRAINING" else "test"
            if split not in rows_by_split:
                raise SimulatedMultimodalSnapshotError("simulation_source_split_is_not_publishable")
            row = _project_case_row(
                context,
                case,
                split=split,
                purpose=purpose,
                source_dataset_id=str(source_manifest["dataset_id"]),
                source_manifest_sha256=str(source_manifest["manifest_sha256"]),
                reviewer=reviewer,
                dlp_processor=dlp_processor,
                occurred_at=now + timedelta(milliseconds=index),
                controlled_labels=controlled_labels,
            )
            rows_by_split[split].append(row)
            projected_cases.append((case, split, row))
        for projected_split, rows in rows_by_split.items():
            rows.sort(key=lambda item: str(item["candidate_id"]))
            frame = pd.DataFrame(rows, columns=DATASET_COLUMNS)
            DATASET_SCHEMA.validate(frame, lazy=True)
            if rows and {str(row["split"]) for row in rows} != {projected_split}:
                raise SimulatedMultimodalSnapshotError("simulation_projected_split_is_invalid")
        _require_projected_group_isolation(rows_by_split)

        prefix = f"tenant/{context.tenant_id}/datasets/snapshots/{snapshot_id}/published"
        schema_hash = _digest("\0".join(DATASET_COLUMNS).encode())
        artifacts = [
            _Artifact(
                kind="parquet",
                split=split,
                object_key=f"{prefix}/{split}.parquet",
                content=_parquet_bytes(rows),
                row_count=len(rows),
                schema_hash=schema_hash,
            )
            for split, rows in rows_by_split.items()
        ]
        artifacts.extend(
            _project_case_artifacts(
                source,
                projected_cases,
                prefix=prefix,
                source_manifest_sha256=str(source_manifest["manifest_sha256"]),
            )
        )
        review = _review_document(
            projected_cases,
            source_manifest_sha256=str(source_manifest["manifest_sha256"]),
            purpose=purpose,
            generated_by=generated_by,
            reviewer=reviewer,
            occurred_at=now,
        )
        review_bytes = _json_bytes(review)
        review_key = f"{prefix}/governance/simulation-review.json"
        artifacts.append(
            _Artifact(
                kind="simulation_review",
                split=None,
                object_key=review_key,
                content=review_bytes,
                row_count=len(projected_cases),
                schema_hash=_digest(SIMULATION_REVIEW_SCHEMA_VERSION.encode()),
            )
        )
        source_manifest_bytes = _json_bytes(source_manifest)
        source_manifest_key = f"{prefix}/governance/source-manifest.json"
        artifacts.append(
            _Artifact(
                kind="simulation_provenance",
                split=None,
                object_key=source_manifest_key,
                content=source_manifest_bytes,
                row_count=len(projected_cases),
                schema_hash=_digest(str(source_manifest["schema_version"]).encode()),
            )
        )
        source_freeze_bytes = _json_bytes(source_freeze)
        source_freeze_key = f"{prefix}/governance/source-evaluation-freeze.json"
        artifacts.append(
            _Artifact(
                kind="simulation_provenance",
                split=None,
                object_key=source_freeze_key,
                content=source_freeze_bytes,
                row_count=len(source_freeze.get("case_ids", ())),
                schema_hash=_digest(str(source_freeze["schema_version"]).encode()),
            )
        )
        feedback_bytes = b"".join(
            _json_line(_feedback_candidate_projection(case, row, purpose=purpose))
            for case, _, row in projected_cases
        )
        artifacts.append(
            _Artifact(
                kind="simulation_feedback",
                split=None,
                object_key=f"{prefix}/governance/feedback-candidates.jsonl",
                content=feedback_bytes,
                row_count=len(projected_cases),
                schema_hash=_digest(b"simulated-feedback-candidate-projection/v1"),
            )
        )
        label_studio_bytes = _json_bytes(
            _label_studio_projection(projected_cases, reviewer=reviewer, purpose=purpose)
        )
        artifacts.append(
            _Artifact(
                kind="simulation_labeling",
                split=None,
                object_key=f"{prefix}/governance/label-studio-consensus.json",
                content=label_studio_bytes,
                row_count=len(projected_cases),
                schema_hash=_digest(b"simulated-label-studio-consensus/v1"),
            )
        )

        split_counts = {split: len(rows) for split, rows in rows_by_split.items()}
        row_count = sum(split_counts.values())
        quality = {
            "contract_version": SIMULATED_MULTIMODAL_CONTRACT,
            "pandera_validation": "PASSED",
            "row_count": row_count,
            "split_counts": split_counts,
            "group_leakage_count": 0,
            "exclusions": [],
            "classification": CLASSIFICATION,
            "production_gold_eligible": False,
        }
        quality_bytes = _json_bytes(quality)
        quality_key = f"{prefix}/quality-report.json"
        artifacts.append(
            _Artifact(
                kind="quality_report",
                split=None,
                object_key=quality_key,
                content=quality_bytes,
                row_count=row_count,
                schema_hash=_digest(SIMULATED_MULTIMODAL_CONTRACT.encode()),
            )
        )
        profile = {
            "purpose": purpose,
            "task_type": SIMULATION_TASK_TYPE,
            "training_output_contract": (
                "industrial-vlm-findings-v1" if purpose == "LOCAL_SIMULATION_VLM_TRAINING" else None
            ),
            "prompt_contract_version": VLM_FINDINGS_PROMPT_CONTRACT,
            "source_dataset_id": source_manifest["dataset_id"],
            "source_manifest_key": source_manifest_key,
            "manifest_sha256": source_manifest["manifest_sha256"],
            "source_evaluation_freeze_key": source_freeze_key,
            "freeze_sha256": source_freeze["freeze_sha256"],
            "review_evidence_key": review_key,
            "review_sha256": review["review_sha256"],
            "generated_by_subject_id": generated_by,
            "reviewed_by_subject_id": reviewer,
            "production_gold_eligible": False,
            "expert_review_required_for_production": True,
        }
        manifest: dict[str, Any] = {
            "snapshot_id": snapshot_id,
            "run_id": run_id,
            "tenant_id": context.tenant_id,
            "contract_version": SIMULATED_MULTIMODAL_CONTRACT,
            "code_version": self._code_version,
            "classification": CLASSIFICATION,
            "production_claim": False,
            "production_release_eligible": False,
            "input_manifest_hash": source_manifest["manifest_sha256"],
            "row_count": row_count,
            "split_counts": split_counts,
            "lineage_status": "CONFIRMED",
            "evaluation_eligible": purpose == "SIMULATION_REFERENCE_EVALUATION",
            "sample_origin_counts": {"real": 0, "synthetic": row_count},
            "synthetic_split_counts": split_counts,
            "simulation_profile": profile,
            "source_work_order_ids": [
                str(row["work_order_id"]) for rows in rows_by_split.values() for row in rows
            ],
            "artifacts": [artifact.manifest_entry() for artifact in artifacts],
        }
        manifest_bytes = _json_bytes(manifest)
        manifest_key = f"{prefix}/manifest.json"
        output_dataset = manifest_key
        self._lineage.emit(
            event_now(
                "START",
                run_id=run_id,
                snapshot_id=snapshot_id,
                tenant_id=context.tenant_id,
                source_work_order_ids=(),
                source_dataset_names=(str(source_manifest["dataset_id"]),),
                output_dataset=output_dataset,
                input_manifest_hash=str(source_manifest["manifest_sha256"]),
            )
        )
        for artifact in artifacts:
            self._store.put_bytes(artifact.object_key, artifact.content)
        self._store.put_bytes(manifest_key, manifest_bytes)
        self._lineage.emit(
            event_now(
                "COMPLETE",
                run_id=run_id,
                snapshot_id=snapshot_id,
                tenant_id=context.tenant_id,
                source_work_order_ids=(),
                source_dataset_names=(str(source_manifest["dataset_id"]),),
                output_dataset=output_dataset,
                input_manifest_hash=str(source_manifest["manifest_sha256"]),
            )
        )
        self._record_snapshot(
            context,
            run_id=run_id,
            snapshot_id=snapshot_id,
            manifest_key=manifest_key,
            manifest_hash=_digest(manifest_bytes),
            quality_key=quality_key,
            artifacts=artifacts,
            split_counts=split_counts,
            source_work_order_ids=tuple(manifest["source_work_order_ids"]),
            output_dataset=output_dataset,
            input_manifest_hash=str(source_manifest["manifest_sha256"]),
            now=now,
        )
        return PublishedSimulationSnapshot(
            snapshot_id=snapshot_id,
            run_id=run_id,
            purpose=purpose,
            manifest_key=manifest_key,
            manifest_hash=_digest(manifest_bytes),
            row_count=row_count,
            split_counts=split_counts,
        )

    def _existing_snapshot(
        self,
        context: TenantContext,
        snapshot_id: str,
    ) -> PublishedSimulationSnapshot | None:
        with self._database.transaction(context) as session:
            snapshot = session.scalar(
                select(DatasetSnapshotRecord).where(
                    DatasetSnapshotRecord.tenant_id == context.tenant_id,
                    DatasetSnapshotRecord.snapshot_id == snapshot_id,
                )
            )
            if snapshot is None:
                return None
            if (
                snapshot.contract_version != SIMULATED_MULTIMODAL_CONTRACT
                or snapshot.status != "CANDIDATE"
                or snapshot.lineage_status != "CONFIRMED"
                or snapshot.manifest_key is None
                or snapshot.manifest_hash is None
            ):
                raise SimulatedMultimodalSnapshotError(
                    "existing_simulation_snapshot_is_not_reusable"
                )
            manifest = _object(
                self._store.get_bytes(snapshot.manifest_key),
                "existing_simulation_manifest_is_invalid",
            )
            if _digest(self._store.get_bytes(snapshot.manifest_key)) != snapshot.manifest_hash:
                raise SimulatedMultimodalSnapshotError(
                    "existing_simulation_manifest_integrity_failed"
                )
            profile = manifest.get("simulation_profile")
            if not isinstance(profile, dict) or not isinstance(profile.get("purpose"), str):
                raise SimulatedMultimodalSnapshotError("existing_simulation_manifest_is_invalid")
            return PublishedSimulationSnapshot(
                snapshot_id=snapshot.snapshot_id,
                run_id=snapshot.run_id,
                purpose=str(profile["purpose"]),
                manifest_key=snapshot.manifest_key,
                manifest_hash=snapshot.manifest_hash,
                row_count=snapshot.row_count,
                split_counts=dict(snapshot.split_counts),
            )

    def _record_snapshot(
        self,
        context: TenantContext,
        *,
        run_id: str,
        snapshot_id: str,
        manifest_key: str,
        manifest_hash: str,
        quality_key: str,
        artifacts: list[_Artifact],
        split_counts: dict[str, int],
        source_work_order_ids: tuple[str, ...],
        output_dataset: str,
        input_manifest_hash: str,
        now: datetime,
    ) -> None:
        row_count = sum(split_counts.values())
        with self._database.transaction(context) as session:
            session.add(
                CurationRunRecord(
                    run_id=run_id,
                    tenant_id=context.tenant_id,
                    window_start=now,
                    window_end=now + timedelta(seconds=1),
                    engine="simulation-local",
                    contract_version=SIMULATED_MULTIMODAL_CONTRACT,
                    code_version=self._code_version,
                    config_hash=_digest(f"{self._code_version}\0{input_manifest_hash}".encode()),
                    input_manifest_hash=input_manifest_hash,
                    input_count=row_count,
                    exclusion_report=[],
                    status="COMPLETED",
                    failure_reason=None,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            session.add(
                DatasetSnapshotRecord(
                    snapshot_id=snapshot_id,
                    tenant_id=context.tenant_id,
                    run_id=run_id,
                    status="CANDIDATE",
                    contract_version=SIMULATED_MULTIMODAL_CONTRACT,
                    input_manifest_hash=input_manifest_hash,
                    manifest_key=manifest_key,
                    manifest_hash=manifest_hash,
                    quality_report_key=quality_key,
                    row_count=row_count,
                    split_counts=split_counts,
                    source_work_order_ids=list(source_work_order_ids),
                    lineage_status="CONFIRMED",
                    training_eligible=True,
                    sample_origin_counts={"real": 0, "synthetic": row_count},
                    synthetic_split_counts=split_counts,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            for artifact in artifacts:
                session.add(
                    DatasetArtifactRecord(
                        artifact_id="artifact-"
                        + sha256(f"{snapshot_id}\0{artifact.object_key}".encode()).hexdigest()[:32],
                        tenant_id=context.tenant_id,
                        snapshot_id=snapshot_id,
                        kind=artifact.kind,
                        split=artifact.split,
                        object_key=artifact.object_key,
                        content_hash=artifact.content_hash,
                        size_bytes=len(artifact.content),
                        row_count=artifact.row_count,
                        schema_hash=artifact.schema_hash,
                        created_at=now,
                        updated_at=now,
                    )
                )
            session.add(
                DataLineageRunRecord(
                    lineage_id=f"lineage-{run_id}",
                    tenant_id=context.tenant_id,
                    run_id=run_id,
                    snapshot_id=snapshot_id,
                    openlineage_run_id=run_id,
                    job_namespace=self._lineage.namespace,
                    job_name=self._lineage.job_name,
                    status="CONFIRMED",
                    source_work_order_ids=list(source_work_order_ids),
                    output_dataset=output_dataset,
                    failure_reason=None,
                    created_at=now,
                    updated_at=now,
                )
            )


def _load_source_cases(root: Path) -> dict[str, list[MultimodalFaultCase]]:
    result: dict[str, list[MultimodalFaultCase]] = {}
    for split in ("train", "validation", "simulation_evaluation"):
        cases: list[MultimodalFaultCase] = []
        try:
            for line in (root / "cases" / f"{split}.jsonl").read_text().splitlines():
                if line.strip():
                    cases.append(MultimodalFaultCase.model_validate_json(line))
        except (OSError, ValueError) as exc:
            raise SimulatedMultimodalSnapshotError("source_multimodal_cases_are_invalid") from exc
        if not cases:
            raise SimulatedMultimodalSnapshotError("source_multimodal_split_is_empty")
        result[split] = cases
    return result


def _require_source_split_isolation(
    cases: dict[str, list[MultimodalFaultCase]],
) -> None:
    case_ids: dict[str, set[str]] = {
        split: {case.case_id for case in values} for split, values in cases.items()
    }
    asset_ids: dict[str, set[str]] = {
        split: {case.asset_id for case in values} for split, values in cases.items()
    }
    splits = tuple(cases)
    if any(
        case_ids[left] & case_ids[right] or asset_ids[left] & asset_ids[right]
        for index, left in enumerate(splits)
        for right in splits[index + 1 :]
    ):
        raise SimulatedMultimodalSnapshotError("source_multimodal_split_leakage_detected")


def _project_case_row(
    context: TenantContext,
    case: MultimodalFaultCase,
    *,
    split: str,
    purpose: str,
    source_dataset_id: str,
    source_manifest_sha256: str,
    reviewer: str,
    dlp_processor: PresidioDlpProcessor,
    occurred_at: datetime,
    controlled_labels: tuple[str, ...],
) -> dict[str, Any]:
    candidate_id = f"sim-feedback-{case.case_id}"
    media_id = f"sim-media-{case.case_id}"
    dlp = dlp_processor.process({"engineer_note": case.engineer_note})
    if dlp.status != "PASSED" or dlp.residual_entity_types:
        raise SimulatedMultimodalSnapshotError("simulation_case_dlp_failed")
    evidence = {
        "classification": CLASSIFICATION,
        "engineer_note": dlp.redacted_content["engineer_note"],
        "asset_model": case.asset_model,
        "site_id": case.site_id,
        "acoustic": case.acoustic.model_dump(mode="json"),
        "telemetry": case.telemetry.model_dump(mode="json"),
        "knowledge_citation": case.target["allowed_citations"][0],
    }
    labels: dict[str, Any] = {
        "fault_code": case.fault_code,
        "root_cause": case.target["root_cause"],
        "severity": case.severity,
        "requires_human_approval": True,
    }
    instruction = build_vlm_findings_instruction(
        (
            "识别图中受控故障区域，并结合脱敏工单证据输出可见故障标签。"
            if purpose == "LOCAL_SIMULATION_VLM_TRAINING"
            else "定位图中异常区域并输出受控可见故障标签。"
        ),
        allowed_labels=controlled_labels,
    )
    if purpose == "LOCAL_SIMULATION_VLM_TRAINING":
        labels["_multimodal_training"] = {
            "review_status": "APPROVED",
            "modality": "image",
            "media_id": media_id,
            "instruction": instruction,
            "answer": json.dumps(
                {
                    "findings": case.target["expected_findings"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "synthetic_media": {
                "schema_version": "simulated-multimodal-fault-dataset/v1",
                "manifest_sha256": source_manifest_sha256,
                "source_type": "PROJECT_GENERATED",
                "reviewer_subject_id": reviewer,
                "production_eligible": False,
            },
        }
    else:
        labels["_evaluation"] = {
            "slices": {
                "fault_code": case.fault_code,
                "severity": case.severity,
                "classification": CLASSIFICATION,
            },
            "required_gates": [
                "media_binding",
                "fault_classification",
                "human_approval_boundary",
            ],
            "allowed_citations": case.target["allowed_citations"],
            "forbidden_substrings": ["production approved", "automatic device control"],
            "expected_decision": "DIAGNOSIS_CANDIDATE",
            "risk": case.severity,
            "contexts": [case.engineer_note],
        }
        labels["_vlm_evaluation"] = {
            "review_status": "APPROVED",
            "modality": "image",
            "media_id": media_id,
            "instruction": instruction,
            "expected_findings": case.target["expected_findings"],
            "forbidden_labels": case.target["forbidden_labels"],
        }
    group_key = "group:" + sha256(f"{context.tenant_id}\0{case.asset_id}".encode()).hexdigest()
    return {
        "tenant_id": context.tenant_id,
        "candidate_id": candidate_id,
        "candidate_version": 1,
        "work_order_id": f"sim-work-{case.case_id}",
        "incident_id": f"sim-incident-{case.case_id}",
        "device_id": case.asset_id,
        "source_event_id": f"sim-source-{case.case_id}",
        "source_content_hash": case.image.sha256,
        "redacted_content_json": _json_text(evidence),
        "labels_json": _json_text(labels),
        "lineage_json": _json_text(
            {
                "classification": CLASSIFICATION,
                "source_dataset_id": source_dataset_id,
                "source_case_id": case.case_id,
                "reviewer_subject_id": reviewer,
                "dlp_policy_version": POLICY_VERSION,
                "dlp_findings_count": len(dlp.findings),
                "production_claim": False,
            }
        ),
        "occurred_at": occurred_at.isoformat(),
        "near_duplicate_group": f"sim-asset:{case.asset_id}",
        "group_key": group_key,
        "split": split,
    }


def _controlled_visual_labels(
    cases: dict[str, list[MultimodalFaultCase]],
) -> tuple[str, ...]:
    labels: set[str] = set()
    for split_cases in cases.values():
        for case in split_cases:
            findings = case.target.get("expected_findings")
            if not isinstance(findings, list):
                raise SimulatedMultimodalSnapshotError("source_visual_findings_are_invalid")
            for finding in findings:
                if not isinstance(finding, dict) or not isinstance(finding.get("label"), str):
                    raise SimulatedMultimodalSnapshotError("source_visual_findings_are_invalid")
                labels.add(str(finding["label"]).strip())
    if not labels:
        raise SimulatedMultimodalSnapshotError("source_visual_label_vocabulary_is_empty")
    return tuple(sorted(labels))


def _project_case_artifacts(
    source: Path,
    projected_cases: list[tuple[MultimodalFaultCase, str, dict[str, Any]]],
    *,
    prefix: str,
    source_manifest_sha256: str,
) -> list[_Artifact]:
    artifacts: list[_Artifact] = []
    copied_knowledge: dict[str, _Artifact] = {}
    media_schema_hash = _digest(b"industrial-governed-media-v1")
    for case, split, row in projected_cases:
        media_id = f"sim-media-{case.case_id}"
        image = (source / case.image.path).read_bytes()
        artifacts.append(
            _Artifact(
                kind="media_image",
                split=split,
                object_key=f"{prefix}/media/{split}/{case.case_id}.png",
                content=image,
                row_count=1,
                schema_hash=media_schema_hash,
                metadata={
                    "source_media_id": media_id,
                    "candidate_ids": [row["candidate_id"]],
                    "mime_type": "image/png",
                    "synthetic": True,
                    "synthetic_manifest_sha256": source_manifest_sha256,
                    "evidence_eligible": False,
                    "golden_dataset_eligible": False,
                },
            )
        )
        artifacts.append(
            _Artifact(
                kind="simulation_acoustic",
                split=split,
                object_key=f"{prefix}/acoustic/{split}/{case.case_id}.wav",
                content=(source / case.acoustic.path).read_bytes(),
                row_count=1,
                schema_hash=_digest(b"simulated-machine-acoustic/v1"),
            )
        )
        artifacts.append(
            _Artifact(
                kind="simulation_telemetry",
                split=split,
                object_key=f"{prefix}/telemetry/{split}/{case.case_id}.json",
                content=(source / case.telemetry.path).read_bytes(),
                row_count=1,
                schema_hash=_digest(b"simulated-industrial-telemetry/v1"),
            )
        )
        if case.knowledge.path not in copied_knowledge:
            copied_knowledge[case.knowledge.path] = _Artifact(
                kind="simulation_knowledge",
                split=None,
                object_key=f"{prefix}/knowledge/{Path(case.knowledge.path).name}",
                content=(source / case.knowledge.path).read_bytes(),
                row_count=1,
                schema_hash=_digest(b"simulated-maintenance-knowledge/v1"),
            )
    artifacts.extend(copied_knowledge.values())
    return artifacts


def _review_document(
    cases: list[tuple[MultimodalFaultCase, str, dict[str, Any]]],
    *,
    source_manifest_sha256: str,
    purpose: str,
    generated_by: str,
    reviewer: str,
    occurred_at: datetime,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": SIMULATION_REVIEW_SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "status": "APPROVED_FOR_LOCAL_SIMULATION",
        "production_approval": False,
        "source_manifest_sha256": source_manifest_sha256,
        "purpose": purpose,
        "generated_by_subject_id": generated_by,
        "reviewed_by_subject_id": reviewer,
        "reviewed_at": occurred_at.isoformat(),
        "case_ids": sorted(case.case_id for case, _, _ in cases),
        "checks": {
            "schema": "PASSED",
            "media_binding": "PASSED",
            "split_isolation": "PASSED",
            "pii": "NO_PERSONAL_DATA_PROJECT_GENERATED",
            "production_boundary": "PASSED",
        },
    }
    document["review_sha256"] = _document_hash(document, "review_sha256")
    return document


def _feedback_candidate_projection(
    case: MultimodalFaultCase,
    row: dict[str, Any],
    *,
    purpose: str,
) -> dict[str, Any]:
    return {
        "schema_version": "simulated-feedback-candidate-projection/v1",
        "classification": CLASSIFICATION,
        "candidate_id": row["candidate_id"],
        "source_event_id": row["source_event_id"],
        "work_order_id": row["work_order_id"],
        "incident_id": row["incident_id"],
        "source_content_hash": row["source_content_hash"],
        "status": "READY_FOR_SIMULATION_CURATION",
        "eligibility_status": "SIMULATION_ONLY",
        "allow_training": purpose == "LOCAL_SIMULATION_VLM_TRAINING",
        "license_status": "PROJECT_GENERATED",
        "source_case_id": case.case_id,
        "production_eligible": False,
    }


def _label_studio_projection(
    cases: list[tuple[MultimodalFaultCase, str, dict[str, Any]]],
    *,
    reviewer: str,
    purpose: str,
) -> dict[str, Any]:
    return {
        "schema_version": "simulated-label-studio-consensus/v1",
        "classification": CLASSIFICATION,
        "project_id": "m7-simulated-multimodal-faults",
        "purpose": purpose,
        "external_service_called": False,
        "production_approval": False,
        "tasks": [
            {
                "external_task_id": f"sim-ls-{case.case_id}",
                "candidate_id": row["candidate_id"],
                "source_case_id": case.case_id,
                "reviewer_subject_id": reviewer,
                "review_status": "APPROVED_FOR_LOCAL_SIMULATION",
                "labels": {
                    "fault_code": case.fault_code,
                    "root_cause": case.target["root_cause"],
                    "expected_findings": case.target["expected_findings"],
                },
            }
            for case, _, row in cases
        ],
    }


def _require_projected_group_isolation(
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> None:
    groups = {
        split: {str(row["group_key"]) for row in rows} for split, rows in rows_by_split.items()
    }
    splits = tuple(groups)
    if any(
        groups[left] & groups[right]
        for index, left in enumerate(splits)
        for right in splits[index + 1 :]
    ):
        raise SimulatedMultimodalSnapshotError("simulation_projected_group_leakage_detected")


def _parquet_bytes(rows: list[dict[str, Any]]) -> bytes:
    frame = pd.DataFrame(rows, columns=DATASET_COLUMNS)
    table = arrow.Table.from_pandas(frame, preserve_index=False)
    output = BytesIO()
    parquet.write_table(table, output, compression="zstd")
    return output.getvalue()


def _object(payload: bytes, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SimulatedMultimodalSnapshotError(reason) from exc
    if not isinstance(value, dict):
        raise SimulatedMultimodalSnapshotError(reason)
    return value


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"


def _json_line(value: Any) -> bytes:
    return _json_text(value).encode() + b"\n"


def _digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _document_hash(value: dict[str, Any], field: str) -> str:
    document = dict(value)
    document.pop(field, None)
    return _digest(_json_text(document).encode())
