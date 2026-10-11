"""Load governed, checksum-verified SFT and post-training records."""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pyarrow.parquet as parquet  # type: ignore[import-untyped]
from sqlalchemy import select

from industrial_ops_agent.data_pipeline.contracts import (
    CONTRACT_VERSION,
    DATASET_COLUMNS,
    DATASET_SCHEMA,
    SIMULATED_MULTIMODAL_CONTRACT,
    SYNTHETIC_AUGMENTATION_CONTRACT,
)
from industrial_ops_agent.data_pipeline.quality import (
    DatasetQualityReportError,
    verify_dataset_quality_report,
)
from industrial_ops_agent.data_pipeline.storage import DatasetStore
from industrial_ops_agent.data_pipeline.training_gate import dataset_training_blockers
from industrial_ops_agent.media.validation import MediaValidationFailure, inspect_media
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    DataLineageRunRecord,
    DatasetArtifactRecord,
    DatasetSnapshotRecord,
    TrainingExperimentRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.training.reward_contracts import (
    LEGACY_REWARD_CONTRACT_VERSION,
    RewardContractError,
    reward_contract,
)
from industrial_ops_agent.training.synthetic_media import (
    SyntheticMediaError,
    verify_synthetic_media_bundle,
)

SYSTEM_INSTRUCTION = (
    "你是工业设备故障根因分类助手。仅依据已经脱敏的工单证据输出受控标签 JSON；"
    "不得补写个人信息、不得编造证据、不得提出或执行设备控制操作。"
    "若证据包含 citation_id，只能将该值原样放入 citations 数组，不得编造引用。"
)

_EVALUATION_CONTROL_FIELDS = (
    "_evaluation",
    "_retrieval_evaluation",
    "_vlm_evaluation",
    "_asr_evaluation",
    "_tts_evaluation",
    "_ppo_safety_evaluation",
)


class TrainingDatasetError(ValueError):
    """A governed snapshot failed integrity, isolation or leakage validation."""


def classification_system_instruction(contract: Any) -> str:
    """Build a closed classification prompt from governed, non-Gold label metadata."""

    if contract is None:
        return SYSTEM_INSTRUCTION
    if not isinstance(contract, dict) or set(contract) != {
        "schema_version",
        "output_key",
        "labels",
    }:
        raise TrainingDatasetError("classification_contract_is_invalid")
    if (
        contract["schema_version"] != "industrial-root-cause-classification/v1"
        or contract["output_key"] != "root_cause_code"
    ):
        raise TrainingDatasetError("classification_contract_is_invalid")
    labels = contract["labels"]
    if not isinstance(labels, dict) or not 2 <= len(labels) <= 64:
        raise TrainingDatasetError("classification_contract_labels_are_invalid")
    normalized: dict[str, str] = {}
    for code, description in labels.items():
        if (
            not isinstance(code, str)
            or re.fullmatch(r"[A-Z][A-Z0-9_]{2,63}", code) is None
            or not isinstance(description, str)
            or description != description.strip()
            or not 1 <= len(description) <= 128
            or "\n" in description
            or "\r" in description
        ):
            raise TrainingDatasetError("classification_contract_labels_are_invalid")
        normalized[code] = description
    taxonomy = "；".join(
        f"{code}={normalized[code]}" for code in sorted(normalized)
    )
    return (
        f"{SYSTEM_INSTRUCTION}"
        "输出 JSON 必须包含 root_cause_code 字段，且其值只能从以下训练域批准标签中选择："
        f"{taxonomy}。除证据要求的 citations 数组外，不得输出其他字段。"
    )


@dataclass(frozen=True, slots=True)
class SftSample:
    prompt: list[dict[str, str]]
    completion: list[dict[str, str]]
    candidate_id: str
    preference_chosen: list[dict[str, str]] | None = None
    preference_rejected: list[dict[str, str]] | None = None
    reward_target: str | None = None
    reward_contract_source_version: str | None = None
    retrieval_query: str | None = None
    retrieval_positive: str | None = None
    retrieval_hard_negatives: tuple[str, ...] = ()
    media_artifact_id: str | None = None
    media_content: bytes | None = None
    media_mime_type: str | None = None
    multimodal_instruction: str | None = None
    multimodal_answer: str | None = None
    transcript: str | None = None
    language: str | None = None
    tts_target_text: str | None = None
    voice_profile_id: str | None = None
    speaker_embedding: tuple[float, ...] = ()
    voice_source_type: str | None = None
    voice_usage_evidence_sha256: str | None = None
    synthetic: bool = False
    synthetic_manifest_sha256: str | None = None

    def trainer_records(self, method: str) -> list[dict[str, Any]]:
        if method in {"LORA", "QLORA"}:
            return [{"prompt": self.prompt, "completion": self.completion}]
        if method == "DPO":
            if self.preference_chosen is None or self.preference_rejected is None:
                raise TrainingDatasetError("dpo_preference_pair_is_missing")
            return [
                {
                    "prompt": self.prompt,
                    "chosen": self.preference_chosen,
                    "rejected": self.preference_rejected,
                }
            ]
        if method == "GRPO":
            if self.reward_target is None or self.reward_contract_source_version is None:
                raise TrainingDatasetError("grpo_reward_target_is_missing")
            return [
                {
                    "prompt": self.prompt,
                    "ground_truth": self.reward_target,
                    "reward_contract_source_version": self.reward_contract_source_version,
                    "candidate_id": self.candidate_id,
                }
            ]
        if method == "PPO":
            return [{"prompt": self.prompt, "candidate_id": self.candidate_id}]
        if method == "EMBEDDING":
            self._require_retrieval_contract()
            record = {
                "anchor": self.retrieval_query,
                "positive": self.retrieval_positive,
            }
            record.update(
                {
                    f"negative_{index}": value
                    for index, value in enumerate(self.retrieval_hard_negatives, start=1)
                }
            )
            return [record]
        if method == "RERANKER":
            self._require_retrieval_contract()
            return [
                {
                    "query": self.retrieval_query,
                    "document": self.retrieval_positive,
                    "label": 1.0,
                },
                *[
                    {
                        "query": self.retrieval_query,
                        "document": negative,
                        "label": 0.0,
                    }
                    for negative in self.retrieval_hard_negatives
                ],
            ]
        if method == "VLM":
            if (
                self.media_content is None
                or self.media_mime_type not in {"image/png", "image/jpeg"}
                or self.multimodal_instruction is None
                or self.multimodal_answer is None
            ):
                raise TrainingDatasetError("vlm_training_sample_is_incomplete")
            return [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": self.multimodal_instruction},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": self.multimodal_answer}],
                        },
                    ],
                    "candidate_id": self.candidate_id,
                    "media_content": self.media_content,
                    "media_mime_type": self.media_mime_type,
                }
            ]
        if method == "ASR":
            if (
                self.media_content is None
                or self.media_mime_type not in {"audio/wav", "audio/flac"}
                or self.transcript is None
                or self.language is None
            ):
                raise TrainingDatasetError("asr_training_sample_is_incomplete")
            return [
                {
                    "media_content": self.media_content,
                    "media_mime_type": self.media_mime_type,
                    "transcript": self.transcript,
                    "language": self.language,
                }
            ]
        if method == "TTS":
            if (
                self.media_content is None
                or self.media_mime_type not in {"audio/wav", "audio/flac"}
                or self.tts_target_text is None
                or self.language is None
                or self.voice_profile_id is None
                or not self.speaker_embedding
                or self.voice_source_type not in {"PLATFORM_SYNTHETIC", "LICENSED_STUDIO"}
                or self.voice_usage_evidence_sha256 is None
            ):
                raise TrainingDatasetError("tts_training_sample_is_incomplete")
            return [
                {
                    "media_content": self.media_content,
                    "media_mime_type": self.media_mime_type,
                    "target_text": self.tts_target_text,
                    "language": self.language,
                    "voice_profile_id": self.voice_profile_id,
                    "speaker_embedding": self.speaker_embedding,
                    "voice_source_type": self.voice_source_type,
                    "voice_usage_evidence_sha256": self.voice_usage_evidence_sha256,
                }
            ]
        raise TrainingDatasetError("training_method_has_no_dataset_contract")

    def _require_retrieval_contract(self) -> None:
        if (
            self.retrieval_query is None
            or self.retrieval_positive is None
            or not self.retrieval_hard_negatives
        ):
            raise TrainingDatasetError("retrieval_training_triplet_is_missing")


@dataclass(frozen=True, slots=True)
class TrainingDatasetBundle:
    method: str
    snapshot_id: str
    manifest_hash: str
    contract_version: str
    sample_origin_counts: dict[str, int]
    train: tuple[SftSample, ...]
    validation: tuple[SftSample, ...]
    split_hashes: dict[str, str]

    @property
    def train_records(self) -> list[dict[str, Any]]:
        return [record for sample in self.train for record in sample.trainer_records(self.method)]

    @property
    def validation_records(self) -> list[dict[str, Any]]:
        return [
            record for sample in self.validation for record in sample.trainer_records(self.method)
        ]

    @property
    def synthetic_train_count(self) -> int:
        return sum(sample.synthetic for sample in self.train)

    @property
    def real_train_count(self) -> int:
        return len(self.train) - self.synthetic_train_count


@dataclass(frozen=True, slots=True)
class _ArtifactContract:
    split: str
    object_key: str
    content_hash: str
    size_bytes: int
    row_count: int
    schema_hash: str


@dataclass(frozen=True, slots=True)
class _MediaArtifactContract:
    source_media_id: str
    split: str
    object_key: str
    content_hash: str
    size_bytes: int
    schema_hash: str
    mime_type: str
    candidate_ids: tuple[str, ...]
    synthetic: bool
    synthetic_manifest_sha256: str | None


@dataclass(frozen=True, slots=True)
class _GovernedMedia:
    source_media_id: str
    split: str
    mime_type: str
    content: bytes
    candidate_ids: tuple[str, ...]
    synthetic: bool
    synthetic_manifest_sha256: str | None


class GovernedSftDatasetLoader:
    def __init__(self, database: Database, store: DatasetStore) -> None:
        self._database = database
        self._store = store

    def load(
        self,
        context: TenantContext,
        experiment: TrainingExperimentRecord,
    ) -> TrainingDatasetBundle:
        if context.tenant_id != experiment.tenant_id:
            raise TrainingDatasetError("experiment_tenant_mismatch")
        with self._database.transaction(context) as session:
            snapshot = session.scalar(
                select(DatasetSnapshotRecord).where(
                    DatasetSnapshotRecord.tenant_id == context.tenant_id,
                    DatasetSnapshotRecord.snapshot_id == experiment.dataset_snapshot_id,
                )
            )
            if snapshot is None:
                raise TrainingDatasetError("dataset_snapshot_not_visible")
            lineage = session.scalar(
                select(DataLineageRunRecord).where(
                    DataLineageRunRecord.tenant_id == context.tenant_id,
                    DataLineageRunRecord.snapshot_id == snapshot.snapshot_id,
                )
            )
            database_artifacts = list(
                session.scalars(
                    select(DatasetArtifactRecord).where(
                        DatasetArtifactRecord.tenant_id == context.tenant_id,
                        DatasetArtifactRecord.snapshot_id == snapshot.snapshot_id,
                    )
                )
            )
            quality_artifacts = [
                artifact for artifact in database_artifacts if artifact.kind == "quality_report"
            ]
            _validate_snapshot(snapshot, experiment, lineage, quality_artifacts)
            manifest_key = snapshot.manifest_key
            expected_manifest_hash = str(snapshot.manifest_hash)
            snapshot_contract = snapshot.contract_version
            snapshot_row_count = snapshot.row_count
            snapshot_split_counts = dict(snapshot.split_counts)
            quality_report_key = snapshot.quality_report_key

        assert manifest_key is not None  # guarded by _validate_snapshot
        assert quality_report_key is not None  # guarded by _validate_snapshot
        assert len(quality_artifacts) == 1  # guarded by _validate_snapshot
        try:
            quality_payload = self._store.get_bytes(quality_report_key)
            verify_dataset_quality_report(snapshot, quality_artifacts[0], quality_payload)
        except (FileNotFoundError, DatasetQualityReportError) as exc:
            raise TrainingDatasetError("dataset_quality_report_integrity_failed") from exc
        manifest_bytes = self._store.get_bytes(manifest_key)
        actual_manifest_hash = _digest_for_expected(
            manifest_bytes,
            expected_manifest_hash,
        )
        if actual_manifest_hash != expected_manifest_hash:
            raise TrainingDatasetError("dataset_manifest_hash_mismatch")
        manifest = _json_object(manifest_bytes, "dataset_manifest_is_invalid_json")
        is_simulation_profile = snapshot_contract == SIMULATED_MULTIMODAL_CONTRACT
        if is_simulation_profile:
            if experiment.method != "VLM" or experiment.task_type != "SIMULATED_MULTIMODAL_VLM":
                raise TrainingDatasetError(
                    "simulated_multimodal_snapshot_requires_simulation_vlm_task"
                )
            sample_origin_counts = _validate_simulated_multimodal_manifest(
                manifest,
                tenant_id=context.tenant_id,
                snapshot_id=experiment.dataset_snapshot_id,
                row_count=snapshot_row_count,
                split_counts=snapshot_split_counts,
            )
        else:
            sample_origin_counts = _validate_manifest(
                manifest,
                tenant_id=context.tenant_id,
                snapshot_id=experiment.dataset_snapshot_id,
                contract_version=snapshot_contract,
                row_count=snapshot_row_count,
                split_counts=snapshot_split_counts,
            )
        if sample_origin_counts["synthetic"] and experiment.method != "VLM":
            raise TrainingDatasetError("synthetic_media_snapshot_requires_vlm_training")

        manifest_artifacts = _artifact_contracts(manifest)
        media_contracts = _media_artifact_contracts(
            manifest,
            simulation_profile=is_simulation_profile,
        )
        if (
            sum(item.synthetic for item in media_contracts.values())
            != sample_origin_counts["synthetic"]
        ):
            raise TrainingDatasetError("synthetic_media_manifest_count_mismatch")
        _cross_check_catalog(manifest_artifacts, media_contracts, database_artifacts)
        if is_simulation_profile:
            self._verify_simulation_provenance(manifest, database_artifacts)
        else:
            self._verify_synthetic_provenance(manifest, database_artifacts)
        governed_media = self._load_governed_media(media_contracts)
        frames: dict[str, pd.DataFrame] = {}
        split_hashes: dict[str, str] = {}
        for split in ("train", "validation"):
            artifact = manifest_artifacts.get(split)
            if artifact is None:
                raise TrainingDatasetError(f"{split}_parquet_artifact_is_missing")
            content = self._store.get_bytes(artifact.object_key)
            if len(content) != artifact.size_bytes or not _digest_matches(
                content, artifact.content_hash
            ):
                raise TrainingDatasetError(f"{split}_parquet_integrity_failed")
            frame = parquet.read_table(BytesIO(content)).to_pandas()
            frame = frame.loc[:, list(DATASET_COLUMNS)]
            DATASET_SCHEMA.validate(frame, lazy=True)
            if len(frame) != artifact.row_count:
                raise TrainingDatasetError(f"{split}_row_count_mismatch")
            if set(frame["split"].astype(str)) != {split}:
                raise TrainingDatasetError(f"{split}_contains_foreign_split_rows")
            if set(frame["tenant_id"].astype(str)) != {context.tenant_id}:
                raise TrainingDatasetError(f"{split}_contains_foreign_tenant_rows")
            frames[split] = frame
            split_hashes[split] = artifact.content_hash

        if frames["train"].empty:
            raise TrainingDatasetError("training_split_is_empty")
        if frames["validation"].empty:
            raise TrainingDatasetError("validation_split_is_empty")
        overlap = set(frames["train"]["group_key"].astype(str)) & set(
            frames["validation"]["group_key"].astype(str)
        )
        if overlap:
            raise TrainingDatasetError("group_leakage_between_train_and_validation")
        hard_negatives_per_query = _hard_negatives_per_query(experiment)
        tts_speaker_embedding_dimension = _tts_speaker_embedding_dimension(experiment)

        return TrainingDatasetBundle(
            method=experiment.method,
            snapshot_id=experiment.dataset_snapshot_id,
            manifest_hash=actual_manifest_hash,
            contract_version=str(manifest["contract_version"]),
            sample_origin_counts=sample_origin_counts,
            train=tuple(
                _sample(
                    row,
                    method=experiment.method,
                    hard_negatives_per_query=hard_negatives_per_query,
                    governed_media=governed_media,
                    tts_speaker_embedding_dimension=tts_speaker_embedding_dimension,
                    simulation_profile=is_simulation_profile,
                )
                for row in frames["train"].to_dict("records")
            ),
            validation=tuple(
                _sample(
                    row,
                    method=experiment.method,
                    hard_negatives_per_query=hard_negatives_per_query,
                    governed_media=governed_media,
                    tts_speaker_embedding_dimension=tts_speaker_embedding_dimension,
                    simulation_profile=is_simulation_profile,
                )
                for row in frames["validation"].to_dict("records")
            ),
            split_hashes=split_hashes,
        )

    def _load_governed_media(
        self,
        contracts: dict[str, _MediaArtifactContract],
    ) -> dict[str, _GovernedMedia]:
        loaded: dict[str, _GovernedMedia] = {}
        for media_id, artifact in contracts.items():
            content = self._store.get_bytes(artifact.object_key)
            if len(content) != artifact.size_bytes or not _digest_matches(
                content, artifact.content_hash
            ):
                raise TrainingDatasetError("media_artifact_integrity_failed")
            try:
                inspection = inspect_media(
                    content,
                    declared_mime=artifact.mime_type,
                    max_bytes=artifact.size_bytes,
                )
            except MediaValidationFailure as exc:
                raise TrainingDatasetError("media_artifact_signature_is_invalid") from exc
            if inspection.content_hash != artifact.content_hash.removeprefix("sha256:"):
                raise TrainingDatasetError("media_artifact_integrity_failed")
            loaded[media_id] = _GovernedMedia(
                source_media_id=media_id,
                split=artifact.split,
                mime_type=artifact.mime_type,
                content=content,
                candidate_ids=artifact.candidate_ids,
                synthetic=artifact.synthetic,
                synthetic_manifest_sha256=artifact.synthetic_manifest_sha256,
            )
        return loaded

    def _verify_synthetic_provenance(
        self,
        manifest: dict[str, Any],
        database_artifacts: list[DatasetArtifactRecord],
    ) -> None:
        entries = manifest.get("synthetic_media", [])
        if not entries:
            return
        catalog = {artifact.object_key: artifact for artifact in database_artifacts}
        expected_kinds = {
            "manifest_key": "synthetic_provenance",
            "source_image_key": "synthetic_source_image",
            "mask_image_key": "synthetic_mask_image",
            "output_image_key": "media_image",
        }
        for entry in entries:
            assert isinstance(entry, dict)  # validated by _validate_manifest
            payloads: dict[str, bytes] = {}
            for field_name, expected_kind in expected_kinds.items():
                object_key = entry.get(field_name)
                artifact = catalog.get(str(object_key))
                if (
                    not isinstance(object_key, str)
                    or artifact is None
                    or artifact.kind != expected_kind
                    or artifact.split != "train"
                ):
                    raise TrainingDatasetError("synthetic_provenance_catalog_is_invalid")
                content = self._store.get_bytes(object_key)
                if len(content) != artifact.size_bytes or not _digest_matches(
                    content, artifact.content_hash
                ):
                    raise TrainingDatasetError("synthetic_provenance_integrity_failed")
                payloads[field_name] = content
            reviewed_manifest = _json_object(
                payloads["manifest_key"],
                "synthetic_provenance_manifest_is_invalid",
            )
            if reviewed_manifest.get("manifest_sha256") != entry.get("manifest_sha256"):
                raise TrainingDatasetError("synthetic_provenance_manifest_binding_changed")
            try:
                verify_synthetic_media_bundle(
                    reviewed_manifest,
                    image_png=payloads["output_image_key"],
                    source_image=payloads["source_image_key"],
                    mask_image=payloads["mask_image_key"],
                )
            except SyntheticMediaError as exc:
                raise TrainingDatasetError("synthetic_provenance_integrity_failed") from exc

    def _verify_simulation_provenance(
        self,
        manifest: dict[str, Any],
        database_artifacts: list[DatasetArtifactRecord],
    ) -> None:
        profile = manifest.get("simulation_profile")
        if not isinstance(profile, dict):
            raise TrainingDatasetError("simulation_profile_is_missing")
        expected = {
            "source_manifest_key": ("simulation_provenance", "manifest_sha256"),
            "source_evaluation_freeze_key": (
                "simulation_provenance",
                "freeze_sha256",
            ),
            "review_evidence_key": ("simulation_review", "review_sha256"),
        }
        catalog = {artifact.object_key: artifact for artifact in database_artifacts}
        loaded: dict[str, dict[str, Any]] = {}
        for field_name, (kind, _) in expected.items():
            object_key = profile.get(field_name)
            artifact = catalog.get(str(object_key))
            if (
                not isinstance(object_key, str)
                or artifact is None
                or artifact.kind != kind
                or artifact.split is not None
            ):
                raise TrainingDatasetError("simulation_provenance_catalog_is_invalid")
            content = self._store.get_bytes(object_key)
            if len(content) != artifact.size_bytes or not _digest_matches(
                content, artifact.content_hash
            ):
                raise TrainingDatasetError("simulation_provenance_integrity_failed")
            loaded[field_name] = _json_object(
                content,
                "simulation_provenance_document_is_invalid",
            )
        source = loaded["source_manifest_key"]
        freeze = loaded["source_evaluation_freeze_key"]
        review = loaded["review_evidence_key"]
        if (
            source.get("classification") != "SIMULATED_NON_PRODUCTION"
            or source.get("production_gold_eligible") is not False
            or source.get("manifest_sha256") != _canonical_document_hash(source, "manifest_sha256")
            or freeze.get("classification") != "SIMULATED_NON_PRODUCTION"
            or freeze.get("production_gold_eligible") is not False
            or freeze.get("freeze_sha256") != _canonical_document_hash(freeze, "freeze_sha256")
            or review.get("status") != "APPROVED_FOR_LOCAL_SIMULATION"
            or review.get("production_approval") is not False
            or review.get("generated_by_subject_id") == review.get("reviewed_by_subject_id")
            or review.get("review_sha256") != _canonical_document_hash(review, "review_sha256")
        ):
            raise TrainingDatasetError("simulation_provenance_governance_is_invalid")
        for field_name, (_, digest_field) in expected.items():
            expected_digest = profile.get(digest_field)
            if expected_digest != loaded[field_name].get(digest_field):
                raise TrainingDatasetError("simulation_provenance_binding_changed")


def _validate_snapshot(
    snapshot: DatasetSnapshotRecord,
    experiment: TrainingExperimentRecord,
    lineage: DataLineageRunRecord | None,
    quality_artifacts: list[DatasetArtifactRecord],
) -> None:
    if dataset_training_blockers(snapshot, lineage, quality_artifacts):
        raise TrainingDatasetError("dataset_snapshot_is_not_training_eligible")
    if snapshot.manifest_hash != experiment.dataset_manifest_hash:
        raise TrainingDatasetError("experiment_manifest_binding_changed")


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    tenant_id: str,
    snapshot_id: str,
    contract_version: str,
    row_count: int,
    split_counts: dict[str, int],
) -> dict[str, int]:
    expected = {
        "tenant_id": tenant_id,
        "snapshot_id": snapshot_id,
        "contract_version": contract_version,
        "row_count": row_count,
        "split_counts": split_counts,
        "lineage_status": "CONFIRMED",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise TrainingDatasetError(f"dataset_manifest_{key}_mismatch")
    if contract_version != CONTRACT_VERSION:
        raise TrainingDatasetError("dataset_contract_version_is_not_supported")
    if sum(split_counts.values()) != row_count:
        raise TrainingDatasetError("dataset_manifest_split_counts_are_invalid")
    counts = manifest.get("sample_origin_counts")
    if counts is None:
        return {"real": row_count, "synthetic": 0}
    if (
        not isinstance(counts, dict)
        or isinstance(counts.get("real"), bool)
        or not isinstance(counts.get("real"), int)
        or isinstance(counts.get("synthetic"), bool)
        or not isinstance(counts.get("synthetic"), int)
        or counts["real"] < 0
        or counts["synthetic"] <= 0
        or counts["real"] + counts["synthetic"] != row_count
        or manifest.get("augmentation_contract_version") != SYNTHETIC_AUGMENTATION_CONTRACT
        or manifest.get("evaluation_eligible") is not False
        or manifest.get("synthetic_split_counts")
        != {"train": counts["synthetic"], "validation": 0, "test": 0}
    ):
        raise TrainingDatasetError("synthetic_media_origin_contract_is_invalid")
    _validate_synthetic_manifest_entries(manifest, counts["synthetic"])
    return {"real": counts["real"], "synthetic": counts["synthetic"]}


def _validate_simulated_multimodal_manifest(
    manifest: dict[str, Any],
    *,
    tenant_id: str,
    snapshot_id: str,
    row_count: int,
    split_counts: dict[str, int],
) -> dict[str, int]:
    expected = {
        "tenant_id": tenant_id,
        "snapshot_id": snapshot_id,
        "contract_version": SIMULATED_MULTIMODAL_CONTRACT,
        "classification": "SIMULATED_NON_PRODUCTION",
        "production_claim": False,
        "production_release_eligible": False,
        "lineage_status": "CONFIRMED",
        "row_count": row_count,
        "split_counts": split_counts,
        "evaluation_eligible": False,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise TrainingDatasetError("simulated_multimodal_manifest_binding_mismatch")
    counts = manifest.get("sample_origin_counts")
    synthetic_splits = manifest.get("synthetic_split_counts")
    profile = manifest.get("simulation_profile")
    if (
        sum(split_counts.values()) != row_count
        or split_counts.get("test") != 0
        or not isinstance(counts, dict)
        or counts != {"real": 0, "synthetic": row_count}
        or synthetic_splits
        != {
            "train": split_counts.get("train", 0),
            "validation": split_counts.get("validation", 0),
            "test": 0,
        }
        or not isinstance(profile, dict)
        or profile.get("purpose") != "LOCAL_SIMULATION_VLM_TRAINING"
        or profile.get("training_output_contract") != "industrial-vlm-findings-v1"
        or profile.get("production_gold_eligible") is not False
        or profile.get("expert_review_required_for_production") is not True
    ):
        raise TrainingDatasetError("simulated_multimodal_origin_contract_is_invalid")
    return {"real": 0, "synthetic": row_count}


def _artifact_contracts(manifest: dict[str, Any]) -> dict[str, _ArtifactContract]:
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise TrainingDatasetError("dataset_manifest_artifacts_are_invalid")
    artifacts: dict[str, _ArtifactContract] = {}
    for raw in entries:
        if not isinstance(raw, dict) or raw.get("kind") != "parquet":
            continue
        split = raw.get("split")
        if split not in {"train", "validation", "test"} or split in artifacts:
            raise TrainingDatasetError("dataset_manifest_parquet_splits_are_invalid")
        try:
            artifact = _ArtifactContract(
                split=split,
                object_key=str(raw["object_key"]),
                content_hash=str(raw["content_hash"]),
                size_bytes=int(raw["size_bytes"]),
                row_count=int(raw["row_count"]),
                schema_hash=str(raw["schema_hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainingDatasetError("dataset_manifest_artifact_is_invalid") from exc
        if artifact.size_bytes <= 0 or artifact.row_count < 0:
            raise TrainingDatasetError("dataset_manifest_artifact_is_invalid")
        artifacts[split] = artifact
    if set(artifacts) != {"train", "validation", "test"}:
        raise TrainingDatasetError("dataset_manifest_requires_all_three_splits")
    return artifacts


def _media_artifact_contracts(
    manifest: dict[str, Any],
    *,
    simulation_profile: bool = False,
) -> dict[str, _MediaArtifactContract]:
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise TrainingDatasetError("dataset_manifest_artifacts_are_invalid")
    artifacts: dict[str, _MediaArtifactContract] = {}
    expected_schema_hash = sha256(b"industrial-governed-media-v1").hexdigest()
    for raw in entries:
        if not isinstance(raw, dict) or raw.get("kind") not in {
            "media_image",
            "media_audio",
        }:
            continue
        try:
            media_id = str(raw["source_media_id"])
            candidate_ids_raw = raw["candidate_ids"]
            if not isinstance(candidate_ids_raw, list):
                raise TypeError
            artifact = _MediaArtifactContract(
                source_media_id=media_id,
                split=str(raw["split"]),
                object_key=str(raw["object_key"]),
                content_hash=str(raw["content_hash"]),
                size_bytes=int(raw["size_bytes"]),
                schema_hash=str(raw["schema_hash"]),
                mime_type=str(raw["mime_type"]),
                candidate_ids=tuple(str(value) for value in candidate_ids_raw),
                synthetic=raw.get("synthetic", False),
                synthetic_manifest_sha256=(
                    str(raw["synthetic_manifest_sha256"])
                    if raw.get("synthetic_manifest_sha256") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainingDatasetError("dataset_manifest_media_artifact_is_invalid") from exc
        if (
            not media_id
            or media_id in artifacts
            or artifact.split not in {"train", "validation", "test"}
            or artifact.size_bytes <= 0
            or not artifact.candidate_ids
            or len(set(artifact.candidate_ids)) != len(artifact.candidate_ids)
            or artifact.schema_hash.removeprefix("sha256:") != expected_schema_hash
            or artifact.mime_type not in {"image/png", "image/jpeg", "audio/wav", "audio/flac"}
            or not isinstance(artifact.synthetic, bool)
            or (
                artifact.synthetic
                and (
                    artifact.split
                    not in ({"train", "validation"} if simulation_profile else {"train"})
                    or artifact.mime_type != "image/png"
                    or len(artifact.candidate_ids) != 1
                    or not _sha256_digest(artifact.synthetic_manifest_sha256)
                    or raw.get("evidence_eligible") is not False
                    or raw.get("golden_dataset_eligible") is not False
                )
            )
            or (not artifact.synthetic and artifact.synthetic_manifest_sha256 is not None)
        ):
            raise TrainingDatasetError("dataset_manifest_media_artifact_is_invalid")
        artifacts[media_id] = artifact
    return artifacts


def _cross_check_catalog(
    manifest_artifacts: dict[str, _ArtifactContract],
    media_artifacts: dict[str, _MediaArtifactContract],
    database_artifacts: list[DatasetArtifactRecord],
) -> None:
    catalog = {
        str(item.split): item
        for item in database_artifacts
        if item.kind == "parquet" and item.split is not None
    }
    if set(catalog) != set(manifest_artifacts):
        raise TrainingDatasetError("dataset_catalog_and_manifest_differ")
    expected_schema_hash = sha256("\0".join(DATASET_COLUMNS).encode()).hexdigest()
    for split, manifest in manifest_artifacts.items():
        record = catalog[split]
        if (
            manifest.object_key != record.object_key
            or manifest.content_hash != record.content_hash
            or manifest.size_bytes != record.size_bytes
            or manifest.row_count != record.row_count
            or manifest.schema_hash != record.schema_hash
            or manifest.schema_hash.removeprefix("sha256:") != expected_schema_hash
        ):
            raise TrainingDatasetError("dataset_catalog_and_manifest_differ")
    media_catalog = {
        item.object_key: item
        for item in database_artifacts
        if item.kind in {"media_image", "media_audio"}
    }
    if set(media_catalog) != {item.object_key for item in media_artifacts.values()}:
        raise TrainingDatasetError("dataset_media_catalog_and_manifest_differ")
    for media_manifest in media_artifacts.values():
        record = media_catalog[media_manifest.object_key]
        expected_kind = (
            "media_image" if media_manifest.mime_type.startswith("image/") else "media_audio"
        )
        if (
            record.kind != expected_kind
            or record.split != media_manifest.split
            or record.content_hash != media_manifest.content_hash
            or record.size_bytes != media_manifest.size_bytes
            or record.row_count != len(media_manifest.candidate_ids)
            or record.schema_hash != media_manifest.schema_hash
        ):
            raise TrainingDatasetError("dataset_media_catalog_and_manifest_differ")


def _sample(
    row: dict[str, Any],
    *,
    method: str,
    hard_negatives_per_query: int,
    governed_media: dict[str, _GovernedMedia],
    tts_speaker_embedding_dimension: int = 0,
    simulation_profile: bool = False,
) -> SftSample:
    evidence = _canonical_json_text(row["redacted_content_json"], "redacted_content_is_invalid")
    labels_value = _json_value(row["labels_json"], "labels_are_invalid")
    if not isinstance(labels_value, dict):
        raise TrainingDatasetError("labels_must_be_an_object")
    classification_contract = labels_value.pop("_classification_contract", None)
    system_instruction = classification_system_instruction(classification_contract)
    post_training = labels_value.pop("_post_training", None)
    retrieval_training = labels_value.pop("_retrieval_training", None)
    multimodal_training = labels_value.pop("_multimodal_training", None)
    for field_name in _EVALUATION_CONTROL_FIELDS:
        labels_value.pop(field_name, None)
    if not labels_value:
        raise TrainingDatasetError("supervised_labels_are_empty_after_control_field_removal")
    labels = json.dumps(labels_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    prompt = [
        {"role": "system", "content": system_instruction},
        {
            "role": "user",
            "content": (f"请根据以下脱敏工业工单证据完成根因分类。\nevidence={evidence}"),
        },
    ]
    completion = [{"role": "assistant", "content": labels}]
    preference_chosen = None
    preference_rejected = None
    reward_target = None
    reward_contract_source_version = None
    retrieval_query = None
    retrieval_positive = None
    retrieval_hard_negatives: tuple[str, ...] = ()
    media_artifact_id = None
    media_content = None
    media_mime_type = None
    multimodal_instruction = None
    multimodal_answer = None
    transcript = None
    language = None
    tts_target_text = None
    voice_profile_id = None
    speaker_embedding: tuple[float, ...] = ()
    voice_source_type = None
    voice_usage_evidence_sha256 = None
    synthetic = False
    synthetic_manifest_sha256 = None
    if method == "DPO":
        if not isinstance(post_training, dict):
            raise TrainingDatasetError("dpo_post_training_contract_is_missing")
        preference = post_training.get("preference")
        if not isinstance(preference, dict) or preference.get("review_status") != "APPROVED":
            raise TrainingDatasetError("dpo_preference_review_is_not_approved")
        chosen_text = _canonical_completion(preference.get("chosen"), "dpo_chosen_is_invalid")
        rejected_text = _canonical_completion(preference.get("rejected"), "dpo_rejected_is_invalid")
        if chosen_text == rejected_text:
            raise TrainingDatasetError("dpo_preference_pair_must_differ")
        preference_chosen = [{"role": "assistant", "content": chosen_text}]
        preference_rejected = [{"role": "assistant", "content": rejected_text}]
    elif method == "GRPO":
        if not isinstance(post_training, dict):
            raise TrainingDatasetError("grpo_reward_target_is_missing")
        legacy_target = post_training.get("reward_target")
        versioned_reward = post_training.get("reward")
        if legacy_target is not None and versioned_reward is not None:
            raise TrainingDatasetError("grpo_reward_contract_is_ambiguous")
        if legacy_target is not None:
            reward_target = _canonical_completion(legacy_target, "grpo_reward_target_is_invalid")
            reward_contract_source_version = LEGACY_REWARD_CONTRACT_VERSION
        elif isinstance(versioned_reward, dict):
            if versioned_reward.get("review_status") != "APPROVED":
                raise TrainingDatasetError("grpo_reward_review_is_not_approved")
            source_version = versioned_reward.get("contract_version")
            if not isinstance(source_version, str):
                raise TrainingDatasetError("grpo_reward_contract_version_is_missing")
            try:
                reward_contract(source_version)
            except RewardContractError as exc:
                raise TrainingDatasetError(str(exc)) from exc
            reward_target = _canonical_completion(
                versioned_reward.get("target"), "grpo_reward_target_is_invalid"
            )
            reward_contract_source_version = source_version
        else:
            raise TrainingDatasetError("grpo_reward_target_is_missing")
    elif method in {"EMBEDDING", "RERANKER"}:
        if (
            not isinstance(retrieval_training, dict)
            or retrieval_training.get("review_status") != "APPROVED"
        ):
            raise TrainingDatasetError("retrieval_training_review_is_not_approved")
        retrieval_query = _required_retrieval_text(
            retrieval_training.get("query"), "retrieval_query_is_invalid"
        )
        retrieval_positive = _required_retrieval_text(
            retrieval_training.get("positive"), "retrieval_positive_is_invalid"
        )
        negatives = retrieval_training.get("hard_negatives")
        if (
            not isinstance(negatives, list)
            or not negatives
            or not all(isinstance(item, str) and item.strip() for item in negatives)
        ):
            raise TrainingDatasetError("retrieval_hard_negatives_are_invalid")
        retrieval_hard_negatives = tuple(item.strip() for item in negatives)
        if len(retrieval_hard_negatives) != hard_negatives_per_query:
            raise TrainingDatasetError("retrieval_hard_negative_count_mismatch")
        if (
            len(set(retrieval_hard_negatives)) != len(retrieval_hard_negatives)
            or retrieval_positive in retrieval_hard_negatives
        ):
            raise TrainingDatasetError("retrieval_triplet_contains_duplicate_documents")
    elif method in {"VLM", "ASR", "TTS"}:
        if (
            not isinstance(multimodal_training, dict)
            or multimodal_training.get("review_status") != "APPROVED"
            or not isinstance(multimodal_training.get("media_id"), str)
        ):
            raise TrainingDatasetError("multimodal_training_review_is_not_approved")
        media_artifact_id = multimodal_training["media_id"].strip()
        media = governed_media.get(media_artifact_id)
        if (
            media is None
            or media.split != str(row["split"])
            or str(row["candidate_id"]) not in media.candidate_ids
        ):
            raise TrainingDatasetError("multimodal_media_binding_is_invalid")
        expected_modality = "image" if method == "VLM" else "audio"
        if multimodal_training.get(
            "modality"
        ) != expected_modality or not media.mime_type.startswith(f"{expected_modality}/"):
            raise TrainingDatasetError("multimodal_media_modality_is_invalid")
        media_content = media.content
        media_mime_type = media.mime_type
        if method == "VLM":
            synthetic_contract = multimodal_training.get("synthetic_media")
            if media.synthetic:
                if simulation_profile:
                    valid_synthetic_contract = (
                        isinstance(synthetic_contract, dict)
                        and synthetic_contract.get("schema_version")
                        == "simulated-multimodal-fault-dataset/v1"
                        and synthetic_contract.get("manifest_sha256")
                        == media.synthetic_manifest_sha256
                        and synthetic_contract.get("source_type") == "PROJECT_GENERATED"
                        and isinstance(synthetic_contract.get("reviewer_subject_id"), str)
                        and synthetic_contract.get("production_eligible") is False
                    )
                else:
                    valid_synthetic_contract = (
                        isinstance(synthetic_contract, dict)
                        and synthetic_contract.get("schema_version") == "synthetic-media/v1"
                        and synthetic_contract.get("manifest_sha256")
                        == media.synthetic_manifest_sha256
                        and isinstance(synthetic_contract.get("generator_release_id"), str)
                        and isinstance(synthetic_contract.get("reviewer_subject_id"), str)
                    )
                if not valid_synthetic_contract:
                    raise TrainingDatasetError("synthetic_media_binding_is_invalid")
                synthetic = True
                synthetic_manifest_sha256 = media.synthetic_manifest_sha256
            elif synthetic_contract is not None:
                raise TrainingDatasetError("synthetic_media_binding_is_invalid")
            instruction = _required_multimodal_text(
                multimodal_training.get("instruction"), "vlm_instruction_is_invalid"
            )
            answer = _required_multimodal_text(
                multimodal_training.get("answer"), "vlm_answer_is_invalid"
            )
            if simulation_profile:
                answer = _simulation_vlm_answer(answer)
            multimodal_instruction = f"{instruction}\n脱敏工单证据：{evidence}"
            multimodal_answer = answer
        elif method == "ASR":
            transcript = _required_multimodal_text(
                multimodal_training.get("transcript"), "asr_transcript_is_invalid"
            )
            language = _required_multimodal_text(
                multimodal_training.get("language"), "asr_language_is_invalid", max_length=32
            )
        else:
            expected_keys = {
                "task",
                "review_status",
                "modality",
                "media_id",
                "target_text",
                "language",
                "voice_profile_id",
                "speaker_embedding",
                "voice_usage",
            }
            if (
                set(multimodal_training) != expected_keys
                or multimodal_training.get("task") != "speech_synthesis"
                or media.mime_type not in {"audio/wav", "audio/flac"}
            ):
                raise TrainingDatasetError("tts_training_contract_is_invalid")
            voice_usage = multimodal_training.get("voice_usage")
            if not isinstance(voice_usage, dict) or set(voice_usage) != {
                "purpose",
                "source_type",
                "license_status",
                "evidence_sha256",
            }:
                raise TrainingDatasetError("tts_voice_usage_is_not_authorized")
            source_type = voice_usage.get("source_type")
            evidence_digest = voice_usage.get("evidence_sha256")
            if (
                voice_usage.get("purpose") != "ENTERPRISE_SAFETY_GUIDANCE"
                or source_type not in {"PLATFORM_SYNTHETIC", "LICENSED_STUDIO"}
                or voice_usage.get("license_status") != "APPROVED"
                or not _sha256_digest(evidence_digest)
            ):
                raise TrainingDatasetError("tts_voice_usage_is_not_authorized")
            raw_embedding = multimodal_training.get("speaker_embedding")
            if (
                tts_speaker_embedding_dimension <= 0
                or not isinstance(raw_embedding, list)
                or len(raw_embedding) != tts_speaker_embedding_dimension
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in raw_embedding
                )
            ):
                raise TrainingDatasetError("tts_speaker_embedding_is_invalid")
            tts_target_text = _required_multimodal_text(
                multimodal_training.get("target_text"),
                "tts_target_text_is_invalid",
                max_length=4096,
            )
            language = _required_multimodal_text(
                multimodal_training.get("language"),
                "tts_language_is_invalid",
                max_length=32,
            )
            voice_profile_id = _required_multimodal_text(
                multimodal_training.get("voice_profile_id"),
                "tts_voice_profile_is_invalid",
                max_length=128,
            )
            speaker_embedding = tuple(float(value) for value in raw_embedding)
            voice_source_type = str(source_type)
            voice_usage_evidence_sha256 = str(evidence_digest)

    return SftSample(
        prompt=prompt,
        completion=completion,
        candidate_id=str(row["candidate_id"]),
        preference_chosen=preference_chosen,
        preference_rejected=preference_rejected,
        reward_target=reward_target,
        reward_contract_source_version=reward_contract_source_version,
        retrieval_query=retrieval_query,
        retrieval_positive=retrieval_positive,
        retrieval_hard_negatives=retrieval_hard_negatives,
        media_artifact_id=media_artifact_id,
        media_content=media_content,
        media_mime_type=media_mime_type,
        multimodal_instruction=multimodal_instruction,
        multimodal_answer=multimodal_answer,
        transcript=transcript,
        language=language,
        tts_target_text=tts_target_text,
        voice_profile_id=voice_profile_id,
        speaker_embedding=speaker_embedding,
        voice_source_type=voice_source_type,
        voice_usage_evidence_sha256=voice_usage_evidence_sha256,
        synthetic=synthetic,
        synthetic_manifest_sha256=synthetic_manifest_sha256,
    )


def _validate_synthetic_manifest_entries(
    manifest: dict[str, Any],
    expected_count: int,
) -> None:
    entries = manifest.get("synthetic_media")
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise TrainingDatasetError("synthetic_media_manifest_count_mismatch")
    identifiers: set[str] = set()
    candidate_ids: set[str] = set()
    media_ids: set[str] = set()
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not _sha256_digest(entry.get("manifest_sha256"))
            or entry.get("split") != "train"
            or entry.get("evidence_eligible") is not False
            or entry.get("golden_dataset_eligible") is not False
            or not isinstance(entry.get("candidate_id"), str)
            or not isinstance(entry.get("media_id"), str)
        ):
            raise TrainingDatasetError("synthetic_media_manifest_entry_is_invalid")
        identifiers.add(str(entry["manifest_sha256"]))
        candidate_ids.add(str(entry["candidate_id"]))
        media_ids.add(str(entry["media_id"]))
    if (
        len(identifiers) != expected_count
        or len(candidate_ids) != expected_count
        or len(media_ids) != expected_count
    ):
        raise TrainingDatasetError("synthetic_media_manifest_entry_is_duplicated")


def _canonical_json_text(value: Any, error: str) -> str:
    parsed = _json_value(value, error)
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_value(value: Any, error: str) -> dict[str, Any] | list[Any]:
    if isinstance(value, (dict, list)):
        return copy.deepcopy(value)
    if not isinstance(value, str):
        raise TrainingDatasetError(error)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise TrainingDatasetError(error) from exc
    if not isinstance(parsed, (dict, list)):
        raise TrainingDatasetError(error)
    return parsed


def _canonical_completion(value: Any, error: str) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise TrainingDatasetError(error)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text
        if not isinstance(parsed, (dict, list)):
            raise TrainingDatasetError(error)
        value = parsed
    if not isinstance(value, (dict, list)):
        raise TrainingDatasetError(error)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _required_retrieval_text(value: Any, error: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 16_384:
        raise TrainingDatasetError(error)
    return value.strip()


def _required_multimodal_text(
    value: Any,
    error: str,
    *,
    max_length: int = 16_384,
) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise TrainingDatasetError(error)
    return value.strip()


def _simulation_vlm_answer(value: str) -> str:
    try:
        answer = json.loads(value)
    except json.JSONDecodeError as exc:
        raise TrainingDatasetError("simulation_vlm_answer_is_not_json") from exc
    if not isinstance(answer, dict) or set(answer) != {"findings"}:
        raise TrainingDatasetError("simulation_vlm_answer_contract_is_invalid")
    findings = answer["findings"]
    if not isinstance(findings, list) or not 1 <= len(findings) <= 100:
        raise TrainingDatasetError("simulation_vlm_findings_are_invalid")
    for finding in findings:
        if not isinstance(finding, dict) or set(finding) != {"label", "region"}:
            raise TrainingDatasetError("simulation_vlm_finding_is_invalid")
        label = finding["label"]
        region = finding["region"]
        if (
            not isinstance(label, str)
            or not label.strip()
            or len(label) > 128
            or not isinstance(region, dict)
            or set(region) != {"x", "y", "width", "height"}
        ):
            raise TrainingDatasetError("simulation_vlm_finding_is_invalid")
        coordinates: dict[str, float] = {}
        for key in ("x", "y", "width", "height"):
            raw = region[key]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TrainingDatasetError("simulation_vlm_region_is_invalid")
            coordinates[key] = float(raw)
        if (
            not 0 <= coordinates["x"] < 1
            or not 0 <= coordinates["y"] < 1
            or not 0 < coordinates["width"] <= 1
            or not 0 < coordinates["height"] <= 1
            or coordinates["x"] + coordinates["width"] > 1
            or coordinates["y"] + coordinates["height"] > 1
        ):
            raise TrainingDatasetError("simulation_vlm_region_is_invalid")
    return json.dumps(answer, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hard_negatives_per_query(experiment: TrainingExperimentRecord) -> int:
    if experiment.method not in {"EMBEDDING", "RERANKER"}:
        return 0
    value = experiment.training_config.get("hard_negatives_per_query", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingDatasetError("hard_negatives_per_query_is_invalid")
    return int(value)


def _tts_speaker_embedding_dimension(experiment: TrainingExperimentRecord) -> int:
    if experiment.method != "TTS":
        return 0
    value = experiment.training_config.get("speaker_embedding_dimension")
    if isinstance(value, bool) or not isinstance(value, int) or not 2 <= value <= 4096:
        raise TrainingDatasetError("tts_speaker_embedding_dimension_is_invalid")
    return value


def _json_object(content: bytes, error: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingDatasetError(error) from exc
    if not isinstance(value, dict):
        raise TrainingDatasetError(error)
    return value


def _digest(content: bytes) -> str:
    return sha256(content).hexdigest()


def _digest_for_expected(content: bytes, expected: str) -> str:
    digest = _digest(content)
    return f"sha256:{digest}" if expected.startswith("sha256:") else digest


def _digest_matches(content: bytes, expected: str) -> bool:
    return _digest(content) == expected.removeprefix("sha256:")


def _sha256_digest(value: object) -> bool:
    if not isinstance(value, str):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def _canonical_document_hash(value: dict[str, Any], field: str) -> str:
    document = copy.deepcopy(value)
    document.pop(field, None)
    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(payload).hexdigest()
