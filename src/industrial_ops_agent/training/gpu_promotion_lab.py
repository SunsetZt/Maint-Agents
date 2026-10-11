"""Project-authorized GPU model-promotion dataset and evidence orchestration."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast
from uuid import NAMESPACE_URL, uuid5

import pyarrow as arrow  # type: ignore[import-untyped]
import pyarrow.parquet as parquet  # type: ignore[import-untyped]
from sqlalchemy import select

from industrial_ops_agent.data_pipeline.contracts import CONTRACT_VERSION, DATASET_COLUMNS
from industrial_ops_agent.data_pipeline.storage import DatasetStore, ImmutableObjectExists
from industrial_ops_agent.evaluation.backend import (
    EvaluationRuntimeConfig,
    ModelEvaluationBackend,
    TransformersModelEvaluationBackend,
)
from industrial_ops_agent.evaluation.dataset import EvaluationCase
from industrial_ops_agent.evaluation.metrics import ModelObservation
from industrial_ops_agent.evaluation.runtime_backend import (
    RuntimeProbeOutcome,
    RuntimeProbeSandbox,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    CurationRunRecord,
    DataLineageRunRecord,
    DatasetArtifactRecord,
    DatasetSnapshotRecord,
    TrainingExperimentRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext

DATASET_PROTOCOL = "project-authorized-gpu-dataset/v1"
AUTHORIZATION_CLASSIFICATION = "PROJECT_OWNED_SYNTHETIC"
STAGING_CLASSIFICATION = "LOCAL_STAGING_PROJECT_AUTHORIZED"
DATA_CLASSIFICATION = "SYNTHETIC_DATA"
CLASSIFICATION_CONTRACT_VERSION = "industrial-root-cause-classification/v1"
BASE_MODEL_ID = "Qwen/Qwen3-0.6B"
BASE_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
BASE_MODEL_LICENSE = "Apache-2.0"
SPLITS = ("train", "validation", "gold")
FORMAL_SPLITS = ("train", "validation", "test")
LOCAL_ROLE_ATTESTATION = "LOCAL_ROLE_SIMULATION"
_EVALUATION_RUNTIME_GATES = (
    "cross_tenant_isolation",
    "unauthorized_tool_execution",
    "t3_control_execution",
    "tool_schema_success",
    "high_risk_approval",
    "valid_citations",
    "replay_reproducibility",
)
_GOLD_VARIANTS = (
    "夜班低负荷复核窗口",
    "计划检修前趋势复核窗口",
    "告警复位失败后的只读复核窗口",
)
_MODEL_FILES = frozenset(
    {
        "LICENSE",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
)
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "version",
        "authorization",
        "generator",
        "base_model",
        "scenarios",
    }
)
_AUTHORIZATION_FIELDS = frozenset(
    {
        "owner",
        "classification",
        "data_classification",
        "allowed_uses",
        "prohibited_uses",
        "contains_customer_data",
        "contains_personal_data",
        "production_claim",
    }
)
_GENERATOR_FIELDS = frozenset({"id", "version", "seed"})
_MODEL_FIELDS = frozenset(
    {
        "model_id",
        "revision",
        "license",
        "remote_code_allowed",
        "chat_template_sha256",
        "files",
    }
)
_MODEL_FILE_FIELDS = frozenset({"path", "size_bytes", "sha256"})
_SCENARIO_FIELDS = frozenset(
    {
        "scenario_id",
        "split",
        "equipment_family",
        "alarm_code",
        "observations",
        "root_cause_code",
        "root_cause",
        "inspection_steps",
        "prohibited_actions",
        "severity",
    }
)
_BOUNDARY_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class GpuPromotionLabError(RuntimeError):
    """The local promotion lab cannot establish governed evidence."""


class GpuPromotionLabContractError(GpuPromotionLabError):
    """A dataset or model identity contract is invalid."""


@dataclass(frozen=True, slots=True)
class DatasetPreparationResult:
    training_snapshot_id: str
    training_manifest_hash: str
    gold_snapshot_id: str
    gold_manifest_hash: str
    dataset_digest: str
    authorization_digest: str
    model_manifest_digest: str
    classification: str
    data_classification: str
    authorization_classification: str
    production_claim: bool


@dataclass(frozen=True, slots=True)
class FormalGoldPreparationResult:
    snapshot_id: str
    manifest_hash: str
    sample_count: int
    slice_counts: dict[str, int]
    dataset_digest: str
    classification: str
    data_classification: str
    production_claim: bool


@dataclass(frozen=True, slots=True)
class _Catalog:
    document: dict[str, Any]
    authorization: dict[str, Any]
    base_model: dict[str, Any]
    scenarios: tuple[dict[str, Any], ...]
    digest: str
    authorization_digest: str
    model_manifest_digest: str


@dataclass(frozen=True, slots=True)
class _Case:
    case_id: str
    split: Literal["train", "validation", "gold"]
    scenario_id: str
    equipment_family: str
    alarm_code: str
    observation: str
    root_cause_code: str
    root_cause: str
    inspection_steps: tuple[str, ...]
    prohibited_actions: tuple[str, ...]
    severity: str

    def leakage_text(self) -> str:
        return "\n".join(
            (
                self.equipment_family,
                self.alarm_code,
                self.observation,
                self.root_cause_code,
                self.root_cause,
                *self.inspection_steps,
                *self.prohibited_actions,
            )
        )


@dataclass(frozen=True, slots=True)
class _Artifact:
    kind: str
    split: str | None
    object_key: str
    content: bytes
    row_count: int
    schema_hash: str

    @property
    def content_hash(self) -> str:
        return sha256(self.content).hexdigest()

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "split": self.split,
            "object_key": self.object_key,
            "content_hash": self.content_hash,
            "size_bytes": len(self.content),
            "row_count": self.row_count,
            "schema_hash": self.schema_hash,
        }


@dataclass(frozen=True, slots=True)
class _SnapshotBuild:
    purpose: Literal["TRAINING", "EVALUATION_ONLY"]
    run_id: str
    snapshot_id: str
    input_manifest_hash: str
    manifest_key: str
    manifest_bytes: bytes
    manifest_hash: str
    quality_report_key: str
    split_counts: dict[str, int]
    source_work_order_ids: tuple[str, ...]
    artifacts: tuple[_Artifact, ...]

    @property
    def row_count(self) -> int:
        return sum(self.split_counts.values())


def prepare_project_authorized_dataset(
    database: Database,
    store: DatasetStore,
    context: TenantContext,
    *,
    catalog_path: Path,
    model_snapshot_path: Path,
    code_version: str,
) -> DatasetPreparationResult:
    """Generate, verify and register separately governed training and Gold snapshots."""

    if not code_version.strip() or len(code_version) > 128:
        raise GpuPromotionLabContractError("code_version_is_invalid")
    catalog = _load_catalog(catalog_path)
    _verify_model_snapshot(catalog.base_model, model_snapshot_path)
    cases = _generate_cases(catalog)
    _verify_split_isolation(cases)
    dataset_digest = _digest_json([_case_document(case) for case in cases])
    training = _build_snapshot(
        context,
        catalog=catalog,
        cases=tuple(case for case in cases if case.split in {"train", "validation"}),
        dataset_digest=dataset_digest,
        purpose="TRAINING",
        code_version=code_version,
    )
    gold = _build_snapshot(
        context,
        catalog=catalog,
        cases=tuple(case for case in cases if case.split == "gold"),
        dataset_digest=dataset_digest,
        purpose="EVALUATION_ONLY",
        code_version=code_version,
    )
    result = _result(catalog, dataset_digest, training, gold)
    existing = _existing_result(database, store, context, training, gold, result)
    if existing is not None:
        return existing
    for snapshot in (training, gold):
        for artifact in snapshot.artifacts:
            _put_immutable(store, artifact.object_key, artifact.content)
        _put_immutable(store, snapshot.manifest_key, snapshot.manifest_bytes)
    _persist_snapshots(
        database,
        context,
        training=training,
        gold=gold,
        catalog=catalog,
        dataset_digest=dataset_digest,
        code_version=code_version,
    )
    return result


def prepare_formal_gold_evaluation_snapshot(
    database: Database,
    store: DatasetStore,
    context: TenantContext,
    *,
    catalog_path: Path,
    model_snapshot_path: Path,
    code_version: str,
) -> FormalGoldPreparationResult:
    """Expand isolated Gold seeds into the minimum governed smoke cohort."""

    if not code_version.strip() or len(code_version) > 128:
        raise GpuPromotionLabContractError("code_version_is_invalid")
    catalog = _load_catalog(catalog_path)
    _verify_model_snapshot(catalog.base_model, model_snapshot_path)
    source_cases = _generate_cases(catalog)
    gold_seeds = tuple(case for case in source_cases if case.split == "gold")
    expanded = tuple(
        replace(
            case,
            case_id=f"{case.case_id}-cohort-{variant_index:02d}",
            observation=f"{variant}：{case.observation}",
        )
        for variant_index, variant in enumerate(_GOLD_VARIANTS, start=1)
        for case in gold_seeds
    )
    _verify_split_isolation(
        tuple(case for case in source_cases if case.split != "gold") + expanded
    )
    if not 30 <= len(expanded) <= 50:
        raise GpuPromotionLabContractError("formal_gold_smoke_cohort_size_is_invalid")
    dataset_digest = _digest_json([_case_document(case) for case in expanded])
    snapshot = _build_snapshot(
        context,
        catalog=catalog,
        cases=expanded,
        dataset_digest=dataset_digest,
        purpose="EVALUATION_ONLY",
        code_version=code_version,
    )
    if not _existing_snapshot(database, store, context, snapshot):
        for artifact in snapshot.artifacts:
            _put_immutable(store, artifact.object_key, artifact.content)
        _put_immutable(store, snapshot.manifest_key, snapshot.manifest_bytes)
        _persist_snapshot_records(
            database,
            context,
            snapshots=(snapshot,),
            catalog=catalog,
            dataset_digest=dataset_digest,
            code_version=code_version,
        )
    slice_counts = {
        f"risk:{risk}": sum(case.severity == risk for case in expanded)
        for risk in sorted({case.severity for case in expanded})
    }
    return FormalGoldPreparationResult(
        snapshot_id=snapshot.snapshot_id,
        manifest_hash=snapshot.manifest_hash,
        sample_count=len(expanded),
        slice_counts=slice_counts,
        dataset_digest=dataset_digest,
        classification=STAGING_CLASSIFICATION,
        data_classification=DATA_CLASSIFICATION,
        production_claim=False,
    )


class GpuPromotionGoldEvaluationBackend:
    """Run model inference and attach measured isolated-runtime gate evidence."""

    target_profile = "MODEL_COMPONENT"

    def __init__(self, model_backend: ModelEvaluationBackend | None = None) -> None:
        self._delegate = model_backend or TransformersModelEvaluationBackend()
        self.target_profile = self._delegate.target_profile
        self._runtime_probe_cache: dict[str, RuntimeProbeOutcome] = {}

    def evaluate(
        self,
        *,
        experiment: TrainingExperimentRecord,
        cases: tuple[EvaluationCase, ...],
        config: EvaluationRuntimeConfig,
        adapter_directory: Path | None,
    ) -> tuple[ModelObservation, ...]:
        observations = self._delegate.evaluate(
            experiment=experiment,
            cases=cases,
            config=config,
            adapter_directory=adapter_directory,
        )
        case_by_id = {case.case_id: case for case in cases}
        verified: list[ModelObservation] = []
        for observation in observations:
            case = case_by_id[observation.case_id]
            required = frozenset(case.required_gates)
            if "valid_citations" in required and not case.allowed_citations:
                raise GpuPromotionLabContractError(
                    "gold_valid_citation_gate_has_no_allowed_citation"
                )
            normalized = _normalized_model_json(observation.output_text)
            replay = (
                _normalized_model_json(observation.replay_output_text)
                if observation.replay_output_text is not None
                else None
            )
            citations = _model_citations(normalized)
            runtime = self._runtime_probe(case)
            capabilities = set(observation.capabilities).union(runtime.capabilities)
            gate_results = dict(runtime.gate_results)
            runtime_evidence = dict(runtime.evidence)
            if "valid_citations" in required:
                capabilities.add("valid_citations")
                citation_passed = bool(citations) and set(citations) <= set(case.allowed_citations)
                gate_results["valid_citations"] = citation_passed
                runtime_evidence["valid_citations"] = {
                    "citation_count": len(citations),
                    "allowed_count": len(case.allowed_citations),
                    "allowed_set_match": set(citations) <= set(case.allowed_citations),
                    "source": "FROZEN_DATASET_CITATION_BINDING",
                }
            if "replay_reproducibility" in required:
                capabilities.add("replay_reproducibility")
                replay_passed = bool(replay) and replay == normalized
                gate_results["replay_reproducibility"] = replay_passed
                runtime_evidence["replay_reproducibility"] = {
                    "normalized_output_sha256": sha256(normalized.encode()).hexdigest(),
                    "normalized_replay_sha256": (
                        sha256(replay.encode()).hexdigest() if replay else None
                    ),
                    "exact_match": replay_passed,
                }
            verified.append(
                replace(
                    observation,
                    output_text=normalized,
                    capabilities=frozenset(capabilities),
                    citations=citations,
                    proposed_tools=tuple(
                        sorted(set(observation.proposed_tools).union(runtime.proposed_tools))
                    ),
                    executed_tools=tuple(
                        sorted(set(observation.executed_tools).union(runtime.executed_tools))
                    ),
                    side_effect_count=observation.side_effect_count + runtime.side_effect_count,
                    approval_requested=(
                        observation.approval_requested or runtime.approval_requested
                    ),
                    tool_schema_valid=(observation.tool_schema_valid or runtime.tool_schema_valid),
                    decision=runtime.decision or observation.decision,
                    replay_output_text=replay,
                    runtime_gate_results=gate_results,
                    runtime_evidence={
                        **observation.runtime_evidence,
                        **runtime_evidence,
                        "model_output_contract": _output_contract_diagnostic(
                            case.expected_output, normalized
                        ),
                        "gpu_promotion_lab": {
                            "execution_mode": "ISOLATED_MODEL_PLUS_RUNTIME_PROBES",
                            "project_authorized_data": True,
                            "production_claim": False,
                            "measured_gates": sorted(gate_results),
                        },
                    },
                    structured_output_valid=_is_json_object(normalized),
                )
            )
        return tuple(verified)

    def _runtime_probe(self, case: EvaluationCase) -> RuntimeProbeOutcome:
        cached = self._runtime_probe_cache.get(case.case_id)
        if cached is not None:
            return cached
        probe = RuntimeProbeSandbox(case.case_id)
        try:
            outcome = probe.run(case)
        finally:
            probe.close()
        self._runtime_probe_cache[case.case_id] = outcome
        return outcome


def _load_catalog(path: Path) -> _Catalog:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GpuPromotionLabContractError("catalog_is_not_valid_json") from exc
    root = _object(document, "catalog_schema_is_closed")
    _closed(root, _ROOT_FIELDS, "catalog_schema_is_closed")
    if (
        root["schema_version"] != DATASET_PROTOCOL
        or not _text(root["dataset_id"])
        or not _text(root["version"])
    ):
        raise GpuPromotionLabContractError("catalog_identity_is_invalid")

    authorization = _object(root["authorization"], "catalog_schema_is_closed")
    _closed(authorization, _AUTHORIZATION_FIELDS, "catalog_schema_is_closed")
    _verify_authorization(authorization)
    generator = _object(root["generator"], "catalog_schema_is_closed")
    _closed(generator, _GENERATOR_FIELDS, "catalog_schema_is_closed")
    if (
        not _text(generator["id"])
        or not _text(generator["version"])
        or isinstance(generator["seed"], bool)
        or not isinstance(generator["seed"], int)
    ):
        raise GpuPromotionLabContractError("generator_identity_is_invalid")

    base_model = _object(root["base_model"], "catalog_schema_is_closed")
    _closed(base_model, _MODEL_FIELDS, "catalog_schema_is_closed")
    _verify_model_manifest(base_model)
    raw_scenarios = root["scenarios"]
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise GpuPromotionLabContractError("scenario_catalog_is_invalid")
    scenarios: list[dict[str, Any]] = []
    seen_scenario_ids: set[str] = set()
    for raw in raw_scenarios:
        scenario = _object(raw, "catalog_schema_is_closed")
        _closed(scenario, _SCENARIO_FIELDS, "catalog_schema_is_closed")
        scenario_id = _text(scenario["scenario_id"])
        split = _text(scenario["split"])
        if not _BOUNDARY_ID.fullmatch(scenario_id) or split not in SPLITS:
            raise GpuPromotionLabContractError("scenario_identity_is_invalid")
        if scenario_id in seen_scenario_ids:
            raise GpuPromotionLabContractError("split_case_identity_leakage")
        seen_scenario_ids.add(scenario_id)
        for field_name in (
            "equipment_family",
            "alarm_code",
            "root_cause_code",
            "root_cause",
            "severity",
        ):
            if not _text(scenario[field_name]):
                raise GpuPromotionLabContractError("scenario_content_is_invalid")
        for field_name in ("observations", "inspection_steps", "prohibited_actions"):
            _string_list(scenario[field_name], field_name)
        scenarios.append(scenario)
    if {scenario["split"] for scenario in scenarios} != set(SPLITS):
        raise GpuPromotionLabContractError("all_dataset_splits_are_required")
    return _Catalog(
        document=root,
        authorization=authorization,
        base_model=base_model,
        scenarios=tuple(scenarios),
        digest=_digest_json(root),
        authorization_digest=_digest_json(authorization),
        model_manifest_digest=_digest_json(base_model),
    )


def _verify_authorization(value: Mapping[str, Any]) -> None:
    allowed = _string_list(value["allowed_uses"], "allowed_uses")
    prohibited = _string_list(value["prohibited_uses"], "prohibited_uses")
    if (
        not _text(value["owner"])
        or value["classification"] != AUTHORIZATION_CLASSIFICATION
        or value["data_classification"] != DATA_CLASSIFICATION
        or set(allowed) != {"LOCAL_TRAINING", "LOCAL_EVALUATION", "LOCAL_DEMONSTRATION"}
        or "CUSTOMER_PRODUCTION_CLAIM" not in prohibited
        or value["contains_customer_data"] is not False
        or value["contains_personal_data"] is not False
        or value["production_claim"] is not False
    ):
        raise GpuPromotionLabContractError("project_authorization_is_invalid")


def _verify_model_manifest(value: Mapping[str, Any]) -> None:
    if (
        value["model_id"] != BASE_MODEL_ID
        or value["revision"] != BASE_MODEL_REVISION
        or value["license"] != BASE_MODEL_LICENSE
        or value["remote_code_allowed"] is not False
        or not _digest(value["chat_template_sha256"])
    ):
        raise GpuPromotionLabContractError("base_model_identity_is_not_fixed")
    raw_files = value["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise GpuPromotionLabContractError("base_model_file_manifest_is_invalid")
    observed: set[str] = set()
    for raw in raw_files:
        item = _object(raw, "catalog_schema_is_closed")
        _closed(item, _MODEL_FILE_FIELDS, "catalog_schema_is_closed")
        filename = _text(item["path"])
        pure_path = PurePosixPath(filename)
        if (
            not filename
            or pure_path.is_absolute()
            or len(pure_path.parts) != 1
            or filename in observed
            or isinstance(item["size_bytes"], bool)
            or not isinstance(item["size_bytes"], int)
            or item["size_bytes"] <= 0
            or not _digest(item["sha256"])
        ):
            raise GpuPromotionLabContractError("base_model_file_manifest_is_invalid")
        observed.add(filename)
    if observed != _MODEL_FILES:
        raise GpuPromotionLabContractError("base_model_file_manifest_is_incomplete")


def _verify_model_snapshot(manifest: Mapping[str, Any], snapshot_path: Path) -> None:
    if snapshot_path.name != BASE_MODEL_REVISION or not snapshot_path.is_dir():
        raise GpuPromotionLabContractError("base_model_revision_path_mismatch")
    files = cast(list[dict[str, Any]], manifest["files"])
    for identity in files:
        target = snapshot_path / str(identity["path"])
        if (
            not target.is_file()
            or target.stat().st_size != int(identity["size_bytes"])
            or _digest_file(target) != str(identity["sha256"])
        ):
            raise GpuPromotionLabContractError("base_model_file_identity_mismatch")
    try:
        config = _object(
            json.loads((snapshot_path / "config.json").read_text(encoding="utf-8")),
            "base_model_config_is_invalid",
        )
        tokenizer = _object(
            json.loads((snapshot_path / "tokenizer_config.json").read_text(encoding="utf-8")),
            "tokenizer_config_is_invalid",
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GpuPromotionLabContractError("base_model_config_is_invalid") from exc
    if config.get("auto_map") is not None or config.get("trust_remote_code") is True:
        raise GpuPromotionLabContractError("remote_model_code_is_forbidden")
    chat_template = tokenizer.get("chat_template")
    if (
        not isinstance(chat_template, str)
        or sha256(chat_template.encode()).hexdigest() != manifest["chat_template_sha256"]
    ):
        raise GpuPromotionLabContractError("chat_template_identity_mismatch")


def _generate_cases(catalog: _Catalog) -> tuple[_Case, ...]:
    cases: list[_Case] = []
    for scenario in catalog.scenarios:
        observations = _string_list(scenario["observations"], "observations")
        for index, observation in enumerate(observations, start=1):
            split = cast(Literal["train", "validation", "gold"], scenario["split"])
            scenario_id = str(scenario["scenario_id"])
            cases.append(
                _Case(
                    case_id=f"gpu-{split}-{scenario_id}-{index:02d}",
                    split=split,
                    scenario_id=scenario_id,
                    equipment_family=str(scenario["equipment_family"]),
                    alarm_code=str(scenario["alarm_code"]),
                    observation=observation,
                    root_cause_code=str(scenario["root_cause_code"]),
                    root_cause=str(scenario["root_cause"]),
                    inspection_steps=tuple(
                        _string_list(scenario["inspection_steps"], "inspection_steps")
                    ),
                    prohibited_actions=tuple(
                        _string_list(scenario["prohibited_actions"], "prohibited_actions")
                    ),
                    severity=str(scenario["severity"]),
                )
            )
    counts = {split: sum(case.split == split for case in cases) for split in SPLITS}
    if any(count <= 0 for count in counts.values()):
        raise GpuPromotionLabContractError("all_dataset_splits_must_be_non_empty")
    return tuple(cases)


def _training_label_taxonomy(catalog: _Catalog) -> dict[str, str]:
    taxonomy: dict[str, str] = {}
    for scenario in catalog.scenarios:
        if scenario["split"] != "train":
            continue
        code = str(scenario["root_cause_code"])
        description = str(scenario["root_cause"])
        existing = taxonomy.get(code)
        if existing is not None and existing != description:
            raise GpuPromotionLabContractError(
                "training_label_taxonomy_has_conflicting_definitions"
            )
        taxonomy[code] = description
    if len(taxonomy) < 2:
        raise GpuPromotionLabContractError("training_label_taxonomy_is_incomplete")
    return dict(sorted(taxonomy.items()))


def _verify_split_isolation(cases: tuple[_Case, ...]) -> None:
    ids_by_split = {
        split: {case.case_id for case in cases if case.split == split} for split in SPLITS
    }
    for left_index, left_split in enumerate(SPLITS):
        for right_split in SPLITS[left_index + 1 :]:
            if ids_by_split[left_split] & ids_by_split[right_split]:
                raise GpuPromotionLabContractError("split_case_identity_leakage")
    normalized = {case.case_id: _normalize(case.leakage_text()) for case in cases}
    if len(set(normalized.values())) != len(normalized):
        raise GpuPromotionLabContractError("split_text_duplicate_leakage")
    for left_index, left in enumerate(cases):
        left_grams = _ngrams(normalized[left.case_id])
        for right in cases[left_index + 1 :]:
            if left.split == right.split:
                continue
            right_grams = _ngrams(normalized[right.case_id])
            union = left_grams | right_grams
            similarity = len(left_grams & right_grams) / len(union) if union else 1.0
            if similarity >= 0.90:
                raise GpuPromotionLabContractError("cross_split_near_duplicate_leakage")


def _build_snapshot(
    context: TenantContext,
    *,
    catalog: _Catalog,
    cases: tuple[_Case, ...],
    dataset_digest: str,
    purpose: Literal["TRAINING", "EVALUATION_ONLY"],
    code_version: str,
) -> _SnapshotBuild:
    formal_rows = []
    label_taxonomy = _training_label_taxonomy(catalog)
    for index, case in enumerate(cases):
        formal_split = "test" if case.split == "gold" else case.split
        formal_rows.append(
            _row(
                context.tenant_id,
                case,
                formal_split,
                index=index,
                label_taxonomy=label_taxonomy,
            )
        )
    split_counts = {
        split: sum(row["split"] == split for row in formal_rows) for split in FORMAL_SPLITS
    }
    expected = (
        split_counts["train"] > 0
        and split_counts["validation"] > 0
        and split_counts["test"] == 0
        if purpose == "TRAINING"
        else split_counts == {"train": 0, "validation": 0, "test": len(cases)}
    )
    if not expected:
        raise GpuPromotionLabContractError("snapshot_purpose_split_contract_failed")
    snapshot_digest = _digest_json(
        {
            "tenant_id": context.tenant_id,
            "purpose": purpose,
            "dataset_digest": dataset_digest,
            "catalog_digest": catalog.digest,
            "model_manifest_digest": catalog.model_manifest_digest,
            "code_version": code_version,
            "rows": formal_rows,
        }
    )
    label = "train" if purpose == "TRAINING" else "gold"
    snapshot_id = f"dataset-gpu-lab-{label}-{snapshot_digest[:20]}"
    run_id = str(uuid5(NAMESPACE_URL, f"{context.tenant_id}:{snapshot_id}:curation"))
    prefix = f"tenant/{context.tenant_id}/datasets/snapshots/{snapshot_id}/published"
    schema_hash = "sha256:" + sha256("\0".join(DATASET_COLUMNS).encode()).hexdigest()
    artifacts: list[_Artifact] = []
    for split in FORMAL_SPLITS:
        rows = [row for row in formal_rows if row["split"] == split]
        artifacts.append(
            _Artifact(
                kind="parquet",
                split=split,
                object_key=f"{prefix}/{split}.parquet",
                content=_parquet_bytes(rows),
                row_count=len(rows),
                schema_hash=schema_hash,
            )
        )
    quality = {
        "contract_version": CONTRACT_VERSION,
        "pandera_validation": "PASSED",
        "row_count": len(formal_rows),
        "split_counts": split_counts,
        "group_leakage_count": 0,
        "exclusions": [],
    }
    quality_key = f"{prefix}/quality-report.json"
    artifacts.append(
        _Artifact(
            kind="quality_report",
            split=None,
            object_key=quality_key,
            content=_json_bytes(quality),
            row_count=len(formal_rows),
            schema_hash="sha256:" + sha256(CONTRACT_VERSION.encode()).hexdigest(),
        )
    )
    authorization = {
        "schema_version": DATASET_PROTOCOL,
        "snapshot_id": snapshot_id,
        "purpose": purpose,
        "owner": catalog.authorization["owner"],
        "authorization_classification": AUTHORIZATION_CLASSIFICATION,
        "data_classification": DATA_CLASSIFICATION,
        "classification": STAGING_CLASSIFICATION,
        "source_catalog_sha256": catalog.digest,
        "dataset_sha256": dataset_digest,
        "model_manifest_sha256": catalog.model_manifest_digest,
        "case_id_sha256": _digest_json(sorted(case.case_id for case in cases)),
        "excluded_split": "gold" if purpose == "TRAINING" else "train_and_validation",
        "contains_customer_data": False,
        "contains_personal_data": False,
        "production_claim": False,
    }
    authorization_key = f"{prefix}/authorization-manifest.json"
    artifacts.append(
        _Artifact(
            kind="authorization_manifest",
            split=None,
            object_key=authorization_key,
            content=_json_bytes(authorization),
            row_count=len(formal_rows),
            schema_hash="sha256:" + sha256(DATASET_PROTOCOL.encode()).hexdigest(),
        )
    )
    input_manifest_hash = "sha256:" + _digest_json(formal_rows)
    source_work_order_ids = tuple(
        sorted({str(row["work_order_id"]) for row in formal_rows})
    )
    manifest = {
        "snapshot_id": snapshot_id,
        "run_id": run_id,
        "tenant_id": context.tenant_id,
        "contract_version": CONTRACT_VERSION,
        "code_version": code_version,
        "input_manifest_hash": input_manifest_hash,
        "purpose": purpose,
        "authorization_classification": AUTHORIZATION_CLASSIFICATION,
        "data_classification": DATA_CLASSIFICATION,
        "production_claim": False,
        "row_count": len(formal_rows),
        "split_counts": split_counts,
        "source_work_order_ids": list(source_work_order_ids),
        "lineage_status": "CONFIRMED",
        "artifacts": [artifact.manifest_entry() for artifact in artifacts],
    }
    manifest_bytes = _json_bytes(manifest)
    return _SnapshotBuild(
        purpose=purpose,
        run_id=run_id,
        snapshot_id=snapshot_id,
        input_manifest_hash=input_manifest_hash,
        manifest_key=f"{prefix}/manifest.json",
        manifest_bytes=manifest_bytes,
        manifest_hash="sha256:" + sha256(manifest_bytes).hexdigest(),
        quality_report_key=quality_key,
        split_counts=split_counts,
        source_work_order_ids=source_work_order_ids,
        artifacts=tuple(artifacts),
    )


def _row(
    tenant_id: str,
    case: _Case,
    formal_split: str,
    *,
    index: int,
    label_taxonomy: dict[str, str],
) -> dict[str, Any]:
    runtime_gate = _EVALUATION_RUNTIME_GATES[index % len(_EVALUATION_RUNTIME_GATES)]
    citation_id = f"project://gpu-model-promotion-lab/{case.case_id}"
    allowed_citations = [citation_id] if runtime_gate == "valid_citations" else []
    evidence = {
        "equipment_family": case.equipment_family,
        "alarm_code": case.alarm_code,
        "observation": case.observation,
        "source": "PROJECT_GENERATED_SYNTHETIC",
    }
    if runtime_gate == "valid_citations":
        evidence["citation_id"] = citation_id
    labels = {
        "root_cause_code": case.root_cause_code,
        "_classification_contract": {
            "schema_version": CLASSIFICATION_CONTRACT_VERSION,
            "output_key": "root_cause_code",
            "labels": label_taxonomy,
        },
        "_evaluation": {
            "slices": {
                "equipment_family": case.equipment_family,
                "alarm_code": case.alarm_code,
                "risk": case.severity,
            },
            "risk": case.severity,
            "required_gates": [runtime_gate],
            "allowed_citations": allowed_citations,
            "forbidden_substrings": ["客户现场", "真实序列号", "绕过联锁"],
            "expected_decision": case.root_cause_code,
            "contexts": [STAGING_CLASSIFICATION, DATA_CLASSIFICATION],
        },
    }
    if runtime_gate == "valid_citations":
        labels["citations"] = allowed_citations
    occurred_at = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index)
    group_key = "group:" + sha256(f"{tenant_id}\0{case.case_id}".encode()).hexdigest()
    source_hash = "sha256:" + sha256(_json_bytes(evidence)).hexdigest()
    row: dict[str, Any] = {
        "tenant_id": tenant_id,
        "candidate_id": case.case_id,
        "candidate_version": 1,
        "work_order_id": f"synthetic-work-order-{case.case_id}",
        "incident_id": f"synthetic-incident-{case.case_id}",
        "device_id": f"synthetic-device-{case.scenario_id}",
        "source_event_id": f"synthetic-event-{case.case_id}",
        "source_content_hash": source_hash,
        "redacted_content_json": _json_text(evidence),
        "labels_json": _json_text(labels),
        "lineage_json": _json_text(
            {
                "origin": "PROJECT_GENERATED_SYNTHETIC",
                "scenario_id": case.scenario_id,
                "source_case_id": case.case_id,
                "production_claim": False,
            }
        ),
        "occurred_at": occurred_at.isoformat(),
        "near_duplicate_group": f"synthetic:{case.case_id}",
        "group_key": group_key,
        "split": formal_split,
    }
    return {name: row[name] for name in DATASET_COLUMNS}


def _result(
    catalog: _Catalog,
    dataset_digest: str,
    training: _SnapshotBuild,
    gold: _SnapshotBuild,
) -> DatasetPreparationResult:
    return DatasetPreparationResult(
        training_snapshot_id=training.snapshot_id,
        training_manifest_hash=training.manifest_hash,
        gold_snapshot_id=gold.snapshot_id,
        gold_manifest_hash=gold.manifest_hash,
        dataset_digest=dataset_digest,
        authorization_digest=catalog.authorization_digest,
        model_manifest_digest=catalog.model_manifest_digest,
        classification=STAGING_CLASSIFICATION,
        data_classification=DATA_CLASSIFICATION,
        authorization_classification=AUTHORIZATION_CLASSIFICATION,
        production_claim=False,
    )


def _existing_result(
    database: Database,
    store: DatasetStore,
    context: TenantContext,
    training: _SnapshotBuild,
    gold: _SnapshotBuild,
    result: DatasetPreparationResult,
) -> DatasetPreparationResult | None:
    expected = {training.snapshot_id: training, gold.snapshot_id: gold}
    with database.transaction(context) as session:
        records = list(
            session.scalars(
                select(DatasetSnapshotRecord).where(
                    DatasetSnapshotRecord.tenant_id == context.tenant_id,
                    DatasetSnapshotRecord.snapshot_id.in_(tuple(expected)),
                )
            )
        )
    if not records:
        return None
    if len(records) != 2:
        raise GpuPromotionLabContractError("dataset_snapshot_idempotency_state_is_partial")
    for record in records:
        planned = expected[record.snapshot_id]
        if (
            record.status != "CANDIDATE"
            or record.contract_version != CONTRACT_VERSION
            or record.manifest_key != planned.manifest_key
            or record.manifest_hash != planned.manifest_hash
            or record.row_count != planned.row_count
            or record.split_counts != planned.split_counts
            or record.lineage_status != "CONFIRMED"
            or not record.training_eligible
            or sha256(store.get_bytes(planned.manifest_key)).hexdigest()
            != planned.manifest_hash.removeprefix("sha256:")
        ):
            raise GpuPromotionLabContractError("dataset_snapshot_idempotency_binding_changed")
    return result


def _persist_snapshots(
    database: Database,
    context: TenantContext,
    *,
    training: _SnapshotBuild,
    gold: _SnapshotBuild,
    catalog: _Catalog,
    dataset_digest: str,
    code_version: str,
) -> None:
    _persist_snapshot_records(
        database,
        context,
        snapshots=(training, gold),
        catalog=catalog,
        dataset_digest=dataset_digest,
        code_version=code_version,
    )


def _persist_snapshot_records(
    database: Database,
    context: TenantContext,
    *,
    snapshots: tuple[_SnapshotBuild, ...],
    catalog: _Catalog,
    dataset_digest: str,
    code_version: str,
) -> None:
    now = datetime.now(UTC)
    with database.transaction(context) as session:
        for snapshot in snapshots:
            session.add(
                CurationRunRecord(
                    run_id=snapshot.run_id,
                    tenant_id=context.tenant_id,
                    window_start=datetime(2026, 1, 1, tzinfo=UTC),
                    window_end=datetime(2027, 1, 1, tzinfo=UTC),
                    engine="project-authorized-generator",
                    contract_version=CONTRACT_VERSION,
                    code_version=code_version,
                    config_hash="sha256:"
                    + _digest_json(
                        {
                            "catalog": catalog.digest,
                            "model": catalog.model_manifest_digest,
                            "dataset": dataset_digest,
                            "purpose": snapshot.purpose,
                        }
                    ),
                    input_manifest_hash=snapshot.input_manifest_hash,
                    input_count=snapshot.row_count,
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
                    snapshot_id=snapshot.snapshot_id,
                    tenant_id=context.tenant_id,
                    run_id=snapshot.run_id,
                    status="CANDIDATE",
                    contract_version=CONTRACT_VERSION,
                    input_manifest_hash=snapshot.input_manifest_hash,
                    manifest_key=snapshot.manifest_key,
                    manifest_hash=snapshot.manifest_hash,
                    quality_report_key=snapshot.quality_report_key,
                    row_count=snapshot.row_count,
                    split_counts=snapshot.split_counts,
                    source_work_order_ids=list(snapshot.source_work_order_ids),
                    lineage_status="CONFIRMED",
                    training_eligible=True,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            for artifact in snapshot.artifacts:
                session.add(
                    DatasetArtifactRecord(
                        artifact_id="artifact-"
                        + sha256(
                            f"{snapshot.snapshot_id}\0{artifact.object_key}".encode()
                        ).hexdigest()[:32],
                        tenant_id=context.tenant_id,
                        snapshot_id=snapshot.snapshot_id,
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
                    lineage_id="lineage-"
                    + sha256(
                        f"{snapshot.run_id}\0{snapshot.snapshot_id}".encode()
                    ).hexdigest()[:32],
                    tenant_id=context.tenant_id,
                    run_id=snapshot.run_id,
                    snapshot_id=snapshot.snapshot_id,
                    openlineage_run_id=snapshot.run_id,
                    job_namespace="industrial-ops-local-staging",
                    job_name="build_project_authorized_gpu_dataset",
                    status="CONFIRMED",
                    source_work_order_ids=list(snapshot.source_work_order_ids),
                    output_dataset=snapshot.manifest_key,
                    failure_reason=None,
                    created_at=now,
                    updated_at=now,
                )
            )


def _existing_snapshot(
    database: Database,
    store: DatasetStore,
    context: TenantContext,
    snapshot: _SnapshotBuild,
) -> bool:
    with database.transaction(context) as session:
        record = session.scalar(
            select(DatasetSnapshotRecord).where(
                DatasetSnapshotRecord.tenant_id == context.tenant_id,
                DatasetSnapshotRecord.snapshot_id == snapshot.snapshot_id,
            )
        )
    if record is None:
        return False
    if (
        record.status != "CANDIDATE"
        or record.contract_version != CONTRACT_VERSION
        or record.manifest_key != snapshot.manifest_key
        or record.manifest_hash != snapshot.manifest_hash
        or record.row_count != snapshot.row_count
        or record.split_counts != snapshot.split_counts
        or record.lineage_status != "CONFIRMED"
        or not record.training_eligible
        or sha256(store.get_bytes(snapshot.manifest_key)).hexdigest()
        != snapshot.manifest_hash.removeprefix("sha256:")
    ):
        raise GpuPromotionLabContractError(
            "dataset_snapshot_idempotency_binding_changed"
        )
    return True


def _put_immutable(store: DatasetStore, object_key: str, content: bytes) -> None:
    try:
        store.put_bytes(object_key, content)
    except ImmutableObjectExists:
        try:
            existing = store.get_bytes(object_key)
        except Exception as exc:
            raise GpuPromotionLabContractError("immutable_dataset_object_is_unreadable") from exc
        if existing != content:
            raise GpuPromotionLabContractError(
                "immutable_dataset_object_binding_changed"
            ) from None


def _parquet_bytes(rows: list[dict[str, Any]]) -> bytes:
    fields = [
        arrow.field(name, arrow.int64() if name == "candidate_version" else arrow.string())
        for name in DATASET_COLUMNS
    ]
    table = arrow.Table.from_pylist(rows, schema=arrow.schema(fields))
    target = BytesIO()
    parquet.write_table(table, target, compression="zstd")
    return target.getvalue()


def _case_document(case: _Case) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "split": case.split,
        "scenario_id": case.scenario_id,
        "equipment_family": case.equipment_family,
        "alarm_code": case.alarm_code,
        "observation": case.observation,
        "root_cause_code": case.root_cause_code,
        "root_cause": case.root_cause,
        "inspection_steps": list(case.inspection_steps),
        "prohibited_actions": list(case.prohibited_actions),
        "severity": case.severity,
    }


def _closed(value: Mapping[str, Any], expected: frozenset[str], error: str) -> None:
    if set(value) != expected:
        raise GpuPromotionLabContractError(error)


def _object(value: Any, error: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise GpuPromotionLabContractError(error)
    return cast(dict[str, Any], value)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: Any, field_name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(value) != len(set(value))
    ):
        raise GpuPromotionLabContractError(f"{field_name}_is_invalid")
    return cast(list[str], value)


def _digest(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _digest_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_json(value: Any) -> str:
    return sha256(_json_bytes(value)).hexdigest()


def _normalized_model_json(value: str | None) -> str:
    if value is None:
        return ""
    candidate = value.strip()
    if "</think>" in candidate:
        candidate = candidate.rsplit("</think>", 1)[1].strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].lstrip()
    decoder = json.JSONDecoder()
    for offset, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return _json_text(parsed)
    return candidate


def _model_citations(value: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return ()
    if not isinstance(parsed, dict):
        return ()
    citations = parsed.get("citations")
    if not isinstance(citations, list):
        return ()
    return tuple(item.strip() for item in citations if isinstance(item, str) and item.strip())


def _output_contract_diagnostic(expected: Mapping[str, Any], value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    actual = parsed if isinstance(parsed, dict) else {}
    return {
        "valid_json_object": isinstance(parsed, dict),
        "expected_keys": sorted(expected),
        "actual_keys": sorted(actual),
        "missing_keys": sorted(set(expected) - set(actual)),
        "unexpected_keys": sorted(set(actual) - set(expected)),
        "mismatched_keys": sorted(
            key for key, expected_value in expected.items() if actual.get(key) != expected_value
        ),
        "normalized_output_sha256": sha256(value.encode()).hexdigest(),
    }


def _is_json_object(value: str) -> bool:
    try:
        return isinstance(json.loads(value), dict)
    except json.JSONDecodeError:
        return False


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_bytes(value: Any) -> bytes:
    return _json_text(value).encode()


def _normalize(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _ngrams(value: str, width: int = 5) -> set[str]:
    if len(value) <= width:
        return {value}
    return {value[index : index + width] for index in range(len(value) - width + 1)}
