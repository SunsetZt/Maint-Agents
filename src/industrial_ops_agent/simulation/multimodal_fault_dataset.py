"""Build and verify a deterministic non-production multimodal fault dataset."""

from __future__ import annotations

import io
import json
import math
import random
import shutil
import struct
import tempfile
import wave
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, ValidationError

CLASSIFICATION = "SIMULATED_NON_PRODUCTION"
DATASET_SCHEMA_VERSION = "simulated-multimodal-fault-dataset/v1"
CASE_SCHEMA_VERSION = "simulated-multimodal-fault-case/v1"
FREEZE_SCHEMA_VERSION = "simulated-multimodal-evaluation-freeze/v1"
DEFAULT_SEED = 20260822
DEFAULT_CASES_PER_SPLIT = 6
MAX_CASES_PER_SPLIT = 200
SPLITS = ("train", "validation", "simulation_evaluation")
MODALITIES = ("image", "acoustic", "telemetry", "engineer_note", "knowledge")


class MultimodalFaultDatasetError(RuntimeError):
    """The generated dataset is unsafe, inconsistent, or has been modified."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactReference(_ClosedModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1)
    media_type: str


class ExpectedFinding(_ClosedModel):
    label: str
    region: dict[str, float]


class CaseGovernance(_ClosedModel):
    source_type: Literal["PROJECT_GENERATED"] = "PROJECT_GENERATED"
    review_status: Literal["SIMULATED_AUTO_LABEL"] = "SIMULATED_AUTO_LABEL"
    training_candidate_eligible: bool
    simulation_evaluation_eligible: bool
    production_evaluation_eligible: Literal[False] = False
    evidence_eligible: Literal[False] = False


class MultimodalFaultCase(_ClosedModel):
    schema_version: Literal["simulated-multimodal-fault-case/v1"] = (
        "simulated-multimodal-fault-case/v1"
    )
    classification: Literal["SIMULATED_NON_PRODUCTION"] = "SIMULATED_NON_PRODUCTION"
    synthetic: Literal[True] = True
    case_id: str
    split: Literal["train", "validation", "simulation_evaluation"]
    asset_id: str
    asset_model: str
    site_id: str
    fault_code: str
    severity: Literal["MEDIUM", "HIGH", "CRITICAL"]
    image: ArtifactReference
    acoustic: ArtifactReference
    telemetry: ArtifactReference
    engineer_note: str
    knowledge: ArtifactReference
    target: dict[str, Any]
    governance: CaseGovernance


class DatasetBuildResult(_ClosedModel):
    output_dir: str
    dataset_id: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_freeze_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_counts: dict[str, int]
    artifact_count: int
    created: bool
    status: Literal["SIMULATED_MULTIMODAL_FAULT_DATASET_VERIFIED"] = (
        "SIMULATED_MULTIMODAL_FAULT_DATASET_VERIFIED"
    )
    classification: Literal["SIMULATED_NON_PRODUCTION"] = "SIMULATED_NON_PRODUCTION"


@dataclass(frozen=True, slots=True)
class _FaultDefinition:
    code: str
    asset_model: str
    severity: Literal["MEDIUM", "HIGH", "CRITICAL"]
    root_cause: str
    visual_label: str
    region: tuple[float, float, float, float]
    acoustic_hz: float
    action: str
    part_number: str
    engineer_note: str
    telemetry_bias: tuple[float, float, float, float, float]


_FAULTS = (
    _FaultDefinition(
        code="PUMP_SEAL_LEAK",
        asset_model="PUMP-X100",
        severity="HIGH",
        root_cause="mechanical_seal_degradation",
        visual_label="oil_leak_near_mechanical_seal",
        region=(0.48, 0.53, 0.22, 0.30),
        acoustic_hz=182.0,
        action="isolate_pump_and_replace_mechanical_seal",
        part_number="SEAL-X100-04",
        engineer_note="泵体联轴器侧出现少量油液，出口压力下降并伴随周期性振动。",
        telemetry_bias=(11.0, 4.8, -0.9, 1.6, -1.8),
    ),
    _FaultDefinition(
        code="MOTOR_BEARING_OVERHEAT",
        asset_model="MOTOR-M220",
        severity="CRITICAL",
        root_cause="drive_end_bearing_lubrication_failure",
        visual_label="overheated_drive_end_bearing",
        region=(0.35, 0.31, 0.24, 0.30),
        acoustic_hz=310.0,
        action="stop_motor_and_replace_drive_end_bearing",
        part_number="BRG-M220-DE",
        engineer_note="电机驱动端温度持续升高，现场听到高频摩擦声，电流波动扩大。",
        telemetry_bias=(28.0, 7.5, 0.1, 4.4, 0.0),
    ),
    _FaultDefinition(
        code="COMPRESSOR_FILTER_BLOCKAGE",
        asset_model="COMP-C80",
        severity="HIGH",
        root_cause="intake_filter_excessive_pressure_drop",
        visual_label="contaminated_intake_filter",
        region=(0.67, 0.20, 0.20, 0.44),
        acoustic_hz=126.0,
        action="replace_intake_filter_and_verify_pressure_drop",
        part_number="FLT-C80-IN",
        engineer_note="压缩机吸气侧压差偏高，排气流量下降，滤芯外观积尘明显。",
        telemetry_bias=(13.0, 2.1, -1.7, 3.2, -3.6),
    ),
    _FaultDefinition(
        code="CONVEYOR_BELT_MISALIGNMENT",
        asset_model="CONV-B50",
        severity="MEDIUM",
        root_cause="return_idler_alignment_drift",
        visual_label="belt_edge_tracking_offset",
        region=(0.19, 0.50, 0.61, 0.22),
        acoustic_hz=94.0,
        action="lockout_conveyor_and_realign_return_idlers",
        part_number="IDLER-B50-R",
        engineer_note="输送带向驱动侧跑偏，边缘有轻微擦痕，回程托辊附近存在重复异响。",
        telemetry_bias=(6.0, 3.9, 0.0, 2.0, -1.1),
    ),
    _FaultDefinition(
        code="GEARBOX_TOOTH_DAMAGE",
        asset_model="GEAR-G90",
        severity="CRITICAL",
        root_cause="second_stage_gear_tooth_spalling",
        visual_label="metallic_debris_at_inspection_port",
        region=(0.43, 0.35, 0.23, 0.28),
        acoustic_hz=246.0,
        action="stop_gearbox_and_inspect_second_stage_gearset",
        part_number="GEAR-G90-S2",
        engineer_note="齿轮箱在固定转角出现冲击，检查口附近发现细小金属屑，油温上升。",
        telemetry_bias=(19.0, 8.3, -0.2, 2.8, -0.4),
    ),
    _FaultDefinition(
        code="VALVE_ACTUATOR_STICTION",
        asset_model="VALVE-V40",
        severity="MEDIUM",
        root_cause="actuator_stem_friction_and_stiction",
        visual_label="actuator_position_deviation",
        region=(0.38, 0.18, 0.26, 0.39),
        acoustic_hz=158.0,
        action="isolate_valve_and_service_actuator_stem",
        part_number="ACT-V40-KIT",
        engineer_note="调节阀指令变化后阀位响应迟滞，执行机构有短时抖动，流量呈阶跃变化。",
        telemetry_bias=(8.0, 2.8, -0.6, 1.1, -2.2),
    ),
)


def build_multimodal_fault_dataset(
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    cases_per_split: int = DEFAULT_CASES_PER_SPLIT,
    train_cases: int | None = None,
    validation_cases: int | None = None,
    evaluation_cases: int | None = None,
) -> DatasetBuildResult:
    """Build once, or verify and reuse an identical immutable output directory."""

    split_case_counts = _split_case_counts(
        cases_per_split=cases_per_split,
        train_cases=train_cases,
        validation_cases=validation_cases,
        evaluation_cases=evaluation_cases,
    )
    if seed < 0:
        raise MultimodalFaultDatasetError("dataset_generation_configuration_invalid")
    output = output_dir.resolve()
    if output.exists():
        result = verify_multimodal_fault_dataset(output)
        manifest = _load_object(output / "manifest.json", "dataset_manifest_is_invalid")
        if (
            manifest.get("seed") != seed
            or _manifest_split_case_counts(manifest) != split_case_counts
        ):
            raise MultimodalFaultDatasetError("dataset_output_exists_with_different_configuration")
        return result.model_copy(update={"created": False})

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _build_into(temporary, seed=seed, split_case_counts=split_case_counts)
        verify_multimodal_fault_dataset(temporary)
        temporary.rename(output)
    except Exception:
        _safe_remove_temporary(temporary, output.parent)
        raise
    result = verify_multimodal_fault_dataset(output)
    return result.model_copy(update={"created": True})


def verify_multimodal_fault_dataset(output_dir: Path) -> DatasetBuildResult:
    root = output_dir.resolve(strict=True)
    if not root.is_dir():
        raise MultimodalFaultDatasetError("dataset_output_is_not_a_directory")
    manifest_path = root / "manifest.json"
    manifest = _load_object(manifest_path, "dataset_manifest_is_invalid")
    observed_manifest_hash = _document_hash(manifest, "manifest_sha256")
    if (
        manifest.get("schema_version") != DATASET_SCHEMA_VERSION
        or manifest.get("classification") != CLASSIFICATION
        or manifest.get("synthetic") is not True
        or manifest.get("production_claim") is not False
        or manifest.get("production_gold_eligible") is not False
        or manifest.get("manifest_sha256") != observed_manifest_hash
    ):
        raise MultimodalFaultDatasetError("dataset_manifest_identity_or_digest_invalid")

    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise MultimodalFaultDatasetError("dataset_artifact_catalog_is_invalid")
    artifact_catalog: dict[str, dict[str, Any]] = {}
    for raw in raw_artifacts:
        if not isinstance(raw, dict):
            raise MultimodalFaultDatasetError("dataset_artifact_catalog_is_invalid")
        relative = _safe_relative_path(raw.get("path"))
        if relative in artifact_catalog:
            raise MultimodalFaultDatasetError("dataset_artifact_path_is_duplicated")
        payload = (root / relative).read_bytes()
        if raw.get("sha256") != _digest(payload) or raw.get("size_bytes") != len(payload):
            raise MultimodalFaultDatasetError("dataset_artifact_integrity_failed")
        artifact_catalog[relative] = raw

    cases_by_split: dict[str, list[MultimodalFaultCase]] = {}
    all_case_ids: set[str] = set()
    assets_by_split: dict[str, set[str]] = {}
    for split in SPLITS:
        case_path = f"cases/{split}.jsonl"
        if case_path not in artifact_catalog:
            raise MultimodalFaultDatasetError("dataset_split_catalog_is_incomplete")
        cases = _parse_cases(root / case_path)
        if not cases or any(case.split != split for case in cases):
            raise MultimodalFaultDatasetError("dataset_split_contains_foreign_cases")
        identifiers = {case.case_id for case in cases}
        if len(identifiers) != len(cases) or all_case_ids & identifiers:
            raise MultimodalFaultDatasetError("dataset_case_identity_is_duplicated")
        all_case_ids.update(identifiers)
        assets_by_split[split] = {case.asset_id for case in cases}
        cases_by_split[split] = cases
        for case in cases:
            _verify_case_artifacts(case, root, artifact_catalog)

    split_counts = {split: len(cases_by_split[split]) for split in SPLITS}
    if manifest.get("split_counts") != split_counts or manifest.get("case_count") != sum(
        split_counts.values()
    ):
        raise MultimodalFaultDatasetError("dataset_split_counts_are_invalid")
    if _manifest_split_case_counts(manifest) != split_counts:
        raise MultimodalFaultDatasetError("dataset_generation_counts_are_invalid")
    if any(
        assets_by_split[left] & assets_by_split[right]
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    ):
        raise MultimodalFaultDatasetError("dataset_asset_leakage_detected")
    leakage = manifest.get("leakage_checks")
    if not isinstance(leakage, dict) or leakage != {
        "asset_overlap_count": 0,
        "case_overlap_count": 0,
        "status": "PASSED",
    }:
        raise MultimodalFaultDatasetError("dataset_leakage_evidence_is_invalid")

    freeze_path = root / "evaluation-freeze.json"
    freeze = _load_object(freeze_path, "evaluation_freeze_is_invalid")
    _verify_evaluation_freeze(
        freeze,
        root=root,
        artifact_catalog=artifact_catalog,
        evaluation_cases=cases_by_split["simulation_evaluation"],
        training_case_ids={
            case.case_id for split in ("train", "validation") for case in cases_by_split[split]
        },
    )
    binding = manifest.get("evaluation_freeze")
    if not isinstance(binding, dict) or binding != {
        "path": "evaluation-freeze.json",
        "sha256": _digest(freeze_path.read_bytes()),
        "freeze_id": freeze["freeze_id"],
    }:
        raise MultimodalFaultDatasetError("evaluation_freeze_binding_is_invalid")

    dataset_id = manifest.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise MultimodalFaultDatasetError("dataset_id_is_invalid")
    return DatasetBuildResult(
        output_dir=str(root),
        dataset_id=dataset_id,
        manifest_sha256=observed_manifest_hash,
        evaluation_freeze_id=str(freeze["freeze_id"]),
        split_counts=split_counts,
        artifact_count=len(artifact_catalog),
        created=False,
    )


def _build_into(root: Path, *, seed: int, split_case_counts: dict[str, int]) -> None:
    artifacts: list[dict[str, Any]] = []
    cases_by_split: dict[str, list[MultimodalFaultCase]] = {}
    knowledge = _write_knowledge_documents(root, artifacts)
    for split_index, split in enumerate(SPLITS):
        cases: list[MultimodalFaultCase] = []
        for case_index in range(split_case_counts[split]):
            fault = _FAULTS[(case_index + split_index * 2) % len(_FAULTS)]
            case = _generate_case(
                root,
                artifacts,
                fault=fault,
                split=split,
                case_index=case_index,
                seed=seed + split_index * 10_000 + case_index,
                knowledge=knowledge[fault.code],
            )
            cases.append(case)
        cases_by_split[split] = cases
        _write_artifact(
            root,
            artifacts,
            f"cases/{split}.jsonl",
            b"".join(_json_line(case.model_dump(mode="json")) for case in cases),
            kind="case_index",
            split=split,
            media_type="application/x-ndjson",
        )

    evaluation_cases = cases_by_split["simulation_evaluation"]
    evaluation_case_ids = {case.case_id for case in evaluation_cases}
    evaluation_paths = {
        "cases/simulation_evaluation.jsonl",
        *(reference.path for case in evaluation_cases for reference in _case_references(case)),
    }
    evaluation_paths.update(case.knowledge.path for case in evaluation_cases)
    evaluation_artifacts = [
        {
            "path": item["path"],
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        }
        for item in artifacts
        if item["path"] in evaluation_paths
    ]
    freeze_payload: dict[str, Any] = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "synthetic": True,
        "purpose": "SIMULATION_EVALUATION_ONLY",
        "production_gold_eligible": False,
        "evidence_eligible": False,
        "immutable": True,
        "case_ids": sorted(evaluation_case_ids),
        "asset_ids": sorted(case.asset_id for case in evaluation_cases),
        "training_case_overlap_count": 0,
        "training_asset_overlap_count": 0,
        "artifacts": sorted(evaluation_artifacts, key=lambda item: item["path"]),
    }
    freeze_payload["freeze_id"] = _canonical_digest(freeze_payload)
    freeze_payload["freeze_sha256"] = _document_hash(freeze_payload, "freeze_sha256")
    freeze_bytes = _json_bytes(freeze_payload)
    _write_artifact(
        root,
        artifacts,
        "evaluation-freeze.json",
        freeze_bytes,
        kind="evaluation_freeze",
        split="simulation_evaluation",
        media_type="application/json",
    )

    split_counts = {split: len(cases) for split, cases in cases_by_split.items()}
    fault_counts = Counter(case.fault_code for cases in cases_by_split.values() for case in cases)
    manifest: dict[str, Any] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "synthetic": True,
        "production_claim": False,
        "production_gold_eligible": False,
        "dataset_id": (
            f"m7-multimodal-fault-v1-seed-{seed}"
            f"-t{split_case_counts['train']}"
            f"-v{split_case_counts['validation']}"
            f"-e{split_case_counts['simulation_evaluation']}"
        ),
        "seed": seed,
        "cases_per_split": (
            next(iter(set(split_case_counts.values())))
            if len(set(split_case_counts.values())) == 1
            else None
        ),
        "configured_split_counts": split_case_counts,
        "case_count": sum(split_counts.values()),
        "split_counts": split_counts,
        "fault_counts": dict(sorted(fault_counts.items())),
        "modalities": list(MODALITIES),
        "governance": {
            "data_owner": "PROJECT_SIMULATION_LAB",
            "authorization": "PROJECT_GENERATED_DATA_ONLY",
            "contains_enterprise_production_data": False,
            "contains_personal_data": False,
            "requires_expert_review_before_training": True,
        },
        "leakage_checks": {
            "asset_overlap_count": 0,
            "case_overlap_count": 0,
            "status": "PASSED",
        },
        "evaluation_freeze": {
            "path": "evaluation-freeze.json",
            "sha256": _digest(freeze_bytes),
            "freeze_id": freeze_payload["freeze_id"],
        },
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
    }
    manifest["manifest_sha256"] = _document_hash(manifest, "manifest_sha256")
    _write_new(root / "manifest.json", _json_bytes(manifest))


def _generate_case(
    root: Path,
    artifacts: list[dict[str, Any]],
    *,
    fault: _FaultDefinition,
    split: str,
    case_index: int,
    seed: int,
    knowledge: ArtifactReference,
) -> MultimodalFaultCase:
    split_token = split.replace("simulation_", "sim-")
    asset_id = f"{fault.asset_model}-{split_token.upper()}-{case_index + 1:03d}"
    case_id = f"m7-{split_token}-{case_index + 1:03d}-{fault.code.lower()}"
    region = _variant_region(fault.region, seed=seed)
    image = _artifact_reference(
        root,
        artifacts,
        f"media/images/{case_id}.png",
        _render_fault_image(fault, asset_id=asset_id, seed=seed, region=region),
        kind="fault_image",
        split=split,
        case_id=case_id,
        media_type="image/png",
    )
    acoustic = _artifact_reference(
        root,
        artifacts,
        f"media/acoustic/{case_id}.wav",
        _render_acoustic_wave(fault, seed=seed),
        kind="machine_acoustic",
        split=split,
        case_id=case_id,
        media_type="audio/wav",
    )
    telemetry_payload = _telemetry_document(fault, asset_id=asset_id, seed=seed)
    telemetry = _artifact_reference(
        root,
        artifacts,
        f"telemetry/{case_id}.json",
        _json_bytes(telemetry_payload),
        kind="telemetry_sequence",
        split=split,
        case_id=case_id,
        media_type="application/json",
    )
    x, y, width, height = region
    alternatives = [candidate.code for candidate in _FAULTS if candidate.code != fault.code][:3]
    case = MultimodalFaultCase(
        case_id=case_id,
        split=split,
        asset_id=asset_id,
        asset_model=fault.asset_model,
        site_id=f"SIM-SITE-{(case_index % 3) + 1}",
        fault_code=fault.code,
        severity=fault.severity,
        image=image,
        acoustic=acoustic,
        telemetry=telemetry,
        engineer_note=f"[模拟脱敏记录] {fault.engineer_note}",
        knowledge=knowledge,
        target={
            "root_cause": fault.root_cause,
            "expected_findings": [
                {
                    "label": fault.visual_label,
                    "region": {"x": x, "y": y, "width": width, "height": height},
                }
            ],
            "forbidden_labels": alternatives,
            "recommended_action": fault.action,
            "required_part_number": fault.part_number,
            "allowed_citations": [f"manual:{fault.code.lower()}"],
            "requires_human_approval": True,
            "device_control_allowed": False,
        },
        governance=CaseGovernance(
            training_candidate_eligible=split in {"train", "validation"},
            simulation_evaluation_eligible=split == "simulation_evaluation",
        ),
    )
    return case


def _write_knowledge_documents(
    root: Path,
    artifacts: list[dict[str, Any]],
) -> dict[str, ArtifactReference]:
    documents: dict[str, ArtifactReference] = {}
    for fault in _FAULTS:
        content = (
            f"# {fault.asset_model} simulated maintenance excerpt\n\n"
            f"Citation ID: `manual:{fault.code.lower()}`\n\n"
            f"Observed condition: `{fault.visual_label}`.\n\n"
            f"Controlled root cause label: `{fault.root_cause}`.\n\n"
            f"Recommended action: `{fault.action}`.\n\n"
            f"Required part: `{fault.part_number}`.\n\n"
            "This project-generated document is for non-production training and evaluation only.\n"
        ).encode()
        documents[fault.code] = _artifact_reference(
            root,
            artifacts,
            f"knowledge/{fault.code.lower()}.md",
            content,
            kind="knowledge_document",
            split="shared",
            case_id=None,
            media_type="text/markdown",
        )
    return documents


def _render_fault_image(
    fault: _FaultDefinition,
    *,
    asset_id: str,
    seed: int,
    region: tuple[float, float, float, float],
) -> bytes:
    rng = random.Random(seed)
    background_delta = rng.randint(-12, 12)
    image = Image.new(
        "RGB",
        (640, 480),
        tuple(max(0, min(255, channel + background_delta)) for channel in (232, 237, 239)),
    )
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 639, 64), fill=(27, 45, 61))
    draw.text((24, 20), f"SIMULATED INDUSTRIAL ASSET  {asset_id}", fill=(255, 255, 255))
    draw.rectangle((80, 145, 510, 350), fill=(87, 116, 136), outline=(32, 54, 68), width=5)
    draw.ellipse((105, 185, 260, 330), fill=(106, 139, 157), outline=(29, 49, 62), width=5)
    draw.rectangle((245, 190, 450, 310), fill=(127, 151, 163), outline=(29, 49, 62), width=5)
    draw.rectangle((450, 170, 555, 330), fill=(82, 101, 112), outline=(29, 49, 62), width=5)
    draw.line((25, 330, 615, 330), fill=(44, 72, 83), width=12)
    draw.line((130, 350, 130, 415), fill=(44, 72, 83), width=16)
    draw.line((500, 350, 500, 415), fill=(44, 72, 83), width=16)
    x, y, width, height = region
    box = (int(x * 640), int(y * 480), int((x + width) * 640), int((y + height) * 480))
    draw.rectangle(box, outline=(238, 75, 43), width=6)
    for index in range(12):
        px = rng.randint(box[0] + 4, max(box[0] + 4, box[2] - 4))
        py = rng.randint(box[1] + 4, max(box[1] + 4, box[3] - 4))
        radius = 3 + index % 5
        color = (190, 60 + index * 4, 25) if "LEAK" in fault.code else (233, 125, 34)
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)
    draw.text(
        (24, 445), f"PROJECT GENERATED | {fault.code} | NOT FIELD EVIDENCE", fill=(116, 37, 28)
    )
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _variant_region(
    base: tuple[float, float, float, float],
    *,
    seed: int,
) -> tuple[float, float, float, float]:
    rng = random.Random(seed ^ 0x5A17C0DE)
    x, y, width, height = base
    width = min(0.80, max(0.08, width + rng.uniform(-0.035, 0.035)))
    height = min(0.80, max(0.08, height + rng.uniform(-0.035, 0.035)))
    x = min(1.0 - width, max(0.0, x + rng.uniform(-0.045, 0.045)))
    y = min(1.0 - height, max(0.0, y + rng.uniform(-0.045, 0.045)))
    return round(x, 4), round(y, 4), round(width, 4), round(height, 4)


def _split_case_counts(
    *,
    cases_per_split: int,
    train_cases: int | None,
    validation_cases: int | None,
    evaluation_cases: int | None,
) -> dict[str, int]:
    counts = {
        "train": cases_per_split if train_cases is None else train_cases,
        "validation": cases_per_split if validation_cases is None else validation_cases,
        "simulation_evaluation": (
            cases_per_split if evaluation_cases is None else evaluation_cases
        ),
    }
    if any(
        isinstance(value, bool) or not 1 <= value <= MAX_CASES_PER_SPLIT
        for value in counts.values()
    ):
        raise MultimodalFaultDatasetError("dataset_generation_configuration_invalid")
    return counts


def _manifest_split_case_counts(manifest: dict[str, Any]) -> dict[str, int]:
    configured = manifest.get("configured_split_counts")
    if isinstance(configured, dict) and set(configured) == set(SPLITS):
        values: dict[str, Any] = {split: configured.get(split) for split in SPLITS}
    else:
        legacy = manifest.get("cases_per_split")
        values = {split: legacy for split in SPLITS}
    parsed: dict[str, int] = {}
    for split in SPLITS:
        value = values[split]
        if isinstance(value, bool) or not isinstance(value, int):
            raise MultimodalFaultDatasetError("dataset_generation_counts_are_invalid")
        parsed[split] = value
    return parsed


def _render_acoustic_wave(fault: _FaultDefinition, *, seed: int) -> bytes:
    rng = random.Random(seed)
    sample_rate = 16_000
    duration_seconds = 1.0
    frame_count = int(sample_rate * duration_seconds)
    frames = bytearray()
    for index in range(frame_count):
        time = index / sample_rate
        carrier = math.sin(2 * math.pi * fault.acoustic_hz * time)
        harmonic = 0.45 * math.sin(2 * math.pi * fault.acoustic_hz * 2.03 * time)
        impulse = 0.35 if index % max(80, int(sample_rate / fault.acoustic_hz)) < 4 else 0.0
        noise = rng.uniform(-0.08, 0.08)
        sample = max(-1.0, min(1.0, (carrier + harmonic + impulse + noise) / 1.9))
        frames.extend(struct.pack("<h", int(sample * 24_000)))
    output = io.BytesIO()
    with wave.open(output, "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(bytes(frames))
    return output.getvalue()


def _telemetry_document(
    fault: _FaultDefinition,
    *,
    asset_id: str,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    baseline = (52.0, 1.2, 5.5, 16.0, 12.0)
    signal_names = ("temperature_c", "vibration_mm_s", "pressure_bar", "current_a", "flow_m3_h")
    samples: list[dict[str, Any]] = []
    for index in range(48):
        progress = index / 47
        values = {
            signal: round(
                base + bias * (0.2 + progress * 0.8) + rng.uniform(-0.15, 0.15),
                4,
            )
            for signal, base, bias in zip(
                signal_names,
                baseline,
                fault.telemetry_bias,
                strict=True,
            )
        }
        samples.append({"minute": index * 5, **values})
    return {
        "schema_version": "simulated-industrial-telemetry/v1",
        "classification": CLASSIFICATION,
        "synthetic": True,
        "asset_id": asset_id,
        "fault_code": fault.code,
        "sample_interval_seconds": 300,
        "signal_order": list(signal_names),
        "samples": samples,
    }


def _verify_case_artifacts(
    case: MultimodalFaultCase,
    root: Path,
    artifact_catalog: dict[str, dict[str, Any]],
) -> None:
    if case.governance.evidence_eligible or case.governance.production_evaluation_eligible:
        raise MultimodalFaultDatasetError("synthetic_case_governance_boundary_changed")
    if case.split == "simulation_evaluation":
        if (
            case.governance.training_candidate_eligible
            or not case.governance.simulation_evaluation_eligible
        ):
            raise MultimodalFaultDatasetError("evaluation_case_purpose_is_invalid")
    elif not case.governance.training_candidate_eligible:
        raise MultimodalFaultDatasetError("training_case_purpose_is_invalid")
    for reference in _case_references(case):
        relative = _safe_relative_path(reference.path)
        artifact = artifact_catalog.get(relative)
        payload = (root / relative).read_bytes()
        if (
            artifact is None
            or reference.sha256 != _digest(payload)
            or reference.size_bytes != len(payload)
            or artifact.get("sha256") != reference.sha256
        ):
            raise MultimodalFaultDatasetError("case_artifact_binding_is_invalid")
    telemetry = _load_object(root / case.telemetry.path, "case_telemetry_is_invalid")
    if (
        telemetry.get("asset_id") != case.asset_id
        or telemetry.get("fault_code") != case.fault_code
        or telemetry.get("classification") != CLASSIFICATION
    ):
        raise MultimodalFaultDatasetError("case_telemetry_binding_is_invalid")


def _verify_evaluation_freeze(
    freeze: dict[str, Any],
    *,
    root: Path,
    artifact_catalog: dict[str, dict[str, Any]],
    evaluation_cases: list[MultimodalFaultCase],
    training_case_ids: set[str],
) -> None:
    if (
        freeze.get("schema_version") != FREEZE_SCHEMA_VERSION
        or freeze.get("classification") != CLASSIFICATION
        or freeze.get("purpose") != "SIMULATION_EVALUATION_ONLY"
        or freeze.get("production_gold_eligible") is not False
        or freeze.get("evidence_eligible") is not False
        or freeze.get("immutable") is not True
        or freeze.get("freeze_sha256") != _document_hash(freeze, "freeze_sha256")
    ):
        raise MultimodalFaultDatasetError("evaluation_freeze_identity_or_digest_invalid")
    freeze_without_hashes = dict(freeze)
    freeze_without_hashes.pop("freeze_sha256", None)
    freeze_id = freeze_without_hashes.pop("freeze_id", None)
    if freeze_id != _canonical_digest(freeze_without_hashes):
        raise MultimodalFaultDatasetError("evaluation_freeze_id_is_invalid")
    case_ids = {case.case_id for case in evaluation_cases}
    if freeze.get("case_ids") != sorted(case_ids) or case_ids & training_case_ids:
        raise MultimodalFaultDatasetError("evaluation_freeze_case_binding_is_invalid")
    asset_ids = sorted(case.asset_id for case in evaluation_cases)
    if freeze.get("asset_ids") != asset_ids:
        raise MultimodalFaultDatasetError("evaluation_freeze_asset_binding_is_invalid")
    frozen_artifacts = freeze.get("artifacts")
    if not isinstance(frozen_artifacts, list) or not frozen_artifacts:
        raise MultimodalFaultDatasetError("evaluation_freeze_artifacts_are_invalid")
    for item in frozen_artifacts:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size_bytes"}:
            raise MultimodalFaultDatasetError("evaluation_freeze_artifacts_are_invalid")
        relative = _safe_relative_path(item["path"])
        catalog = artifact_catalog.get(relative)
        payload = (root / relative).read_bytes()
        if (
            catalog is None
            or item["sha256"] != _digest(payload)
            or item["size_bytes"] != len(payload)
        ):
            raise MultimodalFaultDatasetError("evaluation_freeze_artifact_integrity_failed")


def _case_references(case: MultimodalFaultCase) -> tuple[ArtifactReference, ...]:
    return case.image, case.acoustic, case.telemetry, case.knowledge


def _parse_cases(path: Path) -> list[MultimodalFaultCase]:
    cases: list[MultimodalFaultCase] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cases.append(MultimodalFaultCase.model_validate_json(line))
    except (UnicodeDecodeError, ValidationError, json.JSONDecodeError) as exc:
        raise MultimodalFaultDatasetError("dataset_case_index_is_invalid") from exc
    return cases


def _artifact_reference(
    root: Path,
    artifacts: list[dict[str, Any]],
    relative: str,
    content: bytes,
    *,
    kind: str,
    split: str,
    case_id: str | None,
    media_type: str,
) -> ArtifactReference:
    _write_artifact(
        root,
        artifacts,
        relative,
        content,
        kind=kind,
        split=split,
        case_id=case_id,
        media_type=media_type,
    )
    return ArtifactReference(
        path=relative,
        sha256=_digest(content),
        size_bytes=len(content),
        media_type=media_type,
    )


def _write_artifact(
    root: Path,
    artifacts: list[dict[str, Any]],
    relative: str,
    content: bytes,
    *,
    kind: str,
    split: str,
    media_type: str,
    case_id: str | None = None,
) -> None:
    relative = _safe_relative_path(relative)
    _write_new(root / relative, content)
    artifacts.append(
        {
            "path": relative,
            "sha256": _digest(content),
            "size_bytes": len(content),
            "kind": kind,
            "split": split,
            "case_id": case_id,
            "media_type": media_type,
        }
    )


def _write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(content)


def _safe_remove_temporary(temporary: Path, parent: Path) -> None:
    resolved = temporary.resolve()
    if resolved.parent != parent.resolve() or not resolved.name.startswith("."):
        raise MultimodalFaultDatasetError("temporary_dataset_cleanup_target_is_unsafe")
    shutil.rmtree(resolved, ignore_errors=True)


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MultimodalFaultDatasetError("dataset_artifact_path_is_invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise MultimodalFaultDatasetError("dataset_artifact_path_is_invalid")
    return value


def _load_object(path: Path, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MultimodalFaultDatasetError(reason) from exc
    if not isinstance(value, dict):
        raise MultimodalFaultDatasetError(reason)
    return value


def _json_line(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"


def _canonical_digest(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def _document_hash(value: dict[str, Any], field: str) -> str:
    document = dict(value)
    document.pop(field, None)
    return _canonical_digest(document)


def _digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()
