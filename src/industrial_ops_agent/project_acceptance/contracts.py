"""Closed contracts for the synthetic project-staging acceptance benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

MANIFEST_SCHEMA_VERSION = "project-business-loop-manifest/v1"
MANIFEST_ID = "project-business-loop-pump-seal-leak-v1"
CLASSIFICATION = "SYNTHETIC_PROJECT_ACCEPTANCE"
ALLOWED_CLAIMS = ("PROJECT_STAGING_BUSINESS_ACCEPTANCE",)
PROHIBITED_CLAIMS = (
    "REAL_CUSTOMER_EVIDENCE",
    "ENTERPRISE_PRODUCTION_DATA",
    "TRAINING_GOLD",
    "PRODUCTION_ACCEPTANCE",
)
SOURCE_IMAGE_PATH = (
    "artifacts/m7-vlm-checkpoint-continuation-rescored-lab/object-store/tenant/"
    "tenant-simulation-lab/datasets/snapshots/"
    "dataset-sim-m7-eval-a5d68beddfc298953a61d8a2/published/media/test/"
    "m7-sim-evaluation-003-pump_seal_leak.png"
)
MAX_MANIFEST_BYTES = 128 * 1024
MAX_SOURCE_IMAGE_BYTES = 16 * 1024 * 1024
MAX_STATE_BYTES = 256 * 1024
STATE_DIRECTORY = Path("artifacts/project-business-acceptance/runtime")
DATASET_ID = "dataset-sim-m7-eval-a5d68beddfc298953a61d8a2"
DATASET_VERSION = "published"
SAFE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SAFE_PROCESSOR_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9@._:/+~-]{0,254}$")
RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
PROHIBITED_STATE_KEY_PATTERN = re.compile(
    r"(?:access[_-]?token|client[_-]?secret|credential|authorization|"
    r"raw[_-]?(?:prompt|response|payload|image)|image[_-]?bytes|password|api[_-]?key)",
    re.IGNORECASE,
)
MUTABLE_CHECKPOINT_FACTS = frozenset(
    {"incident_version", "work_order_version", "incident_status", "work_order_status"}
)


class ProjectAcceptanceContractError(ValueError):
    """A stable, safe project acceptance contract failure."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AcceptanceStage(StrEnum):
    """Ordered durable stages in the project business acceptance run."""

    MODEL_EVIDENCE = "MODEL_EVIDENCE"
    QUOTATION = "QUOTATION"
    WORK_ORDER_AUTHORIZATION = "WORK_ORDER_AUTHORIZATION"
    ASSIGNMENT = "ASSIGNMENT"
    FIELD_EXECUTION = "FIELD_EXECUTION"
    RESULT_ACCEPTANCE = "RESULT_ACCEPTANCE"
    RECEIPT = "RECEIPT"


ACCEPTANCE_STAGE_ORDER = (
    AcceptanceStage.MODEL_EVIDENCE,
    AcceptanceStage.QUOTATION,
    AcceptanceStage.WORK_ORDER_AUTHORIZATION,
    AcceptanceStage.ASSIGNMENT,
    AcceptanceStage.FIELD_EXECUTION,
    AcceptanceStage.RESULT_ACCEPTANCE,
    AcceptanceStage.RECEIPT,
)


class AcceptanceCheckpoint(_ClosedModel):
    """Closed restart checkpoint containing identifiers but no sensitive payloads."""

    schema_version: Literal["project-business-checkpoint/v1"] = (
        "project-business-checkpoint/v1"
    )
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    manifest_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tenant_id: Literal["tenant-m1-demo"]
    site_id: Literal["site-m1-demo"]
    asset_id: Literal["asset-m1-pump"]
    expected_release_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    expected_model_manifest_hash: str = Field(pattern=r"^(?:sha256:)?[0-9a-f]{64}$")
    completed_stages: tuple[AcceptanceStage, ...] = ()
    facts: dict[str, str | int | bool] = Field(default_factory=dict)


class AcceptanceActorSubjects(_ClosedModel):
    customer_subject_id: Literal["subject-project-acceptance-customer"]
    after_sales_subject_id: Literal["subject-m7-after-sales"]
    approver_subject_id: Literal["subject-m7-domain-expert"]
    field_subject_id: Literal["subject-m1-engineer"]
    verifier_subject_id: Literal["subject-project-acceptance-verifier"]


class AcceptanceReceipt(_ClosedModel):
    """Secret-safe immutable proof of one completed project-staging run."""

    schema_version: Literal["project-business-acceptance-receipt/v1"] = (
        "project-business-acceptance-receipt/v1"
    )
    acceptance_label: Literal["PROJECT_STAGING_BUSINESS_ACCEPTED"]
    classification: Literal["SYNTHETIC_PROJECT_ACCEPTANCE"]
    target_environment: Literal["STAGING"]
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    manifest_id: Literal["project-business-loop-pump-seal-leak-v1"]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_id: Literal["dataset-sim-m7-eval-a5d68beddfc298953a61d8a2"]
    dataset_version: Literal["published"]
    source_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tenant_id: Literal["tenant-m1-demo"]
    site_id: Literal["site-m1-demo"]
    asset_id: Literal["asset-m1-pump"]
    actors: AcceptanceActorSubjects
    release_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    model_manifest_hash: str = Field(pattern=r"^(?:sha256:)?[0-9a-f]{64}$")
    diagnosis_alias: Literal["industrial-diagnosis-staging"]
    vlm_alias: Literal["industrial-diagnosis-staging"]
    ocr_processor_version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9@._:/+~-]{0,254}$")
    vlm_inference_request_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    diagnosis_inference_request_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
    )
    draft_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    media_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    recognition_run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    evidence_bundle_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    incident_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    incident_version: int = Field(gt=0)
    diagnosis_run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    diagnosis_version: int = Field(gt=0)
    quotation_candidate_id: Literal["sandbox-cpq-standard-repair"]
    quotation_proposal_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    quotation_approval_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    service_quotation_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    quotation_decision_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    work_order_proposal_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    work_order_approval_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    work_order_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    fsm_candidate_profile_id: Literal["sandbox-fsm-profile"]
    assignment_proposal_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    assignment_approval_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    assignment_delivery_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    field_evidence_upload_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    field_evidence_entry_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    field_step_entry_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    field_signature_entry_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    completion_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    verification_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    customer_update_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    closure_proposal_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    closure_approval_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    closure_operation_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    work_order_status: Literal["CLOSED"]
    work_order_version: int = Field(gt=0)
    incident_status: Literal["CLOSED"]
    closure_facts_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class TenantBinding(_ClosedModel):
    tenant_id: Literal["tenant-m1-demo"]
    site_id: Literal["site-m1-demo"]
    asset_id: Literal["asset-m1-pump"]
    asset_model: Literal["PUMP-X100"]


class SourceImage(_ClosedModel):
    path: Literal[
        "artifacts/m7-vlm-checkpoint-continuation-rescored-lab/object-store/tenant/"
        "tenant-simulation-lab/datasets/snapshots/"
        "dataset-sim-m7-eval-a5d68beddfc298953a61d8a2/published/media/test/"
        "m7-sim-evaluation-003-pump_seal_leak.png"
    ]
    media_type: Literal["image/png"]
    size_bytes: int = Field(gt=0, le=MAX_SOURCE_IMAGE_BYTES)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ModelExpectations(_ClosedModel):
    target_environment: Literal["STAGING"]
    diagnosis_alias: Literal["industrial-diagnosis-staging"]
    vlm_alias: Literal["industrial-diagnosis-staging"]
    required_components: tuple[
        Literal["OCR"],
        Literal["VLM"],
        Literal["DIAGNOSIS"],
    ]
    expected_ocr_terms: tuple[Literal["PUMP"]]
    expected_visual_labels: tuple[Literal["pump_seal_leak"]]


class BusinessScenario(_ClosedModel):
    title: str = Field(min_length=1, max_length=200)
    incident_description: str = Field(min_length=20, max_length=4000)
    quotation_candidate_id: Literal["sandbox-cpq-standard-repair"]
    authorization_type: Literal["COVERED_SERVICE"]
    initial_parts_required: Literal[False]
    fsm_candidate_profile_id: Literal["sandbox-fsm-profile"]
    field_assignee_subject_id: Literal["subject-m1-engineer"]
    field_step_code: str = Field(min_length=1, max_length=128)
    field_step_description: str = Field(min_length=3, max_length=2000)
    root_cause: str = Field(min_length=3, max_length=4000)
    customer_signer: str = Field(min_length=1, max_length=255)
    customer_signature_text: str = Field(min_length=1, max_length=1000)
    customer_result_confirmation: str = Field(min_length=3, max_length=2000)
    cost_amount: str = Field(pattern=r"^(0|[1-9][0-9]{0,9})\.[0-9]{2}$")
    satisfaction_rating: Literal[5]


class ExpectedFinalState(_ClosedModel):
    work_order_status: Literal["CLOSED"]
    incident_status: Literal["CLOSED"]
    acceptance_label: Literal["PROJECT_STAGING_BUSINESS_ACCEPTED"]


class ProjectAcceptanceManifest(_ClosedModel):
    schema_version: Literal["project-business-loop-manifest/v1"]
    manifest_id: Literal["project-business-loop-pump-seal-leak-v1"]
    classification: Literal["SYNTHETIC_PROJECT_ACCEPTANCE"]
    allowed_claims: tuple[Literal["PROJECT_STAGING_BUSINESS_ACCEPTANCE"]]
    prohibited_claims: tuple[
        Literal["REAL_CUSTOMER_EVIDENCE"],
        Literal["ENTERPRISE_PRODUCTION_DATA"],
        Literal["TRAINING_GOLD"],
        Literal["PRODUCTION_ACCEPTANCE"],
    ]
    tenant_binding: TenantBinding
    source_image: SourceImage
    model_expectations: ModelExpectations
    business_scenario: BusinessScenario
    expected_final_state: ExpectedFinalState


@dataclass(frozen=True, slots=True)
class ValidatedProjectAcceptanceManifest:
    manifest: ProjectAcceptanceManifest
    manifest_sha256: str
    source_image_path: Path
    source_image_sha256: str


def _safe_source_path(raw: object, *, repository_root: Path) -> Path:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ProjectAcceptanceContractError("source_image_path_invalid")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ProjectAcceptanceContractError("source_image_path_invalid")

    root = repository_root.resolve()
    candidate = root.joinpath(*relative.parts)
    current = root
    try:
        for part in relative.parts:
            current = current / part
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ProjectAcceptanceContractError("source_image_unsafe")
    except FileNotFoundError:
        raise ProjectAcceptanceContractError("source_image_unavailable") from None
    except OSError:
        raise ProjectAcceptanceContractError("source_image_unavailable") from None

    try:
        resolved = candidate.resolve(strict=True)
        metadata = candidate.lstat()
    except OSError:
        raise ProjectAcceptanceContractError("source_image_unavailable") from None
    if not resolved.is_relative_to(root):
        raise ProjectAcceptanceContractError("source_image_path_invalid")
    if not stat.S_ISREG(metadata.st_mode):
        raise ProjectAcceptanceContractError("source_image_unsafe")
    return resolved


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                if size > MAX_SOURCE_IMAGE_BYTES:
                    raise ProjectAcceptanceContractError("source_image_too_large")
                digest.update(block)
    except ProjectAcceptanceContractError:
        raise
    except OSError:
        raise ProjectAcceptanceContractError("source_image_unavailable") from None
    return size, digest.hexdigest()


def _claims_are_exact(document: Mapping[str, Any]) -> bool:
    return (
        document.get("classification") == CLASSIFICATION
        and document.get("allowed_claims") == list(ALLOWED_CLAIMS)
        and document.get("prohibited_claims") == list(PROHIBITED_CLAIMS)
    )


def validate_project_acceptance_manifest(
    document: Mapping[str, Any],
    *,
    repository_root: Path,
    manifest_sha256: str | None = None,
) -> ValidatedProjectAcceptanceManifest:
    """Validate a parsed manifest and its exact repository-bound source image."""

    if not isinstance(document, Mapping):
        raise ProjectAcceptanceContractError("manifest_schema_invalid")
    source_document = document.get("source_image")
    if not isinstance(source_document, Mapping):
        raise ProjectAcceptanceContractError("manifest_schema_invalid")
    raw_path = source_document.get("path")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or "\\" in raw_path
        or PurePosixPath(raw_path).is_absolute()
        or ".." in PurePosixPath(raw_path).parts
        or "." in PurePosixPath(raw_path).parts
    ):
        raise ProjectAcceptanceContractError("source_image_path_invalid")
    if not _claims_are_exact(document):
        raise ProjectAcceptanceContractError("manifest_claims_invalid")
    try:
        manifest = ProjectAcceptanceManifest.model_validate(document)
    except ValidationError:
        raise ProjectAcceptanceContractError("manifest_schema_invalid") from None

    source_path = _safe_source_path(
        manifest.source_image.path,
        repository_root=repository_root,
    )
    size, digest = _sha256_file(source_path)
    if size != manifest.source_image.size_bytes:
        raise ProjectAcceptanceContractError("source_image_size_mismatch")
    if digest != manifest.source_image.sha256:
        raise ProjectAcceptanceContractError("source_image_digest_mismatch")

    if manifest_sha256 is None:
        canonical = json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        manifest_sha256 = hashlib.sha256(canonical).hexdigest()
    return ValidatedProjectAcceptanceManifest(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        source_image_path=source_path,
        source_image_sha256=digest,
    )


def load_project_acceptance_manifest(
    manifest_path: Path,
    *,
    repository_root: Path,
) -> ValidatedProjectAcceptanceManifest:
    """Load a regular bounded JSON manifest and validate all source evidence."""

    try:
        root = repository_root.resolve()
        lexical_manifest = manifest_path.absolute()
        relative_manifest = lexical_manifest.relative_to(root)
        if not relative_manifest.parts:
            raise ProjectAcceptanceContractError("manifest_unsafe")
        current = root
        for part in relative_manifest.parts:
            current = current / part
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ProjectAcceptanceContractError("manifest_unsafe")
        resolved_manifest = lexical_manifest.resolve(strict=True)
    except ProjectAcceptanceContractError:
        raise
    except ValueError:
        raise ProjectAcceptanceContractError("manifest_unsafe") from None
    except OSError:
        raise ProjectAcceptanceContractError("manifest_unavailable") from None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or not resolved_manifest.is_relative_to(root)
        or metadata.st_size > MAX_MANIFEST_BYTES
    ):
        raise ProjectAcceptanceContractError("manifest_unsafe")
    try:
        raw = resolved_manifest.read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ProjectAcceptanceContractError("manifest_invalid") from None
    if not isinstance(document, dict):
        raise ProjectAcceptanceContractError("manifest_schema_invalid")
    return validate_project_acceptance_manifest(
        document,
        repository_root=repository_root,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _has_prohibited_state_content(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            not isinstance(key, str)
            or PROHIBITED_STATE_KEY_PATTERN.search(key) is not None
            or _has_prohibited_state_content(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_has_prohibited_state_content(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return lowered.startswith("bearer ") or "begin private key" in lowered
    return False


def _validate_checkpoint_contract(checkpoint: AcceptanceCheckpoint) -> None:
    if checkpoint.completed_stages != ACCEPTANCE_STAGE_ORDER[: len(checkpoint.completed_stages)]:
        raise ProjectAcceptanceContractError("checkpoint_stage_order_invalid")
    if _has_prohibited_state_content(checkpoint.facts):
        raise ProjectAcceptanceContractError("checkpoint_prohibited_field")


def validate_acceptance_receipt(
    document: Mapping[str, Any] | AcceptanceReceipt,
) -> AcceptanceReceipt:
    """Validate the closed receipt without accepting sensitive or production data."""

    if isinstance(document, AcceptanceReceipt):
        payload: object = document.model_dump(mode="json")
    else:
        payload = document
    if not isinstance(payload, Mapping):
        raise ProjectAcceptanceContractError("receipt_schema_invalid")
    if _has_prohibited_state_content(payload):
        raise ProjectAcceptanceContractError("receipt_prohibited_field")
    try:
        receipt = AcceptanceReceipt.model_validate(payload)
    except ValidationError:
        raise ProjectAcceptanceContractError("receipt_schema_invalid") from None
    actors = receipt.actors
    if len(set(actors.model_dump().values())) != 5:
        raise ProjectAcceptanceContractError("receipt_actor_separation_invalid")
    return receipt


def _required_receipt_id(checkpoint: AcceptanceCheckpoint, name: str) -> str:
    value = checkpoint.facts.get(name)
    if not isinstance(value, str) or not SAFE_IDENTIFIER_PATTERN.fullmatch(value):
        raise ProjectAcceptanceContractError(f"checkpoint_fact_invalid:{name}")
    return value


def _required_receipt_processor(checkpoint: AcceptanceCheckpoint, name: str) -> str:
    value = checkpoint.facts.get(name)
    if not isinstance(value, str) or not SAFE_PROCESSOR_PATTERN.fullmatch(value):
        raise ProjectAcceptanceContractError(f"checkpoint_fact_invalid:{name}")
    return value


def _required_receipt_int(checkpoint: AcceptanceCheckpoint, name: str) -> int:
    value = checkpoint.facts.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProjectAcceptanceContractError(f"checkpoint_fact_invalid:{name}")
    return value


def _required_receipt_digest(checkpoint: AcceptanceCheckpoint, name: str) -> str:
    value = checkpoint.facts.get(name)
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ProjectAcceptanceContractError(f"checkpoint_fact_invalid:{name}")
    return value


def build_acceptance_receipt(checkpoint: AcceptanceCheckpoint) -> AcceptanceReceipt:
    """Build a deterministic receipt only from a complete, closed checkpoint."""

    _validate_checkpoint_contract(checkpoint)
    if checkpoint.completed_stages != ACCEPTANCE_STAGE_ORDER:
        raise ProjectAcceptanceContractError("checkpoint_incomplete")
    receipt = AcceptanceReceipt(
        acceptance_label="PROJECT_STAGING_BUSINESS_ACCEPTED",
        classification="SYNTHETIC_PROJECT_ACCEPTANCE",
        target_environment="STAGING",
        run_id=checkpoint.run_id,
        manifest_id=checkpoint.manifest_id,
        manifest_sha256=checkpoint.manifest_sha256,
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        source_image_sha256=checkpoint.source_image_sha256,
        tenant_id=checkpoint.tenant_id,
        site_id=checkpoint.site_id,
        asset_id=checkpoint.asset_id,
        actors=AcceptanceActorSubjects(
            customer_subject_id="subject-project-acceptance-customer",
            after_sales_subject_id="subject-m7-after-sales",
            approver_subject_id="subject-m7-domain-expert",
            field_subject_id="subject-m1-engineer",
            verifier_subject_id="subject-project-acceptance-verifier",
        ),
        release_id=checkpoint.expected_release_id,
        model_manifest_hash=checkpoint.expected_model_manifest_hash,
        diagnosis_alias="industrial-diagnosis-staging",
        vlm_alias="industrial-diagnosis-staging",
        ocr_processor_version=_required_receipt_processor(
            checkpoint, "ocr_processor_version"
        ),
        vlm_inference_request_id=_required_receipt_id(
            checkpoint, "vlm_inference_request_id"
        ),
        diagnosis_inference_request_id=_required_receipt_id(
            checkpoint, "diagnosis_inference_request_id"
        ),
        draft_id=_required_receipt_id(checkpoint, "draft_id"),
        media_id=_required_receipt_id(checkpoint, "media_id"),
        recognition_run_id=_required_receipt_id(checkpoint, "recognition_run_id"),
        evidence_bundle_id=_required_receipt_id(checkpoint, "evidence_bundle_id"),
        incident_id=_required_receipt_id(checkpoint, "incident_id"),
        incident_version=_required_receipt_int(checkpoint, "incident_version"),
        diagnosis_run_id=_required_receipt_id(checkpoint, "diagnosis_run_id"),
        diagnosis_version=_required_receipt_int(checkpoint, "diagnosis_version"),
        quotation_candidate_id="sandbox-cpq-standard-repair",
        quotation_proposal_id=_required_receipt_id(checkpoint, "quotation_proposal_id"),
        quotation_approval_id=_required_receipt_id(checkpoint, "quotation_approval_id"),
        service_quotation_id=_required_receipt_id(checkpoint, "service_quotation_id"),
        quotation_decision_id=_required_receipt_id(checkpoint, "quotation_decision_id"),
        work_order_proposal_id=_required_receipt_id(checkpoint, "work_order_proposal_id"),
        work_order_approval_id=_required_receipt_id(checkpoint, "work_order_approval_id"),
        work_order_id=_required_receipt_id(checkpoint, "work_order_id"),
        fsm_candidate_profile_id="sandbox-fsm-profile",
        assignment_proposal_id=_required_receipt_id(checkpoint, "assignment_proposal_id"),
        assignment_approval_id=_required_receipt_id(checkpoint, "assignment_approval_id"),
        assignment_delivery_id=_required_receipt_id(checkpoint, "assignment_delivery_id"),
        field_evidence_upload_id=_required_receipt_id(
            checkpoint, "field_evidence_upload_id"
        ),
        field_evidence_entry_id=_required_receipt_id(
            checkpoint, "field_evidence_entry_id"
        ),
        field_step_entry_id=_required_receipt_id(checkpoint, "field_step_entry_id"),
        field_signature_entry_id=_required_receipt_id(
            checkpoint, "field_signature_entry_id"
        ),
        completion_id=_required_receipt_id(checkpoint, "completion_id"),
        verification_id=_required_receipt_id(checkpoint, "verification_id"),
        customer_update_id=_required_receipt_id(checkpoint, "customer_update_id"),
        closure_proposal_id=_required_receipt_id(checkpoint, "closure_proposal_id"),
        closure_approval_id=_required_receipt_id(checkpoint, "closure_approval_id"),
        closure_operation_id=_required_receipt_id(checkpoint, "closure_operation_id"),
        work_order_status="CLOSED",
        work_order_version=_required_receipt_int(checkpoint, "work_order_version"),
        incident_status="CLOSED",
        closure_facts_digest=_required_receipt_digest(
            checkpoint, "closure_facts_digest"
        ),
    )
    return validate_acceptance_receipt(receipt)


def checkpoint_from_acceptance_receipt(receipt: AcceptanceReceipt) -> AcceptanceCheckpoint:
    """Reconstruct only the safe facts required for independent API reconciliation."""

    receipt = validate_acceptance_receipt(receipt)
    metadata_fields = {
        "schema_version",
        "acceptance_label",
        "classification",
        "target_environment",
        "run_id",
        "manifest_id",
        "manifest_sha256",
        "dataset_id",
        "dataset_version",
        "source_image_sha256",
        "tenant_id",
        "site_id",
        "asset_id",
        "actors",
        "release_id",
        "model_manifest_hash",
        "diagnosis_alias",
        "vlm_alias",
        "quotation_candidate_id",
        "fsm_candidate_profile_id",
    }
    document = receipt.model_dump(mode="python")
    facts = {
        key: value
        for key, value in document.items()
        if key not in metadata_fields and isinstance(value, (str, int, bool))
    }
    return AcceptanceCheckpoint(
        run_id=receipt.run_id,
        manifest_id=receipt.manifest_id,
        manifest_sha256=receipt.manifest_sha256,
        source_image_sha256=receipt.source_image_sha256,
        tenant_id=receipt.tenant_id,
        site_id=receipt.site_id,
        asset_id=receipt.asset_id,
        expected_release_id=receipt.release_id,
        expected_model_manifest_hash=receipt.model_manifest_hash,
        completed_stages=ACCEPTANCE_STAGE_ORDER,
        facts=facts,
    )


def _canonical_model_bytes(model: BaseModel) -> bytes:
    return (
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


class ProjectAcceptanceStateStore:
    """Atomic private storage rooted at the one repository-managed state path."""

    def __init__(
        self,
        *,
        repository_root: Path,
        state_root: Path | None = None,
    ) -> None:
        try:
            metadata = repository_root.lstat()
            resolved_repository = repository_root.resolve(strict=True)
        except OSError:
            raise ProjectAcceptanceContractError("state_repository_invalid") from None
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise ProjectAcceptanceContractError("state_repository_invalid")
        expected = resolved_repository / STATE_DIRECTORY
        supplied = state_root or expected
        if ".." in supplied.parts:
            raise ProjectAcceptanceContractError("state_root_invalid")
        if not supplied.is_absolute():
            supplied = resolved_repository / supplied
        if Path(os.path.abspath(supplied)) != expected:
            raise ProjectAcceptanceContractError("state_root_invalid")
        self.repository_root = resolved_repository
        self.state_root = expected
        self._prepare_state_root()

    def _prepare_state_root(self) -> None:
        artifacts = self.repository_root / "artifacts"
        self._ensure_directory(artifacts, private=False)
        project = artifacts / "project-business-acceptance"
        self._ensure_directory(project, private=True)
        self._ensure_directory(self.state_root, private=True)

    @staticmethod
    def _ensure_directory(path: Path, *, private: bool) -> None:
        try:
            if not path.exists() and not path.is_symlink():
                path.mkdir(mode=0o700)
            metadata = path.lstat()
        except OSError:
            raise ProjectAcceptanceContractError("state_directory_invalid") from None
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
        ):
            raise ProjectAcceptanceContractError("state_directory_invalid")
        if private and stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ProjectAcceptanceContractError("state_directory_permissions")

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
            raise ProjectAcceptanceContractError("state_run_id_invalid")

    def _run_directory(self, run_id: str, *, create: bool) -> Path | None:
        self._validate_run_id(run_id)
        run_directory = self.state_root / run_id
        if not run_directory.exists() and not run_directory.is_symlink():
            if not create:
                return None
            try:
                run_directory.mkdir(mode=0o700)
            except OSError:
                raise ProjectAcceptanceContractError("state_directory_invalid") from None
        self._ensure_directory(run_directory, private=True)
        return run_directory

    def checkpoint_path(self, run_id: str) -> Path:
        run_directory = self._run_directory(run_id, create=True)
        assert run_directory is not None
        return run_directory / "checkpoint.json"

    def receipt_path(self, run_id: str) -> Path:
        run_directory = self._run_directory(run_id, create=True)
        assert run_directory is not None
        return run_directory / "receipt.json"

    @staticmethod
    def _read_private_file(path: Path) -> bytes:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            raise
        except OSError:
            raise ProjectAcceptanceContractError("state_file_invalid") from None
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
        ):
            raise ProjectAcceptanceContractError("state_file_unsafe")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ProjectAcceptanceContractError("state_file_permissions")
        if metadata.st_size <= 0 or metadata.st_size > MAX_STATE_BYTES:
            raise ProjectAcceptanceContractError("state_file_invalid")
        try:
            return path.read_bytes()
        except OSError:
            raise ProjectAcceptanceContractError("state_file_invalid") from None

    @staticmethod
    def _decode_document(raw: bytes, *, reason: str) -> Mapping[str, Any]:
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ProjectAcceptanceContractError(reason) from None
        if not isinstance(document, Mapping):
            raise ProjectAcceptanceContractError(reason)
        return document

    @staticmethod
    def _atomic_private_write(path: Path, content: bytes) -> None:
        descriptor = -1
        temporary_name = ""
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                dir=path.parent,
            )
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
            temporary_name = ""
            directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            raise ProjectAcceptanceContractError("state_file_write_failed") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary_name:
                with suppress(FileNotFoundError):
                    os.unlink(temporary_name)

    def load_checkpoint(self, run_id: str) -> AcceptanceCheckpoint | None:
        run_directory = self._run_directory(run_id, create=False)
        if run_directory is None:
            return None
        path = run_directory / "checkpoint.json"
        try:
            raw = self._read_private_file(path)
        except FileNotFoundError:
            return None
        document = self._decode_document(raw, reason="checkpoint_schema_invalid")
        if _has_prohibited_state_content(document):
            raise ProjectAcceptanceContractError("checkpoint_prohibited_field")
        try:
            checkpoint = AcceptanceCheckpoint.model_validate(document)
        except ValidationError:
            raise ProjectAcceptanceContractError("checkpoint_schema_invalid") from None
        _validate_checkpoint_contract(checkpoint)
        return checkpoint

    def write_checkpoint(self, checkpoint: AcceptanceCheckpoint) -> Path:
        try:
            checkpoint = AcceptanceCheckpoint.model_validate(checkpoint)
        except ValidationError:
            raise ProjectAcceptanceContractError("checkpoint_schema_invalid") from None
        _validate_checkpoint_contract(checkpoint)
        path = self.checkpoint_path(checkpoint.run_id)
        previous = self.load_checkpoint(checkpoint.run_id)
        if previous is not None:
            previous_binding = previous.model_dump(
                exclude={"completed_stages", "facts"}, mode="python"
            )
            current_binding = checkpoint.model_dump(
                exclude={"completed_stages", "facts"}, mode="python"
            )
            if previous_binding != current_binding:
                raise ProjectAcceptanceContractError("checkpoint_binding_conflict")
            if checkpoint.completed_stages[: len(previous.completed_stages)] != (
                previous.completed_stages
            ) or any(
                key not in MUTABLE_CHECKPOINT_FACTS and checkpoint.facts.get(key) != value
                for key, value in previous.facts.items()
            ):
                raise ProjectAcceptanceContractError("checkpoint_progress_conflict")
        content = _canonical_model_bytes(checkpoint)
        if previous is None or self._read_private_file(path) != content:
            self._atomic_private_write(path, content)
        return path

    def load_receipt(self, run_id: str) -> AcceptanceReceipt | None:
        run_directory = self._run_directory(run_id, create=False)
        if run_directory is None:
            return None
        path = run_directory / "receipt.json"
        try:
            raw = self._read_private_file(path)
        except FileNotFoundError:
            return None
        document = self._decode_document(raw, reason="receipt_schema_invalid")
        return validate_acceptance_receipt(document)

    def write_receipt(self, receipt: AcceptanceReceipt) -> Path:
        receipt = validate_acceptance_receipt(receipt)
        content = _canonical_model_bytes(receipt)
        path = self.receipt_path(receipt.run_id)
        if path.exists() or path.is_symlink():
            existing = self._read_private_file(path)
            if existing != content:
                raise ProjectAcceptanceContractError("receipt_conflict")
            return path
        self._atomic_private_write(path, content)
        return path
