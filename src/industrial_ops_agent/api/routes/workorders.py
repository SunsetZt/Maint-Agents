"""WorkOrder query and explicit lifecycle commands."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, NoReturn, Self

from fastapi import APIRouter, Body, Depends, Header, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_identity,
    get_object_store,
    get_optional_edge_pack_signer,
    get_recognition_dispatcher,
    get_tool_gateway,
)
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.api.routes.recognition import EvidenceEnvelope, _evidence_envelope
from industrial_ops_agent.application.assets import ResourceNotVisible
from industrial_ops_agent.application.incidents import (
    IdempotencyConflict,
    RecognitionRun,
    RecognitionService,
)
from industrial_ops_agent.approval.models import ApprovalNotVisible
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.edge.contracts import (
    FieldEdgeDiagnosisCandidate,
    FieldEdgeDiagnosisPackClaims,
    FieldEdgeDiagnosisResult,
    FieldEdgePromptBinding,
    FieldEdgeReleaseBinding,
    FieldEdgeRetrievalBinding,
    FieldEdgeRuntimeEvidence,
)
from industrial_ops_agent.edge.signing import Ed25519PackSigner
from industrial_ops_agent.media.service import MediaUploadRejected, TenantObjectStore
from industrial_ops_agent.multimodal.models import (
    EvidenceConfirmationRequired,
    EvidenceVersionConflict,
    FindingDisposition,
    OcrDisposition,
    QrDisposition,
    TranscriptDisposition,
)
from industrial_ops_agent.orchestration.activities import (
    RecognitionActivityInput,
    RecognitionDispatcher,
    RecognitionDispatchUnavailable,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    WorkOrderEdgeDiagnosisCandidateRecord,
    WorkOrderEdgeDiagnosisPackRecord,
)
from industrial_ops_agent.tools.contracts import EnterpriseToolError
from industrial_ops_agent.tools.gateway import (
    PartIssueProposalConflict,
    PartIssueProposalIdempotencyConflict,
    PartMovementProposalConflict,
    PartMovementProposalIdempotencyConflict,
    ToolGateway,
    ToolRateLimitExceeded,
)
from industrial_ops_agent.workorders.edge_diagnosis import (
    EdgeDiagnosisConflict,
    EdgeDiagnosisNotVisible,
    EdgeDiagnosisRejected,
    EdgeDiagnosisUnavailable,
    WorkOrderEdgeDiagnosisService,
)
from industrial_ops_agent.workorders.evidence import (
    FieldEvidenceService,
    FieldEvidenceUploadView,
    FieldEvidenceValidationError,
)
from industrial_ops_agent.workorders.offline_pack import (
    OfflinePackNotAvailable,
    WorkOrderOfflinePackService,
    WorkOrderOfflinePackView,
)
from industrial_ops_agent.workorders.service import (
    DispatchQueueItem,
    WorkOrderClosurePartView,
    WorkOrderClosureReportView,
    WorkOrderConflict,
    WorkOrderControlView,
    WorkOrderFieldEntryView,
    WorkOrderNotVisible,
    WorkOrderPartAccountingEntryView,
    WorkOrderPartAccountingView,
    WorkOrderPartAllocationView,
    WorkOrderPartMovementView,
    WorkOrderRepairHistoryView,
    WorkOrderRepairRoundView,
    WorkOrderService,
    WorkOrderView,
)
from industrial_ops_agent.workorders.voice_guidance import (
    FieldVoiceGuidance,
    FieldVoiceGuidanceService,
)

router = APIRouter(tags=["work-orders"])
EDGE_DIAGNOSIS_ERRORS = (
    OfflinePackNotAvailable,
    WorkOrderNotVisible,
    AuthorizationDenied,
    WorkOrderConflict,
    EdgeDiagnosisUnavailable,
    EdgeDiagnosisNotVisible,
    EdgeDiagnosisConflict,
    EdgeDiagnosisRejected,
)

WorkOrderStatus = Literal[
    "READY",
    "ASSIGNED",
    "ACCEPTED",
    "IN_PROGRESS",
    "ON_HOLD",
    "COMPLETED",
    "VERIFIED",
    "ESCALATED",
    "CLOSED",
    "CANCELLED",
]
WorkOrderPriority = Literal["CRITICAL", "HIGH", "NORMAL", "LOW"]


class AssignBody(BaseModel):
    assignee_subject_id: str = Field(min_length=1, max_length=128)


class CompleteBody(BaseModel):
    root_cause: str
    actions: list[str]
    evidence_ids: list[str]
    part_reservation_ids: list[str] = Field(default_factory=list)
    cost_amount: str | None = None
    customer_confirmation: str | None = None


class VerifyBody(BaseModel):
    passed: bool
    reason: str


class HoldBody(BaseModel):
    reason: str = Field(min_length=3, max_length=2000)
    recovery_condition: str = Field(min_length=3, max_length=2000)


class ResumeBody(BaseModel):
    reason: str = Field(min_length=3, max_length=2000)


class EscalateBody(BaseModel):
    reason: str = Field(min_length=3, max_length=2000)
    recovery_condition: str = Field(min_length=3, max_length=2000)
    responsible_subject_id: str = Field(min_length=1, max_length=128)


class ReplanBody(BaseModel):
    target_status: Literal["READY", "IN_PROGRESS"]
    reason: str = Field(min_length=3, max_length=2000)
    responsible_subject_id: str | None = Field(default=None, max_length=128)
    service_window_start: datetime
    service_window_end: datetime
    sla_due_at: datetime


class RescheduleBody(BaseModel):
    priority: WorkOrderPriority
    reason: str = Field(min_length=3, max_length=2000)
    service_window_start: datetime
    service_window_end: datetime
    sla_due_at: datetime


class FieldAiObservationSourceBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal[
        "OCR_BLOCK",
        "ASR_SEGMENT",
        "VISUAL_FINDING",
        "VIDEO_EVENT",
        "EDGE_DIAGNOSIS_CANDIDATE",
    ]
    source_id: str = Field(min_length=1, max_length=128)


class FieldEntryBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_operation_id: str = Field(min_length=8, max_length=128)
    entry_type: Literal["STEP", "PART", "EVIDENCE", "NOTE", "SIGNATURE", "AI_OBSERVATION"]
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    description: str | None = Field(default=None, min_length=1, max_length=2000)
    step_code: str | None = Field(default=None, min_length=1, max_length=128)
    outcome: Literal["COMPLETED", "BLOCKED"] | None = None
    part_reservation_id: str | None = Field(default=None, min_length=1, max_length=128)
    part_number: str | None = Field(default=None, min_length=1, max_length=128)
    quantity: int | None = Field(default=None, ge=1, le=10000)
    evidence_id: str | None = Field(default=None, min_length=1, max_length=128)
    signed_by: str | None = Field(default=None, min_length=1, max_length=255)
    signature_role: Literal["CUSTOMER", "ENGINEER"] | None = None
    confirmation_text: str | None = Field(default=None, min_length=1, max_length=1000)
    observation: str | None = Field(default=None, min_length=1, max_length=2000)
    field_evidence_upload_id: str | None = Field(default=None, min_length=1, max_length=128)
    recognition_bundle_id: str | None = Field(default=None, min_length=1, max_length=128)
    recognition_bundle_version: int | None = Field(default=None, ge=1)
    edge_diagnosis_candidate_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )
    edge_diagnosis_candidate_version: int | None = Field(default=None, ge=1)
    source_items: list[FieldAiObservationSourceBody] | None = Field(
        default=None,
        min_length=1,
        max_length=32,
    )

    @model_validator(mode="before")
    @classmethod
    def accept_payload_envelope(cls, value: Any) -> Any:
        if isinstance(value, dict) and isinstance(value.get("payload"), dict):
            flattened = dict(value)
            payload = flattened.pop("payload")
            edge_binding_fields = {
                "edge_diagnosis_candidate_id",
                "edge_diagnosis_candidate_version",
            }
            source_items = payload.get("source_items", flattened.get("source_items"))
            has_edge_source = isinstance(source_items, list) and any(
                isinstance(item, dict) and item.get("source_type") == "EDGE_DIAGNOSIS_CANDIDATE"
                for item in source_items
            )
            if (
                edge_binding_fields.intersection(payload)
                or edge_binding_fields.intersection(flattened)
                or has_edge_source
            ):
                allowed_edge_payload_fields = {
                    "observation",
                    "edge_diagnosis_candidate_id",
                    "edge_diagnosis_candidate_version",
                    "source_items",
                }
                if set(payload).difference(allowed_edge_payload_fields):
                    raise ValueError("ai_observation_edge_payload_extra_fields")
            for field in (
                "part_reservation_id",
                "part_number",
                "quantity",
                "observation",
                "field_evidence_upload_id",
                "recognition_bundle_id",
                "recognition_bundle_version",
                "edge_diagnosis_candidate_id",
                "edge_diagnosis_candidate_version",
                "source_items",
            ):
                if field in payload:
                    flattened.setdefault(field, payload[field])
            return flattened
        return value

    @model_validator(mode="after")
    def validate_entry_payload(self) -> Self:
        recognition_binding_present = any(
            value is not None
            for value in (
                self.field_evidence_upload_id,
                self.recognition_bundle_id,
                self.recognition_bundle_version,
            )
        )
        edge_binding_present = any(
            value is not None
            for value in (
                self.edge_diagnosis_candidate_id,
                self.edge_diagnosis_candidate_version,
            )
        )
        ai_observation_complete = False
        if self.entry_type == "AI_OBSERVATION":
            if recognition_binding_present and edge_binding_present:
                raise ValueError("ai_observation_source_mixed")
            if not self.observation or not self.observation.strip() or not self.source_items:
                raise ValueError("ai_observation_entry_fields_incomplete")
            if edge_binding_present:
                ai_observation_complete = bool(
                    self.edge_diagnosis_candidate_id
                    and self.edge_diagnosis_candidate_version
                    and len(self.source_items) == 1
                    and self.source_items[0].source_type == "EDGE_DIAGNOSIS_CANDIDATE"
                    and self.source_items[0].source_id == self.edge_diagnosis_candidate_id
                )
            elif recognition_binding_present:
                ai_observation_complete = bool(
                    self.field_evidence_upload_id
                    and self.recognition_bundle_id
                    and self.recognition_bundle_version
                    and all(
                        item.source_type != "EDGE_DIAGNOSIS_CANDIDATE" for item in self.source_items
                    )
                )
            if not ai_observation_complete:
                raise ValueError("ai_observation_source_invalid")
        required = {
            "STEP": bool(self.description and self.outcome),
            "PART": bool(self.part_reservation_id and self.part_number and self.quantity),
            "EVIDENCE": bool(self.evidence_id),
            "NOTE": bool(self.description),
            "SIGNATURE": bool(self.signed_by and self.signature_role and self.confirmation_text),
            "AI_OBSERVATION": ai_observation_complete,
        }
        if not required[self.entry_type]:
            raise ValueError(f"{self.entry_type.lower()}_entry_fields_incomplete")
        if self.entry_type == "AI_OBSERVATION" and self.source_items is not None:
            source_keys = {(item.source_type, item.source_id) for item in self.source_items}
            if len(source_keys) != len(self.source_items):
                raise ValueError("ai_observation_source_duplicate")
        return self

    def entry_payload(self) -> dict[str, Any]:
        fields = {
            "STEP": ("step_code", "description", "outcome"),
            "PART": ("part_reservation_id", "part_number", "quantity"),
            "EVIDENCE": ("evidence_id", "description"),
            "NOTE": ("description",),
            "SIGNATURE": ("signed_by", "signature_role", "confirmation_text"),
            "AI_OBSERVATION": (
                "observation",
                "field_evidence_upload_id",
                "recognition_bundle_id",
                "recognition_bundle_version",
                "edge_diagnosis_candidate_id",
                "edge_diagnosis_candidate_version",
                "source_items",
            ),
        }[self.entry_type]
        return {
            field: (
                [item.model_dump(mode="json") for item in value]
                if field == "source_items"
                else value
            )
            for field in fields
            if (value := getattr(self, field)) is not None
        }


class FieldCompletionBody(BaseModel):
    root_cause: str = Field(min_length=3, max_length=4000)
    cost_amount: str = Field(min_length=1, max_length=64)


class RevokeOfflinePackBody(BaseModel):
    reason: str = Field(min_length=3, max_length=500)


class ImportEdgeDiagnosisBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_operation_id: str = Field(min_length=8, max_length=128)
    result: FieldEdgeDiagnosisResult


class DecideEdgeDiagnosisBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["ACCEPTED", "REJECTED"]
    reason: str = Field(min_length=3, max_length=1000)


class PartIssueProposalBody(BaseModel):
    work_order_version: int = Field(ge=1)


class PartConsumptionProposalBody(BaseModel):
    work_order_version: int = Field(ge=1)
    field_entry_id: str = Field(min_length=1, max_length=128)


class PartReturnProposalBody(BaseModel):
    work_order_version: int = Field(ge=1)
    quantity: int = Field(ge=1, le=20)


class WorkOrderResponse(BaseModel):
    work_order_id: str
    incident_id: str
    proposal_id: str
    reservation_id: str | None
    creation_mode: Literal["PARTS_RESERVATION", "SERVICE_AUTHORIZATION"]
    authorization_type: Literal["COVERED_SERVICE", "ACCEPTED_QUOTATION"] | None
    diagnosis_run_id: str | None
    diagnosis_version: int | None
    service_quotation_id: str | None
    initial_parts_required: bool
    part_issue_required: bool
    part_accounting_required: bool
    status: str
    assigned_subject_id: str | None
    pending_assignment_operation_id: str | None
    priority: str
    sla_due_at: str | None
    sla_status: str
    service_window_start: str | None
    service_window_end: str | None
    version: int
    created_at: str
    updated_at: str
    legal_actions: list[str]


class WorkOrderEnvelope(BaseModel):
    data: WorkOrderResponse
    meta: dict[str, str]


class FieldVoiceDiagnosisResponse(BaseModel):
    diagnosis_run_id: str
    status: str
    version: int
    conclusion: str | None
    next_checks: list[str]


class FieldRediagnosisProgressResponse(BaseModel):
    diagnosis_run_id: str
    status: str
    version: int
    source_diagnosis_run_id: str
    source_work_order_version: int
    field_entry_count: int
    updated_at: datetime


class FieldVoiceGuidanceResponse(BaseModel):
    work_order_id: str
    work_order_version: int
    work_order_status: str
    incident_id: str
    diagnosis: FieldVoiceDiagnosisResponse | None
    field_reanalysis: FieldRediagnosisProgressResponse | None
    safety_warning: str
    legal_actions: list[
        Literal[
            "START_REALTIME_VOICE",
            "SYNTHESIZE_VOICE_GUIDANCE",
            "REQUEST_FIELD_REANALYSIS",
        ]
    ]


class FieldVoiceGuidanceEnvelope(BaseModel):
    data: FieldVoiceGuidanceResponse
    meta: dict[str, str]


class WorkOrderCenterItemResponse(WorkOrderResponse):
    asset_id: str
    asset_display_name: str | None
    model_code: str | None
    site_id: str | None
    site_name: str | None
    incident_description: str


class WorkOrderControlResponse(BaseModel):
    control_id: str
    work_order_version: int
    command_type: str
    previous_status: str
    target_status: str
    reason: str
    recovery_condition: str | None
    responsible_subject_id: str | None
    service_window_start: str | None
    service_window_end: str | None
    sla_due_at: str | None
    occurred_at: str


class WorkOrderControlsEnvelope(BaseModel):
    data: list[WorkOrderControlResponse]
    meta: dict[str, str]


class WorkOrderFieldEntryResponse(BaseModel):
    entry_id: str
    sequence: int
    client_operation_id: str
    entry_type: str
    payload: dict[str, Any]
    actor_subject_id: str
    occurred_at: str
    recorded_at: str


class WorkOrderFieldEntryEnvelope(BaseModel):
    data: WorkOrderFieldEntryResponse
    meta: dict[str, str]


class WorkOrderFieldEntriesEnvelope(BaseModel):
    data: list[WorkOrderFieldEntryResponse]
    meta: dict[str, str]


class FieldEvidenceUploadResponse(BaseModel):
    evidence_upload_id: str
    work_order_id: str
    media_id: str
    uploaded_by_subject_id: str
    client_operation_id: str
    content_hash: str
    declared_mime: str
    detected_mime: str | None
    size_bytes: int
    work_order_version: int
    scan_state: Literal["PENDING", "CLEAN", "REJECTED", "INFECTED"]
    scan_version: int
    content_credential_status: str
    ready_to_attach: bool
    legal_actions: list[
        Literal[
            "ATTACH",
            "RECOGNIZE",
            "CONFIRM_RECOGNITION",
            "CREATE_AI_OBSERVATION",
        ]
    ]
    recognition_run_id: str | None
    recognition_status: str | None
    recognition_bundle_id: str | None
    recognition_bundle_status: str | None
    recognition_bundle_version: int | None
    recognition_failure_summary: str | None
    occurred_at: datetime
    updated_at: datetime


class FieldEvidenceUploadEnvelope(BaseModel):
    data: FieldEvidenceUploadResponse
    meta: dict[str, Any]


class FieldEvidenceUploadsEnvelope(BaseModel):
    data: list[FieldEvidenceUploadResponse]
    meta: dict[str, Any]


class FieldRecognitionRunResponse(BaseModel):
    recognition_run_id: str
    draft_id: str
    media_id: str
    context_kind: Literal["FIELD_EVIDENCE"]
    work_order_id: str
    field_evidence_upload_id: str
    work_order_version: int
    requested_by_subject_id: str
    workflow_id: str
    status: str
    processor_profile: str
    evidence_bundle_id: str | None
    failure_reason: str | None
    version: int
    created_at: datetime
    updated_at: datetime


class FieldRecognitionRunEnvelope(BaseModel):
    data: FieldRecognitionRunResponse
    meta: dict[str, str | bool]


class FieldOcrBlockDecisionBody(BaseModel):
    disposition: OcrDisposition
    corrected_text: str | None = Field(default=None, max_length=4_000)


class FieldRecognitionStartBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temporal_analysis: bool = False


class FieldTranscriptDecisionBody(BaseModel):
    disposition: TranscriptDisposition
    corrected_text: str | None = Field(default=None, max_length=4_000)


class FieldQrCodeDecisionBody(BaseModel):
    disposition: QrDisposition
    corrected_text: str | None = Field(default=None, max_length=4_096)


class FieldRecognitionConfirmationBody(BaseModel):
    corrections: dict[str, str] = Field(default_factory=dict)
    finding_dispositions: dict[str, FindingDisposition] = Field(default_factory=dict)
    ocr_block_decisions: dict[str, FieldOcrBlockDecisionBody] = Field(default_factory=dict)
    transcript_decisions: dict[str, FieldTranscriptDecisionBody] = Field(default_factory=dict)
    video_event_dispositions: dict[str, FindingDisposition] = Field(default_factory=dict)
    qr_code_decisions: dict[str, FieldQrCodeDecisionBody] = Field(default_factory=dict)


class OfflinePackWorkOrderSnapshotResponse(BaseModel):
    work_order_id: str
    incident_id: str
    status: str
    priority: str
    sla_due_at: str | None
    service_window_start: str | None
    service_window_end: str | None
    version: int


class OfflinePackAssetSnapshotResponse(BaseModel):
    asset_id: str
    display_name: str | None
    model_code: str | None
    serial_number: str | None
    lifecycle_status: str | None
    version: int


class OfflinePackSiteSnapshotResponse(BaseModel):
    site_id: str
    site_name: str


class OfflinePackIncidentSnapshotResponse(BaseModel):
    incident_id: str
    description: str | None
    severity: str | None
    category: str | None


class OfflinePackDiagnosisSnapshotResponse(BaseModel):
    diagnosis_run_id: str
    status: str
    version: int
    conclusion: str | None
    next_checks: list[str]
    recommended_actions: list[str]
    confidence: float | None
    citation_ids: list[str]


class OfflinePackCitationSnapshotResponse(BaseModel):
    citation_id: str
    anchor_kind: str
    page_number: int | None
    excerpt: str | None
    excerpt_checksum: str


class OfflinePackSnapshotResponse(BaseModel):
    work_order: OfflinePackWorkOrderSnapshotResponse
    asset: OfflinePackAssetSnapshotResponse
    site: OfflinePackSiteSnapshotResponse | None
    incident: OfflinePackIncidentSnapshotResponse
    diagnosis: OfflinePackDiagnosisSnapshotResponse | None
    citations: list[OfflinePackCitationSnapshotResponse]
    legal_actions: list[Literal["READ_OFFLINE_SNAPSHOT"]]


class WorkOrderOfflinePackResponse(BaseModel):
    pack_id: str
    status: Literal["ACTIVE", "SUPERSEDED", "REVOKED"]
    schema_version: Literal["field-offline-pack-v1"]
    subject_id: str
    work_order_id: str
    work_order_version: int
    asset_id: str
    asset_version: int
    assignment_id: str
    assignment_assigned_at: str
    snapshot: OfflinePackSnapshotResponse
    content_hash: str
    issued_at: str
    expires_at: str
    revoked_at: str | None
    revoked_by_subject_id: str | None
    revocation_reason: str | None
    version: int
    legal_actions: list[Literal["REVOKE"]]


class WorkOrderOfflinePackEnvelope(BaseModel):
    data: WorkOrderOfflinePackResponse
    meta: dict[str, str | bool | int]


class FieldEdgeDiagnosisPackResponse(BaseModel):
    pack_id: str
    status: Literal["ACTIVE", "SUPERSEDED"]
    schema_version: Literal["field-edge-diagnosis-pack/v1"]
    token: str
    pack_digest: str
    key_id: str
    work_order_id: str
    offline_pack_id: str
    release: FieldEdgeReleaseBinding
    prompt_bundle: FieldEdgePromptBinding
    retrieval: FieldEdgeRetrievalBinding
    issued_at: datetime
    expires_at: datetime
    version: int
    legal_actions: list[Literal["IMPORT_RESULT"]]


class FieldEdgeDiagnosisPackEnvelope(BaseModel):
    data: FieldEdgeDiagnosisPackResponse
    meta: dict[str, str | bool | int]


class FieldEdgeDiagnosisPackSummary(BaseModel):
    pack_id: str
    key_id: str
    expires_at: datetime
    release_id: str
    manifest_hash: str
    model_file: str
    model_content_hash: str
    prompt_bundle_id: str
    prompt_bundle_hash: str
    index_release_id: str
    index_content_checksum: str


class FieldEdgeDiagnosisCandidateResponse(BaseModel):
    candidate_id: str
    pack_id: str
    work_order_id: str
    status: Literal["PENDING_REVIEW", "ACCEPTED", "REJECTED"]
    pack: FieldEdgeDiagnosisPackSummary
    candidate: FieldEdgeDiagnosisCandidate
    runtime_evidence: FieldEdgeRuntimeEvidence
    result_content_hash: str
    security_policy_version: str
    reviewed_by_subject_id: str | None
    review_reason: str | None
    reviewed_at: datetime | None
    version: int
    created_at: datetime
    legal_actions: list[Literal["ACCEPT", "REJECT", "CREATE_AI_OBSERVATION"]]


class FieldEdgeDiagnosisCandidateEnvelope(BaseModel):
    data: FieldEdgeDiagnosisCandidateResponse
    meta: dict[str, str | bool | int]


class FieldEdgeDiagnosisCandidatesEnvelope(BaseModel):
    data: list[FieldEdgeDiagnosisCandidateResponse]
    meta: dict[str, str | int]


class WorkOrderPartAllocationResponse(BaseModel):
    reservation_id: str
    part_number: str
    approved_quantity: int
    status: str
    source: str
    source_record_id: str
    as_of: str
    issue_required: bool
    issue_status: str
    usage_enabled: bool
    legal_actions: list[str]
    part_issue_id: str | None
    proposal_id: str | None
    approval_id: str | None
    operation_id: str | None
    reconciliation_id: str | None
    external_issue_id: str | None
    issue_source: str | None
    issue_source_record_id: str | None
    issue_as_of: str | None


class PartIssueProposalResponse(BaseModel):
    proposal_id: str
    approval_id: str
    operation_id: str
    part_issue_id: str
    tool_id: str
    status: str
    parameters: dict[str, Any]


class PartIssueProposalEnvelope(BaseModel):
    data: PartIssueProposalResponse
    meta: dict[str, str]


class WorkOrderPartAllocationEnvelope(BaseModel):
    data: WorkOrderPartAllocationResponse
    meta: dict[str, str]


class WorkOrderPartAccountingEntryResponse(BaseModel):
    entry_id: str
    quantity: int
    occurred_at: str


class WorkOrderPartMovementResponse(BaseModel):
    movement_id: str
    operation_id: str
    proposal_id: str
    approval_id: str
    movement_kind: str
    field_entry_id: str | None
    quantity: int
    status: str
    external_movement_id: str | None
    source: str | None
    source_record_id: str | None
    as_of: str | None
    reconciliation_id: str | None
    reason: str | None
    created_at: str
    updated_at: str


class WorkOrderPartAccountingResponse(BaseModel):
    mode: str
    work_order_id: str
    work_order_version: int
    reservation_id: str
    part_issue_id: str | None
    part_number: str
    issued_quantity: int
    recorded_usage_quantity: int
    consumed_quantity: int
    returned_quantity: int
    active_consumption_quantity: int
    active_return_quantity: int
    unaccounted_quantity: int
    returnable_quantity: int
    eligible_entries: list[WorkOrderPartAccountingEntryResponse]
    movements: list[WorkOrderPartMovementResponse]
    close_ready: bool
    blocking_reasons: list[str]
    legal_actions: list[str]


class WorkOrderPartAccountingEnvelope(BaseModel):
    data: WorkOrderPartAccountingResponse
    meta: dict[str, str]


class PartMaterialMovementProposalResponse(BaseModel):
    proposal_id: str
    approval_id: str
    operation_id: str
    movement_id: str
    tool_id: str
    status: str
    parameters: dict[str, Any]


class PartMaterialMovementProposalEnvelope(BaseModel):
    data: PartMaterialMovementProposalResponse
    meta: dict[str, str]


class WorkOrderRepairRoundResponse(BaseModel):
    round_number: int
    status: str
    completion_id: str | None
    completed_by_subject_id: str | None
    root_cause: str | None
    actions: list[str]
    evidence_ids: list[str]
    part_reservation_ids: list[str]
    cost_amount: str | None
    customer_confirmation: str | None
    field_entry_sequence_start: int | None
    field_entry_sequence_end: int | None
    completed_at: str | None
    verification_id: str | None
    verifier_subject_id: str | None
    verification_passed: bool | None
    verification_reason: str | None
    verified_at: str | None
    rework_id: str | None
    rework_status: str | None
    rework_reason: str | None
    entry_sequence_checkpoint: int | None
    rework_opened_at: str | None
    rework_resolved_at: str | None


class WorkOrderRepairHistoryResponse(BaseModel):
    work_order_id: str
    current_round: int
    rounds: list[WorkOrderRepairRoundResponse]


class WorkOrderRepairHistoryEnvelope(BaseModel):
    data: WorkOrderRepairHistoryResponse
    meta: dict[str, str]


class WorkOrderClosurePartResponse(BaseModel):
    reservation_id: str
    part_number: str
    quantity: int
    status: str
    source: str
    as_of: str


class WorkOrderClosureReportResponse(BaseModel):
    report_contract_version: str
    closure_facts_digest: str
    work_order_id: str
    incident_id: str
    asset_id: str
    asset_display_name: str | None
    model_code: str | None
    serial_number: str | None
    site_id: str | None
    site_name: str | None
    closed_at: str
    closed_by_subject_id: str
    close_reason: str | None
    close_operation_id: str | None
    close_proposal_id: str | None
    repair_round_count: int
    rework_count: int
    completion_id: str
    completion_round_number: int
    completed_by_subject_id: str
    completed_at: str
    root_cause: str
    actions: list[str]
    evidence_ids: list[str]
    parts: list[WorkOrderClosurePartResponse]
    cost_amount: str | None
    field_customer_confirmation: str | None
    verification_id: str
    verifier_subject_id: str
    verification_reason: str
    verified_at: str
    customer_result_confirmation_id: str | None
    customer_result_accepted: bool
    customer_confirmed_by_subject_id: str | None
    customer_confirmed_at: str | None
    customer_satisfaction_rating: int | None


class WorkOrderClosureReportEnvelope(BaseModel):
    data: WorkOrderClosureReportResponse
    meta: dict[str, str]


class DispatchFactAuthorityResponse(BaseModel):
    contract_version: str
    domain: str
    owner: str
    field_sources: dict[str, str]


class DispatchFactResponse(BaseModel):
    tool_id: str
    tool_call_id: str
    source: str
    source_record_id: str
    as_of: str
    authority: DispatchFactAuthorityResponse
    values: dict[str, str | int | bool | None]


class DispatchDependencyFailureResponse(BaseModel):
    tool_id: str
    reason: str


class DispatchWorkOrderResponse(WorkOrderResponse):
    asset_id: str
    asset_display_name: str | None
    model_code: str | None
    site_id: str | None
    site_name: str | None
    incident_description: str
    schedule: DispatchFactResponse | None
    inventory: DispatchFactResponse | None
    fact_failures: list[DispatchDependencyFailureResponse]


class DispatchMeta(BaseModel):
    request_id: str
    total: int
    limit: int
    offset: int


class DispatchWorkOrdersEnvelope(BaseModel):
    data: list[DispatchWorkOrderResponse]
    meta: DispatchMeta


class WorkOrdersEnvelope(BaseModel):
    data: list[WorkOrderCenterItemResponse]
    meta: DispatchMeta


@router.get("/dispatch/work-orders", response_model=DispatchWorkOrdersEnvelope)
def list_dispatch_work_orders(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    tool_gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    status: Annotated[WorkOrderStatus | None, Query()] = None,
    assigned_subject_id: Annotated[
        str | None,
        Query(min_length=1, max_length=128),
    ] = None,
    priority: Annotated[WorkOrderPriority | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DispatchWorkOrdersEnvelope:
    try:
        items, total = WorkOrderService(database, authorizer).list_dispatch_queue(
            identity,
            status=status,
            assigned_subject_id=assigned_subject_id,
            priority=priority,
            limit=limit,
            offset=offset,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise AppError(
            403,
            "authorization_denied",
            "authorization",
            "Action is not allowed",
        ) from exc
    return DispatchWorkOrdersEnvelope(
        data=[
            _dispatch_view(
                item,
                identity,
                authorizer,
                tool_gateway,
                request_id=request.state.request_id,
            )
            for item in items
        ],
        meta=DispatchMeta(
            request_id=request.state.request_id,
            total=total,
            limit=limit,
            offset=offset,
        ),
    )


@router.get("/work-orders", response_model=WorkOrdersEnvelope)
def list_work_orders(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    status: Annotated[WorkOrderStatus | None, Query()] = None,
    assigned_subject_id: Annotated[
        str | None,
        Query(min_length=1, max_length=128),
    ] = None,
    priority: Annotated[WorkOrderPriority | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> WorkOrdersEnvelope:
    try:
        items, total = WorkOrderService(database, authorizer).list_dispatch_queue(
            identity,
            status=status,
            assigned_subject_id=assigned_subject_id,
            priority=priority,
            limit=limit,
            offset=offset,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise AppError(
            403,
            "authorization_denied",
            "authorization",
            "Action is not allowed",
        ) from exc
    return WorkOrdersEnvelope(
        data=[_center_view(item, identity, authorizer) for item in items],
        meta=DispatchMeta(
            request_id=request.state.request_id,
            total=total,
            limit=limit,
            offset=offset,
        ),
    )


@router.get("/field/work-orders", response_model=WorkOrdersEnvelope, tags=["field-service"])
def list_my_field_work_orders(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    status: Annotated[WorkOrderStatus | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> WorkOrdersEnvelope:
    try:
        authorizer.require(
            identity,
            Action.EXECUTE_WORK_ORDER,
            ResourceContext(identity.tenant_id),
            request_id=request.state.request_id,
        )
        items, total = WorkOrderService(database, authorizer).list_dispatch_queue(
            identity,
            status=status,
            assigned_subject_id=identity.subject_id,
            limit=limit,
            offset=offset,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise AppError(
            403,
            "authorization_denied",
            "authorization",
            "Action is not allowed",
        ) from exc
    return WorkOrdersEnvelope(
        data=[_center_view(item, identity, authorizer) for item in items],
        meta=DispatchMeta(
            request_id=request.state.request_id,
            total=total,
            limit=limit,
            offset=offset,
        ),
    )


@router.get(
    "/field/work-orders/{work_order_id}/voice-guidance",
    response_model=FieldVoiceGuidanceEnvelope,
    tags=["field-service"],
)
def get_field_voice_guidance(
    work_order_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> FieldVoiceGuidanceEnvelope:
    try:
        guidance = FieldVoiceGuidanceService(database, authorizer).get(
            identity,
            work_order_id,
            request_id=request.state.request_id,
        )
    except (WorkOrderNotVisible, AuthorizationDenied, WorkOrderConflict) as exc:
        _raise_field_error(exc)
    return FieldVoiceGuidanceEnvelope(
        data=_field_voice_guidance_response(guidance),
        meta={"request_id": request.state.request_id},
    )


@router.get(
    "/field/work-orders/{work_order_id}/part-allocation",
    response_model=WorkOrderPartAllocationEnvelope,
    tags=["field-service"],
)
def get_field_part_allocation(
    work_order_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> WorkOrderPartAllocationEnvelope:
    try:
        allocation = WorkOrderService(database, authorizer).get_part_allocation(
            identity,
            work_order_id,
            request_id=request.state.request_id,
        )
    except (WorkOrderNotVisible, AuthorizationDenied, WorkOrderConflict) as exc:
        _raise_field_error(exc)
    return WorkOrderPartAllocationEnvelope(
        data=_part_allocation_view(allocation),
        meta={"request_id": request.state.request_id},
    )


@router.get(
    "/field/work-orders/{work_order_id}/part-accounting",
    response_model=WorkOrderPartAccountingEnvelope,
    tags=["field-service"],
)
def get_field_part_accounting(
    work_order_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> WorkOrderPartAccountingEnvelope:
    try:
        accounting = WorkOrderService(database, authorizer).get_part_accounting(
            identity,
            work_order_id,
            request_id=request.state.request_id,
        )
    except (WorkOrderNotVisible, AuthorizationDenied, WorkOrderConflict) as exc:
        _raise_field_error(exc)
    return WorkOrderPartAccountingEnvelope(
        data=_part_accounting_view(accounting),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/field/work-orders/{work_order_id}/part-issue-proposals",
    response_model=PartIssueProposalEnvelope,
    status_code=201,
    tags=["field-service"],
)
async def propose_field_part_issue(
    work_order_id: str,
    body: PartIssueProposalBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=8)],
) -> PartIssueProposalEnvelope:
    try:
        proposal = gateway.propose_part_issue(
            identity,
            work_order_id=work_order_id,
            work_order_version=body.work_order_version,
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise AppError(
            403, "authorization_denied", "authorization", "Action is not allowed"
        ) from exc
    except (ApprovalNotVisible, WorkOrderNotVisible) as exc:
        raise AppError(
            404,
            "resource_not_found_or_not_visible",
            "not_found",
            "Resource not found or not visible",
        ) from exc
    except PartIssueProposalIdempotencyConflict as exc:
        raise AppError(
            409,
            "idempotency_conflict",
            "conflict",
            "Idempotency key was already used for another part issue request",
        ) from exc
    except PartIssueProposalConflict as exc:
        raise AppError(409, exc.reason, "conflict", "Part issue proposal is not allowed") from exc
    return PartIssueProposalEnvelope(
        data=PartIssueProposalResponse(
            proposal_id=proposal.proposal_id,
            approval_id=proposal.approval_id,
            operation_id=proposal.operation_id,
            part_issue_id=proposal.part_issue_id,
            tool_id="parts.issue",
            status="PENDING_APPROVAL",
            parameters=proposal.parameters,
        ),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/field/work-orders/{work_order_id}/part-consumption-proposals",
    response_model=PartMaterialMovementProposalEnvelope,
    status_code=201,
    tags=["field-service"],
)
async def propose_field_part_consumption(
    work_order_id: str,
    body: PartConsumptionProposalBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=8)],
) -> PartMaterialMovementProposalEnvelope:
    return _propose_field_part_movement(
        gateway,
        identity,
        work_order_id=work_order_id,
        work_order_version=body.work_order_version,
        movement_kind="CONSUME",
        field_entry_id=body.field_entry_id,
        quantity=None,
        idempotency_key=idempotency_key,
        request_id=request.state.request_id,
    )


@router.post(
    "/field/work-orders/{work_order_id}/part-return-proposals",
    response_model=PartMaterialMovementProposalEnvelope,
    status_code=201,
    tags=["field-service"],
)
async def propose_field_part_return(
    work_order_id: str,
    body: PartReturnProposalBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=8)],
) -> PartMaterialMovementProposalEnvelope:
    return _propose_field_part_movement(
        gateway,
        identity,
        work_order_id=work_order_id,
        work_order_version=body.work_order_version,
        movement_kind="RETURN",
        field_entry_id=None,
        quantity=body.quantity,
        idempotency_key=idempotency_key,
        request_id=request.state.request_id,
    )


def _propose_field_part_movement(
    gateway: ToolGateway,
    identity: IdentityContext,
    *,
    work_order_id: str,
    work_order_version: int,
    movement_kind: Literal["CONSUME", "RETURN"],
    field_entry_id: str | None,
    quantity: int | None,
    idempotency_key: str,
    request_id: str,
) -> PartMaterialMovementProposalEnvelope:
    try:
        proposal = (
            gateway.propose_part_consumption(
                identity,
                work_order_id=work_order_id,
                work_order_version=work_order_version,
                field_entry_id=field_entry_id or "",
                idempotency_key=idempotency_key,
                request_id=request_id,
            )
            if movement_kind == "CONSUME"
            else gateway.propose_part_return(
                identity,
                work_order_id=work_order_id,
                work_order_version=work_order_version,
                quantity=quantity or 0,
                idempotency_key=idempotency_key,
                request_id=request_id,
            )
        )
    except AuthorizationDenied as exc:
        raise AppError(
            403, "authorization_denied", "authorization", "Action is not allowed"
        ) from exc
    except (ApprovalNotVisible, WorkOrderNotVisible) as exc:
        raise AppError(
            404,
            "resource_not_found_or_not_visible",
            "not_found",
            "Resource not found or not visible",
        ) from exc
    except PartMovementProposalIdempotencyConflict as exc:
        raise AppError(
            409,
            "idempotency_conflict",
            "conflict",
            "Idempotency key was already used for another material movement",
        ) from exc
    except PartMovementProposalConflict as exc:
        raise AppError(
            409, exc.reason, "conflict", "Part movement proposal is not allowed"
        ) from exc
    return PartMaterialMovementProposalEnvelope(
        data=PartMaterialMovementProposalResponse(
            proposal_id=proposal.proposal_id,
            approval_id=proposal.approval_id,
            operation_id=proposal.operation_id,
            movement_id=proposal.movement_id,
            tool_id="parts.consume" if movement_kind == "CONSUME" else "parts.return",
            status="PENDING_APPROVAL",
            parameters=proposal.parameters,
        ),
        meta={"request_id": request_id},
    )


@router.post(
    "/field/work-orders/{work_order_id}/offline-packs",
    response_model=WorkOrderOfflinePackEnvelope,
    status_code=201,
    tags=["field-service"],
)
def create_field_offline_pack(
    work_order_id: str,
    request: Request,
    response: Response,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=8, max_length=255),
    ],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> WorkOrderOfflinePackEnvelope:
    try:
        result = WorkOrderOfflinePackService(database, authorizer).create(
            identity,
            work_order_id,
            expected_work_order_version=_version(if_match),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (
        OfflinePackNotAvailable,
        WorkOrderNotVisible,
        AuthorizationDenied,
        WorkOrderConflict,
    ) as exc:
        _raise_offline_pack_error(exc)
    response.status_code = 201 if result.created else 200
    return WorkOrderOfflinePackEnvelope(
        data=_offline_pack_view(result.pack),
        meta={
            "request_id": request.state.request_id,
            "created": result.created,
            "version": result.pack.version,
        },
    )


@router.get(
    "/field/work-orders/{work_order_id}/offline-packs/current",
    response_model=WorkOrderOfflinePackEnvelope,
    tags=["field-service"],
)
def get_current_field_offline_pack(
    work_order_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> WorkOrderOfflinePackEnvelope:
    try:
        pack = WorkOrderOfflinePackService(database, authorizer).current(
            identity,
            work_order_id,
            request_id=request.state.request_id,
        )
    except (
        OfflinePackNotAvailable,
        WorkOrderNotVisible,
        AuthorizationDenied,
        WorkOrderConflict,
    ) as exc:
        _raise_offline_pack_error(exc)
    return WorkOrderOfflinePackEnvelope(
        data=_offline_pack_view(pack),
        meta={"request_id": request.state.request_id, "version": pack.version},
    )


@router.post(
    "/field/work-orders/{work_order_id}/offline-packs/{pack_id}/revoke",
    response_model=WorkOrderOfflinePackEnvelope,
    tags=["field-service"],
)
def revoke_field_offline_pack(
    work_order_id: str,
    pack_id: str,
    body: RevokeOfflinePackBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> WorkOrderOfflinePackEnvelope:
    try:
        pack = WorkOrderOfflinePackService(database, authorizer).revoke(
            identity,
            work_order_id,
            pack_id,
            expected_pack_version=_version(if_match),
            reason=body.reason,
            request_id=request.state.request_id,
        )
    except (
        OfflinePackNotAvailable,
        WorkOrderNotVisible,
        AuthorizationDenied,
        WorkOrderConflict,
    ) as exc:
        _raise_offline_pack_error(exc)
    return WorkOrderOfflinePackEnvelope(
        data=_offline_pack_view(pack),
        meta={"request_id": request.state.request_id, "version": pack.version},
    )


@router.post(
    "/field/work-orders/{work_order_id}/edge-diagnosis-packs",
    response_model=FieldEdgeDiagnosisPackEnvelope,
    status_code=201,
    tags=["field-service"],
)
async def create_field_edge_diagnosis_pack(
    work_order_id: str,
    request: Request,
    response: Response,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    signer: Annotated[Ed25519PackSigner | None, Depends(get_optional_edge_pack_signer)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=8, max_length=255),
    ],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> FieldEdgeDiagnosisPackEnvelope:
    service = _edge_diagnosis_service(request, database, authorizer, signer)
    try:
        pack, created = service.issue(
            identity,
            work_order_id,
            expected_work_order_version=_version(if_match),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except EDGE_DIAGNOSIS_ERRORS as exc:
        _raise_edge_diagnosis_error(exc)  # type: ignore[arg-type]
    response.status_code = 201 if created else 200
    return FieldEdgeDiagnosisPackEnvelope(
        data=_edge_pack_view(pack),
        meta={
            "request_id": request.state.request_id,
            "created": created,
            "version": pack.version,
        },
    )


@router.post(
    "/field/work-orders/{work_order_id}/edge-diagnosis-candidates/import",
    response_model=FieldEdgeDiagnosisCandidateEnvelope,
    status_code=201,
    tags=["field-service"],
)
async def import_field_edge_diagnosis_candidate(
    work_order_id: str,
    body: ImportEdgeDiagnosisBody,
    request: Request,
    response: Response,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    signer: Annotated[Ed25519PackSigner | None, Depends(get_optional_edge_pack_signer)],
) -> FieldEdgeDiagnosisCandidateEnvelope:
    service = _edge_diagnosis_service(request, database, authorizer, signer)
    try:
        candidate, created = service.import_result(
            identity,
            work_order_id,
            client_operation_id=body.client_operation_id,
            result=body.result,
            request_id=request.state.request_id,
        )
    except EDGE_DIAGNOSIS_ERRORS as exc:
        _raise_edge_diagnosis_error(exc)  # type: ignore[arg-type]
    response.status_code = 201 if created else 200
    return FieldEdgeDiagnosisCandidateEnvelope(
        data=_edge_candidate_view(database, identity, candidate),
        meta={
            "request_id": request.state.request_id,
            "created": created,
            "version": candidate.version,
        },
    )


@router.get(
    "/field/work-orders/{work_order_id}/edge-diagnosis-candidates",
    response_model=FieldEdgeDiagnosisCandidatesEnvelope,
    tags=["field-service"],
)
async def list_field_edge_diagnosis_candidates(
    work_order_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    signer: Annotated[Ed25519PackSigner | None, Depends(get_optional_edge_pack_signer)],
) -> FieldEdgeDiagnosisCandidatesEnvelope:
    service = _edge_diagnosis_service(request, database, authorizer, signer)
    try:
        items = service.list_candidates(
            identity,
            work_order_id,
            request_id=request.state.request_id,
        )
    except EDGE_DIAGNOSIS_ERRORS as exc:
        _raise_edge_diagnosis_error(exc)  # type: ignore[arg-type]
    return FieldEdgeDiagnosisCandidatesEnvelope(
        data=[_edge_candidate_view(database, identity, item) for item in items],
        meta={"request_id": request.state.request_id, "total": len(items)},
    )


@router.post(
    "/field/work-orders/{work_order_id}/edge-diagnosis-candidates/{candidate_id}/decision",
    response_model=FieldEdgeDiagnosisCandidateEnvelope,
    tags=["field-service"],
)
async def decide_field_edge_diagnosis_candidate(
    work_order_id: str,
    candidate_id: str,
    body: DecideEdgeDiagnosisBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    signer: Annotated[Ed25519PackSigner | None, Depends(get_optional_edge_pack_signer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> FieldEdgeDiagnosisCandidateEnvelope:
    service = _edge_diagnosis_service(request, database, authorizer, signer)
    try:
        item = service.decide(
            identity,
            work_order_id,
            candidate_id,
            expected_version=_version(if_match),
            decision=body.decision,
            reason=body.reason,
            request_id=request.state.request_id,
        )
    except EDGE_DIAGNOSIS_ERRORS as exc:
        _raise_edge_diagnosis_error(exc)  # type: ignore[arg-type]
    return FieldEdgeDiagnosisCandidateEnvelope(
        data=_edge_candidate_view(database, identity, item),
        meta={"request_id": request.state.request_id, "version": item.version},
    )


@router.get(
    "/field/work-orders/{work_order_id}/evidence-uploads",
    response_model=FieldEvidenceUploadsEnvelope,
    tags=["field-service"],
)
async def list_field_evidence_uploads(
    work_order_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    object_store: Annotated[TenantObjectStore, Depends(get_object_store)],
) -> FieldEvidenceUploadsEnvelope:
    try:
        uploads = FieldEvidenceService(database, authorizer, object_store).list(
            identity,
            work_order_id,
            request_id=request.state.request_id,
        )
    except (WorkOrderNotVisible, AuthorizationDenied, WorkOrderConflict) as exc:
        _raise_field_error(exc)
    return FieldEvidenceUploadsEnvelope(
        data=[_field_evidence_upload_view(upload) for upload in uploads],
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/field/work-orders/{work_order_id}/evidence-uploads",
    response_model=FieldEvidenceUploadEnvelope,
    status_code=201,
    tags=["field-service"],
)
async def upload_field_evidence(
    work_order_id: str,
    request: Request,
    response: Response,
    content: Annotated[bytes, Body(media_type="application/octet-stream")],
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    object_store: Annotated[TenantObjectStore, Depends(get_object_store)],
    declared_mime: Annotated[str, Header(alias="Content-Type")],
    if_match: Annotated[str, Header(alias="If-Match")],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=8, max_length=128),
    ],
    content_hash: Annotated[
        str,
        Header(alias="X-Content-SHA256", min_length=71, max_length=71),
    ],
) -> FieldEvidenceUploadEnvelope:
    try:
        result = await FieldEvidenceService(database, authorizer, object_store).upload(
            identity,
            work_order_id,
            expected_work_order_version=_version(if_match),
            client_operation_id=idempotency_key,
            expected_content_hash=content_hash,
            declared_mime=declared_mime,
            content=content,
            request_id=request.state.request_id,
        )
    except FieldEvidenceValidationError as exc:
        raise AppError(
            422,
            exc.reason,
            "validation",
            "Field evidence content binding is invalid",
        ) from exc
    except MediaUploadRejected as exc:
        status_code = 415 if exc.reason_code == "mime_type_mismatch" else 422
        raise AppError(
            status_code,
            exc.reason_code,
            "validation",
            "Field evidence upload failed media validation",
            details={"scan_state": exc.media.scan_state.value},
        ) from exc
    except (WorkOrderNotVisible, AuthorizationDenied, WorkOrderConflict) as exc:
        _raise_field_error(exc)
    response.status_code = 201 if result.created else 200
    return FieldEvidenceUploadEnvelope(
        data=_field_evidence_upload_view(result.upload),
        meta={
            "request_id": request.state.request_id,
            "created": result.created,
        },
    )


@router.post(
    "/field/work-orders/{work_order_id}/evidence-uploads/{evidence_upload_id}/recognitions",
    response_model=FieldRecognitionRunEnvelope,
    status_code=202,
    tags=["field-service"],
)
async def start_field_evidence_recognition(
    work_order_id: str,
    evidence_upload_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dispatcher: Annotated[RecognitionDispatcher, Depends(get_recognition_dispatcher)],
    if_match: Annotated[str, Header(alias="If-Match")],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=8, max_length=128),
    ],
    body: Annotated[FieldRecognitionStartBody | None, Body()] = None,
) -> FieldRecognitionRunEnvelope:
    service = RecognitionService(database, authorizer)
    try:
        result = service.enqueue_field(
            identity,
            work_order_id,
            evidence_upload_id,
            expected_work_order_version=_version(if_match),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
            temporal_analysis=body.temporal_analysis if body is not None else False,
        )
        if result.run.status == "QUEUED":
            await dispatcher.dispatch(
                RecognitionActivityInput(
                    recognition_run_id=result.run.recognition_run_id,
                    identity=identity,
                    request_id=request.state.request_id,
                    workflow_id=result.run.workflow_id,
                )
            )
    except (WorkOrderNotVisible, ResourceNotVisible):
        _raise_field_error(WorkOrderNotVisible())
    except (AuthorizationDenied, WorkOrderConflict) as exc:
        _raise_field_error(exc)
    except IdempotencyConflict as exc:
        raise AppError(
            409,
            "field_recognition_idempotency_mismatch",
            "conflict",
            "Field recognition idempotency binding changed",
        ) from exc
    except RecognitionDispatchUnavailable as exc:
        raise AppError(
            503,
            "recognition_workflow_unavailable",
            "dependency",
            "Recognition workflow unavailable",
            retryable=True,
        ) from exc
    return _field_recognition_run_envelope(
        result.run,
        created=result.created,
        request_id=request.state.request_id,
    )


@router.get(
    "/field/work-orders/{work_order_id}/evidence-uploads/{evidence_upload_id}/recognitions",
    response_model=EvidenceEnvelope,
    tags=["field-service"],
)
async def get_field_evidence_recognition(
    work_order_id: str,
    evidence_upload_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> EvidenceEnvelope:
    try:
        bundle = RecognitionService(database, authorizer).get_field_evidence(
            identity,
            work_order_id,
            evidence_upload_id,
            request_id=request.state.request_id,
        )
    except (WorkOrderNotVisible, ResourceNotVisible):
        _raise_field_error(WorkOrderNotVisible())
    except (AuthorizationDenied, WorkOrderConflict) as exc:
        _raise_field_error(exc)
    return _evidence_envelope(
        bundle,
        request.state.request_id,
        video_keyframe_base_path=(
            f"/api/v1/field/work-orders/{work_order_id}/evidence-uploads/"
            f"{evidence_upload_id}/recognitions/video-keyframes"
        ),
    )


@router.get(
    "/field/work-orders/{work_order_id}/evidence-uploads/{evidence_upload_id}/"
    "recognitions/video-keyframes/{frame_id}/image",
    response_class=Response,
    tags=["field-service"],
)
async def read_field_evidence_video_keyframe(
    work_order_id: str,
    evidence_upload_id: str,
    frame_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    object_store: Annotated[TenantObjectStore, Depends(get_object_store)],
) -> Response:
    try:
        content = await RecognitionService(database, authorizer).read_field_video_keyframe(
            identity,
            work_order_id,
            evidence_upload_id,
            frame_id,
            object_store=object_store,
            request_id=request.state.request_id,
        )
    except (WorkOrderNotVisible, ResourceNotVisible):
        _raise_field_error(WorkOrderNotVisible())
    except (AuthorizationDenied, WorkOrderConflict) as exc:
        _raise_field_error(exc)
    return Response(
        content=content,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post(
    "/field/work-orders/{work_order_id}/evidence-uploads/{evidence_upload_id}/recognitions/confirmation",
    response_model=EvidenceEnvelope,
    tags=["field-service"],
)
async def confirm_field_evidence_recognition(
    work_order_id: str,
    evidence_upload_id: str,
    body: FieldRecognitionConfirmationBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> EvidenceEnvelope:
    try:
        bundle = RecognitionService(database, authorizer).confirm_field(
            identity,
            work_order_id,
            evidence_upload_id,
            corrections=body.corrections,
            finding_dispositions=body.finding_dispositions,
            ocr_block_decisions={
                block_id: (decision.disposition, decision.corrected_text)
                for block_id, decision in body.ocr_block_decisions.items()
            },
            transcript_decisions={
                segment_id: (decision.disposition, decision.corrected_text)
                for segment_id, decision in body.transcript_decisions.items()
            },
            video_event_dispositions=body.video_event_dispositions,
            qr_code_decisions={
                candidate_id: (decision.disposition, decision.corrected_text)
                for candidate_id, decision in body.qr_code_decisions.items()
            },
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        )
    except (WorkOrderNotVisible, ResourceNotVisible):
        _raise_field_error(WorkOrderNotVisible())
    except (AuthorizationDenied, WorkOrderConflict) as exc:
        _raise_field_error(exc)
    except EvidenceVersionConflict as exc:
        raise AppError(
            409,
            "evidence_version_conflict",
            "conflict",
            "Evidence version conflict",
            details={"current_version": exc.current_version},
        ) from exc
    except EvidenceConfirmationRequired as exc:
        raise AppError(
            409,
            exc.reason_code,
            "conflict",
            "Field recognition review is incomplete",
        ) from exc
    return _evidence_envelope(
        bundle,
        request.state.request_id,
        video_keyframe_base_path=(
            f"/api/v1/field/work-orders/{work_order_id}/evidence-uploads/"
            f"{evidence_upload_id}/recognitions/video-keyframes"
        ),
    )


@router.get(
    "/field/work-orders/{work_order_id}/entries",
    response_model=WorkOrderFieldEntriesEnvelope,
    tags=["field-service"],
)
def list_field_entries(
    work_order_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> WorkOrderFieldEntriesEnvelope:
    try:
        entries = WorkOrderService(database, authorizer).list_field_entries(
            identity,
            work_order_id,
            request_id=request.state.request_id,
        )
    except (WorkOrderNotVisible, AuthorizationDenied, WorkOrderConflict) as exc:
        _raise_field_error(exc)
    return WorkOrderFieldEntriesEnvelope(
        data=[_field_entry_view(entry) for entry in entries],
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/field/work-orders/{work_order_id}/entries",
    response_model=WorkOrderFieldEntryEnvelope,
    tags=["field-service"],
)
def append_field_entry(
    work_order_id: str,
    body: FieldEntryBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> WorkOrderFieldEntryEnvelope:
    try:
        entry = WorkOrderService(database, authorizer).append_field_entry(
            identity,
            work_order_id,
            client_operation_id=body.client_operation_id,
            entry_type=body.entry_type,
            payload=body.entry_payload(),
            occurred_at=body.occurred_at,
            request_id=request.state.request_id,
        )
    except (WorkOrderNotVisible, AuthorizationDenied, WorkOrderConflict) as exc:
        _raise_field_error(exc)
    return WorkOrderFieldEntryEnvelope(
        data=_field_entry_view(entry),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/field/work-orders/{work_order_id}/complete",
    response_model=WorkOrderEnvelope,
    tags=["field-service"],
)
def complete_field_work_order(
    work_order_id: str,
    body: FieldCompletionBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> WorkOrderEnvelope:
    return _call(
        lambda: WorkOrderService(database, authorizer).complete_from_field_entries(
            identity,
            work_order_id,
            expected_version=_version(if_match),
            root_cause=body.root_cause,
            cost_amount=body.cost_amount,
            request_id=request.state.request_id,
        ),
        request,
        identity,
        authorizer,
    )


@router.get("/work-orders/{work_order_id}", response_model=WorkOrderEnvelope)
def get_work_order(
    work_order_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> WorkOrderEnvelope:
    return _call(
        lambda: WorkOrderService(database, authorizer).get(
            identity, work_order_id, request_id=request.state.request_id
        ),
        request,
        identity,
        authorizer,
    )


@router.get(
    "/work-orders/{work_order_id}/repair-history",
    response_model=WorkOrderRepairHistoryEnvelope,
)
def get_work_order_repair_history(
    work_order_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> WorkOrderRepairHistoryEnvelope:
    try:
        history = WorkOrderService(database, authorizer).repair_history(
            identity,
            work_order_id,
            request_id=request.state.request_id,
        )
    except WorkOrderNotVisible as exc:
        raise AppError(
            404,
            "resource_not_found_or_not_visible",
            "not_found",
            "Resource not found or not visible",
        ) from exc
    except AuthorizationDenied as exc:
        raise AppError(
            403,
            "authorization_denied",
            "authorization",
            "Action is not allowed",
        ) from exc
    return WorkOrderRepairHistoryEnvelope(
        data=_repair_history_view(history),
        meta={"request_id": request.state.request_id},
    )


@router.get(
    "/work-orders/{work_order_id}/closure-report",
    response_model=WorkOrderClosureReportEnvelope,
)
def get_work_order_closure_report(
    work_order_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> WorkOrderClosureReportEnvelope:
    try:
        report = WorkOrderService(database, authorizer).closure_report(
            identity,
            work_order_id,
            request_id=request.state.request_id,
        )
    except WorkOrderNotVisible as exc:
        raise AppError(
            404,
            "resource_not_found_or_not_visible",
            "not_found",
            "Resource not found or not visible",
        ) from exc
    except AuthorizationDenied as exc:
        raise AppError(
            403,
            "authorization_denied",
            "authorization",
            "Action is not allowed",
        ) from exc
    except WorkOrderConflict as exc:
        raise AppError(
            409,
            exc.reason,
            "conflict",
            "WorkOrder closure report is not available",
            details={"current_version": exc.current_version},
        ) from exc
    return WorkOrderClosureReportEnvelope(
        data=_closure_report_view(report),
        meta={"request_id": request.state.request_id},
    )


@router.post("/work-orders/{work_order_id}/assign", response_model=WorkOrderEnvelope)
def assign(
    work_order_id: str,
    body: AssignBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> WorkOrderEnvelope:
    return _call(
        lambda: WorkOrderService(database, authorizer).assign(
            identity,
            work_order_id,
            assignee_subject_id=body.assignee_subject_id,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        ),
        request,
        identity,
        authorizer,
    )


@router.post("/work-orders/{work_order_id}/hold", response_model=WorkOrderEnvelope)
def hold(
    work_order_id: str,
    body: HoldBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> WorkOrderEnvelope:
    return _call(
        lambda: WorkOrderService(database, authorizer).hold(
            identity,
            work_order_id,
            expected_version=_version(if_match),
            reason=body.reason,
            recovery_condition=body.recovery_condition,
            request_id=request.state.request_id,
        ),
        request,
        identity,
        authorizer,
    )


@router.post("/work-orders/{work_order_id}/resume", response_model=WorkOrderEnvelope)
def resume(
    work_order_id: str,
    body: ResumeBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> WorkOrderEnvelope:
    return _call(
        lambda: WorkOrderService(database, authorizer).resume(
            identity,
            work_order_id,
            expected_version=_version(if_match),
            reason=body.reason,
            request_id=request.state.request_id,
        ),
        request,
        identity,
        authorizer,
    )


@router.post("/work-orders/{work_order_id}/escalate", response_model=WorkOrderEnvelope)
def escalate(
    work_order_id: str,
    body: EscalateBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> WorkOrderEnvelope:
    return _call(
        lambda: WorkOrderService(database, authorizer).escalate(
            identity,
            work_order_id,
            expected_version=_version(if_match),
            reason=body.reason,
            recovery_condition=body.recovery_condition,
            responsible_subject_id=body.responsible_subject_id,
            request_id=request.state.request_id,
        ),
        request,
        identity,
        authorizer,
    )


@router.post("/work-orders/{work_order_id}/replan", response_model=WorkOrderEnvelope)
def replan(
    work_order_id: str,
    body: ReplanBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> WorkOrderEnvelope:
    return _call(
        lambda: WorkOrderService(database, authorizer).replan(
            identity,
            work_order_id,
            expected_version=_version(if_match),
            target_status=body.target_status,
            reason=body.reason,
            responsible_subject_id=body.responsible_subject_id,
            service_window_start=body.service_window_start,
            service_window_end=body.service_window_end,
            sla_due_at=body.sla_due_at,
            request_id=request.state.request_id,
        ),
        request,
        identity,
        authorizer,
    )


@router.post("/work-orders/{work_order_id}/reschedule", response_model=WorkOrderEnvelope)
def reschedule(
    work_order_id: str,
    body: RescheduleBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> WorkOrderEnvelope:
    return _call(
        lambda: WorkOrderService(database, authorizer).reschedule(
            identity,
            work_order_id,
            expected_version=_version(if_match),
            priority=body.priority,
            reason=body.reason,
            service_window_start=body.service_window_start,
            service_window_end=body.service_window_end,
            sla_due_at=body.sla_due_at,
            request_id=request.state.request_id,
        ),
        request,
        identity,
        authorizer,
    )


@router.get(
    "/work-orders/{work_order_id}/controls",
    response_model=WorkOrderControlsEnvelope,
)
def list_work_order_controls(
    work_order_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> WorkOrderControlsEnvelope:
    try:
        controls = WorkOrderService(database, authorizer).list_controls(
            identity,
            work_order_id,
            request_id=request.state.request_id,
        )
    except WorkOrderNotVisible as exc:
        raise AppError(
            404,
            "resource_not_found_or_not_visible",
            "not_found",
            "Resource not found or not visible",
        ) from exc
    except AuthorizationDenied as exc:
        raise AppError(
            403,
            "authorization_denied",
            "authorization",
            "Action is not allowed",
        ) from exc
    return WorkOrderControlsEnvelope(
        data=[_control_view(control) for control in controls],
        meta={"request_id": request.state.request_id},
    )


@router.post("/work-orders/{work_order_id}/accept", response_model=WorkOrderEnvelope)
def accept(
    work_order_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> WorkOrderEnvelope:
    return _call(
        lambda: WorkOrderService(database, authorizer).accept(
            identity,
            work_order_id,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        ),
        request,
        identity,
        authorizer,
    )


@router.post("/work-orders/{work_order_id}/start", response_model=WorkOrderEnvelope)
def start(
    work_order_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> WorkOrderEnvelope:
    return _call(
        lambda: WorkOrderService(database, authorizer).start(
            identity,
            work_order_id,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        ),
        request,
        identity,
        authorizer,
    )


@router.post("/work-orders/{work_order_id}/complete", response_model=WorkOrderEnvelope)
def complete(
    work_order_id: str,
    body: CompleteBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> WorkOrderEnvelope:
    return _call(
        lambda: WorkOrderService(database, authorizer).complete(
            identity,
            work_order_id,
            expected_version=_version(if_match),
            root_cause=body.root_cause,
            actions=body.actions,
            evidence_ids=body.evidence_ids,
            part_reservation_ids=body.part_reservation_ids,
            cost_amount=body.cost_amount,
            customer_confirmation=body.customer_confirmation,
            request_id=request.state.request_id,
        ),
        request,
        identity,
        authorizer,
    )


@router.post("/work-orders/{work_order_id}/verify", response_model=WorkOrderEnvelope)
def verify(
    work_order_id: str,
    body: VerifyBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> WorkOrderEnvelope:
    return _call(
        lambda: WorkOrderService(database, authorizer).verify(
            identity,
            work_order_id,
            expected_version=_version(if_match),
            passed=body.passed,
            reason=body.reason,
            request_id=request.state.request_id,
        ),
        request,
        identity,
        authorizer,
    )


@router.post("/work-orders/{work_order_id}/close", response_model=WorkOrderEnvelope)
def close(
    work_order_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> WorkOrderEnvelope:
    return _call(
        lambda: WorkOrderService(database, authorizer).close(
            identity,
            work_order_id,
            expected_version=_version(if_match),
            request_id=request.state.request_id,
        ),
        request,
        identity,
        authorizer,
    )


_DISPATCH_FACT_FIELDS: dict[str, tuple[str, ...]] = {
    "schedule.availability": (
        "site_id",
        "next_available_start",
        "next_available_end",
        "available_technician_count",
    ),
    "parts.availability": ("part_number", "available_quantity"),
}


def _dispatch_view(
    item: DispatchQueueItem,
    identity: IdentityContext,
    authorizer: Authorizer,
    tool_gateway: ToolGateway,
    *,
    request_id: str,
) -> DispatchWorkOrderResponse:
    schedule, schedule_failure = _dispatch_fact(
        tool_gateway,
        identity,
        tool_id="schedule.availability",
        asset_id=item.asset_id,
        request_id=request_id,
    )
    inventory, inventory_failure = _dispatch_fact(
        tool_gateway,
        identity,
        tool_id="parts.availability",
        asset_id=item.asset_id,
        request_id=request_id,
    )
    base = _view(item.work_order, identity, authorizer)
    return DispatchWorkOrderResponse(
        **base.model_dump(),
        asset_id=item.asset_id,
        asset_display_name=item.asset_display_name,
        model_code=item.model_code,
        site_id=item.site_id,
        site_name=item.site_name,
        incident_description=item.incident_description,
        schedule=schedule,
        inventory=inventory,
        fact_failures=[
            failure for failure in (schedule_failure, inventory_failure) if failure is not None
        ],
    )


def _center_view(
    item: DispatchQueueItem,
    identity: IdentityContext,
    authorizer: Authorizer,
) -> WorkOrderCenterItemResponse:
    base = _view(item.work_order, identity, authorizer)
    return WorkOrderCenterItemResponse(
        **base.model_dump(),
        asset_id=item.asset_id,
        asset_display_name=item.asset_display_name,
        model_code=item.model_code,
        site_id=item.site_id,
        site_name=item.site_name,
        incident_description=item.incident_description,
    )


def _dispatch_fact(
    tool_gateway: ToolGateway,
    identity: IdentityContext,
    *,
    tool_id: str,
    asset_id: str,
    request_id: str,
) -> tuple[DispatchFactResponse | None, DispatchDependencyFailureResponse | None]:
    try:
        result = tool_gateway.invoke(
            identity,
            tool_id=tool_id,
            version="1.0.0",
            parameters={"asset_id": asset_id},
            request_id=request_id,
        )
    except AuthorizationDenied:
        return None, DispatchDependencyFailureResponse(
            tool_id=tool_id,
            reason="tool_authorization_denied",
        )
    except EnterpriseToolError as exc:
        return None, DispatchDependencyFailureResponse(tool_id=tool_id, reason=exc.reason)
    except ToolRateLimitExceeded:
        return None, DispatchDependencyFailureResponse(
            tool_id=tool_id,
            reason="tool_rate_limit_exceeded",
        )
    if result.status != "SUCCEEDED" or result.data is None:
        return None, DispatchDependencyFailureResponse(
            tool_id=tool_id,
            reason="tool_result_not_usable",
        )
    data = result.data
    authority = data.get("authority")
    if not isinstance(authority, dict):
        return None, DispatchDependencyFailureResponse(
            tool_id=tool_id,
            reason="tool_authority_declaration_invalid",
        )
    field_sources = authority.get("field_sources")
    if not isinstance(field_sources, dict):
        return None, DispatchDependencyFailureResponse(
            tool_id=tool_id,
            reason="tool_authority_declaration_invalid",
        )
    return (
        DispatchFactResponse(
            tool_id=tool_id,
            tool_call_id=result.tool_call_id,
            source=str(data["source"]),
            source_record_id=str(data["source_record_id"]),
            as_of=str(data["as_of"]),
            authority=DispatchFactAuthorityResponse(
                contract_version=str(authority.get("contract_version", "")),
                domain=str(authority.get("domain", "")),
                owner=str(authority.get("owner", "")),
                field_sources={str(field): str(source) for field, source in field_sources.items()},
            ),
            values={
                field: _dispatch_fact_value(data.get(field))
                for field in _DISPATCH_FACT_FIELDS[tool_id]
            },
        ),
        None,
    )


def _dispatch_fact_value(value: Any) -> str | int | bool | None:
    if value is None or isinstance(value, str | int | bool):
        return value
    return str(value)


def _call(
    operation: Callable[[], WorkOrderView],
    request: Request,
    identity: IdentityContext,
    authorizer: Authorizer,
) -> WorkOrderEnvelope:
    try:
        work = operation()
    except WorkOrderNotVisible as exc:
        raise AppError(
            404,
            "resource_not_found_or_not_visible",
            "not_found",
            "Resource not found or not visible",
        ) from exc
    except AuthorizationDenied as exc:
        raise AppError(
            403,
            "authorization_denied",
            "authorization",
            "Action is not allowed",
        ) from exc
    except WorkOrderConflict as exc:
        raise AppError(
            409,
            exc.reason,
            "conflict",
            "WorkOrder state or version conflict",
            details={"current_version": exc.current_version},
        ) from exc
    return WorkOrderEnvelope(
        data=_view(
            work,
            identity,
            authorizer,
            edge_diagnosis_available=(
                bool(request.app.state.settings.edge_diagnosis_enabled)
                and getattr(request.app.state, "edge_pack_signer", None) is not None
            ),
        ),
        meta={"request_id": request.state.request_id},
    )


def _view(
    work: WorkOrderView,
    identity: IdentityContext,
    authorizer: Authorizer,
    *,
    edge_diagnosis_available: bool = False,
) -> WorkOrderResponse:
    return WorkOrderResponse(
        work_order_id=work.work_order_id,
        incident_id=work.incident_id,
        proposal_id=work.proposal_id,
        reservation_id=work.reservation_id,
        creation_mode=work.creation_mode,
        authorization_type=work.authorization_type,
        diagnosis_run_id=work.diagnosis_run_id,
        diagnosis_version=work.diagnosis_version,
        service_quotation_id=work.service_quotation_id,
        initial_parts_required=work.initial_parts_required,
        part_issue_required=work.part_issue_required,
        part_accounting_required=work.part_accounting_required,
        status=work.status,
        assigned_subject_id=work.assigned_subject_id,
        pending_assignment_operation_id=work.pending_assignment_operation_id,
        priority=work.priority,
        sla_due_at=work.sla_due_at.isoformat() if work.sla_due_at is not None else None,
        sla_status=work.sla_status,
        service_window_start=(
            work.service_window_start.isoformat() if work.service_window_start is not None else None
        ),
        service_window_end=(
            work.service_window_end.isoformat() if work.service_window_end is not None else None
        ),
        version=work.version,
        created_at=work.created_at.isoformat(),
        updated_at=work.updated_at.isoformat(),
        legal_actions=_legal_actions(
            work,
            identity,
            authorizer,
            edge_diagnosis_available=edge_diagnosis_available,
        ),
    )


def _legal_actions(
    work: WorkOrderView,
    identity: IdentityContext,
    authorizer: Authorizer,
    *,
    edge_diagnosis_available: bool = False,
) -> list[str]:
    transitions: dict[str, list[tuple[Action, str]]] = {
        "READY": [(Action.ASSIGN_WORK_ORDER, "ASSIGN")],
        "ASSIGNED": [(Action.EXECUTE_WORK_ORDER, "ACCEPT")],
        "ACCEPTED": [(Action.EXECUTE_WORK_ORDER, "START")],
        "IN_PROGRESS": [
            (Action.EXECUTE_WORK_ORDER, "COMPLETE"),
            (Action.CONTROL_WORK_ORDER, "HOLD"),
            (Action.ESCALATE_WORK_ORDER, "ESCALATE"),
        ],
        "ON_HOLD": [
            (Action.CONTROL_WORK_ORDER, "RESUME"),
            (Action.ESCALATE_WORK_ORDER, "ESCALATE"),
        ],
        "COMPLETED": [(Action.VERIFY_WORK_ORDER, "VERIFY")],
        "VERIFIED": [(Action.CLOSE_WORK_ORDER, "CLOSE")],
        "ESCALATED": [(Action.ESCALATE_WORK_ORDER, "REPLAN")],
    }
    candidates = transitions.get(work.status, [])
    if edge_diagnosis_available and work.status in {"ACCEPTED", "IN_PROGRESS", "ON_HOLD"}:
        candidates.append((Action.EXECUTE_WORK_ORDER, "CREATE_EDGE_DIAGNOSIS_PACK"))
    if work.status not in {"CLOSED", "CANCELLED"}:
        candidates.append((Action.CONTROL_WORK_ORDER, "RESCHEDULE"))
    if work.pending_assignment_operation_id is not None:
        candidates = [item for item in candidates if item[1] not in {"ASSIGN", "RESCHEDULE"}]
    resource = ResourceContext(tenant_id=identity.tenant_id, resource_id=work.work_order_id)
    assignee_only = {
        "ACCEPT",
        "START",
        "COMPLETE",
        "HOLD",
        "RESUME",
        "RESCHEDULE",
        "CREATE_EDGE_DIAGNOSIS_PACK",
    }
    coordinator_roles = {
        "after_sales_engineer",
        "domain_expert",
        "tenant_admin",
    }
    is_coordinator = any(role.value in coordinator_roles for role in identity.roles)
    return [
        label
        for action, label in candidates
        if authorizer.decide(identity, action, resource).allowed
        and (
            label not in assignee_only
            or is_coordinator
            or work.assigned_subject_id == identity.subject_id
        )
    ]


def _control_view(control: WorkOrderControlView) -> WorkOrderControlResponse:
    return WorkOrderControlResponse(
        control_id=control.control_id,
        work_order_version=control.work_order_version,
        command_type=control.command_type,
        previous_status=control.previous_status,
        target_status=control.target_status,
        reason=control.reason,
        recovery_condition=control.recovery_condition,
        responsible_subject_id=control.responsible_subject_id,
        service_window_start=(
            control.service_window_start.isoformat()
            if control.service_window_start is not None
            else None
        ),
        service_window_end=(
            control.service_window_end.isoformat()
            if control.service_window_end is not None
            else None
        ),
        sla_due_at=(control.sla_due_at.isoformat() if control.sla_due_at is not None else None),
        occurred_at=control.occurred_at.isoformat(),
    )


def _offline_pack_view(pack: WorkOrderOfflinePackView) -> WorkOrderOfflinePackResponse:
    return WorkOrderOfflinePackResponse(
        pack_id=pack.pack_id,
        status=pack.status,
        schema_version=pack.schema_version,
        subject_id=pack.subject_id,
        work_order_id=pack.work_order_id,
        work_order_version=pack.work_order_version,
        asset_id=pack.asset_id,
        asset_version=pack.asset_version,
        assignment_id=pack.assignment_id,
        assignment_assigned_at=pack.assignment_assigned_at.isoformat(),
        snapshot=OfflinePackSnapshotResponse.model_validate(pack.snapshot),
        content_hash=pack.content_hash,
        issued_at=pack.issued_at.isoformat(),
        expires_at=pack.expires_at.isoformat(),
        revoked_at=pack.revoked_at.isoformat() if pack.revoked_at is not None else None,
        revoked_by_subject_id=pack.revoked_by_subject_id,
        revocation_reason=pack.revocation_reason,
        version=pack.version,
        legal_actions=pack.legal_actions,
    )


def _edge_diagnosis_service(
    request: Request,
    database: Database,
    authorizer: Authorizer,
    signer: Ed25519PackSigner | None,
) -> WorkOrderEdgeDiagnosisService:
    return WorkOrderEdgeDiagnosisService(
        database,
        authorizer,
        signer,
        enabled=bool(request.app.state.settings.edge_diagnosis_enabled),
    )


def _edge_pack_view(
    pack: WorkOrderEdgeDiagnosisPackRecord,
) -> FieldEdgeDiagnosisPackResponse:
    claims = FieldEdgeDiagnosisPackClaims.model_validate(pack.claims_json)
    return FieldEdgeDiagnosisPackResponse(
        pack_id=pack.pack_id,
        status=pack.status,  # type: ignore[arg-type]
        schema_version=claims.schema_version,
        token=pack.signed_token,
        pack_digest=pack.pack_digest,
        key_id=pack.key_id,
        work_order_id=pack.work_order_id,
        offline_pack_id=pack.offline_pack_id,
        release=claims.release,
        prompt_bundle=claims.prompt_bundle,
        retrieval=claims.retrieval,
        issued_at=pack.issued_at,
        expires_at=pack.expires_at,
        version=pack.version,
        legal_actions=["IMPORT_RESULT"] if pack.status == "ACTIVE" else [],
    )


def _edge_candidate_view(
    database: Database,
    identity: IdentityContext,
    item: WorkOrderEdgeDiagnosisCandidateRecord,
) -> FieldEdgeDiagnosisCandidateResponse:
    with database.transaction(identity.tenant_context) as session:
        pack = session.scalar(
            select(WorkOrderEdgeDiagnosisPackRecord).where(
                WorkOrderEdgeDiagnosisPackRecord.tenant_id == identity.tenant_id,
                WorkOrderEdgeDiagnosisPackRecord.subject_id == identity.subject_id,
                WorkOrderEdgeDiagnosisPackRecord.pack_id == item.pack_id,
            )
        )
        if pack is None:
            raise AppError(
                404,
                "edge_diagnosis_not_found_or_not_visible",
                "not_found",
                "Edge diagnosis candidate is not available",
            )
        pack_summary = FieldEdgeDiagnosisPackSummary(
            pack_id=pack.pack_id,
            key_id=pack.key_id,
            expires_at=pack.expires_at,
            release_id=pack.release_id,
            manifest_hash=pack.manifest_hash,
            model_file=pack.model_file,
            model_content_hash=pack.model_content_hash,
            prompt_bundle_id=pack.prompt_bundle_id,
            prompt_bundle_hash=pack.prompt_bundle_hash,
            index_release_id=pack.index_release_id,
            index_content_checksum=pack.index_content_checksum,
        )
    return FieldEdgeDiagnosisCandidateResponse(
        candidate_id=item.candidate_id,
        pack_id=item.pack_id,
        work_order_id=item.work_order_id,
        status=item.status,  # type: ignore[arg-type]
        pack=pack_summary,
        candidate=FieldEdgeDiagnosisCandidate.model_validate(item.candidate_json),
        runtime_evidence=FieldEdgeRuntimeEvidence.model_validate(item.runtime_evidence_json),
        result_content_hash=item.result_content_hash,
        security_policy_version=item.security_policy_version,
        reviewed_by_subject_id=item.reviewed_by_subject_id,
        review_reason=item.review_reason,
        reviewed_at=item.reviewed_at,
        version=item.version,
        created_at=item.created_at,
        legal_actions=(
            ["ACCEPT", "REJECT"]
            if item.status == "PENDING_REVIEW"
            else ["CREATE_AI_OBSERVATION"]
            if item.status == "ACCEPTED"
            else []
        ),
    )


def _field_evidence_upload_view(
    upload: FieldEvidenceUploadView,
) -> FieldEvidenceUploadResponse:
    return FieldEvidenceUploadResponse(
        evidence_upload_id=upload.evidence_upload_id,
        work_order_id=upload.work_order_id,
        media_id=upload.media_id,
        uploaded_by_subject_id=upload.uploaded_by_subject_id,
        client_operation_id=upload.client_operation_id,
        content_hash=upload.content_hash,
        declared_mime=upload.declared_mime,
        detected_mime=upload.detected_mime,
        size_bytes=upload.size_bytes,
        work_order_version=upload.work_order_version,
        scan_state=upload.scan_state,
        scan_version=upload.scan_version,
        content_credential_status=upload.content_credential_status,
        ready_to_attach=upload.ready_to_attach,
        legal_actions=upload.legal_actions,
        recognition_run_id=upload.recognition_run_id,
        recognition_status=upload.recognition_status,
        recognition_bundle_id=upload.recognition_bundle_id,
        recognition_bundle_status=upload.recognition_bundle_status,
        recognition_bundle_version=upload.recognition_bundle_version,
        recognition_failure_summary=upload.recognition_failure_summary,
        occurred_at=upload.occurred_at,
        updated_at=upload.updated_at,
    )


def _field_recognition_run_envelope(
    run: RecognitionRun,
    *,
    created: bool,
    request_id: str,
) -> FieldRecognitionRunEnvelope:
    if (
        run.context_kind != "FIELD_EVIDENCE"
        or run.work_order_id is None
        or run.field_evidence_upload_id is None
        or run.work_order_version is None
        or run.requested_by_subject_id is None
    ):
        raise RuntimeError("field recognition response binding is incomplete")
    return FieldRecognitionRunEnvelope(
        data=FieldRecognitionRunResponse(
            recognition_run_id=run.recognition_run_id,
            draft_id=run.draft_id,
            media_id=run.media_id,
            context_kind="FIELD_EVIDENCE",
            work_order_id=run.work_order_id,
            field_evidence_upload_id=run.field_evidence_upload_id,
            work_order_version=run.work_order_version,
            requested_by_subject_id=run.requested_by_subject_id,
            workflow_id=run.workflow_id,
            status=run.status,
            processor_profile=run.processor_profile,
            evidence_bundle_id=run.evidence_bundle_id,
            failure_reason=run.failure_reason,
            version=run.version,
            created_at=run.created_at,
            updated_at=run.updated_at,
        ),
        meta={"request_id": request_id, "created": created},
    )


def _field_entry_view(entry: WorkOrderFieldEntryView) -> WorkOrderFieldEntryResponse:
    return WorkOrderFieldEntryResponse(
        entry_id=entry.entry_id,
        sequence=entry.sequence,
        client_operation_id=entry.client_operation_id,
        entry_type=entry.entry_type,
        payload=entry.payload,
        actor_subject_id=entry.actor_subject_id,
        occurred_at=entry.occurred_at.isoformat(),
        recorded_at=entry.recorded_at.isoformat(),
    )


def _repair_history_view(
    history: WorkOrderRepairHistoryView,
) -> WorkOrderRepairHistoryResponse:
    return WorkOrderRepairHistoryResponse(
        work_order_id=history.work_order_id,
        current_round=history.current_round,
        rounds=[_repair_round_view(item) for item in history.rounds],
    )


def _repair_round_view(item: WorkOrderRepairRoundView) -> WorkOrderRepairRoundResponse:
    return WorkOrderRepairRoundResponse(
        round_number=item.round_number,
        status=item.status,
        completion_id=item.completion_id,
        completed_by_subject_id=item.completed_by_subject_id,
        root_cause=item.root_cause,
        actions=item.actions,
        evidence_ids=item.evidence_ids,
        part_reservation_ids=item.part_reservation_ids,
        cost_amount=item.cost_amount,
        customer_confirmation=item.customer_confirmation,
        field_entry_sequence_start=item.field_entry_sequence_start,
        field_entry_sequence_end=item.field_entry_sequence_end,
        completed_at=item.completed_at.isoformat() if item.completed_at else None,
        verification_id=item.verification_id,
        verifier_subject_id=item.verifier_subject_id,
        verification_passed=item.verification_passed,
        verification_reason=item.verification_reason,
        verified_at=item.verified_at.isoformat() if item.verified_at else None,
        rework_id=item.rework_id,
        rework_status=item.rework_status,
        rework_reason=item.rework_reason,
        entry_sequence_checkpoint=item.entry_sequence_checkpoint,
        rework_opened_at=(item.rework_opened_at.isoformat() if item.rework_opened_at else None),
        rework_resolved_at=(
            item.rework_resolved_at.isoformat() if item.rework_resolved_at else None
        ),
    )


def _closure_report_view(
    report: WorkOrderClosureReportView,
) -> WorkOrderClosureReportResponse:
    return WorkOrderClosureReportResponse(
        report_contract_version=report.report_contract_version,
        closure_facts_digest=report.closure_facts_digest,
        work_order_id=report.work_order_id,
        incident_id=report.incident_id,
        asset_id=report.asset_id,
        asset_display_name=report.asset_display_name,
        model_code=report.model_code,
        serial_number=report.serial_number,
        site_id=report.site_id,
        site_name=report.site_name,
        closed_at=report.closed_at.isoformat(),
        closed_by_subject_id=report.closed_by_subject_id,
        close_reason=report.close_reason,
        close_operation_id=report.close_operation_id,
        close_proposal_id=report.close_proposal_id,
        repair_round_count=report.repair_round_count,
        rework_count=report.rework_count,
        completion_id=report.completion_id,
        completion_round_number=report.completion_round_number,
        completed_by_subject_id=report.completed_by_subject_id,
        completed_at=report.completed_at.isoformat(),
        root_cause=report.root_cause,
        actions=report.actions,
        evidence_ids=report.evidence_ids,
        parts=[_closure_part_view(item) for item in report.parts],
        cost_amount=report.cost_amount,
        field_customer_confirmation=report.field_customer_confirmation,
        verification_id=report.verification_id,
        verifier_subject_id=report.verifier_subject_id,
        verification_reason=report.verification_reason,
        verified_at=report.verified_at.isoformat(),
        customer_result_confirmation_id=report.customer_result_confirmation_id,
        customer_result_accepted=report.customer_result_accepted,
        customer_confirmed_by_subject_id=report.customer_confirmed_by_subject_id,
        customer_confirmed_at=(
            report.customer_confirmed_at.isoformat() if report.customer_confirmed_at else None
        ),
        customer_satisfaction_rating=report.customer_satisfaction_rating,
    )


def _closure_part_view(item: WorkOrderClosurePartView) -> WorkOrderClosurePartResponse:
    return WorkOrderClosurePartResponse(
        reservation_id=item.reservation_id,
        part_number=item.part_number,
        quantity=item.quantity,
        status=item.status,
        source=item.source,
        as_of=item.as_of.isoformat(),
    )


def _part_allocation_view(
    item: WorkOrderPartAllocationView,
) -> WorkOrderPartAllocationResponse:
    return WorkOrderPartAllocationResponse(
        reservation_id=item.reservation_id,
        part_number=item.part_number,
        approved_quantity=item.approved_quantity,
        status=item.status,
        source=item.source,
        source_record_id=item.source_record_id,
        as_of=item.as_of.isoformat(),
        issue_required=item.issue_required,
        issue_status=item.issue_status,
        usage_enabled=item.usage_enabled,
        legal_actions=item.legal_actions,
        part_issue_id=item.part_issue_id,
        proposal_id=item.proposal_id,
        approval_id=item.approval_id,
        operation_id=item.operation_id,
        reconciliation_id=item.reconciliation_id,
        external_issue_id=item.external_issue_id,
        issue_source=item.issue_source,
        issue_source_record_id=item.issue_source_record_id,
        issue_as_of=item.issue_as_of.isoformat() if item.issue_as_of is not None else None,
    )


def _part_accounting_entry_view(
    item: WorkOrderPartAccountingEntryView,
) -> WorkOrderPartAccountingEntryResponse:
    return WorkOrderPartAccountingEntryResponse(
        entry_id=item.entry_id,
        quantity=item.quantity,
        occurred_at=item.occurred_at.isoformat(),
    )


def _part_movement_view(
    item: WorkOrderPartMovementView,
) -> WorkOrderPartMovementResponse:
    return WorkOrderPartMovementResponse(
        movement_id=item.movement_id,
        operation_id=item.operation_id,
        proposal_id=item.proposal_id,
        approval_id=item.approval_id,
        movement_kind=item.movement_kind,
        field_entry_id=item.field_entry_id,
        quantity=item.quantity,
        status=item.status,
        external_movement_id=item.external_movement_id,
        source=item.source,
        source_record_id=item.source_record_id,
        as_of=item.as_of.isoformat() if item.as_of is not None else None,
        reconciliation_id=item.reconciliation_id,
        reason=item.reason,
        created_at=item.created_at.isoformat(),
        updated_at=item.updated_at.isoformat(),
    )


def _part_accounting_view(
    item: WorkOrderPartAccountingView,
) -> WorkOrderPartAccountingResponse:
    return WorkOrderPartAccountingResponse(
        mode=item.mode,
        work_order_id=item.work_order_id,
        work_order_version=item.work_order_version,
        reservation_id=item.reservation_id,
        part_issue_id=item.part_issue_id,
        part_number=item.part_number,
        issued_quantity=item.issued_quantity,
        recorded_usage_quantity=item.recorded_usage_quantity,
        consumed_quantity=item.consumed_quantity,
        returned_quantity=item.returned_quantity,
        active_consumption_quantity=item.active_consumption_quantity,
        active_return_quantity=item.active_return_quantity,
        unaccounted_quantity=item.unaccounted_quantity,
        returnable_quantity=item.returnable_quantity,
        eligible_entries=[_part_accounting_entry_view(entry) for entry in item.eligible_entries],
        movements=[_part_movement_view(movement) for movement in item.movements],
        close_ready=item.close_ready,
        blocking_reasons=item.blocking_reasons,
        legal_actions=item.legal_actions,
    )


def _field_voice_guidance_response(
    item: FieldVoiceGuidance,
) -> FieldVoiceGuidanceResponse:
    diagnosis = item.diagnosis
    field_reanalysis = item.field_reanalysis
    return FieldVoiceGuidanceResponse(
        work_order_id=item.work_order_id,
        work_order_version=item.work_order_version,
        work_order_status=item.work_order_status,
        incident_id=item.incident_id,
        diagnosis=(
            FieldVoiceDiagnosisResponse(
                diagnosis_run_id=diagnosis.diagnosis_run_id,
                status=diagnosis.status,
                version=diagnosis.version,
                conclusion=diagnosis.conclusion,
                next_checks=list(diagnosis.next_checks),
            )
            if diagnosis is not None
            else None
        ),
        field_reanalysis=(
            FieldRediagnosisProgressResponse(
                diagnosis_run_id=field_reanalysis.diagnosis_run_id,
                status=field_reanalysis.status,
                version=field_reanalysis.version,
                source_diagnosis_run_id=field_reanalysis.source_diagnosis_run_id,
                source_work_order_version=field_reanalysis.source_work_order_version,
                field_entry_count=field_reanalysis.field_entry_count,
                updated_at=field_reanalysis.updated_at,
            )
            if field_reanalysis is not None
            else None
        ),
        safety_warning=item.safety_warning,
        legal_actions=list(item.legal_actions),  # type: ignore[arg-type]
    )


def _raise_field_error(
    error: WorkOrderNotVisible | AuthorizationDenied | WorkOrderConflict,
) -> NoReturn:
    if isinstance(error, WorkOrderNotVisible):
        raise AppError(
            404,
            "resource_not_found_or_not_visible",
            "not_found",
            "Resource not found or not visible",
        ) from error
    if isinstance(error, AuthorizationDenied):
        raise AppError(
            403,
            "authorization_denied",
            "authorization",
            "Action is not allowed",
        ) from error
    raise AppError(
        409,
        error.reason,
        "conflict",
        "Field execution fact conflicts with the work order",
        details={"current_version": error.current_version},
    ) from error


def _raise_offline_pack_error(
    error: (
        OfflinePackNotAvailable | WorkOrderNotVisible | AuthorizationDenied | WorkOrderConflict
    ),
) -> NoReturn:
    if isinstance(error, OfflinePackNotAvailable):
        raise AppError(
            404,
            "offline_pack_not_available",
            "not_found",
            "No current offline work pack is available",
        ) from error
    _raise_field_error(error)


def _raise_edge_diagnosis_error(
    error: (
        OfflinePackNotAvailable
        | WorkOrderNotVisible
        | AuthorizationDenied
        | WorkOrderConflict
        | EdgeDiagnosisUnavailable
        | EdgeDiagnosisNotVisible
        | EdgeDiagnosisConflict
        | EdgeDiagnosisRejected
    ),
) -> NoReturn:
    if isinstance(error, EdgeDiagnosisUnavailable):
        raise AppError(
            503,
            error.reason,
            "dependency",
            "Governed field edge diagnosis is unavailable",
            retryable=True,
        ) from error
    if isinstance(error, EdgeDiagnosisNotVisible):
        raise AppError(
            404,
            "edge_diagnosis_not_found_or_not_visible",
            "not_found",
            "Edge diagnosis resource not found or not visible",
        ) from error
    if isinstance(error, EdgeDiagnosisRejected):
        raise AppError(
            422,
            error.reason,
            "validation",
            "Edge diagnosis result failed governed validation",
        ) from error
    if isinstance(error, EdgeDiagnosisConflict):
        details = (
            {"current_version": error.current_version}
            if error.current_version is not None
            else None
        )
        raise AppError(
            409,
            error.reason,
            "conflict",
            "Edge diagnosis state or authority binding changed",
            details=details,
        ) from error
    if isinstance(error, OfflinePackNotAvailable):
        _raise_offline_pack_error(error)
    _raise_field_error(error)


def _version(value: str) -> int:
    try:
        return int(value.strip().strip('"'))
    except ValueError as exc:
        raise AppError(
            400,
            "invalid_version_precondition",
            "validation",
            "If-Match is invalid",
        ) from exc
