"""T2 approval decisions and idempotent execution API."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Self

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from industrial_ops_agent.api.dependencies import get_authorizer, get_identity, get_tool_gateway
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.approval.models import (
    ApprovalConflict,
    ApprovalNotVisible,
    ExecutionResult,
    PurchaseDependencyUnavailable,
    PurchaseNotRequired,
    RefundAlreadyRequested,
    RefundNotEligible,
    SeparationOfDutiesViolation,
)
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _utc
from industrial_ops_agent.persistence.models import (
    ActionProposalRecord,
    ApprovalRequestRecord,
    RefundRequestRecord,
)
from industrial_ops_agent.tools.gateway import (
    CustomerNotificationAuthorityUnavailable,
    CustomerNotificationContactConflict,
    FsmAssignmentProfileConflict,
    FsmAssignmentProfileUnavailable,
    PartsReservationProposalDependencyUnavailable,
    PartsReservationProposalIdempotencyConflict,
    PartsReservationProposalNotEligible,
    PurchaseDeliveryProfileConflict,
    PurchaseDeliveryProfileUnavailable,
    PurchaseRequestProjection,
    RefundDeliveryProfileConflict,
    RefundDeliveryProfileUnavailable,
    RepairWorkOrderConflict,
    RepairWorkOrderProposalIdempotencyConflict,
    RepairWorkOrderUnavailable,
    ServiceQuotationConflict,
    ServiceQuotationUnavailable,
    ToolGateway,
)

router = APIRouter(tags=["approvals"])


class DecisionBody(BaseModel):
    decision: str
    reason: str = Field(min_length=3, max_length=500)


class ExecuteBody(BaseModel):
    parameters: dict[str, Any]


class CustomerNotificationProposalBody(BaseModel):
    incident_version: int = Field(ge=1)
    template_id: Literal[
        "DIAGNOSIS_SUMMARY",
        "INFORMATION_REQUEST",
        "SERVICE_PROGRESS",
    ]
    headline: str = Field(min_length=3, max_length=120)
    detail: str = Field(min_length=3, max_length=1000)
    next_step: str = Field(min_length=3, max_length=500)
    channel: Literal["PORTAL", "PORTAL_AND_EMAIL", "PORTAL_AND_SMS"] = "PORTAL"
    contact_point_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_contact_binding(self) -> Self:
        if self.channel == "PORTAL" and self.contact_point_id is not None:
            raise ValueError("portal_contact_not_allowed")
        if self.channel != "PORTAL" and self.contact_point_id is None:
            raise ValueError("external_contact_required")
        return self


class PurchaseRequestProposalBody(BaseModel):
    incident_version: int = Field(ge=1)
    part_number: str = Field(pattern=r"^[A-Z0-9-]{3,64}$")
    quantity: int = Field(ge=1, le=100)
    estimated_unit_cost_minor: int = Field(ge=0, le=1_000_000)
    currency: Literal["CNY", "USD", "EUR"]
    cost_center: str = Field(pattern=r"^[A-Z0-9_-]{2,64}$")
    justification: str = Field(min_length=3, max_length=1000)
    delivery_target: Literal["INTERNAL", "ENTERPRISE_ERP"] = "INTERNAL"
    profile_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_delivery_binding(self) -> Self:
        if self.delivery_target == "INTERNAL" and self.profile_id is not None:
            raise ValueError("internal_purchase_profile_not_allowed")
        if self.delivery_target == "ENTERPRISE_ERP" and self.profile_id is None:
            raise ValueError("enterprise_procurement_profile_required")
        return self


class PartReservationProposalBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_version: int = Field(ge=1)
    part_number: str = Field(pattern=r"^[A-Z0-9-]{3,64}$")
    quantity: int = Field(ge=1, le=20)


class RefundRequestProposalBody(BaseModel):
    work_order_version: int = Field(ge=1)
    amount_minor: int = Field(ge=1, le=1_000_000)
    currency: Literal["CNY", "USD", "EUR"]
    cost_center: str = Field(pattern=r"^[A-Z0-9_-]{2,64}$")
    reason: str = Field(min_length=10, max_length=2000)
    delivery_target: Literal["INTERNAL", "ENTERPRISE_FINANCE"] = "INTERNAL"
    profile_id: str | None = Field(default=None, min_length=1, max_length=128)
    reason_code: Literal[
        "SERVICE_NOT_ACCEPTED",
        "LOW_SATISFACTION",
        "REWORK_COMPENSATION",
        "OTHER_APPROVED_COMPENSATION",
    ] | None = None

    @model_validator(mode="after")
    def validate_delivery_binding(self) -> Self:
        if self.delivery_target == "INTERNAL" and (
            self.profile_id is not None or self.reason_code is not None
        ):
            raise ValueError("internal_refund_profile_not_allowed")
        if self.delivery_target == "ENTERPRISE_FINANCE" and (
            self.profile_id is None or self.reason_code is None
        ):
            raise ValueError("enterprise_refund_profile_required")
        return self


class ServiceQuotationProposalBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )


class RepairWorkOrderProposalBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_version: int = Field(ge=1)
    authorization_type: Literal["COVERED_SERVICE", "ACCEPTED_QUOTATION"]


class WorkOrderAssignmentProposalBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_order_version: int = Field(ge=1)
    assignee_subject_id: str | None = Field(default=None, min_length=1, max_length=128)
    assignment_target: Literal["LOCAL_DIRECTORY", "ENTERPRISE_FSM"] = "LOCAL_DIRECTORY"
    candidate_profile_id: str | None = Field(default=None, min_length=1, max_length=128)
    reason: str = Field(min_length=3, max_length=1000)

    @model_validator(mode="after")
    def validate_assignment_binding(self) -> Self:
        if self.assignment_target == "LOCAL_DIRECTORY" and (
            self.assignee_subject_id is None or self.candidate_profile_id is not None
        ):
            raise ValueError("local_assignment_subject_required")
        if self.assignment_target == "ENTERPRISE_FSM" and (
            self.assignee_subject_id is not None or self.candidate_profile_id is None
        ):
            raise ValueError("enterprise_fsm_candidate_required")
        return self


class WorkOrderClosureProposalBody(BaseModel):
    work_order_version: int = Field(ge=1)
    reason: str = Field(min_length=3, max_length=1000)


class ApprovalResponse(BaseModel):
    approval_id: str
    proposal_id: str
    tool_id: str
    status: str
    parameters: dict[str, Any]
    parameters_hash: str
    resource_version: str | int
    expires_at: datetime
    version: int
    legal_actions: list[str]


class ExecutionResponse(BaseModel):
    status: str
    reservation_id: str | None
    work_order_id: str | None
    reason: str | None
    notification_id: str | None
    notification_delivery_id: str | None
    purchase_request_id: str | None
    refund_request_id: str | None
    assignment_delivery_id: str | None
    part_issue_id: str | None
    part_movement_id: str | None
    execution_attempt_id: str | None
    reconciliation_id: str | None
    service_quotation_id: str | None


class ActionProposalResponse(BaseModel):
    proposal_id: str
    approval_id: str
    operation_id: str
    tool_id: str
    status: str
    parameters: dict[str, Any]


class ActionProposalEnvelope(BaseModel):
    data: ActionProposalResponse
    meta: dict[str, str]


class ServiceQuotationLineItemResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    description: str
    quantity: int
    unit_price: str
    line_total: str


class ServiceQuotationOptionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    candidate_id: str
    source_version: str
    display_name: str
    currency: str
    line_items: list[ServiceQuotationLineItemResponse]
    subtotal: str
    discount: str
    tax: str
    total: str
    as_of: str
    expires_at: str


class ServiceQuotationOptionsResponse(BaseModel):
    incident_id: str
    diagnosis_run_id: str
    diagnosis_version: int
    entitlement_decision: str
    options: list[ServiceQuotationOptionResponse]
    catalog_unavailable_reason: str | None


class ServiceQuotationOptionsEnvelope(BaseModel):
    data: ServiceQuotationOptionsResponse
    meta: dict[str, str]


class ServiceQuotationResponse(BaseModel):
    quotation_id: str
    incident_id: str
    asset_id: str
    diagnosis_run_id: str
    diagnosis_version: int
    version: int
    supersedes_quotation_id: str | None
    proposal_id: str
    approval_id: str
    provider: str
    candidate_id: str
    source_version: str
    source_as_of: datetime
    line_items: list[ServiceQuotationLineItemResponse]
    currency: str
    subtotal: str
    discount: str
    tax: str
    total: str
    entitlement: dict[str, Any]
    status: str
    published_at: datetime
    expires_at: datetime
    state_version: int


class ServiceQuotationListEnvelope(BaseModel):
    data: list[ServiceQuotationResponse]
    meta: dict[str, str]


class RepairWorkOrderOptionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    authorization_type: Literal["COVERED_SERVICE", "ACCEPTED_QUOTATION"]
    fulfillment_mode: Literal["NO_INITIAL_PARTS"]
    quotation_id: str | None
    quotation_version: int | None
    quotation_state_version: int | None
    currency: str | None
    total: str | None


class RepairWorkOrderOptionsResponse(BaseModel):
    incident_id: str
    incident_version: int
    asset_id: str
    diagnosis_run_id: str | None
    diagnosis_version: int | None
    entitlement_decision: str | None
    options: list[RepairWorkOrderOptionResponse]
    unavailable_reason: str | None


class RepairWorkOrderOptionsEnvelope(BaseModel):
    data: RepairWorkOrderOptionsResponse
    meta: dict[str, str]


@router.get(
    "/incidents/{incident_id}/repair-work-order-options",
    response_model=RepairWorkOrderOptionsEnvelope,
)
def list_repair_work_order_options(
    incident_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
) -> RepairWorkOrderOptionsEnvelope:
    try:
        result = gateway.list_repair_work_order_options(
            identity,
            incident_id=incident_id,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ApprovalNotVisible as exc:
        raise _not_found() from exc
    return RepairWorkOrderOptionsEnvelope(
        data=RepairWorkOrderOptionsResponse(
            incident_id=result.incident_id,
            incident_version=result.incident_version,
            asset_id=result.asset_id,
            diagnosis_run_id=result.diagnosis_run_id,
            diagnosis_version=result.diagnosis_version,
            entitlement_decision=result.entitlement_decision,
            options=[
                RepairWorkOrderOptionResponse(**asdict(option))
                for option in result.options
            ],
            unavailable_reason=result.unavailable_reason,
        ),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/incidents/{incident_id}/repair-work-order-proposals",
    response_model=ActionProposalEnvelope,
    status_code=201,
)
def propose_repair_work_order(
    incident_id: str,
    body: RepairWorkOrderProposalBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=8, max_length=255),
    ],
) -> ActionProposalEnvelope:
    try:
        proposal = gateway.propose_repair_work_order(
            identity,
            incident_id=incident_id,
            incident_version=body.incident_version,
            authorization_type=body.authorization_type,
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ApprovalNotVisible as exc:
        raise _not_found() from exc
    except RepairWorkOrderProposalIdempotencyConflict as exc:
        raise AppError(
            409,
            "idempotency_conflict",
            "conflict",
            "Idempotency key was reused with different repair parameters",
        ) from exc
    except RepairWorkOrderConflict as exc:
        raise AppError(
            409,
            "repair_work_order_not_eligible",
            "conflict",
            "Current enterprise facts do not permit this repair WorkOrder",
            details={"reason": exc.reason},
        ) from exc
    except RepairWorkOrderUnavailable as exc:
        raise AppError(
            503,
            "repair_work_order_facts_unavailable",
            "dependency",
            "Authoritative repair eligibility facts are unavailable",
            retryable=True,
            details={"reason": exc.reason},
        ) from exc
    return ActionProposalEnvelope(
        data=ActionProposalResponse(
            proposal_id=proposal.proposal_id,
            approval_id=proposal.approval_id,
            operation_id=proposal.operation_id,
            tool_id="work_order.create",
            status="PENDING_APPROVAL",
            parameters=proposal.parameters,
        ),
        meta={"request_id": request.state.request_id},
    )


@router.get(
    "/incidents/{incident_id}/service-quotation-options",
    response_model=ServiceQuotationOptionsEnvelope,
)
def list_service_quotation_options(
    incident_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
) -> ServiceQuotationOptionsEnvelope:
    try:
        result = gateway.list_service_quotation_options(
            identity,
            incident_id=incident_id,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ApprovalNotVisible as exc:
        raise _not_found() from exc
    except ApprovalConflict as exc:
        raise AppError(
            409,
            "service_quotation_not_eligible",
            "conflict",
            "A terminal diagnosis is required before quotation",
        ) from exc
    return ServiceQuotationOptionsEnvelope(
        data=ServiceQuotationOptionsResponse(
            incident_id=result.incident_id,
            diagnosis_run_id=result.diagnosis_run_id,
            diagnosis_version=result.diagnosis_version,
            entitlement_decision=result.entitlement_decision,
            options=[ServiceQuotationOptionResponse(**asdict(option)) for option in result.options],
            catalog_unavailable_reason=result.catalog_unavailable_reason,
        ),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/incidents/{incident_id}/service-quotation-proposals",
    response_model=ActionProposalEnvelope,
    status_code=201,
)
def propose_service_quotation(
    incident_id: str,
    body: ServiceQuotationProposalBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
) -> ActionProposalEnvelope:
    try:
        proposal = gateway.propose_service_quotation(
            identity,
            incident_id=incident_id,
            candidate_id=body.candidate_id,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ApprovalNotVisible as exc:
        raise _not_found() from exc
    except ServiceQuotationConflict as exc:
        raise AppError(
            409,
            "service_quotation_proposal_conflict",
            "conflict",
            "Current diagnosis, entitlement, or CPQ candidate is not quotable",
            details={"reason": exc.reason},
        ) from exc
    except ServiceQuotationUnavailable as exc:
        raise AppError(
            503,
            "service_quotation_dependency_unavailable",
            "dependency",
            "Authoritative service quotation data is unavailable",
            retryable=True,
            details={"reason": exc.reason},
        ) from exc
    return ActionProposalEnvelope(
        data=ActionProposalResponse(
            proposal_id=proposal.proposal_id,
            approval_id=proposal.approval_id,
            operation_id=proposal.operation_id,
            tool_id="service.quote.publish",
            status="PENDING_APPROVAL",
            parameters=proposal.parameters,
        ),
        meta={"request_id": request.state.request_id},
    )


@router.get(
    "/incidents/{incident_id}/service-quotations",
    response_model=ServiceQuotationListEnvelope,
)
def list_service_quotations(
    incident_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
) -> ServiceQuotationListEnvelope:
    try:
        records = gateway.list_service_quotations(
            identity,
            incident_id=incident_id,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ApprovalNotVisible as exc:
        raise _not_found() from exc
    return ServiceQuotationListEnvelope(
        data=[
            ServiceQuotationResponse(
                quotation_id=record.quotation_id,
                incident_id=record.incident_id,
                asset_id=record.asset_id,
                diagnosis_run_id=record.diagnosis_run_id,
                diagnosis_version=record.diagnosis_version,
                version=record.quotation_version,
                supersedes_quotation_id=record.supersedes_quotation_id,
                proposal_id=record.proposal_id,
                approval_id=record.approval_id,
                provider=record.provider,
                candidate_id=record.candidate_id,
                source_version=record.source_version,
                source_as_of=_utc(record.source_as_of),
                line_items=[
                    ServiceQuotationLineItemResponse(**item)
                    for item in record.line_items_json
                ],
                currency=record.currency,
                subtotal=record.subtotal,
                discount=record.discount,
                tax=record.tax,
                total=record.total,
                entitlement=record.entitlement_json,
                status=record.status,
                published_at=_utc(record.published_at),
                expires_at=_utc(record.expires_at),
                state_version=record.state_version,
            )
            for record in records
        ],
        meta={"request_id": request.state.request_id},
    )


class PurchaseFulfillmentResponse(BaseModel):
    fulfillment_id: str
    external_order_id: str
    status: Literal[
        "APPROVED",
        "ORDERED",
        "IN_TRANSIT",
        "PARTIALLY_RECEIVED",
        "RECEIVED",
        "REJECTED",
        "CANCELLED",
    ]
    requested_quantity: int
    received_quantity: int
    expected_delivery_at: datetime | None
    last_event_occurred_at: datetime
    reason: str | None
    version: int


class PurchaseReservationReadinessResponse(BaseModel):
    status: Literal[
        "AWAITING_ERP_FULFILLMENT",
        "NOT_RECEIVED",
        "AWAITING_WMS_SYNC",
        "FACTS_UNAVAILABLE",
        "REVIEW_REQUIRED",
        "READY_FOR_RESERVATION",
    ]
    reason: str
    next_action: str
    source: str | None
    source_record_id: str | None
    as_of: str | None
    available_quantity: int | None


class PurchaseRequestResponse(BaseModel):
    purchase_request_id: str
    proposal_id: str
    incident_id: str
    asset_id: str
    part_number: str
    quantity: int
    available_quantity_at_submission: int
    estimated_unit_cost_minor: int
    estimated_total_cost_minor: int
    currency: str
    cost_center: str
    justification: str
    delivery_target: str
    provider: str | None
    profile_id: str | None
    profile_version: str | None
    profile_display_name: str | None
    profile_as_of: datetime | None
    profile_expires_at: datetime | None
    external_request_id: str | None
    reconciliation_id: str | None = None
    fulfillment: PurchaseFulfillmentResponse | None = None
    reservation_readiness: PurchaseReservationReadinessResponse | None = None
    status: str
    reason: str | None
    submitted_at: datetime | None
    terminal_at: datetime | None
    last_receipt_at: datetime | None
    version: int
    requested_by_subject_id: str
    inventory_source: str
    inventory_source_record_id: str
    inventory_as_of: datetime
    created_at: datetime


class PurchaseRequestListEnvelope(BaseModel):
    data: list[PurchaseRequestResponse]
    meta: dict[str, str]


class RefundRequestResponse(BaseModel):
    refund_request_id: str
    proposal_id: str
    incident_id: str
    work_order_id: str
    asset_id: str
    customer_subject_id: str
    customer_confirmation_update_id: str
    amount_minor: int
    currency: str
    cost_center: str
    reason: str
    compensation_policy_id: str
    reason_code: str | None
    delivery_target: str
    provider: str | None
    profile_id: str | None
    profile_version: str | None
    profile_display_name: str | None
    profile_as_of: datetime | None
    profile_expires_at: datetime | None
    external_request_id: str | None
    reconciliation_id: str | None = None
    status: str
    payment_status: str
    provider_reason: str | None
    submitted_at: datetime | None
    terminal_at: datetime | None
    last_receipt_at: datetime | None
    version: int
    requested_by_subject_id: str
    created_at: datetime


class RefundRequestListEnvelope(BaseModel):
    data: list[RefundRequestResponse]
    meta: dict[str, str]


class ApprovalEnvelope(BaseModel):
    data: ApprovalResponse
    meta: dict[str, str]


class ApprovalListEnvelope(BaseModel):
    data: list[ApprovalResponse]
    meta: dict[str, str]


class ExecutionEnvelope(BaseModel):
    data: ExecutionResponse
    meta: dict[str, str]


class CustomerNotificationOptionResponse(BaseModel):
    channel: Literal["PORTAL", "PORTAL_AND_EMAIL", "PORTAL_AND_SMS"]
    contact_point_id: str | None
    masked_destination: str
    source_system: str
    source_record_id: str | None
    source_version: str | None
    as_of: str | None
    consent_basis_digest: str | None
    expires_at: str | None


class CustomerNotificationOptionsResponse(BaseModel):
    incident_id: str
    options: list[CustomerNotificationOptionResponse]
    external_unavailable_reason: str | None


class CustomerNotificationOptionsEnvelope(BaseModel):
    data: CustomerNotificationOptionsResponse
    meta: dict[str, str]


class PurchaseRequestOptionResponse(BaseModel):
    delivery_target: Literal["INTERNAL", "ENTERPRISE_ERP"]
    provider: str | None
    profile_id: str | None
    profile_version: str | None
    display_name: str
    as_of: str | None
    expires_at: str | None


class PurchaseRequestOptionsResponse(BaseModel):
    incident_id: str
    options: list[PurchaseRequestOptionResponse]
    erp_unavailable_reason: str | None


class PurchaseRequestOptionsEnvelope(BaseModel):
    data: PurchaseRequestOptionsResponse
    meta: dict[str, str]


class RefundRequestOptionResponse(BaseModel):
    delivery_target: Literal["INTERNAL", "ENTERPRISE_FINANCE"]
    provider: str | None
    profile_id: str | None
    profile_version: str | None
    display_name: str
    as_of: str | None
    expires_at: str | None


class RefundRequestOptionsResponse(BaseModel):
    work_order_id: str
    options: list[RefundRequestOptionResponse]
    finance_unavailable_reason: str | None


class RefundRequestOptionsEnvelope(BaseModel):
    data: RefundRequestOptionsResponse
    meta: dict[str, str]


class WorkOrderAssignmentOptionResponse(BaseModel):
    assignment_target: Literal["LOCAL_DIRECTORY", "ENTERPRISE_FSM"]
    provider: str | None
    candidate_profile_id: str | None
    profile_version: str | None
    assignee_subject_id: str | None
    display_name: str
    matched_skill_codes: list[str]
    service_window_start: str | None
    service_window_end: str | None
    travel_minutes: int | None
    remaining_work_minutes: int | None
    eligibility_code: str | None
    source_record_id: str | None
    as_of: str | None
    expires_at: str | None


class WorkOrderAssignmentOptionsResponse(BaseModel):
    work_order_id: str
    options: list[WorkOrderAssignmentOptionResponse]
    fsm_unavailable_reason: str | None


class WorkOrderAssignmentOptionsEnvelope(BaseModel):
    data: WorkOrderAssignmentOptionsResponse
    meta: dict[str, str]


@router.get(
    "/work-orders/{work_order_id}/assignment-options",
    response_model=WorkOrderAssignmentOptionsEnvelope,
)
def list_work_order_assignment_options(
    work_order_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
) -> WorkOrderAssignmentOptionsEnvelope:
    try:
        result = gateway.list_work_order_assignment_options(
            identity,
            work_order_id=work_order_id,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ApprovalNotVisible as exc:
        raise _not_found() from exc
    except ApprovalConflict as exc:
        raise AppError(
            409,
            "assignment_options_conflict",
            "conflict",
            "Work order is no longer available for assignment",
        ) from exc
    return WorkOrderAssignmentOptionsEnvelope(
        data=WorkOrderAssignmentOptionsResponse(
            work_order_id=result.work_order_id,
            options=[
                WorkOrderAssignmentOptionResponse(
                    assignment_target=option.assignment_target,
                    provider=option.provider,
                    candidate_profile_id=option.candidate_profile_id,
                    profile_version=option.profile_version,
                    assignee_subject_id=option.assignee_subject_id,
                    display_name=option.display_name,
                    matched_skill_codes=list(option.matched_skill_codes),
                    service_window_start=option.service_window_start,
                    service_window_end=option.service_window_end,
                    travel_minutes=option.travel_minutes,
                    remaining_work_minutes=option.remaining_work_minutes,
                    eligibility_code=option.eligibility_code,
                    source_record_id=option.source_record_id,
                    as_of=option.as_of,
                    expires_at=option.expires_at,
                )
                for option in result.options
            ],
            fsm_unavailable_reason=result.fsm_unavailable_reason,
        ),
        meta={"request_id": request.state.request_id},
    )


@router.get(
    "/incidents/{incident_id}/purchase-request-options",
    response_model=PurchaseRequestOptionsEnvelope,
)
def list_purchase_request_options(
    incident_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
) -> PurchaseRequestOptionsEnvelope:
    try:
        result = gateway.list_purchase_request_options(
            identity,
            incident_id=incident_id,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ApprovalNotVisible as exc:
        raise _not_found() from exc
    return PurchaseRequestOptionsEnvelope(
        data=PurchaseRequestOptionsResponse(
            incident_id=result.incident_id,
            options=[
                PurchaseRequestOptionResponse(
                    delivery_target=option.delivery_target,
                    provider=option.provider,
                    profile_id=option.profile_id,
                    profile_version=option.profile_version,
                    display_name=option.display_name,
                    as_of=option.as_of,
                    expires_at=option.expires_at,
                )
                for option in result.options
            ],
            erp_unavailable_reason=result.erp_unavailable_reason,
        ),
        meta={"request_id": request.state.request_id},
    )


@router.get(
    "/work-orders/{work_order_id}/refund-request-options",
    response_model=RefundRequestOptionsEnvelope,
)
def list_refund_request_options(
    work_order_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
) -> RefundRequestOptionsEnvelope:
    try:
        result = gateway.list_refund_request_options(
            identity,
            work_order_id=work_order_id,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ApprovalNotVisible as exc:
        raise _not_found() from exc
    except RefundNotEligible as exc:
        raise AppError(
            409,
            "refund_not_eligible",
            "conflict",
            "Latest customer result does not qualify for compensation",
        ) from exc
    return RefundRequestOptionsEnvelope(
        data=RefundRequestOptionsResponse(
            work_order_id=result.work_order_id,
            options=[
                RefundRequestOptionResponse(
                    delivery_target=option.delivery_target,
                    provider=option.provider,
                    profile_id=option.profile_id,
                    profile_version=option.profile_version,
                    display_name=option.display_name,
                    as_of=option.as_of,
                    expires_at=option.expires_at,
                )
                for option in result.options
            ],
            finance_unavailable_reason=result.finance_unavailable_reason,
        ),
        meta={"request_id": request.state.request_id},
    )


@router.get(
    "/incidents/{incident_id}/customer-notification-options",
    response_model=CustomerNotificationOptionsEnvelope,
)
def list_customer_notification_options(
    incident_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
) -> CustomerNotificationOptionsEnvelope:
    try:
        result = gateway.list_customer_notification_options(
            identity,
            incident_id=incident_id,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ApprovalNotVisible as exc:
        raise _not_found() from exc
    return CustomerNotificationOptionsEnvelope(
        data=CustomerNotificationOptionsResponse(
            incident_id=result.incident_id,
            options=[
                CustomerNotificationOptionResponse(
                    channel=option.channel,
                    contact_point_id=option.contact_point_id,
                    masked_destination=option.masked_destination,
                    source_system=option.source_system,
                    source_record_id=option.source_record_id,
                    source_version=option.source_version,
                    as_of=option.as_of,
                    consent_basis_digest=option.consent_basis_digest,
                    expires_at=option.expires_at,
                )
                for option in result.options
            ],
            external_unavailable_reason=result.external_unavailable_reason,
        ),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/incidents/{incident_id}/customer-notification-proposals",
    response_model=ActionProposalEnvelope,
    status_code=201,
)
def propose_customer_notification(
    incident_id: str,
    body: CustomerNotificationProposalBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
) -> ActionProposalEnvelope:
    try:
        proposal = gateway.propose_customer_notification(
            identity,
            incident_id=incident_id,
            incident_version=body.incident_version,
            template_id=body.template_id,
            headline=body.headline,
            detail=body.detail,
            next_step=body.next_step,
            request_id=request.state.request_id,
            channel=body.channel,
            contact_point_id=body.contact_point_id,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ApprovalNotVisible as exc:
        raise _not_found() from exc
    except ApprovalConflict as exc:
        raise AppError(
            409,
            "notification_proposal_conflict",
            "conflict",
            "Incident state or version changed",
        ) from exc
    except CustomerNotificationContactConflict as exc:
        raise AppError(
            409,
            "notification_contact_conflict",
            "conflict",
            "Customer notification contact is no longer eligible",
            details={"reason": exc.reason},
        ) from exc
    except CustomerNotificationAuthorityUnavailable as exc:
        raise AppError(
            503,
            "customer_contact_authority_unavailable",
            "dependency",
            "Authoritative customer contact data is unavailable",
            retryable=True,
            details={"reason": exc.reason},
        ) from exc
    return ActionProposalEnvelope(
        data=ActionProposalResponse(
            proposal_id=proposal.proposal_id,
            approval_id=proposal.approval_id,
            operation_id=proposal.operation_id,
            tool_id="customer.notify",
            status="PENDING_APPROVAL",
            parameters=proposal.parameters,
        ),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/incidents/{incident_id}/part-reservation-proposals",
    response_model=ActionProposalEnvelope,
    status_code=201,
)
def propose_part_reservation(
    incident_id: str,
    body: PartReservationProposalBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=8, max_length=255),
    ],
) -> ActionProposalEnvelope:
    try:
        proposal = gateway.propose_parts_reservation(
            identity,
            incident_id=incident_id,
            incident_version=body.incident_version,
            part_number=body.part_number,
            quantity=body.quantity,
            request_id=request.state.request_id,
            idempotency_key=idempotency_key,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ApprovalNotVisible as exc:
        raise _not_found() from exc
    except ApprovalConflict as exc:
        raise AppError(
            409,
            "reservation_proposal_conflict",
            "conflict",
            "Incident state or version changed",
        ) from exc
    except PartsReservationProposalNotEligible as exc:
        raise AppError(
            409,
            "reservation_not_eligible",
            "conflict",
            "Current equipment or inventory facts do not permit this reservation proposal",
            details={"reason": exc.reason},
        ) from exc
    except PartsReservationProposalDependencyUnavailable as exc:
        raise AppError(
            503,
            "inventory_dependency_unavailable",
            "dependency",
            "Authoritative inventory is unavailable or invalid",
            retryable=True,
            details={"reason": exc.reason},
        ) from exc
    except PartsReservationProposalIdempotencyConflict as exc:
        raise AppError(
            409,
            "idempotency_conflict",
            "conflict",
            "Idempotency key was reused with different reservation parameters",
        ) from exc
    return ActionProposalEnvelope(
        data=ActionProposalResponse(
            proposal_id=proposal.proposal_id,
            approval_id=proposal.approval_id,
            operation_id=proposal.operation_id,
            tool_id="parts.reserve",
            status="PENDING_APPROVAL",
            parameters=proposal.parameters,
        ),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/incidents/{incident_id}/purchase-request-proposals",
    response_model=ActionProposalEnvelope,
    status_code=201,
)
def propose_purchase_request(
    incident_id: str,
    body: PurchaseRequestProposalBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
) -> ActionProposalEnvelope:
    try:
        proposal = gateway.propose_purchase_request(
            identity,
            incident_id=incident_id,
            incident_version=body.incident_version,
            part_number=body.part_number,
            quantity=body.quantity,
            estimated_unit_cost_minor=body.estimated_unit_cost_minor,
            currency=body.currency,
            cost_center=body.cost_center,
            justification=body.justification,
            request_id=request.state.request_id,
            delivery_target=body.delivery_target,
            profile_id=body.profile_id,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ApprovalNotVisible as exc:
        raise _not_found() from exc
    except PurchaseNotRequired as exc:
        raise AppError(
            409,
            "purchase_not_required",
            "conflict",
            "Current inventory can satisfy the requested quantity",
        ) from exc
    except PurchaseDependencyUnavailable as exc:
        raise AppError(
            503,
            "inventory_dependency_unavailable",
            "dependency",
            "Authoritative inventory is unavailable or stale",
            retryable=True,
            details={"reason": exc.reason},
        ) from exc
    except PurchaseDeliveryProfileUnavailable as exc:
        raise AppError(
            503,
            "enterprise_procurement_profile_unavailable",
            "dependency",
            "Enterprise procurement profile is unavailable",
            retryable=True,
            details={"reason": exc.reason},
        ) from exc
    except PurchaseDeliveryProfileConflict as exc:
        raise AppError(
            409,
            "purchase_delivery_profile_conflict",
            "conflict",
            "Purchase delivery profile is not eligible",
            details={"reason": exc.reason},
        ) from exc
    except ApprovalConflict as exc:
        raise AppError(
            409,
            "purchase_proposal_conflict",
            "conflict",
            "Incident state or version changed",
        ) from exc
    return ActionProposalEnvelope(
        data=ActionProposalResponse(
            proposal_id=proposal.proposal_id,
            approval_id=proposal.approval_id,
            operation_id=proposal.operation_id,
            tool_id="purchase.request",
            status="PENDING_APPROVAL",
            parameters=proposal.parameters,
        ),
        meta={"request_id": request.state.request_id},
    )


@router.get(
    "/incidents/{incident_id}/purchase-requests",
    response_model=PurchaseRequestListEnvelope,
)
def list_purchase_requests(
    incident_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
) -> PurchaseRequestListEnvelope:
    try:
        records = gateway.list_purchase_requests(
            identity,
            incident_id=incident_id,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ApprovalNotVisible as exc:
        raise _not_found() from exc
    return PurchaseRequestListEnvelope(
        data=[_purchase_request(item) for item in records],
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/work-orders/{work_order_id}/refund-request-proposals",
    response_model=ActionProposalEnvelope,
    status_code=201,
)
def propose_refund_request(
    work_order_id: str,
    body: RefundRequestProposalBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
) -> ActionProposalEnvelope:
    try:
        proposal = gateway.propose_refund_request(
            identity,
            work_order_id=work_order_id,
            work_order_version=body.work_order_version,
            amount_minor=body.amount_minor,
            currency=body.currency,
            cost_center=body.cost_center,
            reason=body.reason,
            request_id=request.state.request_id,
            delivery_target=body.delivery_target,
            profile_id=body.profile_id,
            reason_code=body.reason_code,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ApprovalNotVisible as exc:
        raise _not_found() from exc
    except RefundNotEligible as exc:
        raise AppError(
            409,
            "refund_not_eligible",
            "conflict",
            "Latest customer result does not qualify for compensation",
        ) from exc
    except RefundAlreadyRequested as exc:
        raise AppError(
            409,
            "refund_already_requested",
            "conflict",
            "A refund request already exists for this work order",
        ) from exc
    except RefundDeliveryProfileUnavailable as exc:
        raise AppError(
            503,
            "enterprise_finance_profile_unavailable",
            "dependency",
            "Enterprise finance profile is unavailable",
            retryable=True,
            details={"reason": exc.reason},
        ) from exc
    except RefundDeliveryProfileConflict as exc:
        raise AppError(
            409,
            "refund_delivery_profile_conflict",
            "conflict",
            "Refund delivery profile is not eligible",
            details={"reason": exc.reason},
        ) from exc
    except ApprovalConflict as exc:
        raise AppError(
            409,
            "refund_proposal_conflict",
            "conflict",
            "Work order state or version changed",
        ) from exc
    return ActionProposalEnvelope(
        data=ActionProposalResponse(
            proposal_id=proposal.proposal_id,
            approval_id=proposal.approval_id,
            operation_id=proposal.operation_id,
            tool_id="refund.request",
            status="PENDING_APPROVAL",
            parameters=proposal.parameters,
        ),
        meta={"request_id": request.state.request_id},
    )


@router.get(
    "/incidents/{incident_id}/refund-requests",
    response_model=RefundRequestListEnvelope,
)
def list_refund_requests(
    incident_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
) -> RefundRequestListEnvelope:
    try:
        records = gateway.list_refund_requests(
            identity,
            incident_id=incident_id,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ApprovalNotVisible as exc:
        raise _not_found() from exc
    return RefundRequestListEnvelope(
        data=[
            _refund_request(record, reconciliation_id)
            for record, reconciliation_id in records
        ],
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/work-orders/{work_order_id}/assignment-proposals",
    response_model=ActionProposalEnvelope,
    status_code=201,
)
def propose_work_order_assignment(
    work_order_id: str,
    body: WorkOrderAssignmentProposalBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
) -> ActionProposalEnvelope:
    try:
        proposal = gateway.propose_work_order_assignment(
            identity,
            work_order_id=work_order_id,
            work_order_version=body.work_order_version,
            assignee_subject_id=body.assignee_subject_id,
            assignment_target=body.assignment_target,
            candidate_profile_id=body.candidate_profile_id,
            reason=body.reason,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ApprovalNotVisible as exc:
        raise _not_found() from exc
    except ApprovalConflict as exc:
        raise AppError(
            409,
            "assignment_proposal_conflict",
            "conflict",
            "Work order state, version, or assignee eligibility changed",
        ) from exc
    except FsmAssignmentProfileUnavailable as exc:
        raise AppError(
            503,
            "fsm_assignment_unavailable",
            "dependency",
            "Enterprise FSM candidates are temporarily unavailable",
        ) from exc
    except FsmAssignmentProfileConflict as exc:
        raise AppError(
            409,
            "fsm_assignment_candidate_conflict",
            "conflict",
            "Selected enterprise FSM candidate is no longer available",
        ) from exc
    return ActionProposalEnvelope(
        data=ActionProposalResponse(
            proposal_id=proposal.proposal_id,
            approval_id=proposal.approval_id,
            operation_id=proposal.operation_id,
            tool_id="work_order.assign",
            status="PENDING_APPROVAL",
            parameters=proposal.parameters,
        ),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/work-orders/{work_order_id}/closure-proposals",
    response_model=ActionProposalEnvelope,
    status_code=201,
)
def propose_work_order_closure(
    work_order_id: str,
    body: WorkOrderClosureProposalBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
) -> ActionProposalEnvelope:
    try:
        proposal = gateway.propose_work_order_closure(
            identity,
            work_order_id=work_order_id,
            work_order_version=body.work_order_version,
            reason=body.reason,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ApprovalNotVisible as exc:
        raise _not_found() from exc
    except ApprovalConflict as exc:
        raise AppError(
            409,
            "closure_proposal_conflict",
            "conflict",
            "Work order state, version, or closure evidence changed",
        ) from exc
    return ActionProposalEnvelope(
        data=ActionProposalResponse(
            proposal_id=proposal.proposal_id,
            approval_id=proposal.approval_id,
            operation_id=proposal.operation_id,
            tool_id="work_order.close",
            status="PENDING_APPROVAL",
            parameters=proposal.parameters,
        ),
        meta={"request_id": request.state.request_id},
    )


@router.get("/approvals", response_model=ApprovalListEnvelope)
def list_approvals(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> ApprovalListEnvelope:
    try:
        rows = gateway.list_approvals(identity, request_id=request.state.request_id)
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    return ApprovalListEnvelope(
        data=[_approval(a, p, _legal_actions(a, p, identity, authorizer)) for a, p in rows],
        meta={"request_id": request.state.request_id},
    )


@router.get("/approvals/{approval_id}", response_model=ApprovalEnvelope)
def get_approval(
    approval_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> ApprovalEnvelope:
    try:
        approval, proposal = gateway.get_approval(
            identity,
            approval_id,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ApprovalNotVisible as exc:
        raise _not_found() from exc
    return ApprovalEnvelope(
        data=_approval(
            approval,
            proposal,
            _legal_actions(approval, proposal, identity, authorizer),
        ),
        meta={"request_id": request.state.request_id},
    )


@router.post("/approvals/{approval_id}/decisions", response_model=ApprovalEnvelope)
def decide(
    approval_id: str,
    body: DecisionBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> ApprovalEnvelope:
    try:
        approval = gateway.decide(
            identity,
            approval_id=approval_id,
            expected_version=_version(if_match),
            decision=body.decision,
            reason=body.reason,
            request_id=request.state.request_id,
        )
        _, proposal = gateway.get_approval(
            identity, approval_id, request_id=request.state.request_id
        )
    except SeparationOfDutiesViolation as exc:
        raise AppError(
            403, "separation_of_duties", "authorization", "Initiator cannot approve this action"
        ) from exc
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ApprovalNotVisible as exc:
        raise _not_found() from exc
    except ApprovalConflict as exc:
        raise AppError(
            409, "approval_conflict", "conflict", "Approval state or version changed"
        ) from exc
    return ApprovalEnvelope(
        data=_approval(
            approval,
            proposal,
            _legal_actions(approval, proposal, identity, authorizer),
        ),
        meta={"request_id": request.state.request_id},
    )


@router.post(
    "/approvals/{approval_id}/execute",
    response_model=ExecutionEnvelope,
    responses={202: {"model": ExecutionEnvelope}},
)
def execute(
    approval_id: str,
    body: ExecuteBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    if_match: Annotated[str, Header(alias="If-Match")],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
) -> ExecutionEnvelope | JSONResponse:
    try:
        result = gateway.execute_approved_proposal(
            identity,
            approval_id=approval_id,
            parameters=body.parameters,
            request_id=request.state.request_id,
            expected_approval_version=_version(if_match),
            idempotency_key=idempotency_key,
        )
    except AuthorizationDenied as exc:
        raise _forbidden() from exc
    except ApprovalNotVisible as exc:
        raise _not_found() from exc
    if result.status == "DEPENDENCY_UNAVAILABLE" and result.reason is not None and (
        result.reason.startswith("fsm_assignment_")
    ):
        raise AppError(
            409,
            "dependency_unavailable",
            "dependency",
            "Enterprise FSM assignment candidate could not be revalidated",
        )
    envelope = ExecutionEnvelope(
        data=_execution(result),
        meta={"request_id": request.state.request_id},
    )
    if result.status == "RECONCILING":
        return JSONResponse(status_code=202, content=envelope.model_dump(mode="json"))
    return envelope


def _approval(
    approval: ApprovalRequestRecord,
    proposal: ActionProposalRecord,
    legal_actions: list[str],
) -> ApprovalResponse:
    expires = (
        approval.expires_at.replace(tzinfo=UTC)
        if approval.expires_at.tzinfo is None
        else approval.expires_at
    )
    return ApprovalResponse(
        approval_id=approval.approval_id,
        proposal_id=proposal.proposal_id,
        tool_id=proposal.tool_id,
        status=approval.status,
        parameters=proposal.parameters,
        parameters_hash=proposal.parameters_hash,
        resource_version=proposal.resource_version,
        expires_at=expires,
        version=approval.version,
        legal_actions=legal_actions,
    )


def _execution(result: ExecutionResult) -> ExecutionResponse:
    return ExecutionResponse(
        status=result.status,
        reservation_id=result.reservation_id,
        work_order_id=result.work_order_id,
        reason=result.reason,
        notification_id=result.notification_id,
        notification_delivery_id=result.notification_delivery_id,
        purchase_request_id=result.purchase_request_id,
        refund_request_id=result.refund_request_id,
        assignment_delivery_id=result.assignment_delivery_id,
        part_issue_id=result.part_issue_id,
        part_movement_id=result.part_movement_id,
        execution_attempt_id=result.execution_attempt_id,
        reconciliation_id=result.reconciliation_id,
        service_quotation_id=getattr(result, "service_quotation_id", None),
    )


def _purchase_request(
    projection: PurchaseRequestProjection,
) -> PurchaseRequestResponse:
    record = projection.request
    fulfillment = projection.fulfillment
    readiness = projection.reservation_readiness
    return PurchaseRequestResponse(
        purchase_request_id=record.purchase_request_id,
        proposal_id=record.proposal_id,
        incident_id=record.incident_id,
        asset_id=record.asset_id,
        part_number=record.part_number,
        quantity=record.quantity,
        available_quantity_at_submission=record.available_quantity_at_submission,
        estimated_unit_cost_minor=record.estimated_unit_cost_minor,
        estimated_total_cost_minor=record.estimated_total_cost_minor,
        currency=record.currency,
        cost_center=record.cost_center,
        justification=record.justification,
        delivery_target=record.delivery_target,
        provider=record.provider,
        profile_id=record.profile_id,
        profile_version=record.profile_version,
        profile_display_name=record.profile_display_name,
        profile_as_of=_utc(record.profile_as_of) if record.profile_as_of else None,
        profile_expires_at=(
            _utc(record.profile_expires_at) if record.profile_expires_at else None
        ),
        external_request_id=record.external_request_id,
        reconciliation_id=projection.reconciliation_id,
        fulfillment=(
            PurchaseFulfillmentResponse(
                fulfillment_id=fulfillment.fulfillment_id,
                external_order_id=fulfillment.external_order_id,
                status=fulfillment.status,
                requested_quantity=fulfillment.requested_quantity,
                received_quantity=fulfillment.received_quantity,
                expected_delivery_at=(
                    _utc(fulfillment.expected_delivery_at)
                    if fulfillment.expected_delivery_at
                    else None
                ),
                last_event_occurred_at=_utc(fulfillment.last_event_occurred_at),
                reason=fulfillment.reason,
                version=fulfillment.version,
            )
            if fulfillment is not None
            else None
        ),
        reservation_readiness=(
            PurchaseReservationReadinessResponse(
                status=readiness.status,
                reason=readiness.reason,
                next_action=readiness.next_action,
                source=readiness.source,
                source_record_id=readiness.source_record_id,
                as_of=(readiness.as_of.isoformat() if readiness.as_of else None),
                available_quantity=readiness.available_quantity,
            )
            if readiness is not None
            else None
        ),
        status=record.status,
        reason=record.reason,
        submitted_at=_utc(record.submitted_at) if record.submitted_at else None,
        terminal_at=_utc(record.terminal_at) if record.terminal_at else None,
        last_receipt_at=(
            _utc(record.last_receipt_at) if record.last_receipt_at else None
        ),
        version=record.version,
        requested_by_subject_id=record.requested_by_subject_id,
        inventory_source=record.inventory_source,
        inventory_source_record_id=record.inventory_source_record_id,
        inventory_as_of=_utc(record.inventory_as_of),
        created_at=_utc(record.created_at),
    )


def _refund_request(
    record: RefundRequestRecord,
    reconciliation_id: str | None,
) -> RefundRequestResponse:
    return RefundRequestResponse(
        refund_request_id=record.refund_request_id,
        proposal_id=record.proposal_id,
        incident_id=record.incident_id,
        work_order_id=record.work_order_id,
        asset_id=record.asset_id,
        customer_subject_id=record.customer_subject_id,
        customer_confirmation_update_id=record.customer_confirmation_update_id,
        amount_minor=record.amount_minor,
        currency=record.currency,
        cost_center=record.cost_center,
        reason=record.reason,
        compensation_policy_id=record.compensation_policy_id,
        reason_code=record.reason_code,
        delivery_target=record.delivery_target,
        provider=record.provider,
        profile_id=record.profile_id,
        profile_version=record.profile_version,
        profile_display_name=record.profile_display_name,
        profile_as_of=_utc(record.profile_as_of) if record.profile_as_of else None,
        profile_expires_at=(
            _utc(record.profile_expires_at) if record.profile_expires_at else None
        ),
        external_request_id=record.external_request_id,
        reconciliation_id=reconciliation_id,
        status=record.status,
        payment_status=record.payment_status,
        provider_reason=record.provider_reason,
        submitted_at=_utc(record.submitted_at) if record.submitted_at else None,
        terminal_at=_utc(record.terminal_at) if record.terminal_at else None,
        last_receipt_at=(
            _utc(record.last_receipt_at) if record.last_receipt_at else None
        ),
        version=record.version,
        requested_by_subject_id=record.requested_by_subject_id,
        created_at=_utc(record.created_at),
    )


def _legal_actions(
    approval: ApprovalRequestRecord,
    proposal: ActionProposalRecord,
    identity: IdentityContext,
    authorizer: Authorizer,
) -> list[str]:
    if _utc(approval.expires_at) <= datetime.now(UTC):
        return []
    resource = ResourceContext(
        tenant_id=identity.tenant_id,
        resource_id=approval.approval_id,
    )
    if (
        approval.status == "PENDING"
        and proposal.initiator_subject_id != identity.subject_id
        and authorizer.decide(identity, Action.APPROVE_ACTION, resource).allowed
    ):
        return ["APPROVE", "REJECT"]
    if (
        approval.status == "APPROVED"
        and authorizer.decide(identity, Action.EXECUTE_ACTION, resource).allowed
    ):
        return ["EXECUTE"]
    return []


def _version(value: str) -> int:
    try:
        return int(value.strip().strip('"'))
    except ValueError as exc:
        raise AppError(
            400, "invalid_version_precondition", "validation", "If-Match is invalid"
        ) from exc


def _forbidden() -> AppError:
    return AppError(403, "authorization_denied", "authorization", "Action is not allowed")


def _not_found() -> AppError:
    return AppError(
        404, "resource_not_found_or_not_visible", "not_found", "Resource not found or not visible"
    )
