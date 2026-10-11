"""Customer self-service case progress, updates, and result confirmation API."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from typing import Annotated, Literal, NoReturn, Self
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select

from industrial_ops_agent.api.dependencies import get_authorizer, get_database, get_identity
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.customer_portal.service import (
    CustomerCaseConflict,
    CustomerCaseNotVisible,
    CustomerCaseView,
    CustomerPortalService,
    CustomerResolutionView,
    CustomerUpdateView,
    CustomerWorkOrderView,
)
from industrial_ops_agent.maintenance_planning.review_isolation import (
    incident_read_transaction,
    record_incident_disclosure,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    CustomerServiceUpdateRecord,
    IncidentRecord,
    ServiceQuotationDecisionRecord,
    ServiceQuotationRecord,
)

router = APIRouter(prefix="/portal", tags=["customer-portal"])
staff_router = APIRouter(tags=["customer-updates"])

IncidentStatus = Literal[
    "SUBMITTED",
    "NEEDS_INFORMATION",
    "TRIAGED",
    "DIAGNOSING",
    "DIAGNOSED",
    "WORK_ORDER_CREATED",
    "RESOLVED",
    "ESCALATED",
    "CLOSED",
    "CANCELLED",
]


class CustomerUpdateBody(BaseModel):
    client_operation_id: str = Field(min_length=8, max_length=128)
    update_type: Literal["INFORMATION", "EVIDENCE", "RESULT_CONFIRMATION"]
    message: str = Field(min_length=2, max_length=4000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    work_order_id: str | None = Field(default=None, min_length=1, max_length=128)
    result_accepted: bool | None = None
    satisfaction_rating: int | None = Field(default=None, ge=1, le=5)

    @model_validator(mode="after")
    def validate_update(self) -> Self:
        if self.update_type == "EVIDENCE" and not self.evidence_ids:
            raise ValueError("customer_evidence_required")
        if self.update_type == "RESULT_CONFIRMATION" and (
            self.result_accepted is None or self.satisfaction_rating is None
        ):
            raise ValueError("service_result_confirmation_incomplete")
        return self


class CustomerWorkOrderResponse(BaseModel):
    work_order_id: str
    status: str
    priority: str
    sla_due_at: str | None
    assigned_subject_id: str | None
    version: int
    result_accepted: bool | None
    updated_at: str
    legal_actions: list[str]


class CustomerResolutionResponse(BaseModel):
    resolution_id: str
    mode: str
    summary: str
    evidence_ids: list[str]
    resolved_at: str
    result_accepted: bool | None
    confirmation_update_id: str | None
    confirmation_occurred_at: str | None


class CustomerCaseResponse(BaseModel):
    incident_id: str
    asset_id: str
    asset_display_name: str | None
    model_code: str | None
    site_id: str | None
    site_name: str | None
    description: str
    status: str
    version: int
    created_at: str
    updated_at: str
    work_orders: list[CustomerWorkOrderResponse]
    resolution: CustomerResolutionResponse | None
    legal_actions: list[str]


class CustomerUpdateResponse(BaseModel):
    update_id: str
    incident_id: str
    work_order_id: str | None
    client_operation_id: str
    update_type: str
    message: str
    evidence_ids: list[str]
    result_accepted: bool | None
    satisfaction_rating: int | None
    actor_subject_id: str
    occurred_at: str


class PortalMeta(BaseModel):
    request_id: str
    total: int | None = None
    limit: int | None = None
    offset: int | None = None


class CustomerCasesEnvelope(BaseModel):
    data: list[CustomerCaseResponse]
    meta: PortalMeta


class CustomerCaseEnvelope(BaseModel):
    data: CustomerCaseResponse
    meta: PortalMeta


class CustomerUpdatesEnvelope(BaseModel):
    data: list[CustomerUpdateResponse]
    meta: PortalMeta


class CustomerUpdateEnvelope(BaseModel):
    data: CustomerUpdateResponse
    meta: PortalMeta


class CustomerQuotationLineItemResponse(BaseModel):
    code: str
    description: str
    quantity: int
    unit_price: str
    line_total: str


class CustomerQuotationResponse(BaseModel):
    quotation_id: str
    version: int
    status: str
    provider: str
    candidate_id: str
    source_version: str
    line_items: list[CustomerQuotationLineItemResponse]
    currency: str
    subtotal: str
    discount: str
    tax: str
    total: str
    entitlement_decision: str
    published_at: str
    expires_at: str
    state_version: int
    legal_actions: list[str]


class CustomerQuotationHistoryResponse(BaseModel):
    quotation_id: str
    version: int
    status: str
    total: str
    currency: str
    published_at: str
    expires_at: str


class CustomerQuotationViewResponse(BaseModel):
    current: CustomerQuotationResponse | None
    history: list[CustomerQuotationHistoryResponse]


class CustomerQuotationEnvelope(BaseModel):
    data: CustomerQuotationViewResponse
    meta: PortalMeta


class CustomerQuotationDecisionBody(BaseModel):
    model_config = {"extra": "forbid"}

    decision: Literal["ACCEPTED", "REJECTED"]


class CustomerQuotationDecisionResponse(BaseModel):
    decision_id: str
    quotation_id: str
    quotation_version: int
    decision: str
    status: str
    decided_at: str


class CustomerQuotationDecisionEnvelope(BaseModel):
    data: CustomerQuotationDecisionResponse
    meta: PortalMeta


class CustomerQuotationDecisionConflict(Exception):
    pass


@router.get("/cases", response_model=CustomerCasesEnvelope)
def list_customer_cases(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    status: Annotated[IncidentStatus | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> CustomerCasesEnvelope:
    try:
        cases, total = CustomerPortalService(database, authorizer).list_cases(
            identity,
            status=status,
            limit=limit,
            offset=offset,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        _raise_portal_error(exc)
    return CustomerCasesEnvelope(
        data=[_case_response(case) for case in cases],
        meta=PortalMeta(
            request_id=request.state.request_id,
            total=total,
            limit=limit,
            offset=offset,
        ),
    )


@router.get("/cases/{incident_id}", response_model=CustomerCaseEnvelope)
def get_customer_case(
    incident_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> CustomerCaseEnvelope:
    try:
        case = CustomerPortalService(database, authorizer).get_case(
            identity,
            incident_id,
            request_id=request.state.request_id,
        )
    except (CustomerCaseNotVisible, CustomerCaseConflict, AuthorizationDenied) as exc:
        _raise_portal_error(exc)
    return CustomerCaseEnvelope(
        data=_case_response(case),
        meta=PortalMeta(request_id=request.state.request_id),
    )


@router.get(
    "/cases/{incident_id}/updates",
    response_model=CustomerUpdatesEnvelope,
)
def list_customer_updates(
    incident_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> CustomerUpdatesEnvelope:
    try:
        updates = CustomerPortalService(database, authorizer).list_updates(
            identity,
            incident_id,
            request_id=request.state.request_id,
        )
    except (CustomerCaseNotVisible, CustomerCaseConflict, AuthorizationDenied) as exc:
        _raise_portal_error(exc)
    return CustomerUpdatesEnvelope(
        data=[_update_response(update) for update in updates],
        meta=PortalMeta(request_id=request.state.request_id),
    )


@router.post(
    "/cases/{incident_id}/updates",
    response_model=CustomerUpdateEnvelope,
)
def append_customer_update(
    incident_id: str,
    body: CustomerUpdateBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> CustomerUpdateEnvelope:
    try:
        update = CustomerPortalService(database, authorizer).append_update(
            identity,
            incident_id,
            client_operation_id=body.client_operation_id,
            update_type=body.update_type,
            message=body.message,
            evidence_ids=body.evidence_ids,
            work_order_id=body.work_order_id,
            result_accepted=body.result_accepted,
            satisfaction_rating=body.satisfaction_rating,
            request_id=request.state.request_id,
        )
    except (CustomerCaseNotVisible, CustomerCaseConflict, AuthorizationDenied) as exc:
        _raise_portal_error(exc)
    return CustomerUpdateEnvelope(
        data=_update_response(update),
        meta=PortalMeta(request_id=request.state.request_id),
    )


@router.get(
    "/cases/{incident_id}/service-quotation",
    response_model=CustomerQuotationEnvelope,
)
def get_customer_service_quotation(
    incident_id: str,
    request: Request,
    response: Response,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> CustomerQuotationEnvelope:
    try:
        case = CustomerPortalService(database, authorizer).get_case(
            identity,
            incident_id,
            request_id=request.state.request_id,
        )
        authorizer.require(
            identity,
            Action.READ_CUSTOMER_SERVICE_QUOTATION,
            ResourceContext(identity.tenant_id, incident_id, case.asset_id),
            request_id=request.state.request_id,
        )
        with database.transaction(identity.tenant_context) as session:
            incident = session.scalar(
                select(IncidentRecord).where(
                    IncidentRecord.tenant_id == identity.tenant_id,
                    IncidentRecord.incident_id == incident_id,
                    IncidentRecord.reporter_subject_id == identity.subject_id,
                )
            )
            if incident is None:
                raise CustomerCaseNotVisible
            records = list(
                session.scalars(
                    select(ServiceQuotationRecord)
                    .where(
                        ServiceQuotationRecord.tenant_id == identity.tenant_id,
                        ServiceQuotationRecord.incident_id == incident_id,
                    )
                    .order_by(ServiceQuotationRecord.quotation_version.desc())
                )
            )
    except (CustomerCaseNotVisible, CustomerCaseConflict, AuthorizationDenied) as exc:
        _raise_portal_error(exc)
    now = datetime.now(UTC)
    current = next(
        (
            record
            for record in records
            if record.status == "PUBLISHED" and _utc_time(record.expires_at) > now
        ),
        None,
    )
    if current is not None:
        response.headers["ETag"] = f'"{current.state_version}"'
    with database.transaction(identity.tenant_context) as session:
        record_incident_disclosure(
            session,
            identity,
            (incident_id,),
            resource_kind="customer-portal",
            resource_id=request.url.path,
            request_id=request.state.request_id,
        )
    return CustomerQuotationEnvelope(
        data=CustomerQuotationViewResponse(
            current=_customer_quotation_response(current, now) if current else None,
            history=[_customer_quotation_history(record, now) for record in records],
        ),
        meta=PortalMeta(request_id=request.state.request_id),
    )


@router.post(
    "/cases/{incident_id}/service-quotation/decision",
    response_model=CustomerQuotationDecisionEnvelope,
)
def decide_customer_service_quotation(
    incident_id: str,
    body: CustomerQuotationDecisionBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=8, max_length=128),
    ],
) -> CustomerQuotationDecisionEnvelope:
    try:
        expected_version = _etag_version(if_match)
        case = CustomerPortalService(database, authorizer).get_case(
            identity,
            incident_id,
            request_id=request.state.request_id,
        )
        authorizer.require(
            identity,
            Action.DECIDE_CUSTOMER_SERVICE_QUOTATION,
            ResourceContext(identity.tenant_id, incident_id, case.asset_id),
            request_id=request.state.request_id,
        )
        decision, quotation = _record_customer_quotation_decision(
            database,
            identity,
            incident_id=incident_id,
            decision=body.decision,
            expected_version=expected_version,
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except CustomerQuotationDecisionConflict as exc:
        raise AppError(
            409,
            "service_quotation_decision_conflict",
            "conflict",
            "Quotation state, version, or idempotency binding changed",
        ) from exc
    except (CustomerCaseNotVisible, CustomerCaseConflict, AuthorizationDenied) as exc:
        _raise_portal_error(exc)
    return CustomerQuotationDecisionEnvelope(
        data=CustomerQuotationDecisionResponse(
            decision_id=decision.decision_id,
            quotation_id=decision.quotation_id,
            quotation_version=decision.quotation_version,
            decision=decision.decision,
            status=quotation.status,
            decided_at=_utc_time(decision.decided_at).isoformat(),
        ),
        meta=PortalMeta(request_id=request.state.request_id),
    )


@staff_router.get(
    "/incidents/{incident_id}/customer-updates",
    response_model=CustomerUpdatesEnvelope,
)
def list_incident_customer_updates(
    incident_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> CustomerUpdatesEnvelope:
    try:
        updates = CustomerPortalService(database, authorizer).list_updates_for_staff(
            identity,
            incident_id,
            request_id=request.state.request_id,
        )
    except (CustomerCaseNotVisible, CustomerCaseConflict, AuthorizationDenied) as exc:
        _raise_portal_error(exc)
    return CustomerUpdatesEnvelope(
        data=[_update_response(update) for update in updates],
        meta=PortalMeta(request_id=request.state.request_id),
    )


def _record_customer_quotation_decision(
    database: Database,
    identity: IdentityContext,
    *,
    incident_id: str,
    decision: str,
    expected_version: int,
    idempotency_key: str,
    request_id: str,
) -> tuple[ServiceQuotationDecisionRecord, ServiceQuotationRecord]:
    now = datetime.now(UTC)
    with incident_read_transaction(
        database, identity, (incident_id,), request_id=request_id
    ) as session:
        incident = session.scalar(
            select(IncidentRecord)
            .where(
                IncidentRecord.tenant_id == identity.tenant_id,
                IncidentRecord.incident_id == incident_id,
                IncidentRecord.reporter_subject_id == identity.subject_id,
            )
            .with_for_update()
        )
        if incident is None:
            raise CustomerCaseNotVisible
        existing = session.scalar(
            select(ServiceQuotationDecisionRecord).where(
                ServiceQuotationDecisionRecord.tenant_id == identity.tenant_id,
                ServiceQuotationDecisionRecord.reporter_subject_id == identity.subject_id,
                ServiceQuotationDecisionRecord.client_operation_id == idempotency_key,
            )
        )
        if existing is not None:
            quotation = session.scalar(
                select(ServiceQuotationRecord).where(
                    ServiceQuotationRecord.tenant_id == identity.tenant_id,
                    ServiceQuotationRecord.quotation_id == existing.quotation_id,
                    ServiceQuotationRecord.incident_id == incident_id,
                )
            )
            if quotation is None:
                raise CustomerQuotationDecisionConflict
            matches = existing.request_digest == _quotation_decision_digest(
                incident_id, existing.quotation_id, decision, expected_version
            )
            # Pre-version-binding receipts are immutable. Their accepted/rejected
            # quotation retains the single state increment performed at decision time.
            legacy_matches = (
                quotation.status == existing.decision
                and quotation.state_version == expected_version + 1
                and existing.request_digest
                == _quotation_decision_digest(incident_id, existing.quotation_id, decision)
            )
            if not (matches or legacy_matches):
                raise CustomerQuotationDecisionConflict
            # Replay the recorded decision, not a new action on today's quotation.
            # Expiry and PUBLISHED checks apply only when creating a decision.
            return existing, quotation
        quotation = session.scalar(
            select(ServiceQuotationRecord)
            .where(
                ServiceQuotationRecord.tenant_id == identity.tenant_id,
                ServiceQuotationRecord.incident_id == incident_id,
            )
            .order_by(ServiceQuotationRecord.quotation_version.desc())
            .with_for_update()
        )
        if (
            quotation is None
            or quotation.status != "PUBLISHED"
            or _utc_time(quotation.expires_at) <= now
            or quotation.state_version != expected_version
        ):
            raise CustomerQuotationDecisionConflict
        request_digest = _quotation_decision_digest(
            incident_id,
            quotation.quotation_id,
            decision,
            expected_version,
        )
        record = ServiceQuotationDecisionRecord(
            decision_id=f"quotation-decision-{uuid4().hex}",
            tenant_id=identity.tenant_id,
            quotation_id=quotation.quotation_id,
            quotation_version=quotation.quotation_version,
            incident_id=incident_id,
            reporter_subject_id=identity.subject_id,
            decision=decision,
            client_operation_id=idempotency_key,
            request_digest=request_digest,
            decided_at=now,
            created_at=now,
            updated_at=now,
        )
        session.add(record)
        timeline_operation = "quote-" + sha256(idempotency_key.encode()).hexdigest()[:40]
        session.add(
            CustomerServiceUpdateRecord(
                update_id=f"customer-update-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                incident_id=incident_id,
                work_order_id=None,
                client_operation_id=timeline_operation,
                update_type="SERVICE_QUOTATION_DECISION",
                message=(
                    "客户已接受当前服务报价，等待售后按既有流程处理"
                    if decision == "ACCEPTED"
                    else "客户已拒绝当前服务报价，等待售后人工跟进"
                ),
                evidence_ids=[],
                result_accepted=decision == "ACCEPTED",
                satisfaction_rating=None,
                actor_subject_id=identity.subject_id,
                occurred_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        quotation.status = decision
        quotation.state_version += 1
        quotation.updated_at = now
        return record, quotation


def _customer_quotation_response(
    record: ServiceQuotationRecord,
    now: datetime,
) -> CustomerQuotationResponse:
    status = _customer_quotation_status(record, now)
    return CustomerQuotationResponse(
        quotation_id=record.quotation_id,
        version=record.quotation_version,
        status=status,
        provider=record.provider,
        candidate_id=record.candidate_id,
        source_version=record.source_version,
        line_items=[CustomerQuotationLineItemResponse(**item) for item in record.line_items_json],
        currency=record.currency,
        subtotal=record.subtotal,
        discount=record.discount,
        tax=record.tax,
        total=record.total,
        entitlement_decision=str(record.entitlement_json.get("decision", "UNKNOWN")),
        published_at=_utc_time(record.published_at).isoformat(),
        expires_at=_utc_time(record.expires_at).isoformat(),
        state_version=record.state_version,
        legal_actions=["ACCEPT", "REJECT"] if status == "PUBLISHED" else [],
    )


def _customer_quotation_history(
    record: ServiceQuotationRecord,
    now: datetime,
) -> CustomerQuotationHistoryResponse:
    return CustomerQuotationHistoryResponse(
        quotation_id=record.quotation_id,
        version=record.quotation_version,
        status=_customer_quotation_status(record, now),
        total=record.total,
        currency=record.currency,
        published_at=_utc_time(record.published_at).isoformat(),
        expires_at=_utc_time(record.expires_at).isoformat(),
    )


def _customer_quotation_status(
    record: ServiceQuotationRecord,
    now: datetime,
) -> str:
    if record.status == "PUBLISHED" and _utc_time(record.expires_at) <= now:
        return "EXPIRED"
    return record.status


def _quotation_decision_digest(
    incident_id: str,
    quotation_id: str,
    decision: str,
    expected_version: int | None = None,
) -> str:
    payload = json.dumps(
        {
            "incident_id": incident_id,
            "quotation_id": quotation_id,
            "decision": decision,
            **({"expected_version": expected_version} if expected_version is not None else {}),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(payload).hexdigest()


def _etag_version(value: str) -> int:
    normalized = value.strip()
    if normalized.startswith('W/"') and normalized.endswith('"'):
        normalized = normalized[3:-1]
    elif normalized.startswith('"') and normalized.endswith('"'):
        normalized = normalized[1:-1]
    if not normalized.isdigit() or int(normalized) < 1:
        raise CustomerQuotationDecisionConflict
    return int(normalized)


def _utc_time(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _case_response(case: CustomerCaseView) -> CustomerCaseResponse:
    legal_actions = [] if case.status in {"CLOSED", "CANCELLED"} else ["ADD_UPDATE"]
    if (
        case.status == "RESOLVED"
        and case.resolution is not None
        and case.resolution.mode == "REMOTE"
    ):
        legal_actions.append("CONFIRM_REMOTE_RESULT")
    return CustomerCaseResponse(
        incident_id=case.incident_id,
        asset_id=case.asset_id,
        asset_display_name=case.asset_display_name,
        model_code=case.model_code,
        site_id=case.site_id,
        site_name=case.site_name,
        description=case.description,
        status=case.status,
        version=case.version,
        created_at=case.created_at.isoformat(),
        updated_at=case.updated_at.isoformat(),
        work_orders=[_work_order_response(work) for work in case.work_orders],
        resolution=(_resolution_response(case.resolution) if case.resolution is not None else None),
        legal_actions=legal_actions,
    )


def _resolution_response(
    resolution: CustomerResolutionView,
) -> CustomerResolutionResponse:
    return CustomerResolutionResponse(
        resolution_id=resolution.resolution_id,
        mode=resolution.mode,
        summary=resolution.summary,
        evidence_ids=list(resolution.evidence_ids),
        resolved_at=resolution.resolved_at.isoformat(),
        result_accepted=resolution.result_accepted,
        confirmation_update_id=resolution.confirmation_update_id,
        confirmation_occurred_at=(
            resolution.confirmation_occurred_at.isoformat()
            if resolution.confirmation_occurred_at is not None
            else None
        ),
    )


def _work_order_response(work: CustomerWorkOrderView) -> CustomerWorkOrderResponse:
    legal_actions = []
    if work.status in {"COMPLETED", "VERIFIED"}:
        legal_actions.append("CONFIRM_RESULT")
    return CustomerWorkOrderResponse(
        work_order_id=work.work_order_id,
        status=work.status,
        priority=work.priority,
        sla_due_at=work.sla_due_at.isoformat() if work.sla_due_at is not None else None,
        assigned_subject_id=work.assigned_subject_id,
        version=work.version,
        result_accepted=work.result_accepted,
        updated_at=work.updated_at.isoformat(),
        legal_actions=legal_actions,
    )


def _update_response(update: CustomerUpdateView) -> CustomerUpdateResponse:
    return CustomerUpdateResponse(
        update_id=update.update_id,
        incident_id=update.incident_id,
        work_order_id=update.work_order_id,
        client_operation_id=update.client_operation_id,
        update_type=update.update_type,
        message=update.message,
        evidence_ids=list(update.evidence_ids),
        result_accepted=update.result_accepted,
        satisfaction_rating=update.satisfaction_rating,
        actor_subject_id=update.actor_subject_id,
        occurred_at=update.occurred_at.isoformat(),
    )


def _raise_portal_error(
    error: CustomerCaseNotVisible | CustomerCaseConflict | AuthorizationDenied,
) -> NoReturn:
    if isinstance(error, CustomerCaseNotVisible):
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
        "Customer case state or update conflict",
    ) from error
