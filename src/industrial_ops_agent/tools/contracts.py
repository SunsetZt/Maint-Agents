"""Stable contracts shared by governed tool adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ToolTransportBinding:
    """Non-secret deployment identity for one governed tool transport."""

    transport: str
    local_tool_id: str
    remote_tool_name: str
    server_id: str | None = None
    server_version: str | None = None
    required_scopes: tuple[str, ...] = ()
    input_schema_digest: str | None = None


@dataclass(frozen=True, slots=True)
class ReservationResult:
    reservation_id: str
    operation_id: str
    part_number: str
    quantity: int
    source: str
    source_record_id: str
    as_of: datetime


@dataclass(frozen=True, slots=True)
class PartIssueResult:
    issue_id: str
    operation_id: str
    reservation_id: str
    part_number: str
    quantity: int
    status: str
    source: str
    source_record_id: str
    as_of: datetime


@dataclass(frozen=True, slots=True)
class PartMovementResult:
    movement_id: str
    operation_id: str
    movement_kind: str
    part_issue_id: str
    reservation_id: str
    part_number: str
    quantity: int
    status: str
    source: str
    source_record_id: str
    as_of: datetime


@dataclass(frozen=True, slots=True, repr=False)
class NotificationDestination:
    """A raw address that may only be revealed at the delivery adapter boundary."""

    _value: str

    def __post_init__(self) -> None:
        if not self._value:
            raise ValueError("notification destination must not be empty")

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "NotificationDestination(<redacted>)"

    def __str__(self) -> str:
        return "<redacted>"


@dataclass(frozen=True, slots=True)
class NotificationContact:
    """Authoritative contact resolved in memory without persisting its destination."""

    contact_point_id: str
    channel: str
    destination: NotificationDestination
    masked_destination: str
    source_system: str
    source_record_id: str
    source_version: str
    as_of: datetime
    verification_status: str
    consent_status: str
    consent_basis: str
    expires_at: datetime | None


@dataclass(frozen=True, slots=True)
class NotificationDeliveryResult:
    status: str
    provider_message_id: str | None
    reason: str | None
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class NotificationReceipt:
    tenant_id: str
    provider_event_id: str
    operation_id: str
    provider_message_id: str
    status: str
    occurred_at: datetime
    reason: str | None
    signature_digest: str


@dataclass(frozen=True, slots=True)
class ProcurementProfile:
    """Safe, non-secret identity for one tenant-authorized ERP handoff profile."""

    provider: str
    profile_id: str
    profile_version: str
    display_name: str
    as_of: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class ProcurementDeliveryResult:
    status: str
    external_request_id: str | None
    reason: str | None
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class ProcurementReceipt:
    tenant_id: str
    provider_event_id: str
    operation_id: str
    external_request_id: str
    status: str
    occurred_at: datetime
    reason: str | None
    signature_digest: str


@dataclass(frozen=True, slots=True)
class ProcurementFulfillmentReceipt:
    """Normalized order/receipt progress from the bound procurement provider."""

    tenant_id: str
    provider_event_id: str
    operation_id: str
    external_request_id: str
    external_order_id: str
    status: str
    received_quantity: int
    expected_delivery_at: datetime | None
    occurred_at: datetime
    reason: str | None
    signature_digest: str


@dataclass(frozen=True, slots=True)
class RefundProfile:
    """Safe identity for a tenant and customer authorized finance handoff."""

    provider: str
    profile_id: str
    profile_version: str
    display_name: str
    as_of: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class RefundDeliveryResult:
    status: str
    payment_status: str
    external_request_id: str | None
    reason: str | None
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class RefundReceipt:
    tenant_id: str
    provider_event_id: str
    operation_id: str
    external_request_id: str
    status: str
    payment_status: str
    occurred_at: datetime
    reason: str | None
    signature_digest: str


@dataclass(frozen=True, slots=True)
class FsmAssignmentCandidate:
    """Safe projection of one authoritative, named enterprise FSM candidate."""

    provider: str
    candidate_profile_id: str
    profile_version: str
    assignee_subject_id: str
    display_name: str
    matched_skill_codes: tuple[str, ...]
    service_window_start: datetime
    service_window_end: datetime
    travel_minutes: int
    remaining_work_minutes: int
    eligibility_code: str
    source_record_id: str
    as_of: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class FsmAssignmentDeliveryResult:
    status: str
    external_assignment_id: str | None
    reason: str | None
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class FsmAssignmentReceipt:
    tenant_id: str
    provider_event_id: str
    operation_id: str
    candidate_profile_id: str
    assignee_subject_id: str
    external_assignment_id: str
    status: str
    occurred_at: datetime
    reason: str | None
    signature_digest: str


@dataclass(frozen=True, slots=True)
class ServiceQuotationLineItem:
    """One closed, display-safe line item returned by the enterprise CPQ."""

    code: str
    description: str
    quantity: int
    unit_price: str
    line_total: str


@dataclass(frozen=True, slots=True)
class ServiceQuotationCandidate:
    """Safe immutable projection of one authoritative enterprise CPQ candidate."""

    provider: str
    candidate_id: str
    source_version: str
    display_name: str
    incident_ref: str
    asset_ref: str
    diagnosis_ref: str
    currency: str
    line_items: tuple[ServiceQuotationLineItem, ...]
    subtotal: str
    discount: str
    tax: str
    total: str
    as_of: datetime
    expires_at: datetime


class ReservationOutcomeUnknown(RuntimeError):
    """The WMS may have committed a reservation, so callers must reconcile."""


class PartIssueOutcomeUnknown(RuntimeError):
    """The WMS may have committed an issue, so callers must only reconcile."""


class PartMovementOutcomeUnknown(RuntimeError):
    """The WMS may have applied a movement, so callers must only reconcile."""


class EnterpriseToolError(RuntimeError):
    """Stable external-adapter failure without response bodies or credentials."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ReadonlyToolAdapter(Protocol):
    def invoke(self, tool_id: str, parameters: dict[str, Any]) -> dict[str, Any]: ...


class ServiceQuotationCatalogAdapter(Protocol):
    """Provider-neutral, read-only CPQ candidate boundary."""

    provider_id: str

    def list_candidates(
        self,
        *,
        tenant_id: str,
        incident_ref: str,
        asset_ref: str,
        diagnosis_ref: str,
    ) -> tuple[ServiceQuotationCandidate, ...]: ...

    def resolve_candidate(
        self,
        *,
        tenant_id: str,
        candidate_id: str,
        incident_ref: str,
        asset_ref: str,
        diagnosis_ref: str,
    ) -> ServiceQuotationCandidate | None: ...


@runtime_checkable
class ToolTransportBindingProvider(Protocol):
    def transport_binding(self, tool_id: str) -> ToolTransportBinding: ...


class PartsAdapter(Protocol):
    def check_availability(self, *, part_number: str, quantity: int) -> dict[str, object]: ...

    def reserve(
        self,
        *,
        operation_id: str,
        part_number: str,
        quantity: int,
    ) -> ReservationResult: ...

    def reconcile(self, operation_id: str) -> ReservationResult | None: ...

    def issue(
        self,
        *,
        operation_id: str,
        reservation_id: str,
        part_number: str,
        quantity: int,
    ) -> PartIssueResult: ...

    def reconcile_issue(self, operation_id: str) -> PartIssueResult | None: ...

    def apply_movement(
        self,
        *,
        operation_id: str,
        movement_kind: str,
        part_issue_id: str,
        reservation_id: str,
        part_number: str,
        quantity: int,
    ) -> PartMovementResult: ...

    def reconcile_movement(
        self,
        operation_id: str,
        movement_kind: str,
    ) -> PartMovementResult | None: ...


class NotificationDeliveryAdapter(Protocol):
    provider_id: str

    def resolve_contacts(
        self,
        *,
        tenant_id: str,
        recipient_subject_id: str,
    ) -> tuple[NotificationContact, ...]: ...

    def deliver(
        self,
        *,
        operation_id: str,
        contact: NotificationContact,
        rendered_message: str,
    ) -> NotificationDeliveryResult: ...

    def reconcile(self, operation_id: str) -> NotificationDeliveryResult | None: ...

    def verify_receipt(self, raw_body: bytes, signature: str) -> NotificationReceipt: ...


class ProcurementDeliveryAdapter(Protocol):
    provider_id: str

    def resolve_profile(self, *, tenant_id: str) -> ProcurementProfile | None: ...

    def submit(
        self,
        *,
        operation_id: str,
        profile: ProcurementProfile,
        payload: dict[str, object],
    ) -> ProcurementDeliveryResult: ...

    def reconcile(self, operation_id: str) -> ProcurementDeliveryResult | None: ...

    def verify_receipt(self, raw_body: bytes, signature: str) -> ProcurementReceipt: ...

    def verify_fulfillment_receipt(
        self,
        raw_body: bytes,
        signature: str,
    ) -> ProcurementFulfillmentReceipt: ...


class RefundDeliveryAdapter(Protocol):
    provider_id: str

    def resolve_profile(
        self,
        *,
        tenant_id: str,
        customer_subject_id: str,
    ) -> RefundProfile | None: ...

    def submit(
        self,
        *,
        operation_id: str,
        profile: RefundProfile,
        payload: dict[str, object],
    ) -> RefundDeliveryResult: ...

    def reconcile(self, operation_id: str) -> RefundDeliveryResult | None: ...

    def verify_receipt(self, raw_body: bytes, signature: str) -> RefundReceipt: ...


class FsmAssignmentAdapter(Protocol):
    """Provider-neutral read boundary for server-authorized assignment candidates."""

    provider_id: str

    def list_candidates(
        self,
        *,
        tenant_id: str,
        work_order_id: str,
        asset_id: str,
        site_id: str,
        service_window_start: datetime | None,
        service_window_end: datetime | None,
    ) -> tuple[FsmAssignmentCandidate, ...]: ...

    def submit(
        self,
        *,
        operation_id: str,
        candidate: FsmAssignmentCandidate,
        payload: dict[str, object],
    ) -> FsmAssignmentDeliveryResult: ...

    def reconcile(self, operation_id: str) -> FsmAssignmentDeliveryResult | None: ...

    def verify_receipt(self, raw_body: bytes, signature: str) -> FsmAssignmentReceipt: ...


class ToolRateLimiter(Protocol):
    def require(self, tenant_id: str, tool_id: str, limit: int) -> None: ...
