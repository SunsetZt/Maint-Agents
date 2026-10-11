"""Evidence-only handoffs to an external certified equipment-control system."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, Field

from industrial_ops_agent.api.dependencies import get_identity, get_tool_gateway
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.persistence.models import EquipmentControlHandoffRecord
from industrial_ops_agent.tools.gateway import (
    EquipmentControlHandoffConflict,
    EquipmentControlHandoffNotVisible,
    ToolGateway,
)

router = APIRouter(tags=["equipment-control-handoffs"])


class EquipmentControlHandoffBody(BaseModel):
    incident_version: int = Field(ge=1)
    requested_operation: Literal[
        "STOP",
        "REMOTE_START",
        "PLC_PARAMETER_CHANGE",
        "INTERLOCK_OVERRIDE",
    ]
    reason: str = Field(min_length=10, max_length=2000)
    evidence_ids: list[Annotated[str, Field(min_length=1, max_length=128)]] = Field(
        min_length=1,
        max_length=20,
    )


class EquipmentControlHandoffResponse(BaseModel):
    handoff_id: str
    incident_id: str
    asset_id: str
    tool_call_id: str
    requested_tool_id: str
    requested_operation: str
    reason: str
    evidence_ids: list[str]
    target_system: str
    status: str
    requested_by_subject_id: str
    incident_version: int
    created_at: datetime


class EquipmentControlHandoffEnvelope(BaseModel):
    data: EquipmentControlHandoffResponse
    meta: dict[str, str]


class EquipmentControlHandoffListEnvelope(BaseModel):
    data: list[EquipmentControlHandoffResponse]
    meta: dict[str, str]


@router.post(
    "/incidents/{incident_id}/equipment-control-handoffs",
    response_model=EquipmentControlHandoffEnvelope,
    status_code=201,
)
def create_equipment_control_handoff(
    incident_id: str,
    body: EquipmentControlHandoffBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=8, max_length=128)],
) -> EquipmentControlHandoffEnvelope:
    try:
        record = gateway.create_equipment_control_handoff(
            identity,
            incident_id=incident_id,
            incident_version=body.incident_version,
            requested_operation=body.requested_operation,
            reason=body.reason,
            evidence_ids=body.evidence_ids,
            client_operation_id=idempotency_key,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise AppError(403, "forbidden", "authorization", "Action is not permitted") from exc
    except EquipmentControlHandoffNotVisible as exc:
        raise AppError(
            404, "handoff_incident_not_found", "not_found", "Incident not found"
        ) from exc
    except EquipmentControlHandoffConflict as exc:
        raise AppError(
            409,
            "equipment_control_handoff_conflict",
            "conflict",
            "Handoff facts, evidence, version, or idempotency binding are invalid",
            details={"reason": exc.reason},
        ) from exc
    return EquipmentControlHandoffEnvelope(
        data=_response(record),
        meta={"request_id": request.state.request_id},
    )


@router.get(
    "/incidents/{incident_id}/equipment-control-handoffs",
    response_model=EquipmentControlHandoffListEnvelope,
)
def list_equipment_control_handoffs(
    incident_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
) -> EquipmentControlHandoffListEnvelope:
    try:
        records = gateway.list_equipment_control_handoffs(
            identity,
            incident_id=incident_id,
            request_id=request.state.request_id,
        )
    except AuthorizationDenied as exc:
        raise AppError(403, "forbidden", "authorization", "Action is not permitted") from exc
    except EquipmentControlHandoffNotVisible as exc:
        raise AppError(
            404, "handoff_incident_not_found", "not_found", "Incident not found"
        ) from exc
    return EquipmentControlHandoffListEnvelope(
        data=[_response(record) for record in records],
        meta={"request_id": request.state.request_id},
    )


def _response(record: EquipmentControlHandoffRecord) -> EquipmentControlHandoffResponse:
    created_at = (
        record.created_at.replace(tzinfo=UTC)
        if record.created_at.tzinfo is None
        else record.created_at.astimezone(UTC)
    )
    return EquipmentControlHandoffResponse(
        handoff_id=record.handoff_id,
        incident_id=record.incident_id,
        asset_id=record.asset_id,
        tool_call_id=record.tool_call_id,
        requested_tool_id=record.requested_tool_id,
        requested_operation=record.requested_operation,
        reason=record.reason,
        evidence_ids=record.evidence_ids,
        target_system=record.target_system,
        status=record.status,
        requested_by_subject_id=record.requested_by_subject_id,
        incident_version=record.incident_version,
        created_at=created_at,
    )
