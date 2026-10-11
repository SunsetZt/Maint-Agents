"""Provider-authenticated enterprise FSM assignment receipt endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel

from industrial_ops_agent.api.dependencies import get_tool_gateway
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.tools.gateway import (
    FsmAssignmentReceiptRejected,
    ToolGateway,
)

router = APIRouter(
    prefix="/fsm-assignment-receipts",
    tags=["fsm-assignment-receipts"],
)


class FsmAssignmentReceiptResponse(BaseModel):
    delivery_id: str
    status: str
    state_changed: bool
    reconciliation_id: str | None


class FsmAssignmentReceiptEnvelope(BaseModel):
    data: FsmAssignmentReceiptResponse
    meta: dict[str, str]


@router.post("/{provider}", response_model=FsmAssignmentReceiptEnvelope)
async def receive_fsm_assignment_receipt(
    provider: str,
    request: Request,
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    signature: Annotated[str, Header(alias="X-FSM-Signature")],
) -> FsmAssignmentReceiptEnvelope:
    try:
        result = gateway.process_fsm_assignment_receipt(
            provider=provider,
            raw_body=await request.body(),
            signature=signature,
        )
    except FsmAssignmentReceiptRejected as exc:
        raise AppError(
            401,
            "fsm_assignment_receipt_rejected",
            "authentication",
            "FSM assignment receipt authentication failed",
        ) from exc
    return FsmAssignmentReceiptEnvelope(
        data=FsmAssignmentReceiptResponse(
            delivery_id=result.delivery_id,
            status=result.status,
            state_changed=result.state_changed,
            reconciliation_id=result.reconciliation_id,
        ),
        meta={"request_id": request.state.request_id},
    )
