"""Provider-authenticated enterprise finance refund receipt endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel

from industrial_ops_agent.api.dependencies import get_tool_gateway
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.tools.gateway import (
    RefundRequestReceiptRejected,
    ToolGateway,
)

router = APIRouter(
    prefix="/refund-request-receipts",
    tags=["refund-request-receipts"],
)


class RefundRequestReceiptResponse(BaseModel):
    refund_request_id: str
    status: str
    payment_status: str
    state_changed: bool
    reconciliation_id: str | None


class RefundRequestReceiptEnvelope(BaseModel):
    data: RefundRequestReceiptResponse
    meta: dict[str, str]


@router.post("/{provider}", response_model=RefundRequestReceiptEnvelope)
async def receive_refund_request_receipt(
    provider: str,
    request: Request,
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    signature: Annotated[str, Header(alias="X-Refund-Signature")],
) -> RefundRequestReceiptEnvelope:
    try:
        result = gateway.process_refund_request_receipt(
            provider=provider,
            raw_body=await request.body(),
            signature=signature,
        )
    except RefundRequestReceiptRejected as exc:
        raise AppError(
            401,
            "refund_receipt_rejected",
            "authentication",
            "Refund receipt authentication failed",
        ) from exc
    return RefundRequestReceiptEnvelope(
        data=RefundRequestReceiptResponse(
            refund_request_id=result.refund_request_id,
            status=result.status,
            payment_status=result.payment_status,
            state_changed=result.state_changed,
            reconciliation_id=result.reconciliation_id,
        ),
        meta={"request_id": request.state.request_id},
    )
