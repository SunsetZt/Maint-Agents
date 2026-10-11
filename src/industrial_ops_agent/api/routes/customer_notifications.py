"""Provider-authenticated customer notification receipt endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel

from industrial_ops_agent.api.dependencies import get_tool_gateway
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.tools.gateway import (
    CustomerNotificationReceiptRejected,
    ToolGateway,
)

router = APIRouter(
    prefix="/customer-notification-receipts",
    tags=["customer-notification-receipts"],
)


class CustomerNotificationReceiptResponse(BaseModel):
    notification_delivery_id: str
    status: str
    state_changed: bool
    reconciliation_id: str | None


class CustomerNotificationReceiptEnvelope(BaseModel):
    data: CustomerNotificationReceiptResponse
    meta: dict[str, str]


@router.post("/{provider}", response_model=CustomerNotificationReceiptEnvelope)
async def receive_customer_notification_receipt(
    provider: str,
    request: Request,
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    signature: Annotated[str, Header(alias="X-Notification-Signature")],
) -> CustomerNotificationReceiptEnvelope:
    try:
        result = gateway.process_customer_notification_receipt(
            provider=provider,
            raw_body=await request.body(),
            signature=signature,
        )
    except CustomerNotificationReceiptRejected as exc:
        raise AppError(
            401,
            "notification_receipt_rejected",
            "authentication",
            "Notification receipt authentication failed",
        ) from exc
    return CustomerNotificationReceiptEnvelope(
        data=CustomerNotificationReceiptResponse(
            notification_delivery_id=result.delivery_id,
            status=result.status,
            state_changed=result.state_changed,
            reconciliation_id=result.reconciliation_id,
        ),
        meta={"request_id": request.state.request_id},
    )
