"""Provider-authenticated ERP purchase requisition receipt endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel

from industrial_ops_agent.api.dependencies import get_tool_gateway
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.tools.gateway import (
    ProcurementFulfillmentReceiptRejected,
    PurchaseRequestReceiptRejected,
    ToolGateway,
)

router = APIRouter(
    prefix="/purchase-requisition-receipts",
    tags=["purchase-requisition-receipts"],
)
fulfillment_router = APIRouter(
    prefix="/purchase-fulfillment-receipts",
    tags=["purchase-fulfillment-receipts"],
)


class PurchaseRequestReceiptResponse(BaseModel):
    purchase_request_id: str
    status: str
    state_changed: bool
    reconciliation_id: str | None


class PurchaseRequestReceiptEnvelope(BaseModel):
    data: PurchaseRequestReceiptResponse
    meta: dict[str, str]


class ProcurementFulfillmentReceiptResponse(BaseModel):
    purchase_request_id: str
    fulfillment_id: str
    status: str
    received_quantity: int
    state_changed: bool


class ProcurementFulfillmentReceiptEnvelope(BaseModel):
    data: ProcurementFulfillmentReceiptResponse
    meta: dict[str, str]


@router.post("/{provider}", response_model=PurchaseRequestReceiptEnvelope)
async def receive_purchase_request_receipt(
    provider: str,
    request: Request,
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    signature: Annotated[str, Header(alias="X-Procurement-Signature")],
) -> PurchaseRequestReceiptEnvelope:
    try:
        result = gateway.process_purchase_request_receipt(
            provider=provider,
            raw_body=await request.body(),
            signature=signature,
        )
    except PurchaseRequestReceiptRejected as exc:
        raise AppError(
            401,
            "procurement_receipt_rejected",
            "authentication",
            "Procurement receipt authentication failed",
        ) from exc
    return PurchaseRequestReceiptEnvelope(
        data=PurchaseRequestReceiptResponse(
            purchase_request_id=result.purchase_request_id,
            status=result.status,
            state_changed=result.state_changed,
            reconciliation_id=result.reconciliation_id,
        ),
        meta={"request_id": request.state.request_id},
    )


@fulfillment_router.post(
    "/{provider}",
    response_model=ProcurementFulfillmentReceiptEnvelope,
)
async def receive_procurement_fulfillment_receipt(
    provider: str,
    request: Request,
    gateway: Annotated[ToolGateway, Depends(get_tool_gateway)],
    signature: Annotated[str, Header(alias="X-Procurement-Signature")],
) -> ProcurementFulfillmentReceiptEnvelope:
    try:
        result = gateway.process_procurement_fulfillment_receipt(
            provider=provider,
            raw_body=await request.body(),
            signature=signature,
        )
    except ProcurementFulfillmentReceiptRejected as exc:
        raise AppError(
            401,
            "procurement_fulfillment_receipt_rejected",
            "authentication",
            "Procurement fulfillment receipt authentication failed",
        ) from exc
    return ProcurementFulfillmentReceiptEnvelope(
        data=ProcurementFulfillmentReceiptResponse(
            purchase_request_id=result.purchase_request_id,
            fulfillment_id=result.fulfillment_id,
            status=result.status,
            received_quantity=result.received_quantity,
            state_changed=result.state_changed,
        ),
        meta={"request_id": request.state.request_id},
    )
