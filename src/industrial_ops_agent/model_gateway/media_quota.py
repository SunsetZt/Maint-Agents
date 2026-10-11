"""Shared quota admission for media requests within the caller's transaction."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from industrial_ops_agent.model_gateway.service import ModelGatewayError
from industrial_ops_agent.persistence.models import ModelGatewayQuotaRecord, ModelInferenceRecord


def require_media_quota(
    session: Session,
    *,
    tenant_id: str,
    model_alias: str,
    request_class: Literal["ASR", "TTS"],
    reservation: int,
    limit_reason: str,
    include_pending: bool,
    now: datetime,
) -> None:
    quota = session.scalar(
        select(ModelGatewayQuotaRecord)
        .where(
            ModelGatewayQuotaRecord.tenant_id == tenant_id,
            ModelGatewayQuotaRecord.model_alias == model_alias,
        )
        .with_for_update()
    )
    if quota is None or not quota.enabled:
        raise ModelGatewayError("model_quota_unavailable")
    if request_class not in quota.allowed_request_classes:
        raise ModelGatewayError("model_request_class_forbidden")
    if reservation > quota.max_output_tokens:
        raise ModelGatewayError(limit_reason)
    scope = (
        ModelInferenceRecord.tenant_id == tenant_id,
        ModelInferenceRecord.model_alias == model_alias,
    )
    minute_count = int(
        session.scalar(
            select(func.count(ModelInferenceRecord.inference_request_id)).where(
                *scope, ModelInferenceRecord.created_at >= now - timedelta(minutes=1)
            )
        )
        or 0
    )
    if minute_count >= quota.requests_per_minute:
        raise ModelGatewayError("model_request_rate_exceeded")
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    daily_usage = int(
        session.scalar(
            select(
                func.coalesce(
                    func.sum(
                        ModelInferenceRecord.usage_prompt_tokens
                        + ModelInferenceRecord.usage_completion_tokens
                    ),
                    0,
                )
            ).where(*scope, ModelInferenceRecord.created_at >= day_start)
        )
        or 0
    )
    pending_reservation = 0
    if include_pending:
        pending_reservation = int(
            session.scalar(
                select(func.coalesce(func.sum(ModelInferenceRecord.max_output_tokens), 0)).where(
                    *scope,
                    ModelInferenceRecord.created_at >= day_start,
                    ModelInferenceRecord.status == "PENDING",
                )
            )
            or 0
        )
    if daily_usage + pending_reservation + reservation > quota.tokens_per_day:
        raise ModelGatewayError("model_daily_token_quota_exceeded")
