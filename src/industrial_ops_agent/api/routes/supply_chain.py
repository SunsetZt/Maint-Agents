"""Read-only API for cryptographically verified software/model supply-chain evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from industrial_ops_agent.api.dependencies import get_authorizer, get_database, get_identity
from industrial_ops_agent.api.errors import AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import SupplyChainEvidenceRecord
from industrial_ops_agent.supply_chain.service import SupplyChainEvidenceService

router = APIRouter(tags=["supply-chain"])


class SupplyChainEvidenceResponse(BaseModel):
    evidence_id: str
    schema_version: str
    image_repository: str
    image_digest: str
    model_artifact_digest: str
    source_repository: str
    source_revision: str
    sbom_digest: str
    sbom_format: str
    vulnerability_report_digest: str
    vulnerability_scan_status: str
    maximum_vulnerability_severity: str
    vulnerability_scanner: str
    license_report_digest: str
    license_status: str
    provenance_digest: str
    image_signature_digest: str
    model_signature_digest: str
    certificate_identity: str
    certificate_oidc_issuer: str
    verifier_version: str
    verification_status: str
    verification_hash: str
    verified_by_subject_id: str
    verified_at: datetime


class SupplyChainEvidenceListEnvelope(BaseModel):
    data: list[SupplyChainEvidenceResponse]
    meta: dict[str, str | int]


@router.get("/supply-chain/evidence", response_model=SupplyChainEvidenceListEnvelope)
def list_supply_chain_evidence(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    release_ready_only: Annotated[bool, Query()] = False,
    component: Annotated[Literal["VLM", "ASR"] | None, Query()] = None,
    model_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
) -> SupplyChainEvidenceListEnvelope:
    if (component is None) != (model_id is None):
        raise AppError(
            status_code=422,
            code="component_model_filter_required",
            category="validation",
            message="component and model_id must be supplied together",
        )
    reason = None
    try:
        service = SupplyChainEvidenceService(database, authorizer)
        if component is not None and model_id is not None:
            records, reason = service.list_for_component(
                identity,
                component=component,
                model_id=model_id,
                request_id=request.state.request_id,
            )
        else:
            records = service.list(
                identity,
                request_id=request.state.request_id,
                release_ready_only=release_ready_only,
            )
    except AuthorizationDenied as exc:
        raise AppError(
            status_code=403,
            code="supply_chain_evidence_forbidden",
            category="authorization",
            message="Supply-chain evidence access was denied",
        ) from exc
    return SupplyChainEvidenceListEnvelope(
        data=[_response(record) for record in records],
        meta={
            "request_id": request.state.request_id,
            "total": len(records),
            **({"binding_status": "BLOCKED" if reason else "READY"} if component else {}),
            **({"reason": reason} if reason else {}),
        },
    )


def _response(record: SupplyChainEvidenceRecord) -> SupplyChainEvidenceResponse:
    return SupplyChainEvidenceResponse(
        **{field: getattr(record, field) for field in SupplyChainEvidenceResponse.model_fields}
    )
