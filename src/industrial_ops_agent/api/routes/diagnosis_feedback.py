"""Version-bound diagnosis quality feedback API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self

from fastapi import APIRouter, Depends, Header, Request, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from industrial_ops_agent.api.dependencies import get_authorizer, get_database, get_identity
from industrial_ops_agent.api.errors import STANDARD_ERROR_RESPONSES, AppError
from industrial_ops_agent.application.diagnosis_feedback import (
    DiagnosisFeedbackInput,
    DiagnosisFeedbackNotVisible,
    DiagnosisFeedbackRejected,
    DiagnosisFeedbackService,
    DiagnosisFeedbackView,
)
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.persistence.database import Database

router = APIRouter(tags=["diagnosis-feedback"])

Verdict = Literal["HELPFUL", "PARTIALLY_HELPFUL", "NOT_HELPFUL", "UNSAFE"]
IssueCode = Literal[
    "INCORRECT_ROOT_CAUSE",
    "MISSING_EVIDENCE",
    "WRONG_CITATION",
    "OUTDATED_KNOWLEDGE",
    "TOOL_FACT_MISMATCH",
    "INCOMPLETE_NEXT_STEPS",
    "UNSAFE_RECOMMENDATION",
    "OTHER",
]


class DiagnosisFeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnosis_run_id: str = Field(min_length=1, max_length=128)
    verdict: Verdict
    issue_codes: list[IssueCode] = Field(default_factory=list, max_length=8)
    comment: str | None = Field(default=None, max_length=4_000)

    @model_validator(mode="after")
    def validate_basis(self) -> Self:
        if len(set(self.issue_codes)) != len(self.issue_codes):
            raise ValueError("diagnosis_feedback_issue_codes_duplicate")
        comment = self.comment.strip() if self.comment is not None else ""
        if self.verdict in {"NOT_HELPFUL", "UNSAFE"} and (
            not self.issue_codes or len(comment) < 8
        ):
            raise ValueError("diagnosis_feedback_negative_basis_required")
        if self.verdict == "UNSAFE" and "UNSAFE_RECOMMENDATION" not in self.issue_codes:
            raise ValueError("diagnosis_feedback_unsafe_issue_required")
        return self


class DiagnosisFeedbackResponse(BaseModel):
    feedback_id: str
    diagnosis_run_id: str
    incident_id: str
    asset_id: str
    agent_run_id: str
    diagnosis_status: str
    diagnosis_version: int
    report_digest: str
    manifest_digest: str
    model_release_id: str | None
    prompt_bundle_id: str | None
    index_release_id: str | None
    context_hash: str | None
    verdict: str
    issue_codes: list[str]
    comment: str | None
    content_digest: str
    submitted_by_subject_id: str
    submitted_by_role: str
    created_at: datetime


class DiagnosisFeedbackMeta(BaseModel):
    request_id: str
    created: bool | None = None
    can_submit: bool | None = None
    total: int | None = None


class DiagnosisFeedbackEnvelope(BaseModel):
    data: DiagnosisFeedbackResponse
    meta: DiagnosisFeedbackMeta


class DiagnosisFeedbackListEnvelope(BaseModel):
    data: list[DiagnosisFeedbackResponse]
    meta: DiagnosisFeedbackMeta


@router.post(
    "/feedback",
    response_model=DiagnosisFeedbackEnvelope,
    status_code=201,
    responses=STANDARD_ERROR_RESPONSES,
)
def create_diagnosis_feedback(
    body: DiagnosisFeedbackRequest,
    request: Request,
    response: Response,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[
        str, Header(alias="Idempotency-Key", min_length=1, max_length=255)
    ],
) -> DiagnosisFeedbackEnvelope:
    try:
        result = DiagnosisFeedbackService(database, authorizer).create(
            identity,
            DiagnosisFeedbackInput(
                diagnosis_run_id=body.diagnosis_run_id,
                verdict=body.verdict,
                issue_codes=tuple(body.issue_codes),
                comment=body.comment,
            ),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except DiagnosisFeedbackNotVisible as exc:
        raise _not_visible() from exc
    except DiagnosisFeedbackRejected as exc:
        raise _rejected(exc) from exc
    response.status_code = 201 if result.created else 200
    return DiagnosisFeedbackEnvelope(
        data=_response(result.feedback),
        meta=DiagnosisFeedbackMeta(
            request_id=request.state.request_id,
            created=result.created,
        ),
    )


@router.get(
    "/diagnosis-runs/{diagnosis_run_id}/feedback",
    response_model=DiagnosisFeedbackListEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
def list_diagnosis_feedback(
    diagnosis_run_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> DiagnosisFeedbackListEnvelope:
    try:
        result = DiagnosisFeedbackService(database, authorizer).list_for_run(
            identity,
            diagnosis_run_id,
            request_id=request.state.request_id,
        )
    except DiagnosisFeedbackNotVisible as exc:
        raise _not_visible() from exc
    return DiagnosisFeedbackListEnvelope(
        data=[_response(item) for item in result.feedback],
        meta=DiagnosisFeedbackMeta(
            request_id=request.state.request_id,
            can_submit=result.can_submit,
            total=len(result.feedback),
        ),
    )


def _response(value: DiagnosisFeedbackView) -> DiagnosisFeedbackResponse:
    return DiagnosisFeedbackResponse.model_validate(value, from_attributes=True)


def _not_visible() -> AppError:
    return AppError(
        status_code=404,
        code="resource_not_found_or_not_visible",
        category="not_found",
        message="Resource not found or not visible",
    )


def _rejected(exc: DiagnosisFeedbackRejected) -> AppError:
    conflicts = {
        "diagnosis_feedback_idempotency_conflict",
        "diagnosis_feedback_already_exists",
        "diagnosis_feedback_run_not_terminal",
        "diagnosis_feedback_report_not_available",
        "diagnosis_feedback_manifest_not_available",
    }
    return AppError(
        status_code=409 if exc.reason in conflicts else 422,
        code=exc.reason,
        category="conflict" if exc.reason in conflicts else "validation",
        message="Diagnosis quality feedback was rejected",
    )
