"""Stable M1 API error responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

if TYPE_CHECKING:
    from fastapi import FastAPI


class ApiErrorBody(BaseModel):
    category: str
    code: str
    message: str
    retryable: bool
    details: dict[str, Any] | None = None


class ApiErrorEnvelope(BaseModel):
    error: ApiErrorBody
    request_id: str | None = None
    trace_id: str | None = None


STANDARD_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ApiErrorEnvelope, "description": "Invalid request precondition"},
    401: {"model": ApiErrorEnvelope, "description": "Authentication required"},
    403: {"model": ApiErrorEnvelope, "description": "Action is not allowed"},
    404: {"model": ApiErrorEnvelope, "description": "Resource not found or not visible"},
    409: {"model": ApiErrorEnvelope, "description": "Idempotency or version conflict"},
    415: {"model": ApiErrorEnvelope, "description": "Media type mismatch"},
    422: {"model": ApiErrorEnvelope, "description": "Request validation failed"},
    503: {"model": ApiErrorEnvelope, "description": "Required dependency unavailable"},
}


@dataclass(frozen=True, slots=True)
class AppError(Exception):
    """A safe application error that can cross the HTTP boundary."""

    status_code: int
    code: str
    category: str
    message: str
    retryable: bool = False
    details: dict[str, Any] | None = None


def _correlation(request: Request) -> tuple[str, str]:
    return (
        getattr(request.state, "request_id", "unavailable"),
        getattr(request.state, "trace_id", "unavailable"),
    )


def error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    category: str,
    message: str,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Build the closed M1 error envelope."""

    request_id, trace_id = _correlation(request)
    error: dict[str, Any] = {
        "category": category,
        "code": code,
        "message": message,
        "retryable": retryable,
    }
    if details is not None:
        error["details"] = details
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error,
            "request_id": request_id,
            "trace_id": trace_id,
        },
    )


async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    return error_response(
        request,
        status_code=exc.status_code,
        code=exc.code,
        category=exc.category,
        message=exc.message,
        retryable=exc.retryable,
        details=exc.details,
    )


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    structlog.get_logger().info("request_validation_failed", error_count=len(exc.errors()))
    return error_response(
        request,
        status_code=422,
        code="request_validation_failed",
        category="validation",
        message="Request validation failed",
    )


async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    if exc.status_code == 404:
        return error_response(
            request,
            status_code=404,
            code="resource_not_found",
            category="not_found",
            message="Resource not found",
        )
    return error_response(
        request,
        status_code=exc.status_code,
        code="http_request_rejected",
        category="request",
        message="HTTP request rejected",
    )


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    structlog.get_logger().error("unexpected_request_failure", error_type=type(exc).__name__)
    return error_response(
        request,
        status_code=500,
        code="internal_error",
        category="internal",
        message="Internal server error",
        retryable=True,
    )


def register_error_handlers(app: FastAPI) -> None:
    """Install the stable error boundary on an application."""

    from industrial_ops_agent.maintenance_planning.review_isolation import ReviewIsolationConflict

    async def handle_review_conflict(
        request: Request, exc: ReviewIsolationConflict
    ) -> JSONResponse:
        return error_response(
            request,
            status_code=409,
            code=exc.reason,
            category="conflict",
            message="Maintenance review source access is unavailable for this subject.",
        )

    app.add_exception_handler(ReviewIsolationConflict, handle_review_conflict)  # type: ignore[arg-type]
    app.add_exception_handler(AppError, handle_app_error)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, handle_validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, handle_http_error)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, handle_unexpected_error)
