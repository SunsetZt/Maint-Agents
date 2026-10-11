"""Governed enterprise knowledge ingestion and index publication API."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Body, Depends, Header, Query, Request
from pydantic import BaseModel, Field

from industrial_ops_agent.api.dependencies import (
    get_authorizer,
    get_database,
    get_identity,
    get_knowledge_deletion_dispatcher,
    get_knowledge_index_evaluation_dispatcher,
    get_knowledge_ingestion_dispatcher,
    get_object_store,
)
from industrial_ops_agent.api.errors import STANDARD_ERROR_RESPONSES, AppError
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.knowledge.deletion import (
    KnowledgeDeletionActivityInput,
    KnowledgeDeletionDispatcher,
    KnowledgeDeletionDispatchUnavailable,
    KnowledgeDeletionError,
    KnowledgeDeletionService,
    KnowledgeDeletionView,
)
from industrial_ops_agent.knowledge.files import (
    MAX_KNOWLEDGE_FILE_BYTES,
    KnowledgeFileError,
    KnowledgeFileIngestionService,
    KnowledgeFilePlan,
    KnowledgeFileView,
    KnowledgeIngestionActivityInput,
    KnowledgeIngestionDispatcher,
    KnowledgeIngestionDispatchUnavailable,
)
from industrial_ops_agent.knowledge.index_evaluation import (
    KnowledgeIndexEvaluationActivityInput,
    KnowledgeIndexEvaluationDispatcher,
    KnowledgeIndexEvaluationDispatchUnavailable,
    KnowledgeIndexEvaluationError,
    KnowledgeIndexEvaluationService,
    KnowledgeIndexEvaluationView,
)
from industrial_ops_agent.knowledge.ingestion import (
    KnowledgeIngestionError,
    KnowledgeIngestionService,
    KnowledgeReleaseView,
    KnowledgeVersionView,
)
from industrial_ops_agent.knowledge.service import (
    InvalidIndexRelease,
    KnowledgeIndexActivationView,
    KnowledgePublicationService,
)
from industrial_ops_agent.media.service import TenantObjectStore
from industrial_ops_agent.persistence.database import Database

router = APIRouter(tags=["knowledge-governance"])


class KnowledgeDocumentBody(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    source_uri: str = Field(min_length=1, max_length=1024)
    content: str = Field(min_length=20, max_length=500_000)
    classification: Literal["internal", "restricted"] = "internal"
    acl_subject_ids: list[str] = Field(default_factory=list, max_length=100)
    acl_roles: list[str] = Field(default_factory=list, max_length=100)
    device_families: list[str] = Field(default_factory=list, max_length=100)
    device_models: list[str] = Field(default_factory=list, max_length=100)
    valid_from: datetime
    valid_to: datetime | None = None


class KnowledgeSourceFileBody(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    source_filename: str = Field(min_length=1, max_length=255)
    declared_mime: Literal["text/plain", "text/markdown", "application/pdf"]
    classification: Literal["internal", "restricted"] = "internal"
    acl_subject_ids: list[str] = Field(default_factory=list, max_length=100)
    acl_roles: list[str] = Field(default_factory=list, max_length=100)
    device_families: list[str] = Field(default_factory=list, max_length=100)
    device_models: list[str] = Field(default_factory=list, max_length=100)
    valid_from: datetime
    valid_to: datetime | None = None


class KnowledgeDocumentVersionBody(BaseModel):
    content: str = Field(min_length=20, max_length=500_000)
    acl_subject_ids: list[str] = Field(default_factory=list, max_length=100)
    acl_roles: list[str] = Field(default_factory=list, max_length=100)
    device_families: list[str] = Field(default_factory=list, max_length=100)
    device_models: list[str] = Field(default_factory=list, max_length=100)
    valid_from: datetime
    valid_to: datetime | None = None


class KnowledgeReleaseBody(BaseModel):
    name: str = Field(min_length=3, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]+$")
    document_version_ids: list[str] = Field(min_length=1, max_length=1000)


class KnowledgeRollbackBody(BaseModel):
    reason: str = Field(min_length=10, max_length=1000)
    expected_active_release_id: str = Field(min_length=1, max_length=128)


class KnowledgeDeletionBody(BaseModel):
    trigger: Literal["LEGAL_DELETION", "ACCESS_REVOKED", "RETENTION_EXPIRED"]
    reason: str = Field(min_length=8, max_length=1024)


class KnowledgeExpirySweepBody(BaseModel):
    reason: str = Field(min_length=8, max_length=1024)
    limit: int = Field(default=100, ge=1, le=500)


class KnowledgeVersionResponse(BaseModel):
    document_id: str
    document_version_id: str
    version: int
    title: str
    source_uri: str
    classification: str
    source_checksum: str
    content_checksum: str
    parser_version: str
    extraction_metadata: dict[str, Any] | None
    status: str
    state_version: int
    acl_subject_ids: list[str]
    acl_roles: list[str]
    device_families: list[str]
    device_models: list[str]
    valid_from: datetime
    valid_to: datetime | None
    created_by_subject_id: str | None
    reviewed_by_subject_id: str | None
    reviewed_at: datetime | None
    created_at: datetime
    updated_at: datetime
    legal_actions: list[str]


class KnowledgeReleaseResponse(BaseModel):
    release_id: str
    name: str
    version: int
    status: str
    is_active: bool
    activation_version: int
    rollback_eligible: bool
    content_checksum: str
    chunk_count: int
    created_by_subject_id: str | None
    evaluation_id: str | None
    evaluation_status: str | None
    evaluation_metrics: dict[str, Any]
    evaluation_gate_results: dict[str, bool]
    evaluation_failure_codes: list[str]
    evaluated_at: datetime | None
    published_by: str | None
    published_at: datetime | None
    created_at: datetime
    updated_at: datetime
    legal_actions: list[str]


class KnowledgeVersionEnvelope(BaseModel):
    data: KnowledgeVersionResponse
    meta: dict[str, str]


class KnowledgeReleaseEnvelope(BaseModel):
    data: KnowledgeReleaseResponse
    meta: dict[str, str]


class KnowledgeVersionListEnvelope(BaseModel):
    data: list[KnowledgeVersionResponse]
    meta: dict[str, str | int]
    legal_actions: list[str]


class KnowledgeReleaseListEnvelope(BaseModel):
    data: list[KnowledgeReleaseResponse]
    meta: dict[str, str | int]
    legal_actions: list[str]


class KnowledgeIndexActivationResponse(BaseModel):
    activation_id: str
    name: str
    action: Literal["PUBLISH", "ROLLBACK"]
    from_release_id: str | None
    to_release_id: str
    from_content_checksum: str | None
    to_content_checksum: str
    actor_subject_id: str
    reason: str
    retired_search_profile_ids: list[str]
    revoked_graph_release_ids: list[str]
    occurred_at: datetime


class KnowledgeIndexActivationListEnvelope(BaseModel):
    data: list[KnowledgeIndexActivationResponse]
    meta: dict[str, str | int]


class KnowledgeIndexEvaluationResponse(BaseModel):
    evaluation_id: str
    release_id: str
    workflow_id: str
    status: str
    policy_version: str
    evaluator_version: str
    metrics: dict[str, Any]
    gate_results: dict[str, bool]
    failure_codes: list[str]
    requested_by_subject_id: str
    started_at: datetime | None
    completed_at: datetime | None
    attempt_count: int
    version: int
    created_at: datetime
    updated_at: datetime


class KnowledgeIndexEvaluationEnvelope(BaseModel):
    data: KnowledgeIndexEvaluationResponse
    meta: dict[str, str]


class KnowledgeFileAttemptResponse(BaseModel):
    attempt_id: str
    attempt_number: int
    workflow_id: str
    status: str
    failure_reason: str | None
    parser_version: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class KnowledgeFileResponse(BaseModel):
    ingestion_id: str
    workflow_id: str
    title: str
    source_filename: str
    declared_mime: str
    detected_mime: str | None
    classification: str
    status: str
    failure_reason: str | None
    source_checksum: str | None
    size_bytes: int | None
    parser_version: str | None
    document_id: str | None
    document_version_id: str | None
    attempt_count: int
    attempts: list[KnowledgeFileAttemptResponse]
    version: int
    created_at: datetime
    updated_at: datetime
    legal_actions: list[str]


class KnowledgeFileEnvelope(BaseModel):
    data: KnowledgeFileResponse
    meta: dict[str, str]


class KnowledgeFileListEnvelope(BaseModel):
    data: list[KnowledgeFileResponse]
    meta: dict[str, str | int]
    legal_actions: list[str]


class KnowledgeDeletionResponse(BaseModel):
    deletion_id: str
    workflow_id: str
    document_id: str
    document_version_id: str
    trigger: str
    reason: str
    status: str
    target_source_checksum: str
    target_content_checksum: str
    scope: dict[str, Any]
    impact: dict[str, Any]
    verification: dict[str, Any]
    failure_reason: str | None
    requested_by_subject_id: str
    requested_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    attempt_count: int
    version: int
    created_at: datetime
    updated_at: datetime
    legal_actions: list[str]


class KnowledgeDeletionEnvelope(BaseModel):
    data: KnowledgeDeletionResponse
    meta: dict[str, str]


class KnowledgeDeletionListEnvelope(BaseModel):
    data: list[KnowledgeDeletionResponse]
    meta: dict[str, str | int]
    legal_actions: list[str]


@router.post(
    "/knowledge/source-files",
    response_model=KnowledgeFileEnvelope,
    status_code=201,
    responses=STANDARD_ERROR_RESPONSES,
)
async def create_knowledge_source_file(
    body: KnowledgeSourceFileBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    object_store: Annotated[TenantObjectStore, Depends(get_object_store)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
) -> KnowledgeFileEnvelope:
    try:
        result = KnowledgeFileIngestionService(database, authorizer, object_store).create_upload(
            identity,
            KnowledgeFilePlan(
                title=body.title,
                source_filename=body.source_filename,
                declared_mime=body.declared_mime,
                classification=body.classification,
                acl_subject_ids=tuple(body.acl_subject_ids),
                acl_roles=tuple(body.acl_roles),
                device_families=tuple(body.device_families),
                device_models=tuple(body.device_models),
                valid_from=body.valid_from,
                valid_to=body.valid_to,
            ),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeFileError) as exc:
        raise _translate(exc) from exc
    return _file_envelope(result, request.state.request_id)


@router.get(
    "/knowledge/source-files",
    response_model=KnowledgeFileListEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
async def list_knowledge_source_files(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    object_store: Annotated[TenantObjectStore, Depends(get_object_store)],
    status: Annotated[
        Literal[
            "AWAITING_UPLOAD",
            "QUEUED",
            "RUNNING",
            "DRAFT_READY",
            "REJECTED",
            "FAILED",
            "PURGED",
        ]
        | None,
        Query(),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> KnowledgeFileListEnvelope:
    try:
        items, total = KnowledgeFileIngestionService(
            database,
            authorizer,
            object_store,
        ).list(
            identity,
            status=status,
            limit=limit,
            offset=offset,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeFileError) as exc:
        raise _translate(exc) from exc
    return KnowledgeFileListEnvelope(
        data=[_file_response(item) for item in items],
        meta={"request_id": request.state.request_id, "total": total},
        legal_actions=["CREATE_KNOWLEDGE_SOURCE_FILE"],
    )


@router.put(
    "/knowledge/source-files/{ingestion_id}/content",
    response_model=KnowledgeFileEnvelope,
    status_code=202,
    responses=STANDARD_ERROR_RESPONSES,
)
async def upload_knowledge_source_content(
    ingestion_id: str,
    request: Request,
    content: Annotated[
        bytes,
        Body(
            media_type="application/octet-stream",
            min_length=1,
            max_length=MAX_KNOWLEDGE_FILE_BYTES,
        ),
    ],
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    object_store: Annotated[TenantObjectStore, Depends(get_object_store)],
    dispatcher: Annotated[
        KnowledgeIngestionDispatcher,
        Depends(get_knowledge_ingestion_dispatcher),
    ],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> KnowledgeFileEnvelope:
    try:
        result = await KnowledgeFileIngestionService(
            database,
            authorizer,
            object_store,
        ).upload_content(
            identity,
            ingestion_id,
            content=content,
            expected_version=_state_version(if_match),
            request_id=request.state.request_id,
        )
        if result.status == "QUEUED":
            await dispatcher.dispatch(
                KnowledgeIngestionActivityInput(
                    ingestion_id=result.ingestion_id,
                    identity=identity,
                    request_id=request.state.request_id,
                    workflow_id=result.workflow_id,
                )
            )
    except (
        AuthorizationDenied,
        KnowledgeFileError,
        KnowledgeIngestionDispatchUnavailable,
    ) as exc:
        raise _translate(exc) from exc
    return _file_envelope(result, request.state.request_id)


@router.get(
    "/knowledge/source-files/{ingestion_id}",
    response_model=KnowledgeFileEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
async def get_knowledge_source_file(
    ingestion_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    object_store: Annotated[TenantObjectStore, Depends(get_object_store)],
) -> KnowledgeFileEnvelope:
    try:
        result = KnowledgeFileIngestionService(database, authorizer, object_store).get(
            identity,
            ingestion_id,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeFileError) as exc:
        raise _translate(exc) from exc
    return _file_envelope(result, request.state.request_id)


@router.post(
    "/knowledge/source-files/{ingestion_id}/reprocess",
    response_model=KnowledgeFileEnvelope,
    status_code=202,
    responses=STANDARD_ERROR_RESPONSES,
)
async def reprocess_knowledge_source_file(
    ingestion_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    object_store: Annotated[TenantObjectStore, Depends(get_object_store)],
    dispatcher: Annotated[
        KnowledgeIngestionDispatcher,
        Depends(get_knowledge_ingestion_dispatcher),
    ],
    if_match: Annotated[str, Header(alias="If-Match")],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=1, max_length=200),
    ],
) -> KnowledgeFileEnvelope:
    try:
        result = KnowledgeFileIngestionService(
            database,
            authorizer,
            object_store,
        ).reprocess(
            identity,
            ingestion_id,
            expected_version=_state_version(if_match),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
        if result.status == "QUEUED":
            await dispatcher.dispatch(
                KnowledgeIngestionActivityInput(
                    ingestion_id=result.ingestion_id,
                    identity=identity,
                    request_id=request.state.request_id,
                    workflow_id=result.workflow_id,
                )
            )
    except (
        AuthorizationDenied,
        KnowledgeFileError,
        KnowledgeIngestionDispatchUnavailable,
    ) as exc:
        raise _translate(exc) from exc
    return _file_envelope(result, request.state.request_id)


@router.post("/knowledge/documents", response_model=KnowledgeVersionEnvelope, status_code=201)
async def create_knowledge_document(
    body: KnowledgeDocumentBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
) -> KnowledgeVersionEnvelope:
    try:
        version = KnowledgeIngestionService(database, authorizer).create_document(
            identity,
            title=body.title,
            source_uri=body.source_uri,
            content=body.content,
            classification=body.classification,
            acl_subject_ids=tuple(body.acl_subject_ids),
            acl_roles=tuple(body.acl_roles),
            device_families=tuple(body.device_families),
            device_models=tuple(body.device_models),
            valid_from=body.valid_from,
            valid_to=body.valid_to,
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeIngestionError) as exc:
        raise _translate(exc) from exc
    return _version_envelope(version, request.state.request_id, identity)


@router.post(
    "/knowledge/documents/{document_id}/versions",
    response_model=KnowledgeVersionEnvelope,
    status_code=201,
)
async def create_knowledge_document_version(
    document_id: str,
    body: KnowledgeDocumentVersionBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
) -> KnowledgeVersionEnvelope:
    try:
        version = KnowledgeIngestionService(database, authorizer).create_version(
            identity,
            document_id,
            content=body.content,
            acl_subject_ids=tuple(body.acl_subject_ids),
            acl_roles=tuple(body.acl_roles),
            device_families=tuple(body.device_families),
            device_models=tuple(body.device_models),
            valid_from=body.valid_from,
            valid_to=body.valid_to,
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeIngestionError) as exc:
        raise _translate(exc) from exc
    return _version_envelope(version, request.state.request_id, identity)


@router.get(
    "/knowledge/document-versions",
    response_model=KnowledgeVersionListEnvelope,
)
async def list_knowledge_document_versions(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    status: Annotated[
        Literal[
            "DRAFT",
            "REVIEWED",
            "PUBLISHED",
            "DELETION_PENDING",
            "REVOCATION_PENDING",
            "EXPIRY_PENDING",
            "DELETED",
            "REVOKED",
            "EXPIRED",
        ]
        | None,
        Query(),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> KnowledgeVersionListEnvelope:
    try:
        versions, total = KnowledgeIngestionService(database, authorizer).list_versions(
            identity,
            status=status,
            limit=limit,
            offset=offset,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeIngestionError) as exc:
        raise _translate(exc) from exc
    return KnowledgeVersionListEnvelope(
        data=[_version_response(item, identity) for item in versions],
        meta={
            "request_id": request.state.request_id,
            "total": total,
            "limit": limit,
            "offset": offset,
        },
        legal_actions=["CREATE_KNOWLEDGE_DOCUMENT", "CREATE_KNOWLEDGE_INDEX_RELEASE"],
    )


@router.get(
    "/knowledge/document-versions/{document_version_id}",
    response_model=KnowledgeVersionEnvelope,
)
async def get_knowledge_document_version(
    document_version_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> KnowledgeVersionEnvelope:
    try:
        version = KnowledgeIngestionService(database, authorizer).get_version(
            identity,
            document_version_id,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeIngestionError) as exc:
        raise _translate(exc) from exc
    return _version_envelope(version, request.state.request_id, identity)


@router.post(
    "/knowledge/document-versions/{document_version_id}/reviews",
    response_model=KnowledgeVersionEnvelope,
)
async def review_knowledge_document_version(
    document_version_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> KnowledgeVersionEnvelope:
    try:
        version = KnowledgeIngestionService(database, authorizer).review_version(
            identity,
            document_version_id,
            expected_state_version=_state_version(if_match),
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeIngestionError) as exc:
        raise _translate(exc) from exc
    return _version_envelope(version, request.state.request_id, identity)


@router.post(
    "/knowledge/document-versions/{document_version_id}/deletions",
    response_model=KnowledgeDeletionEnvelope,
    status_code=202,
    responses=STANDARD_ERROR_RESPONSES,
)
async def request_knowledge_deletion(
    document_version_id: str,
    body: KnowledgeDeletionBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dispatcher: Annotated[
        KnowledgeDeletionDispatcher,
        Depends(get_knowledge_deletion_dispatcher),
    ],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=1, max_length=200),
    ],
) -> KnowledgeDeletionEnvelope:
    try:
        deletion = KnowledgeDeletionService(database, authorizer).request(
            identity,
            document_version_id,
            trigger=body.trigger,
            reason=body.reason,
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
        await dispatcher.dispatch(_deletion_command(deletion, identity.tenant_id))
    except (
        AuthorizationDenied,
        KnowledgeDeletionError,
        KnowledgeDeletionDispatchUnavailable,
    ) as exc:
        raise _translate(exc) from exc
    return _deletion_envelope(deletion, request.state.request_id)


@router.post(
    "/knowledge/deletions/expiry-sweep",
    response_model=KnowledgeDeletionListEnvelope,
    status_code=202,
    responses=STANDARD_ERROR_RESPONSES,
)
async def sweep_expired_knowledge(
    body: KnowledgeExpirySweepBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dispatcher: Annotated[
        KnowledgeDeletionDispatcher,
        Depends(get_knowledge_deletion_dispatcher),
    ],
) -> KnowledgeDeletionListEnvelope:
    try:
        deletions = KnowledgeDeletionService(database, authorizer).request_expired(
            identity,
            reason=body.reason,
            limit=body.limit,
            request_id=request.state.request_id,
        )
        for deletion in deletions:
            await dispatcher.dispatch(_deletion_command(deletion, identity.tenant_id))
    except (
        AuthorizationDenied,
        KnowledgeDeletionError,
        KnowledgeDeletionDispatchUnavailable,
    ) as exc:
        raise _translate(exc) from exc
    return KnowledgeDeletionListEnvelope(
        data=[_deletion_response(item) for item in deletions],
        meta={"request_id": request.state.request_id, "total": len(deletions)},
        legal_actions=["RUN_KNOWLEDGE_EXPIRY_SWEEP"],
    )


@router.get(
    "/knowledge/deletions",
    response_model=KnowledgeDeletionListEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
async def list_knowledge_deletions(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    status: Annotated[
        Literal["QUEUED", "RUNNING", "COMPLETED", "PARTIAL", "FAILED"] | None,
        Query(),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> KnowledgeDeletionListEnvelope:
    try:
        items, total = KnowledgeDeletionService(database, authorizer).list(
            identity,
            status=status,
            limit=limit,
            offset=offset,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeDeletionError) as exc:
        raise _translate(exc) from exc
    return KnowledgeDeletionListEnvelope(
        data=[_deletion_response(item) for item in items],
        meta={
            "request_id": request.state.request_id,
            "total": total,
            "limit": limit,
            "offset": offset,
        },
        legal_actions=["RUN_KNOWLEDGE_EXPIRY_SWEEP"],
    )


@router.get(
    "/knowledge/deletions/{deletion_id}",
    response_model=KnowledgeDeletionEnvelope,
    responses=STANDARD_ERROR_RESPONSES,
)
async def get_knowledge_deletion(
    deletion_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> KnowledgeDeletionEnvelope:
    try:
        deletion = KnowledgeDeletionService(database, authorizer).get(
            identity,
            deletion_id,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeDeletionError) as exc:
        raise _translate(exc) from exc
    return _deletion_envelope(deletion, request.state.request_id)


@router.post(
    "/knowledge/deletions/{deletion_id}/retry",
    response_model=KnowledgeDeletionEnvelope,
    status_code=202,
    responses=STANDARD_ERROR_RESPONSES,
)
async def retry_knowledge_deletion(
    deletion_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dispatcher: Annotated[
        KnowledgeDeletionDispatcher,
        Depends(get_knowledge_deletion_dispatcher),
    ],
    if_match: Annotated[str, Header(alias="If-Match")],
) -> KnowledgeDeletionEnvelope:
    try:
        deletion = KnowledgeDeletionService(database, authorizer).retry(
            identity,
            deletion_id,
            expected_version=_state_version(if_match),
            request_id=request.state.request_id,
        )
        await dispatcher.dispatch(_deletion_command(deletion, identity.tenant_id))
    except (
        AuthorizationDenied,
        KnowledgeDeletionError,
        KnowledgeDeletionDispatchUnavailable,
    ) as exc:
        raise _translate(exc) from exc
    return _deletion_envelope(deletion, request.state.request_id)


@router.post("/knowledge/index-releases", response_model=KnowledgeReleaseEnvelope, status_code=201)
async def build_knowledge_index_release(
    body: KnowledgeReleaseBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
) -> KnowledgeReleaseEnvelope:
    try:
        release = KnowledgeIngestionService(database, authorizer).build_release(
            identity,
            name=body.name,
            document_version_ids=tuple(body.document_version_ids),
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeIngestionError) as exc:
        raise _translate(exc) from exc
    return _release_envelope(release, request.state.request_id)


@router.get(
    "/knowledge/index-releases",
    response_model=KnowledgeReleaseListEnvelope,
)
async def list_knowledge_index_releases(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    status: Annotated[
        Literal[
            "EVALUATION_PENDING",
            "EVALUATING",
            "CANDIDATE",
            "REJECTED",
            "PUBLISHED",
            "REVOKED",
        ]
        | None,
        Query(),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> KnowledgeReleaseListEnvelope:
    try:
        releases, total = KnowledgeIngestionService(database, authorizer).list_releases(
            identity,
            status=status,
            limit=limit,
            offset=offset,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeIngestionError) as exc:
        raise _translate(exc) from exc
    return KnowledgeReleaseListEnvelope(
        data=[_release_response(item) for item in releases],
        meta={
            "request_id": request.state.request_id,
            "total": total,
            "limit": limit,
            "offset": offset,
        },
        legal_actions=["CREATE_KNOWLEDGE_INDEX_RELEASE"],
    )


@router.get(
    "/knowledge/index-releases/{release_id}",
    response_model=KnowledgeReleaseEnvelope,
)
async def get_knowledge_index_release(
    release_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> KnowledgeReleaseEnvelope:
    try:
        release = KnowledgeIngestionService(database, authorizer).get_release(
            identity,
            release_id,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeIngestionError) as exc:
        raise _translate(exc) from exc
    return _release_envelope(release, request.state.request_id)


@router.post(
    "/knowledge/index-releases/{release_id}/evaluations",
    response_model=KnowledgeIndexEvaluationEnvelope,
    status_code=202,
    responses=STANDARD_ERROR_RESPONSES,
)
async def evaluate_knowledge_index_release(
    release_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    dispatcher: Annotated[
        KnowledgeIndexEvaluationDispatcher,
        Depends(get_knowledge_index_evaluation_dispatcher),
    ],
    idempotency_key: Annotated[
        str, Header(alias="Idempotency-Key", min_length=1, max_length=200)
    ],
) -> KnowledgeIndexEvaluationEnvelope:
    try:
        evaluation = KnowledgeIndexEvaluationService(database, authorizer).start(
            identity,
            release_id,
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
        await dispatcher.dispatch(
            KnowledgeIndexEvaluationActivityInput(
                evaluation_id=evaluation.evaluation_id,
                tenant_id=identity.tenant_id,
                workflow_id=evaluation.workflow_id,
            )
        )
    except (
        AuthorizationDenied,
        KnowledgeIndexEvaluationError,
        KnowledgeIndexEvaluationDispatchUnavailable,
    ) as exc:
        raise _translate(exc) from exc
    return _evaluation_envelope(evaluation, request.state.request_id)


@router.post(
    "/knowledge/index-releases/{release_id}/promote",
    response_model=KnowledgeReleaseEnvelope,
)
async def promote_knowledge_index_release(
    release_id: str,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
) -> KnowledgeReleaseEnvelope:
    try:
        KnowledgePublicationService(database, authorizer).publish(
            identity,
            release_id,
            request_id=request.state.request_id,
        )
        release = KnowledgeIngestionService(database, authorizer).get_release(
            identity,
            release_id,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeIngestionError, InvalidIndexRelease) as exc:
        raise _translate(exc) from exc
    return _release_envelope(release, request.state.request_id)


@router.post(
    "/knowledge/index-releases/{release_id}/rollback",
    response_model=KnowledgeReleaseEnvelope,
)
async def rollback_knowledge_index_release(
    release_id: str,
    body: KnowledgeRollbackBody,
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    if_match: Annotated[str, Header(alias="If-Match")],
    idempotency_key: Annotated[
        str, Header(alias="Idempotency-Key", min_length=8, max_length=200)
    ],
) -> KnowledgeReleaseEnvelope:
    try:
        KnowledgePublicationService(database, authorizer).rollback(
            identity,
            release_id,
            expected_activation_version=_state_version(if_match),
            expected_active_release_id=body.expected_active_release_id,
            reason=body.reason,
            idempotency_key=idempotency_key,
            request_id=request.state.request_id,
        )
        release = KnowledgeIngestionService(database, authorizer).get_release(
            identity,
            release_id,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, KnowledgeIngestionError, InvalidIndexRelease) as exc:
        raise _translate(exc) from exc
    return _release_envelope(release, request.state.request_id)


@router.get(
    "/knowledge/index-activations",
    response_model=KnowledgeIndexActivationListEnvelope,
)
async def list_knowledge_index_activations(
    request: Request,
    identity: Annotated[IdentityContext, Depends(get_identity)],
    database: Annotated[Database, Depends(get_database)],
    authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    name: Annotated[str | None, Query(min_length=3, max_length=128)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> KnowledgeIndexActivationListEnvelope:
    try:
        items = KnowledgePublicationService(database, authorizer).list_activations(
            identity,
            name=name,
            limit=limit,
            request_id=request.state.request_id,
        )
    except (AuthorizationDenied, InvalidIndexRelease) as exc:
        raise _translate(exc) from exc
    return KnowledgeIndexActivationListEnvelope(
        data=[_activation_response(item) for item in items],
        meta={"request_id": request.state.request_id, "total": len(items)},
    )


def _version_envelope(
    value: KnowledgeVersionView,
    request_id: str,
    identity: IdentityContext,
) -> KnowledgeVersionEnvelope:
    return KnowledgeVersionEnvelope(
        data=_version_response(value, identity),
        meta={"request_id": request_id},
    )


def _release_envelope(value: KnowledgeReleaseView, request_id: str) -> KnowledgeReleaseEnvelope:
    return KnowledgeReleaseEnvelope(
        data=_release_response(value),
        meta={"request_id": request_id},
    )


def _evaluation_envelope(
    value: KnowledgeIndexEvaluationView,
    request_id: str,
) -> KnowledgeIndexEvaluationEnvelope:
    return KnowledgeIndexEvaluationEnvelope(
        data=KnowledgeIndexEvaluationResponse(
            **{
                **asdict(value),
                "failure_codes": list(value.failure_codes),
            }
        ),
        meta={"request_id": request_id},
    )


def _deletion_envelope(
    value: KnowledgeDeletionView,
    request_id: str,
) -> KnowledgeDeletionEnvelope:
    return KnowledgeDeletionEnvelope(
        data=_deletion_response(value),
        meta={"request_id": request_id},
    )


def _deletion_response(value: KnowledgeDeletionView) -> KnowledgeDeletionResponse:
    return KnowledgeDeletionResponse(
        **asdict(value),
        legal_actions=["RETRY_KNOWLEDGE_DELETION"]
        if value.status in {"PARTIAL", "FAILED"}
        else [],
    )


def _deletion_command(
    value: KnowledgeDeletionView,
    tenant_id: str,
) -> KnowledgeDeletionActivityInput:
    return KnowledgeDeletionActivityInput(
        deletion_id=value.deletion_id,
        tenant_id=tenant_id,
        workflow_id=value.workflow_id,
    )


def _version_response(
    value: KnowledgeVersionView,
    identity: IdentityContext,
) -> KnowledgeVersionResponse:
    legal_actions = ["CREATE_DOCUMENT_VERSION"]
    if value.status == "DRAFT" and value.created_by_subject_id != identity.subject_id:
        legal_actions.append("REVIEW_KNOWLEDGE_VERSION")
    if value.status in {"REVIEWED", "PUBLISHED"}:
        legal_actions.append("INCLUDE_IN_INDEX_RELEASE")
    if value.status in {"DRAFT", "REVIEWED", "PUBLISHED"}:
        legal_actions.append("REQUEST_KNOWLEDGE_DELETION")
    return KnowledgeVersionResponse(
        **{
            **asdict(value),
            "acl_subject_ids": list(value.acl_subject_ids),
            "acl_roles": list(value.acl_roles),
            "device_families": list(value.device_families),
            "device_models": list(value.device_models),
            "legal_actions": legal_actions,
        }
    )


def _release_response(value: KnowledgeReleaseView) -> KnowledgeReleaseResponse:
    legal_actions: list[str] = []
    if value.status == "EVALUATION_PENDING":
        legal_actions.append("RUN_KNOWLEDGE_INDEX_EVALUATION")
    if value.status == "CANDIDATE" and value.evaluation_status == "PASSED":
        legal_actions.append("PROMOTE_KNOWLEDGE_RELEASE")
    if value.rollback_eligible:
        legal_actions.append("ROLLBACK_KNOWLEDGE_RELEASE")
    return KnowledgeReleaseResponse(
        **{
            **asdict(value),
            "evaluation_failure_codes": list(value.evaluation_failure_codes),
        },
        legal_actions=legal_actions,
    )


def _activation_response(
    value: KnowledgeIndexActivationView,
) -> KnowledgeIndexActivationResponse:
    return KnowledgeIndexActivationResponse(
        **{
            **asdict(value),
            "action": value.action,
            "retired_search_profile_ids": list(value.retired_search_profile_ids),
            "revoked_graph_release_ids": list(value.revoked_graph_release_ids),
        }
    )


def _file_envelope(value: KnowledgeFileView, request_id: str) -> KnowledgeFileEnvelope:
    return KnowledgeFileEnvelope(
        data=_file_response(value),
        meta={"request_id": request_id},
    )


def _file_response(value: KnowledgeFileView) -> KnowledgeFileResponse:
    legal_actions: list[str] = []
    if value.status == "AWAITING_UPLOAD":
        legal_actions.append("UPLOAD_KNOWLEDGE_SOURCE_CONTENT")
    if value.status == "FAILED":
        legal_actions.append("REPROCESS_KNOWLEDGE_SOURCE_FILE")
    if value.status == "QUEUED":
        legal_actions.append("RETRY_KNOWLEDGE_INGESTION_DISPATCH")
    if value.document_version_id is not None:
        legal_actions.append("VIEW_KNOWLEDGE_VERSION")
    return KnowledgeFileResponse(
        ingestion_id=value.ingestion_id,
        workflow_id=value.workflow_id,
        title=value.title,
        source_filename=value.source_filename,
        declared_mime=value.declared_mime,
        detected_mime=value.detected_mime,
        classification=value.classification,
        status=value.status,
        failure_reason=value.failure_reason,
        source_checksum=value.source_checksum,
        size_bytes=value.size_bytes,
        parser_version=value.parser_version,
        document_id=value.document_id,
        document_version_id=value.document_version_id,
        attempt_count=value.attempt_count,
        attempts=[KnowledgeFileAttemptResponse(**asdict(attempt)) for attempt in value.attempts],
        version=value.version,
        created_at=value.created_at,
        updated_at=value.updated_at,
        legal_actions=legal_actions,
    )


def _state_version(value: str) -> int:
    try:
        parsed = int(value.strip().strip('"'))
    except ValueError as exc:
        raise AppError(
            400,
            "invalid_version_precondition",
            "validation",
            "If-Match is invalid",
        ) from exc
    if parsed < 1:
        raise AppError(400, "invalid_version_precondition", "validation", "If-Match is invalid")
    return parsed


def _translate(exc: Exception) -> AppError:
    if isinstance(exc, AuthorizationDenied):
        return AppError(403, "authorization_denied", "authorization", "Action is not allowed")
    if isinstance(exc, InvalidIndexRelease):
        return AppError(
            409,
            str(exc),
            "knowledge_governance",
            "Knowledge index publication gate rejected the operation",
        )
    if isinstance(exc, KnowledgeIngestionDispatchUnavailable):
        return AppError(
            503,
            "knowledge_ingestion_workflow_unavailable",
            "dependency",
            "Knowledge ingestion workflow unavailable",
            retryable=True,
        )
    if isinstance(exc, KnowledgeDeletionDispatchUnavailable):
        return AppError(
            503,
            "knowledge_deletion_workflow_unavailable",
            "dependency",
            "Knowledge deletion workflow unavailable",
            retryable=True,
        )
    if isinstance(exc, KnowledgeIndexEvaluationDispatchUnavailable):
        return AppError(
            503,
            "knowledge_index_evaluation_workflow_unavailable",
            "dependency",
            "Knowledge index evaluation workflow unavailable",
            retryable=True,
        )
    if isinstance(exc, KnowledgeIndexEvaluationError):
        if exc.reason.endswith("_not_visible"):
            return AppError(
                404,
                "resource_not_found_or_not_visible",
                "not_found",
                "Resource not found or not visible",
            )
        if exc.reason in {
            "knowledge_index_evaluation_idempotency_conflict",
            "knowledge_index_evaluation_state_invalid",
            "knowledge_index_evaluation_release_invalid",
            "knowledge_index_evaluation_workflow_stale",
        }:
            return AppError(
                409,
                exc.reason,
                "knowledge_governance",
                "Knowledge index evaluation rejected the operation",
            )
        return AppError(400, exc.reason, "validation", "Index evaluation request is invalid")
    if isinstance(exc, KnowledgeDeletionError):
        if exc.reason.endswith("_not_visible"):
            return AppError(
                404,
                "resource_not_found_or_not_visible",
                "not_found",
                "Resource not found or not visible",
            )
        if exc.reason in {
            "knowledge_deletion_idempotency_conflict",
            "knowledge_deletion_target_inactive",
            "knowledge_deletion_state_changed",
            "knowledge_deletion_state_invalid",
        }:
            return AppError(
                409,
                exc.reason,
                "knowledge_governance",
                "Knowledge deletion rejected the operation",
            )
        return AppError(400, exc.reason, "validation", "Knowledge deletion request is invalid")
    if isinstance(exc, KnowledgeFileError):
        if exc.reason.endswith("_not_visible"):
            return AppError(
                404,
                "resource_not_found_or_not_visible",
                "not_found",
                "Resource not found or not visible",
            )
        if exc.reason in {
            "knowledge_file_idempotency_conflict",
            "knowledge_file_idempotency_state_invalid",
            "knowledge_file_content_conflict",
            "knowledge_file_state_changed",
            "knowledge_file_not_ready",
            "knowledge_file_reprocess_forbidden",
            "knowledge_file_source_state_invalid",
            "knowledge_file_attempt_state_invalid",
            "knowledge_file_workflow_stale",
        }:
            return AppError(
                409,
                exc.reason,
                "knowledge_governance",
                "Knowledge file ingestion rejected the operation",
            )
        return AppError(400, exc.reason, "validation", "Knowledge file request is invalid")
    if not isinstance(exc, KnowledgeIngestionError):
        raise TypeError("unsupported knowledge governance error")
    if exc.reason.endswith("_not_visible"):
        return AppError(
            404,
            "resource_not_found_or_not_visible",
            "not_found",
            "Resource not found or not visible",
        )
    if exc.reason in {
        "knowledge_idempotency_conflict",
        "knowledge_idempotency_state_invalid",
        "knowledge_version_state_changed",
        "knowledge_review_separation_required",
        "knowledge_release_document_version_conflict",
        "knowledge_release_version_not_reviewed",
    }:
        return AppError(
            409,
            exc.reason,
            "knowledge_governance",
            "Knowledge governance rejected the operation",
        )
    return AppError(400, exc.reason, "validation", "Knowledge request is invalid")
