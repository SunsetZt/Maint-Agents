"""Closed interchange contracts for governed disconnected field diagnosis."""

from __future__ import annotations

import json
import math
from datetime import datetime
from hashlib import sha256
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FIELD_EDGE_PACK_SCHEMA_VERSION = "field-edge-diagnosis-pack/v1"
FIELD_EDGE_RESULT_SCHEMA_VERSION = "field-edge-diagnosis-result/v1"
FIELD_EDGE_ISSUER = "industrial-ops-field-edge"
FIELD_EDGE_AUDIENCE = "industrial-ops-field-edge-runner"


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FieldEdgeSnapshotWorkOrder(_ClosedModel):
    work_order_id: str = Field(min_length=1, max_length=128)
    incident_id: str = Field(min_length=1, max_length=128)
    status: str = Field(min_length=1, max_length=32)
    priority: str = Field(min_length=1, max_length=32)
    sla_due_at: str | None = Field(default=None, max_length=64)
    service_window_start: str | None = Field(default=None, max_length=64)
    service_window_end: str | None = Field(default=None, max_length=64)
    version: int = Field(ge=1)


class FieldEdgeSnapshotAsset(_ClosedModel):
    asset_id: str = Field(min_length=1, max_length=128)
    display_name: str | None = Field(default=None, max_length=255)
    model_code: str | None = Field(default=None, max_length=128)
    serial_number: str | None = Field(default=None, max_length=255)
    lifecycle_status: str | None = Field(default=None, max_length=32)
    version: int = Field(ge=1)


class FieldEdgeSnapshotSite(_ClosedModel):
    site_id: str = Field(min_length=1, max_length=128)
    site_name: str = Field(min_length=1, max_length=255)


class FieldEdgeSnapshotIncident(_ClosedModel):
    incident_id: str = Field(min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=4000)
    severity: str | None = Field(default=None, max_length=32)
    category: str | None = Field(default=None, max_length=128)


class FieldEdgeSnapshotDiagnosis(_ClosedModel):
    diagnosis_run_id: str = Field(min_length=1, max_length=128)
    status: str = Field(min_length=1, max_length=32)
    version: int = Field(ge=1)
    conclusion: str | None = Field(default=None, max_length=2000)
    next_checks: list[str] = Field(default_factory=list, max_length=12)
    recommended_actions: list[str] = Field(default_factory=list, max_length=12)
    confidence: float | None = Field(default=None, ge=0, le=1)
    citation_ids: list[str] = Field(default_factory=list, max_length=20)


class FieldEdgeSnapshotCitation(_ClosedModel):
    citation_id: str = Field(min_length=1, max_length=128)
    anchor_kind: str = Field(min_length=1, max_length=64)
    page_number: int | None = Field(default=None, ge=1)
    excerpt: str | None = Field(default=None, max_length=1000)
    excerpt_checksum: str = Field(min_length=1, max_length=128)


class FieldEdgeSnapshot(_ClosedModel):
    work_order: FieldEdgeSnapshotWorkOrder
    asset: FieldEdgeSnapshotAsset
    site: FieldEdgeSnapshotSite | None
    incident: FieldEdgeSnapshotIncident
    diagnosis: FieldEdgeSnapshotDiagnosis | None
    citations: list[FieldEdgeSnapshotCitation] = Field(default_factory=list, max_length=20)
    legal_actions: list[Literal["READ_OFFLINE_SNAPSHOT"]] = Field(max_length=1)


class FieldEdgeWorkOrderBinding(_ClosedModel):
    work_order_id: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=1)
    status: Literal["ACCEPTED", "IN_PROGRESS", "ON_HOLD"]


class FieldEdgeAssetBinding(_ClosedModel):
    asset_id: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=1)


class FieldEdgeAssignmentBinding(_ClosedModel):
    assignment_id: str = Field(min_length=1, max_length=128)
    assigned_at: datetime


class FieldEdgeOfflinePackBinding(_ClosedModel):
    pack_id: str = Field(min_length=1, max_length=128)
    content_hash: str = Field(min_length=1, max_length=128)
    expires_at: datetime
    snapshot: FieldEdgeSnapshot


class FieldEdgeInferenceConfig(_ClosedModel):
    engine: Literal["llama.cpp"]
    context_size: int = Field(ge=1024, le=131_072)
    threads: int = Field(ge=1, le=256)
    batch_size: int = Field(ge=1, le=4096)
    mmap: bool
    mlock: bool
    output_tokens: int = Field(default=512, ge=64, le=2048)
    timeout_seconds: int = Field(default=120, ge=10, le=600)
    seed: Literal[42] = 42
    temperature: Literal[0] = 0


class FieldEdgeReleaseBinding(_ClosedModel):
    release_id: str = Field(min_length=1, max_length=128)
    manifest_hash: str = Field(min_length=1, max_length=128)
    runtime_profile_id: str = Field(min_length=1, max_length=128)
    model_file: str = Field(
        min_length=6,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*\.gguf$",
    )
    model_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    inference_config: FieldEdgeInferenceConfig


class FieldEdgePromptBinding(_ClosedModel):
    prompt_bundle_id: str = Field(min_length=1, max_length=128)
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class FieldEdgeRetrievalBinding(_ClosedModel):
    index_release_id: str = Field(min_length=1, max_length=128)
    content_checksum: str = Field(min_length=1, max_length=128)


class FieldEdgeDiagnosisPackClaims(_ClosedModel):
    schema_version: Literal["field-edge-diagnosis-pack/v1"] = (
        FIELD_EDGE_PACK_SCHEMA_VERSION
    )
    iss: Literal["industrial-ops-field-edge"] = FIELD_EDGE_ISSUER
    aud: Literal["industrial-ops-field-edge-runner"] = FIELD_EDGE_AUDIENCE
    sub: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=64)
    iat: int = Field(ge=0)
    nbf: int = Field(ge=0)
    exp: int = Field(ge=0)
    jti: str = Field(min_length=1, max_length=128)
    work_order: FieldEdgeWorkOrderBinding
    asset: FieldEdgeAssetBinding
    assignment: FieldEdgeAssignmentBinding
    offline_pack: FieldEdgeOfflinePackBinding
    release: FieldEdgeReleaseBinding
    prompt_bundle: FieldEdgePromptBinding
    retrieval: FieldEdgeRetrievalBinding

    @model_validator(mode="after")
    def validate_bindings(self) -> Self:
        if not self.iat <= self.nbf < self.exp:
            raise ValueError("field_edge_pack_time_window_invalid")
        snapshot = self.offline_pack.snapshot
        if (
            snapshot.work_order.work_order_id != self.work_order.work_order_id
            or snapshot.work_order.version != self.work_order.version
            or snapshot.asset.asset_id != self.asset.asset_id
            or snapshot.asset.version != self.asset.version
        ):
            raise ValueError("field_edge_pack_snapshot_binding_invalid")
        return self


class FieldEdgeSignedPackFile(_ClosedModel):
    schema_version: Literal["field-edge-diagnosis-pack/v1"] = (
        FIELD_EDGE_PACK_SCHEMA_VERSION
    )
    token: str = Field(min_length=64, max_length=32_768)
    key_id: str = Field(min_length=1, max_length=128)
    pack_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    def has_valid_pack_digest(self) -> bool:
        return self.pack_digest == f"sha256:{sha256(self.token.encode()).hexdigest()}"


class FieldEdgeDiagnosisCandidate(_ClosedModel):
    conclusion: str = Field(min_length=1, max_length=2000)
    possible_causes: list[str] = Field(default_factory=list, max_length=12)
    next_checks: list[str] = Field(default_factory=list, max_length=12)
    missing_information: list[str] = Field(default_factory=list, max_length=12)
    safety_warnings: list[str] = Field(default_factory=list, max_length=12)
    citation_ids: list[str] = Field(default_factory=list, max_length=20)
    confidence: float = Field(ge=0, le=1)

    @field_validator(
        "possible_causes",
        "next_checks",
        "missing_information",
        "safety_warnings",
    )
    @classmethod
    def validate_text_list(cls, value: list[str]) -> list[str]:
        if any(not item.strip() or len(item) > 500 for item in value):
            raise ValueError("field_edge_candidate_text_invalid")
        return _unique(value, "field_edge_candidate_text_duplicate")

    @field_validator("citation_ids")
    @classmethod
    def validate_citations(cls, value: list[str]) -> list[str]:
        if any(not item.strip() or len(item) > 128 for item in value):
            raise ValueError("field_edge_candidate_citation_invalid")
        return _unique(value, "field_edge_candidate_citation_duplicate")

    @field_validator("confidence")
    @classmethod
    def validate_finite_confidence(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("field_edge_candidate_confidence_invalid")
        return value


class FieldEdgeRuntimeEvidence(_ClosedModel):
    engine: Literal["llama.cpp"]
    runtime_attestation: Literal["UNATTESTED"]
    model_file: str = Field(
        min_length=6,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*\.gguf$",
    )
    model_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    prompt_bundle_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    started_at: datetime
    completed_at: datetime
    exit_code: Literal[0]

    @model_validator(mode="after")
    def validate_clock(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError("field_edge_runtime_clock_invalid")
        return self


class FieldEdgeDiagnosisResult(_ClosedModel):
    schema_version: Literal["field-edge-diagnosis-result/v1"] = (
        FIELD_EDGE_RESULT_SCHEMA_VERSION
    )
    pack_token: str = Field(min_length=64, max_length=32_768)
    candidate: FieldEdgeDiagnosisCandidate
    runtime_evidence: FieldEdgeRuntimeEvidence
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    def has_valid_content_hash(self) -> bool:
        return self.content_hash == field_edge_result_content_hash(
            self.pack_token,
            self.candidate,
            self.runtime_evidence,
        )


def field_edge_result_content_hash(
    pack_token: str,
    candidate: FieldEdgeDiagnosisCandidate,
    runtime_evidence: FieldEdgeRuntimeEvidence,
) -> str:
    return _digest(
        {
            "schema_version": FIELD_EDGE_RESULT_SCHEMA_VERSION,
            "pack_token": pack_token,
            "candidate": candidate.model_dump(mode="json"),
            "runtime_evidence": runtime_evidence.model_dump(mode="json"),
        }
    )


def canonical_digest(value: object) -> str:
    """Return the shared protocol hash for JSON-safe server-owned values."""

    return _digest(value)


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{sha256(encoded).hexdigest()}"


def _unique(value: list[str], reason: str) -> list[str]:
    if len(value) != len(set(value)):
        raise ValueError(reason)
    return value
