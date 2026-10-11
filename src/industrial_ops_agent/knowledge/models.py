"""Stable contracts shared by knowledge retrieval and the diagnosis graph."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    tenant_id: str
    subject_id: str
    roles: frozenset[str]
    release_id: str
    device_family: str | None
    device_model: str
    query: str
    as_of: datetime
    limit: int = 6


@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    chunk_id: str
    citation_id: str
    document_id: str
    document_version_id: str
    document_version: int
    title: str
    content: str
    content_checksum: str
    source_checksum: str
    page_number: int | None
    device_model: str
    sparse_rank: int | None
    vector_rank: int | None
    fused_score: float
    reranker_score: float | None = None
    final_rank: int | None = None

    def citation_payload(self) -> dict[str, str | int | float | None]:
        return {
            "citation_id": self.citation_id,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_version_id": self.document_version_id,
            "document_version": self.document_version,
            "title": self.title,
            "source_checksum": self.source_checksum,
            "content_checksum": self.content_checksum,
            "page_number": self.page_number,
            "device_model": self.device_model,
            "fused_score": round(self.fused_score, 6),
            "reranker_score": (
                round(self.reranker_score, 6) if self.reranker_score is not None else None
            ),
            "final_rank": self.final_rank,
        }


@dataclass(frozen=True, slots=True)
class RetrievalGuardTrace:
    """Non-sensitive proof that policy filtering preceded every recall path."""

    release_id: str
    backend: str
    filter_stage: str
    policy_dimensions: tuple[str, ...]
    sparse_candidate_count: int
    vector_candidate_count: int
    fused_candidate_count: int
    returned_count: int
    unauthorized_candidate_count: int


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    evidence: tuple[RetrievedEvidence, ...]
    guard: RetrievalGuardTrace


@dataclass(frozen=True, slots=True)
class CitationView:
    citation_id: str
    document_id: str
    document_version_id: str
    document_version: int
    chunk_id: str
    title: str
    source_uri: str
    source_checksum: str
    content_checksum: str
    content_excerpt: str
    page_number: int | None
    bounding_box: dict[str, float] | None
    start_offset: int
    end_offset: int
