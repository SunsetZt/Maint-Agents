"""Prompt-free manifest for the exact governed context sent to one model call."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal

ContextSourceType = Literal["QUERY", "EVIDENCE", "MEMORY", "TOOL_RESULT"]
ContextTruncationReason = Literal["COUNT_LIMIT", "CHARACTER_LIMIT", "TOKEN_BUDGET"]

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,254}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "trace_id",
        "prompt_bundle_id",
        "prompt_bundle_hash",
        "model_release_id",
        "retrieval_index_id",
        "evidence",
        "memories",
        "tool_results",
        "token_budget",
        "truncated_items",
        "rendered_context_hash",
        "context_hash",
    }
)


@dataclass(frozen=True, slots=True)
class ContextReference:
    reference_id: str
    content_hash: str

    def __post_init__(self) -> None:
        _validate_identifier(self.reference_id, "context_reference_id_invalid")
        if _SHA256.fullmatch(self.content_hash) is None:
            raise ValueError("context_reference_hash_invalid")

    def as_dict(self) -> dict[str, str]:
        return {"id": self.reference_id, "content_hash": self.content_hash}


@dataclass(frozen=True, slots=True)
class ContextTruncation:
    source_type: ContextSourceType
    source_id: str
    reason: ContextTruncationReason
    original_units: int
    included_units: int

    def __post_init__(self) -> None:
        _validate_identifier(self.source_id, "context_truncation_source_invalid")
        if self.original_units <= 0 or not 0 <= self.included_units < self.original_units:
            raise ValueError("context_truncation_units_invalid")

    def as_dict(self) -> dict[str, str | int]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "reason": self.reason,
            "original_units": self.original_units,
            "included_units": self.included_units,
        }


@dataclass(frozen=True, slots=True)
class ContextManifest:
    trace_id: str
    prompt_bundle_id: str | None
    prompt_bundle_hash: str | None
    model_release_id: str
    retrieval_index_id: str | None
    evidence: tuple[ContextReference, ...]
    memories: tuple[ContextReference, ...]
    tool_results: tuple[ContextReference, ...]
    token_budget: int
    truncated_items: tuple[ContextTruncation, ...]
    rendered_context_hash: str
    context_hash: str
    schema_version: str = "context-manifest/v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "prompt_bundle_id": self.prompt_bundle_id,
            "prompt_bundle_hash": self.prompt_bundle_hash,
            "model_release_id": self.model_release_id,
            "retrieval_index_id": self.retrieval_index_id,
            "evidence": [item.as_dict() for item in self.evidence],
            "memories": [item.as_dict() for item in self.memories],
            "tool_results": [item.as_dict() for item in self.tool_results],
            "token_budget": self.token_budget,
            "truncated_items": [item.as_dict() for item in self.truncated_items],
            "rendered_context_hash": self.rendered_context_hash,
            "context_hash": self.context_hash,
        }


def build_context_manifest(
    *,
    trace_id: str,
    prompt_bundle_id: str | None,
    prompt_bundle_hash: str | None,
    model_release_id: str,
    retrieval_index_id: str | None,
    evidence: tuple[ContextReference, ...],
    memories: tuple[ContextReference, ...],
    tool_results: tuple[ContextReference, ...],
    token_budget: int,
    truncated_items: tuple[ContextTruncation, ...],
    messages: tuple[dict[str, Any], ...],
    response_schema: dict[str, Any],
) -> ContextManifest:
    _validate_identifier(trace_id, "context_trace_id_invalid")
    _validate_identifier(model_release_id, "context_model_release_id_invalid")
    if prompt_bundle_id is not None:
        _validate_identifier(prompt_bundle_id, "context_prompt_bundle_id_invalid")
    if retrieval_index_id is not None:
        _validate_identifier(retrieval_index_id, "context_retrieval_index_id_invalid")
    if (prompt_bundle_id is None) != (prompt_bundle_hash is None):
        raise ValueError("context_prompt_bundle_binding_incomplete")
    if prompt_bundle_hash is not None and _SHA256.fullmatch(prompt_bundle_hash) is None:
        raise ValueError("context_prompt_bundle_hash_invalid")
    if not 1 <= token_budget <= 1_000_000:
        raise ValueError("context_token_budget_invalid")
    for references in (evidence, memories, tool_results):
        ids = [item.reference_id for item in references]
        if len(ids) != len(set(ids)):
            raise ValueError("context_reference_duplicate")
    rendered_context_hash = canonical_sha256(
        {"messages": messages, "response_schema": response_schema}
    )
    hash_basis = {
        "schema_version": "context-manifest/v1",
        "prompt_bundle_id": prompt_bundle_id,
        "prompt_bundle_hash": prompt_bundle_hash,
        "model_release_id": model_release_id,
        "retrieval_index_id": retrieval_index_id,
        "evidence": [item.as_dict() for item in evidence],
        "memories": [item.as_dict() for item in memories],
        "tool_results": [item.as_dict() for item in tool_results],
        "token_budget": token_budget,
        "truncated_items": [item.as_dict() for item in truncated_items],
        "rendered_context_hash": rendered_context_hash,
    }
    return ContextManifest(
        trace_id=trace_id,
        prompt_bundle_id=prompt_bundle_id,
        prompt_bundle_hash=prompt_bundle_hash,
        model_release_id=model_release_id,
        retrieval_index_id=retrieval_index_id,
        evidence=evidence,
        memories=memories,
        tool_results=tool_results,
        token_budget=token_budget,
        truncated_items=truncated_items,
        rendered_context_hash=rendered_context_hash,
        context_hash=canonical_sha256(hash_basis),
    )


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{sha256(encoded).hexdigest()}"


def context_manifest_is_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != _MANIFEST_KEYS:
        return False
    prompt_bundle_id = value.get("prompt_bundle_id")
    prompt_bundle_hash = value.get("prompt_bundle_hash")
    if value.get("schema_version") != "context-manifest/v1":
        return False
    if not _is_identifier(value.get("trace_id")) or not _is_identifier(
        value.get("model_release_id")
    ):
        return False
    if prompt_bundle_id is not None and not _is_identifier(prompt_bundle_id):
        return False
    if (prompt_bundle_id is None) != (prompt_bundle_hash is None):
        return False
    if prompt_bundle_hash is not None and not _is_sha256(prompt_bundle_hash):
        return False
    retrieval_index_id = value.get("retrieval_index_id")
    if retrieval_index_id is not None and not _is_identifier(retrieval_index_id):
        return False
    if not all(
        _references_are_valid(value.get(field))
        for field in ("evidence", "memories", "tool_results")
    ):
        return False
    token_budget = value.get("token_budget")
    if (
        not isinstance(token_budget, int)
        or isinstance(token_budget, bool)
        or not 1 <= token_budget <= 1_000_000
    ):
        return False
    if not _truncations_are_valid(value.get("truncated_items")):
        return False
    context_hash = value.get("context_hash")
    if (
        not _is_sha256(value.get("rendered_context_hash"))
        or not isinstance(context_hash, str)
        or not _is_sha256(context_hash)
    ):
        return False
    hash_basis = {
        key: value[key]
        for key in (
            "schema_version",
            "prompt_bundle_id",
            "prompt_bundle_hash",
            "model_release_id",
            "retrieval_index_id",
            "evidence",
            "memories",
            "tool_results",
            "token_budget",
            "truncated_items",
            "rendered_context_hash",
        )
    }
    return canonical_sha256(hash_basis) == context_hash


def _references_are_valid(value: object) -> bool:
    if not isinstance(value, list) or len(value) > 10_000:
        return False
    ids: list[str] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"id", "content_hash"}:
            return False
        reference_id = item.get("id")
        if (
            not isinstance(reference_id, str)
            or not _is_identifier(reference_id)
            or not _is_sha256(item.get("content_hash"))
        ):
            return False
        ids.append(reference_id)
    return len(ids) == len(set(ids))


def _truncations_are_valid(value: object) -> bool:
    if not isinstance(value, list) or len(value) > 10_000:
        return False
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "source_type",
            "source_id",
            "reason",
            "original_units",
            "included_units",
        }:
            return False
        original = item.get("original_units")
        included = item.get("included_units")
        if (
            item.get("source_type") not in {"QUERY", "EVIDENCE", "MEMORY", "TOOL_RESULT"}
            or not _is_identifier(item.get("source_id"))
            or item.get("reason") not in {"COUNT_LIMIT", "CHARACTER_LIMIT", "TOKEN_BUDGET"}
            or not isinstance(original, int)
            or isinstance(original, bool)
            or not isinstance(included, int)
            or isinstance(included, bool)
            or original <= 0
            or not 0 <= included < original
        ):
            return False
    return True


def _is_identifier(value: object) -> bool:
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _validate_identifier(value: str, reason: str) -> None:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(reason)
