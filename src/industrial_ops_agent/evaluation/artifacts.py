"""Immutable evaluation reports and safe PEFT adapter extraction."""

from __future__ import annotations

import json
import tarfile
from collections.abc import Mapping
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

from industrial_ops_agent.data_pipeline.storage import DatasetStore, ImmutableObjectExists
from industrial_ops_agent.experiments.service import ArtifactInput


class EvaluationArtifactError(RuntimeError):
    pass


def extract_adapter_bundle(content: bytes, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    total_size = 0
    with tarfile.open(fileobj=BytesIO(content), mode="r:gz") as archive:
        members = archive.getmembers()
        if not members or len(members) > 10_000:
            raise EvaluationArtifactError("adapter_archive_member_count_is_invalid")
        for member in members:
            path = Path(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or member.issym()
                or member.islnk()
                or not member.isfile()
            ):
                raise EvaluationArtifactError("adapter_archive_contains_unsafe_member")
            total_size += member.size
            if total_size > 8 * 1024 * 1024 * 1024:
                raise EvaluationArtifactError("adapter_archive_is_too_large")
            source = archive.extractfile(member)
            if source is None:
                raise EvaluationArtifactError("adapter_archive_member_is_unreadable")
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())


def verify_artifact_bytes(content: bytes, *, content_hash: str, size_bytes: int) -> None:
    if len(content) != size_bytes or sha256(content).hexdigest() != content_hash:
        raise EvaluationArtifactError("training_adapter_integrity_failed")


def store_case_evidence(
    store: DatasetStore,
    *,
    tenant_id: str,
    job_id: str,
    report: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> ArtifactInput:
    content = json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    content_hash = sha256(content).hexdigest()
    object_key = (
        f"tenant/{tenant_id}/evaluations/jobs/{job_id}/"
        f"case-evidence-{content_hash}.json"
    )
    try:
        store.put_bytes(object_key, content)
    except ImmutableObjectExists:
        if store.get_bytes(object_key) != content:
            raise EvaluationArtifactError("immutable_evaluation_artifact_conflict") from None
    return ArtifactInput(
        kind="case_evidence",
        object_key=object_key,
        content_hash=content_hash,
        size_bytes=len(content),
        metadata={"format": "json", **dict(metadata)},
    )
