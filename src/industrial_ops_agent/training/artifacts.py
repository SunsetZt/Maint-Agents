"""Deterministic packaging for adapter/checkpoint artifacts."""

from __future__ import annotations

import gzip
import tarfile
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from industrial_ops_agent.data_pipeline.storage import (
    DatasetStore,
    ImmutableObjectExists,
)
from industrial_ops_agent.experiments.service import ArtifactInput


class TrainingArtifactError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    descriptor: ArtifactInput
    content: bytes


def package_directory(directory: Path) -> bytes:
    """Create a byte-reproducible gzip tarball without host ownership or timestamps."""

    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise TrainingArtifactError("training_output_directory_is_empty")
    raw_tar = BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in files:
            relative = path.relative_to(directory).as_posix()
            info = archive.gettarinfo(str(path), arcname=relative)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with path.open("rb") as source:
                archive.addfile(info, source)
    compressed = BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0, filename="") as target:
        target.write(raw_tar.getvalue())
    return compressed.getvalue()


def extract_training_bundle(content: bytes, destination: Path) -> None:
    """Extract a governed training bundle without links or path traversal."""

    destination.mkdir(parents=True, exist_ok=False)
    total_size = 0
    try:
        with tarfile.open(fileobj=BytesIO(content), mode="r:gz") as archive:
            members = archive.getmembers()
            if not members or len(members) > 25_000:
                raise TrainingArtifactError("training_bundle_member_count_is_invalid")
            for member in members:
                relative = Path(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or member.issym()
                    or member.islnk()
                    or not member.isfile()
                ):
                    raise TrainingArtifactError("training_bundle_contains_unsafe_member")
                total_size += member.size
                if total_size > 16 * 1024 * 1024 * 1024:
                    raise TrainingArtifactError("training_bundle_is_too_large")
                source = archive.extractfile(member)
                if source is None:
                    raise TrainingArtifactError("training_bundle_member_is_unreadable")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read())
    except tarfile.TarError as exc:
        raise TrainingArtifactError("training_bundle_is_not_a_valid_tar_gzip") from exc


def store_adapter_bundle(
    store: DatasetStore,
    *,
    tenant_id: str,
    experiment_id: str,
    directory: Path,
    metadata: dict[str, object],
) -> StoredArtifact:
    return store_training_bundle(
        store,
        tenant_id=tenant_id,
        experiment_id=experiment_id,
        directory=directory,
        kind="adapter_bundle",
        name="adapter",
        serialization="safetensors",
        metadata=metadata,
    )


def store_training_bundle(
    store: DatasetStore,
    *,
    tenant_id: str,
    experiment_id: str,
    directory: Path,
    kind: str,
    name: str,
    serialization: str,
    metadata: dict[str, object],
) -> StoredArtifact:
    content = package_directory(directory)
    content_hash = sha256(content).hexdigest()
    object_key = (
        f"tenant/{tenant_id}/experiments/{experiment_id}/"
        f"{name}-{content_hash}.tar.gz"
    )
    try:
        store.put_bytes(object_key, content)
    except ImmutableObjectExists:
        if store.get_bytes(object_key) != content:
            raise TrainingArtifactError("immutable_training_artifact_conflict") from None
    descriptor = ArtifactInput(
        kind=kind,
        object_key=object_key,
        content_hash=content_hash,
        size_bytes=len(content),
        metadata={"format": "tar.gz", "serialization": serialization, **metadata},
    )
    return StoredArtifact(descriptor=descriptor, content=content)
