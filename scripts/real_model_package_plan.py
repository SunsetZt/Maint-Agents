"""Retain existing packaging metadata; never attest model/package validity."""
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import sys
import time

from real_model_inspect import _Fault, _Inspector, _require, _SMOL_MODEL, _SMOL_REVISION

_LIMIT = 1024 * 1024
_SOURCE_KEYS = {"diagnosis_adapter_source", "vlm_cache_source", "vlm_adapter_source"}
_SHARED_KEYS = _SOURCE_KEYS | {"diagnosis_source_image", "diagnosis_source_digest",
                               "vlm_revision", "gpu_probe_image"}
_COMPONENT_KEYS = {"volume", "base_model_id", "model_id", "package_digest",
                   "base_files", "adapter_files", "base_revision"}
_LOCAL_ADAPTER_KEYS = {"diagnosis_source_volume", "diagnosis_source_package_digest",
                       "diagnosis_adapter_bundle", "diagnosis_adapter_content_hash",
                       "diagnosis_serving_source_files"}


def _text(value):
    _require(isinstance(value, str) and 0 < len(value) <= 4096
             and not any(ord(c) < 32 or ord(c) == 127 for c in value), "PLAN_SCHEMA")


def _hex(value):
    _require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value), "PLAN_SCHEMA")


def _validate(plan):
    keys = _SHARED_KEYS | {c + "_" + k for c in ("diagnosis", "vlm") for k in _COMPONENT_KEYS}
    base_vlm = "schema_version" in plan
    if base_vlm:
        _require(type(plan["schema_version"]) is int and plan["schema_version"] == 2, "PLAN_SCHEMA")
        keys = (keys - {"vlm_adapter_source"}) | {
            "schema_version", "vlm_artifact_kind", "vlm_source_manifest_sha256"}
        _require(plan.get("vlm_artifact_kind") == "BASE_CHECKPOINT"
                 and plan.get("vlm_base_model_id") == _SMOL_MODEL
                 and plan.get("vlm_revision") == plan.get("vlm_base_revision") == _SMOL_REVISION
                 and plan.get("vlm_adapter_files") == []
                 and plan.get("diagnosis_base_model_id") == "Qwen/Qwen3-0.6B"
                 and plan.get("diagnosis_model_id") != plan.get("vlm_model_id"), "PLAN_SCHEMA")
        _hex(plan.get("vlm_source_manifest_sha256"))
    local_adapter = "diagnosis_adapter_content_hash" in plan
    if local_adapter:
        _require(base_vlm, "PLAN_SCHEMA")
        keys |= _LOCAL_ADAPTER_KEYS
        _hex(plan.get("diagnosis_adapter_content_hash"))
        _hex(plan.get("diagnosis_source_package_digest"))
        sources = plan.get("diagnosis_serving_source_files")
        _require(isinstance(sources, dict) and set(sources) == {"owned_runtime.py", "router_runtime.py"}, "PLAN_SCHEMA")
        for value in sources.values():
            _hex(value)
    _require(set(plan) == keys, "PLAN_SCHEMA")
    for key, value in plan.items():
        if key in {"schema_version", "diagnosis_serving_source_files"}:
            continue
        if key.endswith("_files"):
            if base_vlm and key == "vlm_adapter_files":
                continue
            _require(isinstance(value, list) and value, "PLAN_SCHEMA")
            seen = set()
            for row in value:
                _require(isinstance(row, dict) and set(row) == {"path", "size_bytes", "sha256"}, "PLAN_SCHEMA")
                relative = row["path"]
                _text(relative)
                path = PurePosixPath(relative)
                _require(not path.is_absolute() and ".." not in path.parts
                         and relative not in (".", "") and str(path) == relative
                         and "\\" not in relative and relative not in seen, "PLAN_SCHEMA")
                _require(type(row["size_bytes"]) is int and 0 <= row["size_bytes"] < 2**63, "PLAN_SCHEMA")
                _hex(row["sha256"])
                seen.add(relative)
        else:
            _text(value)
            if key in _SOURCE_KEYS or key == "diagnosis_adapter_bundle":
                _require(value.startswith("/") and ".." not in PurePosixPath(value).parts, "PLAN_SCHEMA")
            elif key.endswith("_image"):
                reference = r"[A-Za-z0-9][A-Za-z0-9._/:-]*"
                qualified = reference + r"@sha256:[a-f0-9]{64}"
                # Source identity is checked separately by verify_diagnosis_source_image.
                pattern = qualified if key == "gpu_probe_image" else reference + r"(?:@sha256:[a-f0-9]{64})?"
                _require("://" not in value and re.fullmatch(pattern, value), "PLAN_SCHEMA")
                if key == "diagnosis_source_image" and "@" in value:
                    _require(value.rsplit("@", 1)[1] == plan["diagnosis_source_digest"], "PLAN_SCHEMA")
            elif key == "diagnosis_source_digest":
                _require(re.fullmatch(r"sha256:[a-f0-9]{64}", value), "PLAN_SCHEMA")
            elif key.endswith("_package_digest"):
                _hex(value)
            else:
                _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/+-]{0,511}", value), "PLAN_SCHEMA")


def _existing_matches(directory_fd, name, data):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                 and stat.S_IMODE(info.st_mode) == 0o600 and info.st_size == len(data),
                 "SNAPSHOT_CONFLICT")
        _require(stream.read(_LIMIT + 1) == data, "SNAPSHOT_CONFLICT")


def verified_vlm_source(root):
    """Hash the fixed base snapshot without importing a model framework or following links."""
    reader = _Inspector()
    reader.deadline = time.monotonic() + 240
    manifest = Path(__file__).resolve().parents[1] / "infra/model-runtime/smolvlm-500m.json"
    source = reader.source_manifest(manifest)
    expected = {row["path"] for row in source["files"]}
    directory = reader.parent(root / "config.json")
    try:
        names = set()
        with os.scandir(directory) as entries:
            for entry in entries:
                names.add(entry.name)
                _require(len(names) <= len(expected), "SOURCE_FILE_SET_CHANGED")
        _require(names == expected, "SOURCE_FILE_SET_CHANGED")
    finally:
        os.close(directory)
    started, consumed = time.monotonic(), 0
    for row in source["files"]:
        path = root / row["path"]
        before = reader.file_stat(path)
        _require(before.st_size == row["size_bytes"], "SOURCE_CONTENT_CHANGED")
        parent = reader.parent(path)
        try:
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        finally:
            os.close(parent)
        digest, size = sha256(), 0
        with os.fdopen(fd, "rb") as stream:
            def identity(info):
                return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
            _require(identity(before) == identity(os.fstat(stream.fileno())), "SOURCE_CONTENT_CHANGED")
            while block := stream.read(_LIMIT):
                reader.check()
                size += len(block)
                consumed += len(block)
                _require(size <= row["size_bytes"], "SOURCE_CONTENT_CHANGED")
                digest.update(block)
                time.sleep(max(0, consumed / (32 * _LIMIT) - (time.monotonic() - started)))
            _require(identity(before) == identity(os.fstat(stream.fileno())), "SOURCE_CONTENT_CHANGED")
        _require(size == row["size_bytes"] and digest.hexdigest() == row["sha256"], "SOURCE_CONTENT_CHANGED")
    return source, reader.metadata[str(manifest)]["sha256"]


def retain(source, destination):
    if not destination.is_absolute():
        destination = Path.cwd() / destination
    reader = _Inspector()
    plan = reader.read_json(source)
    _validate(plan)
    data = (json.dumps(plan, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _require(len(data) <= _LIMIT, "METADATA_LIMIT")
    digest = sha256(data).hexdigest()
    name = "real-model-package-plan-" + digest + ".json"
    snapshot = destination / name
    directory_fd = reader.parent(snapshot)
    temporary = None
    try:
        info = os.fstat(directory_fd)
        _require(info.st_uid == os.getuid() and not info.st_mode & 0o022, "UNSAFE_DESTINATION")
        temporary = ".package-plan-" + secrets.token_hex(16)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory_fd)
        # The exact owned name is the only temporary artifact this function removes.
        try:
            with os.fdopen(fd, "wb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, name, src_dir_fd=directory_fd,
                        dst_dir_fd=directory_fd, follow_symlinks=False)
            except FileExistsError:
                _existing_matches(directory_fd, name, data)
            os.fsync(directory_fd)
        finally:
            os.unlink(temporary, dir_fd=directory_fd)
            temporary = None
        # A renamed output directory must not yield an apparently reusable path.
        current_fd = reader.parent(snapshot)
        try:
            current = os.fstat(current_fd)
            _require((current.st_dev, current.st_ino) == (info.st_dev, info.st_ino), "UNSAFE_DESTINATION")
        finally:
            os.close(current_fd)
    finally:
        os.close(directory_fd)
    return {"package_plan": str(snapshot), "sha256": digest,
            "size_bytes": len(data), "purpose": "COLD_PROBE_INPUT_ONLY"}


def main(argv):
    try:
        _require(len(argv) == 2, "INVALID_ARGUMENTS")
        source, destination = map(Path, argv)
        descriptor = retain(source, destination)
    except (OSError, _Fault, ValueError, TypeError, KeyError, RecursionError):
        print("PACKAGE_PLAN_SNAPSHOT_REFUSED", file=sys.stderr)
        return 3
    print(json.dumps(descriptor, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
