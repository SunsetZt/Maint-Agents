"""Atomic evidence report writes with distinct path and return contracts."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _replace_json(target: Path, value: object) -> None:
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(root: Path, path: Path, *, reported_path: Path | None = None) -> dict[str, Any]:
    target = path.resolve(strict=True)
    return {
        "path": (reported_path or target.relative_to(root)).as_posix(),
        "size_bytes": target.stat().st_size,
        "sha256": file_sha256(target),
    }

def write_resolved_json(path: Path, value: object) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    _replace_json(target, value)


def write_relative_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _replace_json(path, value)


def write_document(path: Path, document: dict[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    _replace_json(target, document)
    return target


def write_model_report(report: BaseModel, path: Path) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
