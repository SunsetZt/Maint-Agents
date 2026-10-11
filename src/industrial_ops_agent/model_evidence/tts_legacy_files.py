"""Legacy TTS evidence contracts; preserve original rejection and ASCII digest rules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from industrial_ops_agent.domain.json import legacy_ascii_canonical
from industrial_ops_agent.model_evidence.report_files import file_sha256


@dataclass(frozen=True, slots=True)
class LegacyTtsEvidenceFiles:
    error_type: type[RuntimeError]

    def directory_manifest(self, directory: Path) -> list[dict[str, Any]]:
        root = directory.resolve(strict=True)
        entries: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if relative.parts and relative.parts[0] == ".cache":
                continue
            if path.is_symlink():
                raise self.error_type("TTS artifact contains a symbolic link")
            if not path.is_file():
                continue
            if path.stat().st_size <= 0:
                raise self.error_type("TTS artifact contains an empty file")
            entries.append(
                {
                    "path": relative.as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
        if not entries:
            raise self.error_type("TTS artifact directory is empty")
        return entries

    def inside_file(self, root: Path, value: Path) -> Path:
        target = value if value.is_absolute() else root / value
        target = target.resolve(strict=True)
        self.require_inside(root, target)
        if not target.is_file():
            raise self.error_type("TTS evidence path is not a file")
        return target

    def inside_directory(self, root: Path, value: Path) -> Path:
        target = value if value.is_absolute() else root / value
        target = target.resolve(strict=True)
        self.require_inside(root, target)
        if not target.is_dir():
            raise self.error_type("TTS evidence path is not a directory")
        return target

    def require_inside(self, root: Path, target: Path) -> None:
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise self.error_type("TTS evidence path escaped repository") from exc

    def load_object(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise self.error_type("TTS evidence JSON is invalid") from exc
        if not isinstance(value, dict):
            raise self.error_type("TTS evidence JSON is invalid")
        return value

    def require_image_digest(self, value: str) -> None:
        if not (
            value.startswith("sha256:")
            and len(value) == 71
            and all(character in "0123456789abcdef" for character in value[7:])
        ):
            raise self.error_type("TTS runtime image digest is invalid")

    def component_evidence(
        self, root: Path, paths: dict[str, Any], metadata: dict[str, tuple[str, str, str]]
    ) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for role in ("base_model", "vocoder", "asr", "xvector"):
            path = paths["components"][role]
            files = self.directory_manifest(path)
            model_id, revision, weight_sha256 = metadata[role]
            evidence.append(
                {
                    "role": role,
                    "model_id": model_id,
                    "revision": revision,
                    "weight_sha256": weight_sha256,
                    "path": path.relative_to(root).as_posix(),
                    "manifest_sha256": legacy_ascii_digest(files),
                    "files": files,
                }
            )
        return evidence


def legacy_ascii_digest(value: object) -> str:
    return sha256(legacy_ascii_canonical(value)).hexdigest()
