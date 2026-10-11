"""Historical TTS source identities; current model and data bindings remain strict."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

# Only these exact retired source identities can be represented without executable
# source files. Dataset, model, report, approval and runtime bindings remain live checks.
_RETIRED_SOURCES = MappingProxyType(
    {
        "src/industrial_ops_agent/simulation/tts_enterprise_value_lab.py": (
            61900,
            "d31e678afe9f105062de91be367e7955d646dcd1530f9770eeaa8bd9e128bd84",
        ),
        "src/industrial_ops_agent/simulation/tts_recovery_value_lab.py": (
            84283,
            "a79d40e935d85f8f0b7d351d25bc0677ab038eec2dc9f0c3fabc4f97946ecbf0",
        ),
        "src/industrial_ops_agent/simulation/tts_calibrated_value_lab.py": (
            88996,
            "b0587ba8fb048871896c78abce3abead85ff464ac763c05b8b3d1279b4ec8751",
        ),
        "src/industrial_ops_agent/simulation/tts_final_value_lab.py": (
            98355,
            "8648f9bb80a20494ce831347461b71aa75e46dcb13e885149834b0738a001deb",
        ),
        "src/industrial_ops_agent/simulation/tts_latency_value_lab.py": (
            105805,
            "dd41768c19afd46cdabf991ba61d843c0b1d1acc8f6af2fe22e9a1a8a25e61b5",
        ),
        "src/industrial_ops_agent/simulation/tts_strong_asr_value_lab.py": (
            116634,
            "3282a1f53abbd5e07d89268c5c8f0d39e9ab22284659720f795328ba2f71c3a5",
        ),
    }
)


def retired_source_binding(
    repo_root: Path, relative_path: str, *, error_type: type[RuntimeError] = RuntimeError
) -> dict[str, Any]:
    expected = _RETIRED_SOURCES.get(relative_path)
    if expected is None:
        raise error_type("unrecognized retired experiment source")
    root = repo_root.resolve(strict=True)
    path = root / relative_path
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise error_type("retired experiment source escaped the repository")
    size_bytes, digest = expected
    # Restored executable files must match their original source identities.
    # Missing files are allowed only for this closed set of explicitly retired sources.
    if path.exists():
        if not path.is_file():
            raise error_type("retired experiment source changed")
        actual = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                actual.update(chunk)
        observed = (path.stat().st_size, actual.hexdigest())
        if observed != expected:
            raise error_type("retired experiment source changed")
    return {"path": relative_path, "size_bytes": size_bytes, "sha256": digest}


# This is the source identity recorded before the import-only migration. It is
# not an accepted digest for a new runtime release, nor a blanket source exemption.
_TTS_RUNTIME_PATH = "src/industrial_ops_agent/multimodal/tts_runtime.py"
_PRE_MIGRATION_TTS_RUNTIME = (
    18667,
    "9fd3f3927c7eab4fe38719c310c6125facd24f005e16a97d5dd882e9def271ff",
)


def verify_tts_runtime_source(
    repo_root: Path,
    *,
    path: str,
    size_bytes: int,
    digest: str,
    error_type: type[RuntimeError] = RuntimeError,
) -> None:
    """Verify a live binding or the exact pre-migration TTS source identity.

    Historical receipts keep their original bytes and digest. Only reversal of
    the single evidence-reader import is permitted; any other edit fails. New
    receipts must still be created from the actual source via file_binding().
    """
    root = repo_root.resolve(strict=True)
    if path != _TTS_RUNTIME_PATH:
        raise error_type("tts_runtime_source_path_changed")
    source = root / path
    if source.is_symlink() or not source.resolve().is_relative_to(root):
        raise error_type("tts_runtime_source_escaped_repository")
    if not source.is_file():
        raise error_type("tts_runtime_source_missing")
    content = source.read_bytes()
    expected = (size_bytes, digest)
    if (len(content), sha256(content).hexdigest()) == expected:
        return
    migrated_import = b"    from industrial_ops_agent.model_evidence.tts_base import ("
    if expected == _PRE_MIGRATION_TTS_RUNTIME and content.count(migrated_import) == 1:
        original = content.replace(
            migrated_import,
            b"    from industrial_ops_agent.simulation.tts_enterprise_value_lab import (",
            1,
        )
        if (len(original), sha256(original).hexdigest()) == expected:
            return
    raise error_type("tts_runtime_source_changed")
