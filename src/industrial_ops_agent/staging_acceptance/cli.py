from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from industrial_ops_agent.staging_acceptance.collector import StagingAcceptanceCollector
from industrial_ops_agent.staging_acceptance.contracts import (
    StagingAcceptancePlan,
    StagingAcceptanceReport,
)
from industrial_ops_agent.staging_acceptance.dynamic_contracts import (
    DynamicAcceptanceResults,
)
from industrial_ops_agent.staging_acceptance.dynamic_manifest import (
    build_dynamic_acceptance_manifest,
)
from industrial_ops_agent.staging_acceptance.runner import (
    RunnerInputError,
    StagingDynamicRunner,
)
from industrial_ops_agent.staging_acceptance.runner_contracts import (
    StagingDynamicExecutionPlan,
)

MAX_PLAN_BYTES = 64 * 1024
MAX_EVIDENCE_BYTES = 1024 * 1024
MAX_EXECUTION_PLAN_BYTES = 128 * 1024
ModelT = TypeVar("ModelT", bound=BaseModel)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-staging-acceptance")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser(
        "preflight",
        help="collect a read-only Staging control-plane preflight evidence bundle",
    )
    preflight.add_argument("--plan", required=True, type=Path)
    preflight.add_argument("--output", required=True, type=Path)
    aggregate = commands.add_parser(
        "aggregate",
        help="aggregate static and dynamic Staging evidence into a review manifest",
    )
    aggregate.add_argument("--preflight", required=True, type=Path)
    aggregate.add_argument("--results", required=True, type=Path)
    aggregate.add_argument("--output", required=True, type=Path)
    dynamic_run = commands.add_parser(
        "run",
        help="run authorized dynamic Staging scenarios using a digest-pinned driver",
    )
    dynamic_run.add_argument("--preflight", required=True, type=Path)
    dynamic_run.add_argument("--execution-plan", required=True, type=Path)
    dynamic_run.add_argument("--confirm-context", required=True)
    dynamic_run.add_argument("--evidence-dir", required=True, type=Path)
    dynamic_run.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        return _run_dynamic_scenarios(
            args.preflight,
            args.execution_plan,
            args.confirm_context,
            args.evidence_dir,
            args.output,
        )
    if args.command == "aggregate":
        return _run_aggregate(args.preflight, args.results, args.output)
    return _run_preflight(args.plan, args.output)


def _run_dynamic_scenarios(
    preflight_path: Path,
    execution_plan_path: Path,
    confirm_context: str,
    evidence_dir: Path,
    output: Path,
) -> int:
    try:
        preflight, _ = _load_model(
            preflight_path,
            StagingAcceptanceReport,
            max_bytes=MAX_EVIDENCE_BYTES,
        )
        execution_plan, _ = _load_model(
            execution_plan_path,
            StagingDynamicExecutionPlan,
            max_bytes=MAX_EXECUTION_PLAN_BYTES,
        )
    except (OSError, UnicodeError, ValueError, ValidationError, json.JSONDecodeError):
        _error("invalid_execution_plan")
        return 64
    if (
        output.exists()
        or output.is_symlink()
        or evidence_dir.exists()
        or evidence_dir.is_symlink()
    ):
        _error("output_exists")
        return 73
    if not _has_safe_existing_parent(output) or not _has_safe_existing_parent(
        evidence_dir
    ):
        _error("output_write_failed")
        return 73
    try:
        results = StagingDynamicRunner().run(
            preflight,
            execution_plan,
            confirm_context=confirm_context,
            evidence_root=evidence_dir,
        )
    except RunnerInputError:
        _error("invalid_execution_plan")
        return 64
    except FileExistsError:
        _error("output_exists")
        return 73
    except OSError:
        _error("output_write_failed")
        return 73
    try:
        _write_json_exclusive(output, results.model_dump(mode="json"))
    except FileExistsError:
        _error("output_exists")
        return 73
    except OSError:
        _error("output_write_failed")
        return 73
    complete = len(results.scenarios) == 15 and all(
        item.status == "PASSED" and all(assertion.passed for assertion in item.assertions)
        for item in results.scenarios
    )
    status = "PASSED" if complete else "BLOCKED"
    print(
        json.dumps(
            {
                "run_id": execution_plan.run_id,
                "status": status,
                "scenario_count": len(results.scenarios),
                "evidence_dir": str(evidence_dir),
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if complete else 2


def _run_preflight(plan_path: Path, output: Path) -> int:
    try:
        plan = _load_plan(plan_path)
    except (OSError, UnicodeError, ValueError, ValidationError, json.JSONDecodeError):
        _error("invalid_plan")
        return 64
    if output.exists() or output.is_symlink():
        _error("output_exists")
        return 73
    report = StagingAcceptanceCollector().collect(plan)
    try:
        _write_json_exclusive(output, report.model_dump(mode="json"))
    except FileExistsError:
        _error("output_exists")
        return 73
    except OSError:
        _error("output_write_failed")
        return 73
    print(
        json.dumps(
            {
                "report_id": report.report_id,
                "status": report.status,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 2 if report.status == "BLOCKED" else 0


def _run_aggregate(preflight_path: Path, results_path: Path, output: Path) -> int:
    try:
        preflight, preflight_bytes = _load_model(
            preflight_path,
            StagingAcceptanceReport,
            max_bytes=MAX_EVIDENCE_BYTES,
        )
        results, _ = _load_model(
            results_path,
            DynamicAcceptanceResults,
            max_bytes=MAX_EVIDENCE_BYTES,
        )
    except (OSError, UnicodeError, ValueError, ValidationError, json.JSONDecodeError):
        _error("invalid_acceptance_evidence")
        return 64
    if output.exists() or output.is_symlink():
        _error("output_exists")
        return 73
    try:
        manifest = build_dynamic_acceptance_manifest(
            preflight,
            results,
            preflight_artifact_digest=_digest(preflight_bytes),
        )
    except (TypeError, ValueError, ValidationError):
        _error("invalid_acceptance_evidence")
        return 64
    try:
        _write_json_exclusive(output, manifest.model_dump(mode="json"))
    except FileExistsError:
        _error("output_exists")
        return 73
    except OSError:
        _error("output_write_failed")
        return 73
    print(
        json.dumps(
            {
                "manifest_id": manifest.manifest_id,
                "status": manifest.status,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 2 if manifest.status == "BLOCKED" else 0


def _load_plan(path: Path) -> StagingAcceptancePlan:
    try:
        plan, _ = _load_model(path, StagingAcceptancePlan, max_bytes=MAX_PLAN_BYTES)
        return plan
    except (UnicodeError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        if isinstance(exc, ValueError) and "size" in str(exc):
            raise ValueError("staging acceptance plan size is invalid") from exc
        raise ValueError("invalid staging acceptance plan") from exc


def _load_model(
    path: Path,
    model_type: type[ModelT],
    *,
    max_bytes: int,
) -> tuple[ModelT, bytes]:
    raw = _read_bounded_regular_file(path, max_bytes=max_bytes)
    payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_closed_object)
    if not isinstance(payload, dict):
        raise ValueError("JSON evidence must be an object")
    return model_type.model_validate(payload), raw


def _read_bounded_regular_file(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb", closefd=True) as stream:
        metadata = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > max_bytes
        ):
            raise ValueError("evidence file size is invalid")
        raw = stream.read(max_bytes + 1)
    if not raw or len(raw) > max_bytes:
        raise ValueError("evidence file size is invalid")
    return raw


def _closed_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    parent = path.parent
    parent_metadata = parent.lstat()
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise OSError("output parent is not a directory")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = parent / f".{path.name}.tmp-{uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _has_safe_existing_parent(path: Path) -> bool:
    try:
        metadata = path.parent.lstat()
    except OSError:
        return False
    return stat.S_ISDIR(metadata.st_mode)


def _error(code: str) -> None:
    print(json.dumps({"error": code}, sort_keys=True), file=sys.stderr)


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
