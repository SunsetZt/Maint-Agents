#!/usr/bin/env python3
"""Thin CLI for the typed project-staging business acceptance orchestrator."""

from __future__ import annotations

import argparse
import stat
import sys
from pathlib import Path
from uuid import uuid4

from industrial_ops_agent.project_acceptance import (
    AcceptanceCheckpoint,
    AcceptanceStage,
    ProjectAcceptanceContractError,
    ProjectAcceptanceStateStore,
    build_acceptance_receipt,
    load_project_acceptance_manifest,
)
from industrial_ops_agent.project_acceptance.orchestrator import (
    AcceptanceActor,
    ProjectAcceptanceFailure,
    ProjectAcceptanceOrchestrator,
    UrllibPublicApiTransport,
)


def _regular_private_text(path: Path) -> str:
    try:
        metadata = path.lstat()
    except OSError:
        raise ProjectAcceptanceFailure("actor_token_file_invalid") from None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or metadata.st_size <= 0
        or metadata.st_size > 16_384
    ):
        raise ProjectAcceptanceFailure("actor_token_file_invalid")
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        raise ProjectAcceptanceFailure("actor_token_file_invalid") from None
    if not value or len(value) > 16_384:
        raise ProjectAcceptanceFailure("actor_token_file_invalid")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the governed project-staging public-API business loop."
    )
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-release-id", required=True)
    parser.add_argument("--expected-model-manifest-hash", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--workflow-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    for actor in AcceptanceActor:
        parser.add_argument(
            f"--{actor.value.replace('_', '-')}-token-file",
            type=Path,
            required=True,
        )
    return parser


def main() -> int:
    args = _parser().parse_args()
    run_id = args.run_id or uuid4().hex
    try:
        validated = load_project_acceptance_manifest(
            args.manifest,
            repository_root=args.repository_root,
        )
        state_store = ProjectAcceptanceStateStore(
            repository_root=args.repository_root,
            state_root=args.state_root,
        )
        prior_checkpoint = state_store.load_checkpoint(run_id)
        tokens = {
            actor: _regular_private_text(getattr(args, f"{actor.value}_token_file"))
            for actor in AcceptanceActor
        }
        transport = UrllibPublicApiTransport(
            args.api_url,
            actor_tokens=tokens,
            timeout_seconds=args.request_timeout_seconds,
        )

        def progress(stage: AcceptanceStage, facts: object) -> None:
            del facts
            print(f"PROJECT_BUSINESS_STAGE_OK stage={stage.value}", flush=True)

        def persist(checkpoint: AcceptanceCheckpoint) -> None:
            state_store.write_checkpoint(checkpoint)

        orchestrator = ProjectAcceptanceOrchestrator(
            transport=transport,
            manifest=validated,
            run_id=run_id,
            expected_release_id=args.expected_release_id,
            expected_model_manifest_hash=args.expected_model_manifest_hash,
            poll_seconds=args.poll_seconds,
            workflow_timeout_seconds=args.workflow_timeout_seconds,
            progress=progress,
            checkpoint_sink=persist,
        )
        checkpoint = orchestrator.run(checkpoint=prior_checkpoint)
        receipt = build_acceptance_receipt(checkpoint)
        orchestrator.reconcile_receipt(receipt)
        receipt_path = state_store.write_receipt(receipt)
    except (ProjectAcceptanceContractError, ProjectAcceptanceFailure) as exc:
        print(f"Project business orchestration failed: {exc}", file=sys.stderr)
        return 3
    print(
        "PROJECT_BUSINESS_ORCHESTRATION_COMPLETE "
        f"run_id={checkpoint.run_id} "
        f"incident_id={checkpoint.facts['incident_id']} "
        f"work_order_id={checkpoint.facts['work_order_id']}",
        flush=True,
    )
    print(
        "PROJECT_STAGING_BUSINESS_ACCEPTED "
        f"run_id={checkpoint.run_id} "
        f"receipt={receipt_path.relative_to(args.repository_root.resolve()).as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
