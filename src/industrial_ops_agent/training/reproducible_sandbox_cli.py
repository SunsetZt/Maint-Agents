from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from industrial_ops_agent.experiments.mlflow import ExperimentTracker, MlflowRestTracker
from industrial_ops_agent.training.reproducible_sandbox import (
    MARKER,
    RuntimeIdentity,
    SandboxBackend,
    SandboxReport,
    TransformersPeftSandboxBackend,
    run_sandbox,
)


def execute(
    *,
    dataset_root: Path,
    output_directory: Path,
    backend: SandboxBackend,
    tracker: ExperimentTracker,
    runtime: RuntimeIdentity,
) -> SandboxReport:
    """Run the approved core through a CLI-owned production consumer boundary."""

    return run_sandbox(
        dataset_root=dataset_root,
        output_directory=output_directory,
        backend=backend,
        tracker=tracker,
        runtime=runtime,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model-training-sandbox")
    parser.add_argument("--contract-probe", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--mlflow-url")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.contract_probe:
        print("MODEL_TRAINING_SANDBOX_CLI_READY")
        return 0
    if args.dataset_root is None or args.output_directory is None or not args.mlflow_url:
        raise SystemExit("dataset-root, output-directory and mlflow-url are required")
    image_digest = os.getenv("IOAP_SANDBOX_IMAGE_DIGEST", "")
    report = execute(
        dataset_root=args.dataset_root,
        output_directory=args.output_directory,
        backend=TransformersPeftSandboxBackend(),
        tracker=MlflowRestTracker(args.mlflow_url, timeout_seconds=10.0),
        runtime=RuntimeIdentity(image_digest=image_digest),
    )
    print(
        json.dumps(
            {
                "marker": MARKER,
                "decision": report.decision,
                "selected_method": report.selected_method,
                "mlflow_run_ids": dict(report.mlflow_run_ids),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report.decision == "CANDIDATE_ELIGIBLE" else 4


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
