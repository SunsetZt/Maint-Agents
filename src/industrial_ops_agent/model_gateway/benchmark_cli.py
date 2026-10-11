"""CLI for runtime benchmark execution and fair report comparison."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from industrial_ops_agent.model_gateway.benchmark import (
    InferenceBenchmarkError,
    InferenceBenchmarkPlan,
    InferenceBenchmarkRunner,
    OpenAiStreamingBenchmarkTransport,
    compare_runtime_reports,
    write_benchmark_document,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-inference-benchmark")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run one immutable runtime benchmark plan")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--api-key", default=os.getenv("IOAP_INFERENCE_BENCHMARK_API_KEY", ""))
    run.add_argument("--allow-plain-http", action="store_true")

    compare = commands.add_parser("compare", help="compare reports with one shared contract")
    compare.add_argument("--report", type=Path, action="append", required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            document = _load_json(args.plan)
            plan = InferenceBenchmarkPlan.model_validate(document)
            report = asyncio.run(
                _run_benchmark(
                    plan,
                    api_key=args.api_key,
                    allow_plain_http=args.allow_plain_http,
                )
            )
            write_benchmark_document(str(args.output), report)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0 if report["status"] == "PASSED" else 2

        reports = [_load_json(path) for path in args.report]
        comparison = compare_runtime_reports(reports)
        write_benchmark_document(str(args.output), comparison)
        print(json.dumps(comparison, ensure_ascii=False, sort_keys=True))
        return 0 if comparison["status"] == "PASSED" else 2
    except (InferenceBenchmarkError, ValidationError, OSError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "reason": _safe_reason(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


async def _run_benchmark(
    plan: InferenceBenchmarkPlan,
    *,
    api_key: str,
    allow_plain_http: bool,
) -> dict[str, Any]:
    async with OpenAiStreamingBenchmarkTransport(
        plan.runtime.endpoint_url,
        api_key,
        timeout_seconds=plan.execution.timeout_seconds,
        allow_plain_http=allow_plain_http,
    ) as transport:
        return await InferenceBenchmarkRunner().run(plan, transport)


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise InferenceBenchmarkError("benchmark_document_must_be_an_object")
    return document


def _safe_reason(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "benchmark_plan_validation_failed"
    if isinstance(exc, json.JSONDecodeError):
        return "benchmark_document_is_not_json"
    if isinstance(exc, OSError):
        return "benchmark_document_io_failed"
    return str(exc)[:255] or "benchmark_command_failed"


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
