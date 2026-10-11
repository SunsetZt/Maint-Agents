"""Small local-only control client for the supplier A2A sandbox."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any
from urllib.request import Request, urlopen


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-supplier-sandbox")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    commands.add_parser("reset")
    commands.add_parser("clear-fault")
    fault = commands.add_parser("fault")
    fault.add_argument("mode")
    fault.add_argument("remaining", type=int)
    return parser


def run(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    base_url = os.environ.get(
        "IOAP_SUPPLIER_SANDBOX_BASE_URL",
        "http://127.0.0.1:8092",
    ).rstrip("/")
    token = os.environ.get("IOAP_SANDBOX_CONTROL_TOKEN", "")
    if not token:
        raise SystemExit("supplier sandbox control token is not configured")
    method, path, body = _request_arguments(args)
    raw = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        f"{base_url}{path}",
        data=raw,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urlopen(request, timeout=5) as response:
        document = json.loads(response.read())
    print(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2))


def _request_arguments(args: argparse.Namespace) -> tuple[str, str, dict[str, Any] | None]:
    if args.command == "status":
        return "GET", "/supplier-sandbox/v1/state", None
    if args.command == "reset":
        return "POST", "/supplier-sandbox/v1/reset", None
    if args.command == "clear-fault":
        return "DELETE", "/supplier-sandbox/v1/fault", None
    if args.command == "fault":
        return "PUT", "/supplier-sandbox/v1/fault", {
            "mode": str(args.mode).upper(),
            "remaining": args.remaining,
        }
    raise SystemExit("unsupported supplier sandbox command")


if __name__ == "__main__":
    run()
