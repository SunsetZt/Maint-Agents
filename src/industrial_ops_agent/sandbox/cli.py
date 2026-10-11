from __future__ import annotations

import argparse
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="enterprise-sandbox")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    commands.add_parser("reset")
    fault = commands.add_parser("fault")
    fault.add_argument("provider")
    fault.add_argument("mode", choices=("UNAVAILABLE", "REJECT", "OUTCOME_UNKNOWN", "STALE"))
    fault.add_argument("remaining", type=int)
    commands.add_parser("clear-fault").add_argument("provider")
    complete = commands.add_parser("complete")
    for name in ("provider", "operation_id", "status"):
        complete.add_argument(name)
    complete.add_argument("--payment-status")
    complete.add_argument("--reason")
    fulfill = commands.add_parser("fulfill")
    fulfill.add_argument("operation_id")
    fulfill.add_argument("status")
    fulfill.add_argument("received_quantity", type=int)
    fulfill.add_argument("--reason")
    commands.add_parser("verify")
    commands.add_parser("acceptance")
    return parser


def _request(method: str, path: str, payload: dict[str, object] | None) -> object:
    token = os.environ.get("IOAP_SANDBOX_CONTROL_TOKEN", "")
    if not token:
        raise RuntimeError("sandbox control credential is not configured")
    base_url = os.environ.get("IOAP_SANDBOX_BASE_URL", "http://127.0.0.1:8090").rstrip("/")
    raw = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(  # noqa: S310 - the container-local endpoint is operator configured
        f"{base_url}{path}",
        data=raw,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urlopen(request, timeout=10) as response:  # noqa: S310
            return json.loads(response.read())
    except HTTPError as exc:
        raise RuntimeError(f"sandbox control request rejected with HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("sandbox control endpoint unavailable") from exc


def run(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    command: str = args.command
    if command == "acceptance":
        from industrial_ops_agent.sandbox.acceptance import (
            SandboxAcceptanceCredentials,
            execute_sandbox_acceptance,
        )

        result = execute_sandbox_acceptance(
            os.environ.get(
                "IOAP_SANDBOX_BASE_URL",
                "http://127.0.0.1:8090",
            ),
            SandboxAcceptanceCredentials.from_environment(),
        ).model_dump(mode="json")
    elif command in {"status", "verify"}:
        result = _request("GET", "/sandbox/v1/state", None)
    elif command == "reset":
        result = _request("POST", "/sandbox/v1/reset", {})
    elif command == "fault":
        result = _request(
            "PUT",
            f"/sandbox/v1/faults/{args.provider}",
            {"mode": args.mode, "remaining": args.remaining},
        )
    elif command == "clear-fault":
        result = _request("DELETE", f"/sandbox/v1/faults/{args.provider}", None)
    elif command == "complete":
        payload: dict[str, object] = {
            "status": args.status,
            "payment_status": args.payment_status,
            "reason": args.reason,
        }
        result = _request(
            "POST", f"/sandbox/v1/completions/{args.provider}/{args.operation_id}", payload
        )
    else:
        payload = {
            "status": args.status,
            "received_quantity": args.received_quantity,
            "reason": args.reason,
        }
        result = _request(
            "POST", f"/sandbox/v1/procurement-fulfillments/{args.operation_id}", payload
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    run()
