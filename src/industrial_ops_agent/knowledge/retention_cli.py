"""Invoke the authorized expiry sweep from Airflow without exposing its bearer token."""

from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def run() -> None:
    parser = argparse.ArgumentParser(description="Run the governed knowledge expiry sweep")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--reason",
        default="Scheduled tenant retention policy expiry sweep",
    )
    args = parser.parse_args()
    token = os.environ.get("IOAP_RETENTION_SERVICE_TOKEN", "").strip()
    if not token:
        raise SystemExit("IOAP_RETENTION_SERVICE_TOKEN is required")
    api_url = os.environ.get("IOAP_API_URL", "http://api:8000/api/v1").rstrip("/")
    payload = json.dumps(
        {"reason": args.reason, "limit": args.limit},
        separators=(",", ":"),
    ).encode()
    request = Request(  # noqa: S310 - URL is an operator-owned deployment setting
        f"{api_url}/knowledge/deletions/expiry-sweep",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=60) as response:  # noqa: S310
            result = json.loads(response.read())
    except HTTPError as exc:
        sys.stderr.write(f"knowledge expiry sweep rejected with HTTP {exc.code}\n")
        raise SystemExit(1) from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        sys.stderr.write("knowledge expiry sweep dependency failed\n")
        raise SystemExit(1) from exc
    records = result.get("data", [])
    print(
        json.dumps(
            {
                "requested": len(records),
                "deletion_ids": [record.get("deletion_id") for record in records],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
