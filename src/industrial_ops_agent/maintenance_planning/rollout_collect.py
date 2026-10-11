"""Explicit, read-only Compose/Kubernetes reader inventories for an independent signer.

No automatic platform detection, deployment, signing, activation, or generated test evidence.
Writes a new file exclusively; original before/after observations are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from industrial_ops_agent.maintenance_planning.review_isolation import (
    READER_COVERAGE_DIGEST,
    REVIEW_POLICY_VERSION,
)
from industrial_ops_agent.maintenance_planning.rollout_evidence import (
    MAX_RECEIPT_BYTES,
    ReaderInstance,
    ReaderInventory,
    ReaderRolloutReceipt,
    ReaderScope,
)

REVISION_LABEL = "org.opencontainers.image.revision"
COVERAGE_LABEL = "io.industrial-ops.review-coverage"


def _json(argv: list[str]) -> Any:
    result = subprocess.run(argv, check=True, capture_output=True, timeout=30)  # noqa: S603
    if len(result.stdout) > 8_388_608:
        raise ValueError("runtime inventory exceeds bounded collection size")
    return json.loads(result.stdout)


def _compose(
    context: str, scope: str, workloads: dict[str, int]
) -> tuple[str, list[ReaderInstance]]:
    base = ["docker", "--context", context]
    daemon_id = _json([*base, "info", "--format", "{{json .ID}}"])
    if not isinstance(daemon_id, str) or not daemon_id:
        raise ValueError("daemon identity unavailable")
    # List all containers, including stopped/draining instances; filter running below.
    result = subprocess.run(
        [
            *base,
            "container",
            "ls",
            "--all",
            "--no-trunc",
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={scope}",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )  # noqa: S603
    if len(result.stdout) > 262_144:
        raise ValueError("too many containers")
    ids = result.stdout.decode("ascii").splitlines()
    if not ids:
        raise ValueError("empty reader inventory")
    instances = []
    for container_id in ids:
        container = _json([*base, "container", "inspect", container_id])[0]
        labels = container["Config"].get("Labels") or {}
        workload = labels.get("com.docker.compose.service")
        if workload not in workloads or not container["State"]["Running"]:
            continue
        image_id = container["Image"]
        image = _json([*base, "image", "inspect", image_id])[0]
        labels = image["Config"].get("Labels") or {}
        instances.append(
            ReaderInstance(
                instance_id=container["Id"],
                workload=workload,
                image_digest=image_id,
                git_revision=labels.get(REVISION_LABEL, ""),
                reader_coverage_digest=labels.get(COVERAGE_LABEL, ""),
                ready=(
                    container["State"].get("Health", {}).get("Status") == "healthy"
                    and not container["State"].get("Restarting", False)
                    and not container["State"].get("Paused", False)
                ),
            )
        )
    return daemon_id, instances


def _kubernetes(
    context: str, scope: str, workloads: dict[str, int], after: bool
) -> tuple[str, list[ReaderInstance]]:
    base = ["kubectl", "--context", context, "--request-timeout=25s"]
    identity = _json([*base, "get", "namespace", "kube-system", "-o", "json"])["metadata"]["uid"]
    pods = _json([*base, "-n", scope, "get", "pods", "-o", "json"])["items"]
    instances = []
    for workload, expected in workloads.items():
        deployment = _json([*base, "-n", scope, "get", "deployment", workload, "-o", "json"])
        selector = deployment["spec"]["selector"]
        if not selector.get("matchLabels") or selector.get("matchExpressions"):
            raise ValueError("reader deployment requires an explicit matchLabels selector")
        status = deployment.get("status", {})
        if after and (
            deployment["spec"].get("replicas", 1) != expected
            or status.get("observedGeneration") != deployment["metadata"]["generation"]
            or any(
                status.get(key) != expected
                for key in ("replicas", "updatedReplicas", "readyReplicas", "availableReplicas")
            )
        ):
            raise ValueError("reader deployment has not converged")
        containers = deployment["spec"]["template"]["spec"]["containers"]
        if len(containers) != 1:
            raise ValueError("collector requires a single explicitly identified reader container")
        name = containers[0]["name"]
        for pod in pods:
            metadata = pod["metadata"]
            if not all(
                metadata.get("labels", {}).get(k) == v for k, v in selector["matchLabels"].items()
            ):
                continue
            if pod.get("status", {}).get("phase") in {"Succeeded", "Failed"}:
                continue
            states = [
                s for s in pod.get("status", {}).get("containerStatuses", []) if s["name"] == name
            ]
            if len(states) != 1 or not states[0].get("imageID"):
                raise ValueError("reader image identity is unavailable")
            state = states[0]
            image = state["imageID"]
            if "sha256:" not in image:
                raise ValueError("reader image has no immutable digest")
            digest = "sha256:" + image.rsplit("sha256:", 1)[1]
            annotations = metadata.get("annotations", {})
            instances.append(
                ReaderInstance(
                    instance_id=metadata["uid"],
                    workload=workload,
                    image_digest=digest,
                    git_revision=annotations.get(REVISION_LABEL, ""),
                    reader_coverage_digest=annotations.get(COVERAGE_LABEL, ""),
                    ready=(
                        not metadata.get("deletionTimestamp")
                        and state.get("ready") is True
                        and any(
                            c.get("type") == "Ready" and c.get("status") == "True"
                            for c in pod.get("status", {}).get("conditions", [])
                        )
                    ),
                )
            )
    return str(identity), instances


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("before", "after"))
    parser.add_argument("--runtime", choices=("compose", "kubernetes"), required=True)
    parser.add_argument("--context", required=True, help="Explicit Docker context or kube context")
    parser.add_argument("--scope", required=True, help="Compose project or Kubernetes namespace")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument(
        "--workload", action="append", required=True, help="Reader service/deployment=count"
    )
    parser.add_argument("--before", type=Path)
    parser.add_argument("--git-revision")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == "after" and not (args.before and args.git_revision):
        parser.error("after requires --before and --git-revision")
    try:
        pairs = [value.rsplit("=", 1) for value in args.workload]
        workloads = {name: int(count) for name, count in pairs}
        if len(workloads) != len(pairs):
            raise ValueError("duplicate reader workload")
        # Validate CLI scope/counts before passing identifiers to a runtime command.
        header = ReaderScope(
            tenant_id=args.tenant,
            environment_id=args.environment,
            runtime=args.runtime,
            runtime_identity="pending",
            scope=args.scope,
            workloads=workloads,
        ).model_dump()
        runtime_id, instances = (
            _compose(args.context, args.scope, workloads)
            if args.runtime == "compose"
            else _kubernetes(args.context, args.scope, workloads, args.phase == "after")
        )
        inventory = ReaderInventory(
            **{
                **header,
                "runtime_identity": runtime_id,
                "observed_at": datetime.now(UTC),
                "instances": instances,
            }
        )
        output: ReaderInventory | ReaderRolloutReceipt = inventory
        if args.phase == "after":
            before = ReaderInventory.model_validate_json(_read(args.before, MAX_RECEIPT_BYTES))
            output = ReaderRolloutReceipt(
                schema_version="maintenance-review-rollout-v2",
                policy_version=REVIEW_POLICY_VERSION,
                reader_coverage_digest=READER_COVERAGE_DIGEST,
                git_revision=args.git_revision,
                before=before,
                after=inventory,
            )
            if before.model_dump(exclude={"observed_at", "instances"}) != inventory.model_dump(
                exclude={"observed_at", "instances"}
            ):
                raise ValueError("before and after inventories belong to different scopes")
        encoded = output.model_dump_json(indent=2).encode("utf-8")
        if len(encoded) > MAX_RECEIPT_BYTES:
            raise ValueError("receipt exceeds upload limit")
        with args.output.open("xb") as stream:
            stream.write(encoded)
    except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError):
        # Do not print runtime stderr, credentials, raw inspect documents or command environments.
        print(
            "Reader inventory collection failed; no deployment or signing was performed.",
            file=sys.stderr,
        )
        return 1
    print(
        "Reader inventory saved; administrator confirmation is required before activation."
    )
    return 0


def _read(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("input exceeds collection limit")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
