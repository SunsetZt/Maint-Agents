"""Small MLflow Tracking REST boundary used by the API and training workers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import httpx


class TrackingUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TrackingRun:
    run_id: str


class ExperimentTracker(Protocol):
    def start_run(
        self,
        *,
        experiment_name: str,
        run_name: str,
        params: dict[str, str],
        tags: dict[str, str],
    ) -> TrackingRun: ...

    def complete_run(self, run_id: str, *, metrics: dict[str, float]) -> None: ...

    def fail_run(self, run_id: str, *, reason_code: str) -> None: ...


class MlflowRestTracker:
    """MLflow v2 REST client without importing the heavyweight training SDK."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 5.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def start_run(
        self,
        *,
        experiment_name: str,
        run_name: str,
        params: dict[str, str],
        tags: dict[str, str],
    ) -> TrackingRun:
        try:
            with httpx.Client(base_url=self._base_url, timeout=self._timeout_seconds) as client:
                experiment_id = self._ensure_experiment(client, experiment_name)
                now_ms = _milliseconds()
                response = client.post(
                    "/api/2.0/mlflow/runs/create",
                    json={
                        "experiment_id": experiment_id,
                        "run_name": run_name,
                        "start_time": now_ms,
                        "tags": [
                            {"key": key, "value": value} for key, value in sorted(tags.items())
                        ],
                    },
                )
                response.raise_for_status()
                run_id = str(response.json()["run"]["info"]["run_id"])
                batch = client.post(
                    "/api/2.0/mlflow/runs/log-batch",
                    json={
                        "run_id": run_id,
                        "metrics": [],
                        "params": [
                            {"key": key, "value": value} for key, value in sorted(params.items())
                        ],
                        "tags": [],
                    },
                )
                batch.raise_for_status()
                return TrackingRun(run_id=run_id)
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise TrackingUnavailable("MLflow tracking request failed") from exc

    def complete_run(self, run_id: str, *, metrics: dict[str, float]) -> None:
        try:
            with httpx.Client(base_url=self._base_url, timeout=self._timeout_seconds) as client:
                now_ms = _milliseconds()
                batch = client.post(
                    "/api/2.0/mlflow/runs/log-batch",
                    json={
                        "run_id": run_id,
                        "metrics": [
                            {"key": key, "value": value, "timestamp": now_ms, "step": 0}
                            for key, value in sorted(metrics.items())
                        ],
                        "params": [],
                        "tags": [],
                    },
                )
                batch.raise_for_status()
                update = client.post(
                    "/api/2.0/mlflow/runs/update",
                    json={"run_id": run_id, "status": "FINISHED", "end_time": now_ms},
                )
                update.raise_for_status()
        except httpx.HTTPError as exc:
            raise TrackingUnavailable("MLflow tracking request failed") from exc

    def fail_run(self, run_id: str, *, reason_code: str) -> None:
        try:
            with httpx.Client(base_url=self._base_url, timeout=self._timeout_seconds) as client:
                now_ms = _milliseconds()
                batch = client.post(
                    "/api/2.0/mlflow/runs/log-batch",
                    json={
                        "run_id": run_id,
                        "metrics": [],
                        "params": [],
                        "tags": [{"key": "ioap.failure_reason", "value": reason_code}],
                    },
                )
                batch.raise_for_status()
                update = client.post(
                    "/api/2.0/mlflow/runs/update",
                    json={"run_id": run_id, "status": "FAILED", "end_time": now_ms},
                )
                update.raise_for_status()
        except httpx.HTTPError as exc:
            raise TrackingUnavailable("MLflow tracking request failed") from exc

    @staticmethod
    def _ensure_experiment(client: httpx.Client, name: str) -> str:
        current = client.get(
            "/api/2.0/mlflow/experiments/get-by-name",
            params={"experiment_name": name},
        )
        if current.status_code == 200:
            return str(current.json()["experiment"]["experiment_id"])
        if current.status_code != 404:
            current.raise_for_status()
        created = client.post(
            "/api/2.0/mlflow/experiments/create",
            json={"name": name},
        )
        if created.status_code == 200:
            return str(created.json()["experiment_id"])
        # A concurrent creator can win between get-by-name and create.
        if created.status_code in {400, 409}:
            current = client.get(
                "/api/2.0/mlflow/experiments/get-by-name",
                params={"experiment_name": name},
            )
            current.raise_for_status()
            return str(current.json()["experiment"]["experiment_id"])
        created.raise_for_status()
        raise TrackingUnavailable("MLflow experiment could not be resolved")


def _milliseconds() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)
