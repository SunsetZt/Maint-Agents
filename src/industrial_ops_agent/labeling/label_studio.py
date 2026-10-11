"""Typed Label Studio API adapter that never logs tokens or response bodies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from industrial_ops_agent.secrets import (
    SecretName,
    SecretProvider,
    SecretReadError,
    SecretValue,
)


@dataclass(frozen=True, slots=True)
class LabelAnnotation:
    external_annotation_id: str
    external_version: str
    reviewer_subject_id: str
    labels: dict[str, Any]
    submitted_at: datetime


@dataclass(frozen=True, slots=True)
class LabelTaskCreated:
    external_task_id: str
    external_version: str
    payload_hash: str


@dataclass(frozen=True, slots=True)
class LabelTaskSnapshot:
    external_task_id: str
    external_version: str
    payload_hash: str
    annotations: tuple[LabelAnnotation, ...]


class LabelStudioAdapter(Protocol):
    def create_task(
        self,
        *,
        project_id: str,
        data: dict[str, Any],
        source_data_version: str,
        idempotency_key: str,
    ) -> LabelTaskCreated: ...

    def fetch_task(self, *, external_task_id: str) -> LabelTaskSnapshot: ...


class LabelStudioUnavailable(RuntimeError):
    pass


class LabelStudioHttpAdapter:
    def __init__(
        self,
        *,
        base_url: str,
        api_token: SecretValue,
        timeout_seconds: float = 5.0,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.hostname in {"label-studio", "localhost"}
        ):
            raise ValueError("Label Studio must use HTTPS outside the local integration network")
        if not parsed.netloc:
            raise ValueError("Label Studio base URL is invalid")
        self._base_url = base_url.rstrip("/")
        self._api_token = api_token
        self._timeout_seconds = timeout_seconds

    def create_task(
        self,
        *,
        project_id: str,
        data: dict[str, Any],
        source_data_version: str,
        idempotency_key: str,
    ) -> LabelTaskCreated:
        payload = {
            "project": _project_number(project_id),
            "data": data,
            "meta": {"source_data_version": source_data_version},
        }
        response = self._request(
            "POST",
            "/api/tasks/",
            payload,
            idempotency_key=idempotency_key,
        )
        return LabelTaskCreated(
            external_task_id=str(_required(response, "id")),
            external_version=str(_required(response, "updated_at")),
            payload_hash=_payload_hash(_object(response.get("data", data))),
        )

    def fetch_task(self, *, external_task_id: str) -> LabelTaskSnapshot:
        response = self._request("GET", f"/api/tasks/{external_task_id}/", None)
        annotations = tuple(
            _annotation(value)
            for value in response.get("annotations", [])
            if isinstance(value, dict) and not value.get("was_cancelled", False)
        )
        data = _object(_required(response, "data"))
        return LabelTaskSnapshot(
            external_task_id=str(_required(response, "id")),
            external_version=str(_required(response, "updated_at")),
            payload_hash=_payload_hash(data),
            annotations=annotations,
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Token {self._api_token.reveal()}",
        }
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload, ensure_ascii=False).encode()
        request = Request(  # noqa: S310 - URL scheme is validated in __init__
            self._base_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310
                value = json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LabelStudioUnavailable("label_studio_unavailable") from exc
        if not isinstance(value, dict):
            raise LabelStudioUnavailable("label_studio_invalid_response")
        return value


class LazyLabelStudioAdapter:
    """Resolve the Label Studio token only when the offline integration is used."""

    def __init__(
        self,
        *,
        base_url: str,
        secret_provider: SecretProvider,
        project_key: str,
        project_id: int,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._base_url = base_url
        self._secret_provider = secret_provider
        self._project_key = project_key
        self._project_id = project_id
        self._timeout_seconds = timeout_seconds

    def create_task(
        self,
        *,
        project_id: str,
        data: dict[str, Any],
        source_data_version: str,
        idempotency_key: str,
    ) -> LabelTaskCreated:
        if project_id != self._project_key:
            raise ValueError("Label Studio project is not approved")
        return self._adapter().create_task(
            project_id=str(self._project_id),
            data=data,
            source_data_version=source_data_version,
            idempotency_key=idempotency_key,
        )

    def fetch_task(self, *, external_task_id: str) -> LabelTaskSnapshot:
        return self._adapter().fetch_task(external_task_id=external_task_id)

    def _adapter(self) -> LabelStudioHttpAdapter:
        try:
            token = self._secret_provider.get(SecretName.LABEL_STUDIO_API_TOKEN)
        except SecretReadError as exc:
            raise LabelStudioUnavailable("label_studio_unavailable") from exc
        return LabelStudioHttpAdapter(
            base_url=self._base_url,
            api_token=token,
            timeout_seconds=self._timeout_seconds,
        )


def _annotation(value: dict[str, Any]) -> LabelAnnotation:
    reviewer = value.get("completed_by")
    if isinstance(reviewer, dict):
        reviewer = reviewer.get("id")
    if reviewer is None:
        raise LabelStudioUnavailable("label_studio_annotation_has_no_reviewer")
    updated_at = _timestamp(value.get("updated_at") or value.get("created_at"))
    annotation_id = str(_required(value, "id"))
    return LabelAnnotation(
        external_annotation_id=annotation_id,
        external_version=str(value.get("updated_at") or value.get("created_at") or annotation_id),
        reviewer_subject_id=f"label-studio-user-{reviewer}",
        labels={"result": value.get("result", [])},
        submitted_at=updated_at,
    )


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise LabelStudioUnavailable("label_studio_timestamp_missing")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _required(value: dict[str, Any], name: str) -> Any:
    result = value.get(name)
    if result is None:
        raise LabelStudioUnavailable("label_studio_invalid_response")
    return result


def _object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LabelStudioUnavailable("label_studio_invalid_response")
    return value


def _project_number(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError("Label Studio HTTP project_id must be numeric") from exc


def _payload_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + sha256(encoded.encode()).hexdigest()
