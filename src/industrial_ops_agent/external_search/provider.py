"""Tavily Search API boundary with a closed request and response contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Any, Protocol

import httpx

from industrial_ops_agent.secrets import SecretValue


class ExternalSearchProviderError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ExternalSearchProviderItem:
    title: str
    url: str
    content: str
    score: float
    published_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ExternalSearchProviderResult:
    request_id: str | None
    usage_credits: int | None
    items: tuple[ExternalSearchProviderItem, ...]


class ExternalSearchProvider(Protocol):
    provider_name: str

    def search(
        self,
        query: str,
        *,
        include_domains: tuple[str, ...],
        max_results: int,
    ) -> ExternalSearchProviderResult: ...


class TavilyExternalSearchProvider:
    """Use only Tavily summaries; generated answers and raw page content stay disabled."""

    provider_name = "TAVILY"

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: SecretValue,
        timeout_seconds: float,
        project_id: str | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._project_id = project_id

    def search(
        self,
        query: str,
        *,
        include_domains: tuple[str, ...],
        max_results: int,
    ) -> ExternalSearchProviderResult:
        headers = {
            "Authorization": f"Bearer {self._api_key.reveal()}",
            "Content-Type": "application/json",
        }
        if self._project_id is not None:
            headers["X-Project-ID"] = self._project_id
        payload = {
            "query": query,
            "search_depth": "basic",
            "topic": "general",
            "max_results": max_results,
            "include_domains": list(include_domains),
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "auto_parameters": False,
        }
        try:
            with httpx.Client(timeout=self._timeout_seconds, follow_redirects=False) as client:
                response = client.post(self._endpoint, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise ExternalSearchProviderError("external_search_timeout") from exc
        except httpx.HTTPError as exc:
            raise ExternalSearchProviderError("external_search_provider_unavailable") from exc
        if response.status_code == 429:
            raise ExternalSearchProviderError("external_search_rate_limited")
        if response.status_code in {401, 403}:
            raise ExternalSearchProviderError("external_search_provider_auth_failed")
        if response.status_code != 200:
            raise ExternalSearchProviderError("external_search_provider_unavailable")
        try:
            body = response.json()
        except ValueError as exc:
            raise ExternalSearchProviderError("external_search_response_invalid") from exc
        return _parse_response(body, max_results=max_results)


def _parse_response(body: object, *, max_results: int) -> ExternalSearchProviderResult:
    if not isinstance(body, Mapping):
        raise ExternalSearchProviderError("external_search_response_invalid")
    raw_items = body.get("results")
    if not isinstance(raw_items, list) or len(raw_items) > max_results:
        raise ExternalSearchProviderError("external_search_response_invalid")
    items: list[ExternalSearchProviderItem] = []
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            raise ExternalSearchProviderError("external_search_response_invalid")
        title = raw.get("title")
        url = raw.get("url")
        content = raw.get("content")
        score = raw.get("score")
        if (
            not isinstance(title, str)
            or not isinstance(url, str)
            or not isinstance(content, str)
            or not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not isfinite(float(score))
            or not 0 <= float(score) <= 1
        ):
            raise ExternalSearchProviderError("external_search_response_invalid")
        items.append(
            ExternalSearchProviderItem(
                title=title,
                url=url,
                content=content,
                score=float(score),
                published_at=_published_at(raw.get("published_date")),
            )
        )
    request_id = body.get("request_id")
    if request_id is not None and (
        not isinstance(request_id, str) or not 1 <= len(request_id) <= 128
    ):
        raise ExternalSearchProviderError("external_search_response_invalid")
    usage = body.get("usage")
    usage_credits: int | None = None
    if usage is not None:
        if not isinstance(usage, Mapping):
            raise ExternalSearchProviderError("external_search_response_invalid")
        credits = usage.get("credits")
        if credits is not None:
            if not isinstance(credits, int) or isinstance(credits, bool) or credits < 0:
                raise ExternalSearchProviderError("external_search_response_invalid")
            usage_credits = credits
    return ExternalSearchProviderResult(
        request_id=request_id,
        usage_credits=usage_credits,
        items=tuple(items),
    )


def _published_at(value: Any) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ExternalSearchProviderError("external_search_response_invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ExternalSearchProviderError("external_search_response_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ExternalSearchProviderError("external_search_response_invalid")
    return parsed
