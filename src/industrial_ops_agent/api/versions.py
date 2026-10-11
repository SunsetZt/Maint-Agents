"""Version preconditions grouped by their existing, distinct wire contracts."""

from __future__ import annotations

import re

from industrial_ops_agent.api.errors import AppError


def positive_version(value: str) -> int:
    try:
        version = int(value.strip().strip('"'))
    except ValueError as exc:
        raise _invalid_version() from exc
    if version < 1:
        raise _invalid_version()
    return version


def integer_version(value: str) -> int:
    try:
        return int(value.strip().strip('"'))
    except ValueError as exc:
        raise _invalid_version() from exc


def numeric_precondition(value: str) -> int:
    try:
        return int(value.strip().strip('"'))
    except ValueError as exc:
        raise AppError(
            status_code=400,
            code="invalid_if_match",
            category="precondition",
            message="If-Match must contain a numeric version",
        ) from exc


def decimal_version(value: str) -> int:
    match = re.fullmatch(r'"?([1-9][0-9]*)"?', value.strip())
    if match is None:
        raise _invalid_version()
    return int(match.group(1))


def weak_positive_version(value: str) -> int:
    normalized = value.strip().removeprefix("W/").strip('"')
    try:
        version = int(normalized)
    except ValueError as exc:
        raise _invalid_version() from exc
    if version < 1:
        raise _invalid_version()
    return version


def _invalid_version() -> AppError:
    return AppError(
        status_code=400,
        code="invalid_version_precondition",
        category="validation",
        message="If-Match is invalid",
    )
