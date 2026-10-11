"""Trusted tenant context shared by all infrastructure adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_boundary_identifier(value: str, *, field: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} is not a canonical boundary identifier")
    return value


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Server-derived tenant and subject identity for one operation."""

    tenant_id: str
    subject_id: str

    def __post_init__(self) -> None:
        validate_boundary_identifier(self.tenant_id, field="tenant_id")
        validate_boundary_identifier(self.subject_id, field="subject_id")
