"""Validated identity context created only from a verified OIDC token."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from industrial_ops_agent.persistence.tenant import (
    TenantContext,
    validate_boundary_identifier,
)


class Role(StrEnum):
    CUSTOMER_CONTACT = "customer_contact"
    FIELD_ENGINEER = "field_engineer"
    AFTER_SALES_ENGINEER = "after_sales_engineer"
    DOMAIN_EXPERT = "domain_expert"
    TENANT_ADMIN = "tenant_admin"
    SECURITY_AUDITOR = "security_auditor"
    DATA_STEWARD = "data_steward"
    ANNOTATION_ADMIN = "annotation_admin"
    MODEL_ENGINEER = "model_engineer"
    MODEL_EVALUATOR = "model_evaluator"
    MODEL_RELEASE_APPROVER = "model_release_approver"
    MODEL_RELEASE_OPERATOR = "model_release_operator"
    MODEL_DEPLOYMENT_CONTROLLER = "model_deployment_controller"
    PLATFORM_OPERATOR = "platform_operator"


@dataclass(frozen=True, slots=True)
class IdentityContext:
    subject_id: str
    oidc_subject: str
    tenant_id: str
    roles: frozenset[Role]
    asset_ids: frozenset[str]
    site_ids: frozenset[str]
    issued_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        TenantContext(tenant_id=self.tenant_id, subject_id=self.subject_id)
        if not self.oidc_subject or len(self.oidc_subject) > 255:
            raise ValueError("oidc subject is invalid")
        if not self.roles:
            raise ValueError("at least one mapped role is required")
        if self.expires_at <= self.issued_at:
            raise ValueError("token expiry must be after its issue time")

    @property
    def tenant_context(self) -> TenantContext:
        return TenantContext(tenant_id=self.tenant_id, subject_id=self.subject_id)

    @classmethod
    def from_claims(cls, claims: dict[str, Any]) -> IdentityContext:
        subject_id = _required_string(claims, "subject_id")
        tenant_id = _required_string(claims, "tenant_id")
        oidc_subject = _required_string(claims, "sub")
        validate_boundary_identifier(subject_id, field="subject_id")
        validate_boundary_identifier(tenant_id, field="tenant_id")
        roles = _mapped_roles(claims.get("roles"))
        asset_ids = _identifier_set(claims.get("asset_ids", []), field="asset_ids")
        site_ids = _identifier_set(claims.get("site_ids", []), field="site_ids")
        return cls(
            subject_id=subject_id,
            oidc_subject=oidc_subject,
            tenant_id=tenant_id,
            roles=roles,
            asset_ids=asset_ids,
            site_ids=site_ids,
            issued_at=_numeric_date(claims, "iat"),
            expires_at=_numeric_date(claims, "exp"),
        )


def _required_string(claims: dict[str, Any], name: str) -> str:
    value = claims.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"required claim is invalid: {name}")
    return value


def _mapped_roles(value: Any) -> frozenset[Role]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("roles claim is invalid")
    mapped = frozenset(Role(item) for item in value if item in Role._value2member_map_)
    if not mapped:
        raise ValueError("roles claim has no approved mapping")
    return mapped


def _identifier_set(value: Any, *, field: str) -> frozenset[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} claim is invalid")
    return frozenset(validate_boundary_identifier(item, field=field) for item in value)


def _numeric_date(claims: dict[str, Any], name: str) -> datetime:
    value = claims.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"numeric date claim is invalid: {name}")
    return datetime.fromtimestamp(value, tz=UTC)
