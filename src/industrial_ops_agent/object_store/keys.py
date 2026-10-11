"""Canonical MinIO object-key construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from industrial_ops_agent.persistence.errors import TenantBoundaryViolation
from industrial_ops_agent.persistence.tenant import TenantContext, validate_boundary_identifier


class ObjectZone(StrEnum):
    QUARANTINE = "quarantine"
    CLEAN = "clean"
    GENERATED = "generated"


@dataclass(frozen=True, slots=True)
class TenantObjectKey:
    tenant_id: str
    zone: ObjectZone
    value: str

    def assert_owned_by(self, context: TenantContext) -> None:
        if self.tenant_id != context.tenant_id:
            raise TenantBoundaryViolation("object key is outside the active tenant")


def _object_key(
    context: TenantContext,
    zone: ObjectZone,
    draft_id: str,
    media_id: str,
) -> TenantObjectKey:
    draft_id = validate_boundary_identifier(draft_id, field="draft_id")
    media_id = validate_boundary_identifier(media_id, field="media_id")
    value = f"tenant/{context.tenant_id}/{zone.value}/{draft_id}/{media_id}"
    return TenantObjectKey(tenant_id=context.tenant_id, zone=zone, value=value)


def quarantine_object_key(context: TenantContext, draft_id: str, media_id: str) -> TenantObjectKey:
    return _object_key(context, ObjectZone.QUARANTINE, draft_id, media_id)


def clean_object_key(context: TenantContext, draft_id: str, media_id: str) -> TenantObjectKey:
    return _object_key(context, ObjectZone.CLEAN, draft_id, media_id)


def generated_object_key(
    context: TenantContext, source_id: str, artifact_id: str
) -> TenantObjectKey:
    """Build a tenant-owned key for immutable model-generated media."""

    return _object_key(context, ObjectZone.GENERATED, source_id, artifact_id)


def knowledge_source_object_key(
    context: TenantContext,
    zone: ObjectZone,
    ingestion_id: str,
) -> TenantObjectKey:
    ingestion_id = validate_boundary_identifier(ingestion_id, field="ingestion_id")
    value = f"tenant/{context.tenant_id}/{zone.value}/knowledge/{ingestion_id}/source"
    return TenantObjectKey(tenant_id=context.tenant_id, zone=zone, value=value)


def tenant_object_key_from_value(
    context: TenantContext,
    value: str,
) -> TenantObjectKey:
    """Rehydrate only an already-canonical key owned by the active tenant."""

    prefix = f"tenant/{context.tenant_id}/"
    if not value.startswith(prefix):
        raise TenantBoundaryViolation("object key is outside the active tenant")
    remainder = value.removeprefix(prefix)
    zone_value, separator, tail = remainder.partition("/")
    if not separator or not tail or "//" in value or value.endswith("/"):
        raise TenantBoundaryViolation("object key is not canonical")
    try:
        zone = ObjectZone(zone_value)
    except ValueError as exc:
        raise TenantBoundaryViolation("object key has an invalid zone") from exc
    key = TenantObjectKey(tenant_id=context.tenant_id, zone=zone, value=value)
    key.assert_owned_by(context)
    return key
