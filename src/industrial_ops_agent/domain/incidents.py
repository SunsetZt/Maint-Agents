"""Immutable M1 incident-draft aggregate and concurrency invariants."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from hashlib import sha256
from typing import Any


class DraftVersionConflict(Exception):
    """The supplied version lost an optimistic-concurrency race."""

    def __init__(self, current_version: int) -> None:
        super().__init__("incident draft version conflict")
        self.current_version = current_version


class DraftStateConflict(Exception):
    """A command is not legal for the draft's current lifecycle state."""


class IncidentSubmissionBlocked(Exception):
    """A submit precondition changed before the command was committed."""

    def __init__(self, reason_code: str) -> None:
        super().__init__("incident submission precondition failed")
        self.reason_code = reason_code


class IncidentVersionConflict(Exception):
    def __init__(self, current_version: int) -> None:
        super().__init__("incident version conflict")
        self.current_version = current_version


class IncidentStateConflict(Exception):
    """A formal Incident command is illegal for its current status."""


@dataclass(frozen=True, slots=True)
class DraftTimelineEvent:
    sequence: int
    event_type: str
    occurred_at: datetime
    payload_metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Incident:
    """Formal incident created from one immutable draft/evidence snapshot."""

    incident_id: str
    tenant_id: str
    asset_id: str
    source_draft_id: str
    evidence_bundle_id: str
    reporter_subject_id: str
    description: str
    status: str
    version: int
    diagnosis_run_ids: tuple[str, ...]
    created_at: datetime
    updated_at: datetime

    def triage(self, *, expected_version: int, occurred_at: datetime) -> Incident:
        if expected_version != self.version:
            raise IncidentVersionConflict(self.version)
        if self.status != "SUBMITTED":
            raise IncidentStateConflict("only a SUBMITTED Incident can be triaged")
        return replace(
            self,
            status="TRIAGED",
            version=self.version + 1,
            updated_at=occurred_at,
        )


@dataclass(frozen=True, slots=True)
class IncidentDraft:
    draft_id: str
    tenant_id: str
    asset_id: str
    author_subject_id: str
    description: str
    version: int
    status: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(
        cls,
        *,
        draft_id: str,
        tenant_id: str,
        asset_id: str,
        author_subject_id: str,
        description: str,
        occurred_at: datetime,
    ) -> tuple[IncidentDraft, DraftTimelineEvent]:
        draft = cls(
            draft_id=draft_id,
            tenant_id=tenant_id,
            asset_id=asset_id,
            author_subject_id=author_subject_id,
            description=description,
            version=1,
            status="DRAFT",
            created_at=occurred_at,
            updated_at=occurred_at,
        )
        event = DraftTimelineEvent(
            sequence=1,
            event_type="draft.created",
            occurred_at=occurred_at,
            payload_metadata={"fields": ["asset_id", "description"]},
        )
        return draft, event

    def update_description(
        self,
        *,
        description: str,
        expected_version: int,
        occurred_at: datetime,
    ) -> tuple[IncidentDraft, DraftTimelineEvent]:
        if expected_version != self.version:
            raise DraftVersionConflict(self.version)
        if self.status != "DRAFT":
            raise DraftStateConflict("only a DRAFT can be updated")
        next_version = self.version + 1
        updated = replace(
            self,
            description=description,
            version=next_version,
            updated_at=occurred_at,
        )
        event = DraftTimelineEvent(
            sequence=next_version,
            event_type="draft.description_updated",
            occurred_at=occurred_at,
            payload_metadata={"fields": ["description"]},
        )
        return updated, event

    def submit(
        self,
        *,
        incident_id: str,
        evidence_bundle_id: str,
        expected_version: int,
        evidence_confirmed: bool,
        all_media_clean: bool,
        device_authorized: bool,
        occurred_at: datetime,
    ) -> tuple[IncidentDraft, Incident, DraftTimelineEvent]:
        """Create the formal incident without implicitly starting diagnosis."""

        if expected_version != self.version:
            raise DraftVersionConflict(self.version)
        if self.status != "DRAFT":
            raise DraftStateConflict("only a DRAFT can be submitted")
        if not evidence_confirmed:
            raise IncidentSubmissionBlocked("evidence_not_confirmed")
        if not all_media_clean:
            raise IncidentSubmissionBlocked("media_not_clean")
        if not device_authorized:
            raise IncidentSubmissionBlocked("device_access_revoked")
        if not evidence_bundle_id:
            raise IncidentSubmissionBlocked("evidence_bundle_missing")

        next_version = self.version + 1
        submitted_draft = replace(
            self,
            status="SUBMITTED",
            version=next_version,
            updated_at=occurred_at,
        )
        incident = Incident(
            incident_id=incident_id,
            tenant_id=self.tenant_id,
            asset_id=self.asset_id,
            source_draft_id=self.draft_id,
            evidence_bundle_id=evidence_bundle_id,
            reporter_subject_id=self.author_subject_id,
            description=self.description,
            status="SUBMITTED",
            version=1,
            diagnosis_run_ids=(),
            created_at=occurred_at,
            updated_at=occurred_at,
        )
        event = DraftTimelineEvent(
            sequence=next_version,
            event_type="draft.submitted",
            occurred_at=occurred_at,
            payload_metadata={
                "evidence_bundle_id": evidence_bundle_id,
                "incident_id": incident_id,
            },
        )
        return submitted_draft, incident, event


def create_request_hash(*, asset_id: str, description: str) -> str:
    """Hash a canonical request body so idempotency storage never keeps its text."""

    canonical = json.dumps(
        {"asset_id": asset_id, "description": description},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(canonical.encode()).hexdigest()
