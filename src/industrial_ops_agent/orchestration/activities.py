"""Side-effecting Activity implementations kept outside deterministic workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from industrial_ops_agent.application.incidents import RecognitionService
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.media.service import TenantObjectStore
from industrial_ops_agent.multimodal.models import EvidenceBundle
from industrial_ops_agent.multimodal.processor import MultimodalProcessor


@dataclass(frozen=True, slots=True)
class RecognitionActivityInput:
    """Serializable command captured when an authorized recognition run is queued."""

    recognition_run_id: str
    identity: IdentityContext
    request_id: str
    workflow_id: str | None = None


class RecognitionDispatcher(Protocol):
    """API-to-workflow boundary implemented by Temporal in the next M2 slice."""

    async def dispatch(self, command: RecognitionActivityInput) -> None: ...


class RecognitionDispatchUnavailable(RuntimeError):
    """The durable workflow backend could not accept a recognition command."""


class RecognitionActivity:
    """Read only clean media, invoke providers, and atomically persist evidence."""

    def __init__(
        self,
        service: RecognitionService,
        object_store: TenantObjectStore,
        processor: MultimodalProcessor,
    ) -> None:
        self._service = service
        self._object_store = object_store
        self._processor = processor

    async def execute(self, command: RecognitionActivityInput) -> EvidenceBundle:
        return await self._service.process(
            command.identity,
            command.recognition_run_id,
            object_store=self._object_store,
            processor=self._processor,
            request_id=command.request_id,
        )
