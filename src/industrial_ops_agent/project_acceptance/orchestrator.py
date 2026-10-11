"""Typed public-API state machine for project-staging business acceptance."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Protocol

from industrial_ops_agent.project_acceptance.contracts import (
    ACCEPTANCE_STAGE_ORDER,
    AcceptanceReceipt,
    ProjectAcceptanceContractError,
    ValidatedProjectAcceptanceManifest,
    checkpoint_from_acceptance_receipt,
    validate_acceptance_receipt,
)
from industrial_ops_agent.project_acceptance.contracts import (
    AcceptanceCheckpoint as AcceptanceCheckpoint,
)
from industrial_ops_agent.project_acceptance.contracts import (
    AcceptanceStage as AcceptanceStage,
)

MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_ERROR_BYTES = 64 * 1024
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SAFE_ERROR_CODE_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class AcceptanceActor(StrEnum):
    """One least-privilege identity used by the public workflow."""

    CUSTOMER = "customer"
    AFTER_SALES = "after_sales"
    APPROVER = "approver"
    FIELD = "field"
    VERIFIER = "verifier"


PUBLIC_API_STAGE_ORDER = ACCEPTANCE_STAGE_ORDER

FORBIDDEN_PUBLIC_PATH_FRAGMENTS = (
    "/parts",
    "/wms",
    "/internal",
    "/admin",
)


class ProjectAcceptanceFailure(RuntimeError):
    """Stable, non-sensitive orchestration failure."""

    def __init__(
        self,
        reason: str,
        *,
        stage: AcceptanceStage | None = None,
    ) -> None:
        self.reason = reason
        self.stage = stage
        message = f"{stage.value}:{reason}" if stage is not None else reason
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ApiRequest:
    """One bounded request against an existing public API route."""

    actor: AcceptanceActor
    method: Literal["GET", "POST"]
    path: str
    expected_statuses: frozenset[int]
    json_body: Mapping[str, Any] | None = None
    content: bytes | None = None
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ApiResponse:
    """Validated public API envelope data and selected response metadata."""

    status: int
    data: object
    headers: Mapping[str, str] = field(default_factory=dict)


class PublicApiTransport(Protocol):
    """Transport boundary used by the orchestration state machine."""

    def request(self, request: ApiRequest) -> ApiResponse: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_: object, **__: object) -> None:
        return None


class UrllibPublicApiTransport:
    """Secret-safe loopback transport with closed JSON envelope validation."""

    def __init__(
        self,
        api_url: str,
        *,
        actor_tokens: Mapping[AcceptanceActor, str],
        timeout_seconds: float,
    ) -> None:
        normalized = api_url.rstrip("/")
        parsed = urllib.parse.urlsplit(normalized)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ProjectAcceptanceFailure("api_url_must_be_loopback_without_credentials")
        if (
            not _is_finite_number(timeout_seconds)
            or timeout_seconds <= 0
            or timeout_seconds > 120
        ):
            raise ProjectAcceptanceFailure("api_timeout_invalid")
        if set(actor_tokens) != set(AcceptanceActor) or any(
            not isinstance(token, str) or not token or len(token) > 16_384
            for token in actor_tokens.values()
        ):
            raise ProjectAcceptanceFailure("actor_tokens_incomplete")
        self._api_url = normalized
        self._actor_tokens = dict(actor_tokens)
        self._timeout_seconds = timeout_seconds
        self._opener = urllib.request.build_opener(_NoRedirect())

    def request(self, request: ApiRequest) -> ApiResponse:
        _validate_public_path(request.path)
        if request.json_body is not None and request.content is not None:
            raise ProjectAcceptanceFailure("api_request_body_ambiguous")
        body = request.content
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._actor_tokens[request.actor]}",
            "User-Agent": "ioap-project-business-acceptance/1",
            **request.headers,
        }
        if request.json_body is not None:
            body = json.dumps(
                request.json_body,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"
        wire_request = urllib.request.Request(
            self._api_url + request.path,
            data=body,
            headers=headers,
            method=request.method,
        )
        try:
            with self._opener.open(
                wire_request,
                timeout=self._timeout_seconds,
            ) as response:
                payload = response.read(MAX_RESPONSE_BYTES + 1)
                if len(payload) > MAX_RESPONSE_BYTES:
                    raise ProjectAcceptanceFailure("api_response_too_large")
                if response.status not in request.expected_statuses:
                    raise ProjectAcceptanceFailure(f"api_unexpected_status:{response.status}")
                content_type = response.headers.get_content_type().lower()
                if content_type != "application/json" and not content_type.endswith("+json"):
                    raise ProjectAcceptanceFailure("api_response_content_type_invalid")
                response_headers = {name.lower(): value for name, value in response.headers.items()}
                status = response.status
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400:
                raise ProjectAcceptanceFailure("api_redirect_rejected") from None
            code = _safe_http_error_code(exc.read(MAX_ERROR_BYTES + 1))
            raise ProjectAcceptanceFailure(f"api_http_error:{exc.code}:{code}") from None
        except ProjectAcceptanceFailure:
            raise
        except (TimeoutError, urllib.error.URLError, OSError):
            raise ProjectAcceptanceFailure("api_transport_unavailable") from None

        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ProjectAcceptanceFailure("api_response_invalid") from None
        if (
            not isinstance(document, dict)
            or set(document) != {"data", "meta"}
            or not isinstance(document["meta"], dict)
            or not isinstance(document["meta"].get("request_id"), str)
            or not isinstance(document["data"], (dict, list))
        ):
            raise ProjectAcceptanceFailure("api_envelope_invalid")
        return ApiResponse(
            status=status,
            data=document["data"],
            headers=response_headers,
        )


ProgressCallback = Callable[[AcceptanceStage, Mapping[str, str | int | bool]], None]
CheckpointSink = Callable[[AcceptanceCheckpoint], None]


class ProjectAcceptanceOrchestrator:
    """Run and reconcile the seven-stage governed public-API workflow."""

    def __init__(
        self,
        *,
        transport: PublicApiTransport,
        manifest: ValidatedProjectAcceptanceManifest,
        run_id: str,
        expected_release_id: str,
        expected_model_manifest_hash: str,
        poll_seconds: float = 2.0,
        workflow_timeout_seconds: float = 900.0,
        progress: ProgressCallback | None = None,
        checkpoint_sink: CheckpointSink | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ProjectAcceptanceFailure("run_id_invalid")
        if not _is_safe_identifier(expected_release_id):
            raise ProjectAcceptanceFailure("expected_release_id_invalid")
        if (
            not expected_model_manifest_hash
            or len(expected_model_manifest_hash) > 128
            or not re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", expected_model_manifest_hash)
        ):
            raise ProjectAcceptanceFailure("expected_model_manifest_hash_invalid")
        if (
            not _is_finite_number(poll_seconds)
            or not _is_finite_number(workflow_timeout_seconds)
            or poll_seconds < 0
            or workflow_timeout_seconds <= 0
        ):
            raise ProjectAcceptanceFailure("workflow_timeout_invalid")
        self._transport = transport
        self._manifest = manifest
        self._run_id = run_id
        self._expected_release_id = expected_release_id
        self._expected_model_manifest_hash = expected_model_manifest_hash
        self._poll_seconds = poll_seconds
        self._workflow_timeout_seconds = workflow_timeout_seconds
        self._progress = progress
        self._checkpoint_sink = checkpoint_sink
        self._monotonic = monotonic
        self._sleep = sleep
        self._deadline = 0.0

    def run(
        self,
        *,
        checkpoint: AcceptanceCheckpoint | None = None,
        stop_after: AcceptanceStage | None = None,
    ) -> AcceptanceCheckpoint:
        """Run from a fresh or reconciled checkpoint and return safe stage facts."""

        current = checkpoint or self._new_checkpoint()
        self._validate_checkpoint(current)
        self._deadline = self._monotonic() + self._workflow_timeout_seconds
        completed = set(current.completed_stages)

        for stage in PUBLIC_API_STAGE_ORDER:
            try:
                if stage in completed:
                    self._reconcile(stage, current)
                else:
                    updates = self._execute(stage, current)
                    current = self._advance(current, stage, updates)
                    if self._checkpoint_sink is not None:
                        try:
                            self._checkpoint_sink(current)
                        except ProjectAcceptanceContractError as exc:
                            raise ProjectAcceptanceFailure(
                                f"checkpoint_persist_failed:{exc.reason}"
                            ) from None
                    completed.add(stage)
                    if self._progress is not None:
                        self._progress(stage, updates)
            except ProjectAcceptanceFailure as exc:
                if exc.stage is not None:
                    raise
                raise ProjectAcceptanceFailure(exc.reason, stage=stage) from None
            except (KeyError, TypeError, ValueError):
                raise ProjectAcceptanceFailure(
                    "api_data_invalid",
                    stage=stage,
                ) from None
            if stop_after is stage:
                return current
        return current

    def reconcile_receipt(self, receipt: AcceptanceReceipt) -> None:
        """Read current public APIs and reconcile one immutable receipt."""

        try:
            validated = validate_acceptance_receipt(receipt)
            checkpoint = checkpoint_from_acceptance_receipt(validated)
            self._validate_checkpoint(checkpoint)
            self._reconcile_quotation(checkpoint)
            self._reconcile_work_order(checkpoint, minimum_status="CLOSED")
            self._reconcile_closed_state(checkpoint)
            self._reconcile_report(checkpoint)
            self._reconcile_model_evidence(checkpoint)
        except (ProjectAcceptanceContractError, ProjectAcceptanceFailure) as exc:
            reason = exc.reason
            raise ProjectAcceptanceFailure(
                f"receipt_reconciliation_failed:{reason}"
            ) from None

    def _new_checkpoint(self) -> AcceptanceCheckpoint:
        binding = self._manifest.manifest.tenant_binding
        return AcceptanceCheckpoint(
            run_id=self._run_id,
            manifest_id=self._manifest.manifest.manifest_id,
            manifest_sha256=self._manifest.manifest_sha256,
            source_image_sha256=self._manifest.source_image_sha256,
            tenant_id=binding.tenant_id,
            site_id=binding.site_id,
            asset_id=binding.asset_id,
            expected_release_id=self._expected_release_id,
            expected_model_manifest_hash=self._expected_model_manifest_hash,
        )

    def _validate_checkpoint(self, checkpoint: AcceptanceCheckpoint) -> None:
        expected = self._new_checkpoint()
        bound_fields = (
            "schema_version",
            "run_id",
            "manifest_id",
            "manifest_sha256",
            "source_image_sha256",
            "tenant_id",
            "site_id",
            "asset_id",
            "expected_release_id",
            "expected_model_manifest_hash",
        )
        if any(getattr(checkpoint, name) != getattr(expected, name) for name in bound_fields):
            raise ProjectAcceptanceFailure("checkpoint_binding_mismatch")
        completed = checkpoint.completed_stages
        if completed != PUBLIC_API_STAGE_ORDER[: len(completed)]:
            raise ProjectAcceptanceFailure("checkpoint_stage_order_invalid")
        if any(
            re.search(r"token|credential|authorization|prompt|response|raw", key, re.I)
            for key in checkpoint.facts
        ):
            raise ProjectAcceptanceFailure("checkpoint_contains_prohibited_fact")

    def _advance(
        self,
        checkpoint: AcceptanceCheckpoint,
        stage: AcceptanceStage,
        updates: Mapping[str, str | int | bool],
    ) -> AcceptanceCheckpoint:
        facts = {**checkpoint.facts, **updates}
        return checkpoint.model_copy(
            update={
                "completed_stages": (*checkpoint.completed_stages, stage),
                "facts": facts,
            }
        )

    def _execute(
        self,
        stage: AcceptanceStage,
        checkpoint: AcceptanceCheckpoint,
    ) -> dict[str, str | int | bool]:
        if stage is AcceptanceStage.MODEL_EVIDENCE:
            return self._model_evidence()
        if stage is AcceptanceStage.QUOTATION:
            return self._quotation(checkpoint)
        if stage is AcceptanceStage.WORK_ORDER_AUTHORIZATION:
            return self._work_order_authorization(checkpoint)
        if stage is AcceptanceStage.ASSIGNMENT:
            return self._assignment(checkpoint)
        if stage is AcceptanceStage.FIELD_EXECUTION:
            return self._field_execution(checkpoint)
        if stage is AcceptanceStage.RESULT_ACCEPTANCE:
            return self._result_acceptance(checkpoint)
        if stage is AcceptanceStage.RECEIPT:
            return self._receipt(checkpoint)
        raise ProjectAcceptanceFailure("stage_unknown")

    def _reconcile(
        self,
        stage: AcceptanceStage,
        checkpoint: AcceptanceCheckpoint,
    ) -> None:
        if stage is AcceptanceStage.MODEL_EVIDENCE:
            self._reconcile_model_evidence(checkpoint)
        elif stage is AcceptanceStage.QUOTATION:
            self._reconcile_quotation(checkpoint)
        elif stage is AcceptanceStage.WORK_ORDER_AUTHORIZATION:
            self._reconcile_work_order(checkpoint, minimum_status="READY")
        elif stage is AcceptanceStage.ASSIGNMENT:
            self._reconcile_work_order(checkpoint, minimum_status="ASSIGNED")
        elif stage is AcceptanceStage.FIELD_EXECUTION:
            self._reconcile_work_order(checkpoint, minimum_status="COMPLETED")
        elif stage is AcceptanceStage.RESULT_ACCEPTANCE:
            self._reconcile_closed_state(checkpoint)
        elif stage is AcceptanceStage.RECEIPT:
            self._reconcile_report(checkpoint)

    def _model_evidence(self) -> dict[str, str | int | bool]:
        scenario = self._manifest.manifest.business_scenario
        binding = self._manifest.manifest.tenant_binding
        image = self._read_bound_image()

        draft = self._post_object(
            AcceptanceActor.CUSTOMER,
            "/incident-drafts",
            operation="draft",
            expected_statuses={200, 201},
            json_body={
                "asset_id": binding.asset_id,
                "description": scenario.incident_description,
            },
        )
        draft_id = _required_id(draft, "draft_id")
        draft_version = _required_int(draft, "version")
        media = self._post_object(
            AcceptanceActor.CUSTOMER,
            f"/incident-drafts/{draft_id}/media",
            operation="draft-media",
            expected_statuses={201},
            content=image,
            headers={"Content-Type": self._manifest.manifest.source_image.media_type},
        )
        media_id = _required_id(media, "media_id")
        self._wait_object(
            AcceptanceActor.CUSTOMER,
            f"/media/{media_id}/status",
            status_field="scan_state",
            success={"CLEAN"},
            failures={"REJECTED", "INFECTED", "FAILED"},
            reason="media_scan",
        )

        recognition = self._post_object(
            AcceptanceActor.CUSTOMER,
            f"/incident-drafts/{draft_id}/recognition-runs",
            operation="recognition",
            expected_statuses={200, 202},
            json_body={"media_id": media_id, "processor_profile": "local-core-v1"},
        )
        recognition_id = _required_id(recognition, "recognition_run_id")
        self._wait_object(
            AcceptanceActor.CUSTOMER,
            f"/incident-drafts/{draft_id}/recognition-runs/{recognition_id}",
            status_field="status",
            success={"SUCCEEDED"},
            failures={"FAILED", "CANCELLED", "CANCELED"},
            reason="recognition",
        )
        evidence = self._get_object(
            AcceptanceActor.CUSTOMER,
            f"/incident-drafts/{draft_id}/evidence",
        )
        vlm_execution = self._validate_recognition_evidence(evidence)
        evidence_version = _required_int(evidence, "version")
        bundle_id = _required_id(evidence, "bundle_id")

        confirmation = self._post_object(
            AcceptanceActor.CUSTOMER,
            f"/incident-drafts/{draft_id}/recognition-confirmations",
            operation="recognition-confirmation",
            expected_statuses={200},
            headers={"If-Match": _etag(evidence_version)},
            json_body=self._confirmation_body(evidence, bundle_id=bundle_id),
        )
        if str(confirmation.get("status", "")).upper() != "CONFIRMED":
            raise ProjectAcceptanceFailure("recognition_confirmation_incomplete")

        incident = self._post_object(
            AcceptanceActor.CUSTOMER,
            f"/incident-drafts/{draft_id}/submit",
            operation="incident-submit",
            expected_statuses={200, 201},
            headers={"If-Match": _etag(draft_version)},
            json_body={"evidence_bundle_id": bundle_id},
        )
        incident_id = _required_id(incident, "incident_id")
        if incident.get("asset_id") != binding.asset_id:
            raise ProjectAcceptanceFailure("incident_binding_mismatch")
        triaged = self._post_object(
            AcceptanceActor.AFTER_SALES,
            f"/incidents/{incident_id}/triage",
            operation="incident-triage",
            expected_statuses={200},
            headers={"If-Match": _etag(_required_int(incident, "version"))},
        )
        diagnosis = self._post_object(
            AcceptanceActor.AFTER_SALES,
            f"/incidents/{incident_id}/diagnoses",
            operation="diagnosis",
            expected_statuses={200, 202},
            headers={"If-Match": _etag(_required_int(triaged, "version"))},
        )
        diagnosis_id = _required_id(diagnosis, "diagnosis_run_id")
        completed_diagnosis = self._wait_object(
            AcceptanceActor.AFTER_SALES,
            f"/diagnosis-runs/{diagnosis_id}",
            status_field="status",
            success={"COMPLETED"},
            failures={
                "FAILED",
                "CANCELLED",
                "CANCELED",
                "NEEDS_DATA",
                "NEEDS_INPUT",
                "ESCALATED",
            },
            reason="diagnosis",
        )
        diagnosis_execution = self._validate_execution(
            completed_diagnosis.get("model_execution"),
            component="diagnosis",
            request_class="DIAGNOSIS",
        )
        current_incident = self._get_object(
            AcceptanceActor.AFTER_SALES,
            f"/incidents/{incident_id}",
        )
        if (
            current_incident.get("asset_id") != binding.asset_id
            or str(current_incident.get("status", "")).upper() != "DIAGNOSED"
        ):
            raise ProjectAcceptanceFailure("incident_diagnosis_state_mismatch")
        return {
            "draft_id": draft_id,
            "media_id": media_id,
            "recognition_run_id": recognition_id,
            "evidence_bundle_id": bundle_id,
            "incident_id": incident_id,
            "incident_version": _required_int(current_incident, "version"),
            "diagnosis_run_id": diagnosis_id,
            "diagnosis_version": _required_int(completed_diagnosis, "version"),
            "vlm_inference_request_id": _required_id(
                vlm_execution,
                "inference_request_id",
            ),
            "diagnosis_inference_request_id": _required_id(
                diagnosis_execution,
                "inference_request_id",
            ),
            "ocr_processor_version": _ocr_processor(evidence),
        }

    def _quotation(
        self,
        checkpoint: AcceptanceCheckpoint,
    ) -> dict[str, str | int | bool]:
        incident_id = _fact_id(checkpoint, "incident_id")
        expected_candidate = self._manifest.manifest.business_scenario.quotation_candidate_id
        options = self._get_object(
            AcceptanceActor.AFTER_SALES,
            f"/incidents/{incident_id}/service-quotation-options",
        )
        option = _select_exact_option(
            options.get("options"),
            {"candidate_id": expected_candidate},
            reason="quotation_candidate_unavailable",
        )
        proposal = self._post_object(
            AcceptanceActor.AFTER_SALES,
            f"/incidents/{incident_id}/service-quotation-proposals",
            operation="quotation-proposal",
            expected_statuses={201},
            json_body={"candidate_id": option["candidate_id"]},
        )
        execution = self._approve_and_execute("quotation", proposal)
        quotation_id = _required_id(execution, "service_quotation_id")
        quotation_view = self._get_response(
            AcceptanceActor.CUSTOMER,
            f"/portal/cases/{incident_id}/service-quotation",
            expected_statuses={200},
        )
        view = _object_data(quotation_view.data)
        current = _mapping(view.get("current"), reason="customer_quotation_missing")
        if current.get("quotation_id") != quotation_id:
            raise ProjectAcceptanceFailure("customer_quotation_binding_mismatch")
        current_etag = _response_etag(quotation_view)
        decision = self._post_object(
            AcceptanceActor.CUSTOMER,
            f"/portal/cases/{incident_id}/service-quotation/decision",
            operation="quotation-customer-acceptance",
            expected_statuses={200},
            headers={"If-Match": current_etag},
            json_body={"decision": "ACCEPTED"},
        )
        if str(decision.get("status", "")).upper() != "ACCEPTED":
            raise ProjectAcceptanceFailure("customer_quotation_not_accepted")
        return {
            "quotation_proposal_id": _required_id(proposal, "proposal_id"),
            "quotation_approval_id": _required_id(proposal, "approval_id"),
            "service_quotation_id": quotation_id,
            "quotation_decision_id": _required_id(decision, "decision_id"),
        }

    def _work_order_authorization(
        self,
        checkpoint: AcceptanceCheckpoint,
    ) -> dict[str, str | int | bool]:
        incident_id = _fact_id(checkpoint, "incident_id")
        current_incident = self._get_object(
            AcceptanceActor.AFTER_SALES,
            f"/incidents/{incident_id}",
        )
        options = self._get_object(
            AcceptanceActor.AFTER_SALES,
            f"/incidents/{incident_id}/repair-work-order-options",
        )
        option = _select_exact_option(
            options.get("options"),
            {
                "authorization_type": "COVERED_SERVICE",
                "fulfillment_mode": "NO_INITIAL_PARTS",
                "quotation_id": None,
                "quotation_version": None,
                "quotation_state_version": None,
            },
            reason="repair_authorization_unavailable",
        )
        option_version = _required_int(options, "incident_version")
        if option_version != _required_int(current_incident, "version"):
            raise ProjectAcceptanceFailure("repair_authorization_version_drift")
        proposal = self._post_object(
            AcceptanceActor.AFTER_SALES,
            f"/incidents/{incident_id}/repair-work-order-proposals",
            operation="work-order-proposal",
            expected_statuses={201},
            json_body={
                "incident_version": option_version,
                "authorization_type": option["authorization_type"],
            },
        )
        execution = self._approve_and_execute("work-order", proposal)
        work_order_id = _required_id(execution, "work_order_id")
        work_order = self._get_object(
            AcceptanceActor.AFTER_SALES,
            f"/work-orders/{work_order_id}",
        )
        self._validate_no_parts_work_order(
            work_order,
            incident_id=incident_id,
            minimum_status="READY",
        )
        return {
            "work_order_proposal_id": _required_id(proposal, "proposal_id"),
            "work_order_approval_id": _required_id(proposal, "approval_id"),
            "work_order_id": work_order_id,
            "work_order_version": _required_int(work_order, "version"),
        }

    def _assignment(
        self,
        checkpoint: AcceptanceCheckpoint,
    ) -> dict[str, str | int | bool]:
        work_order_id = _fact_id(checkpoint, "work_order_id")
        scenario = self._manifest.manifest.business_scenario
        work_order = self._get_object(
            AcceptanceActor.AFTER_SALES,
            f"/work-orders/{work_order_id}",
        )
        options = self._get_object(
            AcceptanceActor.AFTER_SALES,
            f"/work-orders/{work_order_id}/assignment-options",
        )
        option = _select_exact_option(
            options.get("options"),
            {
                "assignment_target": "ENTERPRISE_FSM",
                "candidate_profile_id": scenario.fsm_candidate_profile_id,
                "assignee_subject_id": scenario.field_assignee_subject_id,
            },
            reason="fsm_assignment_candidate_unavailable",
        )
        proposal = self._post_object(
            AcceptanceActor.AFTER_SALES,
            f"/work-orders/{work_order_id}/assignment-proposals",
            operation="assignment-proposal",
            expected_statuses={201},
            json_body={
                "work_order_version": _required_int(work_order, "version"),
                "assignment_target": "ENTERPRISE_FSM",
                "candidate_profile_id": option["candidate_profile_id"],
                "reason": "Project-staging enterprise FSM assignment candidate reviewed.",
            },
        )
        execution = self._approve_and_execute("assignment", proposal)
        delivery_id = _required_id(execution, "assignment_delivery_id")
        assigned = self._get_object(
            AcceptanceActor.FIELD,
            f"/work-orders/{work_order_id}",
        )
        self._validate_no_parts_work_order(
            assigned,
            incident_id=_fact_id(checkpoint, "incident_id"),
            minimum_status="ASSIGNED",
        )
        if assigned.get("assigned_subject_id") != scenario.field_assignee_subject_id:
            raise ProjectAcceptanceFailure("fsm_assignment_subject_mismatch")
        return {
            "assignment_proposal_id": _required_id(proposal, "proposal_id"),
            "assignment_approval_id": _required_id(proposal, "approval_id"),
            "assignment_delivery_id": delivery_id,
            "field_assignee_subject_id": scenario.field_assignee_subject_id,
            "work_order_version": _required_int(assigned, "version"),
        }

    def _field_execution(
        self,
        checkpoint: AcceptanceCheckpoint,
    ) -> dict[str, str | int | bool]:
        work_order_id = _fact_id(checkpoint, "work_order_id")
        scenario = self._manifest.manifest.business_scenario
        current = self._get_object(
            AcceptanceActor.FIELD,
            f"/work-orders/{work_order_id}",
        )
        accepted = self._post_object(
            AcceptanceActor.FIELD,
            f"/work-orders/{work_order_id}/accept",
            operation="field-accept",
            expected_statuses={200},
            headers={"If-Match": _etag(_required_int(current, "version"))},
        )
        current = self._get_object(
            AcceptanceActor.FIELD,
            f"/work-orders/{work_order_id}",
        )
        if _required_int(current, "version") != _required_int(accepted, "version"):
            raise ProjectAcceptanceFailure("field_accept_version_drift")
        self._post_object(
            AcceptanceActor.FIELD,
            f"/work-orders/{work_order_id}/start",
            operation="field-start",
            expected_statuses={200},
            headers={"If-Match": _etag(_required_int(current, "version"))},
        )
        current = self._get_object(
            AcceptanceActor.FIELD,
            f"/work-orders/{work_order_id}",
        )
        image = self._read_bound_image()
        upload = self._post_object(
            AcceptanceActor.FIELD,
            f"/field/work-orders/{work_order_id}/evidence-uploads",
            operation="field-evidence-upload",
            expected_statuses={200, 201},
            content=image,
            headers={
                "Content-Type": self._manifest.manifest.source_image.media_type,
                "If-Match": _etag(_required_int(current, "version")),
                "X-Content-SHA256": f"sha256:{self._manifest.source_image_sha256}",
            },
        )
        evidence_upload_id = _required_id(upload, "evidence_upload_id")
        clean_upload = self._wait_for_field_upload(
            work_order_id,
            evidence_upload_id,
        )
        if clean_upload.get("content_hash") != (f"sha256:{self._manifest.source_image_sha256}"):
            raise ProjectAcceptanceFailure("field_evidence_digest_mismatch")
        evidence_entry = self._field_entry(
            work_order_id,
            operation="field-entry-evidence",
            body={
                "entry_type": "EVIDENCE",
                "evidence_id": evidence_upload_id,
            },
        )
        step_entry = self._field_entry(
            work_order_id,
            operation="field-entry-step",
            body={
                "entry_type": "STEP",
                "step_code": scenario.field_step_code,
                "description": scenario.field_step_description,
                "outcome": "COMPLETED",
            },
        )
        signature_entry = self._field_entry(
            work_order_id,
            operation="field-entry-signature",
            body={
                "entry_type": "SIGNATURE",
                "signed_by": scenario.customer_signer,
                "signature_role": "CUSTOMER",
                "confirmation_text": scenario.customer_signature_text,
            },
        )
        current = self._get_object(
            AcceptanceActor.FIELD,
            f"/work-orders/{work_order_id}",
        )
        completed = self._post_object(
            AcceptanceActor.FIELD,
            f"/field/work-orders/{work_order_id}/complete",
            operation="field-complete",
            expected_statuses={200},
            headers={"If-Match": _etag(_required_int(current, "version"))},
            json_body={
                "root_cause": scenario.root_cause,
                "cost_amount": scenario.cost_amount,
            },
        )
        if str(completed.get("status", "")).upper() != "COMPLETED":
            raise ProjectAcceptanceFailure("field_completion_incomplete")
        return {
            "field_evidence_upload_id": evidence_upload_id,
            "field_evidence_entry_id": _required_id(evidence_entry, "entry_id"),
            "field_step_entry_id": _required_id(step_entry, "entry_id"),
            "field_signature_entry_id": _required_id(signature_entry, "entry_id"),
            "work_order_version": _required_int(completed, "version"),
        }

    def _result_acceptance(
        self,
        checkpoint: AcceptanceCheckpoint,
    ) -> dict[str, str | int | bool]:
        work_order_id = _fact_id(checkpoint, "work_order_id")
        incident_id = _fact_id(checkpoint, "incident_id")
        scenario = self._manifest.manifest.business_scenario
        current = self._get_object(
            AcceptanceActor.VERIFIER,
            f"/work-orders/{work_order_id}",
        )
        verified = self._post_object(
            AcceptanceActor.VERIFIER,
            f"/work-orders/{work_order_id}/verify",
            operation="independent-verification",
            expected_statuses={200},
            headers={"If-Match": _etag(_required_int(current, "version"))},
            json_body={
                "passed": True,
                "reason": "Independent project-staging repair result verification passed.",
            },
        )
        if str(verified.get("status", "")).upper() != "VERIFIED":
            raise ProjectAcceptanceFailure("independent_verification_incomplete")
        customer_update = self._post_object(
            AcceptanceActor.CUSTOMER,
            f"/portal/cases/{incident_id}/updates",
            operation="customer-result-confirmation",
            expected_statuses={200},
            json_body={
                "client_operation_id": self._idempotency_key("customer-result-confirmation"),
                "update_type": "RESULT_CONFIRMATION",
                "message": scenario.customer_result_confirmation,
                "work_order_id": work_order_id,
                "result_accepted": True,
                "satisfaction_rating": scenario.satisfaction_rating,
            },
        )
        customer_update_id = _required_id(customer_update, "update_id")
        current = self._get_object(
            AcceptanceActor.AFTER_SALES,
            f"/work-orders/{work_order_id}",
        )
        closure = self._post_object(
            AcceptanceActor.AFTER_SALES,
            f"/work-orders/{work_order_id}/closure-proposals",
            operation="closure-proposal",
            expected_statuses={201},
            json_body={
                "work_order_version": _required_int(current, "version"),
                "reason": "Completion, verification, and customer confirmation reconciled.",
            },
        )
        close_execution = self._approve_and_execute("closure", closure)
        if _required_id(close_execution, "work_order_id") != work_order_id:
            raise ProjectAcceptanceFailure("closure_work_order_binding_mismatch")
        closed_work = self._get_object(
            AcceptanceActor.AFTER_SALES,
            f"/work-orders/{work_order_id}",
        )
        if str(closed_work.get("status", "")).upper() != "CLOSED":
            raise ProjectAcceptanceFailure("work_order_not_closed")
        current_incident = self._get_object(
            AcceptanceActor.AFTER_SALES,
            f"/incidents/{incident_id}",
        )
        if str(current_incident.get("status", "")).upper() != "RESOLVED":
            raise ProjectAcceptanceFailure("incident_not_resolved")
        closed_incident = self._post_object(
            AcceptanceActor.AFTER_SALES,
            f"/incidents/{incident_id}/closure",
            operation="incident-closure",
            expected_statuses={200},
            headers={"If-Match": _etag(_required_int(current_incident, "version"))},
            json_body={
                "confirmation_type": "CUSTOMER",
                "customer_update_id": customer_update_id,
                "reason": "Customer accepted the verified project-staging repair result.",
            },
        )
        if str(closed_incident.get("status", "")).upper() != "CLOSED":
            raise ProjectAcceptanceFailure("incident_not_closed")
        return {
            "customer_update_id": customer_update_id,
            "closure_proposal_id": _required_id(closure, "proposal_id"),
            "closure_approval_id": _required_id(closure, "approval_id"),
            "closure_operation_id": _required_id(closure, "operation_id"),
            "work_order_status": "CLOSED",
            "work_order_version": _required_int(closed_work, "version"),
            "incident_status": "CLOSED",
            "incident_version": _required_int(closed_incident, "version"),
        }

    def _receipt(
        self,
        checkpoint: AcceptanceCheckpoint,
    ) -> dict[str, str | int | bool]:
        work_order_id = _fact_id(checkpoint, "work_order_id")
        first = self._get_object(
            AcceptanceActor.AFTER_SALES,
            f"/work-orders/{work_order_id}/closure-report",
        )
        second = self._get_object(
            AcceptanceActor.AFTER_SALES,
            f"/work-orders/{work_order_id}/closure-report",
        )
        first_digest = _required_str(first, "closure_facts_digest")
        second_digest = _required_str(second, "closure_facts_digest")
        expected = {
            "work_order_id": work_order_id,
            "incident_id": _fact_id(checkpoint, "incident_id"),
            "asset_id": checkpoint.asset_id,
            "site_id": checkpoint.site_id,
            "close_operation_id": _fact_id(checkpoint, "closure_operation_id"),
            "close_proposal_id": _fact_id(checkpoint, "closure_proposal_id"),
            "completed_by_subject_id": "subject-m1-engineer",
            "verifier_subject_id": "subject-project-acceptance-verifier",
            "customer_result_confirmation_id": _fact_id(checkpoint, "customer_update_id"),
            "customer_confirmed_by_subject_id": "subject-project-acceptance-customer",
        }
        if (
            first_digest != second_digest
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", first_digest)
            or first.get("report_contract_version") != "work-order-closure-report/v1"
            or first.get("customer_result_accepted") is not True
            or any(first.get(name) != value for name, value in expected.items())
        ):
            raise ProjectAcceptanceFailure("closure_report_drift")
        return {
            "closure_facts_digest": first_digest,
            "completion_id": _required_id(first, "completion_id"),
            "verification_id": _required_id(first, "verification_id"),
        }

    def _reconcile_model_evidence(
        self,
        checkpoint: AcceptanceCheckpoint,
    ) -> None:
        incident_id = _fact_id(checkpoint, "incident_id")
        diagnosis_id = _fact_id(checkpoint, "diagnosis_run_id")
        draft_id = _fact_id(checkpoint, "draft_id")
        incident = self._get_object(
            AcceptanceActor.AFTER_SALES,
            f"/incidents/{incident_id}",
        )
        allowed_statuses = {"DIAGNOSED", "WORK_ORDER_CREATED", "RESOLVED", "CLOSED"}
        if (
            incident.get("asset_id") != checkpoint.asset_id
            or incident.get("reporter_subject_id") != "subject-project-acceptance-customer"
            or str(incident.get("status", "")).upper() not in allowed_statuses
            or diagnosis_id not in incident.get("diagnosis_run_ids", [])
        ):
            raise ProjectAcceptanceFailure("resume_incident_binding_mismatch")
        diagnosis = self._get_object(
            AcceptanceActor.AFTER_SALES,
            f"/diagnosis-runs/{diagnosis_id}",
        )
        if str(diagnosis.get("status", "")).upper() != "COMPLETED":
            raise ProjectAcceptanceFailure("resume_diagnosis_incomplete")
        execution = self._validate_execution(
            diagnosis.get("model_execution"),
            component="diagnosis",
            request_class="DIAGNOSIS",
        )
        if execution.get("inference_request_id") != checkpoint.facts.get(
            "diagnosis_inference_request_id"
        ):
            raise ProjectAcceptanceFailure("resume_diagnosis_binding_mismatch")
        evidence = self._get_object(
            AcceptanceActor.CUSTOMER,
            f"/incident-drafts/{draft_id}/evidence",
        )
        vlm = self._validate_recognition_evidence(evidence)
        if evidence.get("bundle_id") != checkpoint.facts.get("evidence_bundle_id") or vlm.get(
            "inference_request_id"
        ) != checkpoint.facts.get("vlm_inference_request_id"):
            raise ProjectAcceptanceFailure("resume_recognition_binding_mismatch")

    def _reconcile_quotation(self, checkpoint: AcceptanceCheckpoint) -> None:
        incident_id = _fact_id(checkpoint, "incident_id")
        quotation_id = _fact_id(checkpoint, "service_quotation_id")
        quotations = self._get_list(
            AcceptanceActor.AFTER_SALES,
            f"/incidents/{incident_id}/service-quotations",
        )
        matching = next(
            (
                item
                for item in quotations
                if item.get("quotation_id") == quotation_id
                and str(item.get("status", "")).upper() == "ACCEPTED"
            ),
            None,
        )
        if matching is None:
            raise ProjectAcceptanceFailure("resume_quotation_binding_mismatch")

    def _reconcile_work_order(
        self,
        checkpoint: AcceptanceCheckpoint,
        *,
        minimum_status: str,
    ) -> None:
        work_order_id = _fact_id(checkpoint, "work_order_id")
        work_order = self._get_object(
            AcceptanceActor.AFTER_SALES,
            f"/work-orders/{work_order_id}",
        )
        self._validate_no_parts_work_order(
            work_order,
            incident_id=_fact_id(checkpoint, "incident_id"),
            minimum_status=minimum_status,
        )
        if minimum_status != "READY" and work_order.get("assigned_subject_id") != (
            self._manifest.manifest.business_scenario.field_assignee_subject_id
        ):
            raise ProjectAcceptanceFailure("resume_assignment_binding_mismatch")

    def _reconcile_closed_state(self, checkpoint: AcceptanceCheckpoint) -> None:
        work_order_id = _fact_id(checkpoint, "work_order_id")
        incident_id = _fact_id(checkpoint, "incident_id")
        work_order = self._get_object(
            AcceptanceActor.AFTER_SALES,
            f"/work-orders/{work_order_id}",
        )
        incident = self._get_object(
            AcceptanceActor.AFTER_SALES,
            f"/incidents/{incident_id}",
        )
        if (
            str(work_order.get("status", "")).upper() != "CLOSED"
            or str(incident.get("status", "")).upper() != "CLOSED"
            or _required_int(work_order, "version")
            != _fact_int(checkpoint, "work_order_version")
            or _required_int(incident, "version") != _fact_int(checkpoint, "incident_version")
        ):
            raise ProjectAcceptanceFailure("resume_closed_state_mismatch")

    def _reconcile_report(self, checkpoint: AcceptanceCheckpoint) -> None:
        report_updates = self._receipt(checkpoint)
        if any(checkpoint.facts.get(name) != value for name, value in report_updates.items()):
            raise ProjectAcceptanceFailure("resume_closure_report_mismatch")

    def _validate_recognition_evidence(
        self,
        evidence: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if evidence.get("source_sha256") not in {
            None,
            self._manifest.source_image_sha256,
        }:
            raise ProjectAcceptanceFailure("recognition_source_digest_mismatch")
        processor = _ocr_processor(evidence)
        normalized_processor = processor.lower()
        if any(
            marker in normalized_processor
            for marker in ("development", "fallback", "mock", "synthetic")
        ):
            raise ProjectAcceptanceFailure("ocr_processor_not_release_bound")
        processors = _mapping(
            evidence.get("processor_versions"),
            reason="ocr_processor_missing",
        )
        if processors.get("ocr.release_id") != self._expected_release_id:
            raise ProjectAcceptanceFailure("ocr_release_binding_mismatch")
        ocr_text = " ".join(
            str(block.get("text", ""))
            for block in _mapping_list(evidence.get("ocr_blocks"), reason="ocr_blocks_invalid")
        )
        if any(
            term not in ocr_text
            for term in self._manifest.manifest.model_expectations.expected_ocr_terms
        ):
            raise ProjectAcceptanceFailure("ocr_expected_terms_missing")
        labels = {
            str(item.get("label", ""))
            for item in _mapping_list(
                evidence.get("visual_findings"),
                reason="visual_findings_invalid",
            )
        }
        if not set(self._manifest.manifest.model_expectations.expected_visual_labels).issubset(
            labels
        ):
            raise ProjectAcceptanceFailure("visual_finding_expected_label_missing")
        executions = _mapping_list(
            evidence.get("model_executions"),
            reason="vlm_execution_missing",
        )
        execution = next(
            (item for item in executions if item.get("component") == "vlm"),
            None,
        )
        return self._validate_execution(
            execution,
            component="vlm",
            request_class="VLM",
        )

    def _validate_execution(
        self,
        raw: object,
        *,
        component: str,
        request_class: str,
    ) -> Mapping[str, Any]:
        execution = _mapping(raw, reason=f"{component}_execution_missing")
        alias = execution.get("model_alias")
        expected_alias = (
            self._manifest.manifest.model_expectations.vlm_alias
            if component == "vlm"
            else self._manifest.manifest.model_expectations.diagnosis_alias
        )
        if alias is not None and alias != expected_alias:
            raise ProjectAcceptanceFailure(f"{component}_execution_mismatch")
        if (
            execution.get("component") != component
            or execution.get("request_class") != request_class
            or execution.get("target_environment") != "STAGING"
            or execution.get("release_id") != self._expected_release_id
            or execution.get("manifest_hash") != self._expected_model_manifest_hash
            or execution.get("guardrail_decision") != "ALLOWED"
            or not _positive_number(execution.get("latency_ms"))
            or not _positive_int(execution.get("prompt_tokens"))
            or not _positive_int(execution.get("completion_tokens"))
        ):
            raise ProjectAcceptanceFailure(f"{component}_execution_mismatch")
        _required_id(execution, "inference_request_id")
        return execution

    def _validate_no_parts_work_order(
        self,
        work_order: Mapping[str, Any],
        *,
        incident_id: str,
        minimum_status: str,
    ) -> None:
        status_order = {
            "READY": 0,
            "ASSIGNED": 1,
            "ACCEPTED": 2,
            "IN_PROGRESS": 3,
            "COMPLETED": 4,
            "VERIFIED": 5,
            "CLOSED": 6,
        }
        status = str(work_order.get("status", "")).upper()
        if (
            work_order.get("incident_id") != incident_id
            or work_order.get("creation_mode") != "SERVICE_AUTHORIZATION"
            or work_order.get("authorization_type") != "COVERED_SERVICE"
            or work_order.get("initial_parts_required") is not False
            or work_order.get("part_issue_required") is not False
            or work_order.get("part_accounting_required") is not False
            or status not in status_order
            or status_order[status] < status_order[minimum_status]
        ):
            raise ProjectAcceptanceFailure("no_parts_work_order_binding_mismatch")

    def _confirmation_body(
        self,
        evidence: Mapping[str, Any],
        *,
        bundle_id: str,
    ) -> dict[str, Any]:
        return {
            "bundle_id": bundle_id,
            "corrections": {
                _required_id(item, "entity_id"): str(
                    item.get("original_value") or item.get("normalized_value") or ""
                )
                for item in _mapping_list(
                    evidence.get("extracted_entities"),
                    reason="extracted_entities_invalid",
                )
                if item.get("validation_status") != "VALID"
            },
            "finding_dispositions": {
                _required_id(item, "finding_id"): "ACCEPTED"
                for item in _mapping_list(
                    evidence.get("visual_findings"),
                    reason="visual_findings_invalid",
                )
            },
            "ocr_block_decisions": {
                _required_id(item, "block_id"): {"disposition": "ACCEPTED"}
                for item in _mapping_list(
                    evidence.get("ocr_blocks"),
                    reason="ocr_blocks_invalid",
                )
            },
            "qr_code_decisions": {
                _required_id(item, "candidate_id"): {
                    "disposition": ("REJECTED" if item.get("security_findings") else "ACCEPTED")
                }
                for item in _mapping_list(
                    evidence.get("qr_codes", []),
                    reason="qr_codes_invalid",
                )
            },
        }

    def _approve_and_execute(
        self,
        operation: str,
        proposal: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        approval_id = _required_id(proposal, "approval_id")
        parameters = _mapping(
            proposal.get("parameters"),
            reason="approval_parameters_missing",
        )
        approved = self._post_object(
            AcceptanceActor.APPROVER,
            f"/approvals/{approval_id}/decisions",
            operation=f"{operation}-approval",
            expected_statuses={200},
            headers={"If-Match": _etag(1)},
            json_body={
                "decision": "APPROVED",
                "reason": f"Independent {operation} facts and risk boundary reviewed.",
            },
        )
        if str(approved.get("status", "")).upper() != "APPROVED":
            raise ProjectAcceptanceFailure(f"{operation}_approval_incomplete")
        approval_version = _required_int(approved, "version")
        while True:
            executed = self._post_object(
                AcceptanceActor.AFTER_SALES,
                f"/approvals/{approval_id}/execute",
                operation=f"{operation}-execute",
                expected_statuses={200, 202},
                headers={"If-Match": _etag(approval_version)},
                json_body={"parameters": dict(parameters)},
            )
            status = str(executed.get("status", "")).upper()
            if status == "SUCCEEDED":
                return executed
            if status != "RECONCILING":
                raise ProjectAcceptanceFailure(f"{operation}_execution_failed:{status}")
            self._wait_tick(f"{operation}_execution")

    def _field_entry(
        self,
        work_order_id: str,
        *,
        operation: str,
        body: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        operation_id = self._idempotency_key(operation)
        return self._post_object(
            AcceptanceActor.FIELD,
            f"/field/work-orders/{work_order_id}/entries",
            operation=operation,
            expected_statuses={200},
            json_body={
                "client_operation_id": operation_id,
                "occurred_at": datetime.now(UTC).isoformat(),
                **body,
            },
        )

    def _wait_for_field_upload(
        self,
        work_order_id: str,
        evidence_upload_id: str,
    ) -> Mapping[str, Any]:
        while True:
            uploads = self._get_list(
                AcceptanceActor.FIELD,
                f"/field/work-orders/{work_order_id}/evidence-uploads",
            )
            upload = next(
                (item for item in uploads if item.get("evidence_upload_id") == evidence_upload_id),
                None,
            )
            if upload is None:
                raise ProjectAcceptanceFailure("field_evidence_upload_missing")
            state = str(upload.get("scan_state", "")).upper()
            if state == "CLEAN" and upload.get("ready_to_attach") is True:
                return upload
            if state in {"REJECTED", "INFECTED", "FAILED"}:
                raise ProjectAcceptanceFailure(f"field_evidence_scan_failed:{state}")
            self._wait_tick("field_evidence_scan")

    def _wait_object(
        self,
        actor: AcceptanceActor,
        path: str,
        *,
        status_field: str,
        success: set[str],
        failures: set[str],
        reason: str,
    ) -> Mapping[str, Any]:
        transient_failures = 0
        while True:
            try:
                data = self._get_object(actor, path)
                transient_failures = 0
            except ProjectAcceptanceFailure as exc:
                if exc.reason != "api_transport_unavailable" or transient_failures >= 2:
                    raise
                transient_failures += 1
                self._wait_tick(reason)
                continue
            status = str(data.get(status_field, "")).upper()
            if status in success:
                return data
            if status in failures:
                raise ProjectAcceptanceFailure(f"{reason}_failed:{status}")
            self._wait_tick(reason)

    def _wait_tick(self, reason: str) -> None:
        if self._monotonic() >= self._deadline:
            raise ProjectAcceptanceFailure(f"{reason}_timeout")
        self._sleep(self._poll_seconds)

    def _read_bound_image(self) -> bytes:
        try:
            content = self._manifest.source_image_path.read_bytes()
        except OSError:
            raise ProjectAcceptanceFailure("source_image_unavailable") from None
        if hashlib.sha256(content).hexdigest() != self._manifest.source_image_sha256:
            raise ProjectAcceptanceFailure("source_image_digest_drift")
        return content

    def _idempotency_key(self, operation: str) -> str:
        normalized = re.sub(r"[^a-z0-9-]+", "-", operation.lower()).strip("-")
        key = f"project-acceptance-{self._run_id}-{normalized}"
        if not normalized or len(key) > 128:
            raise ProjectAcceptanceFailure("idempotency_operation_invalid")
        return key

    def _get_response(
        self,
        actor: AcceptanceActor,
        path: str,
        *,
        expected_statuses: set[int],
    ) -> ApiResponse:
        return self._request(
            actor,
            "GET",
            path,
            expected_statuses=expected_statuses,
        )

    def _get_object(
        self,
        actor: AcceptanceActor,
        path: str,
    ) -> Mapping[str, Any]:
        response = self._get_response(actor, path, expected_statuses={200})
        return _object_data(response.data)

    def _get_list(
        self,
        actor: AcceptanceActor,
        path: str,
    ) -> list[Mapping[str, Any]]:
        response = self._get_response(actor, path, expected_statuses={200})
        return _list_data(response.data)

    def _post_object(
        self,
        actor: AcceptanceActor,
        path: str,
        *,
        operation: str,
        expected_statuses: set[int],
        json_body: Mapping[str, Any] | None = None,
        content: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        response = self._request(
            actor,
            "POST",
            path,
            operation=operation,
            expected_statuses=expected_statuses,
            json_body=json_body,
            content=content,
            headers=headers,
        )
        return _object_data(response.data)

    def _request(
        self,
        actor: AcceptanceActor,
        method: Literal["GET", "POST"],
        path: str,
        *,
        expected_statuses: set[int],
        operation: str | None = None,
        json_body: Mapping[str, Any] | None = None,
        content: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> ApiResponse:
        _validate_public_path(path)
        request_headers = dict(headers or {})
        if method != "GET":
            if operation is None:
                raise ProjectAcceptanceFailure("mutation_idempotency_missing")
            request_headers["Idempotency-Key"] = self._idempotency_key(operation)
        response = self._transport.request(
            ApiRequest(
                actor=actor,
                method=method,
                path=path,
                expected_statuses=frozenset(expected_statuses),
                json_body=json_body,
                content=content,
                headers=request_headers,
            )
        )
        if response.status not in expected_statuses:
            raise ProjectAcceptanceFailure(f"api_unexpected_status:{response.status}")
        if not isinstance(response.data, (Mapping, list)):
            raise ProjectAcceptanceFailure("api_data_invalid")
        return response


def _validate_public_path(path: str) -> None:
    parsed = urllib.parse.urlsplit(path)
    if (
        not path.startswith("/")
        or path.startswith("//")
        or parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or any(part in {".", ".."} for part in parsed.path.split("/"))
        or any(fragment in path.lower() for fragment in FORBIDDEN_PUBLIC_PATH_FRAGMENTS)
    ):
        raise ProjectAcceptanceFailure("api_path_invalid")


def _safe_http_error_code(raw: bytes) -> str:
    if len(raw) > MAX_ERROR_BYTES:
        return "error_body_too_large"
    try:
        document = json.loads(raw.decode("utf-8"))
        error = document.get("error", {}) if isinstance(document, dict) else {}
        code = error.get("code") if isinstance(error, dict) else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "non_json_error"
    return (
        str(code)
        if isinstance(code, str) and SAFE_ERROR_CODE_PATTERN.fullmatch(code)
        else "unknown_error"
    )


def _object_data(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectAcceptanceFailure("api_data_invalid")
    return value


def _list_data(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ProjectAcceptanceFailure("api_data_invalid")
    return list(value)


def _mapping(value: object, *, reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectAcceptanceFailure(reason)
    return value


def _mapping_list(value: object, *, reason: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ProjectAcceptanceFailure(reason)
    return list(value)


def _required_str(value: Mapping[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise ProjectAcceptanceFailure(f"api_field_invalid:{name}")
    return item


def _required_id(value: Mapping[str, Any], name: str) -> str:
    item = _required_str(value, name)
    if not _is_safe_identifier(item):
        raise ProjectAcceptanceFailure(f"api_identifier_invalid:{name}")
    return item


def _required_int(value: Mapping[str, Any], name: str) -> int:
    item = value.get(name)
    if isinstance(item, bool) or not isinstance(item, int) or item < 1:
        raise ProjectAcceptanceFailure(f"api_field_invalid:{name}")
    return item


def _positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _positive_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value > 0


def _is_safe_identifier(value: str) -> bool:
    return bool(ID_PATTERN.fullmatch(value))


def _etag(version: int) -> str:
    if version < 1:
        raise ProjectAcceptanceFailure("resource_version_invalid")
    return f'"{version}"'


def _response_etag(response: ApiResponse) -> str:
    value = next(
        (header_value for name, header_value in response.headers.items() if name.lower() == "etag"),
        None,
    )
    if not isinstance(value, str) or not re.fullmatch(r'"[1-9][0-9]*"', value):
        raise ProjectAcceptanceFailure("response_etag_missing")
    return value


def _select_exact_option(
    raw: object,
    expected: Mapping[str, object],
    *,
    reason: str,
) -> Mapping[str, Any]:
    options = _mapping_list(raw, reason=reason)
    matches = [
        option
        for option in options
        if all(option.get(name) == value for name, value in expected.items())
    ]
    if len(matches) != 1:
        raise ProjectAcceptanceFailure(reason)
    return matches[0]


def _fact_id(checkpoint: AcceptanceCheckpoint, name: str) -> str:
    value = checkpoint.facts.get(name)
    if not isinstance(value, str) or not _is_safe_identifier(value):
        raise ProjectAcceptanceFailure(f"checkpoint_fact_missing:{name}")
    return value


def _fact_int(checkpoint: AcceptanceCheckpoint, name: str) -> int:
    value = checkpoint.facts.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProjectAcceptanceFailure(f"checkpoint_fact_missing:{name}")
    return value


def _ocr_processor(evidence: Mapping[str, Any]) -> str:
    processors = _mapping(
        evidence.get("processor_versions"),
        reason="ocr_processor_missing",
    )
    value = processors.get("ocr")
    if not isinstance(value, str) or not value:
        raise ProjectAcceptanceFailure("ocr_processor_missing")
    return value
