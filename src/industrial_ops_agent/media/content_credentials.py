"""Bounded C2PA inspection and signing for industrial media provenance."""

from __future__ import annotations

import importlib
import io
import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Any, Protocol

CONTENT_CREDENTIAL_REPORT_VERSION = "media-content-credential/v1"
PLATFORM_SYNTHETIC_REVIEW_ASSERTION = "org.cppagan.industrial-ops.synthetic-review"
SUPPORTED_CONTENT_CREDENTIAL_MIME_TYPES = frozenset(
    {
        "application/pdf",
        "audio/flac",
        "audio/wav",
        "image/jpeg",
        "image/png",
        "video/mp4",
    }
)


class ContentCredentialStatus(StrEnum):
    PENDING_INSPECTION = "PENDING_INSPECTION"
    NOT_INSPECTED = "NOT_INSPECTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    ABSENT = "ABSENT"
    PRESENT_TRUSTED = "PRESENT_TRUSTED"
    PRESENT_VALID = "PRESENT_VALID"
    PRESENT_UNVERIFIED = "PRESENT_UNVERIFIED"
    PRESENT_INVALID = "PRESENT_INVALID"
    INSPECTION_UNAVAILABLE = "INSPECTION_UNAVAILABLE"
    INSPECTION_ERROR = "INSPECTION_ERROR"


@dataclass(frozen=True, slots=True)
class ContentCredentialReport:
    status: ContentCredentialStatus
    sdk_version: str | None = None
    validation_state: str | None = None
    active_manifest: str | None = None
    claim_generator: str | None = None
    actions: tuple[str, ...] = ()
    digital_source_types: tuple[str, ...] = ()
    validation_results_digest: str | None = None
    reviewed_manifest_hash: str | None = None
    reviewer_subject_hash: str | None = None
    generator_release_id: str | None = None
    evidence_eligible: bool | None = None
    golden_dataset_eligible: bool | None = None
    reason_code: str | None = None

    def document(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": CONTENT_CREDENTIAL_REPORT_VERSION,
            "status": self.status.value,
            "sdk_version": self.sdk_version,
            "validation_state": self.validation_state,
            "active_manifest": self.active_manifest,
            "claim_generator": self.claim_generator,
            "actions": list(self.actions),
            "digital_source_types": list(self.digital_source_types),
            "validation_results_digest": self.validation_results_digest,
            "reviewed_manifest_hash": self.reviewed_manifest_hash,
            "reviewer_subject_hash": self.reviewer_subject_hash,
            "generator_release_id": self.generator_release_id,
            "evidence_eligible": self.evidence_eligible,
            "golden_dataset_eligible": self.golden_dataset_eligible,
            "reason_code": self.reason_code,
            "absence_is_not_fraud_evidence": True,
        }
        payload["report_digest"] = _canonical_digest(payload)
        return payload


class ContentCredentialInspector(Protocol):
    def inspect(self, content: bytes, media_type: str) -> ContentCredentialReport: ...


@dataclass(frozen=True, slots=True)
class C2paContentCredentialInspector:
    """Inspect embedded credentials without fetching remote manifests."""

    trust_anchors_pem: str | None = None

    def inspect(self, content: bytes, media_type: str) -> ContentCredentialReport:
        normalized_mime = media_type.strip().casefold()
        if normalized_mime not in SUPPORTED_CONTENT_CREDENTIAL_MIME_TYPES:
            return ContentCredentialReport(ContentCredentialStatus.NOT_SUPPORTED)
        try:
            c2pa = importlib.import_module("c2pa")
        except ImportError:
            return ContentCredentialReport(
                ContentCredentialStatus.INSPECTION_UNAVAILABLE,
                reason_code="c2pa_runtime_unavailable",
            )
        settings_document: dict[str, Any] = {
            "verify": {
                "remote_manifest_fetch": False,
                "verify_cert_anchors": self.trust_anchors_pem is not None,
            }
        }
        if self.trust_anchors_pem is not None:
            settings_document["trust"] = {"trust_anchors": self.trust_anchors_pem}
        try:
            settings = c2pa.Settings.from_dict(settings_document)
            with c2pa.Context(settings) as context:
                reader = c2pa.Reader.try_create(
                    normalized_mime,
                    io.BytesIO(content),
                    context=context,
                )
                if reader is None:
                    return ContentCredentialReport(
                        ContentCredentialStatus.ABSENT,
                        sdk_version=str(c2pa.sdk_version()),
                    )
                with reader:
                    active = reader.get_active_manifest()
                    manifest_store = _json_object(reader.json())
                    if isinstance(active, dict):
                        active = dict(active)
                        active.setdefault("label", manifest_store.get("active_manifest"))
                    validation_state = str(reader.get_validation_state())
                    validation_results = reader.get_validation_results()
                    return _report_from_active_manifest(
                        active,
                        validation_state=validation_state,
                        validation_results=validation_results,
                        sdk_version=str(c2pa.sdk_version()),
                    )
        except Exception:
            return ContentCredentialReport(
                ContentCredentialStatus.INSPECTION_ERROR,
                sdk_version=_safe_sdk_version(c2pa),
                reason_code="c2pa_inspection_failed",
            )


@dataclass(frozen=True, slots=True)
class C2paSigningMaterial:
    algorithm: str
    certificate_pem: str
    sign_callback: Callable[[bytes], bytes]
    timestamp_authority_url: str | None = None


@dataclass(frozen=True, slots=True)
class CredentialedMedia:
    content: bytes
    report: ContentCredentialReport


class ContentCredentialSigner(Protocol):
    def sign_generated_media(
        self,
        *,
        generated_content: bytes,
        generated_media_type: str,
        source_content: bytes,
        source_media_type: str,
        title: str,
        reviewed_manifest_hash: str,
        reviewer_subject_hash: str,
        generator_release_id: str,
    ) -> CredentialedMedia: ...


@dataclass(frozen=True, slots=True)
class C2paSdkContentCredentialSigner:
    """Sign through a callback so production keys can remain in KMS or HSM."""

    material: C2paSigningMaterial
    inspector: ContentCredentialInspector

    def sign_generated_media(
        self,
        *,
        generated_content: bytes,
        generated_media_type: str,
        source_content: bytes,
        source_media_type: str,
        title: str,
        reviewed_manifest_hash: str,
        reviewer_subject_hash: str,
        generator_release_id: str,
    ) -> CredentialedMedia:
        if (
            generated_media_type not in SUPPORTED_CONTENT_CREDENTIAL_MIME_TYPES
            or source_media_type not in SUPPORTED_CONTENT_CREDENTIAL_MIME_TYPES
            or not _sha256_digest(reviewed_manifest_hash)
            or not _sha256_digest(reviewer_subject_hash)
            or not title.strip()
            or not generator_release_id.strip()
        ):
            raise ValueError("c2pa_signing_request_invalid")
        try:
            c2pa = importlib.import_module("c2pa")
            algorithm = getattr(c2pa.C2paSigningAlg, self.material.algorithm.upper())
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("c2pa_signing_runtime_unavailable") from exc
        manifest_definition = {
            "claim_generator_info": [
                {"name": "industrial-ops-agent-platform", "version": "0.1.0"}
            ],
            "title": title.strip(),
            "assertions": [
                {
                    "label": PLATFORM_SYNTHETIC_REVIEW_ASSERTION,
                    "data": {
                        "reviewed_manifest_hash": reviewed_manifest_hash,
                        "reviewer_subject_hash": reviewer_subject_hash,
                        "generator_release_id": generator_release_id,
                        "evidence_eligible": False,
                        "golden_dataset_eligible": False,
                    },
                }
            ],
        }
        output = io.BytesIO()
        try:
            with (
                c2pa.Context() as context,
                c2pa.Signer.from_callback(
                    self.material.sign_callback,
                    algorithm,
                    self.material.certificate_pem,
                    self.material.timestamp_authority_url,
                ) as signer,
                c2pa.Builder(manifest_definition, context) as builder,
            ):
                builder.set_intent(
                    c2pa.C2paBuilderIntent.CREATE,
                    c2pa.C2paDigitalSourceType.TRAINED_ALGORITHMIC_MEDIA,
                )
                ingredient = json.dumps(
                    {
                        "title": "governed-source-media",
                        "relationship": "componentOf",
                        "instance_id": f"sha256:{sha256(source_content).hexdigest()}",
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                builder.add_ingredient(
                    ingredient,
                    source_media_type,
                    io.BytesIO(source_content),
                )
                builder.sign(
                    signer,
                    generated_media_type,
                    io.BytesIO(generated_content),
                    output,
                )
        except Exception as exc:
            raise RuntimeError("c2pa_signing_failed") from exc
        signed = output.getvalue()
        report = self.inspector.inspect(signed, generated_media_type)
        if report.status not in {
            ContentCredentialStatus.PRESENT_TRUSTED,
            ContentCredentialStatus.PRESENT_VALID,
            ContentCredentialStatus.PRESENT_UNVERIFIED,
        }:
            raise RuntimeError("c2pa_signed_media_verification_failed")
        return CredentialedMedia(content=signed, report=report)


def _report_from_active_manifest(
    active: Any,
    *,
    validation_state: str,
    validation_results: Any,
    sdk_version: str,
) -> ContentCredentialReport:
    if not isinstance(active, dict):
        return ContentCredentialReport(
            ContentCredentialStatus.PRESENT_INVALID,
            sdk_version=sdk_version,
            validation_state=validation_state,
            reason_code="c2pa_active_manifest_invalid",
        )
    normalized_state = validation_state.casefold()
    status = {
        "trusted": ContentCredentialStatus.PRESENT_TRUSTED,
        "valid": ContentCredentialStatus.PRESENT_VALID,
        "invalid": ContentCredentialStatus.PRESENT_INVALID,
    }.get(normalized_state, ContentCredentialStatus.PRESENT_UNVERIFIED)
    actions, source_types = _extract_actions(active)
    review_binding = _extract_synthetic_review_binding(active)
    return ContentCredentialReport(
        status=status,
        sdk_version=sdk_version,
        validation_state=validation_state,
        active_manifest=_optional_text(active.get("label")),
        claim_generator=_claim_generator(active),
        actions=tuple(actions),
        digital_source_types=tuple(source_types),
        validation_results_digest=_canonical_digest(validation_results),
        reviewed_manifest_hash=_optional_text(
            review_binding.get("reviewed_manifest_hash")
        ),
        reviewer_subject_hash=_optional_text(
            review_binding.get("reviewer_subject_hash")
        ),
        generator_release_id=_optional_text(review_binding.get("generator_release_id")),
        evidence_eligible=_optional_bool(review_binding.get("evidence_eligible")),
        golden_dataset_eligible=_optional_bool(
            review_binding.get("golden_dataset_eligible")
        ),
    )


def _extract_actions(active: dict[str, Any]) -> tuple[list[str], list[str]]:
    actions: set[str] = set()
    source_types: set[str] = set()
    for assertion in _assertion_candidates(active):
        if not isinstance(assertion, dict):
            continue
        label = assertion.get("label")
        if not isinstance(label, str) or not (
            label == "c2pa.actions" or label.startswith("c2pa.actions.")
        ):
            continue
        data = assertion.get("data")
        values = data.get("actions") if isinstance(data, dict) else None
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, dict):
                continue
            action = _optional_text(value.get("action"))
            source_type = _optional_text(value.get("digitalSourceType"))
            if action is not None:
                actions.add(action)
            if source_type is not None:
                source_types.add(source_type)
    return sorted(actions), sorted(source_types)


def _extract_synthetic_review_binding(active: dict[str, Any]) -> dict[str, Any]:
    for assertion in _assertion_candidates(active):
        if not isinstance(assertion, dict):
            continue
        data = assertion.get("data")
        if (
            assertion.get("label") == PLATFORM_SYNTHETIC_REVIEW_ASSERTION
            and isinstance(data, dict)
        ):
            return {str(key): value for key, value in data.items()}
    return {}


def _assertion_candidates(active: dict[str, Any]) -> list[Any]:
    assertions = active.get("assertions", [])
    if isinstance(assertions, dict):
        candidates: list[Any] = []
        for label, assertion in assertions.items():
            if isinstance(assertion, dict) and "label" not in assertion:
                candidates.append({"label": label, **assertion})
            else:
                candidates.append(assertion)
        return candidates
    return assertions if isinstance(assertions, list) else []


def _claim_generator(active: dict[str, Any]) -> str | None:
    direct = _optional_text(active.get("claim_generator"))
    if direct is not None:
        return direct
    info = active.get("claim_generator_info")
    if not isinstance(info, list) or not info or not isinstance(info[0], dict):
        return None
    name = _optional_text(info[0].get("name"))
    version = _optional_text(info[0].get("version"))
    if name is None:
        return None
    return f"{name}/{version}" if version is not None else name


def _safe_sdk_version(c2pa: Any) -> str | None:
    try:
        return str(c2pa.sdk_version())
    except Exception:
        return None


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256(encoded).hexdigest()
