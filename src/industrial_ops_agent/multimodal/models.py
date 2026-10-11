"""Immutable OCR/VLM evidence and human-confirmation domain model."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from hashlib import sha256

from industrial_ops_agent.guardrails.prompt_injection import (
    PROMPT_INJECTION_POLICY_VERSION,
    PromptInjectionGuard,
)

CRITICAL_ENTITY_TYPES = frozenset(
    {"alarm_code", "serial_number", "part_number", "measurement", "unit"}
)
TEMPORAL_VIDEO_EVENT_TYPES = frozenset(
    {"motion_candidate", "signal_pattern_candidate", "action_sequence_candidate"}
)
REVIEWABLE_VIDEO_EVENT_TYPES = frozenset(
    {"visual_candidate", *TEMPORAL_VIDEO_EVENT_TYPES}
)


class EvidenceSchemaError(ValueError):
    """Processor output does not satisfy the evidence contract."""


class EvidenceVersionConflict(Exception):
    def __init__(self, current_version: int) -> None:
        super().__init__("evidence bundle version conflict")
        self.current_version = current_version


class EvidenceConfirmationRequired(Exception):
    def __init__(self, reason_code: str) -> None:
        super().__init__("evidence confirmation is incomplete")
        self.reason_code = reason_code


class EntityValidationStatus(StrEnum):
    VALID = "VALID"
    REQUIRES_CONFIRMATION = "REQUIRES_CONFIRMATION"
    INVALID = "INVALID"


class EvidenceStatus(StrEnum):
    READY_FOR_CONFIRMATION = "READY_FOR_CONFIRMATION"
    CONFIRMED = "CONFIRMED"


class FindingDisposition(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class OcrDisposition(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    CORRECTED = "CORRECTED"


class QrPayloadKind(StrEnum):
    TEXT = "TEXT"
    HTTPS_URL = "HTTPS_URL"
    OTHER_URI = "OTHER_URI"


class QrDisposition(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    CORRECTED = "CORRECTED"


class TranscriptDisposition(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


SECURITY_FINDING_SOURCE_TYPES = frozenset(
    {"OCR_BLOCK", "ASR_SEGMENT", "VISUAL_FINDING", "VIDEO_EVENT", "QR_CODE"}
)


@dataclass(frozen=True, slots=True)
class EvidenceSecurityFinding:
    source_type: str
    source_id: str
    policy_version: str
    pattern_id: str
    category: str
    severity: str
    content_hash: str

    def __post_init__(self) -> None:
        if (
            self.source_type not in SECURITY_FINDING_SOURCE_TYPES
            or not self.source_id.strip()
            or not self.policy_version.strip()
            or not self.pattern_id.strip()
            or not self.category.strip()
            or self.severity != "HIGH"
            or len(self.content_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.content_hash)
        ):
            raise EvidenceSchemaError("multimodal security finding is invalid")

    def as_dict(self) -> dict[str, str]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "policy_version": self.policy_version,
            "pattern_id": self.pattern_id,
            "category": self.category,
            "severity": self.severity,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Normalized source-region coordinates in the inclusive 0..1 image space."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if any(value < 0 or value > 1 for value in values):
            raise EvidenceSchemaError("bounding box coordinates must be normalized")
        if self.width <= 0 or self.height <= 0:
            raise EvidenceSchemaError("bounding box must have a positive area")
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise EvidenceSchemaError("bounding box exceeds the source image")


@dataclass(frozen=True, slots=True)
class OcrBlock:
    block_id: str
    page_number: int
    text: str
    bbox: BoundingBox
    confidence: float
    source_frame_id: str | None = None

    def __post_init__(self) -> None:
        if not self.block_id or self.page_number < 1 or not self.text.strip():
            raise EvidenceSchemaError("OCR block identity, page and text are required")
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class QrCodeCandidate:
    candidate_id: str
    text: str
    payload_kind: QrPayloadKind
    bbox: BoundingBox
    source_frame_id: str | None = None
    security_findings: tuple[EvidenceSecurityFinding, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.candidate_id.strip()
            or not self.text.strip()
            or len(self.text.encode("utf-8")) > 4_096
            or (self.source_frame_id is not None and not self.source_frame_id.strip())
        ):
            raise EvidenceSchemaError("QR candidate contract is invalid")
        identities: set[tuple[str, str, str]] = set()
        for finding in self.security_findings:
            identity = (finding.pattern_id, finding.category, finding.content_hash)
            if (
                finding.source_type != "QR_CODE"
                or finding.source_id != self.candidate_id
                or identity in identities
            ):
                raise EvidenceSchemaError("QR security finding source is invalid")
            identities.add(identity)


@dataclass(frozen=True, slots=True)
class ExtractedEntity:
    entity_id: str
    entity_type: str
    value: str
    normalized_value: str
    confidence: float
    validation_status: EntityValidationStatus
    source_block_id: str
    validation_reason: str | None = None

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.entity_id,
                self.entity_type,
                self.value,
                self.normalized_value,
                self.source_block_id,
            )
        ):
            raise EvidenceSchemaError("extracted entity fields cannot be empty")
        _validate_confidence(self.confidence)
        if (
            self.validation_status is not EntityValidationStatus.VALID
            and not self.validation_reason
        ):
            raise EvidenceSchemaError("non-valid entities require a validation reason")

    @property
    def critical(self) -> bool:
        return self.entity_type in CRITICAL_ENTITY_TYPES


@dataclass(frozen=True, slots=True)
class VisualFinding:
    finding_id: str
    label: str
    bbox: BoundingBox
    confidence: float
    evidence_level: str
    model_release_id: str
    description: str
    source_frame_id: str | None = None

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.finding_id,
                self.label,
                self.evidence_level,
                self.model_release_id,
                self.description,
            )
        ):
            raise EvidenceSchemaError("visual finding fields cannot be empty")
        if self.evidence_level not in {"candidate", "supporting"}:
            raise EvidenceSchemaError("VLM findings can only be candidate evidence")
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class AsrEntityCandidate:
    entity_id: str
    source_segment_id: str
    entity_type: str
    value: str
    normalized_value: str
    confidence: float
    start_offset: int
    end_offset: int
    requires_confirmation: bool

    def __post_init__(self) -> None:
        if (
            not self.entity_id.strip()
            or not self.source_segment_id.strip()
            or self.entity_type not in CRITICAL_ENTITY_TYPES
            or not self.value.strip()
            or not self.normalized_value.strip()
            or self.start_offset < 0
            or self.end_offset <= self.start_offset
        ):
            raise EvidenceSchemaError("ASR entity candidate contract is invalid")
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class AsrSegment:
    segment_id: str
    start_ms: int
    end_ms: int
    text: str
    language: str
    confidence: float
    model_release_id: str
    source_audio_track_id: str | None = None
    hotword_profile_id: str | None = None
    entity_candidates: tuple[AsrEntityCandidate, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.segment_id.strip()
            or self.start_ms < 0
            or self.end_ms <= self.start_ms
            or not self.text.strip()
            or self.language not in {"zh", "en"}
            or not self.model_release_id.strip()
            or (
                self.source_audio_track_id is not None
                and not self.source_audio_track_id.strip()
            )
            or (self.hotword_profile_id is not None and not self.hotword_profile_id.strip())
        ):
            raise EvidenceSchemaError("ASR segment contract is invalid")
        if (
            len({item.entity_id for item in self.entity_candidates})
            != len(self.entity_candidates)
            or any(item.source_segment_id != self.segment_id for item in self.entity_candidates)
            or any(item.end_offset > len(self.text) for item in self.entity_candidates)
            or any(
                self.text[item.start_offset : item.end_offset] != item.value
                for item in self.entity_candidates
            )
        ):
            raise EvidenceSchemaError("ASR entity candidate is not anchored to its segment")
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class VideoKeyframe:
    frame_id: str
    timestamp_ms: int
    image_sha256: str
    sampling_reason: str
    image_jpeg: bytes | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not self.frame_id.strip()
            or self.timestamp_ms < 0
            or len(self.image_sha256) != 64
            or self.sampling_reason not in {"first_frame", "periodic", "scene_change"}
        ):
            raise EvidenceSchemaError("video keyframe contract is invalid")
        if self.image_jpeg is not None and (
            len(self.image_jpeg) > 2 * 1024 * 1024
            or not self.image_jpeg.startswith(b"\xff\xd8\xff")
            or sha256(self.image_jpeg).hexdigest() != self.image_sha256
        ):
            raise EvidenceSchemaError("video keyframe artifact is invalid")


@dataclass(frozen=True, slots=True)
class VideoEvent:
    event_id: str
    event_type: str
    start_ms: int
    end_ms: int
    keyframe_ids: tuple[str, ...]
    confidence: float
    description: str
    finding_ids: tuple[str, ...] = ()
    model_release_id: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.event_id.strip()
            or self.event_type
            not in {
                "periodic_observation",
                "scene_change",
                "visual_candidate",
                *TEMPORAL_VIDEO_EVENT_TYPES,
            }
            or self.start_ms < 0
            or self.end_ms < self.start_ms
            or not self.keyframe_ids
            or not self.description.strip()
            or len(set(self.keyframe_ids)) != len(self.keyframe_ids)
            or len(set(self.finding_ids)) != len(self.finding_ids)
            or (self.event_type == "visual_candidate") != bool(self.finding_ids)
            or (self.event_type in TEMPORAL_VIDEO_EVENT_TYPES) != bool(self.model_release_id)
            or (
                self.event_type in TEMPORAL_VIDEO_EVENT_TYPES
                and (len(self.keyframe_ids) < 2 or self.end_ms <= self.start_ms)
            )
        ):
            raise EvidenceSchemaError("video event contract is invalid")
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class HumanCorrection:
    correction_id: str
    entity_id: str
    original_value: str
    corrected_value: str
    reviewer_subject_id: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class FindingReview:
    finding_id: str
    disposition: FindingDisposition
    reviewer_subject_id: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class OcrReview:
    block_id: str
    disposition: OcrDisposition
    original_text: str
    corrected_text: str | None
    reviewer_subject_id: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class QrReview:
    candidate_id: str
    disposition: QrDisposition
    original_text: str
    corrected_text: str | None
    reviewer_subject_id: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class TranscriptReview:
    segment_id: str
    disposition: TranscriptDisposition
    original_text: str
    corrected_text: str | None
    reviewer_subject_id: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class VideoEventReview:
    event_id: str
    disposition: FindingDisposition
    reviewer_subject_id: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    bundle_id: str
    tenant_id: str
    draft_id: str
    asset_id: str
    media_id: str
    source_sha256: str
    source_type: str
    ocr_blocks: tuple[OcrBlock, ...]
    asr_segments: tuple[AsrSegment, ...]
    visual_findings: tuple[VisualFinding, ...]
    video_keyframes: tuple[VideoKeyframe, ...]
    video_events: tuple[VideoEvent, ...]
    extracted_entities: tuple[ExtractedEntity, ...]
    human_corrections: tuple[HumanCorrection, ...]
    ocr_reviews: tuple[OcrReview, ...]
    finding_reviews: tuple[FindingReview, ...]
    transcript_reviews: tuple[TranscriptReview, ...]
    video_event_reviews: tuple[VideoEventReview, ...]
    processor_versions: dict[str, str]
    status: EvidenceStatus
    version: int
    created_at: datetime
    updated_at: datetime
    qr_codes: tuple[QrCodeCandidate, ...] = ()
    qr_reviews: tuple[QrReview, ...] = ()
    security_policy_version: str | None = None
    security_findings: tuple[EvidenceSecurityFinding, ...] = ()
    automation_eligible: bool = False
    automation_blockers: tuple[str, ...] = ("evidence_not_confirmed",)

    @property
    def automatic_diagnosis_eligible(self) -> bool:
        """Only confirmed, supported and safely reviewed evidence may enter automation."""

        return (
            self.status is EvidenceStatus.CONFIRMED
            and self.automation_eligible
            and self.processor_versions.get("vlm_status") != "unsupported_model"
        )

    @classmethod
    def create(
        cls,
        *,
        bundle_id: str,
        tenant_id: str,
        draft_id: str,
        asset_id: str,
        media_id: str,
        source_sha256: str,
        source_type: str,
        ocr_blocks: tuple[OcrBlock, ...],
        visual_findings: tuple[VisualFinding, ...],
        extracted_entities: tuple[ExtractedEntity, ...],
        processor_versions: dict[str, str],
        occurred_at: datetime,
        asr_segments: tuple[AsrSegment, ...] = (),
        video_keyframes: tuple[VideoKeyframe, ...] = (),
        video_events: tuple[VideoEvent, ...] = (),
        qr_codes: tuple[QrCodeCandidate, ...] = (),
        security_policy_version: str | None = PROMPT_INJECTION_POLICY_VERSION,
        security_findings: tuple[EvidenceSecurityFinding, ...] = (),
    ) -> EvidenceBundle:
        if not all(value.strip() for value in (bundle_id, tenant_id, draft_id, asset_id, media_id)):
            raise EvidenceSchemaError("evidence identity fields cannot be empty")
        if len(source_sha256) != 64:
            raise EvidenceSchemaError("evidence source checksum is invalid")
        if source_type not in {"image", "scanned_document", "video"}:
            raise EvidenceSchemaError("evidence source type is unsupported")
        if not processor_versions or any(
            not key.strip() or not value.strip() for key, value in processor_versions.items()
        ):
            raise EvidenceSchemaError("processor versions are required")
        block_ids = {block.block_id for block in ocr_blocks}
        if len(block_ids) != len(ocr_blocks):
            raise EvidenceSchemaError("OCR block IDs must be unique")
        if any(entity.source_block_id not in block_ids for entity in extracted_entities):
            raise EvidenceSchemaError("entity source block is missing")
        if len({item.entity_id for item in extracted_entities}) != len(extracted_entities):
            raise EvidenceSchemaError("entity IDs must be unique")
        if len({item.finding_id for item in visual_findings}) != len(visual_findings):
            raise EvidenceSchemaError("visual finding IDs must be unique")
        segment_ids = {item.segment_id for item in asr_segments}
        if len(segment_ids) != len(asr_segments):
            raise EvidenceSchemaError("ASR segment IDs must be unique")
        frame_ids = {item.frame_id for item in video_keyframes}
        if len(frame_ids) != len(video_keyframes):
            raise EvidenceSchemaError("video keyframe IDs must be unique")
        if len({item.event_id for item in video_events}) != len(video_events):
            raise EvidenceSchemaError("video event IDs must be unique")
        if any(not set(item.keyframe_ids).issubset(frame_ids) for item in video_events):
            raise EvidenceSchemaError("video event references a missing keyframe")
        finding_frames = {item.finding_id: item.source_frame_id for item in visual_findings}
        if any(not set(item.finding_ids).issubset(finding_frames) for item in video_events):
            raise EvidenceSchemaError("video event references a missing visual finding")
        if any(
            any(
                finding_frames[finding_id] not in item.keyframe_ids
                for finding_id in item.finding_ids
            )
            for item in video_events
        ):
            raise EvidenceSchemaError("video event finding is not anchored to its keyframes")
        frame_timestamps = {item.frame_id: item.timestamp_ms for item in video_keyframes}
        if any(
            item.start_ms > min(frame_timestamps[frame_id] for frame_id in item.keyframe_ids)
            or item.end_ms < max(frame_timestamps[frame_id] for frame_id in item.keyframe_ids)
            for item in video_events
        ):
            raise EvidenceSchemaError("video event time range does not cover its keyframes")
        if source_type == "video" and not video_keyframes:
            raise EvidenceSchemaError("video evidence requires keyframes")
        if source_type == "video" and any(item.image_jpeg is None for item in video_keyframes):
            raise EvidenceSchemaError("new video evidence requires keyframe artifacts")
        missing_frame_anchor = any(item.source_frame_id is None for item in ocr_blocks) or any(
            item.source_frame_id is None for item in visual_findings
        )
        if source_type == "video" and missing_frame_anchor:
            raise EvidenceSchemaError("video OCR and VLM results require a frame anchor")
        if source_type != "video" and (asr_segments or video_keyframes or video_events):
            raise EvidenceSchemaError("non-video evidence cannot contain video results")
        missing_frame_reference = any(
            item.source_frame_id is not None and item.source_frame_id not in frame_ids
            for item in ocr_blocks
        ) or any(
            item.source_frame_id is not None and item.source_frame_id not in frame_ids
            for item in visual_findings
        )
        if missing_frame_reference:
            raise EvidenceSchemaError("recognition result references a missing video frame")
        _validate_qr_codes(
            qr_codes,
            source_type=source_type,
            frame_ids=frame_ids,
            security_policy_version=security_policy_version,
        )
        _validate_security_finding_sources(
            security_policy_version,
            security_findings,
            ocr_blocks=ocr_blocks,
            asr_segments=asr_segments,
            visual_findings=visual_findings,
            video_events=video_events,
        )
        return cls(
            bundle_id=bundle_id,
            tenant_id=tenant_id,
            draft_id=draft_id,
            asset_id=asset_id,
            media_id=media_id,
            source_sha256=source_sha256,
            source_type=source_type,
            ocr_blocks=ocr_blocks,
            asr_segments=asr_segments,
            visual_findings=visual_findings,
            video_keyframes=video_keyframes,
            video_events=video_events,
            extracted_entities=extracted_entities,
            human_corrections=(),
            ocr_reviews=(),
            finding_reviews=(),
            transcript_reviews=(),
            video_event_reviews=(),
            processor_versions=dict(processor_versions),
            status=EvidenceStatus.READY_FOR_CONFIRMATION,
            version=1,
            created_at=occurred_at,
            updated_at=occurred_at,
            qr_codes=qr_codes,
            qr_reviews=(),
            security_policy_version=security_policy_version,
            security_findings=security_findings,
            automation_eligible=False,
            automation_blockers=_initial_automation_blockers(
                security_findings=security_findings,
                qr_codes=qr_codes,
            ),
        )

    def confirm(
        self,
        *,
        reviewer_subject_id: str,
        corrections: dict[str, str],
        finding_dispositions: dict[str, FindingDisposition],
        expected_version: int,
        occurred_at: datetime,
        transcript_decisions: dict[str, tuple[TranscriptDisposition, str | None]] | None = None,
        ocr_block_decisions: dict[str, tuple[OcrDisposition, str | None]] | None = None,
        video_event_dispositions: dict[str, FindingDisposition] | None = None,
        qr_code_decisions: dict[str, tuple[QrDisposition, str | None]] | None = None,
    ) -> EvidenceBundle:
        if expected_version != self.version:
            raise EvidenceVersionConflict(self.version)
        if self.status is EvidenceStatus.CONFIRMED:
            raise EvidenceConfirmationRequired("evidence_already_confirmed")
        if not reviewer_subject_id:
            raise EvidenceConfirmationRequired("reviewer_missing")
        if not self.security_policy_version:
            raise EvidenceConfirmationRequired("evidence_security_check_missing")

        entities = {entity.entity_id: entity for entity in self.extracted_entities}
        blocks = {block.block_id: block for block in self.ocr_blocks}
        findings = {finding.finding_id: finding for finding in self.visual_findings}
        segments = {segment.segment_id: segment for segment in self.asr_segments}
        qr_codes = {candidate.candidate_id: candidate for candidate in self.qr_codes}
        reviewable_video_events = {
            event.event_id: event
            for event in self.video_events
            if event.event_type in REVIEWABLE_VIDEO_EVENT_TYPES
        }
        transcript_decisions = transcript_decisions or {}
        if not set(corrections).issubset(entities):
            raise EvidenceConfirmationRequired("unknown_entity")
        if set(finding_dispositions) != set(findings):
            raise EvidenceConfirmationRequired("visual_findings_not_reviewed")
        if set(transcript_decisions) != set(segments):
            raise EvidenceConfirmationRequired("asr_segments_not_reviewed")
        if ocr_block_decisions is not None and set(ocr_block_decisions) != set(blocks):
            raise EvidenceConfirmationRequired("ocr_blocks_not_reviewed")
        if video_event_dispositions is not None and set(
            video_event_dispositions
        ) != set(reviewable_video_events):
            raise EvidenceConfirmationRequired("video_events_not_reviewed")
        qr_code_decisions = qr_code_decisions or {}
        if set(qr_code_decisions) != set(qr_codes):
            raise EvidenceConfirmationRequired("qr_codes_not_reviewed")

        _validate_security_dispositions(
            self,
            finding_dispositions=finding_dispositions,
            transcript_decisions=transcript_decisions,
            ocr_block_decisions=ocr_block_decisions or {},
            video_event_dispositions=video_event_dispositions or {},
        )

        correction_rows: list[HumanCorrection] = []
        for entity_id, corrected_value in sorted(corrections.items()):
            if not corrected_value.strip():
                raise EvidenceConfirmationRequired("empty_entity_correction")
            entity = entities[entity_id]
            correction_key = f"{self.bundle_id}\0{entity_id}\0{self.version + 1}\0{corrected_value}"
            correction_rows.append(
                HumanCorrection(
                    correction_id=f"correction-{sha256(correction_key.encode()).hexdigest()}",
                    entity_id=entity_id,
                    original_value=entity.value,
                    corrected_value=corrected_value.strip(),
                    reviewer_subject_id=reviewer_subject_id,
                    occurred_at=occurred_at,
                )
            )

        for entity in self.extracted_entities:
            if (
                entity.critical
                and entity.validation_status is not EntityValidationStatus.VALID
                and entity.entity_id not in corrections
            ):
                raise EvidenceConfirmationRequired("critical_entity_not_confirmed")

        reviews = tuple(
            FindingReview(
                finding_id=finding_id,
                disposition=disposition,
                reviewer_subject_id=reviewer_subject_id,
                occurred_at=occurred_at,
            )
            for finding_id, disposition in sorted(finding_dispositions.items())
        )
        ocr_reviews: list[OcrReview] = []
        for block_id, (disposition, corrected_text) in sorted(
            (ocr_block_decisions or {}).items()
        ):
            block = blocks[block_id]
            normalized_correction = corrected_text.strip() if corrected_text else None
            if disposition is OcrDisposition.CORRECTED and normalized_correction is None:
                raise EvidenceConfirmationRequired("ocr_block_correction_required")
            if disposition is not OcrDisposition.CORRECTED and normalized_correction is not None:
                raise EvidenceConfirmationRequired("ocr_block_correction_not_allowed")
            ocr_reviews.append(
                OcrReview(
                    block_id=block_id,
                    disposition=disposition,
                    original_text=block.text,
                    corrected_text=normalized_correction,
                    reviewer_subject_id=reviewer_subject_id,
                    occurred_at=occurred_at,
                )
            )
        qr_reviews: list[QrReview] = []
        guard = PromptInjectionGuard()
        for candidate_id, (qr_disposition, corrected_text) in sorted(
            qr_code_decisions.items()
        ):
            candidate = qr_codes[candidate_id]
            normalized_correction = corrected_text.strip() if corrected_text else None
            if qr_disposition is QrDisposition.CORRECTED and normalized_correction is None:
                raise EvidenceConfirmationRequired("qr_code_correction_required")
            if qr_disposition is not QrDisposition.CORRECTED and normalized_correction is not None:
                raise EvidenceConfirmationRequired("qr_code_correction_not_allowed")
            if candidate.security_findings and qr_disposition is QrDisposition.ACCEPTED:
                raise EvidenceConfirmationRequired("unsafe_evidence_accepted")
            if qr_disposition is QrDisposition.CORRECTED:
                _require_safe_correction(guard, normalized_correction, "qr_code")
            qr_reviews.append(
                QrReview(
                    candidate_id=candidate_id,
                    disposition=qr_disposition,
                    original_text=candidate.text,
                    corrected_text=normalized_correction,
                    reviewer_subject_id=reviewer_subject_id,
                    occurred_at=occurred_at,
                )
            )
        transcript_reviews: list[TranscriptReview] = []
        for segment_id, (transcript_disposition, corrected_text) in sorted(
            transcript_decisions.items()
        ):
            segment = segments[segment_id]
            normalized_correction = corrected_text.strip() if corrected_text else None
            if (
                transcript_disposition is TranscriptDisposition.ACCEPTED
                and any(item.requires_confirmation for item in segment.entity_candidates)
                and normalized_correction is None
            ):
                raise EvidenceConfirmationRequired("asr_entity_confirmation_required")
            if transcript_disposition is TranscriptDisposition.ACCEPTED and not (
                normalized_correction or segment.text.strip()
            ):
                raise EvidenceConfirmationRequired("asr_segment_text_missing")
            transcript_reviews.append(
                TranscriptReview(
                    segment_id=segment_id,
                    disposition=transcript_disposition,
                    original_text=segment.text,
                    corrected_text=normalized_correction,
                    reviewer_subject_id=reviewer_subject_id,
                    occurred_at=occurred_at,
                )
            )
        video_event_reviews = tuple(
            VideoEventReview(
                event_id=event_id,
                disposition=disposition,
                reviewer_subject_id=reviewer_subject_id,
                occurred_at=occurred_at,
            )
            for event_id, disposition in sorted(
                (video_event_dispositions or {}).items()
            )
        )
        return replace(
            self,
            human_corrections=tuple(correction_rows),
            ocr_reviews=tuple(ocr_reviews),
            finding_reviews=reviews,
            transcript_reviews=tuple(transcript_reviews),
            video_event_reviews=video_event_reviews,
            qr_reviews=tuple(qr_reviews),
            status=EvidenceStatus.CONFIRMED,
            version=self.version + 1,
            updated_at=occurred_at,
            automation_eligible=(
                self.processor_versions.get("vlm_status") != "unsupported_model"
            ),
            automation_blockers=(
                ()
                if self.processor_versions.get("vlm_status") != "unsupported_model"
                else ("unsupported_vlm_profile",)
            ),
    )


def _initial_automation_blockers(
    *,
    security_findings: tuple[EvidenceSecurityFinding, ...],
    qr_codes: tuple[QrCodeCandidate, ...],
) -> tuple[str, ...]:
    blockers: list[str] = []
    if security_findings:
        blockers.append("security_findings_unresolved")
    if qr_codes:
        blockers.append("qr_codes_not_reviewed")
    return tuple(blockers) or ("evidence_not_confirmed",)


def _validate_qr_codes(
    qr_codes: tuple[QrCodeCandidate, ...],
    *,
    source_type: str,
    frame_ids: set[str],
    security_policy_version: str | None,
) -> None:
    if len(qr_codes) > 384 or len({item.candidate_id for item in qr_codes}) != len(qr_codes):
        raise EvidenceSchemaError("QR candidate IDs are invalid")
    for candidate in qr_codes:
        if source_type == "video":
            if candidate.source_frame_id not in frame_ids:
                raise EvidenceSchemaError("video QR candidate requires a frame anchor")
        elif candidate.source_frame_id is not None:
            raise EvidenceSchemaError("image QR candidate cannot reference a video frame")
        if candidate.security_findings and not security_policy_version:
            raise EvidenceSchemaError("QR security findings require a policy version")
        if any(
            finding.policy_version != security_policy_version
            for finding in candidate.security_findings
        ):
            raise EvidenceSchemaError("QR security policy version is invalid")


def _validate_security_finding_sources(
    policy_version: str | None,
    findings: tuple[EvidenceSecurityFinding, ...],
    *,
    ocr_blocks: tuple[OcrBlock, ...],
    asr_segments: tuple[AsrSegment, ...],
    visual_findings: tuple[VisualFinding, ...],
    video_events: tuple[VideoEvent, ...],
) -> None:
    if findings and not policy_version:
        raise EvidenceSchemaError("security findings require a policy version")
    source_ids = {
        "OCR_BLOCK": {item.block_id for item in ocr_blocks},
        "ASR_SEGMENT": {item.segment_id for item in asr_segments},
        "VISUAL_FINDING": {item.finding_id for item in visual_findings},
        "VIDEO_EVENT": {item.event_id for item in video_events},
    }
    identities: set[tuple[str, str, str, str]] = set()
    for finding in findings:
        identity = (
            finding.source_type,
            finding.source_id,
            finding.pattern_id,
            finding.content_hash,
        )
        if (
            finding.policy_version != policy_version
            or finding.source_id not in source_ids[finding.source_type]
            or identity in identities
        ):
            raise EvidenceSchemaError("multimodal security finding source is invalid")
        identities.add(identity)


def _validate_security_dispositions(
    bundle: EvidenceBundle,
    *,
    finding_dispositions: dict[str, FindingDisposition],
    transcript_decisions: dict[str, tuple[TranscriptDisposition, str | None]],
    ocr_block_decisions: dict[str, tuple[OcrDisposition, str | None]],
    video_event_dispositions: dict[str, FindingDisposition],
) -> None:
    if not bundle.security_findings:
        return
    guard = PromptInjectionGuard()
    unsafe_sources = {
        (finding.source_type, finding.source_id)
        for finding in bundle.security_findings
    }
    for source_type, source_id in sorted(unsafe_sources):
        if source_type == "OCR_BLOCK":
            ocr_decision = ocr_block_decisions.get(source_id)
            if ocr_decision is None or ocr_decision[0] is OcrDisposition.ACCEPTED:
                raise EvidenceConfirmationRequired("unsafe_evidence_accepted")
            if ocr_decision[0] is OcrDisposition.CORRECTED:
                _require_safe_correction(guard, ocr_decision[1], "ocr_block")
        elif source_type == "ASR_SEGMENT":
            transcript_decision = transcript_decisions.get(source_id)
            if transcript_decision is None:
                raise EvidenceConfirmationRequired("unsafe_evidence_accepted")
            if transcript_decision[0] is TranscriptDisposition.ACCEPTED:
                _require_safe_correction(guard, transcript_decision[1], "asr_segment")
        elif source_type == "VISUAL_FINDING":
            if finding_dispositions.get(source_id) is not FindingDisposition.REJECTED:
                raise EvidenceConfirmationRequired("unsafe_evidence_accepted")
        elif source_type == "VIDEO_EVENT":
            if video_event_dispositions.get(source_id) is not FindingDisposition.REJECTED:
                raise EvidenceConfirmationRequired("unsafe_evidence_accepted")


def _require_safe_correction(
    guard: PromptInjectionGuard,
    corrected_text: str | None,
    source_role: str,
) -> None:
    if not corrected_text or not corrected_text.strip():
        raise EvidenceConfirmationRequired("unsafe_evidence_correction")
    if guard.inspect_text(corrected_text.strip(), source_role=source_role).decision != "ALLOWED":
        raise EvidenceConfirmationRequired("unsafe_evidence_correction")


def _validate_confidence(value: float) -> None:
    if value < 0 or value > 1:
        raise EvidenceSchemaError("confidence must be between 0 and 1")
