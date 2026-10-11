"""Governed Diffusers inpainting bundles for training-only synthetic media."""

from __future__ import annotations

import importlib
import io
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any, Literal, Protocol

from PIL import Image

from industrial_ops_agent.media.content_credentials import (
    ContentCredentialSigner,
    ContentCredentialStatus,
)

SYNTHETIC_MEDIA_SCHEMA_VERSION = "synthetic-media/v1"
MAX_SYNTHETIC_IMAGE_BYTES = 10 * 1024 * 1024
MAX_SYNTHETIC_IMAGE_PIXELS = 16_777_216


class SyntheticMediaError(ValueError):
    """The generation request or provenance bundle is unsafe or inconsistent."""


@dataclass(frozen=True, slots=True)
class SyntheticGenerationRequest:
    source_image: bytes
    mask_image: bytes
    asset_model: str
    defect_label: str
    prompt: str
    negative_prompt: str
    seed: int
    num_inference_steps: int
    guidance_scale: float
    generated_by_subject_id: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class GeneratedSyntheticImage:
    image_png: bytes
    generator_release_id: str


@dataclass(frozen=True, slots=True)
class SyntheticMediaBundle:
    image_png: bytes
    manifest: dict[str, Any]


class SyntheticInpaintingBackend(Protocol):
    def generate(self, request: SyntheticGenerationRequest) -> GeneratedSyntheticImage: ...


@dataclass(slots=True)
class DiffusersInpaintingBackend:
    """Lazily load an official Diffusers inpainting AutoPipeline on a worker."""

    model_id: str
    model_revision: str
    device: Literal["cuda", "cpu"] = "cuda"
    _pipeline: Any | None = None
    _diffusers_version: str | None = None

    def __post_init__(self) -> None:
        self.model_id = self.model_id.strip()
        self.model_revision = self.model_revision.strip()
        if not self.model_id or not self.model_revision or self.device not in {"cuda", "cpu"}:
            raise SyntheticMediaError("diffusers_model_configuration_invalid")

    def generate(self, request: SyntheticGenerationRequest) -> GeneratedSyntheticImage:
        source, mask = _validated_images(request.source_image, request.mask_image)
        pipeline, torch, diffusers_version = self._load_pipeline()
        generator = torch.Generator(device=self.device).manual_seed(request.seed)
        try:
            result = pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or None,
                image=source,
                mask_image=mask,
                generator=generator,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
            )
            images = result.images
            image = images[0]
        except Exception as exc:
            raise SyntheticMediaError("diffusers_inpainting_failed") from exc
        if not isinstance(image, Image.Image):
            raise SyntheticMediaError("diffusers_output_invalid")
        output = io.BytesIO()
        image.convert("RGB").save(output, format="PNG", optimize=True)
        return GeneratedSyntheticImage(
            image_png=output.getvalue(),
            generator_release_id=(
                f"diffusers-{diffusers_version}:{self.model_id}@{self.model_revision}"
            ),
        )

    def _load_pipeline(self) -> tuple[Any, Any, str]:
        try:
            diffusers = importlib.import_module("diffusers")
            torch = importlib.import_module("torch")
            auto_pipeline = diffusers.AutoPipelineForInpainting
        except ImportError as exc:  # pragma: no cover - training image owns the optional stack
            raise SyntheticMediaError("diffusers_runtime_unavailable") from exc
        if self.device == "cuda" and not torch.cuda.is_available():
            raise SyntheticMediaError("diffusers_cuda_unavailable")
        if self._pipeline is None:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            try:
                self._pipeline = auto_pipeline.from_pretrained(
                    self.model_id,
                    revision=self.model_revision,
                    torch_dtype=dtype,
                    use_safetensors=True,
                ).to(self.device)
            except Exception as exc:
                raise SyntheticMediaError("diffusers_model_load_failed") from exc
            self._diffusers_version = str(diffusers.__version__)
        return self._pipeline, torch, self._diffusers_version or str(diffusers.__version__)


def generate_synthetic_media_bundle(
    request: SyntheticGenerationRequest,
    backend: SyntheticInpaintingBackend,
) -> SyntheticMediaBundle:
    _validate_request(request)
    _validated_images(request.source_image, request.mask_image)
    generated = backend.generate(request)
    _validated_png(generated.image_png)
    if not generated.generator_release_id.strip():
        raise SyntheticMediaError("generator_release_missing")
    prompt_payload = {
        "asset_model": request.asset_model.strip(),
        "defect_label": request.defect_label.strip(),
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
    }
    manifest: dict[str, Any] = {
        "schema_version": SYNTHETIC_MEDIA_SCHEMA_VERSION,
        "synthetic": True,
        "purpose": "TRAINING_CANDIDATE",
        "evidence_eligible": False,
        "golden_dataset_eligible": False,
        "training_candidate_eligible": False,
        "asset_model": request.asset_model,
        "defect_label": request.defect_label,
        "generator_release_id": generated.generator_release_id,
        "prompt_hash": _canonical_hash(prompt_payload),
        "seed": request.seed,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        "source_image_sha256": _digest(request.source_image),
        "mask_image_sha256": _digest(request.mask_image),
        "output_image_sha256": _digest(generated.image_png),
        "generated_by_subject_id": request.generated_by_subject_id.strip(),
        "generated_at": request.occurred_at.isoformat(),
        "review": {
            "status": "PENDING_EXPERT_REVIEW",
            "reviewer_subject_id": None,
            "notes": None,
            "reviewed_at": None,
        },
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    verify_synthetic_media_bundle(
        manifest,
        image_png=generated.image_png,
        source_image=request.source_image,
        mask_image=request.mask_image,
    )
    return SyntheticMediaBundle(generated.image_png, manifest)


def review_synthetic_media_bundle(
    manifest: dict[str, Any],
    *,
    image_png: bytes,
    source_image: bytes,
    mask_image: bytes,
    reviewer_subject_id: str,
    decision: Literal["APPROVED", "REJECTED"],
    notes: str,
    occurred_at: datetime,
) -> dict[str, Any]:
    verify_synthetic_media_bundle(
        manifest,
        image_png=image_png,
        source_image=source_image,
        mask_image=mask_image,
    )
    if "content_credentials" in manifest:
        raise SyntheticMediaError("synthetic_media_review_must_precede_c2pa_signing")
    generated_by = str(manifest["generated_by_subject_id"]).strip()
    normalized_reviewer = reviewer_subject_id.strip()
    if not normalized_reviewer or normalized_reviewer == generated_by:
        raise SyntheticMediaError("synthetic_media_reviewer_separation_required")
    normalized_notes = notes.strip()
    if not normalized_notes or len(normalized_notes) > 2_000:
        raise SyntheticMediaError("synthetic_media_review_notes_invalid")
    updated: dict[str, Any] = deepcopy(manifest)
    updated["training_candidate_eligible"] = decision == "APPROVED"
    updated["review"] = {
        "status": decision,
        "reviewer_subject_id": normalized_reviewer,
        "notes": normalized_notes,
        "reviewed_at": occurred_at.isoformat(),
    }
    updated.pop("manifest_sha256", None)
    updated["manifest_sha256"] = _manifest_hash(updated)
    verify_synthetic_media_bundle(
        updated,
        image_png=image_png,
        source_image=source_image,
        mask_image=mask_image,
    )
    return updated


def credential_synthetic_media_bundle(
    manifest: dict[str, Any],
    *,
    image_png: bytes,
    source_image: bytes,
    mask_image: bytes,
    signer: ContentCredentialSigner,
) -> SyntheticMediaBundle:
    """Embed a reviewed synthetic media claim without making it field evidence."""

    verify_synthetic_media_bundle(
        manifest,
        image_png=image_png,
        source_image=source_image,
        mask_image=mask_image,
    )
    review = manifest.get("review")
    if not isinstance(review, dict) or review.get("status") != "APPROVED":
        raise SyntheticMediaError("synthetic_media_c2pa_requires_approved_review")
    if "content_credentials" in manifest:
        raise SyntheticMediaError("synthetic_media_c2pa_already_embedded")
    reviewer = review.get("reviewer_subject_id")
    if not isinstance(reviewer, str) or not reviewer:
        raise SyntheticMediaError("synthetic_media_c2pa_review_binding_invalid")
    reviewed_manifest_hash = str(manifest["manifest_sha256"]).removeprefix("sha256:")
    reviewer_subject_hash = sha256(reviewer.encode()).hexdigest()
    try:
        credentialed = signer.sign_generated_media(
            generated_content=image_png,
            generated_media_type="image/png",
            source_content=source_image,
            source_media_type=_image_media_type(source_image),
            title=f"{manifest['asset_model']} {manifest['defect_label']}",
            reviewed_manifest_hash=reviewed_manifest_hash,
            reviewer_subject_hash=reviewer_subject_hash,
            generator_release_id=str(manifest["generator_release_id"]),
        )
    except (RuntimeError, ValueError) as exc:
        raise SyntheticMediaError("synthetic_media_c2pa_signing_failed") from exc
    _validated_png(credentialed.content)
    if credentialed.report.status not in {
        ContentCredentialStatus.PRESENT_TRUSTED,
        ContentCredentialStatus.PRESENT_VALID,
        ContentCredentialStatus.PRESENT_UNVERIFIED,
    }:
        raise SyntheticMediaError("synthetic_media_c2pa_verification_failed")
    report = credentialed.report.document()
    updated: dict[str, Any] = deepcopy(manifest)
    updated["output_image_sha256"] = _digest(credentialed.content)
    updated["content_credentials"] = {
        "standard": "C2PA",
        "embedded": True,
        "unsigned_output_image_sha256": _digest(image_png),
        "signed_output_image_sha256": _digest(credentialed.content),
        "reviewed_manifest_sha256": str(manifest["manifest_sha256"]),
        "reviewer_subject_hash": f"sha256:{reviewer_subject_hash}",
        "report": report,
    }
    updated.pop("manifest_sha256", None)
    updated["manifest_sha256"] = _manifest_hash(updated)
    verify_synthetic_media_bundle(
        updated,
        image_png=credentialed.content,
        source_image=source_image,
        mask_image=mask_image,
    )
    return SyntheticMediaBundle(image_png=credentialed.content, manifest=updated)


def verify_synthetic_media_bundle(
    manifest: dict[str, Any],
    *,
    image_png: bytes,
    source_image: bytes,
    mask_image: bytes,
) -> None:
    _validated_images(source_image, mask_image)
    _validated_png(image_png)
    if (
        manifest.get("schema_version") != SYNTHETIC_MEDIA_SCHEMA_VERSION
        or manifest.get("synthetic") is not True
        or manifest.get("purpose") != "TRAINING_CANDIDATE"
        or manifest.get("evidence_eligible") is not False
        or manifest.get("golden_dataset_eligible") is not False
        or manifest.get("source_image_sha256") != _digest(source_image)
        or manifest.get("mask_image_sha256") != _digest(mask_image)
        or manifest.get("output_image_sha256") != _digest(image_png)
    ):
        raise SyntheticMediaError("synthetic_media_bundle_integrity_failed")
    expected_manifest_hash = _manifest_hash(manifest)
    if manifest.get("manifest_sha256") != expected_manifest_hash:
        raise SyntheticMediaError("synthetic_media_manifest_integrity_failed")
    _verify_content_credentials(manifest, image_png)
    generated_at = _aware_datetime(manifest.get("generated_at"))
    generated_by = manifest.get("generated_by_subject_id")
    seed = manifest.get("seed")
    steps = manifest.get("num_inference_steps")
    guidance = manifest.get("guidance_scale")
    if (
        not _normalized_text(manifest.get("asset_model"))
        or not _normalized_text(manifest.get("defect_label"))
        or not _normalized_text(manifest.get("generator_release_id"))
        or not _sha256_digest(manifest.get("prompt_hash"))
        or not isinstance(generated_by, str)
        or not _normalized_text(generated_by)
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**63
        or isinstance(steps, bool)
        or not isinstance(steps, int)
        or not 1 <= steps <= 200
        or isinstance(guidance, bool)
        or not isinstance(guidance, (int, float))
        or not math.isfinite(float(guidance))
        or not 0 <= float(guidance) <= 30
        or generated_at is None
    ):
        raise SyntheticMediaError("synthetic_media_manifest_metadata_invalid")
    review = manifest.get("review")
    if not isinstance(review, dict):
        raise SyntheticMediaError("synthetic_media_review_invalid")
    status = review.get("status")
    eligible = manifest.get("training_candidate_eligible")
    reviewer = review.get("reviewer_subject_id")
    reviewed_at = _aware_datetime(review.get("reviewed_at"))
    notes = review.get("notes")
    if (
        status not in {"PENDING_EXPERT_REVIEW", "APPROVED", "REJECTED"}
        or eligible is not (status == "APPROVED")
        or (
            status == "PENDING_EXPERT_REVIEW"
            and any(
                review.get(field) is not None
                for field in ("reviewer_subject_id", "notes", "reviewed_at")
            )
        )
        or (
            status in {"APPROVED", "REJECTED"}
            and (
                not all(
                    review.get(field)
                    for field in ("reviewer_subject_id", "notes", "reviewed_at")
                )
                or not isinstance(reviewer, str)
                or not _normalized_text(reviewer)
                or reviewer == generated_by
                or not isinstance(notes, str)
                or not notes.strip()
                or notes != notes.strip()
                or len(notes) > 2_000
                or reviewed_at is None
                or reviewed_at < generated_at
            )
        )
    ):
        raise SyntheticMediaError("synthetic_media_review_invalid")


def _validate_request(request: SyntheticGenerationRequest) -> None:
    if (
        not request.asset_model.strip()
        or not request.defect_label.strip()
        or not request.prompt.strip()
        or not request.generated_by_subject_id.strip()
        or request.occurred_at.tzinfo is None
        or not 0 <= request.seed < 2**63
        or not 1 <= request.num_inference_steps <= 200
        or not math.isfinite(request.guidance_scale)
        or not 0 <= request.guidance_scale <= 30
    ):
        raise SyntheticMediaError("synthetic_generation_request_invalid")
    if len(request.prompt) > 4_000 or len(request.negative_prompt) > 4_000:
        raise SyntheticMediaError("synthetic_generation_prompt_too_long")


def _validated_images(
    source_content: bytes,
    mask_content: bytes,
) -> tuple[Image.Image, Image.Image]:
    source = _open_image(source_content, "synthetic_source_image_invalid").convert("RGB")
    mask = _open_image(mask_content, "synthetic_mask_image_invalid").convert("L")
    if source.size != mask.size:
        raise SyntheticMediaError("synthetic_mask_dimensions_mismatch")
    extrema = mask.getextrema()
    if extrema is None or extrema[0] == extrema[1]:
        raise SyntheticMediaError("synthetic_mask_requires_selected_region")
    return source, mask


def _validated_png(content: bytes) -> Image.Image:
    image = _open_image(content, "synthetic_output_image_invalid")
    if image.format != "PNG":
        raise SyntheticMediaError("synthetic_output_must_be_png")
    return image


def _image_media_type(content: bytes) -> str:
    image = _open_image(content, "synthetic_source_image_invalid")
    if image.format == "PNG":
        return "image/png"
    if image.format == "JPEG":
        return "image/jpeg"
    raise SyntheticMediaError("synthetic_c2pa_source_format_unsupported")


def _verify_content_credentials(manifest: dict[str, Any], image_png: bytes) -> None:
    credential = manifest.get("content_credentials")
    if credential is None:
        return
    if not isinstance(credential, dict):
        raise SyntheticMediaError("synthetic_media_c2pa_binding_invalid")
    report = credential.get("report")
    if not isinstance(report, dict):
        raise SyntheticMediaError("synthetic_media_c2pa_binding_invalid")
    report_basis = {key: value for key, value in report.items() if key != "report_digest"}
    allowed_statuses = {
        ContentCredentialStatus.PRESENT_TRUSTED.value,
        ContentCredentialStatus.PRESENT_VALID.value,
        ContentCredentialStatus.PRESENT_UNVERIFIED.value,
    }
    unsigned_output_hash = credential.get("unsigned_output_image_sha256")
    reviewed_manifest_hash = credential.get("reviewed_manifest_sha256")
    reviewer_subject_hash = credential.get("reviewer_subject_hash")
    reconstructed_reviewed = deepcopy(manifest)
    reconstructed_reviewed.pop("content_credentials", None)
    reconstructed_reviewed["output_image_sha256"] = unsigned_output_hash
    review = reconstructed_reviewed.get("review")
    reviewer = review.get("reviewer_subject_id") if isinstance(review, dict) else None
    expected_reviewer_hash = (
        f"sha256:{sha256(reviewer.encode()).hexdigest()}"
        if isinstance(reviewer, str) and reviewer
        else None
    )
    if (
        credential.get("standard") != "C2PA"
        or credential.get("embedded") is not True
        or credential.get("signed_output_image_sha256") != _digest(image_png)
        or credential.get("signed_output_image_sha256") != manifest.get("output_image_sha256")
        or not _sha256_digest(unsigned_output_hash)
        or not _sha256_digest(reviewed_manifest_hash)
        or reviewed_manifest_hash != _manifest_hash(reconstructed_reviewed)
        or not _sha256_digest(reviewer_subject_hash)
        or reviewer_subject_hash != expected_reviewer_hash
        or report.get("schema_version") != "media-content-credential/v1"
        or report.get("status") not in allowed_statuses
        or report.get("absence_is_not_fraud_evidence") is not True
        or report.get("reviewed_manifest_hash")
        != str(reviewed_manifest_hash).removeprefix("sha256:")
        or report.get("reviewer_subject_hash")
        != str(reviewer_subject_hash).removeprefix("sha256:")
        or report.get("generator_release_id") != manifest.get("generator_release_id")
        or report.get("evidence_eligible") is not False
        or report.get("golden_dataset_eligible") is not False
        or report.get("report_digest") != _canonical_hash(report_basis).removeprefix("sha256:")
    ):
        raise SyntheticMediaError("synthetic_media_c2pa_binding_invalid")


def _open_image(content: bytes, reason: str) -> Image.Image:
    if not content or len(content) > MAX_SYNTHETIC_IMAGE_BYTES:
        raise SyntheticMediaError(reason)
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except Exception as exc:
        raise SyntheticMediaError(reason) from exc
    if image.width * image.height > MAX_SYNTHETIC_IMAGE_PIXELS:
        raise SyntheticMediaError(reason)
    return image


def _manifest_hash(manifest: dict[str, Any]) -> str:
    basis = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return _canonical_hash(basis)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return _digest(encoded)


def _digest(content: bytes) -> str:
    return f"sha256:{sha256(content).hexdigest()}"


def _normalized_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _sha256_digest(value: object) -> bool:
    if not isinstance(value, str):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def _aware_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None
