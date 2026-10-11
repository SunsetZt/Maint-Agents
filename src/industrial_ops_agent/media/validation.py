"""Bounded signature, size, and decompression-risk validation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8\xff"
_PDF_SIGNATURE = b"%PDF-"
_FLAC_SIGNATURE = b"fLaC"
_RIFF_SIGNATURE = b"RIFF"
_WAVE_SIGNATURE = b"WAVE"
_WEBM_SIGNATURE = b"\x1a\x45\xdf\xa3"
_ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_MAX_PNG_PIXELS = 40_000_000
_MAX_PDF_COMPRESSED_STREAMS = 128


class MediaValidationFailure(ValueError):
    def __init__(self, reason_code: str, *, detected_mime: str | None = None) -> None:
        super().__init__("media validation failed")
        self.reason_code = reason_code
        self.detected_mime = detected_mime


@dataclass(frozen=True, slots=True)
class MediaInspection:
    detected_mime: str
    size_bytes: int
    content_hash: str


def inspect_media(
    content: bytes,
    *,
    declared_mime: str,
    max_bytes: int,
) -> MediaInspection:
    """Validate supported media using bytes, never caller-declared type alone."""

    if len(content) > max_bytes:
        raise MediaValidationFailure("media_too_large")
    normalized_mime = declared_mime.split(";", 1)[0].strip().casefold()
    normalized_mime = {
        "audio/x-wav": "audio/wav",
        "audio/wave": "audio/wav",
        "audio/x-flac": "audio/flac",
    }.get(normalized_mime, normalized_mime)
    if content.startswith(_ZIP_SIGNATURES):
        raise MediaValidationFailure(
            "compressed_archive_not_allowed",
            detected_mime="application/zip",
        )
    detected = _detect_mime(content)
    if detected is None:
        raise MediaValidationFailure("unsupported_media_type")
    if detected != normalized_mime:
        raise MediaValidationFailure("mime_type_mismatch", detected_mime=detected)
    _check_compression_risk(content, detected)
    return MediaInspection(
        detected_mime=detected,
        size_bytes=len(content),
        content_hash=sha256(content).hexdigest(),
    )


def _detect_mime(content: bytes) -> str | None:
    if content.startswith(_PNG_SIGNATURE):
        return "image/png"
    if content.startswith(_JPEG_SIGNATURE):
        return "image/jpeg"
    if content.startswith(_PDF_SIGNATURE):
        return "application/pdf"
    if content.startswith(_FLAC_SIGNATURE):
        return "audio/flac"
    if (
        content.startswith(_RIFF_SIGNATURE)
        and len(content) >= 12
        and content[8:12] == _WAVE_SIGNATURE
    ):
        return "audio/wav"
    if len(content) >= 12 and content[4:8] == b"ftyp":
        return "video/mp4"
    if content.startswith(_WEBM_SIGNATURE):
        return "video/webm"
    return None


def _check_compression_risk(content: bytes, detected_mime: str) -> None:
    if detected_mime == "image/png":
        if len(content) < 24 or content[12:16] != b"IHDR":
            raise MediaValidationFailure("malformed_media", detected_mime=detected_mime)
        width = int.from_bytes(content[16:20], "big")
        height = int.from_bytes(content[20:24], "big")
        if width < 1 or height < 1 or width * height > _MAX_PNG_PIXELS:
            raise MediaValidationFailure("compression_risk", detected_mime=detected_mime)
    if (
        detected_mime == "application/pdf"
        and content.count(b"/FlateDecode") > _MAX_PDF_COMPRESSED_STREAMS
    ):
        raise MediaValidationFailure("compression_risk", detected_mime=detected_mime)
    if detected_mime == "audio/wav" and (
        len(content) < 44
        or b"fmt " not in content[:128]
        or b"data" not in content[:512]
    ):
        raise MediaValidationFailure("malformed_media", detected_mime=detected_mime)
    if detected_mime == "audio/flac" and len(content) < 42:
        raise MediaValidationFailure("malformed_media", detected_mime=detected_mime)
    if detected_mime == "video/mp4":
        box_size = int.from_bytes(content[:4], "big")
        if len(content) < 24 or box_size < 16 or box_size > len(content):
            raise MediaValidationFailure("malformed_media", detected_mime=detected_mime)
    if detected_mime == "video/webm" and len(content) < 16:
        raise MediaValidationFailure("malformed_media", detected_mime=detected_mime)
