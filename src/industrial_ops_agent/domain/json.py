"""Canonical JSON digests with explicit legacy and strict contracts."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any


def legacy_canonical_digest(value: Any) -> str:
    return f"sha256:{legacy_canonical_hex_digest(value)}"


def legacy_canonical_hex_digest(value: Any) -> str:
    """Legacy JSON hashing without a prefix; keep NaN and Unicode behavior."""
    return sha256(legacy_canonical_bytes(value)).hexdigest()


def strict_document_digest(document: Any) -> str:
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()

def legacy_canonical_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def legacy_canonical_bytes(value: object) -> bytes:
    return legacy_canonical_text(value).encode()


def legacy_ascii_canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
