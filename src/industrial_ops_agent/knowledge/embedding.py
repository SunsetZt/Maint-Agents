"""Deterministic local embedding used by the development retrieval profile."""

from __future__ import annotations

import math
import re
from hashlib import sha256

EMBEDDING_DIMENSIONS = 16
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[\u4e00-\u9fff]")


def tokenize(value: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN.findall(value))


def deterministic_embedding(value: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    for token in tokenize(value):
        digest = sha256(token.encode()).digest()
        index = digest[0] % EMBEDDING_DIMENSIONS
        vector[index] += 1.0 if digest[1] & 1 else -1.0
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0:
        return vector
    return [component / norm for component in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))
