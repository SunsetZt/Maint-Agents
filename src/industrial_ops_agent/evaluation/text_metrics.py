"""Deterministic token-sequence metrics without model or governance dependencies."""

from __future__ import annotations


def token_edit_distance(expected: tuple[str, ...], actual: tuple[str, ...]) -> int:
    previous = list(range(len(actual) + 1))
    for expected_index, expected_item in enumerate(expected, start=1):
        current = [expected_index]
        for actual_index, actual_item in enumerate(actual, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[actual_index] + 1,
                    previous[actual_index - 1] + (expected_item != actual_item),
                )
            )
        previous = current
    return previous[-1]
