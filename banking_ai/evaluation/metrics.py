"""Metric helpers for the evaluation harness."""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Optional


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (0 for an empty list)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[rank]


def safe_ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def entity_matches(expected: Any, actual: Any) -> bool:
    """Numbers compare with a small tolerance, strings case-insensitively."""
    if actual is None:
        return False
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) < 0.01
    return str(expected).lower() == str(actual).lower()


def entity_scores(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[int, int]:
    """Return ``(matched_fields, expected_fields)``."""
    matched = sum(entity_matches(v, actual.get(k)) for k, v in expected.items())
    return matched, len(expected)


def tool_sets_equal(expected: Iterable[str], actual: Iterable[str]) -> bool:
    return set(expected) == set(actual)


def contains_all(text: str, needles: Iterable[str]) -> bool:
    lowered = text.lower()
    return all(n.lower() in lowered for n in needles)


def reciprocal_rank(expected_docs: list[str], retrieved_docs: list[str]) -> float:
    for rank, doc in enumerate(retrieved_docs, 1):
        if doc in expected_docs:
            return 1.0 / rank
    return 0.0


_PLACEHOLDER = re.compile(r"\{(\w+):([^{}]+)\}")


def resolve_placeholders(text: str, resolver: Any) -> str:
    """Replace ``{kind:arg:arg}`` placeholders using ``resolver(kind, args) -> str``."""
    return _PLACEHOLDER.sub(lambda m: resolver(m.group(1), m.group(2).split(":")), text)
