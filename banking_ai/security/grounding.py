"""Hallucination guard: every money/percent figure in an answer must exist in the retrieved facts."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

_NUMBER = re.compile(
    r"(?P<cur>₹|rs\.?\s*|inr\s*)?(?P<num>\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?P<pct>\s*%)?",
    re.IGNORECASE,
)
_DATE_LIKE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def _numbers_in(text: str, only_significant: bool) -> set[float]:
    found: set[float] = set()
    text = _DATE_LIKE.sub(" ", text)
    for match in _NUMBER.finditer(text):
        value = float(match.group("num").replace(",", ""))
        significant = bool(match.group("cur") or match.group("pct")) or value >= 1000 or "." in match.group("num")
        if only_significant and not significant:
            continue
        found.add(round(value, 2))
    return found


def collect_fact_numbers(obj: Any) -> set[float]:
    """Collect every number appearing anywhere inside a nested facts structure."""
    numbers: set[float] = set()
    if isinstance(obj, bool) or obj is None:
        return numbers
    if isinstance(obj, (int, float)):
        numbers.add(round(float(obj), 2))
    elif isinstance(obj, str):
        numbers |= _numbers_in(obj, only_significant=False)
    elif isinstance(obj, dict):
        for value in obj.values():
            numbers |= collect_fact_numbers(value)
    elif isinstance(obj, (list, tuple, set)):
        for value in obj:
            numbers |= collect_fact_numbers(value)
    return numbers


def _close(a: float, b: float) -> bool:
    tolerance = 1.0 if max(abs(a), abs(b)) >= 1000 else 0.011
    return abs(a - b) <= tolerance


@dataclass
class GroundingResult:
    """Result of checking a response against known facts."""

    ok: bool
    ungrounded: list[float] = field(default_factory=list)


class GroundingValidator:
    """Rejects responses that contain significant numbers not present in the facts."""

    def check(self, response: str, facts: Any, extra_allowed: Iterable[float] = ()) -> GroundingResult:
        allowed = collect_fact_numbers(facts) | {round(float(x), 2) for x in extra_allowed}
        ungrounded = sorted(
            n for n in _numbers_in(response, only_significant=True)
            if not any(_close(n, known) for known in allowed)
        )
        return GroundingResult(ok=not ungrounded, ungrounded=ungrounded)
