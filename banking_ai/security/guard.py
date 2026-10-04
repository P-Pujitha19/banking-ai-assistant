"""Input validation and prompt-injection / cross-customer access guard."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

CUSTOMER_ID_PATTERN = re.compile(r"^CUST\d{4}$")
_CUSTOMER_REF = re.compile(r"\bcust\s?-?(\d{4})\b", re.IGNORECASE)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|your)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|your)",
    r"(reveal|show|print|repeat|display|leak)\s+(me\s+)?(your\s+|the\s+)?(system|hidden|internal|initial)\s+(prompt|instructions?|message)",
    r"\bsystem\s+prompt\b",
    r"\byou\s+are\s+now\b",
    r"\b(developer|debug|admin|god)\s+mode\b",
    r"\bjailbreak\b",
    r"\bpretend\s+(to\s+be|you\s+are)\b",
    r"override\s+(your\s+)?(rules|instructions|safety|guardrails)",
    r"bypass\s+(the\s+)?(security|authentication|verification|guardrails)",
    r"<\s*/?\s*(system|assistant|tool)\s*>",
]
_CROSS_CUSTOMER_PATTERNS = [
    r"\b(another|other|someone\s+else'?s?|somebody\s+else'?s?)\s+(customer|person|user|account)",
    r"\ball\s+(the\s+)?(customers|accounts|users)\b",
    r"\bevery\s+customer\b",
]
_INJECTION = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
_CROSS = [re.compile(p, re.IGNORECASE) for p in _CROSS_CUSTOMER_PATTERNS]


@dataclass(frozen=True)
class GuardResult:
    """Outcome of inspecting one user message."""

    allowed: bool
    cleaned: str
    category: Optional[str] = None
    user_message: Optional[str] = None


def validate_customer_id(customer_id: str) -> bool:
    """Return True if the id has the ``CUSTnnnn`` format."""
    return bool(CUSTOMER_ID_PATTERN.match(customer_id or ""))


class InputGuard:
    """Sanitises user input and blocks injection or cross-customer access attempts."""

    def __init__(self, max_chars: int = 500) -> None:
        self.max_chars = max_chars

    def inspect(self, message: str, customer_id: str) -> GuardResult:
        """Validate ``message`` for the authenticated ``customer_id``."""
        cleaned = unicodedata.normalize("NFKC", message or "")
        cleaned = _CONTROL_CHARS.sub(" ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if not cleaned:
            return GuardResult(False, cleaned, "empty", "Please type your banking question.")
        if len(cleaned) > self.max_chars:
            return GuardResult(
                False, cleaned[: self.max_chars], "too_long",
                f"Your message is too long. Please keep it under {self.max_chars} characters.",
            )
        if any(p.search(cleaned) for p in _INJECTION):
            return GuardResult(
                False, cleaned, "prompt_injection",
                "I can only help with your own banking queries and can't follow instructions "
                "that change how I operate. How can I help with your accounts or loans?",
            )
        referenced = {m.group(1) for m in _CUSTOMER_REF.finditer(cleaned)}
        own = customer_id[-4:] if customer_id else ""
        if any(ref != own for ref in referenced) or any(p.search(cleaned) for p in _CROSS):
            return GuardResult(
                False, cleaned, "cross_customer",
                "For your security I can only discuss your own accounts. "
                "I can't access information about other customers.",
            )
        return GuardResult(True, cleaned)
