"""Privacy masking helpers used for API responses and log lines."""
from __future__ import annotations

import re
from typing import Any

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_LONG_DIGITS = re.compile(r"(?<![\w.])\d{9,18}(?![\w.])")
_API_KEY = re.compile(r"\b(?:sk-[A-Za-z0-9_\-]{16,}|AIza[0-9A-Za-z_\-]{20,})\b")
_PASSWORD = re.compile(r"(?i)\b(password|passwd|pin|otp|cvv)\s*[:=]?\s*\S+")
_SENSITIVE_KEYS = ("api_key", "apikey", "password", "token", "secret", "authorization", "x-api-key")


def mask_account_number(account_number: str) -> str:
    """Return ``XXXX XXXX 1234`` style masking, keeping only the last 4 digits."""
    digits = re.sub(r"\D", "", str(account_number))
    return f"XXXX XXXX {digits[-4:]}" if len(digits) >= 4 else "XXXX XXXX"


def mask_phone(phone: str) -> str:
    """Mask a phone number except its last 4 digits."""
    digits = re.sub(r"\D", "", str(phone))
    return f"XXXXXX{digits[-4:]}" if len(digits) >= 4 else "XXXXXX"


def mask_email(email: str) -> str:
    """Mask an e-mail address, keeping the first character and the domain."""
    local, _, domain = str(email).partition("@")
    return f"{local[:1]}***@{domain}" if domain else "***"


def mask_text(text: str) -> str:
    """Mask e-mails, long digit runs (account/phone numbers), API keys and secrets in free text."""
    text = _API_KEY.sub("[REDACTED_KEY]", text)
    text = _PASSWORD.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
    text = _EMAIL.sub(lambda m: mask_email(m.group(0)), text)
    return _LONG_DIGITS.sub(lambda m: "X" * (len(m.group(0)) - 4) + m.group(0)[-4:], text)


def mask_mapping(data: Any) -> Any:
    """Recursively mask strings inside dicts/lists; redact values of secret-looking keys."""
    if isinstance(data, dict):
        masked = {}
        for key, value in data.items():
            if any(token in str(key).lower() for token in _SENSITIVE_KEYS):
                masked[key] = "[REDACTED]"
            else:
                masked[key] = mask_mapping(value)
        return masked
    if isinstance(data, (list, tuple)):
        return [mask_mapping(item) for item in data]
    if isinstance(data, str):
        return mask_text(data)
    return data
