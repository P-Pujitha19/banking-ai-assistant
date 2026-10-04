"""Text normalisation shared by the intent classifier and the entity extractor."""
from __future__ import annotations

import re
import unicodedata

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "twenty five": 25, "thirty": 30, "forty": 40, "fifty": 50,
    "a": 1, "an": 1,
}
_UNITS = r"(?:lakhs?|lacs?|crores?|thousand|k\b|years?|yrs?|months?|transactions?|txns?)"
_WORD_NUM = re.compile(
    r"\b(" + "|".join(sorted((w for w in _NUMBER_WORDS if w not in {"a", "an"}), key=len, reverse=True)) + r")\b(?=\s*[- ]?\s*" + _UNITS + r")",
    re.IGNORECASE,
)
_WORD_NUM_AFTER_LAST = re.compile(r"\b(last|past|latest|previous|top|first)\s+(" + "|".join(w for w in _NUMBER_WORDS if w not in {"a", "an"}) + r")\b", re.IGNORECASE)
_A_UNIT = re.compile(r"\b(a|an)\s+(?=(?:lakh|lac|crore|thousand)\b)", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Unicode-normalise, lowercase and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"\s+", " ", text).strip()


def words_to_numbers(text: str) -> str:
    """Convert spelled-out numbers next to units ("five lakh", "last three") into digits."""
    text = _A_UNIT.sub("1 ", text)
    text = _WORD_NUM.sub(lambda m: str(_NUMBER_WORDS[m.group(1).lower()]), text)
    return _WORD_NUM_AFTER_LAST.sub(lambda m: f"{m.group(1)} {_NUMBER_WORDS[m.group(2).lower()]}", text)


def prepare_for_intent(text: str) -> str:
    """Normalise a query for intent classification.

    Specific values (amounts, ids, dates) are replaced by placeholders so the model learns the *shape*
    of a request rather than memorising numbers: ``"5 lakh for 5 years"`` -> ``"<num> lakh for <num> years"``.
    """
    t = words_to_numbers(normalize_text(text))
    t = re.sub(r"\bcust\s?-?\d{4}\b", " <id> ", t)
    t = re.sub(r"\btxn\s?-?\d+\b", " <txn> ", t)
    t = re.sub(r"\d{4}-\d{2}-\d{2}", " <date> ", t)
    t = re.sub(r"\d+(?:,\d+)*(?:\.\d+)?", " <num> ", t)
    t = t.replace("₹", " rs ")
    return re.sub(r"\s+", " ", t).strip()


_TOKEN = re.compile(r"<num>|<id>|<txn>|<date>|[a-z]+|%")


def tokenize(text: str) -> list[str]:
    """Split prepared text into tokens (placeholders are kept intact)."""
    return _TOKEN.findall(prepare_for_intent(text))
