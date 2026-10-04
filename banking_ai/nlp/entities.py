"""Entity extraction: deterministic rules first, optional LLM structured extraction as a safety net.

Why a hybrid?  Money amounts ("5 lakh"), tenures ("5 years") and ids follow strict patterns, so regular
expressions are fast, free, deterministic and testable.  An LLM with a Pydantic output schema is only
invoked when the rules missed a slot that the detected intent needs (e.g. unusual phrasing).
"""
from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from banking_ai.config.logging_config import get_logger
from banking_ai.nlp.preprocessing import normalize_text, words_to_numbers

logger = get_logger("nlp.entities")

LOAN_TYPES = ("personal_loan", "home_loan", "car_loan", "education_loan")
_MULTIPLIERS = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7, "cr": 1e7,
                "thousand": 1e3, "k": 1e3, "lpa": 1e5, "million": 1e6, "mn": 1e6}
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
_MONTH_ABBR = {m[:3]: i + 1 for i, m in enumerate(_MONTHS)}

_LOAN_KEYWORDS = [
    ("home_loan", r"\b(home|housing|house|mortgage|property)\s+loans?\b|\bmortgage\b"),
    ("personal_loan", r"\bpersonal\s+loans?\b"),
    ("car_loan", r"\b(car|auto|vehicle|automobile)\s+loans?\b"),
    ("education_loan", r"\b(education|educational|student|study)\s+loans?\b"),
]
_NUM = re.compile(
    r"(?P<cur>₹|rs\.?|inr)?\s*(?P<num>\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(?P<unit>lpa\b|lakhs?|lacs?|crores?|cr\b|thousand|k\b|million|mn\b)?",
    re.IGNORECASE,
)
_INCOME_WORDS = re.compile(r"(salary|income|earn|earns|earning|take.?home|ctc|paid)\b(?:\W+\w+){0,3}\W*$")
_PER_MONTH = re.compile(r"^\s*(per month|a month|monthly|/\s*month|pm\b|p\.m)")
_ANNUAL = re.compile(r"^\s*(per annum|annual|annually|per year|a year|yearly|lpa|p\.a)|\b(ctc|annual)\b")
_LOAN_CONTEXT = re.compile(r"\b(loan|borrow|emi|finance|afford|mortgage|lend|instal+ments?|installments?)\b")


class ExtractedEntities(BaseModel):
    """Structured entities understood from a customer message (Pydantic schema, also used for LLM extraction)."""

    customer_id: Optional[str] = Field(default=None, description="Customer id like CUST1001 if written in the message")
    account_number_last4: Optional[str] = Field(default=None, description="Last four digits of an account number mentioned")
    account_type: Optional[str] = Field(default=None, description="savings | current | salary")
    loan_type: Optional[str] = Field(default=None, description="personal_loan | home_loan | car_loan | education_loan")
    loan_types: list[str] = Field(default_factory=list, description="All loan types mentioned (for comparisons)")
    loan_amount: Optional[float] = Field(default=None, description="Requested loan amount in INR")
    amount: Optional[float] = Field(default=None, description="Any other money amount in INR")
    tenure_months: Optional[int] = Field(default=None, description="Loan tenure in months (5 years = 60)")
    interest_rate: Optional[float] = Field(default=None, description="Annual interest rate percent")
    monthly_income: Optional[float] = Field(default=None, description="Monthly salary/income in INR")
    number_of_transactions: Optional[int] = Field(default=None, description="How many transactions to show")
    transaction_type: Optional[str] = Field(default=None, description="credit | debit")
    transaction_channel: Optional[str] = Field(default=None, description="upi | atm | neft | imps | rtgs | pos | card | cheque")
    transaction_id: Optional[str] = Field(default=None, description="Transaction id like TXN10000221")
    period: Optional[str] = Field(default=None, description="this_month | last_month | today | yesterday | last_N_days | month_N ...")
    start_date: Optional[str] = Field(default=None, description="ISO start date")
    end_date: Optional[str] = Field(default=None, description="ISO end date")

    def non_empty(self) -> dict[str, Any]:
        """Return only populated fields (drops None and empty lists)."""
        return {k: v for k, v in self.model_dump().items() if v not in (None, [], "")}

    def merged_with(self, fallback: "ExtractedEntities") -> "ExtractedEntities":
        """Fill gaps in ``self`` using values from ``fallback`` (self always wins)."""
        merged = fallback.model_dump()
        merged.update(self.non_empty())
        return ExtractedEntities(**merged)


class RuleEntityExtractor:
    """Regex/lexicon based extractor for Indian banking phrasing (lakh, crore, k, years...)."""

    def extract(self, text: str) -> ExtractedEntities:
        t = words_to_numbers(normalize_text(text))
        ent: dict[str, Any] = {}
        used: list[tuple[int, int]] = []

        def free(span: tuple[int, int]) -> bool:
            return not any(span[0] < u[1] and u[0] < span[1] for u in used)

        def take(match: re.Match) -> None:
            used.append(match.span())

        if (m := re.search(r"\bcust\s?-?(\d{4})\b", t)):
            ent["customer_id"] = f"CUST{m.group(1)}"; take(m)
        if (m := re.search(r"\btxn\s?-?(\d{6,})\b", t)):
            ent["transaction_id"] = f"TXN{m.group(1)}"; take(m)
        if (m := re.search(r"(?:ending|ends with|ending with|last 4 digits(?: of| are)?|xxxx\s?xxxx)\s*(\d{4})\b", t)):
            ent["account_number_last4"] = m.group(1); take(m)
        elif (m := re.search(r"(?<![\d,.])\d{9,18}(?![\d,.])", t)):
            ent["account_number_last4"] = m.group(0)[-4:]; take(m)  # only the last 4 digits are ever kept

        self._dates(t, ent, used)
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)", t):
            if free(m.span()):
                ent["interest_rate"] = float(m.group(1)); take(m)
        self._tenure_and_period(t, ent, used, free)
        self._period_keywords(t, ent)
        self._counts(t, ent, used, free)
        self._amounts(t, ent, used, free)
        self._categoricals(t, ent)
        return ExtractedEntities(**ent)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _dates(t: str, ent: dict[str, Any], used: list[tuple[int, int]]) -> None:
        iso = list(re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", t))
        dmy = list(re.finditer(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", t))
        found = [f"{m.group(1)}-{m.group(2)}-{m.group(3)}" for m in iso]
        found += [f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}" for m in dmy]
        for m in iso + dmy:
            used.append(m.span())
        if found:
            found.sort()
            ent["start_date"] = found[0]
            ent["end_date"] = found[-1]

    @staticmethod
    def _tenure_and_period(t: str, ent: dict[str, Any], used: list, free: Any) -> None:
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*-?\s*(years?|yrs?|yr|months?|mos?)\b", t):
            if not free(m.span()):
                continue
            value, unit = float(m.group(1)), m.group(2)
            before = t[max(0, m.start() - 12):m.start()]
            as_months = int(round(value * 12)) if unit.startswith(("y",)) else int(round(value))
            if re.search(r"\b(last|past|previous|since)\s*$", before):
                if unit.startswith("m"):
                    ent["period"] = f"last_{int(value)}_months"
                used.append(m.span())
                continue
            ent["tenure_months"] = as_months
            used.append(m.span())

    @staticmethod
    def _period_keywords(t: str, ent: dict[str, Any]) -> None:
        if "period" in ent or ent.get("start_date"):
            return
        mapping = [
            (r"\b(this|current) month\b", "this_month"), (r"\b(last|previous) month\b", "last_month"),
            (r"\btoday\b", "today"), (r"\byesterday\b", "yesterday"), (r"\bthis week\b", "this_week"),
            (r"\b(last|previous) week\b", "last_week"), (r"\bpast week\b", "last_7_days"), (r"\bpast month\b", "last_30_days"),
            (r"\bthis year\b", "this_year"),
        ]
        for pattern, value in mapping:
            if re.search(pattern, t):
                ent["period"] = value
                return
        if (m := re.search(r"\b(?:last|past)\s+(\d{1,3})\s+days?\b", t)):
            ent["period"] = f"last_{int(m.group(1))}_days"; return
        if (m := re.search(r"\b(?:in|during|for|of)\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b", t)):
            ent["period"] = f"month_{_MONTHS.index(m.group(1)) + 1}"

    @staticmethod
    def _counts(t: str, ent: dict[str, Any], used: list, free: Any) -> None:
        patterns = [
            r"\b(\d{1,3})\s+(?:recent\s+|latest\s+|last\s+)?(?:transactions?|txns?|entries|payments|debits|credits)\b",
            r"\b(?:last|past|latest|recent|previous|top|first)\s+(\d{1,3})\b(?!\s*(?:years?|yrs?|months?|days?|weeks?|lakh|lac|crore|%|k\b|digits))",
        ]
        for pattern in patterns:
            m = re.search(pattern, t)
            if m and free(m.span(1)) and 1 <= int(m.group(1)) <= 100:
                ent["number_of_transactions"] = int(m.group(1))
                used.append(m.span(1))
                return

    @staticmethod
    def _amounts(t: str, ent: dict[str, Any], used: list, free: Any) -> None:
        has_loan_context = bool(_LOAN_CONTEXT.search(t))
        for m in _NUM.finditer(t):
            if not free(m.span()):
                continue
            unit = (m.group("unit") or "").lower()
            value = float(m.group("num").replace(",", "")) * _MULTIPLIERS.get(unit, 1)
            has_marker = bool(m.group("cur") or unit)
            if not has_marker and (value < 100 or 1900 <= value <= 2100):
                continue
            before, after = t[max(0, m.start() - 35):m.start()], t[m.end():m.end() + 25]
            if _INCOME_WORDS.search(before) or (_PER_MONTH.search(after) and not re.search(r"(loan|borrow|emi)\W*(?:of|for)?\W*$", before)):
                income = value / 12 if unit == "lpa" or _ANNUAL.search(after) or re.search(r"\b(ctc|annual)\b", before) else value
                ent.setdefault("monthly_income", round(income, 2))
            elif has_loan_context:
                ent.setdefault("loan_amount", value)
                ent.setdefault("amount", value)
            else:
                ent.setdefault("amount", value)
            used.append(m.span())

    @staticmethod
    def _categoricals(t: str, ent: dict[str, Any]) -> None:
        loans = [(m.start(), key) for key, pattern in _LOAN_KEYWORDS for m in [re.search(pattern, t)] if m]
        loans.sort()
        ordered = [key for _, key in loans]
        if ordered:
            ent["loan_types"] = ordered
            if len(ordered) == 1:
                ent["loan_type"] = ordered[0]
        if (m := re.search(r"\b(savings?|current|salary)\s+(?:bank\s+)?account\b", t)) or (m := re.search(r"\b(savings?)\b", t)):
            ent["account_type"] = "savings" if m.group(1).startswith("saving") else m.group(1)
        if re.search(r"\b(credit(?!\s*(?:score|card|limit))|credited|deposited|received|incoming)\b", t):
            ent["transaction_type"] = "credit"
        if re.search(r"\b(debit|debits|debited|deducted|deduction|withdraw\w*|spent|spending|outgoing)\b", t):
            ent["transaction_type"] = "debit"
        for channel in ("upi", "atm", "neft", "imps", "rtgs", "pos", "card", "cheque"):
            if re.search(rf"\b{channel}\b", t):
                ent["transaction_channel"] = channel
                break


class HybridEntityExtractor:
    """Rules first; falls back to LLM structured output only for slots the intent needs."""

    REQUIRED_BY_INTENT = {
        "emi_calculation": ("loan_amount", "tenure_months"),
        "loan_eligibility": ("loan_amount",),
    }

    def __init__(self, rules: Optional[RuleEntityExtractor] = None, llm_extractor: Any = None) -> None:
        self.rules = rules or RuleEntityExtractor()
        self.llm_extractor = llm_extractor

    def extract(self, text: str, intent: Optional[str] = None) -> ExtractedEntities:
        entities = self.rules.extract(text)
        if self.llm_extractor is None or intent not in self.REQUIRED_BY_INTENT:
            return entities
        missing = [f for f in self.REQUIRED_BY_INTENT[intent] if getattr(entities, f) is None]
        if not missing:
            return entities
        try:
            llm_entities = self.llm_extractor.extract(text)
            logger.info("llm entity extraction used", extra={"fields": {"missing_before": missing}})
            return entities.merged_with(llm_entities)
        except Exception:  # LLM errors must never break the request
            logger.warning("llm entity extraction failed; using rule output", exc_info=True)
            return entities
