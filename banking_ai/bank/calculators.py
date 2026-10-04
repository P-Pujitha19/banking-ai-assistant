"""Pure financial calculators (EMI, affordability) and Indian currency formatting."""
from __future__ import annotations

from banking_ai.bank.errors import InvalidRequestError


def calculate_emi(principal: float, annual_rate_pct: float, tenure_months: int) -> float:
    """Return the monthly EMI using the reducing-balance formula.

    EMI = P * r * (1 + r)^n / ((1 + r)^n - 1) with r = annual_rate / 12 / 100.
    """
    if principal <= 0:
        raise InvalidRequestError("Loan amount must be greater than zero.")
    if tenure_months <= 0:
        raise InvalidRequestError("Tenure must be at least one month.")
    if annual_rate_pct < 0:
        raise InvalidRequestError("Interest rate cannot be negative.")
    monthly_rate = annual_rate_pct / 12 / 100
    if monthly_rate == 0:
        return round(principal / tenure_months, 2)
    growth = (1 + monthly_rate) ** tenure_months
    return round(principal * monthly_rate * growth / (growth - 1), 2)


def emi_breakdown(principal: float, annual_rate_pct: float, tenure_months: int) -> dict[str, float]:
    """Return EMI, total payable and total interest."""
    emi = calculate_emi(principal, annual_rate_pct, tenure_months)
    total_payment = round(emi * tenure_months, 2)
    return {
        "principal": round(principal, 2), "annual_rate_pct": annual_rate_pct, "tenure_months": tenure_months,
        "emi": emi, "total_payment": total_payment, "total_interest": round(total_payment - principal, 2),
    }


def max_principal_for_emi(emi_capacity: float, annual_rate_pct: float, tenure_months: int) -> float:
    """Largest principal whose EMI does not exceed ``emi_capacity`` (inverse of :func:`calculate_emi`)."""
    if emi_capacity <= 0 or tenure_months <= 0:
        return 0.0
    monthly_rate = annual_rate_pct / 12 / 100
    if monthly_rate == 0:
        return emi_capacity * tenure_months
    growth = (1 + monthly_rate) ** tenure_months
    return emi_capacity * (growth - 1) / (monthly_rate * growth)


def format_inr(amount: float, decimals: bool | None = None) -> str:
    """Format a number with Indian digit grouping, e.g. ``₹1,23,456`` (decimals only when needed)."""
    value = round(float(amount), 2)
    show_decimals = (value % 1 != 0) if decimals is None else decimals
    sign = "-" if value < 0 else ""
    whole, _, frac = f"{abs(value):.2f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return f"{sign}₹{whole}" + (f".{frac}" if show_decimals else "")


def format_months(months: int) -> str:
    """Human-readable tenure, e.g. 60 -> '5 years', 18 -> '1 year 6 months'."""
    years, rest = divmod(int(months), 12)
    parts = []
    if years:
        parts.append(f"{years} year{'s' if years != 1 else ''}")
    if rest:
        parts.append(f"{rest} month{'s' if rest != 1 else ''}")
    return " ".join(parts) or "0 months"
