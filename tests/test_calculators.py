import pytest

from banking_ai.bank.calculators import calculate_emi, emi_breakdown, format_inr, format_months, max_principal_for_emi
from banking_ai.bank.errors import InvalidRequestError


def test_emi_known_value():
    assert calculate_emi(500000, 10, 60) == 10623.52


def test_emi_zero_interest_is_simple_division():
    assert calculate_emi(120000, 0, 12) == 10000.0


def test_emi_breakdown_totals_are_consistent():
    b = emi_breakdown(500000, 10, 60)
    assert b["total_payment"] == pytest.approx(b["emi"] * 60)
    assert b["total_interest"] == pytest.approx(b["total_payment"] - 500000)


def test_max_principal_is_inverse_of_emi():
    emi = calculate_emi(750000, 9.5, 84)
    assert max_principal_for_emi(emi, 9.5, 84) == pytest.approx(750000, rel=1e-3)


@pytest.mark.parametrize("args", [(0, 10, 12), (1000, -1, 12), (1000, 10, 0)])
def test_emi_rejects_invalid_input(args):
    with pytest.raises(InvalidRequestError):
        calculate_emi(*args)


@pytest.mark.parametrize("value,expected", [(75000, "₹75,000"), (500000, "₹5,00,000"), (1234567.5, "₹12,34,567.50"), (999, "₹999"), (10000000, "₹1,00,00,000")])
def test_indian_currency_format(value, expected):
    assert format_inr(value) == expected


def test_format_months():
    assert format_months(60) == "5 years" and format_months(18) == "1 year 6 months"
