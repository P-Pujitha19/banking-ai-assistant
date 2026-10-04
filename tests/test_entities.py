import pytest

from banking_ai.nlp.entities import ExtractedEntities, HybridEntityExtractor, RuleEntityExtractor

rules = RuleEntityExtractor()


@pytest.mark.parametrize("text,expected", [
    ("Show my last 5 transactions", {"number_of_transactions": 5}),
    ("show the last five transactions", {"number_of_transactions": 5}),
    ("Can I get a personal loan of 5 lakh for 5 years?", {"loan_type": "personal_loan", "loan_amount": 500000, "tenure_months": 60}),
    ("Calculate EMI for 5 lakh for 5 years at 10%", {"loan_amount": 500000, "tenure_months": 60, "interest_rate": 10}),
    ("I earn 90k per month, can I get a home loan of 30 lakh for 20 years?", {"monthly_income": 90000, "loan_amount": 3000000, "tenure_months": 240, "loan_type": "home_loan"}),
    ("my salary is 60000 and I need a car loan of two lakh for 36 months", {"monthly_income": 60000, "loan_amount": 200000, "tenure_months": 36, "loan_type": "car_loan"}),
    ("Show my transactions from this month", {"period": "this_month"}),
    ("show upi debit transactions last month", {"transaction_channel": "upi", "transaction_type": "debit", "period": "last_month"}),
    ("transactions between 2026-08-01 and 2026-08-31", {"start_date": "2026-08-01", "end_date": "2026-08-31"}),
    ("explain transaction TXN10000221", {"transaction_id": "TXN10000221"}),
    ("minimum balance for a savings account", {"account_type": "savings"}),
    ("what about 10 years?", {"tenure_months": 120}),
    ("Compare personal loan and home loan", {"loan_types": ["personal_loan", "home_loan"]}),
    ("balance for CUST1001", {"customer_id": "CUST1001"}),
])
def test_rule_extraction(text, expected):
    got = rules.extract(text).non_empty()
    for key, value in expected.items():
        assert got.get(key) == value, f"{key}: {got}"


def test_account_number_is_reduced_to_last_four():
    entities = rules.extract("balance of account 501012345678")
    assert entities.account_number_last4 == "5678"
    assert "501012345678" not in str(entities.non_empty())


def test_last_n_months_is_a_period_not_a_tenure():
    entities = rules.extract("show transactions for last 3 months")
    assert entities.period == "last_3_months" and entities.tenure_months is None


def test_credit_score_is_not_a_credit_transaction():
    assert rules.extract("what is my credit score").transaction_type is None


def test_merge_prefers_primary_values():
    primary = ExtractedEntities(loan_amount=100000.0)
    fallback = ExtractedEntities(loan_amount=999.0, tenure_months=24)
    merged = primary.merged_with(fallback)
    assert merged.loan_amount == 100000.0 and merged.tenure_months == 24


class FakeLlmExtractor:
    def __init__(self):
        self.calls = 0

    def extract(self, text):
        self.calls += 1
        return ExtractedEntities(tenure_months=84)


def test_llm_extractor_only_used_when_a_required_slot_is_missing():
    llm = FakeLlmExtractor()
    hybrid = HybridEntityExtractor(llm_extractor=llm)
    hybrid.extract("emi for 5 lakh for 5 years", "emi_calculation")  # nothing missing
    assert llm.calls == 0
    out = hybrid.extract("emi for 5 lakh over seven years", "emi_calculation")  # tenure phrased unusually
    assert llm.calls in (0, 1)
    assert out.loan_amount == 500000
