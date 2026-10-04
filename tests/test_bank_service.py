import pytest

from banking_ai.bank.errors import InvalidRequestError, NotFoundError


def test_data_volume(bank_service):
    d = bank_service.data
    assert len(d.customers) >= 20 and len(d.transactions) >= 100 and len(d.loans) >= 3 and len(d.interest_rates) >= 5


def test_balance_is_masked_and_correct(bank_service):
    acct = bank_service.get_balance("CUST1001")["accounts"][0]
    assert acct["account_number"].startswith("XXXX XXXX ") and len(acct["account_number"]) == 14
    assert acct["available_balance"] == pytest.approx(acct["balance"] - acct["lien_amount"])


def test_customer_endpoint_masks_contact_details(bank_service):
    c = bank_service.get_customer("CUST1001")
    assert "***" in c["email"] and c["phone"].startswith("XXXXXX")


def test_transactions_sorted_limited_and_filtered(bank_service):
    result = bank_service.list_transactions("CUST1001", limit=5)
    dates = [t["date"] for t in result["transactions"]]
    assert len(dates) == 5 and dates == sorted(dates, reverse=True)
    debits = bank_service.list_transactions("CUST1001", limit=50, txn_type="debit")["transactions"]
    assert debits and all(t["type"] == "debit" for t in debits)
    month = bank_service.list_transactions("CUST1001", limit=50, period="this_month")["transactions"]
    assert month and all(t["date"].startswith(bank_service.as_of.strftime("%Y-%m")) for t in month)


def test_transaction_ownership_is_enforced(bank_service):
    other_txn = bank_service.list_transactions("CUST1002", limit=1)["transactions"][0]["transaction_id"]
    with pytest.raises(NotFoundError):
        bank_service.get_transaction("CUST1001", other_txn)
    assert bank_service.get_transaction("CUST1002", other_txn)["transaction_id"] == other_txn


def test_unknown_customer_and_bad_inputs(bank_service):
    with pytest.raises(NotFoundError):
        bank_service.get_balance("CUST9999")
    with pytest.raises(InvalidRequestError):
        bank_service.list_transactions("CUST1001", period="someday")
    with pytest.raises(InvalidRequestError):
        bank_service.list_transactions("CUST1001", start_date="2026-13-45")


def test_eligibility_positive_and_negative(bank_service):
    ok = bank_service.check_loan_eligibility("CUST1001", "personal_loan", 500000, 60)
    assert ok["eligible"] and ok["estimated_emi"] > 0 and ok["max_eligible_amount"] >= 500000
    no = bank_service.check_loan_eligibility("CUST1003", "personal_loan", 500000, 60)  # low income + low score
    assert not no["eligible"] and any(not c["passed"] for c in no["checks"])


def test_eligibility_uses_bank_income_not_declared_income(bank_service):
    result = bank_service.check_loan_eligibility("CUST1003", "personal_loan", 100000, 36, declared_monthly_income=5_000_000)
    assert not result["eligible"] and result["monthly_income_on_record"] == 24000


def test_explain_charge_and_payment(bank_service):
    charge = bank_service.explain_transaction("CUST1002")
    assert charge["is_bank_charge"] and "charge" in charge["label"].lower() or "GST" in charge["label"]


def test_charges_and_rates(bank_service):
    assert bank_service.get_charges("CUST1001")["charge_schedule"]
    assert bank_service.get_interest_rates("personal_loan")["rates"][0]["rate_min"] == 10.99
    with pytest.raises(NotFoundError):
        bank_service.get_interest_rates("bitcoin")
