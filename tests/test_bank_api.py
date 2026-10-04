"""HTTP-level tests of the mock banking API (skipped when FastAPI/httpx are not installed)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from banking_ai.api.bank_app import create_bank_app  # noqa: E402


@pytest.fixture(scope="module")
def client(settings, bank_service):
    return TestClient(create_bank_app(settings.with_overrides(bank_api_key="test-key"), bank_service))


H = {"X-API-Key": "test-key"}


def test_requires_api_key(client):
    assert client.get("/api/loans").status_code == 401
    assert client.get("/api/loans", headers=H).status_code == 200


@pytest.mark.parametrize("path", [
    "/api/customers/CUST1001", "/api/accounts/CUST1001", "/api/accounts/CUST1001/balance",
    "/api/accounts/CUST1001/transactions?limit=3", "/api/accounts/CUST1001/charges", "/api/loans/personal_loan",
    "/api/interest-rates", "/api/accounts/CUST1002/deductions/explain"])
def test_get_endpoints(client, path):
    assert client.get(path, headers=H).status_code == 200


def test_validation_and_not_found(client):
    assert client.get("/api/customers/BAD", headers=H).status_code == 422
    assert client.get("/api/customers/CUST9999", headers=H).status_code == 404
    assert client.get("/api/accounts/CUST1001/transactions?limit=500", headers=H).status_code == 422


def test_transaction_lookup_is_scoped_to_owner(client):
    txn = client.get("/api/accounts/CUST1002/transactions?limit=1", headers=H).json()["transactions"][0]["transaction_id"]
    assert client.get(f"/api/transactions/{txn}?customer_id=CUST1002", headers=H).status_code == 200
    assert client.get(f"/api/transactions/{txn}?customer_id=CUST1001", headers=H).status_code == 404


def test_post_endpoints(client):
    emi = client.post("/api/emi/calculate", json={"principal": 500000, "annual_rate": 10, "tenure_months": 60}, headers=H).json()
    assert emi["emi"] == 10623.52
    ok = client.post("/api/loans/eligibility", json={"customer_id": "CUST1001", "loan_type": "personal_loan", "loan_amount": 500000, "tenure_months": 60}, headers=H)
    assert ok.status_code == 200 and ok.json()["eligible"] is True
    assert client.post("/api/loans/eligibility", json={"customer_id": "CUST1001", "loan_type": "yacht_loan"}, headers=H).status_code == 422
