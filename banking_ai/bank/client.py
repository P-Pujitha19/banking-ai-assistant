"""Bank clients: HTTP (calls the FastAPI mock bank) and in-process (same logic, no network)."""
from __future__ import annotations

from typing import Any, Optional, Protocol

from banking_ai.bank.errors import BankAPIError, BankError, BankUnavailableError
from banking_ai.bank.service import BankService
from banking_ai.config.settings import Settings


class BankClient(Protocol):
    """Interface every bank client implements (mirrors the mock API endpoints)."""

    def get_customer(self, customer_id: str) -> dict[str, Any]: ...
    def get_accounts(self, customer_id: str) -> dict[str, Any]: ...
    def get_balance(self, customer_id: str, account_last4: Optional[str] = None) -> dict[str, Any]: ...
    def list_transactions(self, customer_id: str, **filters: Any) -> dict[str, Any]: ...
    def get_transaction(self, customer_id: str, transaction_id: str) -> dict[str, Any]: ...
    def explain_transaction(self, customer_id: str, transaction_id: Optional[str] = None) -> dict[str, Any]: ...
    def list_loans(self) -> dict[str, Any]: ...
    def get_loan(self, loan_type: str) -> dict[str, Any]: ...
    def get_interest_rates(self, product_type: Optional[str] = None) -> dict[str, Any]: ...
    def check_eligibility(self, customer_id: str, loan_type: str, loan_amount: Optional[float] = None,
                          tenure_months: Optional[int] = None, declared_monthly_income: Optional[float] = None) -> dict[str, Any]: ...
    def calculate_emi(self, principal: float, annual_rate: float, tenure_months: int) -> dict[str, Any]: ...
    def get_charges(self, customer_id: str) -> dict[str, Any]: ...


class InProcessBankClient:
    """Calls :class:`BankService` directly; errors are converted to :class:`BankAPIError`."""

    def __init__(self, service: BankService) -> None:
        self._service = service

    def _call(self, fn: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return fn(*args, **kwargs)
        except BankError as exc:
            raise BankAPIError(exc.status_code, exc.message) from exc

    def get_customer(self, customer_id): return self._call(self._service.get_customer, customer_id)
    def get_accounts(self, customer_id): return self._call(self._service.get_accounts, customer_id)
    def get_balance(self, customer_id, account_last4=None): return self._call(self._service.get_balance, customer_id, account_last4)
    def list_transactions(self, customer_id, **filters): return self._call(self._service.list_transactions, customer_id, **filters)
    def get_transaction(self, customer_id, transaction_id): return self._call(self._service.get_transaction, customer_id, transaction_id)
    def explain_transaction(self, customer_id, transaction_id=None): return self._call(self._service.explain_transaction, customer_id, transaction_id)
    def list_loans(self): return self._call(self._service.list_loans)
    def get_loan(self, loan_type): return self._call(self._service.get_loan, loan_type)
    def get_interest_rates(self, product_type=None): return self._call(self._service.get_interest_rates, product_type)
    def check_eligibility(self, customer_id, loan_type, loan_amount=None, tenure_months=None, declared_monthly_income=None):
        return self._call(self._service.check_loan_eligibility, customer_id, loan_type, loan_amount, tenure_months, declared_monthly_income)
    def calculate_emi(self, principal, annual_rate, tenure_months): return self._call(self._service.calculate_emi, principal, annual_rate, tenure_months)
    def get_charges(self, customer_id): return self._call(self._service.get_charges, customer_id)


class HttpBankClient:
    """Calls the FastAPI mock banking API over HTTP."""

    def __init__(self, base_url: str, api_key: str = "", timeout: float = 5.0, session: Any = None) -> None:
        import requests  # imported lazily so in-process mode has no HTTP dependency

        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._session = session or requests.Session()
        self._headers = {"X-API-Key": api_key} if api_key else {}
        self._requests = requests

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._session.request(method, f"{self._base}{path}", headers=self._headers,
                                             timeout=self._timeout, **kwargs)
        except self._requests.RequestException as exc:
            raise BankUnavailableError("Could not reach the banking service.") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail") or response.json().get("message") or response.text
            except ValueError:
                detail = response.text
            raise BankAPIError(response.status_code, str(detail)[:200])
        return response.json()

    def get_customer(self, customer_id): return self._request("GET", f"/api/customers/{customer_id}")
    def get_accounts(self, customer_id): return self._request("GET", f"/api/accounts/{customer_id}")
    def get_balance(self, customer_id, account_last4=None):
        return self._request("GET", f"/api/accounts/{customer_id}/balance", params={"account_last4": account_last4})
    def list_transactions(self, customer_id, **filters):
        params = {("type" if k == "txn_type" else k): v for k, v in filters.items() if v is not None}
        return self._request("GET", f"/api/accounts/{customer_id}/transactions", params=params)
    def get_transaction(self, customer_id, transaction_id):
        return self._request("GET", f"/api/transactions/{transaction_id}", params={"customer_id": customer_id})
    def explain_transaction(self, customer_id, transaction_id=None):
        return self._request("GET", f"/api/accounts/{customer_id}/deductions/explain", params={"transaction_id": transaction_id})
    def list_loans(self): return self._request("GET", "/api/loans")
    def get_loan(self, loan_type): return self._request("GET", f"/api/loans/{loan_type}")
    def get_interest_rates(self, product_type=None): return self._request("GET", "/api/interest-rates", params={"product_type": product_type})
    def check_eligibility(self, customer_id, loan_type, loan_amount=None, tenure_months=None, declared_monthly_income=None):
        body = {"customer_id": customer_id, "loan_type": loan_type, "loan_amount": loan_amount,
                "tenure_months": tenure_months, "declared_monthly_income": declared_monthly_income}
        return self._request("POST", "/api/loans/eligibility", json=body)
    def calculate_emi(self, principal, annual_rate, tenure_months):
        return self._request("POST", "/api/emi/calculate", json={"principal": principal, "annual_rate": annual_rate, "tenure_months": tenure_months})
    def get_charges(self, customer_id): return self._request("GET", f"/api/accounts/{customer_id}/charges")


def build_bank_client(settings: Settings) -> BankClient:
    """Create the bank client selected by ``BANK_API_MODE``."""
    if settings.bank_api_mode == "inprocess":
        from banking_ai.bank.data_store import load_bank_data

        data = load_bank_data(settings.bank_data_dir, settings.bank_as_of_date or None)
        return InProcessBankClient(BankService(data))
    return HttpBankClient(settings.bank_api_url, settings.bank_api_key, settings.bank_api_timeout_seconds)
