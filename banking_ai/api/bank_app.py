"""FastAPI *mock core-banking* service (synthetic data only).

Run:  uvicorn banking_ai.api.bank_app:app --port 8001
Docs: http://127.0.0.1:8001/docs
"""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse

from banking_ai.api.schemas import EligibilityRequest, EmiRequest
from banking_ai.bank.data_store import load_bank_data
from banking_ai.bank.errors import BankError
from banking_ai.bank.service import BankService
from banking_ai.config.logging_config import configure_logging, get_logger
from banking_ai.config.settings import Settings, get_settings

CUSTOMER_ID = Path(pattern=r"^CUST\d{4}$", description="Customer id such as CUST1001")


def create_bank_app(settings: Optional[Settings] = None, service: Optional[BankService] = None) -> FastAPI:
    """Application factory (the factory makes the app easy to test with a temporary data set)."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_file, settings.log_json)
    logger = get_logger("bank_api")
    bank = service or BankService(load_bank_data(settings.bank_data_dir, settings.bank_as_of_date or None))
    app = FastAPI(title="Mock Core Banking API", version="1.0.0",
                  description="Synthetic banking backend for the AI Banking Assistant. Contains NO real data. "
                              "Account numbers are masked; send the X-API-Key header when BANK_API_KEY is set.")

    def require_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
        if settings.bank_api_key and x_api_key != settings.bank_api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    @app.exception_handler(BankError)
    async def bank_error_handler(_: Request, exc: BankError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.get("/health", tags=["system"])
    def health() -> dict:
        return {"status": "ok", "bank_date": bank.as_of.isoformat(), "customers": len(bank.data.customers)}

    guard = [Depends(require_key)]

    @app.get("/api/customers/{customer_id}", tags=["customers"], dependencies=guard)
    def get_customer(customer_id: str = CUSTOMER_ID) -> dict:
        return bank.get_customer(customer_id)

    @app.get("/api/accounts/{customer_id}", tags=["accounts"], dependencies=guard)
    def get_accounts(customer_id: str = CUSTOMER_ID) -> dict:
        return bank.get_accounts(customer_id)

    @app.get("/api/accounts/{customer_id}/balance", tags=["accounts"], dependencies=guard)
    def get_balance(customer_id: str = CUSTOMER_ID, account_last4: Optional[str] = Query(default=None, pattern=r"^\d{4}$")) -> dict:
        return bank.get_balance(customer_id, account_last4)

    @app.get("/api/accounts/{customer_id}/transactions", tags=["transactions"], dependencies=guard)
    def list_transactions(
        customer_id: str = CUSTOMER_ID, limit: int = Query(5, ge=1, le=50),
        period: Optional[str] = Query(default=None, description="this_month, last_month, today, yesterday, this_week, last_week, last_N_days, month_N"),
        start_date: Optional[str] = Query(default=None, description="YYYY-MM-DD"), end_date: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
        type: Optional[str] = Query(default=None, pattern="^(credit|debit)$"), channel: Optional[str] = None,
        account_last4: Optional[str] = Query(default=None, pattern=r"^\d{4}$"),
    ) -> dict:
        return bank.list_transactions(customer_id, limit, period, start_date, end_date, type, channel, account_last4)

    @app.get("/api/accounts/{customer_id}/charges", tags=["accounts"], dependencies=guard)
    def get_charges(customer_id: str = CUSTOMER_ID) -> dict:
        return bank.get_charges(customer_id)

    @app.get("/api/accounts/{customer_id}/deductions/explain", tags=["transactions"], dependencies=guard)
    def explain(customer_id: str = CUSTOMER_ID, transaction_id: Optional[str] = Query(default=None, pattern=r"^TXN\d{6,}$")) -> dict:
        return bank.explain_transaction(customer_id, transaction_id)

    @app.get("/api/transactions/{transaction_id}", tags=["transactions"], dependencies=guard)
    def get_transaction(transaction_id: str = Path(pattern=r"^TXN\d{6,}$"), customer_id: str = Query(pattern=r"^CUST\d{4}$")) -> dict:
        """The owning customer_id is required so one customer can never read another's transaction."""
        return bank.get_transaction(customer_id, transaction_id)

    @app.get("/api/loans", tags=["loans"], dependencies=guard)
    def list_loans() -> dict:
        return bank.list_loans()

    @app.get("/api/loans/{loan_type}", tags=["loans"], dependencies=guard)
    def get_loan(loan_type: str) -> dict:
        return bank.get_loan(loan_type)

    @app.get("/api/interest-rates", tags=["rates"], dependencies=guard)
    def interest_rates(product_type: Optional[str] = None) -> dict:
        return bank.get_interest_rates(product_type)

    @app.post("/api/loans/eligibility", tags=["loans"], dependencies=guard)
    def eligibility(body: EligibilityRequest) -> dict:
        return bank.check_loan_eligibility(body.customer_id, body.loan_type, body.loan_amount, body.tenure_months, body.declared_monthly_income)

    @app.post("/api/emi/calculate", tags=["calculators"], dependencies=guard)
    def emi(body: EmiRequest) -> dict:
        return bank.calculate_emi(body.principal, body.annual_rate, body.tenure_months)

    logger.info("mock bank api ready")
    return app


app = create_bank_app()
