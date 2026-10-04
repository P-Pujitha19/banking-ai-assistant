"""The banking tools, their registry and the safe executor that runs them."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from banking_ai.bank.client import BankClient
from banking_ai.bank.errors import BankAPIError
from banking_ai.config.logging_config import get_logger, log_event
from banking_ai.rag.service import RagService
from banking_ai.tools import schemas as s

logger = get_logger("tools")

# Where each tool's data comes from (the four-way separation required by the design)
CUSTOMER_DATA, PRODUCT_CATALOG, KNOWLEDGE_BASE, CALCULATION = "customer_data", "product_catalog", "knowledge_base", "calculation"


@dataclass
class ToolSpec:
    name: str
    description: str
    args_schema: Any
    source_type: str
    func: Callable[..., dict[str, Any]]


@dataclass
class ToolOutcome:
    """Result of one tool execution."""

    name: str
    args: dict[str, Any]
    ok: bool
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    source_type: str = ""
    latency_ms: float = 0.0

    def to_public(self) -> dict[str, Any]:
        return {"tool": self.name, "args": self.args, "ok": self.ok, "source": self.source_type,
                "latency_ms": round(self.latency_ms, 1), "error": self.error}


class BankingToolkit:
    """Tools bound to ONE authenticated customer (the id never comes from the user or the LLM)."""

    def __init__(self, customer_id: str, bank: BankClient, rag: RagService) -> None:
        self.customer_id, self.bank, self.rag = customer_id, bank, rag

    # ---- customer data -------------------------------------------------
    def get_account_balance(self, account_last4: Optional[str] = None) -> dict[str, Any]:
        return self.bank.get_balance(self.customer_id, account_last4)

    def get_account_details(self) -> dict[str, Any]:
        return {"customer": self.bank.get_customer(self.customer_id), "accounts": self.bank.get_accounts(self.customer_id)["accounts"]}

    def get_transaction_history(self, limit: int = 5, period: Optional[str] = None, start_date: Optional[str] = None,
                                end_date: Optional[str] = None, transaction_type: Optional[str] = None,
                                channel: Optional[str] = None, account_last4: Optional[str] = None) -> dict[str, Any]:
        return self.bank.list_transactions(self.customer_id, limit=limit, period=period, start_date=start_date, end_date=end_date,
                                           txn_type=transaction_type, channel=channel, account_last4=account_last4)

    def get_recent_transaction(self, transaction_type: Optional[str] = None) -> dict[str, Any]:
        return self.bank.list_transactions(self.customer_id, limit=1, txn_type=transaction_type)

    def check_loan_eligibility(self, loan_type: str, loan_amount: Optional[float] = None, tenure_months: Optional[int] = None,
                               monthly_income: Optional[float] = None) -> dict[str, Any]:
        return self.bank.check_eligibility(self.customer_id, loan_type, loan_amount, tenure_months, monthly_income)

    def get_account_charges(self) -> dict[str, Any]:
        return self.bank.get_charges(self.customer_id)

    def explain_transaction(self, transaction_id: Optional[str] = None) -> dict[str, Any]:
        return self.bank.explain_transaction(self.customer_id, transaction_id)

    # ---- product catalogue ---------------------------------------------
    def get_loan_products(self, loan_type: Optional[str] = None) -> dict[str, Any]:
        return self.bank.get_loan(loan_type) if loan_type else self.bank.list_loans()

    def get_interest_rates(self, product_type: Optional[str] = None) -> dict[str, Any]:
        return self.bank.get_interest_rates(product_type)

    def compare_loan_products(self, loan_types: list[str]) -> dict[str, Any]:
        types = list(dict.fromkeys(loan_types))
        if len(types) < 2:
            types = ["personal_loan", "home_loan"]
        return {"products": [self.bank.get_loan(t) for t in types[:4]]}

    # ---- calculator -------------------------------------------------------
    def calculate_emi(self, principal: float, tenure_months: int, annual_rate: Optional[float] = None,
                      loan_type: Optional[str] = None) -> dict[str, Any]:
        rate_source, loan_name = "customer_supplied", None
        if annual_rate is None:
            loan = self.bank.get_loan(loan_type or "personal_loan")
            annual_rate, rate_source, loan_name = loan["indicative_rate"], "product_indicative_rate", loan["display_name"]
        result = self.bank.calculate_emi(principal, annual_rate, tenure_months)
        return {**result, "rate_source": rate_source, "loan_name": loan_name}

    # ---- RAG ------------------------------------------------------------
    def search_knowledge_base(self, query: str) -> dict[str, Any]:
        result = self.rag.retrieve(query)
        return {"grounded": result.grounded, "query": query,
                "passages": [{"title": c.title, "section": c.section, "score": c.score, "lines": c.lines} for c in result.chunks]}

    def specs(self) -> dict[str, ToolSpec]:
        table = [
            ("get_account_balance", "Get the customer's account balance and available balance.", s.BalanceArgs, CUSTOMER_DATA, self.get_account_balance),
            ("get_account_details", "Get the customer's profile and account details (type, status, branch, minimum balance).", s.NoArgs, CUSTOMER_DATA, self.get_account_details),
            ("get_transaction_history", "List the customer's transactions, newest first, optionally filtered by count, period, dates, type or channel.", s.TransactionHistoryArgs, CUSTOMER_DATA, self.get_transaction_history),
            ("get_recent_transaction", "Get the customer's single most recent transaction (optionally the latest credit or debit).", s.RecentTransactionArgs, CUSTOMER_DATA, self.get_recent_transaction),
            ("check_loan_eligibility", "Check the customer's eligibility for a loan product and affordability of an amount/tenure.", s.EligibilityArgs, CUSTOMER_DATA, self.check_loan_eligibility),
            ("get_account_charges", "Get the fee schedule for the customer's account types and recent charges deducted.", s.NoArgs, CUSTOMER_DATA, self.get_account_charges),
            ("explain_transaction", "Explain why a transaction or the most recent charge/deduction happened.", s.ExplainTransactionArgs, CUSTOMER_DATA, self.explain_transaction),
            ("get_loan_products", "Get loan product details (amounts, tenure, rates, fees, eligibility limits).", s.LoanProductsArgs, PRODUCT_CATALOG, self.get_loan_products),
            ("get_interest_rates", "Get official deposit and loan interest rates.", s.InterestRatesArgs, PRODUCT_CATALOG, self.get_interest_rates),
            ("compare_loan_products", "Compare two or more loan products side by side.", s.CompareLoansArgs, PRODUCT_CATALOG, self.compare_loan_products),
            ("calculate_emi", "Calculate the monthly EMI, total interest and total payment for a loan.", s.EmiArgs, CALCULATION, self.calculate_emi),
            ("search_knowledge_base", "Search general banking knowledge (policies, minimum balance, features, limits, FAQs).", s.KnowledgeSearchArgs, KNOWLEDGE_BASE, self.search_knowledge_base),
        ]
        return {n: ToolSpec(n, d, a, st, f) for n, d, a, st, f in table}


class ToolExecutor:
    """Validates arguments, runs a tool, times it, and converts every failure into a safe ToolOutcome."""

    def __init__(self, specs: dict[str, ToolSpec]) -> None:
        self.specs = specs
        self.outcomes: list[ToolOutcome] = []

    def execute(self, name: str, args: Optional[dict[str, Any]] = None) -> ToolOutcome:
        args = dict(args or {})
        started = time.perf_counter()
        spec = self.specs.get(name)
        if spec is None:
            outcome = ToolOutcome(name, args, False, error="unknown_tool")
        else:
            try:
                validated = spec.args_schema(**args).model_dump()
                data = spec.func(**validated)
                outcome = ToolOutcome(name, validated, True, data=data, source_type=spec.source_type)
            except BankAPIError as exc:
                outcome = ToolOutcome(name, args, False, error=f"bank_api_{exc.status_code}: {exc.message}", source_type=spec.source_type)
            except (ValueError, TypeError) as exc:  # argument validation problems
                outcome = ToolOutcome(name, args, False, error=f"invalid_arguments: {str(exc)[:120]}", source_type=spec.source_type)
            except Exception:  # never leak internals to the user/LLM
                logger.exception("tool crashed: %s", name)
                outcome = ToolOutcome(name, args, False, error="internal_error", source_type=spec.source_type)
        outcome.latency_ms = (time.perf_counter() - started) * 1000
        self.outcomes.append(outcome)
        log_event(logger, "tool_executed", tool=name, ok=outcome.ok, latency_ms=round(outcome.latency_ms, 1), source=outcome.source_type)
        return outcome

    def execute_json(self, name: str, **args: Any) -> str:
        """Run a tool and return a JSON string (what is fed back to the LLM as a ToolMessage)."""
        outcome = self.execute(name, args)
        return json.dumps({"ok": outcome.ok, "data": outcome.data, "error": outcome.error}, default=str, ensure_ascii=False)
