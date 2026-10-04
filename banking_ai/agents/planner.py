"""Declarative tool planner used in demo mode (no LLM) and as the safety net when the LLM calls no tool.

With a real LLM the *model* chooses tools (see ``langchain_agent.py``). This planner encodes the same policy as a
table: intent + entities -> tool calls. It is deterministic so the system can be tested and run without API keys.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from banking_ai.nlp.entities import ExtractedEntities

ACCOUNT_PRODUCT = {"savings": "savings_account", "salary": "salary_account", "current": "current_account"}


@dataclass
class PlannedCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    calls: list[PlannedCall] = field(default_factory=list)
    clarification: Optional[str] = None
    notes: list[str] = field(default_factory=list)


def _clean(args: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in args.items() if v is not None}


def _balance(e: ExtractedEntities, q: str) -> Plan:
    return Plan([PlannedCall("get_account_balance", _clean({"account_last4": e.account_number_last4}))])


def _history(e: ExtractedEntities, q: str) -> Plan:
    filtered = bool(e.period or e.start_date or e.end_date)
    limit = e.number_of_transactions or (20 if filtered else 5)
    return Plan([PlannedCall("get_transaction_history", _clean({
        "limit": limit, "period": e.period, "start_date": e.start_date, "end_date": e.end_date,
        "transaction_type": e.transaction_type, "channel": e.transaction_channel.upper() if e.transaction_channel else None,
        "account_last4": e.account_number_last4}))])


def _recent(e: ExtractedEntities, q: str) -> Plan:
    return Plan([PlannedCall("get_recent_transaction", _clean({"transaction_type": e.transaction_type}))])


def _eligibility(e: ExtractedEntities, q: str) -> Plan:
    loan_type = e.loan_type or (e.loan_types[0] if e.loan_types else None)
    notes = [] if loan_type else ["I assumed a personal loan since no loan type was mentioned."]
    return Plan([PlannedCall("check_loan_eligibility", _clean({
        "loan_type": loan_type or "personal_loan", "loan_amount": e.loan_amount or e.amount,
        "tenure_months": e.tenure_months, "monthly_income": e.monthly_income}))], notes=notes)


def _loan_info(e: ExtractedEntities, q: str) -> Plan:
    return Plan([PlannedCall("get_loan_products", _clean({"loan_type": e.loan_type}))])


def _rates(e: ExtractedEntities, q: str) -> Plan:
    text = q.lower()
    product = e.loan_type or ACCOUNT_PRODUCT.get(e.account_type or "")
    if product is None:
        if "fixed deposit" in text or " fd" in f" {text}":
            product = "fixed_deposit"
        elif "recurring" in text or " rd" in f" {text}":
            product = "recurring_deposit"
    return Plan([PlannedCall("get_interest_rates", _clean({"product_type": product}))])


def _account_info(e: ExtractedEntities, q: str) -> Plan:
    return Plan([PlannedCall("get_account_details"), PlannedCall("search_knowledge_base", {"query": q})])


def _charges(e: ExtractedEntities, q: str) -> Plan:
    return Plan([PlannedCall("get_account_charges")])


def _min_balance(e: ExtractedEntities, q: str) -> Plan:
    calls = [PlannedCall("search_knowledge_base", {"query": q})]
    if not e.account_type:
        calls.append(PlannedCall("get_account_details"))
    return Plan(calls)


def _emi(e: ExtractedEntities, q: str) -> Plan:
    principal, tenure = e.loan_amount or e.amount, e.tenure_months
    missing = [label for label, value in (("loan amount", principal), ("tenure (for example 5 years)", tenure)) if not value]
    if missing:
        return Plan(clarification="To calculate your EMI I need the " + " and the ".join(missing) + ". You can also add an interest rate, otherwise I'll use the product's indicative rate.")
    return Plan([PlannedCall("calculate_emi", _clean({"principal": principal, "tenure_months": tenure,
                                                      "annual_rate": e.interest_rate, "loan_type": e.loan_type}))])


def _explain(e: ExtractedEntities, q: str) -> Plan:
    return Plan([PlannedCall("explain_transaction", _clean({"transaction_id": e.transaction_id}))])


def _compare(e: ExtractedEntities, q: str) -> Plan:
    return Plan([PlannedCall("compare_loan_products", {"loan_types": e.loan_types})])


ROUTES: dict[str, Callable[[ExtractedEntities, str], Plan]] = {
    "balance_inquiry": _balance, "transaction_history": _history, "recent_transaction": _recent,
    "loan_eligibility": _eligibility, "loan_information": _loan_info, "interest_rate": _rates,
    "account_information": _account_info, "account_charges": _charges, "minimum_balance": _min_balance,
    "emi_calculation": _emi, "transaction_explanation": _explain, "loan_comparison": _compare,
}


def plan_for(intent: str, entities: ExtractedEntities, query: str) -> Plan:
    """Return the tool plan for an intent (empty plan for small talk / fallback)."""
    route = ROUTES.get(intent)
    return route(entities, query) if route else Plan()
