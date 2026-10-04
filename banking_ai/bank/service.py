"""Business logic of the mock bank. Used by the FastAPI mock API and by the in-process client."""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Optional

from banking_ai.bank import calculators
from banking_ai.bank.data_store import BankData
from banking_ai.bank.errors import InvalidRequestError, NotFoundError
from banking_ai.security.masking import mask_account_number, mask_email, mask_phone

MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
               "october", "november", "december"]
ELIGIBILITY_DISCLAIMER = ("This is an indicative assessment based on the profile held by the bank; "
                          "final approval is subject to document verification and credit appraisal.")


class BankService:
    """Read-only banking operations over :class:`BankData`."""

    def __init__(self, data: BankData) -> None:
        self.data = data

    # ------------------------------------------------------------------ helpers
    @property
    def as_of(self) -> date:
        """The bank's current business date."""
        return self.data.as_of

    def _customer(self, customer_id: str) -> dict[str, Any]:
        customer = self.data.customers.get(customer_id)
        if customer is None:
            raise NotFoundError("Customer not found.")
        return customer

    def _accounts(self, customer_id: str) -> list[dict[str, Any]]:
        self._customer(customer_id)
        return self.data.accounts_by_customer.get(customer_id, [])

    def _loan(self, loan_type: str) -> dict[str, Any]:
        loan = self.data.loans.get(loan_type)
        if loan is None:
            raise NotFoundError(f"Unknown loan type '{loan_type}'. Available: {', '.join(self.data.loans)}.")
        return loan

    @staticmethod
    def _public_account(account: dict[str, Any]) -> dict[str, Any]:
        return {
            "account_number": mask_account_number(account["account_number"]),
            "account_type": account["account_type"], "display_name": account["display_name"],
            "status": account["status"], "currency": account["currency"], "branch": account["branch"],
            "ifsc": account["ifsc"], "opened_on": account["opened_on"], "is_primary": account["is_primary"],
            "min_balance_required": account["min_balance_required"], "interest_rate": account["interest_rate"],
        }

    @staticmethod
    def _age(dob: str, today: date) -> int:
        born = date.fromisoformat(dob)
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

    def resolve_period(self, period: str) -> tuple[date, date]:
        """Translate symbolic periods (``this_month``, ``last_30_days``...) into ``(start, end)`` dates."""
        today = self.as_of
        p = period.lower().strip()
        if p == "today":
            return today, today
        if p == "yesterday":
            return today - timedelta(days=1), today - timedelta(days=1)
        if p == "this_week":
            return today - timedelta(days=today.weekday()), today
        if p == "last_week":
            end = today - timedelta(days=today.weekday() + 1)
            return end - timedelta(days=6), end
        if p == "this_month":
            return today.replace(day=1), today
        if p == "last_month":
            end = today.replace(day=1) - timedelta(days=1)
            return end.replace(day=1), end
        if p == "this_year":
            return date(today.year, 1, 1), today
        if (m := re.fullmatch(r"last_(\d{1,3})_days", p)):
            return today - timedelta(days=int(m.group(1)) - 1), today
        if (m := re.fullmatch(r"last_(\d{1,2})_months", p)):
            return today - timedelta(days=30 * int(m.group(1))), today
        if (m := re.fullmatch(r"month_(\d{1,2})", p)) and 1 <= int(m.group(1)) <= 12:
            month = int(m.group(1))
            year = today.year if month <= today.month else today.year - 1
            first = date(year, month, 1)
            nxt = date(year + (month == 12), month % 12 + 1, 1)
            return first, min(nxt - timedelta(days=1), today)
        raise InvalidRequestError(f"Unsupported period '{period}'.")

    # ------------------------------------------------------------------ customer / accounts
    def get_customer(self, customer_id: str) -> dict[str, Any]:
        c = self._customer(customer_id)
        return {
            "customer_id": c["customer_id"], "full_name": c["full_name"], "city": c["city"],
            "employment_type": c["employment_type"], "customer_since": c["customer_since"],
            "kyc_status": c["kyc_status"], "email": mask_email(c["email"]), "phone": mask_phone(c["phone"]),
            "monthly_income": c["monthly_income"],
        }

    def get_accounts(self, customer_id: str) -> dict[str, Any]:
        accounts = [self._public_account(a) for a in self._accounts(customer_id)]
        return {"customer_id": customer_id, "accounts": accounts, "count": len(accounts)}

    def get_balance(self, customer_id: str, account_last4: Optional[str] = None) -> dict[str, Any]:
        accounts = self._accounts(customer_id)
        if account_last4:
            accounts = [a for a in accounts if a["account_number"].endswith(account_last4)]
            if not accounts:
                raise NotFoundError("No account found with those last four digits.")
        rows = [{
            "account_number": mask_account_number(a["account_number"]), "account_type": a["account_type"],
            "display_name": a["display_name"], "balance": a["balance"], "available_balance": a["available_balance"],
            "lien_amount": a["lien_amount"], "currency": a["currency"], "is_primary": a["is_primary"],
        } for a in sorted(accounts, key=lambda a: not a["is_primary"])]
        return {"customer_id": customer_id, "as_of": self.as_of.isoformat(), "accounts": rows}

    # ------------------------------------------------------------------ transactions
    @staticmethod
    def _public_txn(txn: dict[str, Any]) -> dict[str, Any]:
        return {
            "transaction_id": txn["transaction_id"], "date": txn["date"], "time": txn["time"],
            "description": txn["description"], "amount": txn["amount"], "type": txn["type"],
            "channel": txn["channel"], "category": txn["category"], "balance_after": txn["balance_after"],
            "account_number": mask_account_number(txn["account_number"]),
        }

    def list_transactions(
        self, customer_id: str, limit: int = 5, period: Optional[str] = None, start_date: Optional[str] = None,
        end_date: Optional[str] = None, txn_type: Optional[str] = None, channel: Optional[str] = None,
        account_last4: Optional[str] = None,
    ) -> dict[str, Any]:
        numbers = {a["account_number"] for a in self._accounts(customer_id)
                   if not account_last4 or a["account_number"].endswith(account_last4)}
        if not numbers:
            raise NotFoundError("No matching account found.")
        limit = max(1, min(int(limit or 5), 50))
        start = end = None
        if period:
            start, end = self.resolve_period(period)
        try:
            start = date.fromisoformat(start_date) if start_date else start
            end = date.fromisoformat(end_date) if end_date else end
        except ValueError as exc:
            raise InvalidRequestError("Dates must use the YYYY-MM-DD format.") from exc
        if start and end and start > end:
            raise InvalidRequestError("start_date must not be after end_date.")
        if txn_type and txn_type not in {"credit", "debit"}:
            raise InvalidRequestError("type must be 'credit' or 'debit'.")

        matching = []
        for t in self.data.transactions:
            if t["account_number"] not in numbers:
                continue
            d = date.fromisoformat(t["date"])
            if (start and d < start) or (end and d > end) or (txn_type and t["type"] != txn_type):
                continue
            if channel and t["channel"].lower() != channel.lower():
                continue
            matching.append(t)
        matching.sort(key=lambda t: (t["date"], t["time"], t["transaction_id"]), reverse=True)
        selected = matching[:limit]
        return {
            "customer_id": customer_id, "returned": len(selected), "total_matching": len(matching),
            "period": {"start": start.isoformat() if start else None, "end": end.isoformat() if end else None},
            "filters": {"type": txn_type, "channel": channel},
            "totals_matching": {
                "credits": round(sum(t["amount"] for t in matching if t["type"] == "credit"), 2),
                "debits": round(sum(t["amount"] for t in matching if t["type"] == "debit"), 2),
            },
            "transactions": [self._public_txn(t) for t in selected],
        }

    def get_transaction(self, customer_id: str, transaction_id: str) -> dict[str, Any]:
        owned = {a["account_number"] for a in self._accounts(customer_id)}
        txn = self.data.txn_by_id.get(transaction_id)
        if txn is None or txn["account_number"] not in owned:  # same error for "not yours" and "missing"
            raise NotFoundError("Transaction not found.")
        return self._public_txn(txn)

    def explain_transaction(self, customer_id: str, transaction_id: Optional[str] = None) -> dict[str, Any]:
        """Explain a specific transaction, or the most recent charge/deduction when no id is given."""
        owned = {a["account_number"] for a in self._accounts(customer_id)}
        if transaction_id:
            txn = self.data.txn_by_id.get(transaction_id)
            if txn is None or txn["account_number"] not in owned:
                raise NotFoundError("Transaction not found.")
        else:
            debits = sorted((t for t in self.data.transactions if t["account_number"] in owned and t["type"] == "debit"),
                            key=lambda t: (t["date"], t["time"], t["transaction_id"]), reverse=True)
            if not debits:
                raise NotFoundError("No debit transactions found.")
            charges = [t for t in debits if t["category"] == "fee"]
            txn = charges[0] if charges else debits[0]
        entry = next((c for c in self.data.charge_catalog if c["match"] in txn["description"].upper()), None)
        if entry:
            label, explanation, is_charge = entry["label"], entry["explanation"], entry["is_charge"]
        else:
            merchant = txn.get("merchant") or "a merchant"
            label = "Payment"
            explanation = f"A {txn['channel']} payment of this amount was made to {merchant} from your account."
            is_charge = False
        return {
            "transaction": self._public_txn(txn), "label": label, "explanation": explanation, "is_bank_charge": is_charge,
            "selected_by": "transaction_id" if transaction_id else "most_recent_charge_or_debit",
        }

    # ------------------------------------------------------------------ products
    def list_loans(self) -> dict[str, Any]:
        loans = [self._loan_summary(l) for l in self.data.loans.values()]
        return {"loans": loans, "count": len(loans)}

    def get_loan(self, loan_type: str) -> dict[str, Any]:
        return self._loan_summary(self._loan(loan_type))

    @staticmethod
    def _loan_summary(loan: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in loan.items()}

    def get_interest_rates(self, product_type: Optional[str] = None) -> dict[str, Any]:
        rates = self.data.interest_rates
        if product_type:
            rates = [r for r in rates if r["product_type"] == product_type]
            if not rates:
                known = ", ".join(r["product_type"] for r in self.data.interest_rates)
                raise NotFoundError(f"Unknown product '{product_type}'. Available: {known}.")
        return {"as_of": self.as_of.isoformat(), "rates": rates}

    def get_charges(self, customer_id: str) -> dict[str, Any]:
        accounts = self._accounts(customer_id)
        owned = {a["account_number"] for a in accounts}
        types = sorted({a["account_type"] for a in accounts})
        since = self.as_of - timedelta(days=90)
        recent = [self._public_txn(t) for t in sorted(self.data.transactions, key=lambda t: (t["date"], t["time"]), reverse=True)
                  if t["account_number"] in owned and t["category"] == "fee" and date.fromisoformat(t["date"]) >= since]
        return {
            "customer_id": customer_id,
            "account_types": [self.data.charges["account_types"][t] | {"account_type": t} for t in types],
            "charge_schedule": self.data.charges["items"], "gst_rate_pct": self.data.charges["gst_rate_pct"],
            "recent_charges_90_days": recent[:10],
            "total_recent_charges_90_days": round(sum(t["amount"] for t in recent), 2),
        }

    # ------------------------------------------------------------------ calculators
    def calculate_emi(self, principal: float, annual_rate: float, tenure_months: int) -> dict[str, Any]:
        if tenure_months > 480:
            raise InvalidRequestError("Tenure cannot exceed 480 months.")
        if annual_rate > 50:
            raise InvalidRequestError("Interest rate looks invalid (max 50%).")
        return calculators.emi_breakdown(principal, annual_rate, tenure_months)

    def indicative_rate(self, loan: dict[str, Any], credit_score: int) -> float:
        """Linear pricing between ``rate_max`` (score 700) and ``rate_min`` (score 800)."""
        weight = min(max((credit_score - 700) / 100, 0.0), 1.0)
        return round(loan["rate_max"] - (loan["rate_max"] - loan["rate_min"]) * weight, 2)

    def check_loan_eligibility(
        self, customer_id: str, loan_type: str, loan_amount: Optional[float] = None,
        tenure_months: Optional[int] = None, declared_monthly_income: Optional[float] = None,
    ) -> dict[str, Any]:
        customer, loan = self._customer(customer_id), self._loan(loan_type)
        tenure = int(tenure_months or loan["default_tenure_months"])
        assumptions = []
        if not tenure_months:
            assumptions.append(f"Tenure of {calculators.format_months(tenure)} assumed (product default).")
        income = float(customer["monthly_income"])
        if declared_monthly_income and abs(declared_monthly_income - income) > 1:
            assumptions.append("The decision uses the income on record with the bank, not the declared income.")
        age = self._age(customer["date_of_birth"], self.as_of)
        rate = self.indicative_rate(loan, customer["credit_score"])
        capacity = max(0.0, income * loan["foir_limit"] - customer["existing_emi"])
        max_amount = min(loan["max_amount"], calculators.max_principal_for_emi(capacity, rate, tenure))
        max_amount = float(int(max_amount // 1000 * 1000))

        checks = [
            {"check": "age", "passed": loan["min_age"] <= age <= loan["max_age"],
             "detail": f"Allowed age {loan['min_age']}-{loan['max_age']} years."},
            {"check": "monthly_income", "passed": income >= loan["min_monthly_income"],
             "detail": f"Minimum monthly income {calculators.format_inr(loan['min_monthly_income'])}."},
            {"check": "credit_score", "passed": customer["credit_score"] >= loan["min_credit_score"],
             "detail": f"Minimum credit score {loan['min_credit_score']}."},
            {"check": "tenure", "passed": loan["min_tenure_months"] <= tenure <= loan["max_tenure_months"],
             "detail": f"Allowed tenure {calculators.format_months(loan['min_tenure_months'])} to "
                       f"{calculators.format_months(loan['max_tenure_months'])}."},
        ]
        emi = emi_to_income = None
        if loan_amount:
            emi = calculators.calculate_emi(loan_amount, rate, tenure)
            emi_to_income = round((emi + customer["existing_emi"]) / income, 4)
            checks.append({"check": "loan_amount_range", "passed": loan["min_amount"] <= loan_amount <= loan["max_amount"],
                           "detail": f"Allowed amount {calculators.format_inr(loan['min_amount'])} to {calculators.format_inr(loan['max_amount'])}."})
            checks.append({"check": "affordability", "passed": emi_to_income <= loan["foir_limit"],
                           "detail": f"Total EMIs must not exceed {int(loan['foir_limit'] * 100)}% of monthly income."})
        profile_checks = {"age", "monthly_income", "credit_score", "tenure"}
        profile_ok = all(c["passed"] for c in checks if c["check"] in profile_checks) and max_amount >= loan["min_amount"]
        eligible = profile_ok and all(c["passed"] for c in checks)
        return {
            "customer_id": customer_id, "loan_type": loan_type, "loan_name": loan["display_name"],
            "requested_amount": loan_amount, "tenure_months": tenure, "eligible": eligible,
            "indicative_rate": rate, "estimated_emi": emi, "emi_to_income_ratio": emi_to_income,
            "existing_monthly_emi": customer["existing_emi"], "monthly_income_on_record": income,
            "max_eligible_amount": max_amount if profile_ok else 0.0, "profile_eligible": profile_ok, "checks": checks, "assumptions": assumptions,
            "processing_fee_pct": loan["processing_fee_pct"], "disclaimer": ELIGIBILITY_DISCLAIMER,
        }
