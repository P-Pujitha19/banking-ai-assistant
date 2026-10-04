"""Deterministic, fully grounded response composer.

Used (a) in demo mode when no LLM is configured, (b) for greetings/help/fallback, and (c) as the safe fallback
whenever an LLM answer fails the grounding check.  Every number it prints comes straight from tool results.
"""
from __future__ import annotations

from typing import Any, Optional

from banking_ai.bank.calculators import format_inr as inr
from banking_ai.bank.calculators import format_months
from banking_ai.nlp.entities import ExtractedEntities
from banking_ai.rag.service import RetrievalResult, extract_relevant_lines
from banking_ai.tools.toolkit import ToolOutcome

CAPABILITIES = ("check your **balance**, view **transactions**, understand **charges or deductions**, see **account details**, "
                "explore **loan products and interest rates**, **check loan eligibility**, **calculate EMIs**, "
                "**compare loans**, and answer general questions such as minimum balance or transaction limits")
SERVICE_ERROR = ("I wasn't able to retrieve that information right now, and I don't want to guess with your banking details. "
                 "Please try again in a moment or contact customer support.")
PERIOD_WORDS = {"this_month": "this month", "last_month": "last month", "today": "today", "yesterday": "yesterday",
                "this_week": "this week", "last_week": "last week", "this_year": "this year"}
KB_TRIGGERS = ("about", "feature", "limit", "benefit", "tell me", "explain", "how", "what is", "what are")


def _pct(value: float) -> str:
    return f"{value:.2f}%"


class TemplateResponder:
    """Builds answers from verified tool outcomes."""

    def compose(self, intent: str, entities: ExtractedEntities, outcomes: list[ToolOutcome], query: str,
                notes: Optional[list[str]] = None) -> str:
        small_talk = {"greeting": self._greeting, "goodbye": self._goodbye, "help": self._help, "fallback": self._fallback}
        if intent in small_talk:
            return small_talk[intent]()
        by_name = {o.name: o for o in outcomes}
        if outcomes and not any(o.ok for o in outcomes):
            return self._failure(outcomes)
        builders = {
            "balance_inquiry": self._balance, "transaction_history": self._history, "recent_transaction": self._recent,
            "loan_eligibility": self._eligibility, "loan_information": self._loan_info, "interest_rate": self._rates,
            "account_information": self._account_info, "account_charges": self._charges, "minimum_balance": self._min_balance,
            "emi_calculation": self._emi, "transaction_explanation": self._explain, "loan_comparison": self._compare,
        }
        builder = builders.get(intent)
        if builder is None or not outcomes:
            return self._fallback()
        text = builder(by_name, entities, query)
        failed = [o for o in outcomes if not o.ok]
        if failed and text:
            text += "\n\n_Note: I couldn't retrieve some of the supporting information, so this answer may be incomplete._"
        if notes:
            text += "\n\n" + " ".join(f"_{n}_" for n in notes)
        return text

    # ------------------------------------------------------------------ small talk
    @staticmethod
    def _greeting() -> str:
        return f"Hello! I'm your banking assistant. I can help you {CAPABILITIES}. What would you like to do?"

    @staticmethod
    def _goodbye() -> str:
        return "Thank you for banking with us. Have a wonderful day! 👋"

    @staticmethod
    def _help() -> str:
        return (f"I can help you {CAPABILITIES}.\n\nTry: *\"What is my balance?\"*, *\"Show my last 5 transactions\"*, "
                "*\"Can I get a personal loan of 5 lakh for 5 years?\"* or *\"Calculate EMI for 5 lakh for 5 years at 10%\"*.")

    @staticmethod
    def _fallback() -> str:
        return ("I'm sorry, I can only help with banking questions, and I'm not sure I understood that. "
                f"I can help you {CAPABILITIES}. Could you rephrase your question?")

    @staticmethod
    def _failure(outcomes: list[ToolOutcome]) -> str:
        errors = " ".join(o.error or "" for o in outcomes)
        if "404" in errors and "Customer not found" in errors:
            return "I couldn't find your customer profile. Please contact customer support."
        if "404" in errors and ("Transaction not found" in errors or "No debit" in errors):
            return "I couldn't find a matching transaction. Please check the transaction ID or try a different date range."
        if "invalid_arguments" in errors or "422" in errors:
            return "Some of the details look invalid (for example the amount, tenure or dates). Could you check them and try again?"
        return SERVICE_ERROR

    # ------------------------------------------------------------------ customer data
    @staticmethod
    def _balance(o: dict[str, ToolOutcome], e: ExtractedEntities, q: str) -> str:
        accounts = o["get_account_balance"].data["accounts"]
        if not accounts:
            return "I couldn't find any accounts linked to your profile."

        def lien(a: dict[str, Any]) -> str:
            return f"; {inr(a['lien_amount'])} is on hold" if a["lien_amount"] else ""

        if len(accounts) == 1:
            a = accounts[0]
            return (f"Your available balance in your {a['display_name']} ({a['account_number']}) is **{inr(a['available_balance'])}** "
                    f"(ledger balance {inr(a['balance'])}{lien(a)}).")
        lines = [f"- {a['display_name']} ({a['account_number']}): available **{inr(a['available_balance'])}** (ledger {inr(a['balance'])}{lien(a)})" for a in accounts]
        return "Here are your account balances:\n" + "\n".join(lines)

    @staticmethod
    def _txn_line(t: dict[str, Any]) -> str:
        sign = "Credit" if t["type"] == "credit" else "Debit"
        return f"- {t['date']} · {t['description']} · {sign} **{inr(t['amount'])}** · balance {inr(t['balance_after'])}"

    def _history(self, o: dict[str, ToolOutcome], e: ExtractedEntities, q: str) -> str:
        d = o["get_transaction_history"].data
        if not d["transactions"]:
            return "I couldn't find any transactions matching that request."
        period = d["period"]
        when = ""
        if period["start"] or period["end"]:
            when = f" from {period['start'] or 'the beginning'} to {period['end'] or 'today'}"
        head = f"Here are your latest {d['returned']} transaction{'s' if d['returned'] != 1 else ''}{when}:"
        text = head + "\n" + "\n".join(self._txn_line(t) for t in d["transactions"])
        if d["total_matching"] > d["returned"]:
            text += f"\n\nShowing {d['returned']} of {d['total_matching']} matching transactions - ask for a specific number to see more."
        if when:
            text += f"\n\nTotals for the period: credits {inr(d['totals_matching']['credits'])}, debits {inr(d['totals_matching']['debits'])}."
        return text

    @staticmethod
    def _recent(o: dict[str, ToolOutcome], e: ExtractedEntities, q: str) -> str:
        txns = o["get_recent_transaction"].data["transactions"]
        if not txns:
            return "I couldn't find a matching transaction on your account."
        t = txns[0]
        kind = t["type"]
        return (f"Your most recent {kind} was **{inr(t['amount'])}** on {t['date']} at {t['time']} - {t['description']} "
                f"({t['channel']}). Your balance after this transaction was {inr(t['balance_after'])}.")

    @staticmethod
    def _explain(o: dict[str, ToolOutcome], e: ExtractedEntities, q: str) -> str:
        d = o["explain_transaction"].data
        t = d["transaction"]
        text = (f"**{d['label']}** - {inr(t['amount'])} was {'debited' if t['type'] == 'debit' else 'credited'} on {t['date']} "
                f"({t['description']}).\n\n{d['explanation']}")
        if d["selected_by"] != "transaction_id":
            text += "\n\nThis is the most recent charge/deduction on your account. If you mean a different one, send me its transaction ID."
        return text

    @staticmethod
    def _eligibility(o: dict[str, ToolOutcome], e: ExtractedEntities, q: str) -> str:
        d = o["check_loan_eligibility"].data
        tenure = format_months(d["tenure_months"])
        failed = [c for c in d["checks"] if not c["passed"]]
        if d["eligible"]:
            if d["requested_amount"]:
                text = (f"Good news - based on your profile you appear **eligible** for a {d['loan_name']} of **{inr(d['requested_amount'])}** over {tenure}.\n\n"
                        f"- Estimated EMI: **{inr(d['estimated_emi'])}** per month at an indicative {_pct(d['indicative_rate'])} p.a.\n"
                        f"- Your total EMIs would be {d['emi_to_income_ratio'] * 100:.1f}% of your monthly income (existing EMIs included).\n"
                        f"- Maximum you could borrow over {tenure}: {inr(d['max_eligible_amount'])}.")
            else:
                text = (f"You appear **eligible** for a {d['loan_name']}. Based on your income and existing commitments you could borrow up to "
                        f"**{inr(d['max_eligible_amount'])}** over {tenure} at an indicative {_pct(d['indicative_rate'])} p.a.\n\n"
                        "How much would you like to borrow, and for how long? I can check that and calculate the EMI.")
        else:
            reasons = "\n".join(f"- {c['check'].replace('_', ' ').title()}: {c['detail']}" for c in failed) or "- Profile does not meet the criteria."
            text = f"Based on your profile, you don't currently appear eligible for a {d['loan_name']}" + \
                   (f" of {inr(d['requested_amount'])}" if d["requested_amount"] else "") + f". The checks that didn't pass:\n{reasons}"
            if d.get("profile_eligible") and d["max_eligible_amount"]:
                text += f"\n\nA smaller amount may work: up to **{inr(d['max_eligible_amount'])}** over {tenure}."
        extras = d["assumptions"] + [d["disclaimer"]]
        return text + "\n\n" + " ".join(f"_{x}_" for x in extras)

    @staticmethod
    def _charges(o: dict[str, ToolOutcome], e: ExtractedEntities, q: str) -> str:
        d = o["get_account_charges"].data
        gst = d["gst_rate_pct"]
        lines = ["**Charges for your account(s)**"]
        for acct in d["account_types"]:
            penalty = f"; minimum-balance penalty {inr(acct['penalty'])} + GST" if acct["penalty"] else "; no minimum-balance penalty"
            lines.append(f"- {acct['display_name']}: minimum balance {inr(acct['min_balance'])} ({acct['min_balance_basis']}){penalty}")
        lines.append("\n**Common fees** (plus " + f"{gst}% GST where applicable)")
        for item in d["charge_schedule"]:
            amount = "Free" if item["amount"] == 0 else f"{inr(item['amount'])} {item['frequency']}"
            lines.append(f"- {item['name']}: {amount}")
        if d["recent_charges_90_days"]:
            lines.append(f"\n**Charged to your account in the last 90 days:** {inr(d['total_recent_charges_90_days'])} in total")
            lines += [f"- {t['date']} · {t['description']} · {inr(t['amount'])}" for t in d["recent_charges_90_days"][:4]]
        return "\n".join(lines)

    # ------------------------------------------------------------------ product catalogue
    @staticmethod
    def _loan_info(o: dict[str, ToolOutcome], e: ExtractedEntities, q: str) -> str:
        data = o["get_loan_products"].data
        if "loans" in data:
            lines = [f"- **{l['display_name']}**: {inr(l['min_amount'])} to {inr(l['max_amount'])}, {format_months(l['min_tenure_months'])} to "
                     f"{format_months(l['max_tenure_months'])}, {_pct(l['rate_min'])} - {_pct(l['rate_max'])} p.a." for l in data["loans"]]
            return "We offer these loans:\n" + "\n".join(lines) + "\n\nAsk me to compare two loans, check your eligibility, or calculate an EMI."
        l = data
        return (f"**{l['display_name']}** - {l['purpose']}.\n"
                f"- Amount: {inr(l['min_amount'])} to {inr(l['max_amount'])}\n"
                f"- Tenure: {format_months(l['min_tenure_months'])} to {format_months(l['max_tenure_months'])}\n"
                f"- Interest rate: {_pct(l['rate_min'])} to {_pct(l['rate_max'])} p.a. (indicative {_pct(l['indicative_rate'])})\n"
                f"- Processing fee: {l['processing_fee_pct']}% + GST\n- Prepayment: {l['prepayment_charge']}\n"
                f"- Eligibility: age {l['min_age']}-{l['max_age']}, monthly income from {inr(l['min_monthly_income'])}, credit score {l['min_credit_score']}+")

    @staticmethod
    def _rates(o: dict[str, ToolOutcome], e: ExtractedEntities, q: str) -> str:
        rates = o["get_interest_rates"].data["rates"]
        if len(rates) == 1:
            r = rates[0]
            return f"**{r['product_name']} interest rate**\n" + "\n".join(f"- {d}" for d in r["details"])
        deposits = [r for r in rates if r["category"] == "deposit"]
        loans = [r for r in rates if r["category"] == "loan"]
        lines = ["**Deposit rates**"] + [f"- {r['product_name']}: " + (f"{_pct(r['rate_min'])}" if r["rate_min"] == r["rate_max"] else f"{_pct(r['rate_min'])} - {_pct(r['rate_max'])}") + " p.a." for r in deposits]
        lines += ["", "**Loan rates**"] + [f"- {r['product_name']}: {_pct(r['rate_min'])} - {_pct(r['rate_max'])} p.a." for r in loans]
        return "\n".join(lines) + "\n\nAsk about a specific product for the full rate details."

    @staticmethod
    def _compare(o: dict[str, ToolOutcome], e: ExtractedEntities, q: str) -> str:
        products = o["compare_loan_products"].data["products"]
        names = [p["display_name"] for p in products]
        rows = [
            ("Interest rate", [f"{_pct(p['rate_min'])} - {_pct(p['rate_max'])}" for p in products]),
            ("Loan amount", [f"{inr(p['min_amount'])} - {inr(p['max_amount'])}" for p in products]),
            ("Tenure", [f"{format_months(p['min_tenure_months'])} - {format_months(p['max_tenure_months'])}" for p in products]),
            ("Min. monthly income", [inr(p["min_monthly_income"]) for p in products]),
            ("Min. credit score", [str(p["min_credit_score"]) for p in products]),
            ("Processing fee", [f"{p['processing_fee_pct']}% + GST" for p in products]),
            ("Prepayment", [p["prepayment_charge"] for p in products]),
            ("Best for", [p["purpose"] for p in products]),
        ]
        table = ["| Feature | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        table += [f"| {label} | " + " | ".join(values) + " |" for label, values in rows]
        return "\n".join(table)

    @staticmethod
    def _emi(o: dict[str, ToolOutcome], e: ExtractedEntities, q: str) -> str:
        d = o["calculate_emi"].data
        text = (f"For a loan of **{inr(d['principal'])}** over {format_months(d['tenure_months'])} at {_pct(d['annual_rate_pct'])} p.a.:\n\n"
                f"- Monthly EMI: **{inr(d['emi'])}**\n- Total interest: {inr(d['total_interest'])}\n- Total payment: {inr(d['total_payment'])}")
        if d["rate_source"] == "product_indicative_rate":
            text += f"\n\n_No rate was given, so I used the indicative {d['loan_name']} rate of {_pct(d['annual_rate_pct'])} p.a._"
        return text

    # ------------------------------------------------------------------ knowledge base (+ account data)
    def _account_info(self, o: dict[str, ToolOutcome], e: ExtractedEntities, q: str) -> str:
        d = o["get_account_details"].data
        lines = [f"**{d['customer']['full_name']}**, customer since {d['customer']['customer_since']}. Your accounts:"]
        for a in d["accounts"]:
            lines.append(f"- **{a['display_name']}** {a['account_number']} - {a['status']}, opened {a['opened_on']}, {a['branch']} "
                         f"(IFSC {a['ifsc']}). Minimum balance {inr(a['min_balance_required'])}, interest {_pct(a['interest_rate'])} p.a.")
        kb = self._kb_lines(o, q) if any(t in q.lower() for t in KB_TRIGGERS) else []
        if kb:
            lines += ["", "**Good to know**"] + [f"- {ln} _[{src}]_" for ln, src in kb]
        return "\n".join(lines)

    def _min_balance(self, o: dict[str, ToolOutcome], e: ExtractedEntities, q: str) -> str:
        parts: list[str] = []
        kb = self._kb_lines(o, q)
        if kb:
            parts += [f"- {ln} _[{src}]_" for ln, src in kb]
        details = o.get("get_account_details")
        if details and details.ok and not e.account_type:
            for a in details.data["accounts"]:
                need = "no minimum balance (salary credit required each month)" if a["account_type"] == "salary" else f"a minimum of {inr(a['min_balance_required'])}"
                parts.append(f"- Your {a['display_name']} ({a['account_number']}) requires {need}.")
        if not parts:
            return "I couldn't find the minimum balance rules in our knowledge base. Please contact customer support for exact details."
        return "**Minimum balance**\n" + "\n".join(parts)

    @staticmethod
    def _kb_lines(o: dict[str, ToolOutcome], query: str) -> list[tuple[str, str]]:
        outcome = o.get("search_knowledge_base")
        if not outcome or not outcome.ok or not outcome.data["grounded"]:
            return []
        from banking_ai.rag.vector_store import RetrievedChunk

        chunks = [RetrievedChunk("", "", p["title"], p["section"], "", p["lines"], p["score"]) for p in outcome.data["passages"]]
        return extract_relevant_lines(query, RetrievalResult(query, chunks, True))
