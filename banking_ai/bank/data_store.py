"""Loads the synthetic banking JSON files into memory."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional


@dataclass
class BankData:
    """In-memory view of the synthetic bank database."""

    customers: dict[str, dict[str, Any]]
    accounts_by_customer: dict[str, list[dict[str, Any]]]
    transactions: list[dict[str, Any]]
    loans: dict[str, dict[str, Any]]
    interest_rates: list[dict[str, Any]]
    charges: dict[str, Any]
    charge_catalog: list[dict[str, Any]]
    as_of: date
    txn_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)


def _read(data_dir: Path, name: str) -> Any:
    path = data_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Generate the synthetic data first: python scripts/generate_data.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_bank_data(data_dir: Path, as_of: Optional[str] = None) -> BankData:
    """Load all bank JSON files. ``as_of`` defaults to the latest transaction date (the bank's 'today')."""
    customers = {c["customer_id"]: c for c in _read(data_dir, "customers")}
    accounts_by_customer: dict[str, list[dict[str, Any]]] = {}
    for account in _read(data_dir, "accounts"):
        accounts_by_customer.setdefault(account["customer_id"], []).append(account)
    transactions = _read(data_dir, "transactions")
    as_of_date = date.fromisoformat(as_of) if as_of else max(date.fromisoformat(t["date"]) for t in transactions)
    return BankData(
        customers=customers,
        accounts_by_customer=accounts_by_customer,
        transactions=transactions,
        loans={l["loan_type"]: l for l in _read(data_dir, "loans")},
        interest_rates=_read(data_dir, "interest_rates"),
        charges=_read(data_dir, "charges"),
        charge_catalog=_read(data_dir, "charge_catalog"),
        as_of=as_of_date,
        txn_by_id={t["transaction_id"]: t for t in transactions},
    )
