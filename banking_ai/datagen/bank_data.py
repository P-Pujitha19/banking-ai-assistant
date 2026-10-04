"""Generate completely fictional, deterministic banking data (customers, accounts, transactions...)."""
from __future__ import annotations

import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from banking_ai.datagen import catalog

FIRST_NAMES = ["Aarav", "Vivaan", "Aditya", "Arjun", "Ishaan", "Kabir", "Rohan", "Karthik", "Rahul", "Siddharth",
               "Ananya", "Diya", "Isha", "Kavya", "Meera", "Neha", "Priya", "Riya", "Sneha", "Tanvi", "Pooja",
               "Nikhil", "Divya", "Manoj"]
LAST_NAMES = ["Sharma", "Verma", "Iyer", "Nair", "Reddy", "Patel", "Gupta", "Mehta", "Rao", "Menon", "Joshi", "Kulkarni",
              "Das", "Singh", "Bose", "Chopra"]
CITIES = ["Bengaluru", "Mumbai", "Pune", "Chennai", "Hyderabad", "Delhi", "Kolkata", "Ahmedabad", "Kochi", "Jaipur"]
EMPLOYMENT = ["salaried", "salaried", "salaried", "self_employed", "business_owner"]
UPI_MERCHANTS = [("SWIGGY", "food"), ("ZOMATO", "food"), ("AMAZON PAY", "shopping"), ("BIGBASKET", "groceries"),
                 ("UBER INDIA", "travel"), ("IRCTC", "travel"), ("JIO RECHARGE", "utilities"), ("AIRTEL POSTPAID", "utilities"),
                 ("BESCOM ELECTRICITY", "utilities"), ("NETFLIX", "entertainment"), ("BOOKMYSHOW", "entertainment"),
                 ("APOLLO PHARMACY", "health"), ("DMART", "groceries"), ("FLIPKART", "shopping"), ("PETROL PUMP HP", "fuel")]
PEOPLE = ["RAHUL K", "SNEHA M", "AMIT S", "PRIYA R", "VIKRAM T", "ANITA D"]


def _round(x: float) -> float:
    return round(float(x), 2)


def generate_customers(rng: random.Random, count: int = 24) -> list[dict[str, Any]]:
    customers = []
    for i in range(count):
        cid = f"CUST{1001 + i}"
        first, last = FIRST_NAMES[i % len(FIRST_NAMES)], LAST_NAMES[(i * 5 + 3) % len(LAST_NAMES)]
        employment = EMPLOYMENT[i % len(EMPLOYMENT)]
        income = rng.choice([28000, 35000, 42000, 55000, 65000, 80000, 95000, 120000, 150000, 210000])
        customers.append({
            "customer_id": cid, "full_name": f"{first} {last}", "date_of_birth": f"{rng.randint(1968, 2001)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "city": CITIES[i % len(CITIES)], "employment_type": employment, "monthly_income": income,
            "credit_score": rng.randint(640, 820), "existing_emi": rng.choice([0, 0, 0, 4500, 8000, 12500, 20000]) if income >= 40000 else 0,
            "email": f"{first.lower()}.{last.lower()}{i}@example.com", "phone": f"+91 90000{i:05d}", "customer_since": f"{rng.randint(2012, 2024)}-{rng.randint(1, 12):02d}-15",
            "kyc_status": "verified",
        })
    # Deterministic profiles used by documentation and tests
    customers[0].update({"full_name": "Aarav Sharma", "monthly_income": 85000, "credit_score": 782, "existing_emi": 8000,
                         "date_of_birth": "1990-05-14", "employment_type": "salaried", "city": "Bengaluru"})
    customers[1].update({"full_name": "Vivaan Verma", "monthly_income": 60000, "credit_score": 745, "existing_emi": 0,
                         "date_of_birth": "1988-11-03", "employment_type": "salaried", "city": "Mumbai"})
    customers[2].update({"full_name": "Aditya Iyer", "monthly_income": 24000, "credit_score": 655, "existing_emi": 0,
                         "date_of_birth": "1999-02-21", "employment_type": "salaried", "city": "Chennai"})
    return customers


def generate_accounts(rng: random.Random, customers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accounts = []
    for i, cust in enumerate(customers):
        if i == 0 or i % 3 == 1:
            types = ["savings"]
        elif cust["employment_type"] == "salaried":
            types = ["salary"]
        else:
            types = ["savings"]
        if cust["employment_type"] == "business_owner":
            types.append("current")
        if i % 7 == 4:
            types.append("savings") if "savings" not in types else None
        for j, acct_type in enumerate(types):
            spec = catalog.ACCOUNT_TYPES[acct_type]
            number = f"5010{rng.randint(10_000_000, 99_999_999)}"
            accounts.append({
                "account_number": number, "customer_id": cust["customer_id"], "account_type": acct_type,
                "display_name": spec["display_name"], "ifsc": "FICT0000101", "branch": f"{cust['city']} Main Branch",
                "status": "active", "currency": "INR", "opened_on": cust["customer_since"], "is_primary": j == 0,
                "min_balance_required": spec["min_balance"], "interest_rate": spec["interest_rate"],
                "balance": 0.0, "available_balance": 0.0, "lien_amount": 0.0,
            })
    return accounts


def _month_start(d: date, offset: int) -> date:
    month_index = d.year * 12 + (d.month - 1) + offset
    return date(month_index // 12, month_index % 12 + 1, 1)


def generate_transactions(rng: random.Random, customers: list[dict[str, Any]], accounts: list[dict[str, Any]],
                          anchor: date) -> list[dict[str, Any]]:
    cust_by_id = {c["customer_id"]: c for c in customers}
    txns: list[dict[str, Any]] = []
    counter = 10_000_000
    start_day = anchor - timedelta(days=95)
    for acct_idx, acct in enumerate(accounts):
        cust = cust_by_id[acct["customer_id"]]
        events: list[tuple[date, str, float, str, str, str, str]] = []  # date, desc, signed amt, channel, category, merchant, kind

        def add(d: date, desc: str, amount: float, channel: str, category: str, merchant: str = "", kind: str = "") -> None:
            if start_day <= d <= anchor:
                events.append((d, desc, _round(amount), channel, category, merchant, kind))

        for offset in (-3, -2, -1, 0):
            first = _month_start(anchor, offset)
            if acct["account_type"] in ("salary", "savings") and cust["employment_type"] == "salaried" and acct["is_primary"]:
                add(first, f"SALARY CREDIT NEFT ACME TECH", cust["monthly_income"], "NEFT", "salary", "ACME TECH", "salary")
            if cust["existing_emi"] and acct["is_primary"]:
                add(first + timedelta(days=4), "EMI AUTO DEBIT LOAN A/C 4821", -cust["existing_emi"], "ECS", "emi", "LOAN EMI", "emi")
            # a couple of early-month UPI payments so "this month" is never empty
            for k in range(2):
                m, cat = rng.choice(UPI_MERCHANTS)
                add(first + timedelta(days=k), f"UPI/{m}/{rng.randint(10**8, 10**9 - 1)}", -rng.choice([149, 219, 349, 480, 799, 1250]), "UPI", cat, m, "upi")
        day = start_day
        while day <= anchor:
            if rng.random() < 0.30:
                m, cat = rng.choice(UPI_MERCHANTS)
                add(day, f"UPI/{m}/{rng.randint(10**8, 10**9 - 1)}", -rng.choice([99, 149, 249, 399, 560, 820, 1450, 2299, 3600]), "UPI", cat, m, "upi")
            if rng.random() < 0.05:
                add(day, "ATM CASH WITHDRAWAL", -rng.choice([2000, 3000, 5000, 10000]), "ATM", "cash", "ATM", "atm")
            if rng.random() < 0.04:
                add(day, f"NEFT-{rng.choice(PEOPLE)}", rng.choice([-5000, -12000, 8000, 15000, -3500]), "NEFT", "transfer", rng.choice(PEOPLE), "neft")
            day += timedelta(days=1)
        # Fees / charges (first 10 customers get a rich set so explanation queries always have material)
        if acct["is_primary"] and acct["account_type"] != "salary" or acct_idx < 12:
            add(anchor - timedelta(days=rng.randint(2, 20)), "SMS ALERT CHARGES Q3", -15.0, "CHARGES", "fee", "BANK", "fee")
            add(anchor - timedelta(days=rng.randint(2, 20)), "GST ON SMS ALERT CHARGES @18%", -2.7, "CHARGES", "fee", "BANK", "fee")
        if acct_idx % 3 == 0:
            add(anchor - timedelta(days=rng.randint(25, 60)), "DEBIT CARD ANNUAL FEE", -250.0, "CHARGES", "fee", "BANK", "fee")
            add(anchor - timedelta(days=rng.randint(25, 60)), "GST ON DEBIT CARD ANNUAL FEE @18%", -45.0, "CHARGES", "fee", "BANK", "fee")
        if acct_idx % 4 == 1:
            add(anchor - timedelta(days=rng.randint(5, 40)), "ATM WDL FEE - OTHER BANK", -21.0, "CHARGES", "fee", "BANK", "fee")
            add(anchor - timedelta(days=rng.randint(5, 40)), "GST ON ATM WDL FEE @18%", -3.78, "CHARGES", "fee", "BANK", "fee")
        if acct["account_type"] == "savings":
            add(_month_start(anchor, -1) + timedelta(days=27), "SB INTEREST CREDIT", _round(rng.uniform(80, 420)), "SYSTEM", "interest", "BANK", "interest")
        if acct_idx % 9 == 5:
            add(anchor - timedelta(days=rng.randint(10, 30)), "MIN BAL PENALTY", -300.0, "CHARGES", "fee", "BANK", "fee")
            add(anchor - timedelta(days=rng.randint(10, 30)), "GST ON MIN BAL PENALTY @18%", -54.0, "CHARGES", "fee", "BANK", "fee")

        events.sort(key=lambda e: e[0])
        cumulative, lowest = 0.0, 0.0
        for e in events:
            cumulative += e[2]
            lowest = min(lowest, cumulative)
        opening = max(0.0, -lowest) + rng.choice([6500, 12000, 21000, 38000, 52000, 74500, 118000])
        if acct_idx == 0:
            opening = max(0.0, -lowest) + 61000  # makes CUST1001 balance "healthy"
        balance = opening
        for e in events:
            balance = _round(balance + e[2])
            counter += rng.randint(1, 7)
            when = datetime.combine(e[0], datetime.min.time()) + timedelta(hours=rng.randint(7, 21), minutes=rng.randint(0, 59))
            txns.append({
                "transaction_id": f"TXN{counter}", "account_number": acct["account_number"], "customer_id": acct["customer_id"],
                "date": e[0].isoformat(), "time": when.strftime("%H:%M"), "description": e[1], "amount": abs(e[2]),
                "type": "credit" if e[2] > 0 else "debit", "channel": e[3], "category": e[4], "merchant": e[5],
                "balance_after": balance, "reference": f"REF{rng.randint(10**9, 10**10 - 1)}",
            })
        acct["balance"] = _round(balance)
        lien = rng.choice([0.0, 0.0, 0.0, 500.0, 1500.0])
        acct["lien_amount"] = lien
        acct["available_balance"] = _round(balance - lien)
    return txns


def build_all(anchor: date, seed: int = 42) -> dict[str, Any]:
    rng = random.Random(seed)
    customers = generate_customers(rng)
    accounts = generate_accounts(rng, customers)
    txns = generate_transactions(rng, customers, accounts, anchor)
    return {
        "customers": customers, "accounts": accounts, "transactions": txns,
        "loans": [{"loan_type": k, **v} for k, v in catalog.LOAN_PRODUCTS.items()],
        "interest_rates": catalog.INTEREST_RATES,
        "charges": {"items": catalog.CHARGES, "gst_rate_pct": catalog.GST_RATE_PCT,
                    "account_types": catalog.ACCOUNT_TYPES},
        "charge_catalog": catalog.CHARGE_CATALOG,
    }


def write_all(out_dir: Path, anchor: date, seed: int = 42) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = build_all(anchor, seed)
    for name, payload in data.items():
        (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {k: (len(v) if isinstance(v, list) else len(v.get("items", []))) for k, v in data.items()}
