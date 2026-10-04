"""Static product catalogue for the *fictional* bank (single source of truth for synthetic data)."""
from __future__ import annotations

ACCOUNT_TYPES = {
    "savings": {
        "display_name": "Savings Account",
        "min_balance": 5000, "min_balance_basis": "average monthly balance",
        "penalty": 300, "interest_rate": 3.5,
        "atm_free_own": 5, "atm_free_other": 3,
    },
    "salary": {
        "display_name": "Salary Account",
        "min_balance": 0, "min_balance_basis": "zero balance while a monthly salary credit is received",
        "penalty": 0, "interest_rate": 3.5,
        "atm_free_own": 8, "atm_free_other": 3,
    },
    "current": {
        "display_name": "Current Account",
        "min_balance": 25000, "min_balance_basis": "average quarterly balance",
        "penalty": 750, "interest_rate": 0.0,
        "atm_free_own": 10, "atm_free_other": 3,
    },
}

LOAN_PRODUCTS = {
    "personal_loan": {
        "display_name": "Personal Loan", "rate_min": 10.99, "rate_max": 16.5, "indicative_rate": 11.49,
        "min_amount": 50000, "max_amount": 2500000, "min_tenure_months": 12, "max_tenure_months": 72,
        "default_tenure_months": 60, "min_monthly_income": 25000, "min_credit_score": 700,
        "min_age": 21, "max_age": 60, "foir_limit": 0.50, "processing_fee_pct": 1.5,
        "prepayment_charge": "2% of outstanding principal within 12 months; nil afterwards",
        "purpose": "Unsecured loan for medical, travel, wedding or other personal needs",
    },
    "home_loan": {
        "display_name": "Home Loan", "rate_min": 8.5, "rate_max": 9.75, "indicative_rate": 8.65,
        "min_amount": 500000, "max_amount": 50000000, "min_tenure_months": 60, "max_tenure_months": 360,
        "default_tenure_months": 240, "min_monthly_income": 40000, "min_credit_score": 725,
        "min_age": 21, "max_age": 65, "foir_limit": 0.55, "processing_fee_pct": 0.5,
        "prepayment_charge": "Nil on floating-rate home loans",
        "purpose": "Secured loan to buy, construct or renovate a house; financing up to 80% of property value",
    },
    "car_loan": {
        "display_name": "Car Loan", "rate_min": 8.75, "rate_max": 11.5, "indicative_rate": 9.25,
        "min_amount": 100000, "max_amount": 5000000, "min_tenure_months": 12, "max_tenure_months": 84,
        "default_tenure_months": 60, "min_monthly_income": 30000, "min_credit_score": 700,
        "min_age": 21, "max_age": 65, "foir_limit": 0.50, "processing_fee_pct": 1.0,
        "prepayment_charge": "3% of outstanding principal within 12 months; nil afterwards",
        "purpose": "Secured loan to buy a new or pre-owned car; financing up to 90% of on-road price",
    },
    "education_loan": {
        "display_name": "Education Loan", "rate_min": 8.9, "rate_max": 12.0, "indicative_rate": 9.75,
        "min_amount": 50000, "max_amount": 7500000, "min_tenure_months": 12, "max_tenure_months": 180,
        "default_tenure_months": 120, "min_monthly_income": 20000, "min_credit_score": 680,
        "min_age": 18, "max_age": 60, "foir_limit": 0.50, "processing_fee_pct": 0.5,
        "prepayment_charge": "Nil",
        "purpose": "Loan for tuition and living costs of higher education in India or abroad (co-applicant income considered)",
    },
}

INTEREST_RATES = [
    {"product_type": "savings_account", "product_name": "Savings Account", "category": "deposit",
     "rate_min": 3.5, "rate_max": 4.0,
     "details": ["3.50% p.a. on balances up to ₹10,00,000", "4.00% p.a. on balances above ₹10,00,000"]},
    {"product_type": "salary_account", "product_name": "Salary Account", "category": "deposit",
     "rate_min": 3.5, "rate_max": 3.5, "details": ["3.50% p.a. on all balances"]},
    {"product_type": "current_account", "product_name": "Current Account", "category": "deposit",
     "rate_min": 0.0, "rate_max": 0.0, "details": ["No interest is paid on current accounts"]},
    {"product_type": "fixed_deposit", "product_name": "Fixed Deposit", "category": "deposit",
     "rate_min": 3.5, "rate_max": 7.0,
     "details": ["7 to 45 days: 3.50% p.a.", "46 to 180 days: 4.75% p.a.", "181 to 364 days: 5.75% p.a.",
                 "1 year to less than 2 years: 6.75% p.a.", "2 years to less than 3 years: 7.00% p.a.",
                 "3 years to 5 years: 6.75% p.a.", "Senior citizens get an additional 0.50% p.a."]},
    {"product_type": "recurring_deposit", "product_name": "Recurring Deposit", "category": "deposit",
     "rate_min": 6.5, "rate_max": 6.5, "details": ["6.50% p.a. for tenures of 6 months to 10 years"]},
] + [
    {"product_type": key, "product_name": p["display_name"], "category": "loan",
     "rate_min": p["rate_min"], "rate_max": p["rate_max"],
     "details": [f"{p['rate_min']:.2f}% to {p['rate_max']:.2f}% p.a.; indicative rate {p['indicative_rate']:.2f}% p.a. "
                 "(final rate depends on credit score and loan amount)"]}
    for key, p in LOAN_PRODUCTS.items()
]

CHARGES = [
    {"code": "SMS_ALERT", "name": "SMS alert charges", "amount": 15.0, "frequency": "per quarter", "gst_applicable": True},
    {"code": "DEBIT_CARD_ANNUAL", "name": "Debit card annual fee", "amount": 250.0, "frequency": "per year", "gst_applicable": True},
    {"code": "ATM_OTHER_BANK", "name": "ATM withdrawal beyond free limit", "amount": 21.0, "frequency": "per transaction", "gst_applicable": True},
    {"code": "CHEQUE_BOOK", "name": "Cheque book (after 25 free leaves per year)", "amount": 4.0, "frequency": "per leaf", "gst_applicable": True},
    {"code": "IMPS_SMALL", "name": "IMPS up to ₹1,000", "amount": 0.0, "frequency": "per transaction", "gst_applicable": False},
    {"code": "IMPS_MID", "name": "IMPS above ₹1,000 and up to ₹1,00,000", "amount": 5.0, "frequency": "per transaction", "gst_applicable": True},
    {"code": "IMPS_HIGH", "name": "IMPS above ₹1,00,000 and up to ₹5,00,000", "amount": 15.0, "frequency": "per transaction", "gst_applicable": True},
    {"code": "NEFT_ONLINE", "name": "NEFT / RTGS / UPI (online)", "amount": 0.0, "frequency": "per transaction", "gst_applicable": False},
    {"code": "DUPLICATE_STATEMENT", "name": "Duplicate physical statement", "amount": 100.0, "frequency": "per request", "gst_applicable": True},
]
GST_RATE_PCT = 18

CHARGE_CATALOG = [
    {"match": "SMS ALERT", "label": "SMS alert service charge", "is_charge": True,
     "explanation": "A quarterly charge of ₹15 for SMS transaction alerts, plus 18% GST on the charge."},
    {"match": "GST ON", "label": "GST on bank charges", "is_charge": True,
     "explanation": "18% GST that is applied on bank service charges (such as SMS alerts, card fees and penalties)."},
    {"match": "DEBIT CARD ANNUAL", "label": "Debit card annual fee", "is_charge": True,
     "explanation": "The yearly maintenance fee of ₹250 (plus GST) for your debit card."},
    {"match": "ATM WDL FEE", "label": "ATM withdrawal fee", "is_charge": True,
     "explanation": "Charged when you exceed the free ATM transactions in a month: ₹21 plus GST per transaction."},
    {"match": "MIN BAL PENALTY", "label": "Minimum balance penalty", "is_charge": True,
     "explanation": "Charged when the average balance falls below the minimum required for the account type (₹300 for a savings account)."},
    {"match": "EMI AUTO DEBIT", "label": "Loan EMI auto-debit", "is_charge": False,
     "explanation": "The scheduled monthly instalment (EMI) for your loan, auto-debited on the due date."},
    {"match": "SB INTEREST", "label": "Savings interest credit", "is_charge": False,
     "explanation": "Interest paid by the bank on your savings balance."},
    {"match": "SALARY", "label": "Salary credit", "is_charge": False,
     "explanation": "Monthly salary credited by your employer."},
]
