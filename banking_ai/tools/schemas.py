"""Pydantic argument schemas for the banking tools (also the JSON schemas the LLM sees when tool-calling).

Note: there is deliberately NO ``customer_id`` argument. The authenticated customer is injected by the system,
so a manipulated prompt can never make a tool read another customer's data.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class NoArgs(BaseModel):
    """Tool without arguments."""


class BalanceArgs(BaseModel):
    account_last4: Optional[str] = Field(default=None, description="Last 4 digits of one account; omit for all accounts")


class TransactionHistoryArgs(BaseModel):
    limit: int = Field(default=5, ge=1, le=50, description="How many transactions to return (1-50)")
    period: Optional[str] = Field(default=None, description="this_month, last_month, today, yesterday, this_week, last_week, last_N_days, month_N")
    start_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    transaction_type: Optional[str] = Field(default=None, description="credit or debit")
    channel: Optional[str] = Field(default=None, description="UPI, ATM, NEFT, IMPS, ECS, CHARGES")
    account_last4: Optional[str] = Field(default=None, description="Last 4 digits of the account")


class RecentTransactionArgs(BaseModel):
    transaction_type: Optional[str] = Field(default=None, description="credit or debit; omit for any")


class LoanProductsArgs(BaseModel):
    loan_type: Optional[str] = Field(default=None, description="personal_loan, home_loan, car_loan, education_loan; omit for all")


class EligibilityArgs(BaseModel):
    loan_type: str = Field(description="personal_loan, home_loan, car_loan or education_loan")
    loan_amount: Optional[float] = Field(default=None, gt=0, description="Requested amount in INR")
    tenure_months: Optional[int] = Field(default=None, ge=1, le=480, description="Tenure in months")
    monthly_income: Optional[float] = Field(default=None, ge=0, description="Income stated by the customer (informational)")


class InterestRatesArgs(BaseModel):
    product_type: Optional[str] = Field(default=None, description="savings_account, salary_account, current_account, fixed_deposit, recurring_deposit, personal_loan, home_loan, car_loan, education_loan")


class EmiArgs(BaseModel):
    principal: float = Field(gt=0, description="Loan amount in INR")
    tenure_months: int = Field(ge=1, le=480, description="Tenure in months")
    annual_rate: Optional[float] = Field(default=None, ge=0, le=50, description="Annual interest rate in percent; omit to use the product's indicative rate")
    loan_type: Optional[str] = Field(default=None, description="Used to pick the indicative rate when annual_rate is omitted")


class ExplainTransactionArgs(BaseModel):
    transaction_id: Optional[str] = Field(default=None, description="e.g. TXN10000221; omit to explain the most recent charge")


class CompareLoansArgs(BaseModel):
    loan_types: list[str] = Field(default_factory=list, description="Two or more loan types to compare")


class KnowledgeSearchArgs(BaseModel):
    query: str = Field(description="The customer's question, rephrased as a search query")
