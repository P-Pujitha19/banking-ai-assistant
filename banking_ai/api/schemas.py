"""Pydantic request/response models for both APIs."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

LoanType = Literal["personal_loan", "home_loan", "car_loan", "education_loan"]


class EligibilityRequest(BaseModel):
    customer_id: str = Field(pattern=r"^CUST\d{4}$", description="Customer id, e.g. CUST1001")
    loan_type: LoanType
    loan_amount: Optional[float] = Field(default=None, gt=0, le=1_000_000_000)
    tenure_months: Optional[int] = Field(default=None, ge=1, le=480)
    declared_monthly_income: Optional[float] = Field(default=None, ge=0, le=100_000_000)


class EmiRequest(BaseModel):
    principal: float = Field(gt=0, le=1_000_000_000)
    annual_rate: float = Field(ge=0, le=50)
    tenure_months: int = Field(ge=1, le=480)


class ChatRequest(BaseModel):
    customer_id: str = Field(pattern=r"^CUST\d{4}$", description="Authenticated customer (demo: selected in the UI)")
    message: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = Field(default=None, max_length=64)
    debug: bool = Field(default=False, description="Include the developer execution summary (demo mode only)")


class DebugInfo(BaseModel):
    request_id: str
    intent: str
    confidence: float
    intent_backend: str
    entities: dict[str, Any]
    tools: list[dict[str, Any]]
    sources: list[str]
    rag_documents: list[dict[str, Any]]
    response_source: str
    grounding_ok: Optional[bool] = None
    notes: list[str]
    latency_ms: float
    mode: str


class ChatResponse(BaseModel):
    session_id: str
    request_id: str
    response: str
    status: str
    latency_ms: float
    debug: Optional[DebugInfo] = None
