"""Request / response schemas. Field names and order follow the model's 24 input features."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

NonNeg = Field(ge=0)


class Transaction(BaseModel):
    """One card transaction, as in the dataset (Credit Card Fraud Detection 2026, CC0)."""

    model_config = ConfigDict(extra="forbid")

    merchant_category: str
    card_type: str
    auth_method: str
    channel: str
    device_type: str
    is_foreign_transaction: bool
    is_new_merchant: bool
    used_vpn: bool
    ip_country_mismatch: bool
    billing_shipping_mismatch: bool
    is_ai_generated_scam_attempt: bool
    amount_usd: float = Field(gt=0, description="Transaction amount, USD")
    hours_since_last_txn: float = NonNeg
    distance_from_home_km: float = NonNeg
    card_age_months: float = NonNeg
    customer_age: float = Field(ge=0, le=120)
    account_balance_usd: float = NonNeg
    velocity_score: float = NonNeg
    merchant_risk_score: float = Field(ge=0, le=100)
    txn_count_last_24h: float = NonNeg
    cvv_retry_count: float = NonNeg
    prior_disputes: float = NonNeg
    time_of_day_hour: float = Field(ge=0, le=23)
    day_of_week: float = Field(ge=0, le=6)


class Contribution(BaseModel):
    feature: str
    value: str | float | bool
    contribution: float = Field(description="coefficient x model input, in log-odds")


class Prediction(BaseModel):
    score: float = Field(description="P(fraud) from the logistic model, 0..1")
    decision: Literal["approve", "step-up"]
    threshold: float
    threshold_basis: str
    logit: float
    top_contributions: list[Contribution] = Field(description="3 largest |contribution|")
    model: str = Field(description="pipeline spec id")


class Health(BaseModel):
    status: Literal["ok"]
    model: str
    n_features: int
    threshold: float
