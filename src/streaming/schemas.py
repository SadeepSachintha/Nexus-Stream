from typing import List, Optional
from pydantic import BaseModel, Field
import time


class RawTransaction(BaseModel):
    """Raw financial transaction payload emitted by payment gateways or simulators."""
    transaction_id: str = Field(..., description="Unique transaction ID")
    user_id: str = Field(..., description="Unique account/user ID")
    card_id: str = Field(..., description="Masked card identifier")
    amount: float = Field(..., gt=0, description="Transaction monetary value")
    currency: str = Field(default="USD", description="Currency code (e.g. USD, EUR)")
    merchant_id: str = Field(..., description="Merchant identifier")
    merchant_category: str = Field(..., description="Category code (e.g., electronics, travel, grocery)")
    latitude: float = Field(..., description="Geographical latitude of origin")
    longitude: float = Field(..., description="Geographical longitude of origin")
    timestamp: float = Field(default_factory=time.time, description="Epoch timestamp in seconds")
    device_id: Optional[str] = Field(default=None, description="Originating client device identifier")
    is_fraud_ground_truth: Optional[bool] = Field(default=None, description="Ground truth indicator for testing")


class EnrichedTransaction(RawTransaction):
    """Enriched transaction payload with stateful sliding window behavioral features."""
    tx_count_last_10m: int = Field(default=1, description="Number of transactions by this card in the last 10 minutes")
    tx_count_last_1h: int = Field(default=1, description="Number of transactions by this card in the last 1 hour")
    amount_rolling_avg_1h: float = Field(default=0.0, description="Rolling average transaction amount over last 1 hour")
    amount_deviation_from_avg: float = Field(default=0.0, description="Z-score deviation from rolling average amount")
    geo_distance_velocity_kmh: float = Field(default=0.0, description="Haversine velocity from previous transaction in km/h")
    is_night_transaction: int = Field(default=0, description="1 if transaction occurred between 00:00 and 06:00 local time, else 0")
    category_risk_index: float = Field(default=0.1, description="Baseline risk weighting for merchant category")


class FraudPredictionResponse(BaseModel):
    """Response returned by the low-latency inference endpoint."""
    transaction_id: str
    user_id: str
    fraud_probability: float = Field(..., description="Calibrated risk score between 0.0 and 1.0")
    prediction: str = Field(..., description="LEGITIMATE | REVIEW_REQUIRED | FRAUD_SUSPECTED")
    action: str = Field(..., description="ALLOW | STEP_UP_AUTH | BLOCK_AND_FLAG")
    latency_ms: float = Field(..., description="End-to-end inference compute latency in milliseconds")
    top_risk_factors: List[str] = Field(default_factory=list, description="List of primary driving risk factors")
    timestamp: float = Field(default_factory=time.time)


class AnalyticsSummaryResponse(BaseModel):
    """Rolling analytical aggregate metrics served from DuckDB."""
    window: str = "rolling_1h"
    total_transactions: int
    flagged_fraudulent: int
    review_required: int
    fraud_rate_percentage: float
    avg_processing_latency_ms: float
    prevented_capital_loss_usd: float
