from typing import List, Optional, Dict, Any
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
    device_card_count: int = Field(default=1, description="Number of distinct cards observed on this device in the last 24h")
    card_device_count: int = Field(default=1, description="Number of distinct devices used by this card in the last 24h")
    ring_cluster_size: int = Field(default=1, description="Total connected entities in the bipartite fraud ring cluster")
    is_fraud_ring_suspected: bool = Field(default=False, description="True if device sharing indicates a coordinated fraud ring")


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


class DeadLetterEvent(BaseModel):
    """Dead-letter event capturing poison pills and malformed transactions for audit and replay."""
    dlq_id: str = Field(..., description="Unique dead-letter record identifier")
    source_topic: str = Field(default="transactions.raw", description="Originating event stream")
    error_type: str = Field(..., description="MALFORMED_JSON | SCHEMA_VALIDATION_ERROR | PROCESSING_FAILURE")
    error_message: str = Field(..., description="Detailed diagnostic error message or trace")
    raw_payload: str = Field(..., description="Raw string payload that failed execution")
    retry_count: int = Field(default=0, description="Number of times replay has been attempted")
    status: str = Field(default="PENDING", description="PENDING | REPLAYED | DISCARDED")
    timestamp: float = Field(default_factory=time.time)


class DLQSummaryResponse(BaseModel):
    """Telemetry metrics on captured poison pills."""
    total_dlq_events: int
    pending_events: int
    replayed_events: int
    discarded_events: int


class FeatureDriftMetric(BaseModel):
    """Statistical drift analysis for an individual numerical feature."""
    feature_name: str
    psi_score: float = Field(..., description="Population Stability Index (<0.1 stable, >=0.2 significant drift)")
    ks_statistic: float = Field(..., description="Two-sample Kolmogorov-Smirnov test statistic")
    ks_p_value: float = Field(..., description="p-value for KS test")
    wasserstein_distance: float = Field(..., description="Earth Mover's Distance between distributions")
    drift_detected: bool = Field(..., description="True if PSI >= 0.20 or (KS stat >= 0.15 and p_val < 0.05)")
    status: str = Field(..., description="STABLE | MODERATE_SHIFT | CRITICAL_DRIFT")


class DriftReportResponse(BaseModel):
    """Overall dataset and feature-level concept drift telemetry."""
    dataset_drift_detected: bool
    drifted_feature_count: int
    total_features: int
    drift_share: float = Field(..., description="Percentage of monitored features exhibiting significant drift")
    features: Dict[str, FeatureDriftMetric]
    sample_size: int = Field(..., description="Number of production transactions analyzed")
    timestamp: float = Field(default_factory=time.time)


