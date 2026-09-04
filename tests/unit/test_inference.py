import pytest
import time
from fastapi.testclient import TestClient
from src.streaming.schemas import EnrichedTransaction, RawTransaction
from src.inference.predictor import FraudPredictor
from src.inference.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_predictor_scoring_and_latency():
    predictor = FraudPredictor()
    
    # 1. Normal/Legitimate transaction
    normal_tx = EnrichedTransaction(
        transaction_id="tx_norm_10",
        user_id="usr_norm",
        card_id="card_norm",
        amount=42.50,
        currency="USD",
        merchant_id="m10",
        merchant_category="grocery",
        latitude=40.7128,
        longitude=-74.0060,
        timestamp=time.time(),
        tx_count_last_10m=1,
        tx_count_last_1h=2,
        amount_rolling_avg_1h=45.0,
        amount_deviation_from_avg=0.1,
        geo_distance_velocity_kmh=15.0,
        is_night_transaction=0,
        category_risk_index=0.08,
    )
    res_norm = predictor.predict(normal_tx)
    assert res_norm.prediction == "LEGITIMATE"
    assert res_norm.action == "ALLOW"
    assert res_norm.fraud_probability < 0.40
    assert res_norm.latency_ms < 15.0, f"Inference took too long: {res_norm.latency_ms}ms"

    # 2. Fraudulent transaction (severe impossible travel + spike)
    fraud_tx = EnrichedTransaction(
        transaction_id="tx_fraud_20",
        user_id="usr_victim",
        card_id="card_victim",
        amount=5200.0,
        currency="USD",
        merchant_id="m99",
        merchant_category="crypto",
        latitude=35.6762,
        longitude=139.6503,
        timestamp=time.time(),
        tx_count_last_10m=8,
        tx_count_last_1h=10,
        amount_rolling_avg_1h=50.0,
        amount_deviation_from_avg=8.5,
        geo_distance_velocity_kmh=1500.0,
        is_night_transaction=1,
        category_risk_index=0.95,
    )
    res_fraud = predictor.predict(fraud_tx)
    assert res_fraud.prediction == "FRAUD_SUSPECTED"
    assert res_fraud.action == "BLOCK_AND_FLAG"
    assert res_fraud.fraud_probability >= 0.75
    assert len(res_fraud.top_risk_factors) > 0


def test_api_health_endpoint(client):
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "uptime_seconds" in data


def test_api_predict_endpoint(client):
    payload = {
        "transaction_id": "tx_api_01",
        "user_id": "usr_api",
        "card_id": "card_api",
        "amount": 2500.00,
        "currency": "USD",
        "merchant_id": "merch_lux",
        "merchant_category": "jewelry",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "timestamp": time.time(),
        "tx_count_last_10m": 6,
        "tx_count_last_1h": 8,
        "amount_rolling_avg_1h": 40.0,
        "amount_deviation_from_avg": 5.2,
        "geo_distance_velocity_kmh": 900.0,
        "is_night_transaction": 1,
        "category_risk_index": 0.85,
    }
    response = client.post("/v1/predict-fraud", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "tx_api_01"
    assert data["prediction"] in ("FRAUD_SUSPECTED", "REVIEW_REQUIRED")
    assert "latency_ms" in data


def test_api_analytics_and_metrics(client):
    summary_resp = client.get("/v1/analytics/summary")
    assert summary_resp.status_code == 200
    sdata = summary_resp.json()
    assert "total_transactions" in sdata
    assert "fraud_rate_percentage" in sdata

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert "nexus_transactions_total" in metrics_resp.text
