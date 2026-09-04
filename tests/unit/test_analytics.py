import os
import pytest
import time
from src.analytics.store import DuckDBAnalyticsStore
from src.streaming.schemas import RawTransaction, EnrichedTransaction, FraudPredictionResponse


@pytest.fixture
def temp_store(tmp_path):
    db_file = str(tmp_path / "test_analytics.duckdb")
    store = DuckDBAnalyticsStore(db_path=db_file)
    try:
        yield store
    finally:
        store.close()


def test_duckdb_raw_insertion(temp_store):
    tx = RawTransaction(
        transaction_id="tx_test_01",
        user_id="usr_test",
        card_id="card_test",
        amount=125.50,
        currency="USD",
        merchant_id="m100",
        merchant_category="electronics",
        latitude=40.7128,
        longitude=-74.0060,
        timestamp=time.time(),
    )
    temp_store.insert_raw_transaction(tx)

    conn = temp_store._get_connection()
    count = conn.execute("SELECT COUNT(*) FROM transactions_raw WHERE transaction_id = 'tx_test_01'").fetchone()[0]
    conn.close()
    assert count == 1


def test_duckdb_scored_insertion_and_alerts(temp_store):
    now = time.time()
    enriched = EnrichedTransaction(
        transaction_id="tx_fraud_01",
        user_id="usr_fraud",
        card_id="card_fraud",
        amount=3500.0,
        currency="USD",
        merchant_id="m999",
        merchant_category="crypto",
        latitude=40.7128,
        longitude=-74.0060,
        timestamp=now,
        tx_count_last_10m=9,
        tx_count_last_1h=12,
        amount_rolling_avg_1h=80.0,
        amount_deviation_from_avg=6.5,
        geo_distance_velocity_kmh=1200.0,
        is_night_transaction=1,
        category_risk_index=0.95,
    )

    pred = FraudPredictionResponse(
        transaction_id="tx_fraud_01",
        user_id="usr_fraud",
        fraud_probability=0.965,
        prediction="FRAUD_SUSPECTED",
        action="BLOCK_AND_FLAG",
        latency_ms=8.5,
        top_risk_factors=["amount_deviation_from_avg", "impossible_travel"],
        timestamp=now,
    )

    temp_store.insert_scored_transaction(enriched, pred)

    # Verify summary
    summary = temp_store.get_rolling_summary(window_seconds=3600)
    assert summary.total_transactions == 1
    assert summary.flagged_fraudulent == 1
    assert summary.fraud_rate_percentage == 100.0
    assert summary.prevented_capital_loss_usd == 3500.0

    # Verify alert table registration
    alerts = temp_store.get_recent_alerts(limit=10)
    assert len(alerts) == 1
    assert alerts[0]["transaction_id"] == "tx_fraud_01"
    assert alerts[0]["status"] == "UNREVIEWED"

    # Test resolving alert
    temp_store.update_alert_status(alerts[0]["alert_id"], "CONFIRMED_FRAUD")
    updated_alerts = temp_store.get_recent_alerts(limit=10)
    assert updated_alerts[0]["status"] == "CONFIRMED_FRAUD"
