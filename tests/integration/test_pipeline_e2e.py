import pytest
import time
from src.simulator.profiles import UserProfile
from src.streaming.features import StreamFeatureStateManager
from src.inference.predictor import FraudPredictor
from src.analytics.store import DuckDBAnalyticsStore


def test_full_pipeline_end_to_end(tmp_path):
    db_file = str(tmp_path / "e2e_analytics.duckdb")
    store = DuckDBAnalyticsStore(db_path=db_file)
    feature_mgr = StreamFeatureStateManager()
    predictor = FraudPredictor()

    user = UserProfile("usr_e2e_01")
    t0 = time.time() - 600.0

    # 1. Simulate 3 consecutive legitimate transactions
    scored_results = []
    for i in range(3):
        tx_time = t0 + (i * 120.0)
        raw_tx = user.generate_legitimate_transaction(tx_time)
        store.insert_raw_transaction(raw_tx)

        enriched = feature_mgr.enrich(raw_tx)
        pred = predictor.predict(enriched)
        store.insert_scored_transaction(enriched, pred)
        scored_results.append(pred)

    # Legitimate transactions should be allowed
    for p in scored_results:
        assert p.prediction == "LEGITIMATE"
        assert p.action == "ALLOW"

    # 2. Inject an impossible travel fraud attack
    fraud_time = t0 + 400.0
    fraud_tx = user.generate_fraud_impossible_travel(fraud_time)
    store.insert_raw_transaction(fraud_tx)

    enriched_fraud = feature_mgr.enrich(fraud_tx)
    assert enriched_fraud.geo_distance_velocity_kmh > 750.0  # Impossible travel detected

    pred_fraud = predictor.predict(enriched_fraud)
    store.insert_scored_transaction(enriched_fraud, pred_fraud)

    assert pred_fraud.prediction == "FRAUD_SUSPECTED"
    assert pred_fraud.action == "BLOCK_AND_FLAG"
    assert any("geo_distance_velocity_kmh" in f or "impossible travel" in f for f in pred_fraud.top_risk_factors)

    # 3. Verify DuckDB OLAP Aggregates
    summary = store.get_rolling_summary(window_seconds=3600)
    assert summary.total_transactions == 4
    assert summary.flagged_fraudulent == 1
    assert summary.fraud_rate_percentage == 25.0
    assert summary.prevented_capital_loss_usd == fraud_tx.amount

    # 4. Verify Alert Generation and Dispute Triage
    alerts = store.get_recent_alerts(limit=10)
    assert len(alerts) == 1
    assert alerts[0]["transaction_id"] == fraud_tx.transaction_id
    assert alerts[0]["status"] == "UNREVIEWED"

    # Resolve dispute
    store.update_alert_status(alerts[0]["alert_id"], "CONFIRMED_FRAUD")
    updated_alerts = store.get_recent_alerts(limit=10)
    assert updated_alerts[0]["status"] == "CONFIRMED_FRAUD"
