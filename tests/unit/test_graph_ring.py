import pytest
import time
from src.streaming.graph import FraudRingGraphEngine
from src.streaming.features import StreamFeatureStateManager
from src.streaming.schemas import RawTransaction
from src.inference.predictor import FraudPredictor


def test_bipartite_graph_ring_detection():
    engine = FraudRingGraphEngine(ttl_seconds=86400.0, ring_threshold=3)
    t0 = 10000.0

    # Card 1 on Device A
    dev_cnt, card_cnt, cluster_sz, is_ring = engine.record_interaction("card_01", "dev_A", t0)
    assert dev_cnt == 1
    assert card_cnt == 1
    assert cluster_sz == 2  # card_01 + dev_A
    assert not is_ring

    # Card 2 on Device A (Same device, 2nd card)
    dev_cnt, card_cnt, cluster_sz, is_ring = engine.record_interaction("card_02", "dev_A", t0 + 100)
    assert dev_cnt == 2
    assert cluster_sz == 3  # card_01, card_02, dev_A
    assert not is_ring

    # Card 3 on Device A (Same device, 3rd card -> Fraud Ring Triggered!)
    dev_cnt, card_cnt, cluster_sz, is_ring = engine.record_interaction("card_03", "dev_A", t0 + 200)
    assert dev_cnt == 3
    assert cluster_sz == 4
    assert is_ring

    # Verify detected fraud rings report
    rings = engine.get_detected_fraud_rings()
    assert len(rings) == 1
    assert rings[0]["anchor_device"] == "dev_A"
    assert rings[0]["card_count"] == 3
    assert "card_01" in rings[0]["cards"]
    assert "card_02" in rings[0]["cards"]
    assert "card_03" in rings[0]["cards"]


def test_graph_ttl_pruning():
    engine = FraudRingGraphEngine(ttl_seconds=3600.0, ring_threshold=3)
    t0 = 1000.0

    engine.record_interaction("card_01", "dev_A", t0)
    engine.record_interaction("card_02", "dev_A", t0 + 100)
    assert len(engine.devices_to_cards["dev_A"]) == 2

    # Prune with timestamp 2 hours later (7200s > 3600s TTL)
    engine.prune_stale_edges(t0 + 7200.0)
    assert len(engine.devices_to_cards["dev_A"]) == 0


def test_feature_manager_graph_enrichment():
    feature_mgr = StreamFeatureStateManager(ring_threshold=3)
    t0 = time.time()
    shared_device = "dev_botnet_cluster_01"

    # Simulate 3 different users with 3 different cards on 1 shared device
    raw_txs = [
        RawTransaction(
            transaction_id=f"tx_ring_{i}",
            user_id=f"usr_victim_{i}",
            card_id=f"card_stolen_{i}",
            amount=500.0 + i * 50,
            currency="USD",
            merchant_id="m_crypto",
            merchant_category="crypto",
            latitude=40.7128,
            longitude=-74.0060,
            timestamp=t0 + (i * 30),
            device_id=shared_device,
        )
        for i in range(3)
    ]

    en1 = feature_mgr.enrich(raw_txs[0])
    assert en1.device_card_count == 1
    assert not en1.is_fraud_ring_suspected

    en2 = feature_mgr.enrich(raw_txs[1])
    assert en2.device_card_count == 2
    assert not en2.is_fraud_ring_suspected

    en3 = feature_mgr.enrich(raw_txs[2])
    assert en3.device_card_count == 3
    assert en3.is_fraud_ring_suspected  # Triggered!


def test_predictor_fraud_ring_policy_override():
    feature_mgr = StreamFeatureStateManager(ring_threshold=3)
    predictor = FraudPredictor()
    t0 = time.time()
    shared_device = "dev_syndicate_botnet"

    # 3rd transaction on shared device
    for i in range(3):
        raw = RawTransaction(
            transaction_id=f"tx_bot_{i}",
            user_id=f"usr_bot_{i}",
            card_id=f"card_bot_{i}",
            amount=320.0,
            currency="USD",
            merchant_id="m10",
            merchant_category="electronics",
            latitude=40.7128,
            longitude=-74.0060,
            timestamp=t0 + (i * 10),
            device_id=shared_device,
        )
        enriched = feature_mgr.enrich(raw)

    # Score the 3rd enriched transaction
    prediction = predictor.predict(enriched)
    assert prediction.prediction == "FRAUD_SUSPECTED"
    assert prediction.action == "BLOCK_AND_FLAG"
    assert prediction.fraud_probability >= 0.90
    assert any("device_sharing_fraud_ring" in f for f in prediction.top_risk_factors)
