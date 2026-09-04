import pytest
import time
from src.streaming.schemas import RawTransaction
from src.streaming.features import (
    haversine_distance_km,
    calculate_travel_velocity_kmh,
    get_category_risk,
    is_night_transaction,
    CardSlidingWindowState,
    StreamFeatureStateManager,
)


def test_haversine_distance():
    # NYC (40.7128, -74.0060) to London (51.5074, -0.1278) ~5570 km
    dist = haversine_distance_km(40.7128, -74.0060, 51.5074, -0.1278)
    assert 5500 < dist < 5650

    # Same location should be 0.0
    zero_dist = haversine_distance_km(40.7128, -74.0060, 40.7128, -74.0060)
    assert pytest.approx(zero_dist, abs=1e-4) == 0.0


def test_calculate_travel_velocity():
    t1 = 1000.0
    t2 = t1 + 3600.0  # 1 hour later

    # Moving 60 km in 1 hour -> 60 km/h
    # Approximate ~0.54 degrees latitude ~60km
    vel = calculate_travel_velocity_kmh(40.0, -74.0, t1, 40.54, -74.0, t2)
    assert 55.0 <= vel <= 65.0

    # Impossible travel: NYC to London (5570 km) in 15 minutes (0.25h) -> impossible speed
    t_fast = t1 + 900.0
    fast_vel = calculate_travel_velocity_kmh(40.7128, -74.0060, t1, 51.5074, -0.1278, t_fast)
    assert fast_vel >= 1000.0

    # Initial transaction (no previous coordinate) should have 0 velocity
    assert calculate_travel_velocity_kmh(None, None, None, 40.0, -74.0, t1) == 0.0


def test_category_risk():
    assert get_category_risk("crypto") == 0.95
    assert get_category_risk("grocery") == 0.08
    assert get_category_risk("unknown_cat") == 0.20


def test_card_sliding_window_state():
    state = CardSlidingWindowState(max_window_seconds=3600.0)
    base_time = 10000.0

    # 1. First transaction: $50
    tx1 = RawTransaction(
        transaction_id="tx_1",
        user_id="usr_1",
        card_id="card_1",
        amount=50.0,
        currency="USD",
        merchant_id="m1",
        merchant_category="grocery",
        latitude=40.7128,
        longitude=-74.0060,
        timestamp=base_time,
    )
    en1 = state.update_and_enrich(tx1)
    assert en1.tx_count_last_10m == 1
    assert en1.tx_count_last_1h == 1
    assert en1.amount_rolling_avg_1h == 50.0
    assert en1.geo_distance_velocity_kmh == 0.0

    # 2. Second transaction 2 minutes later: $60
    tx2 = RawTransaction(
        transaction_id="tx_2",
        user_id="usr_1",
        card_id="card_1",
        amount=60.0,
        currency="USD",
        merchant_id="m2",
        merchant_category="restaurant",
        latitude=40.7135,
        longitude=-74.0050,
        timestamp=base_time + 120.0,
    )
    en2 = state.update_and_enrich(tx2)
    assert en2.tx_count_last_10m == 2
    assert en2.tx_count_last_1h == 2
    assert en2.amount_rolling_avg_1h == 55.0

    # 3. Third transaction 2 hours later (evicts tx1 and tx2 from 1h window)
    tx3 = RawTransaction(
        transaction_id="tx_3",
        user_id="usr_1",
        card_id="card_1",
        amount=100.0,
        currency="USD",
        merchant_id="m3",
        merchant_category="electronics",
        latitude=40.7140,
        longitude=-74.0040,
        timestamp=base_time + 7200.0,
    )
    en3 = state.update_and_enrich(tx3)
    assert en3.tx_count_last_10m == 1
    assert en3.tx_count_last_1h == 1
    assert en3.amount_rolling_avg_1h == 100.0
