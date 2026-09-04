import math
from collections import deque
from typing import Dict, Optional, Tuple
import datetime

from src.streaming.schemas import RawTransaction, EnrichedTransaction

CATEGORY_RISK_MAP: Dict[str, float] = {
    "crypto": 0.95,
    "jewelry": 0.85,
    "electronics": 0.75,
    "gaming": 0.70,
    "luxury_goods": 0.65,
    "travel": 0.60,
    "clothing": 0.30,
    "entertainment": 0.25,
    "restaurant": 0.15,
    "pharmacy": 0.10,
    "grocery": 0.08,
    "utilities": 0.05,
    "general": 0.20,
}


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on the Earth 
    using the Haversine formula (returned in kilometers).
    """
    # Earth radius in kilometers
    r = 6371.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def calculate_travel_velocity_kmh(
    lat1: Optional[float],
    lon1: Optional[float],
    t1: Optional[float],
    lat2: float,
    lon2: float,
    t2: float,
) -> float:
    """
    Calculate physical velocity (km/h) between two geographical coordinates.
    Flags impossible travel when velocity exceeds commercial flight thresholds (>800-900 km/h).
    """
    if lat1 is None or lon1 is None or t1 is None:
        return 0.0

    time_diff_hours = (t2 - t1) / 3600.0
    if time_diff_hours <= 0:
        # Simultaneous or negative delta: if distance > 5km, flag as severe instant jump
        dist = haversine_distance_km(lat1, lon1, lat2, lon2)
        return 5000.0 if dist > 5.0 else 0.0

    dist_km = haversine_distance_km(lat1, lon1, lat2, lon2)
    velocity = dist_km / time_diff_hours
    return round(min(velocity, 5000.0), 2)


def get_category_risk(category: str) -> float:
    """Lookup category baseline risk index."""
    return CATEGORY_RISK_MAP.get(category.lower(), CATEGORY_RISK_MAP["general"])


def is_night_transaction(timestamp: float) -> int:
    """Return 1 if transaction occurs between 00:00 and 06:00 UTC/local time, else 0."""
    dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
    return 1 if dt.hour < 6 else 0


class CardSlidingWindowState:
    """
    Stateful rolling window representation per card/account.
    Maintains historical transaction records within a 1-hour event-time window
    and tracks physical coordinates to detect velocity anomalies.
    """

    def __init__(self, max_window_seconds: float = 3600.0):
        self.max_window_seconds = max_window_seconds
        # Stores tuples: (timestamp, amount)
        self.history = deque()
        self.last_lat: Optional[float] = None
        self.last_lon: Optional[float] = None
        self.last_timestamp: Optional[float] = None

    def prune_stale_events(self, current_timestamp: float) -> None:
        """Evicts events outside the rolling window [current_timestamp - max_window_seconds, current_timestamp]."""
        cutoff = current_timestamp - self.max_window_seconds
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

    def update_and_enrich(self, tx: RawTransaction) -> EnrichedTransaction:
        """
        Enriches a raw transaction by calculating stateful window features 
        and updates internal state.
        """
        self.prune_stale_events(tx.timestamp)

        # 1. Frequency features
        # 10 minute window count (600 seconds)
        ten_min_cutoff = tx.timestamp - 600.0
        tx_count_last_10m = sum(1 for t, _ in self.history if t >= ten_min_cutoff) + 1
        tx_count_last_1h = len(self.history) + 1

        # 2. Monetary statistics
        if self.history:
            amounts = [amt for _, amt in self.history]
            baseline_avg = sum(amounts) / len(amounts)
            if len(amounts) > 1:
                variance = sum((x - baseline_avg) ** 2 for x in amounts) / (len(amounts) - 1)
                rolling_std = math.sqrt(variance)
            else:
                rolling_std = baseline_avg * 0.4
            
            # Z-Score deviation against baseline
            std_safe = max(rolling_std, 10.0)
            z_score = (tx.amount - baseline_avg) / std_safe
            updated_rolling_avg = (sum(amounts) + tx.amount) / (len(amounts) + 1)
        else:
            updated_rolling_avg = tx.amount
            z_score = 0.0

        # 3. Geographical travel velocity
        velocity_kmh = calculate_travel_velocity_kmh(
            self.last_lat, self.last_lon, self.last_timestamp,
            tx.latitude, tx.longitude, tx.timestamp
        )

        # 4. Contextual risk
        night_tx = is_night_transaction(tx.timestamp)
        cat_risk = get_category_risk(tx.merchant_category)

        # Record this transaction into state
        self.history.append((tx.timestamp, tx.amount))
        self.last_lat = tx.latitude
        self.last_lon = tx.longitude
        self.last_timestamp = tx.timestamp

        return EnrichedTransaction(
            transaction_id=tx.transaction_id,
            user_id=tx.user_id,
            card_id=tx.card_id,
            amount=tx.amount,
            currency=tx.currency,
            merchant_id=tx.merchant_id,
            merchant_category=tx.merchant_category,
            latitude=tx.latitude,
            longitude=tx.longitude,
            timestamp=tx.timestamp,
            device_id=tx.device_id,
            is_fraud_ground_truth=tx.is_fraud_ground_truth,
            tx_count_last_10m=tx_count_last_10m,
            tx_count_last_1h=tx_count_last_1h,
            amount_rolling_avg_1h=round(updated_rolling_avg, 2),
            amount_deviation_from_avg=round(z_score, 2),
            geo_distance_velocity_kmh=velocity_kmh,
            is_night_transaction=night_tx,
            category_risk_index=cat_risk,
        )


from src.streaming.graph import FraudRingGraphEngine


class StreamFeatureStateManager:
    """
    Thread-safe / multi-key state manager indexing rolling card windows
    and maintaining the bipartite entity fraud ring graph.
    """

    def __init__(self, graph_ttl_seconds: float = 86400.0, ring_threshold: int = 3):
        self._states: Dict[str, CardSlidingWindowState] = {}
        self.graph_engine = FraudRingGraphEngine(ttl_seconds=graph_ttl_seconds, ring_threshold=ring_threshold)

    def enrich(self, tx: RawTransaction) -> EnrichedTransaction:
        key = tx.card_id or tx.user_id
        if key not in self._states:
            self._states[key] = CardSlidingWindowState()
        enriched = self._states[key].update_and_enrich(tx)

        # Graph entity relationship tracking
        dev_card_cnt, card_dev_cnt, cluster_sz, is_ring = self.graph_engine.record_interaction(
            card_id=tx.card_id,
            device_id=tx.device_id,
            timestamp=tx.timestamp,
        )

        enriched.device_card_count = dev_card_cnt
        enriched.card_device_count = card_dev_cnt
        enriched.ring_cluster_size = cluster_sz
        enriched.is_fraud_ring_suspected = is_ring

        return enriched

    def reset(self) -> None:
        self._states.clear()
        self.graph_engine.reset()
