import random
import uuid
import time
from typing import Dict, List, Tuple
from src.streaming.schemas import RawTransaction

# Global Hubs for Realistic User Geography
GEOGRAPHIC_HUBS = [
    {"city": "New York", "lat": 40.7128, "lon": -74.0060},
    {"city": "London", "lat": 51.5074, "lon": -0.1278},
    {"city": "San Francisco", "lat": 37.7749, "lon": -122.4194},
    {"city": "Tokyo", "lat": 35.6762, "lon": 139.6503},
    {"city": "Singapore", "lat": 1.3521, "lon": 103.8198},
    {"city": "Frankfurt", "lat": 50.1109, "lon": 8.6821},
    {"city": "Sydney", "lat": -33.8688, "lon": 151.2093},
]

LEGIT_CATEGORIES = ["grocery", "restaurant", "utilities", "pharmacy", "clothing", "entertainment", "general"]
HIGH_RISK_CATEGORIES = ["crypto", "jewelry", "electronics", "gaming", "luxury_goods"]


class UserProfile:
    """Represents an authentic cardholder with regular spending habits and baseline geography."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.card_id = f"card_{uuid.uuid4().hex[:8]}"
        self.home_hub = random.choice(GEOGRAPHIC_HUBS)
        self.mean_spend = random.uniform(25.0, 95.0)
        self.fav_categories = random.sample(LEGIT_CATEGORIES, k=3)
        self.current_lat = self.home_hub["lat"] + random.uniform(-0.05, 0.05)
        self.current_lon = self.home_hub["lon"] + random.uniform(-0.05, 0.05)
        self.last_tx_time = time.time() - random.uniform(300, 7200)

    def generate_legitimate_transaction(self, current_time: float) -> RawTransaction:
        # Small geographical drift within city radius
        self.current_lat += random.uniform(-0.005, 0.005)
        self.current_lon += random.uniform(-0.005, 0.005)
        self.last_tx_time = current_time

        amount = round(random.lognormvariate(math_log_mean := random.uniform(3.0, 4.0), 0.6), 2)
        category = random.choice(self.fav_categories)

        return RawTransaction(
            transaction_id=f"tx_{uuid.uuid4().hex[:12]}",
            user_id=self.user_id,
            card_id=self.card_id,
            amount=amount,
            currency="USD",
            merchant_id=f"merch_{random.randint(100, 999)}",
            merchant_category=category,
            latitude=round(self.current_lat, 6),
            longitude=round(self.current_lon, 6),
            timestamp=current_time,
            device_id=f"dev_{self.user_id[:8]}",
            is_fraud_ground_truth=False,
        )

    def generate_fraud_burst_transaction(self, current_time: float) -> RawTransaction:
        """Card testing or automated bot attack: rapid succession of transactions."""
        self.last_tx_time = current_time
        return RawTransaction(
            transaction_id=f"tx_{uuid.uuid4().hex[:12]}",
            user_id=self.user_id,
            card_id=self.card_id,
            amount=round(random.uniform(90.0, 450.0), 2),
            currency="USD",
            merchant_id=f"merch_{random.randint(500, 999)}",
            merchant_category=random.choice(HIGH_RISK_CATEGORIES),
            latitude=round(self.current_lat, 6),
            longitude=round(self.current_lon, 6),
            timestamp=current_time,
            device_id=f"dev_attacker_{random.randint(10, 99)}",
            is_fraud_ground_truth=True,
        )

    def generate_fraud_amount_spike(self, current_time: float) -> RawTransaction:
        """Massive monetary spike deviating 5x-10x from baseline."""
        self.last_tx_time = current_time
        return RawTransaction(
            transaction_id=f"tx_{uuid.uuid4().hex[:12]}",
            user_id=self.user_id,
            card_id=self.card_id,
            amount=round(random.uniform(2200.0, 7800.0), 2),
            currency="USD",
            merchant_id=f"merch_lux_{random.randint(100, 200)}",
            merchant_category=random.choice(HIGH_RISK_CATEGORIES),
            latitude=round(self.current_lat, 6),
            longitude=round(self.current_lon, 6),
            timestamp=current_time,
            device_id=f"dev_{self.user_id[:8]}",
            is_fraud_ground_truth=True,
        )

    def generate_fraud_impossible_travel(self, current_time: float) -> RawTransaction:
        """Transaction appearing in a distant continent within minutes of previous purchase."""
        # Pick another hub far away
        distant_hub = random.choice([h for h in GEOGRAPHIC_HUBS if h["city"] != self.home_hub["city"]])
        new_lat = distant_hub["lat"] + random.uniform(-0.02, 0.02)
        new_lon = distant_hub["lon"] + random.uniform(-0.02, 0.02)

        # Update last known pos to distant hub
        self.current_lat = new_lat
        self.current_lon = new_lon
        self.last_tx_time = current_time

        return RawTransaction(
            transaction_id=f"tx_{uuid.uuid4().hex[:12]}",
            user_id=self.user_id,
            card_id=self.card_id,
            amount=round(random.uniform(350.0, 1800.0), 2),
            currency="USD",
            merchant_id=f"merch_intl_{random.randint(100, 500)}",
            merchant_category=random.choice(["electronics", "jewelry", "travel"]),
            latitude=round(new_lat, 6),
            longitude=round(new_lon, 6),
            timestamp=current_time,
            device_id=f"dev_foreign_ip_{random.randint(1, 255)}",
            is_fraud_ground_truth=True,
        )

    def generate_fraud_ring_transaction(self, current_time: float, shared_device_id: str) -> RawTransaction:
        """Coordinated fraud ring attack: distinct cardholder transacting from a shared botnet device."""
        self.last_tx_time = current_time
        return RawTransaction(
            transaction_id=f"tx_{uuid.uuid4().hex[:12]}",
            user_id=self.user_id,
            card_id=self.card_id,
            amount=round(random.uniform(450.0, 2800.0), 2),
            currency="USD",
            merchant_id=f"merch_syndicate_{random.randint(10, 50)}",
            merchant_category=random.choice(["crypto", "electronics", "gaming"]),
            latitude=round(self.current_lat, 6),
            longitude=round(self.current_lon, 6),
            timestamp=current_time,
            device_id=shared_device_id,
            is_fraud_ground_truth=True,
        )

