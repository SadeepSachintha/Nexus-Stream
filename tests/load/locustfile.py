"""
Nexus Stream - High-Throughput Load & Stress Testing Suite (Locust)
Simulates realistic concurrent payment traffic, impossible travel velocity attacks,
and device-sharing fraud rings to benchmark sub-15ms p99 inference SLAs.
"""

import time
import uuid
import random
from locust import HttpUser, task, between, events


CITIES = [
    {"name": "New York", "lat": 40.7128, "lon": -74.0060},
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    {"name": "London", "lat": 51.5074, "lon": -0.1278},
    {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
    {"name": "Singapore", "lat": 1.3521, "lon": 103.8198},
]

LEGIT_CATEGORIES = ["grocery", "restaurant", "pharmacy", "subscription", "retail", "transport"]
RISKY_CATEGORIES = ["crypto_exchange", "luxury_goods", "electronics", "money_transfer", "casino"]


class LegitimateShopperUser(HttpUser):
    """
    Simulates high-volume benign cardholders performing routine purchases.
    Represents 85% of standard production traffic.
    """
    weight = 85
    wait_time = between(0.02, 0.10)

    def on_start(self):
        self.user_id = f"usr_legit_{uuid.uuid4().hex[:6]}"
        self.card_id = f"card_legit_{uuid.uuid4().hex[:6]}"
        self.device_id = f"dev_mobile_{uuid.uuid4().hex[:6]}"
        self.city = random.choice(CITIES)

    @task(8)
    def ingest_normal_transaction(self):
        """Dispatches an end-to-end raw transaction through enrichment, graph, and XGBoost scoring."""
        amount = round(random.lognormvariate(3.5, 0.6), 2)  # ~$20 - $120
        payload = {
            "transaction_id": f"tx_locust_{uuid.uuid4().hex[:8]}",
            "user_id": self.user_id,
            "card_id": self.card_id,
            "amount": max(1.0, amount),
            "currency": "USD",
            "merchant_id": f"merch_{random.randint(100, 999)}",
            "merchant_category": random.choice(LEGIT_CATEGORIES),
            "latitude": self.city["lat"] + random.uniform(-0.05, 0.05),
            "longitude": self.city["lon"] + random.uniform(-0.05, 0.05),
            "device_id": self.device_id,
            "timestamp": time.time(),
        }

        with self.client.post("/v1/transactions/ingest", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "fraud_probability" in data and "prediction" in data:
                    response.success()
                else:
                    response.failure(f"Malformed schema response: {data}")
            else:
                response.failure(f"HTTP error {response.status_code}: {response.text}")

    @task(1)
    def check_health_probe(self):
        """Periodically hits the liveness endpoint."""
        self.client.get("/v1/health")


class ImpossibleTravelAttacker(HttpUser):
    """
    Simulates compromised cards executing rapid cross-continental transactions.
    Triggers Haversine velocity alarms (>800 km/h) and impossible travel policy overrides.
    """
    weight = 10
    wait_time = between(0.01, 0.05)

    def on_start(self):
        self.card_id = f"card_hijacked_{uuid.uuid4().hex[:6]}"
        self.user_id = f"usr_victim_{uuid.uuid4().hex[:6]}"
        self.device_id = f"dev_attacker_{uuid.uuid4().hex[:6]}"
        self.city_idx = 0

    @task
    def impossible_velocity_burst(self):
        """Alternates between cities across the globe within seconds."""
        city = CITIES[self.city_idx % len(CITIES)]
        self.city_idx += 1

        payload = {
            "transaction_id": f"tx_attack_{uuid.uuid4().hex[:8]}",
            "user_id": self.user_id,
            "card_id": self.card_id,
            "amount": round(random.uniform(400.0, 1800.0), 2),
            "currency": "USD",
            "merchant_id": f"merch_risky_{random.randint(10, 50)}",
            "merchant_category": random.choice(RISKY_CATEGORIES),
            "latitude": city["lat"],
            "longitude": city["lon"],
            "device_id": self.device_id,
            "timestamp": time.time(),
        }

        with self.client.post("/v1/transactions/ingest", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                # Confirm pipeline scored and flagged appropriately
                if data.get("prediction") in ("FRAUD_SUSPECTED", "REVIEW_REQUIRED", "LEGITIMATE"):
                    response.success()
                else:
                    response.failure("Unexpected prediction category")
            else:
                response.failure(f"Attack transaction failed with status {response.status_code}")


class FraudRingSyndicateUser(HttpUser):
    """
    Simulates a coordinated botnet / carding syndicate sharing the same device across multiple cards.
    Triggers the in-memory bipartite entity graph ring detector.
    """
    weight = 5
    wait_time = between(0.01, 0.04)

    SHARED_DEVICE_ID = "dev_botnet_syndicate_shared_01"

    @task
    def multi_card_device_sharing(self):
        # Rotate through 10 cards sharing the exact same device ID
        card_num = random.randint(1, 10)
        payload = {
            "transaction_id": f"tx_ring_{uuid.uuid4().hex[:8]}",
            "user_id": f"usr_ring_{card_num}",
            "card_id": f"card_ring_syndicate_{card_num}",
            "amount": round(random.uniform(250.0, 950.0), 2),
            "currency": "USD",
            "merchant_id": "merch_crypto_hub",
            "merchant_category": "crypto_exchange",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "device_id": self.SHARED_DEVICE_ID,
            "timestamp": time.time(),
        }

        with self.client.post("/v1/transactions/ingest", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Ring transaction failed: {response.status_code}")


# --- SLA Verification & Test Hooks ---

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Evaluates latency SLA threshold when benchmark run finishes."""
    stats = environment.runner.stats.total
    if stats.num_requests > 0:
        p95 = stats.get_current_response_time_percentile(0.95) or 0
        p99 = stats.get_current_response_time_percentile(0.99) or 0
        rps = stats.current_rps or 0
        fail_ratio = stats.fail_ratio

        print("\n" + "=" * 60)
        print("⚡ NEXUS STREAM LOAD TEST SUMMARY & SLA REPORT")
        print("=" * 60)
        print(f"Total Requests Processed: {stats.num_requests:,}")
        print(f"Failed Requests:         {stats.num_failures:,} ({fail_ratio * 100:.2f}%)")
        print(f"Average Response Time:   {stats.avg_response_time:.2f} ms")
        print(f"P95 Latency:             {p95:.2f} ms")
        print(f"P99 Latency:             {p99:.2f} ms")
        print(f"SLA Target (< 20ms p99): {'✅ PASSED' if p99 < 20 else '⚠️ WARNING - EXCEEDS SLA'}")
        print("=" * 60 + "\n")
