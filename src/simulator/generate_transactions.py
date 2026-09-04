import argparse
import asyncio
import json
import random
import sys
import time
from typing import List
import requests

from src.simulator.profiles import UserProfile
from src.streaming.schemas import RawTransaction

try:
    from confluent_kafka import Producer
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False


class TransactionGenerator:
    """
    High-throughput synthetic financial transaction generator.
    Simulates thousands of concurrent bank cardholders and injects
    realistic fraud anomalies at a configurable target rate.
    """

    def __init__(
        self,
        rate: float = 100.0,
        fraud_rate: float = 0.015,
        target: str = "api",
        api_url: str = "http://localhost:8000/v1/transactions/ingest",
        kafka_broker: str = "localhost:9092",
        topic: str = "transactions.raw",
        num_users: int = 150,
    ):
        self.rate = rate
        self.fraud_rate = fraud_rate
        self.target = target
        self.api_url = api_url
        self.topic = topic
        self.users: List[UserProfile] = [UserProfile(f"usr_{i:04d}") for i in range(num_users)]
        self.kafka_producer = None

        if self.target == "kafka":
            if not HAS_KAFKA:
                print("[!] confluent-kafka not installed. Falling back to stdout mode.")
                self.target = "stdout"
            else:
                conf = {"bootstrap.servers": kafka_broker, "client.id": "nexus-simulator"}
                self.kafka_producer = Producer(conf)

    def generate_event(self) -> RawTransaction:
        user = random.choice(self.users)
        now = time.time()
        is_fraud = random.random() < self.fraud_rate

        if not is_fraud:
            return user.generate_legitimate_transaction(now)

        # Inject fraud attack
        attack_type = random.choice(["burst", "amount_spike", "impossible_travel"])
        if attack_type == "burst":
            return user.generate_fraud_burst_transaction(now)
        elif attack_type == "amount_spike":
            return user.generate_fraud_amount_spike(now)
        else:
            return user.generate_fraud_impossible_travel(now)

    def dispatch(self, tx: RawTransaction) -> None:
        payload = tx.model_dump()

        if self.target == "stdout":
            tag = "🔴 [FRAUD]" if tx.is_fraud_ground_truth else "🟢 [LEGIT]"
            print(f"{tag} {tx.transaction_id} | ${tx.amount:>7.2f} | {tx.merchant_category:<12} | user: {tx.user_id}")
        
        elif self.target == "api":
            try:
                requests.post(self.api_url, json=payload, timeout=2.0)
            except Exception as e:
                # Silently catch network drops during startup
                pass

        elif self.target == "kafka" and self.kafka_producer:
            key = tx.card_id.encode("utf-8")
            val = json.dumps(payload).encode("utf-8")
            self.kafka_producer.produce(self.topic, key=key, value=val)
            self.kafka_producer.poll(0)

    def run(self, duration_sec: float = 0.0) -> None:
        interval = 1.0 / self.rate
        start_time = time.time()
        count = 0
        fraud_count = 0

        print(f"[*] Starting Nexus Transaction Simulator...")
        print(f"    - Target:       {self.target}")
        print(f"    - Rate:         {self.rate} tx/sec")
        print(f"    - Fraud Ratio:  {self.fraud_rate * 100:.1f}%")
        if duration_sec > 0:
            print(f"    - Duration:     {duration_sec}s")

        try:
            while True:
                t0 = time.perf_counter()
                tx = self.generate_event()
                self.dispatch(tx)

                count += 1
                if tx.is_fraud_ground_truth:
                    fraud_count += 1

                if count % max(1, int(self.rate * 2)) == 0:
                    elapsed = time.time() - start_time
                    tps = count / elapsed if elapsed > 0 else 0
                    print(f"[*] Emitted: {count:,} transactions | Fraud: {fraud_count:,} | Actual TPS: {tps:.1f}")

                if duration_sec > 0 and (time.time() - start_time) >= duration_sec:
                    break

                compute_time = time.perf_counter() - t0
                sleep_time = interval - compute_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[*] Stopping generator...")
        finally:
            if self.kafka_producer:
                self.kafka_producer.flush()
            print(f"[+] Total transactions emitted: {count} ({fraud_count} fraud injected).")


def main():
    parser = argparse.ArgumentParser(description="Nexus Stream Synthetic Transaction Simulator")
    parser.add_argument("--rate", type=float, default=50.0, help="Transactions per second (default: 50)")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration in seconds (0 = indefinite)")
    parser.add_argument("--target", type=str, choices=["api", "kafka", "stdout"], default="api", help="Dispatch target")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1/transactions/ingest", help="FastAPI direct ingest endpoint")
    parser.add_argument("--kafka-broker", type=str, default="localhost:9092", help="Kafka/Redpanda bootstrap server")
    parser.add_argument("--topic", type=str, default="transactions.raw", help="Target topic")
    parser.add_argument("--fraud-rate", type=float, default=0.02, help="Probability of fraud anomaly injection")
    args = parser.parse_args()

    generator = TransactionGenerator(
        rate=args.rate,
        fraud_rate=args.fraud_rate,
        target=args.target,
        api_url=args.api_url,
        kafka_broker=args.kafka_broker,
        topic=args.topic,
    )
    generator.run(duration_sec=args.duration)


if __name__ == "__main__":
    main()
