import argparse
import json
import os
import sys
import time
from typing import Optional
import uuid
import requests

from src.streaming.schemas import RawTransaction, EnrichedTransaction, FraudPredictionResponse, DeadLetterEvent
from src.streaming.features import StreamFeatureStateManager
from src.inference.predictor import FraudPredictor
from src.analytics.store import DuckDBAnalyticsStore

try:
    from confluent_kafka import Consumer, Producer, KafkaError
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False


class StreamProcessor:
    """
    Stateful streaming engine for Nexus Stream.
    Consumes raw transactions, computes sliding window behavioral features,
    dispatches to the XGBoost inference engine, and routes scored events to sinks.
    """

    def __init__(
        self,
        kafka_broker: str = "localhost:9092",
        input_topic: str = "transactions.raw",
        output_topic: str = "transactions.enriched",
        alert_topic: str = "transactions.fraud-alerts",
        dlq_topic: str = "transactions.dlq",
        inference_url: Optional[str] = "http://localhost:8000/v1/predict-fraud",
        db_path: str = "nexus_analytics.duckdb",
        use_local_predictor: bool = False,
    ):
        self.kafka_broker = kafka_broker
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.alert_topic = alert_topic
        self.dlq_topic = dlq_topic
        self.inference_url = inference_url
        self.use_local_predictor = use_local_predictor

        self.feature_manager = StreamFeatureStateManager()
        self.store = DuckDBAnalyticsStore(db_path=db_path)
        self.local_predictor = FraudPredictor() if use_local_predictor else None

        self.consumer = None
        self.producer = None
        if HAS_KAFKA:
            consumer_conf = {
                "bootstrap.servers": kafka_broker,
                "group.id": "nexus-stream-processor",
                "auto.offset.reset": "latest",
            }
            producer_conf = {
                "bootstrap.servers": kafka_broker,
                "client.id": "nexus-stream-processor-out",
            }
            try:
                self.consumer = Consumer(consumer_conf)
                self.producer = Producer(producer_conf)
            except Exception as e:
                print(f"[!] Warning: Failed to connect Kafka consumer/producer: {e}")

    def score_transaction(self, enriched: EnrichedTransaction) -> FraudPredictionResponse:
        """Score via local C-API predictor or FastAPI endpoint."""
        if self.use_local_predictor and self.local_predictor:
            return self.local_predictor.predict(enriched)

        if self.inference_url:
            try:
                resp = requests.post(self.inference_url, json=enriched.model_dump(), timeout=1.0)
                if resp.status_code == 200:
                    return FraudPredictionResponse(**resp.json())
            except Exception:
                pass

        # Fallback to local predictor if API unavailable
        if not self.local_predictor:
            self.local_predictor = FraudPredictor()
        return self.local_predictor.predict(enriched)

    def process_event(self, raw_tx: RawTransaction) -> None:
        """Processes a single raw transaction through enrichment and scoring."""
        # 1. State enrichment
        enriched = self.feature_manager.enrich(raw_tx)

        # 2. ML Inference
        prediction = self.score_transaction(enriched)

        # 3. DuckDB Sink
        self.store.insert_scored_transaction(enriched, prediction)

        # 4. Kafka Egress
        if self.producer:
            key = enriched.card_id.encode("utf-8")
            val = json.dumps({
                "enriched": enriched.model_dump(),
                "scoring": prediction.model_dump(),
            }).encode("utf-8")
            self.producer.produce(self.output_topic, key=key, value=val)

            if prediction.prediction in ("FRAUD_SUSPECTED", "REVIEW_REQUIRED"):
                self.producer.produce(self.alert_topic, key=key, value=val)
            self.producer.poll(0)

        # Console Log for Visibility
        tag = "🔴 [FRAUD]" if prediction.prediction == "FRAUD_SUSPECTED" else ("🟡 [REVIEW]" if prediction.prediction == "REVIEW_REQUIRED" else "🟢 [ALLOW]")
        print(f"{tag} {prediction.transaction_id} | Score: {prediction.fraud_probability:.3f} | Latency: {prediction.latency_ms:.2f}ms | Risk: {', '.join(prediction.top_risk_factors) if prediction.top_risk_factors else 'None'}")

    def handle_poison_pill(self, raw_payload: str, error_type: str, error_msg: str) -> DeadLetterEvent:
        """Isolates malformed payloads into DuckDB and Kafka DLQ topic without stalling offset progress."""
        dlq_id = f"dlq_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
        event = DeadLetterEvent(
            dlq_id=dlq_id,
            source_topic=self.input_topic,
            error_type=error_type,
            error_message=error_msg,
            raw_payload=raw_payload,
            retry_count=0,
            status="PENDING",
            timestamp=time.time(),
        )

        # Sinks to DuckDB
        self.store.insert_dlq_event(event)

        # Routes to Kafka DLQ Topic
        if self.producer:
            key = dlq_id.encode("utf-8")
            val = json.dumps(event.model_dump()).encode("utf-8")
            self.producer.produce(self.dlq_topic, key=key, value=val)
            self.producer.poll(0)

        print(f"⚠️ [POISON PILL -> DLQ] {dlq_id} | Type: {error_type} | Reason: {error_msg[:60]}")
        return event

    def run(self) -> None:
        if not self.consumer:
            print("[!] Kafka consumer not initialized. Stream processor requires active broker.")
            return

        self.consumer.subscribe([self.input_topic])
        print(f"[*] Stream processor listening on topic '{self.input_topic}' from broker '{self.kafka_broker}'...")

        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        print(f"[!] Consumer error: {msg.error()}")
                        break

                raw_str = msg.value().decode("utf-8", errors="replace")

                # 1. Check for JSON decoding corruption
                try:
                    payload = json.loads(raw_str)
                except json.JSONDecodeError as je:
                    self.handle_poison_pill(raw_str, "MALFORMED_JSON", str(je))
                    continue

                # 2. Check for Pydantic schema validation failures
                try:
                    raw_tx = RawTransaction(**payload)
                except Exception as ve:
                    self.handle_poison_pill(raw_str, "SCHEMA_VALIDATION_ERROR", str(ve))
                    continue

                # 3. Process business logic & handle runtime exceptions
                try:
                    self.process_event(raw_tx)
                except Exception as pe:
                    self.handle_poison_pill(raw_str, "PROCESSING_FAILURE", str(pe))

        except KeyboardInterrupt:
            print("\n[*] Stopping stream processor...")
        finally:
            self.consumer.close()
            if self.producer:
                self.producer.flush()


def main():
    parser = argparse.ArgumentParser(description="Nexus Stream Processor")
    parser.add_argument("--broker", type=str, default="localhost:9092", help="Kafka/Redpanda bootstrap server")
    parser.add_argument("--input-topic", type=str, default="transactions.raw")
    parser.add_argument("--output-topic", type=str, default="transactions.enriched")
    parser.add_argument("--inference-url", type=str, default="http://localhost:8000/v1/predict-fraud")
    parser.add_argument("--local", action="store_true", help="Run local predictor without HTTP overhead")
    args = parser.parse_args()

    processor = StreamProcessor(
        kafka_broker=args.broker,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        inference_url=args.inference_url,
        use_local_predictor=args.local,
    )
    processor.run()


if __name__ == "__main__":
    main()
