"""
Apache Flink Stateful Stream Processing Job for Nexus Stream
Implements event-time sliding windows, bounded out-of-orderness watermarking,
and stateful feature enrichment over transaction streams.
"""

import json
import os
import sys
from typing import Dict, Any

# Flink imports are wrapped to allow linting and testing even when PyFlink JVM is not in local path
try:
    from pyflink.common import WatermarkStrategy, Duration, Types
    from pyflink.datastream import StreamExecutionEnvironment, KeyedProcessFunction, RuntimeContext
    from pyflink.datastream.state import ValueStateDescriptor, ListStateDescriptor
    from pyflink.datastream.connectors.kafka import KafkaSource, KafkaSink, KafkaRecordSerializationSchema
    from pyflink.common.serialization import SimpleStringSchema
    HAS_PYFLINK = True
except ImportError:
    HAS_PYFLINK = False

from src.streaming.features import (
    haversine_distance_km,
    calculate_travel_velocity_kmh,
    get_category_risk,
    is_night_transaction
)


if HAS_PYFLINK:
    class StatefulEnrichmentFunction(KeyedProcessFunction):
        """
        Keyed process function maintaining 1-hour sliding state per card_id.
        Computes real-time behavioral features on event stream.
        """

        def open(self, runtime_context: RuntimeContext):
            # List of (timestamp, amount) within last hour
            self.history_state = runtime_context.get_list_state(
                ListStateDescriptor("history_state", Types.STRING())
            )
            # Last known location and timestamp: "lat,lon,timestamp"
            self.last_pos_state = runtime_context.get_value_state(
                ValueStateDescriptor("last_pos_state", Types.STRING())
            )

        def process_element(self, value_str: str, ctx: 'KeyedProcessFunction.Context'):
            tx = json.loads(value_str)
            current_time = float(tx.get("timestamp", 0.0))
            amount = float(tx.get("amount", 0.0))
            lat = float(tx.get("latitude", 0.0))
            lon = float(tx.get("longitude", 0.0))
            category = str(tx.get("merchant_category", "general"))

            # 1. Prune events older than 1 hour (3600s)
            cutoff_1h = current_time - 3600.0
            cutoff_10m = current_time - 600.0
            
            retained_events = []
            history_iter = self.history_state.get()
            if history_iter:
                for entry in history_iter:
                    t, a = entry.split(":")
                    t_val, a_val = float(t), float(a)
                    if t_val >= cutoff_1h:
                        retained_events.append((t_val, a_val))

            # 2. Window Counts
            tx_count_10m = sum(1 for t, _ in retained_events if t >= cutoff_10m) + 1
            tx_count_1h = len(retained_events) + 1

            # 3. Rolling Monetary Statistics & Z-Score
            if retained_events:
                amounts = [a for _, a in retained_events]
                rolling_avg = sum(amounts) / len(amounts)
                variance = sum((x - rolling_avg) ** 2 for x in amounts) / max(1, len(amounts) - 1)
                std = variance ** 0.5
                z_score = (amount - rolling_avg) / max(std, 10.0)
            else:
                rolling_avg = amount
                z_score = 0.0

            # 4. Travel Velocity
            last_pos = self.last_pos_state.value()
            if last_pos:
                p_lat, p_lon, p_t = [float(x) for x in last_pos.split(",")]
                velocity_kmh = calculate_travel_velocity_kmh(p_lat, p_lon, p_t, lat, lon, current_time)
            else:
                velocity_kmh = 0.0

            # 5. Update State
            retained_events.append((current_time, amount))
            self.history_state.update([f"{t}:{a}" for t, a in retained_events])
            self.last_pos_state.update(f"{lat},{lon},{current_time}")

            # 6. Build Enriched Payload
            enriched_tx = dict(tx)
            enriched_tx.update({
                "tx_count_last_10m": tx_count_10m,
                "tx_count_last_1h": tx_count_1h,
                "amount_rolling_avg_1h": round(rolling_avg, 2),
                "amount_deviation_from_avg": round(z_score, 2),
                "geo_distance_velocity_kmh": velocity_kmh,
                "is_night_transaction": is_night_transaction(current_time),
                "category_risk_index": get_category_risk(category),
            })

            yield json.dumps(enriched_tx)


def build_flink_pipeline(
    bootstrap_servers: str = "redpanda:9092",
    input_topic: str = "transactions.raw",
    output_topic: str = "transactions.enriched",
):
    """Initializes and runs the Apache Flink DataStream job."""
    if not HAS_PYFLINK:
        print("[!] PyFlink not installed in current environment. Flink job must run inside Flink container.")
        return

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(2)
    env.enable_checkpointing(5000)  # Checkpoint every 5 seconds

    kafka_source = KafkaSource.builder() \
        .set_bootstrap_servers(bootstrap_servers) \
        .set_topics(input_topic) \
        .set_group_id("flink-fraud-detector") \
        .set_value_only_deserializer(SimpleStringSchema()) \
        .build()

    watermark_strategy = WatermarkStrategy \
        .for_bounded_out_of_orderness(Duration.of_seconds(5)) \
        .with_timestamp_assigner(lambda event, _: int(json.loads(event).get("timestamp", 0) * 1000))

    stream = env.from_source(kafka_source, watermark_strategy, "Kafka-Raw-Transactions")

    enriched_stream = stream \
        .key_by(lambda event: json.loads(event).get("card_id", "default")) \
        .process(StatefulEnrichmentFunction(), output_type=Types.STRING())

    kafka_sink = KafkaSink.builder() \
        .set_bootstrap_servers(bootstrap_servers) \
        .set_record_serializer(
            KafkaRecordSerializationSchema.builder()
            .set_topic(output_topic)
            .set_value_serialization_schema(SimpleStringSchema())
            .build()
        ) \
        .build()

    enriched_stream.sink_to(kafka_sink)

    print(f"[*] Executing Flink job: streaming from {input_topic} -> {output_topic}...")
    env.execute("Nexus-Stream-Stateful-Fraud-Enrichment")


if __name__ == "__main__":
    broker = os.getenv("KAFKA_BROKER", "localhost:9092")
    build_flink_pipeline(bootstrap_servers=broker)
