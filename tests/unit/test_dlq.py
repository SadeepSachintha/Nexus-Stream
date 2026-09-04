import json
import pytest
import time
from fastapi.testclient import TestClient
from src.analytics.store import DuckDBAnalyticsStore
from src.streaming.schemas import DeadLetterEvent, RawTransaction
from src.streaming.processor import StreamProcessor
from src.inference.app import app


@pytest.fixture
def temp_store(tmp_path):
    db_file = str(tmp_path / "test_dlq.duckdb")
    store = DuckDBAnalyticsStore(db_path=db_file)
    try:
        yield store
    finally:
        store.close()


def test_dlq_duckdb_storage_and_summary(temp_store):
    now = time.time()
    event = DeadLetterEvent(
        dlq_id="dlq_test_001",
        source_topic="transactions.raw",
        error_type="MALFORMED_JSON",
        error_message="JSONDecodeError: Unterminated string",
        raw_payload='{"broken": true',
        retry_count=0,
        status="PENDING",
        timestamp=now,
    )
    temp_store.insert_dlq_event(event)

    events = temp_store.get_dlq_events(status="PENDING")
    assert len(events) == 1
    assert events[0]["dlq_id"] == "dlq_test_001"
    assert events[0]["error_type"] == "MALFORMED_JSON"

    summary = temp_store.get_dlq_summary()
    assert summary.total_dlq_events == 1
    assert summary.pending_events == 1
    assert summary.replayed_events == 0

    # Mark replayed
    temp_store.update_dlq_status("dlq_test_001", "REPLAYED", increment_retry=True)
    summary2 = temp_store.get_dlq_summary()
    assert summary2.pending_events == 0
    assert summary2.replayed_events == 1


def test_stream_processor_poison_pill_isolation(tmp_path):
    db_file = str(tmp_path / "proc_dlq.duckdb")
    processor = StreamProcessor(
        kafka_broker="localhost:9092",
        input_topic="transactions.raw",
        dlq_topic="transactions.dlq",
        db_path=db_file,
        use_local_predictor=True,
    )
    try:
        # 1. Simulate malformed JSON poison pill
        bad_json_str = '{"transaction_id": "tx_bad", "amount": 100.0, "unclosed...'
        dlq_ev1 = processor.handle_poison_pill(bad_json_str, "MALFORMED_JSON", "JSONDecodeError")
        assert dlq_ev1.error_type == "MALFORMED_JSON"
        assert dlq_ev1.status == "PENDING"

        # 2. Simulate schema validation failure (missing user_id, invalid amount)
        bad_schema_str = json.dumps({"transaction_id": "tx_no_user", "amount": -50.0})
        dlq_ev2 = processor.handle_poison_pill(bad_schema_str, "SCHEMA_VALIDATION_ERROR", "amount must be > 0")
        assert dlq_ev2.error_type == "SCHEMA_VALIDATION_ERROR"

        # Verify both isolated in DuckDB without pipeline stall
        events = processor.store.get_dlq_events()
        assert len(events) == 2
    finally:
        processor.store.close()


def test_api_dlq_lifecycle_and_replay():
    with TestClient(app) as client:
        # 1. Inject a test poison pill
        inject_resp = client.post("/v1/dlq/inject-test?poison_type=missing_field")
        assert inject_resp.status_code == 200
        dlq_item = inject_resp.json()
        dlq_id = dlq_item["dlq_id"]

        # 2. Check DLQ list
        list_resp = client.get("/v1/dlq?status=PENDING")
        assert list_resp.status_code == 200
        assert any(x["dlq_id"] == dlq_id for x in list_resp.json())

        # 3. Check DLQ summary
        summary_resp = client.get("/v1/dlq/summary")
        assert summary_resp.status_code == 200
        assert summary_resp.json()["total_dlq_events"] >= 1

        # 4. Replay with corrected payload
        corrected = {
            "transaction_id": "tx_recovered_01",
            "user_id": "usr_fixed",
            "card_id": "card_fixed",
            "amount": 125.0,
            "currency": "USD",
            "merchant_id": "m1",
            "merchant_category": "grocery",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "timestamp": time.time(),
        }
        replay_resp = client.post(f"/v1/dlq/{dlq_id}/replay", json=corrected)
        assert replay_resp.status_code == 200
        scored = replay_resp.json()
        assert scored["transaction_id"] == "tx_recovered_01"
        assert scored["prediction"] in ("LEGITIMATE", "REVIEW_REQUIRED", "FRAUD_SUSPECTED")

        # 5. Check DLQ status transitioned to REPLAYED
        replayed_list = client.get("/v1/dlq?status=REPLAYED")
        assert replayed_list.status_code == 200
        assert any(x["dlq_id"] == dlq_id for x in replayed_list.json())
