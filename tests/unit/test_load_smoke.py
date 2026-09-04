import pytest
from fastapi.testclient import TestClient

from tests.load.benchmark_cli import generate_benchmark_payload
from src.inference.app import app


def test_benchmark_payload_structure():
    """Validates that benchmark generator produces compliant RawTransaction payloads."""
    for i in range(10):
        p = generate_benchmark_payload(i)
        assert "transaction_id" in p
        assert "amount" in p
        assert p["amount"] > 0
        assert "card_id" in p
        assert "merchant_category" in p


def test_benchmark_payload_e2e_scoring():
    """Scores benchmark payloads through FastAPI TestClient to ensure 0 schema regressions."""
    with TestClient(app) as client:
        for i in range(5):
            payload = generate_benchmark_payload(i)
            res = client.post("/v1/transactions/ingest", json=payload)
            assert res.status_code == 200
            data = res.json()
            assert "fraud_probability" in data
            assert data["prediction"] in ("LEGITIMATE", "REVIEW_REQUIRED", "FRAUD_SUSPECTED")
            assert data["latency_ms"] >= 0.0


def test_run_async_benchmark_smoke():
    """Smoke test for benchmark CLI in-process execution."""
    import asyncio
    from tests.load.benchmark_cli import run_async_benchmark

    code = asyncio.run(run_async_benchmark(
        base_url="http://benchmark.local",
        total_requests=10,
        concurrency=2,
        endpoint="/v1/predict-fraud",
        sla_target_ms=35.0,
        in_process=True
    ))
    assert code == 0
