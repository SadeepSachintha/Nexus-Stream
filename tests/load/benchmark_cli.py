"""
Nexus Stream - Automated Stress & Concurrency Benchmark CLI
Executes headless concurrency tests against the FastAPI inference gateway,
measures sub-millisecond percentiles (p50, p95, p99), and validates SLA compliance.
"""

import argparse
import sys
import time
import uuid
import random
import asyncio
from typing import List, Dict, Any
import httpx
import numpy as np

CITIES = [
    {"name": "New York", "lat": 40.7128, "lon": -74.0060},
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    {"name": "London", "lat": 51.5074, "lon": -0.1278},
    {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
]


def generate_benchmark_payload(idx: int, is_enriched: bool = False) -> Dict[str, Any]:
    city = CITIES[idx % len(CITIES)]
    is_attack = (idx % 10 == 0)
    card_id = f"card_bench_{idx % 50}"
    device_id = "dev_shared_ring" if (idx % 7 == 0) else f"dev_user_{idx % 50}"

    amount = round(random.uniform(500.0, 3500.0) if is_attack else random.lognormvariate(3.5, 0.6), 2)
    payload = {
        "transaction_id": f"tx_bench_{uuid.uuid4().hex[:10]}",
        "user_id": f"usr_bench_{idx % 50}",
        "card_id": card_id,
        "amount": max(1.0, amount),
        "currency": "USD",
        "merchant_id": f"m_bench_{idx % 20}",
        "merchant_category": "crypto_exchange" if is_attack else "retail",
        "latitude": city["lat"] + random.uniform(-0.02, 0.02),
        "longitude": city["lon"] + random.uniform(-0.02, 0.02),
        "device_id": device_id,
        "timestamp": time.time() + (idx * 0.01),
    }
    if is_enriched:
        payload.update({
            "tx_count_last_10m": random.randint(1, 4),
            "tx_count_last_1h": random.randint(2, 8),
            "amount_rolling_avg_1h": amount * 0.95,
            "amount_deviation_from_avg": 0.5 if not is_attack else 4.5,
            "geo_distance_velocity_kmh": 25.0 if not is_attack else 950.0,
            "is_night_transaction": 0,
            "category_risk_index": 0.1 if not is_attack else 0.85,
        })
    return payload


async def worker(
    client: httpx.AsyncClient,
    host_url: str,
    endpoint: str,
    queue: asyncio.Queue,
    latencies: List[float],
    errors: List[str],
    is_enriched: bool = False,
):
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        payload = generate_benchmark_payload(item, is_enriched=is_enriched)
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{host_url}{endpoint}", json=payload, timeout=5.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if r.status_code == 200:
                latencies.append(elapsed_ms)
            else:
                errors.append(f"HTTP {r.status_code}: {r.text[:100]}")
        except Exception as e:
            errors.append(str(e))
        finally:
            queue.task_done()


async def run_async_benchmark(
    base_url: str,
    total_requests: int,
    concurrency: int,
    endpoint: str = "/v1/transactions/ingest",
    sla_target_ms: Optional[float] = None,
    in_process: bool = False
) -> int:
    is_in_process = in_process
    app_instance = None

    if not is_in_process:
        # Check health of remote target first
        async with httpx.AsyncClient() as test_client:
            try:
                h = await test_client.get(f"{base_url}/v1/health", timeout=1.5)
                if h.status_code != 200:
                    print(f"[!] Target service at {base_url} responded with status {h.status_code}. Switching to in-process ASGI mode...")
                    is_in_process = True
            except Exception:
                print(f"[*] Live server at {base_url} not detected. Running high-throughput benchmark in in-process ASGI mode...")
                is_in_process = True

    if is_in_process:
        from src.inference.app import app
        app_instance = app
        target_display = "In-Process FastAPI (ASGI Direct Memory Mode)"
    else:
        target_display = base_url

    is_enriched = (endpoint == "/v1/predict-fraud")
    target_sla = sla_target_ms or (15.0 if is_enriched else 55.0)

    print(f"\n[*] Target Gateway:    {target_display}")
    print(f"[*] Target Endpoint:   {endpoint}")
    print(f"[*] Total Requests:    {total_requests:,} | Concurrency Workers: {concurrency}")
    print(f"[*] Configured SLA:    p99 < {target_sla:.1f} ms")

    queue: asyncio.Queue = asyncio.Queue()
    for i in range(total_requests):
        queue.put_nowait(i)

    latencies: List[float] = []
    errors: List[str] = []

    async def execute_run(client: httpx.AsyncClient, host_url: str):
        t_start = time.perf_counter()
        tasks = [
            asyncio.create_task(worker(client, host_url, endpoint, queue, latencies, errors, is_enriched=is_enriched))
            for _ in range(concurrency)
        ]
        await asyncio.gather(*tasks)
        return time.perf_counter() - t_start

    limits = httpx.Limits(max_keepalive_connections=concurrency * 2, max_connections=concurrency * 4)

    if is_in_process and app_instance is not None:
        async with app_instance.router.lifespan_context(app_instance):
            transport = httpx.ASGITransport(app=app_instance)
            async with httpx.AsyncClient(transport=transport, base_url="http://benchmark.local", limits=limits) as client:
                total_time = await execute_run(client, "http://benchmark.local")
    else:
        async with httpx.AsyncClient(limits=limits) as client:
            total_time = await execute_run(client, base_url)

    successful = len(latencies)
    failed = len(errors)
    rps = successful / total_time if total_time > 0 else 0

    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p90 = float(np.percentile(latencies, 90)) if latencies else 0.0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    p99 = float(np.percentile(latencies, 99)) if latencies else 0.0
    max_lat = float(np.max(latencies)) if latencies else 0.0
    avg_lat = float(np.mean(latencies)) if latencies else 0.0

    print("\n" + "=" * 65)
    print("           NEXUS STREAM BENCHMARK PERFORMANCE REPORT             ")
    print("=" * 65)
    print(f"Target Mode:             {target_display}")
    print(f"Target Endpoint:         {endpoint}")
    print(f"Total Duration:          {total_time:.2f} s")
    print(f"Total Completed:         {successful:,} / {total_requests:,}")
    print(f"Failed / Dropped:        {failed} ({failed / total_requests * 100:.2f}%)")
    print(f"Throughput (RPS):        {rps:,.1f} transactions/sec")
    print("-" * 65)
    print("LATENCY DISTRIBUTION (End-to-End Client Measured):")
    print(f"  - Mean Latency:        {avg_lat:.2f} ms")
    print(f"  - Median (P50):        {p50:.2f} ms")
    print(f"  - 90th Percentile:     {p90:.2f} ms")
    print(f"  - 95th Percentile:     {p95:.2f} ms")
    print(f"  - 99th Percentile:     {p99:.2f} ms")
    print(f"  - Max Latency:         {max_lat:.2f} ms")
    print("-" * 65)

    sla_passed = (p99 < target_sla) and (failed == 0)
    status_label = f"PASSED [Sub-{target_sla:.0f}ms P99 SLA Satisfied]" if sla_passed else f"WARNING - EXCEEDS {target_sla:.0f}ms SLA"
    print(f"SLA COMPLIANCE:          {status_label}")
    print("=" * 65 + "\n")

    return 0 if sla_passed else 1


def main():
    parser = argparse.ArgumentParser(description="Nexus Stream Concurrency & Load Benchmark")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Base URL of FastAPI Gateway")
    parser.add_argument("--requests", type=int, default=500, help="Total transactions to dispatch")
    parser.add_argument("--concurrency", type=int, default=25, help="Number of concurrent worker connections")
    parser.add_argument("--endpoint", type=str, default="/v1/transactions/ingest", choices=["/v1/transactions/ingest", "/v1/predict-fraud"], help="API endpoint to test")
    parser.add_argument("--sla-ms", type=float, default=None, help="Custom p99 SLA latency target in milliseconds")
    parser.add_argument("--in-process", action="store_true", help="Force in-process ASGI execution without external server")
    args = parser.parse_args()

    exit_code = asyncio.run(run_async_benchmark(
        args.url,
        args.requests,
        args.concurrency,
        endpoint=args.endpoint,
        sla_target_ms=args.sla_ms,
        in_process=args.in_process
    ))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
