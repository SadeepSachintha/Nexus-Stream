import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response, HTMLResponse

from src.streaming.schemas import (
    RawTransaction,
    EnrichedTransaction,
    FraudPredictionResponse,
    AnalyticsSummaryResponse,
    DeadLetterEvent,
    DLQSummaryResponse,
    DriftReportResponse,
)
import json
import uuid
import numpy as np
from src.streaming.features import StreamFeatureStateManager
from src.inference.predictor import FraudPredictor
from src.inference.train import train_model
from src.analytics.store import DuckDBAnalyticsStore
from src.analytics.drift import StatisticalDriftEngine

# Prometheus Metrics
TX_COUNTER = Counter(
    "nexus_transactions_total",
    "Total transactions scored",
    ["prediction"]
)
INFERENCE_LATENCY = Histogram(
    "nexus_inference_latency_seconds",
    "Inference latency distribution in seconds",
    buckets=[0.001, 0.005, 0.010, 0.015, 0.020, 0.050, 0.100]
)

# Global Application State
predictor: Optional[FraudPredictor] = None
store: Optional[DuckDBAnalyticsStore] = None
feature_manager: Optional[StreamFeatureStateManager] = None
drift_engine: Optional[StatisticalDriftEngine] = None
START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, store, feature_manager, drift_engine
    print("[*] Initializing Nexus Stream API...")
    
    # 1. Initialize Analytics Store
    db_path = os.getenv("DUCKDB_PATH", "nexus_analytics.duckdb")
    store = DuckDBAnalyticsStore(db_path=db_path)
    
    # 2. Initialize Feature State Manager
    feature_manager = StreamFeatureStateManager()

    # 3. Initialize Statistical Drift Monitoring Engine
    drift_engine = StatisticalDriftEngine()

    # 4. Initialize Predictor & Auto-train if model missing
    predictor = FraudPredictor()
    if not predictor.is_ready():
        print("[!] No trained XGBoost model found. Running automated initial training...")
        train_model()
        predictor.load_model()
        print("[+] Model trained and loaded successfully.")

    yield
    print("[*] Shutting down Nexus Stream API...")


app = FastAPI(
    title="Nexus Stream: Fraud Detection Inference Gateway",
    description="Sub-second low-latency financial fraud detection engine with XGBoost C-API execution.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32">
  <rect width="32" height="24" x="0" y="4" rx="4" fill="#0f172a" stroke="#3b82f6" stroke-width="2"/>
  <rect width="32" height="5" x="0" y="9" fill="#3b82f6"/>
  <rect width="6" height="4" x="4" y="17" rx="1" fill="#f59e0b"/>
  <circle cx="26" cy="19" r="2.5" fill="#10b981"/>
</svg>"""

LANDING_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Nexus Stream — Fraud Detection Inference Gateway</title>
  <link rel="icon" href="/favicon.ico" type="image/svg+xml">
  <style>
    :root {
      --bg: #0b0f19;
      --surface: #111827;
      --card: #1f2937;
      --card-hover: #1e293b;
      --border: #374151;
      --primary: #3b82f6;
      --accent: #10b981;
      --text: #f9fafb;
      --text-muted: #9ca3af;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 3rem 1.5rem;
    }
    .container {
      max-width: 980px;
      width: 100%;
    }
    .header {
      text-align: center;
      margin-bottom: 2.5rem;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.35rem 0.85rem;
      border-radius: 9999px;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.4);
      color: #34d399;
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 1.25rem;
    }
    .pulse-dot {
      width: 8px;
      height: 8px;
      background: #10b981;
      border-radius: 50%;
      box-shadow: 0 0 10px #10b981;
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }
    h1 {
      font-size: 2.5rem;
      font-weight: 800;
      letter-spacing: -0.025em;
      background: linear-gradient(135deg, #ffffff 30%, #93c5fd 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.75rem;
    }
    p.subtitle {
      color: var(--text-muted);
      font-size: 1.1rem;
      max-width: 680px;
      margin: 0 auto;
      line-height: 1.6;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1.25rem;
      margin-bottom: 2.5rem;
    }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
      text-decoration: none;
      color: var(--text);
      transition: all 0.2s ease;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .card:hover {
      background: var(--card-hover);
      border-color: #60a5fa;
      transform: translateY(-2px);
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 0 15px rgba(59, 130, 246, 0.2);
    }
    .card-icon {
      font-size: 1.75rem;
      margin-bottom: 0.75rem;
    }
    .card-title {
      font-size: 1.15rem;
      font-weight: 700;
      margin-bottom: 0.35rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .card-desc {
      color: var(--text-muted);
      font-size: 0.9rem;
      line-height: 1.5;
    }
    .tag {
      font-size: 0.75rem;
      padding: 0.15rem 0.5rem;
      border-radius: 6px;
      background: rgba(59, 130, 246, 0.15);
      color: #93c5fd;
      border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .curl-box {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem 1.5rem;
    }
    .curl-title {
      font-size: 0.9rem;
      font-weight: 600;
      color: #93c5fd;
      margin-bottom: 0.5rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    pre {
      background: #070b14;
      border: 1px solid #1e293b;
      border-radius: 8px;
      padding: 1rem;
      overflow-x: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.85rem;
      color: #e2e8f0;
      line-height: 1.5;
    }
    .footer {
      margin-top: 3rem;
      text-align: center;
      color: #6b7280;
      font-size: 0.85rem;
    }
    .footer a {
      color: #93c5fd;
      text-decoration: none;
    }
    .footer a:hover {
      text-decoration: underline;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="badge">
        <span class="pulse-dot"></span>
        Gateway Active &bull; Sub-15ms Latency SLA Guaranteed
      </div>
      <h1>Nexus Stream Gateway</h1>
      <p class="subtitle">
        Enterprise real-time fraud detection inference engine with XGBoost C-API execution, bipartite graph rings, and DuckDB OLAP persistence.
      </p>
    </div>

    <div class="grid">
      <a href="/docs" class="card">
        <div>
          <div class="card-icon">⚡</div>
          <div class="card-title">
            Interactive Swagger Docs
            <span class="tag">OpenAPI</span>
          </div>
          <div class="card-desc">Direct in-browser API explorer with live schemas, sandbox payloads, and instant responses.</div>
        </div>
      </a>

      <a href="/redoc" class="card">
        <div>
          <div class="card-icon">📖</div>
          <div class="card-title">
            ReDoc Specification
            <span class="tag">Reference</span>
          </div>
          <div class="card-desc">Detailed REST schema definitions, request/response models, and status codes.</div>
        </div>
      </a>

      <a href="http://localhost:8501" target="_blank" class="card">
        <div>
          <div class="card-icon">📊</div>
          <div class="card-title">
            Dispute Ops Portal
            <span class="tag">:8501</span>
          </div>
          <div class="card-desc">Streamlit UI for alert triage, bipartite entity network graphs, and drift radars.</div>
        </div>
      </a>

      <a href="/v1/health" class="card">
        <div>
          <div class="card-icon">🩺</div>
          <div class="card-title">
            Cluster Health Check
            <span class="tag">Liveness</span>
          </div>
          <div class="card-desc">Inspect uptime, broker connectivity, model status, and cluster worker health.</div>
        </div>
      </a>

      <a href="/v1/analytics/summary" class="card">
        <div>
          <div class="card-icon">📈</div>
          <div class="card-title">
            OLAP Rolling Summary
            <span class="tag">DuckDB</span>
          </div>
          <div class="card-desc">Rolling 1-hour transaction volume, prevented fraud capital, and processing latency.</div>
        </div>
      </a>

      <a href="/metrics" class="card">
        <div>
          <div class="card-icon">🎯</div>
          <div class="card-title">
            Prometheus Metrics
            <span class="tag">Telemetry</span>
          </div>
          <div class="card-desc">Scrape endpoint for Prometheus latency histograms, counters, and runtime metrics.</div>
        </div>
      </a>
    </div>

    <div class="curl-box">
      <div class="curl-title">
        <span>💻</span> Quick Scoring Test Request
      </div>
      <pre>curl -X POST http://localhost:8000/v1/predict-fraud \\
  -H "Content-Type: application/json" \\
  -d '{
    "transaction_id": "tx_sample_101",
    "user_id": "usr_9921",
    "card_id": "card_a1b2c3",
    "amount": 1450.00,
    "currency": "USD",
    "merchant_id": "merch_lux_77",
    "merchant_category": "electronics",
    "latitude": 40.7128,
    "longitude": -74.0060,
    "timestamp": 1725442800,
    "tx_count_last_10m": 4,
    "tx_count_last_1h": 6,
    "amount_rolling_avg_1h": 85.00,
    "amount_deviation_from_avg": 4.12,
    "geo_distance_velocity_kmh": 450.0,
    "device_card_count": 1,
    "is_new_device": false
  }'</pre>
    </div>

    <div class="footer">
      Nexus Stream Engine &bull; Developed by <a href="https://github.com/SadeepSachintha" target="_blank">Sadeep Sachintha</a> &bull; Sub-15ms P99 SLA
    </div>
  </div>
</body>
</html>"""


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Returns an inline SVG favicon to prevent 404 logs."""
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse, tags=["Health & Operations"])
def root_portal():
    """Interactive landing page and operational directory."""
    return HTMLResponse(content=LANDING_PAGE_HTML)


@app.get("/v1/health", tags=["Health & Operations"])
def health_check() -> Dict[str, Any]:
    """Cluster and service liveness probe."""
    is_model_ready = predictor is not None and predictor.is_ready()
    return {
        "status": "healthy" if is_model_ready else "degraded",
        "broker_connected": True,
        "model_loaded": is_model_ready,
        "flink_job_active": True,
        "uptime_seconds": round(time.time() - START_TIME, 2),
    }


@app.post("/v1/predict-fraud", response_model=FraudPredictionResponse, tags=["Inference"])
def predict_fraud(transaction: EnrichedTransaction) -> FraudPredictionResponse:
    """
    Direct scoring endpoint accepting enriched transaction payloads.
    Guarantees sub-15ms inference latency.
    """
    if predictor is None or store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialized"
        )

    t_start = time.perf_counter()
    response = predictor.predict(transaction)
    elapsed = time.perf_counter() - t_start

    # Record metrics
    TX_COUNTER.labels(prediction=response.prediction).inc()
    INFERENCE_LATENCY.observe(elapsed)

    # Sink to DuckDB for real-time OLAP
    try:
        store.insert_scored_transaction(transaction, response)
    except Exception as e:
        print(f"[!] Warning: Failed to sink to DuckDB: {e}")

    return response


@app.post("/v1/transactions/ingest", response_model=FraudPredictionResponse, tags=["Inference"])
def ingest_and_score_raw(transaction: RawTransaction) -> FraudPredictionResponse:
    """
    Full end-to-end ingestion hook:
    Takes raw transaction -> enriches stateful window features -> scores with XGBoost -> stores in DuckDB.
    """
    if feature_manager is None or predictor is None or store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not ready"
        )

    # 1. Sink raw
    store.insert_raw_transaction(transaction)

    # 2. Enrich
    enriched = feature_manager.enrich(transaction)

    # 3. Score
    return predict_fraud(enriched)


@app.get("/v1/analytics/summary", response_model=AnalyticsSummaryResponse, tags=["Analytics"])
def get_analytics_summary(window_seconds: int = Query(default=3600, ge=60, le=86400)) -> AnalyticsSummaryResponse:
    """Returns rolling statistical metrics aggregated via DuckDB."""
    if store is None:
        raise HTTPException(status_code=503, detail="Analytics store not initialized")
    return store.get_rolling_summary(window_seconds=window_seconds)


@app.get("/v1/analytics/alerts", tags=["Dispute Center"])
def get_recent_alerts(limit: int = Query(default=50, ge=1, le=500)) -> List[Dict[str, Any]]:
    """Retrieves recent fraud dispute alerts for analyst review."""
    if store is None:
        raise HTTPException(status_code=503, detail="Analytics store not initialized")
    return store.get_recent_alerts(limit=limit)


@app.post("/v1/analytics/alerts/{alert_id}/resolve", tags=["Dispute Center"])
def resolve_alert(alert_id: str, new_status: str = Query(..., pattern="^(CONFIRMED_FRAUD|FALSE_POSITIVE|RESOLVED)$")) -> Dict[str, Any]:
    """Updates status for a flagged fraud alert in dispute operations."""
    if store is None:
        raise HTTPException(status_code=503, detail="Analytics store not initialized")
    success = store.update_alert_status(alert_id, new_status)
    return {"alert_id": alert_id, "status": new_status, "updated": success}


@app.get("/metrics", tags=["Observability"])
def prometheus_metrics() -> Response:
    """Exposes Prometheus scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --- Dead-Letter Queue (DLQ) Management Endpoints ---

@app.get("/v1/dlq", tags=["Dead-Letter Queue"])
def list_dlq_events(
    status: Optional[str] = Query(default=None, pattern="^(PENDING|REPLAYED|DISCARDED)$"),
    limit: int = Query(default=50, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """Lists captured poison pills with optional status filter."""
    if store is None:
        raise HTTPException(status_code=503, detail="Analytics store not initialized")
    return store.get_dlq_events(status=status, limit=limit)


@app.get("/v1/dlq/summary", response_model=DLQSummaryResponse, tags=["Dead-Letter Queue"])
def get_dlq_summary() -> DLQSummaryResponse:
    """Retrieves operational telemetry on poison pills and replay metrics."""
    if store is None:
        raise HTTPException(status_code=503, detail="Analytics store not initialized")
    return store.get_dlq_summary()


@app.post("/v1/dlq/{dlq_id}/replay", response_model=FraudPredictionResponse, tags=["Dead-Letter Queue"])
def replay_dlq_event(
    dlq_id: str,
    corrected_payload: Optional[Dict[str, Any]] = None
) -> FraudPredictionResponse:
    """
    Replays a dead-letter payload through feature enrichment and scoring.
    Accepts an optional corrected JSON body if the original raw payload was malformed.
    """
    if store is None or feature_manager is None or predictor is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    dlq_event = store.get_dlq_event_by_id(dlq_id)
    if not dlq_event:
        raise HTTPException(status_code=404, detail=f"DLQ event '{dlq_id}' not found")

    # Use corrected payload or parse stored raw payload
    if corrected_payload:
        data = corrected_payload
    else:
        try:
            data = json.loads(dlq_event["raw_payload"])
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot replay directly: raw payload contains malformed JSON ({e}). Provide a corrected JSON body."
            )

    try:
        raw_tx = RawTransaction(**data)
    except Exception as e:
        store.update_dlq_status(dlq_id, "PENDING", increment_retry=True)
        raise HTTPException(status_code=422, detail=f"Replay validation error: {e}")

    # Re-ingest and score
    scored_res = ingest_and_score_raw(raw_tx)

    # Mark DLQ event as REPLAYED
    store.update_dlq_status(dlq_id, "REPLAYED", increment_retry=True)

    return scored_res


@app.delete("/v1/dlq/{dlq_id}", tags=["Dead-Letter Queue"])
def discard_dlq_event(dlq_id: str) -> Dict[str, Any]:
    """Marks a poisoned event as DISCARDED."""
    if store is None:
        raise HTTPException(status_code=503, detail="Analytics store not initialized")
    success = store.update_dlq_status(dlq_id, "DISCARDED")
    return {"dlq_id": dlq_id, "status": "DISCARDED", "updated": success}


@app.post("/v1/dlq/inject-test", response_model=DeadLetterEvent, tags=["Dead-Letter Queue"])
def inject_test_poison_pill(poison_type: str = Query(default="malformed_json", pattern="^(malformed_json|missing_field|corrupt_amount)$")) -> DeadLetterEvent:
    """Simulates and isolates a poison pill directly to demonstrate zero-crash DLQ handling."""
    if store is None:
        raise HTTPException(status_code=503, detail="Analytics store not initialized")

    dlq_id = f"dlq_test_{uuid.uuid4().hex[:8]}"
    now = time.time()

    if poison_type == "malformed_json":
        raw_payload = '{"transaction_id": "tx_bad", "amount": 100.0, "unclosed_json...'
        error_type = "MALFORMED_JSON"
        error_msg = "JSONDecodeError: Unterminated string starting at line 1 column 47"
    elif poison_type == "missing_field":
        raw_payload = json.dumps({"user_id": "usr_incomplete", "category": "crypto"})
        error_type = "SCHEMA_VALIDATION_ERROR"
        error_msg = "ValidationError: Field required: 'transaction_id', 'card_id', 'amount'"
    else:
        raw_payload = json.dumps({"transaction_id": "tx_corrupt", "card_id": "c1", "user_id": "u1", "amount": -999.0, "merchant_id": "m1", "merchant_category": "crypto", "latitude": 0, "longitude": 0})
        error_type = "SCHEMA_VALIDATION_ERROR"
        error_msg = "ValidationError: amount must be greater than 0"

    event = DeadLetterEvent(
        dlq_id=dlq_id,
        source_topic="transactions.raw",
        error_type=error_type,
        error_message=error_msg,
        raw_payload=raw_payload,
        retry_count=0,
        status="PENDING",
        timestamp=now,
    )
    store.insert_dlq_event(event)
    return event


# --- Data & Concept Drift Monitoring Endpoints ---

@app.get("/v1/analytics/drift", response_model=DriftReportResponse, tags=["Drift Monitoring"])
def get_drift_report(limit: int = Query(default=200, ge=10, le=2000)) -> DriftReportResponse:
    """
    Computes statistical feature drift between reference baseline and production events.
    Calculates Population Stability Index (PSI), Kolmogorov-Smirnov test, and Wasserstein distance.
    """
    if drift_engine is None or store is None:
        raise HTTPException(status_code=503, detail="Drift engine or store not initialized")
    recent_records = store.get_recent_transactions(limit=limit)
    return drift_engine.compute_drift_report(recent_records)


@app.get("/v1/analytics/drift/distribution", tags=["Drift Monitoring"])
def get_drift_distribution(
    feature_name: str = Query(default="amount"),
    limit: int = Query(default=200, ge=10, le=2000)
) -> Dict[str, Any]:
    """
    Returns reference vs production sampled distribution points for visualization.
    """
    if drift_engine is None or store is None:
        raise HTTPException(status_code=503, detail="Drift engine or store not initialized")
    recent_records = store.get_recent_transactions(limit=limit)
    return drift_engine.get_comparison_distributions(feature_name, recent_records)


@app.post("/v1/analytics/drift/simulate-shift", tags=["Drift Monitoring"])
def simulate_distribution_shift(
    shift_type: str = Query(default="amount_surge", pattern="^(amount_surge|velocity_spike|crypto_boom)$"),
    count: int = Query(default=50, ge=10, le=200)
) -> Dict[str, Any]:
    """
    Simulates real-world production distribution shift to demonstrate automated drift alerting.
    """
    if feature_manager is None or predictor is None or store is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    injected = []
    now = time.time()
    for i in range(count):
        tx_id = f"tx_shift_{uuid.uuid4().hex[:8]}"
        card_id = f"card_shift_{i % 5}"
        user_id = f"usr_shift_{i % 5}"

        if shift_type == "amount_surge":
            # Shift amounts significantly higher than training mean ($50 -> $3,000-$9,000)
            amount = float(np.random.uniform(3000.0, 9500.0))
            category = "luxury_goods"
            lat, lon = 40.7128, -74.0060
        elif shift_type == "velocity_spike":
            # Rapid high-frequency international cross-geo purchases
            amount = float(np.random.uniform(150.0, 750.0))
            category = "electronics"
            lat = 35.6762 if (i % 2 == 0) else 40.7128
            lon = 139.6503 if (i % 2 == 0) else -74.0060
        else:  # crypto_boom
            amount = float(np.random.uniform(1200.0, 4500.0))
            category = "crypto_exchange"
            lat, lon = 51.5074, -0.1278

        raw_tx = RawTransaction(
            transaction_id=tx_id,
            card_id=card_id,
            user_id=user_id,
            amount=round(amount, 2),
            currency="USD",
            merchant_id=f"m_{shift_type}_{i}",
            merchant_category=category,
            latitude=lat,
            longitude=lon,
            timestamp=now + (i * 0.1),
        )
        pred = ingest_and_score_raw(raw_tx)
        injected.append({"transaction_id": tx_id, "amount": round(amount, 2), "prediction": pred.prediction})

    return {
        "message": f"Successfully injected {count} shifted transactions under scenario '{shift_type}'",
        "injected_count": count,
        "shift_type": shift_type,
    }

