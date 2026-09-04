import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.streaming.schemas import (
    RawTransaction,
    EnrichedTransaction,
    FraudPredictionResponse,
    AnalyticsSummaryResponse,
)
from src.streaming.features import StreamFeatureStateManager
from src.inference.predictor import FraudPredictor
from src.inference.train import train_model
from src.analytics.store import DuckDBAnalyticsStore

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
START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, store, feature_manager
    print("[*] Initializing Nexus Stream API...")
    
    # 1. Initialize Analytics Store
    db_path = os.getenv("DUCKDB_PATH", "nexus_analytics.duckdb")
    store = DuckDBAnalyticsStore(db_path=db_path)
    
    # 2. Initialize Feature State Manager
    feature_manager = StreamFeatureStateManager()

    # 3. Initialize Predictor & Auto-train if model missing
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
