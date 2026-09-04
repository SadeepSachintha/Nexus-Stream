import duckdb
import os
import threading
import time
from typing import Any, Dict, List, Optional
import json

from src.streaming.schemas import (
    RawTransaction,
    EnrichedTransaction,
    FraudPredictionResponse,
    AnalyticsSummaryResponse,
    DeadLetterEvent,
    DLQSummaryResponse,
)


class DuckDBAnalyticsStore:
    """
    Embedded in-memory / file-backed columnar OLAP analytical store using DuckDB.
    Supports high-velocity transaction sinks and sub-millisecond aggregate queries
    for dashboards and dispute centers without locking the stream.
    """

    def __init__(self, db_path: str = "nexus_analytics.duckdb"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self.db_path)

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                # 1. Raw transactions stream table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS transactions_raw (
                        transaction_id VARCHAR PRIMARY KEY,
                        user_id VARCHAR,
                        card_id VARCHAR,
                        amount DOUBLE,
                        currency VARCHAR,
                        merchant_id VARCHAR,
                        merchant_category VARCHAR,
                        latitude DOUBLE,
                        longitude DOUBLE,
                        timestamp DOUBLE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # 2. Scored and enriched transactions table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS transactions_scored (
                        transaction_id VARCHAR PRIMARY KEY,
                        user_id VARCHAR,
                        card_id VARCHAR,
                        amount DOUBLE,
                        currency VARCHAR,
                        merchant_category VARCHAR,
                        tx_count_last_10m INTEGER,
                        amount_deviation_from_avg DOUBLE,
                        geo_distance_velocity_kmh DOUBLE,
                        fraud_probability DOUBLE,
                        prediction VARCHAR,
                        action VARCHAR,
                        latency_ms DOUBLE,
                        top_risk_factors VARCHAR,
                        timestamp DOUBLE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # 3. Dedicated fraud alert & dispute triage table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS fraud_alerts (
                        alert_id VARCHAR PRIMARY KEY,
                        transaction_id VARCHAR,
                        user_id VARCHAR,
                        card_id VARCHAR,
                        amount DOUBLE,
                        merchant_category VARCHAR,
                        fraud_probability DOUBLE,
                        prediction VARCHAR,
                        action VARCHAR,
                        top_risk_factors VARCHAR,
                        status VARCHAR DEFAULT 'UNREVIEWED',
                        timestamp DOUBLE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # 4. Dead-Letter Queue (DLQ) table for poison pills
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS dlq_events (
                        dlq_id VARCHAR PRIMARY KEY,
                        source_topic VARCHAR,
                        error_type VARCHAR,
                        error_message VARCHAR,
                        raw_payload VARCHAR,
                        retry_count INTEGER DEFAULT 0,
                        status VARCHAR DEFAULT 'PENDING',
                        timestamp DOUBLE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            finally:
                conn.close()

    def insert_raw_transaction(self, tx: RawTransaction) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO transactions_raw (
                        transaction_id, user_id, card_id, amount, currency,
                        merchant_id, merchant_category, latitude, longitude, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tx.transaction_id, tx.user_id, tx.card_id, tx.amount, tx.currency,
                    tx.merchant_id, tx.merchant_category, tx.latitude, tx.longitude, tx.timestamp
                ))
            finally:
                conn.close()

    def insert_scored_transaction(
        self,
        tx: EnrichedTransaction,
        pred: FraudPredictionResponse
    ) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                risk_factors_str = json.dumps(pred.top_risk_factors)
                conn.execute("""
                    INSERT OR REPLACE INTO transactions_scored (
                        transaction_id, user_id, card_id, amount, currency,
                        merchant_category, tx_count_last_10m, amount_deviation_from_avg,
                        geo_distance_velocity_kmh, fraud_probability, prediction,
                        action, latency_ms, top_risk_factors, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tx.transaction_id, tx.user_id, tx.card_id, tx.amount, tx.currency,
                    tx.merchant_category, tx.tx_count_last_10m, tx.amount_deviation_from_avg,
                    tx.geo_distance_velocity_kmh, pred.fraud_probability, pred.prediction,
                    pred.action, pred.latency_ms, risk_factors_str, tx.timestamp
                ))

                # If high risk or review required, register in alert table
                if pred.prediction in ("FRAUD_SUSPECTED", "REVIEW_REQUIRED"):
                    alert_id = f"alt_{tx.transaction_id}"
                    conn.execute("""
                        INSERT OR REPLACE INTO fraud_alerts (
                            alert_id, transaction_id, user_id, card_id, amount,
                            merchant_category, fraud_probability, prediction,
                            action, top_risk_factors, status, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'UNREVIEWED', ?)
                    """, (
                        alert_id, tx.transaction_id, tx.user_id, tx.card_id, tx.amount,
                        tx.merchant_category, pred.fraud_probability, pred.prediction,
                        pred.action, risk_factors_str, tx.timestamp
                    ))
            finally:
                conn.close()

    def get_rolling_summary(self, window_seconds: int = 3600) -> AnalyticsSummaryResponse:
        """Computes rolling aggregate statistics for the API and dashboard."""
        with self._lock:
            conn = self._get_connection()
            try:
                now = time.time()
                cutoff = now - window_seconds
                query = """
                    SELECT 
                        COUNT(*) as total_count,
                        COALESCE(SUM(CASE WHEN prediction = 'FRAUD_SUSPECTED' THEN 1 ELSE 0 END), 0) as fraud_count,
                        COALESCE(SUM(CASE WHEN prediction = 'REVIEW_REQUIRED' THEN 1 ELSE 0 END), 0) as review_count,
                        COALESCE(AVG(latency_ms), 0.0) as avg_latency,
                        COALESCE(SUM(CASE WHEN prediction = 'FRAUD_SUSPECTED' THEN amount ELSE 0.0 END), 0.0) as prevented_loss
                    FROM transactions_scored
                    WHERE timestamp >= ?
                """
                res = conn.execute(query, (cutoff,)).fetchone()
                if not res or res[0] == 0:
                    # Fallback to all-time if window is empty during startup
                    fallback_query = """
                        SELECT 
                            COUNT(*) as total_count,
                            COALESCE(SUM(CASE WHEN prediction = 'FRAUD_SUSPECTED' THEN 1 ELSE 0 END), 0) as fraud_count,
                            COALESCE(SUM(CASE WHEN prediction = 'REVIEW_REQUIRED' THEN 1 ELSE 0 END), 0) as review_count,
                            COALESCE(AVG(latency_ms), 0.0) as avg_latency,
                            COALESCE(SUM(CASE WHEN prediction = 'FRAUD_SUSPECTED' THEN amount ELSE 0.0 END), 0.0) as prevented_loss
                        FROM transactions_scored
                    """
                    res = conn.execute(fallback_query).fetchone()

                total_tx = int(res[0]) if res and res[0] else 0
                fraud_tx = int(res[1]) if res and res[1] else 0
                review_tx = int(res[2]) if res and res[2] else 0
                avg_latency = float(res[3]) if res and res[3] else 0.0
                prevented_loss = float(res[4]) if res and res[4] else 0.0

                fraud_rate = (fraud_tx / total_tx * 100.0) if total_tx > 0 else 0.0

                return AnalyticsSummaryResponse(
                    window=f"rolling_{window_seconds // 3600}h" if window_seconds >= 3600 else f"rolling_{window_seconds}s",
                    total_transactions=total_tx,
                    flagged_fraudulent=fraud_tx,
                    review_required=review_tx,
                    fraud_rate_percentage=round(fraud_rate, 3),
                    avg_processing_latency_ms=round(avg_latency, 2),
                    prevented_capital_loss_usd=round(prevented_loss, 2),
                )
            finally:
                conn.close()

    def get_recent_transactions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns the most recent scored transactions as a list of dicts."""
        with self._lock:
            conn = self._get_connection()
            try:
                df = conn.execute("""
                    SELECT 
                        transaction_id, user_id, card_id, amount, currency,
                        merchant_category, tx_count_last_10m, amount_deviation_from_avg,
                        geo_distance_velocity_kmh, fraud_probability, prediction,
                        action, latency_ms, top_risk_factors, timestamp
                    FROM transactions_scored
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,)).fetchdf()
                return df.to_dict(orient="records")
            finally:
                conn.close()

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns the most recent fraud dispute alerts."""
        with self._lock:
            conn = self._get_connection()
            try:
                df = conn.execute("""
                    SELECT * FROM fraud_alerts
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,)).fetchdf()
                return df.to_dict(orient="records")
            finally:
                conn.close()

    def update_alert_status(self, alert_id: str, new_status: str) -> bool:
        """Updates review status of a dispute alert (e.g., 'CONFIRMED_FRAUD', 'FALSE_POSITIVE', 'RESOLVED')."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("""
                    UPDATE fraud_alerts
                    SET status = ?
                    WHERE alert_id = ?
                """, (new_status, alert_id))
                return True
            finally:
                conn.close()

    def get_category_risk_breakdown(self) -> List[Dict[str, Any]]:
        """Returns aggregated transaction metrics by merchant category."""
        with self._lock:
            conn = self._get_connection()
            try:
                df = conn.execute("""
                    SELECT 
                        merchant_category,
                        COUNT(*) as total_count,
                        COALESCE(SUM(CASE WHEN prediction = 'FRAUD_SUSPECTED' THEN 1 ELSE 0 END), 0) as fraud_count,
                        ROUND(AVG(amount), 2) as avg_amount,
                        ROUND(AVG(fraud_probability), 4) as avg_risk_score
                    FROM transactions_scored
                    GROUP BY merchant_category
                    ORDER BY fraud_count DESC, total_count DESC
                """).fetchdf()
                return df.to_dict(orient="records")
            finally:
                conn.close()

    # --- Dead-Letter Queue (DLQ) Operations ---

    def insert_dlq_event(self, event: DeadLetterEvent) -> None:
        """Sinks a poisoned or malformed payload into the DLQ table."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO dlq_events (
                        dlq_id, source_topic, error_type, error_message,
                        raw_payload, retry_count, status, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.dlq_id, event.source_topic, event.error_type,
                    event.error_message, event.raw_payload, event.retry_count,
                    event.status, event.timestamp
                ))
            finally:
                conn.close()

    def get_dlq_events(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves dead-letter events with optional status filtering (PENDING, REPLAYED, DISCARDED)."""
        with self._lock:
            conn = self._get_connection()
            try:
                if status:
                    df = conn.execute("""
                        SELECT * FROM dlq_events
                        WHERE status = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (status, limit)).fetchdf()
                else:
                    df = conn.execute("""
                        SELECT * FROM dlq_events
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (limit,)).fetchdf()
                return df.to_dict(orient="records")
            finally:
                conn.close()

    def get_dlq_event_by_id(self, dlq_id: str) -> Optional[Dict[str, Any]]:
        """Finds a single DLQ event by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                res = conn.execute("SELECT * FROM dlq_events WHERE dlq_id = ?", (dlq_id,)).fetchdf()
                if not res.empty:
                    return res.to_dict(orient="records")[0]
                return None
            finally:
                conn.close()

    def update_dlq_status(self, dlq_id: str, new_status: str, increment_retry: bool = False) -> bool:
        """Updates DLQ record status and optionally increments the retry counter."""
        with self._lock:
            conn = self._get_connection()
            try:
                if increment_retry:
                    conn.execute("""
                        UPDATE dlq_events
                        SET status = ?, retry_count = retry_count + 1
                        WHERE dlq_id = ?
                    """, (new_status, dlq_id))
                else:
                    conn.execute("""
                        UPDATE dlq_events
                        SET status = ?
                        WHERE dlq_id = ?
                    """, (new_status, dlq_id))
                return True
            finally:
                conn.close()

    def get_dlq_summary(self) -> DLQSummaryResponse:
        """Returns aggregate metrics on poison pills and replay status."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = """
                    SELECT 
                        COUNT(*) as total_events,
                        COALESCE(SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END), 0) as pending_cnt,
                        COALESCE(SUM(CASE WHEN status = 'REPLAYED' THEN 1 ELSE 0 END), 0) as replayed_cnt,
                        COALESCE(SUM(CASE WHEN status = 'DISCARDED' THEN 1 ELSE 0 END), 0) as discarded_cnt
                    FROM dlq_events
                """
                res = conn.execute(query).fetchone()
                return DLQSummaryResponse(
                    total_dlq_events=int(res[0]) if res and res[0] else 0,
                    pending_events=int(res[1]) if res and res[1] else 0,
                    replayed_events=int(res[2]) if res and res[2] else 0,
                    discarded_events=int(res[3]) if res and res[3] else 0,
                )
            finally:
                conn.close()

