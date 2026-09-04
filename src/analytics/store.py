import duckdb
import os
import queue
import threading
import time
import sys
import atexit
import weakref
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

_ACTIVE_STORES = weakref.WeakSet()


def _cleanup_all_stores():
    """Teardown hook ensuring all background DuckDB writer threads are joined before interpreter exit."""
    stores = list(_ACTIVE_STORES)
    for s in stores:
        try:
            s.close()
        except Exception:
            pass


atexit.register(_cleanup_all_stores)


class DuckDBAnalyticsStore:
    """
    Embedded in-memory / file-backed columnar OLAP analytical store using DuckDB.
    Supports high-velocity transaction sinks and sub-millisecond aggregate queries
    for dashboards and dispute centers without locking the stream.
    """

    def __init__(self, db_path: str = "nexus_analytics.duckdb"):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._raw_queue: queue.Queue = queue.Queue(maxsize=100000)
        self._scored_queue: queue.Queue = queue.Queue(maxsize=100000)
        self._running = True
        self._initialized = False
        self._init_db()
        self._initialized = True
        self._writer_thread = threading.Thread(target=self._background_flush_loop, daemon=True)
        self._writer_thread.start()
        _ACTIVE_STORES.add(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Gracefully halts the background flush thread and executes final queue sync."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        if hasattr(self, "_writer_thread") and self._writer_thread.is_alive():
            if threading.current_thread() != self._writer_thread:
                try:
                    self._writer_thread.join(timeout=0.5)
                except Exception:
                    pass

        try:
            self.flush()
        except Exception:
            pass

    def _raw_connect(self) -> duckdb.DuckDBPyConnection:
        if getattr(sys, "is_finalizing", lambda: False)():
            raise RuntimeError("Python interpreter is shutting down")
        return duckdb.connect(self.db_path)

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        if self._initialized:
            self.flush()
        return self._raw_connect()

    def flush(self) -> None:
        if not self._initialized or getattr(sys, "is_finalizing", lambda: False)():
            return

        raw_items = []
        while not self._raw_queue.empty() and len(raw_items) < 1000:
            try:
                raw_items.append(self._raw_queue.get_nowait())
            except queue.Empty:
                break

        scored_items = []
        alerts = []
        while not self._scored_queue.empty() and len(scored_items) < 1000:
            try:
                sc_item, alt_item = self._scored_queue.get_nowait()
                scored_items.append(sc_item)
                if alt_item:
                    alerts.append(alt_item)
            except queue.Empty:
                break

        if raw_items or scored_items or alerts:
            with self._lock:
                if getattr(sys, "is_finalizing", lambda: False)():
                    return
                conn = self._raw_connect()
                try:
                    if raw_items:
                        conn.executemany("""
                            INSERT OR REPLACE INTO transactions_raw (
                                transaction_id, user_id, card_id, amount, currency,
                                merchant_id, merchant_category, latitude, longitude, timestamp
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, raw_items)
                        for _ in raw_items:
                            self._raw_queue.task_done()

                    if scored_items:
                        conn.executemany("""
                            INSERT OR REPLACE INTO transactions_scored (
                                transaction_id, user_id, card_id, amount, currency,
                                merchant_category, tx_count_last_10m, tx_count_last_1h,
                                amount_rolling_avg_1h, amount_deviation_from_avg,
                                geo_distance_velocity_kmh, is_night_transaction,
                                category_risk_index, fraud_probability, prediction,
                                action, latency_ms, top_risk_factors, timestamp
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, scored_items)
                        for _ in scored_items:
                            self._scored_queue.task_done()

                    if alerts:
                        conn.executemany("""
                            INSERT OR REPLACE INTO fraud_alerts (
                                alert_id, transaction_id, user_id, card_id, amount,
                                merchant_category, fraud_probability, prediction,
                                action, top_risk_factors, status, timestamp
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'UNREVIEWED', ?)
                        """, alerts)
                finally:
                    conn.close()

    def _background_flush_loop(self) -> None:
        while self._running:
            try:
                if getattr(sys, "is_finalizing", lambda: False)():
                    break
                time.sleep(0.05)
                if not self._running or getattr(sys, "is_finalizing", lambda: False)():
                    break
                self.flush()
            except Exception:
                break

    def _init_db(self) -> None:
        with self._lock:
            conn = self._raw_connect()
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
                        tx_count_last_1h INTEGER DEFAULT 1,
                        amount_rolling_avg_1h DOUBLE DEFAULT 0.0,
                        amount_deviation_from_avg DOUBLE,
                        geo_distance_velocity_kmh DOUBLE,
                        is_night_transaction INTEGER DEFAULT 0,
                        category_risk_index DOUBLE DEFAULT 0.1,
                        fraud_probability DOUBLE,
                        prediction VARCHAR,
                        action VARCHAR,
                        latency_ms DOUBLE,
                        top_risk_factors VARCHAR,
                        timestamp DOUBLE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                for col_name, col_type in [
                    ("tx_count_last_1h", "INTEGER DEFAULT 1"),
                    ("amount_rolling_avg_1h", "DOUBLE DEFAULT 0.0"),
                    ("is_night_transaction", "INTEGER DEFAULT 0"),
                    ("category_risk_index", "DOUBLE DEFAULT 0.1"),
                ]:
                    try:
                        conn.execute(f"ALTER TABLE transactions_scored ADD COLUMN IF NOT EXISTS {col_name} {col_type};")
                    except Exception:
                        pass

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
        """Enqueues raw transaction into asynchronous batch writer for non-blocking OLAP persistence."""
        self._raw_queue.put((
            tx.transaction_id, tx.user_id, tx.card_id, tx.amount, tx.currency,
            tx.merchant_id, tx.merchant_category, tx.latitude, tx.longitude, tx.timestamp
        ))

    def insert_scored_transaction(
        self,
        tx: EnrichedTransaction,
        pred: FraudPredictionResponse
    ) -> None:
        """Enqueues scored event and alert triggers into asynchronous batch writer."""
        risk_factors_str = json.dumps(pred.top_risk_factors)
        sc_tuple = (
            tx.transaction_id, tx.user_id, tx.card_id, tx.amount, tx.currency,
            tx.merchant_category, tx.tx_count_last_10m, getattr(tx, "tx_count_last_1h", 1),
            getattr(tx, "amount_rolling_avg_1h", tx.amount), tx.amount_deviation_from_avg,
            tx.geo_distance_velocity_kmh, 1 if getattr(tx, "is_night_transaction", False) else 0,
            getattr(tx, "category_risk_index", 0.1), pred.fraud_probability, pred.prediction,
            pred.action, pred.latency_ms, risk_factors_str, tx.timestamp
        )

        alt_tuple = None
        if pred.prediction in ("FRAUD_SUSPECTED", "REVIEW_REQUIRED"):
            alert_id = f"alt_{tx.transaction_id}"
            alt_tuple = (
                alert_id, tx.transaction_id, tx.user_id, tx.card_id, tx.amount,
                tx.merchant_category, pred.fraud_probability, pred.prediction,
                pred.action, risk_factors_str, tx.timestamp
            )

        self._scored_queue.put((sc_tuple, alt_tuple))

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
                    SELECT *
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

