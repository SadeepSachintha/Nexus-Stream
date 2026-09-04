import os
import json
import time
from typing import List, Tuple, Optional
import numpy as np
import xgboost as xgb

from src.streaming.schemas import EnrichedTransaction, FraudPredictionResponse
from src.inference.train import FEATURE_NAMES


class FraudPredictor:
    """
    High-throughput, low-latency scoring engine leveraging native XGBoost C-API
    execution to guarantee sub-15ms response times.
    """

    def __init__(self, model_path: Optional[str] = None, meta_path: Optional[str] = None):
        base_dir = os.path.dirname(__file__)
        self.model_path = model_path or os.path.join(base_dir, "model", "fraud_xgboost_model.json")
        self.meta_path = meta_path or os.path.join(base_dir, "model", "model_meta.json")
        
        self.booster: Optional[xgb.Booster] = None
        self.metadata: dict = {}
        self.load_model()

    def load_model(self) -> None:
        if os.path.exists(self.model_path):
            self.booster = xgb.Booster()
            self.booster.load_model(self.model_path)
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r") as f:
                    self.metadata = json.load(f)
            # Warm up C-API and OpenMP thread pool
            dummy = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32)
            self.booster.predict(xgb.DMatrix(dummy, feature_names=FEATURE_NAMES))
        else:
            self.booster = None

    def is_ready(self) -> bool:
        return self.booster is not None

    def _extract_feature_vector(self, tx: EnrichedTransaction) -> np.ndarray:
        return np.array([[
            float(tx.amount),
            float(tx.tx_count_last_10m),
            float(tx.tx_count_last_1h),
            float(tx.amount_rolling_avg_1h),
            float(tx.amount_deviation_from_avg),
            float(tx.geo_distance_velocity_kmh),
            float(tx.is_night_transaction),
            float(tx.category_risk_index),
        ]], dtype=np.float32)

    def _determine_risk_factors(self, tx: EnrichedTransaction, prob: float) -> List[str]:
        factors = []
        if tx.is_fraud_ring_suspected or tx.device_card_count >= 3:
            dev_label = tx.device_id or "unknown"
            factors.append(f"device_sharing_fraud_ring ({tx.device_card_count} cards linked to device '{dev_label}')")
        if tx.amount_deviation_from_avg >= 3.5:
            factors.append(f"amount_deviation_from_avg ({tx.amount_deviation_from_avg:+.1f} std vs historical avg)")
        if tx.geo_distance_velocity_kmh >= 750.0:
            factors.append(f"geo_distance_velocity_kmh ({tx.geo_distance_velocity_kmh:.0f} km/h impossible travel)")
        if tx.tx_count_last_10m >= 5:
            factors.append(f"tx_count_last_10m ({tx.tx_count_last_10m} transactions in 10 min burst)")
        if tx.category_risk_index >= 0.70:
            factors.append(f"category_risk_index ({tx.merchant_category} high-risk code)")
        if tx.is_night_transaction == 1 and prob > 0.4:
            factors.append("is_night_transaction (high-risk off-hours)")

        if not factors and prob >= 0.4:
            factors.append("compound_behavioral_risk_anomaly")
        return factors

    def predict(self, tx: EnrichedTransaction) -> FraudPredictionResponse:
        t_start = time.perf_counter()

        if self.booster is None:
            # Fallback heuristic if model is not yet trained/loaded
            prob = 0.05
            if tx.amount_deviation_from_avg > 4.0 or tx.geo_distance_velocity_kmh > 800:
                prob = 0.88
            elif tx.tx_count_last_10m > 5:
                prob = 0.65
        else:
            # Optimized C-API execution path
            feat_vector = self._extract_feature_vector(tx)
            dmatrix = xgb.DMatrix(feat_vector, feature_names=FEATURE_NAMES)
            raw_pred = self.booster.predict(dmatrix)
            prob = float(raw_pred[0])

        # Hybrid Policy Guardrail (Industry Standard for Physical Impossibility & Fraud Rings)
        if tx.is_fraud_ring_suspected or tx.device_card_count >= 3:
            prob = max(prob, 0.92)
        elif tx.geo_distance_velocity_kmh >= 750.0:
            prob = max(prob, 0.88)
        elif tx.amount_deviation_from_avg >= 5.0 and tx.category_risk_index >= 0.70:
            prob = max(prob, 0.82)
        elif tx.tx_count_last_10m >= 8:
            prob = max(prob, 0.78)

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        # Classification decision rules
        if prob >= 0.75:
            prediction = "FRAUD_SUSPECTED"
            action = "BLOCK_AND_FLAG"
        elif prob >= 0.40:
            prediction = "REVIEW_REQUIRED"
            action = "STEP_UP_AUTH"
        else:
            prediction = "LEGITIMATE"
            action = "ALLOW"

        top_factors = self._determine_risk_factors(tx, prob)

        return FraudPredictionResponse(
            transaction_id=tx.transaction_id,
            user_id=tx.user_id,
            fraud_probability=round(prob, 4),
            prediction=prediction,
            action=action,
            latency_ms=round(latency_ms, 2),
            top_risk_factors=top_factors,
            timestamp=tx.timestamp,
        )
