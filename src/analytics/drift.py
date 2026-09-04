import json
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.streaming.schemas import FeatureDriftMetric, DriftReportResponse
from src.inference.train import FEATURE_NAMES, generate_synthetic_fraud_dataset


class StatisticalDriftEngine:
    """
    Sub-second statistical data & concept drift monitoring engine.
    Computes Population Stability Index (PSI), Two-Sample Kolmogorov-Smirnov (KS) test,
    and Wasserstein Distance between baseline training distributions and live streaming data.
    """

    def __init__(self, reference_path: Optional[str] = None):
        base_dir = os.path.join(os.path.dirname(__file__), "..", "inference", "model")
        self.reference_path = reference_path or os.path.join(base_dir, "baseline_reference.json")
        self.reference_data: Dict[str, np.ndarray] = {}
        self._load_or_generate_reference()

    def _load_or_generate_reference(self) -> None:
        """Loads reference baseline distribution or synthesizes initial ground truth."""
        if os.path.exists(self.reference_path):
            with open(self.reference_path, "r") as f:
                raw_dict = json.load(f)
                for feat in FEATURE_NAMES:
                    if feat in raw_dict:
                        self.reference_data[feat] = np.array(raw_dict[feat], dtype=np.float32)

        if not self.reference_data or "amount" not in self.reference_data:
            # Fallback: generate baseline sample immediately
            df = generate_synthetic_fraud_dataset(n_samples=2000, fraud_rate=0.01)
            for feat in FEATURE_NAMES:
                self.reference_data[feat] = df[feat].values.astype(np.float32)

    @staticmethod
    def calculate_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
        """
        Calculates the Population Stability Index (PSI) between reference and current distributions.
        Industry Standard:
          - PSI < 0.10: Stable distribution
          - 0.10 <= PSI < 0.20: Moderate distribution shift
          - PSI >= 0.20: Significant concept/data drift
        """
        if len(reference) == 0 or len(current) == 0:
            return 0.0

        # Create quantile bins using reference distribution
        percentiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(reference, percentiles)
        bin_edges = np.unique(bin_edges)  # remove duplicates for low-variance features
        if len(bin_edges) < 2:
            return 0.0

        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Calculate counts per bin
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to percentages with small epsilon to prevent div/log by zero
        eps = 1e-4
        ref_pct = (ref_counts / len(reference)) + eps
        curr_pct = (curr_counts / len(current)) + eps

        # Normalize percentages
        ref_pct /= np.sum(ref_pct)
        curr_pct /= np.sum(curr_pct)

        # PSI = sum((Actual - Expected) * ln(Actual / Expected))
        psi_val = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(round(max(0.0, psi_val), 4))

    @staticmethod
    def calculate_ks_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """Runs the two-sample Kolmogorov-Smirnov test returning (statistic, p_value)."""
        if len(reference) < 5 or len(current) < 5:
            return 0.0, 1.0
        res = stats.ks_2samp(reference, current)
        return float(round(res.statistic, 4)), float(round(res.pvalue, 6))

    @staticmethod
    def calculate_wasserstein(reference: np.ndarray, current: np.ndarray) -> float:
        """Calculates Earth Mover's (Wasserstein) Distance between distributions."""
        if len(reference) < 2 or len(current) < 2:
            return 0.0
        dist = stats.wasserstein_distance(reference, current)
        return float(round(dist, 4))

    def analyze_feature(self, feature_name: str, current_vals: np.ndarray) -> FeatureDriftMetric:
        """Analyzes statistical drift for a single feature."""
        ref_vals = self.reference_data.get(feature_name, np.array([0.0]))
        psi = self.calculate_psi(ref_vals, current_vals)
        ks_stat, ks_pval = self.calculate_ks_test(ref_vals, current_vals)
        wasserstein = self.calculate_wasserstein(ref_vals, current_vals)

        # Drift criteria: Significant if PSI >= 0.20 or strong KS deviation
        is_drift = (psi >= 0.20) or (ks_stat >= 0.22 and ks_pval < 0.01)

        if psi >= 0.20 or (ks_stat >= 0.25 and ks_pval < 0.01):
            status = "CRITICAL_DRIFT"
        elif psi >= 0.10 or (ks_stat >= 0.15 and ks_pval < 0.05):
            status = "MODERATE_SHIFT"
        else:
            status = "STABLE"

        return FeatureDriftMetric(
            feature_name=feature_name,
            psi_score=psi,
            ks_statistic=ks_stat,
            ks_p_value=ks_pval,
            wasserstein_distance=wasserstein,
            drift_detected=is_drift,
            status=status,
        )

    def compute_drift_report(self, current_records: List[Dict[str, Any]]) -> DriftReportResponse:
        """Generates a complete multi-feature drift report from live production events."""
        if not current_records:
            # Empty fallback
            empty_feats = {
                f: FeatureDriftMetric(
                    feature_name=f,
                    psi_score=0.0,
                    ks_statistic=0.0,
                    ks_p_value=1.0,
                    wasserstein_distance=0.0,
                    drift_detected=False,
                    status="STABLE"
                )
                for f in FEATURE_NAMES
            }
            return DriftReportResponse(
                dataset_drift_detected=False,
                drifted_feature_count=0,
                total_features=len(FEATURE_NAMES),
                drift_share=0.0,
                features=empty_feats,
                sample_size=0,
            )

        df_curr = pd.DataFrame(current_records)
        feature_metrics: Dict[str, FeatureDriftMetric] = {}
        drifted_count = 0

        for f in FEATURE_NAMES:
            if f in df_curr.columns:
                curr_vals = df_curr[f].dropna().values.astype(np.float32)
                metric = self.analyze_feature(f, curr_vals)
            else:
                metric = FeatureDriftMetric(
                    feature_name=f,
                    psi_score=0.0,
                    ks_statistic=0.0,
                    ks_p_value=1.0,
                    wasserstein_distance=0.0,
                    drift_detected=False,
                    status="STABLE"
                )
            feature_metrics[f] = metric
            if metric.drift_detected:
                drifted_count += 1

        total_feats = len(FEATURE_NAMES)
        drift_share = round((drifted_count / total_feats) * 100.0, 1)
        dataset_drift = (drifted_count >= 2) or (drift_share >= 30.0)

        return DriftReportResponse(
            dataset_drift_detected=dataset_drift,
            drifted_feature_count=drifted_count,
            total_features=total_feats,
            drift_share=drift_share,
            features=feature_metrics,
            sample_size=len(df_curr),
        )

    def get_comparison_distributions(self, feature_name: str, current_records: List[Dict[str, Any]], max_points: int = 500) -> Dict[str, List[float]]:
        """Returns sampled arrays for reference vs production distribution overlay plots."""
        ref = self.reference_data.get(feature_name, np.array([]))
        ref_sample = [float(x) for x in np.random.choice(ref, size=min(max_points, len(ref)), replace=False)] if len(ref) > 0 else []

        curr_sample = []
        if current_records:
            df = pd.DataFrame(current_records)
            if feature_name in df.columns:
                vals = df[feature_name].dropna().values
                curr_sample = [float(x) for x in np.random.choice(vals, size=min(max_points, len(vals)), replace=False)] if len(vals) > 0 else []

        return {
            "reference": ref_sample,
            "current": curr_sample,
        }
