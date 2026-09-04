import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.analytics.drift import StatisticalDriftEngine
from src.inference.app import app
from src.streaming.schemas import DriftReportResponse


def test_psi_identical_distributions():
    engine = StatisticalDriftEngine()
    data = np.random.normal(loc=100.0, scale=15.0, size=1000)
    psi = engine.calculate_psi(data, data)
    assert psi < 0.05, f"Expected PSI for identical distributions to be near 0, got {psi}"


def test_psi_divergent_distributions():
    engine = StatisticalDriftEngine()
    ref = np.random.normal(loc=50.0, scale=10.0, size=2000)
    shifted = np.random.normal(loc=500.0, scale=50.0, size=2000)
    psi = engine.calculate_psi(ref, shifted)
    assert psi >= 0.20, f"Expected critical PSI (>= 0.20) for 10x shifted distribution, got {psi}"


def test_ks_and_wasserstein_metrics():
    engine = StatisticalDriftEngine()
    ref = np.random.exponential(scale=10.0, size=1000)
    shifted = np.random.exponential(scale=50.0, size=1000)

    stat, pval = engine.calculate_ks_test(ref, shifted)
    assert stat > 0.20
    assert pval < 0.01

    w_dist = engine.calculate_wasserstein(ref, shifted)
    assert w_dist > 10.0


def test_feature_analysis_status():
    engine = StatisticalDriftEngine()
    ref = np.random.lognormal(mean=3.5, sigma=0.8, size=2000)
    engine.reference_data["amount"] = ref

    # Test stable
    stable_sample = np.random.lognormal(mean=3.5, sigma=0.8, size=300)
    metric_stable = engine.analyze_feature("amount", stable_sample)
    assert metric_stable.status in ("STABLE", "MODERATE_SHIFT")
    assert not metric_stable.drift_detected

    # Test critical drift
    drifted_sample = np.random.uniform(5000.0, 10000.0, size=300)
    metric_drift = engine.analyze_feature("amount", drifted_sample)
    assert metric_drift.status == "CRITICAL_DRIFT"
    assert metric_drift.drift_detected


def test_compute_drift_report_and_endpoints():
    with TestClient(app) as client:
        # Check initial drift endpoint
        res = client.get("/v1/analytics/drift?limit=50")
        assert res.status_code == 200
        data = res.json()
        assert "dataset_drift_detected" in data
        assert "features" in data
        assert "amount" in data["features"]

        # Check distribution endpoint
        dist_res = client.get("/v1/analytics/drift/distribution?feature_name=amount&limit=50")
        assert dist_res.status_code == 200
        dist_data = dist_res.json()
        assert "reference" in dist_data
        assert "current" in dist_data

        # Simulate shift
        shift_res = client.post("/v1/analytics/drift/simulate-shift?shift_type=amount_surge&count=20")
        assert shift_res.status_code == 200
        assert shift_res.json()["injected_count"] == 20

        # After shift, query drift report
        drift_after = client.get("/v1/analytics/drift?limit=50")
        assert drift_after.status_code == 200
        report_obj = DriftReportResponse(**drift_after.json())
        assert report_obj.sample_size >= 20
