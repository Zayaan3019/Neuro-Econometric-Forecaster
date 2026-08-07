"""
Correctness tests for the Population Stability Index (PSI) implementation
(app/monitoring/psi.py), against known-distribution shifts with expected
PSI magnitude ranges (per the conventional bands: <0.1 stable, 0.1-0.25
moderate, >0.25 alert -- see module docstring for the threshold rationale).
"""
import numpy as np
import pandas as pd
import pytest

from app.monitoring.psi import compute_psi, psi_report, PSI_ALERT_THRESHOLD, PSI_MODERATE_THRESHOLD


class TestComputePSI:
    def test_identical_distribution_near_zero(self):
        rng = np.random.RandomState(0)
        ref = rng.normal(0, 1, 5000)
        same = rng.normal(0, 1, 5000)
        psi = compute_psi(ref, same)
        assert psi < PSI_MODERATE_THRESHOLD

    def test_small_shift_is_moderate_or_stable(self):
        rng = np.random.RandomState(0)
        ref = rng.normal(0, 1, 5000)
        shifted = rng.normal(0.2, 1, 5000)
        psi = compute_psi(ref, shifted)
        assert psi < PSI_ALERT_THRESHOLD

    def test_large_shift_triggers_alert(self):
        rng = np.random.RandomState(0)
        ref = rng.normal(0, 1, 5000)
        shifted = rng.normal(2.0, 1, 5000)
        psi = compute_psi(ref, shifted)
        assert psi >= PSI_ALERT_THRESHOLD

    def test_disjoint_distributions_very_large_psi(self):
        rng = np.random.RandomState(0)
        ref = rng.normal(0, 1, 5000)
        disjoint = rng.normal(20, 1, 5000)
        psi = compute_psi(ref, disjoint)
        assert psi > 5.0

    def test_variance_only_shift_detected(self):
        """PSI (bucket-based) should detect a pure variance change even with equal means."""
        rng = np.random.RandomState(0)
        ref = rng.normal(0, 1, 5000)
        var_shift = rng.normal(0, 3, 5000)
        psi = compute_psi(ref, var_shift)
        assert psi > PSI_MODERATE_THRESHOLD

    def test_psi_symmetry_not_assumed(self):
        """PSI is not symmetric in general (bins are defined on the reference) -- sanity check it runs both ways."""
        rng = np.random.RandomState(0)
        a = rng.normal(0, 1, 3000)
        b = rng.normal(1, 1, 3000)
        psi_ab = compute_psi(a, b)
        psi_ba = compute_psi(b, a)
        assert psi_ab > 0 and psi_ba > 0

    def test_empty_input_returns_nan(self):
        assert np.isnan(compute_psi(np.array([]), np.array([1, 2, 3])))


class TestPSIReport:
    def test_report_flags_drifted_feature_only(self):
        rng = np.random.RandomState(0)
        n = 2000
        ref_df = pd.DataFrame({
            "stable_feat": rng.normal(0, 1, n),
            "drifted_feat": rng.normal(0, 1, n),
        })
        cur_df = pd.DataFrame({
            "stable_feat": rng.normal(0, 1, n),
            "drifted_feat": rng.normal(3, 1, n),
        })
        report = psi_report(ref_df, cur_df, ["stable_feat", "drifted_feat"])
        assert set(report["feature"]) == {"stable_feat", "drifted_feat"}
        drifted_row = report[report["feature"] == "drifted_feat"].iloc[0]
        stable_row = report[report["feature"] == "stable_feat"].iloc[0]
        assert drifted_row["status"] == "alert"
        assert stable_row["status"] == "stable"
        assert drifted_row["psi"] > stable_row["psi"]
