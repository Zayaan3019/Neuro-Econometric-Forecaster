"""
Causality tests for the HMM regime detector (app/models/regime.py).

The entire point of CausalRegimeDetector is that its filtered posteriors,
when used as a historical feature, must never depend on future observations.
A documented bug in a sibling project used Viterbi-decoded (non-causal)
regime history as a model feature; these tests exist specifically to catch
a regression of that bug in this project.
"""
import numpy as np
import pandas as pd
import pytest

from app.models.regime import CausalRegimeDetector


@pytest.fixture
def two_regime_data():
    """First half low-vol, second half high-vol; features = [return, rolling vol]."""
    np.random.seed(0)
    n = 400
    low = np.random.normal(0, 0.5, n // 2)
    high = np.random.normal(0, 3.0, n // 2)
    ret = np.concatenate([low, high])
    vol5 = pd.Series(ret).rolling(5, min_periods=1).std().fillna(0).values
    X = np.column_stack([ret, vol5])
    return X, n


class TestCausalRegimeDetector:
    def test_fit_identifies_high_vol_state(self, two_regime_data):
        X, n = two_regime_data
        det = CausalRegimeDetector(n_states=2, random_state=42).fit(X)
        filtered = det.filtered_posteriors(X)
        p_high = filtered[:, det.high_vol_state_]
        assert p_high[: n // 2].mean() < 0.1
        assert p_high[n // 2 :].mean() > 0.8

    def test_filtered_differs_from_smoothed(self, two_regime_data):
        """Smoothed (forward-backward) uses future info and should differ from filtered."""
        X, _ = two_regime_data
        det = CausalRegimeDetector(n_states=2, random_state=42).fit(X)
        filtered = det.filtered_posteriors(X)
        smoothed = det.smoothed_posteriors(X)
        assert np.abs(filtered - smoothed).mean() > 1e-4

    def test_causality_prefix_invariance(self, two_regime_data):
        """filtered_posteriors[:t] must be identical whether or not data beyond t exists."""
        X, _ = two_regime_data
        det = CausalRegimeDetector(n_states=2, random_state=42).fit(X)
        t_cut = 250
        filtered_full = det.filtered_posteriors(X)
        filtered_truncated = det.filtered_posteriors(X[:t_cut])
        np.testing.assert_allclose(filtered_full[:t_cut], filtered_truncated, atol=1e-10)

    def test_causality_future_perturbation_invariance(self, two_regime_data):
        """Filtered posteriors for t < t_cut must not change when future data is altered."""
        X, _ = two_regime_data
        det = CausalRegimeDetector(n_states=2, random_state=42).fit(X)
        t_cut = 250
        filtered_full = det.filtered_posteriors(X)

        X_perturbed = X.copy()
        rng = np.random.RandomState(99)
        X_perturbed[t_cut:] = rng.normal(50, 10, X_perturbed[t_cut:].shape)
        filtered_perturbed = det.filtered_posteriors(X_perturbed)

        np.testing.assert_allclose(filtered_full[:t_cut], filtered_perturbed[:t_cut], atol=1e-10)

    def test_high_vol_probability_matches_filtered_column(self, two_regime_data):
        X, _ = two_regime_data
        det = CausalRegimeDetector(n_states=2, random_state=42).fit(X)
        prob = det.high_vol_probability(X)
        filtered = det.filtered_posteriors(X)
        np.testing.assert_allclose(prob, filtered[:, det.high_vol_state_])

    def test_posteriors_sum_to_one(self, two_regime_data):
        X, _ = two_regime_data
        det = CausalRegimeDetector(n_states=2, random_state=42).fit(X)
        filtered = det.filtered_posteriors(X)
        np.testing.assert_allclose(filtered.sum(axis=1), 1.0, atol=1e-8)
