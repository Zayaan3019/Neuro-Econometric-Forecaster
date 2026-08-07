"""
Tests for the Diebold-Mariano test (app/engine/statistical_tests.py).
"""
import numpy as np
import pytest

from app.engine.statistical_tests import diebold_mariano_test


class TestDieboldMariano:
    def test_detects_model1_more_accurate(self):
        rng = np.random.RandomState(0)
        n = 500
        actual = rng.normal(0, 1, n)
        f1 = actual + rng.normal(0, 0.3, n)  # much better forecast
        f2 = actual + rng.normal(0, 1.5, n)  # much worse forecast
        result = diebold_mariano_test(actual, f1, f2)
        assert result["p_value"] < 0.01
        assert result["dm_statistic"] < 0
        assert result["interpretation"] == "model1 significantly more accurate"

    def test_detects_model2_more_accurate(self):
        rng = np.random.RandomState(1)
        n = 500
        actual = rng.normal(0, 1, n)
        f1 = actual + rng.normal(0, 1.5, n)
        f2 = actual + rng.normal(0, 0.3, n)
        result = diebold_mariano_test(actual, f1, f2)
        assert result["p_value"] < 0.01
        assert result["dm_statistic"] > 0
        assert result["interpretation"] == "model2 significantly more accurate"

    def test_equally_accurate_not_significant(self):
        rng = np.random.RandomState(2)
        n = 500
        actual = rng.normal(0, 1, n)
        f1 = actual + rng.normal(0, 1.0, n)
        f2 = actual + rng.normal(0, 1.0, n)
        result = diebold_mariano_test(actual, f1, f2)
        assert result["p_value"] > 0.05
        assert result["interpretation"] == "no significant difference in forecast accuracy"

    def test_perfect_forecast_vs_noise(self):
        rng = np.random.RandomState(3)
        n = 300
        actual = rng.normal(0, 1, n)
        f1 = actual.copy()  # perfect
        f2 = rng.normal(0, 1, n)  # unrelated noise
        result = diebold_mariano_test(actual, f1, f2)
        assert result["mean_loss_model1"] < result["mean_loss_model2"]
        assert result["dm_statistic"] < 0

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            diebold_mariano_test(np.zeros(10), np.zeros(9), np.zeros(10))
