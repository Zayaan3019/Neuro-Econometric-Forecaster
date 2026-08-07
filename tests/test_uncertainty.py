"""
Tests for MC-Dropout uncertainty quantification (app/engine/uncertainty.py).

These verify the MACHINERY (sampling produces variation, coverage/PIT are
computed correctly from known synthetic distributions) -- NOT that the
trained NeuroEconometricNet is well-calibrated, which can only be checked
empirically against real held-out data (see models_saved/walk_forward_report.json
for the actual calibration numbers on real data).
"""
import numpy as np
import torch
import torch.nn as nn
import pytest

from app.engine.uncertainty import (
    enable_mc_dropout, prediction_intervals, empirical_coverage,
    pit_values, pit_uniformity_test, calibration_report,
)


class TinyDropoutNet(nn.Module):
    """Minimal net with dropout, standing in for NeuroEconometricNet in unit tests."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(4, 1)

    def forward(self, feat, ardl, vol, sent):
        x = torch.relu(self.fc(feat.mean(dim=1)))
        x = self.drop(x)
        pred = self.out(x)
        alpha = torch.full_like(pred, 0.5)
        return pred, alpha, pred


class TestEnableMCDropout:
    def test_dropout_active_produces_varying_outputs(self):
        torch.manual_seed(0)
        model = TinyDropoutNet()
        enable_mc_dropout(model)
        feat = torch.randn(8, 5, 4)
        ardl = torch.zeros(8, 1)
        vol = torch.zeros(8, 5)
        sent = torch.zeros(8, 1)

        outputs = [model(feat, ardl, vol, sent)[0].detach() for _ in range(20)]
        stacked = torch.stack(outputs)
        assert stacked.std(dim=0).mean().item() > 0, "MC-dropout samples should vary run to run"

    def test_eval_mode_deterministic(self):
        torch.manual_seed(0)
        model = TinyDropoutNet()
        model.eval()
        feat = torch.randn(8, 5, 4)
        ardl = torch.zeros(8, 1)
        vol = torch.zeros(8, 5)
        sent = torch.zeros(8, 1)
        out1 = model(feat, ardl, vol, sent)[0].detach()
        out2 = model(feat, ardl, vol, sent)[0].detach()
        assert torch.allclose(out1, out2), "full eval mode (dropout off) must be deterministic"


class TestCalibrationDiagnostics:
    def test_well_calibrated_gaussian_samples(self):
        """
        If MC samples truly are draws from N(mu_i, sigma_i) and actuals are
        also drawn from that SAME distribution, empirical coverage of the
        nominal x% interval should be close to x% (within sampling noise).
        """
        rng = np.random.RandomState(0)
        n_points = 2000
        n_samples = 200
        mu = rng.normal(0, 1, n_points)
        sigma = np.abs(rng.normal(1, 0.2, n_points)) + 0.5

        samples = rng.normal(mu[None, :], sigma[None, :], size=(n_samples, n_points))
        actuals = rng.normal(mu, sigma)

        report = calibration_report(samples, actuals, coverage_levels=(0.5, 0.8, 0.9))
        for cov, stats in report["coverage"].items():
            assert abs(stats["gap"]) < 0.08, (
                f"well-calibrated synthetic data should show empirical coverage "
                f"close to nominal {cov}, got {stats['empirical']}"
            )
        # PIT values from a truly well-calibrated model should not be
        # rejected as non-uniform at conventional significance.
        assert report["pit_ks_test"]["ks_pvalue"] > 0.01

    def test_overconfident_intervals_show_undercoverage(self):
        """If predictive samples are artificially too narrow, empirical coverage must fall below nominal."""
        rng = np.random.RandomState(1)
        n_points = 1000
        n_samples = 200
        mu = rng.normal(0, 1, n_points)
        true_sigma = 2.0
        narrow_sigma = 0.2  # model is overconfident: samples too tight

        samples = rng.normal(mu[None, :], narrow_sigma, size=(n_samples, n_points))
        actuals = rng.normal(mu, true_sigma)

        report = calibration_report(samples, actuals, coverage_levels=(0.9,))
        assert report["coverage"][0.9]["empirical"] < 0.9 - 0.1, \
            "overconfident (too-narrow) intervals must show under-coverage"

    def test_empirical_coverage_basic(self):
        actuals = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
        lower = np.array([-1, -1, -1, -1, -1])
        upper = np.array([1, 1, 1, 1, 1])
        cov = empirical_coverage(actuals, lower, upper)
        assert cov == 2 / 5  # only 0 and 1 fall inside [-1, 1]; 2, 3, 10 do not

    def test_pit_values_uniform_for_true_model(self):
        rng = np.random.RandomState(2)
        n_points, n_samples = 500, 300
        mu = rng.normal(0, 1, n_points)
        samples = rng.normal(mu[None, :], 1.0, size=(n_samples, n_points))
        actuals = rng.normal(mu, 1.0)
        pit = pit_values(samples, actuals)
        assert pit.min() >= 0 and pit.max() <= 1
        ks = pit_uniformity_test(pit)
        assert ks["ks_pvalue"] > 0.01
