"""
Calibrated Uncertainty via MC-Dropout.

The NeuroEconometricNet already has dropout layers throughout the neural
branch and fusion gate (see app/models/neural.py, app/models/fusion.py).
MC-Dropout (Gal & Ghahramani, 2016) reinterprets dropout-at-inference-time
as approximate Bayesian inference: running N stochastic forward passes
with dropout left ACTIVE (rather than the usual model.eval() which turns
dropout off) and treating the resulting distribution of predictions as an
approximate predictive distribution.

This module does two things:
1. Produce N-sample predictive distributions -> mean/quantiles/intervals.
2. VERIFY (not assume) that the resulting intervals are calibrated, via:
   - PIT (Probability Integral Transform) histogram: for a well-calibrated
     model, PIT values (the predictive CDF evaluated at the realised
     outcome) should be ~Uniform(0,1). Systematic skew/humps indicate
     mis-calibration.
   - Reliability diagram: empirical coverage of nominal x% intervals
     (e.g. does the 90% interval actually contain the true value ~90% of
     the time on held-out data?).

Per the project rule: never claim "90% intervals" without checking
empirical coverage on a held-out set. The functions here compute that
coverage directly from data; nothing about calibration is assumed.
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def enable_mc_dropout(model: nn.Module) -> None:
    """
    Put the model in eval mode (BatchNorm/etc. frozen) EXCEPT dropout
    layers, which are switched back to train mode so they keep sampling.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            module.train()


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    feat: torch.Tensor,
    ardl: torch.Tensor,
    vol: torch.Tensor,
    sent: torch.Tensor,
    n_samples: int = 50,
) -> np.ndarray:
    """
    Run N stochastic forward passes with dropout active.

    Returns:
        samples: (n_samples, batch) array of predictions.
    """
    enable_mc_dropout(model)
    samples = []
    for _ in range(n_samples):
        pred, _, _ = model(feat, ardl, vol, sent)
        samples.append(pred.cpu().numpy().flatten())
    model.eval()  # restore full eval mode for any subsequent deterministic use
    return np.stack(samples, axis=0)  # (n_samples, batch)


def prediction_intervals(samples: np.ndarray, coverage_levels=(0.5, 0.8, 0.9)) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """
    Args:
        samples: (n_samples, N) MC-dropout predictive samples.
        coverage_levels: nominal central-interval coverages to report.

    Returns:
        {coverage: (lower[N], upper[N])}
    """
    intervals = {}
    for cov in coverage_levels:
        alpha = 1 - cov
        lower = np.quantile(samples, alpha / 2, axis=0)
        upper = np.quantile(samples, 1 - alpha / 2, axis=0)
        intervals[cov] = (lower, upper)
    return intervals


def empirical_coverage(actuals: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of actuals falling inside [lower, upper] -- the ONLY honest way to state 'X% interval'."""
    inside = (actuals >= lower) & (actuals <= upper)
    return float(np.mean(inside))


def pit_values(samples: np.ndarray, actuals: np.ndarray) -> np.ndarray:
    """
    Probability Integral Transform: for each observation, the fraction of
    MC-dropout samples <= the realised actual value -- i.e. the predictive
    CDF evaluated at the truth. For a well-calibrated predictive
    distribution, PIT values should be ~Uniform(0,1) (Dawid, 1984).

    Args:
        samples: (n_samples, N) MC-dropout predictive samples.
        actuals: (N,) realised values.

    Returns:
        (N,) array of PIT values in [0, 1].
    """
    n_samples = samples.shape[0]
    pit = np.mean(samples <= actuals[None, :], axis=0)
    return pit


def pit_uniformity_test(pit: np.ndarray) -> Dict[str, float]:
    """
    Kolmogorov-Smirnov test of PIT values against Uniform(0,1).
    A small p-value is evidence AGAINST calibration (reject uniformity).
    """
    from scipy import stats
    ks_stat, ks_p = stats.kstest(pit, "uniform")
    return {"ks_statistic": float(ks_stat), "ks_pvalue": float(ks_p)}


def calibration_report(
    samples: np.ndarray,
    actuals: np.ndarray,
    coverage_levels=(0.5, 0.8, 0.9),
) -> Dict:
    """
    Full calibration diagnostic: empirical coverage per nominal level +
    PIT histogram counts + KS uniformity test. All numbers computed
    directly from `samples`/`actuals` -- nothing assumed or hardcoded.
    """
    intervals = prediction_intervals(samples, coverage_levels)
    coverage = {}
    for cov, (lower, upper) in intervals.items():
        emp = empirical_coverage(actuals, lower, upper)
        coverage[cov] = {
            "nominal": cov,
            "empirical": emp,
            "gap": emp - cov,
        }

    pit = pit_values(samples, actuals)
    pit_hist, pit_edges = np.histogram(pit, bins=10, range=(0, 1))
    ks = pit_uniformity_test(pit)

    report = {
        "n_predictions": int(len(actuals)),
        "n_mc_samples": int(samples.shape[0]),
        "coverage": coverage,
        "pit_histogram_counts": pit_hist.tolist(),
        "pit_histogram_edges": pit_edges.tolist(),
        "pit_ks_test": ks,
        "pred_std_mean": float(samples.std(axis=0).mean()),
    }
    logger.info(
        "Calibration report: "
        + ", ".join(
            f"{int(c*100)}% nominal -> {v['empirical']*100:.1f}% empirical"
            for c, v in coverage.items()
        )
        + f" | PIT KS p={ks['ks_pvalue']:.4f}"
    )
    return report
