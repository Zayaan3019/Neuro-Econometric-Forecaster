"""
Statistical tests for comparing forecast accuracy across models.
"""
from typing import Dict, Literal
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def diebold_mariano_test(
    actual: np.ndarray,
    forecast1: np.ndarray,
    forecast2: np.ndarray,
    h: int = 1,
    loss: Literal["squared", "absolute"] = "squared",
) -> Dict[str, float]:
    """
    Diebold-Mariano (1995) test of equal predictive accuracy.

    H0: E[L(e1_t)] = E[L(e2_t)]  (forecast1 and forecast2 are equally accurate)
    H1: E[L(e1_t)] != E[L(e2_t)]

    A significantly NEGATIVE DM statistic means forecast1 (typically the
    "new"/hybrid model here) has LOWER loss than forecast2 (the baseline) --
    i.e. forecast1 is more accurate. A significantly POSITIVE statistic
    means forecast2 is more accurate. |DM| not significant => cannot
    distinguish the two forecasts' accuracy at the given confidence level.

    Implements the standard Harvey-Leybourne-Newbold (1997) small-sample
    correction and a Newey-West HAC variance estimate with (h-1) lags,
    which is the textbook prescription for h-step-ahead forecast errors
    (for h=1, this reduces to just the sample variance of the loss
    differential -- no serial-correlation correction needed, but we
    compute it generally so the function is correct for h>1 forecasts too).

    Args:
        actual: (N,) realised values.
        forecast1: (N,) forecasts from model 1 (e.g. the hybrid model).
        forecast2: (N,) forecasts from model 2 (e.g. ARIMA baseline).
        h: forecast horizon (1 for one-step-ahead).
        loss: 'squared' or 'absolute' loss function.

    Returns:
        {"dm_statistic": float, "p_value": float, "n": int,
         "mean_loss1": float, "mean_loss2": float}
    """
    actual = np.asarray(actual, dtype=float)
    f1 = np.asarray(forecast1, dtype=float)
    f2 = np.asarray(forecast2, dtype=float)
    n = len(actual)
    assert len(f1) == n and len(f2) == n, "actual/forecast1/forecast2 must be same length"

    e1 = actual - f1
    e2 = actual - f2

    if loss == "squared":
        l1, l2 = e1 ** 2, e2 ** 2
    elif loss == "absolute":
        l1, l2 = np.abs(e1), np.abs(e2)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    d = l1 - l2
    d_mean = float(d.mean())

    # Newey-West HAC long-run variance with (h-1) lags
    max_lag = max(h - 1, 0)
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, max_lag + 1):
        cov = np.cov(d[lag:], d[:-lag])[0, 1]
        var_d += 2 * (1 - lag / (max_lag + 1)) * cov
    var_d = max(var_d, 1e-300)

    dm_stat = d_mean / np.sqrt(var_d / n)

    # Harvey-Leybourne-Newbold small-sample correction
    hln_correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_corrected = dm_stat * hln_correction

    # Student-t distribution with n-1 df (HLN recommendation) rather than
    # the asymptotic normal -- more conservative / appropriate in finite samples.
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_corrected), df=n - 1))

    result = {
        "dm_statistic": float(dm_stat_corrected),
        "p_value": float(p_value),
        "n": int(n),
        "mean_loss_model1": float(l1.mean()),
        "mean_loss_model2": float(l2.mean()),
        "loss_function": loss,
        "horizon": h,
        "interpretation": (
            "model1 significantly more accurate" if p_value < 0.05 and dm_stat_corrected < 0 else
            "model2 significantly more accurate" if p_value < 0.05 and dm_stat_corrected > 0 else
            "no significant difference in forecast accuracy"
        ),
    }
    logger.info(
        f"Diebold-Mariano test: DM={result['dm_statistic']:.4f}, p={result['p_value']:.4f} "
        f"-> {result['interpretation']}"
    )
    return result
