"""
Baseline forecasters for benchmarking the hybrid NeuroEconometricNet.

Both baselines operate on the daily RETURN series (the same target the
hybrid model predicts), forecast strictly causally, and are evaluated on
exactly the same walk-forward test timestamps as the hybrid model.
"""
from typing import Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import ar_select_order
import logging
import warnings

logger = logging.getLogger(__name__)


def persistence_forecast(n: int) -> np.ndarray:
    """
    Naive 'persistence' baseline for a return series: predicted return = 0
    for every step (equivalent to 'tomorrow's price = today's price', the
    standard random-walk-null forecast for financial price levels).
    """
    return np.zeros(n)


def sign_persistence_forecast(prior_returns: np.ndarray) -> np.ndarray:
    """Alternate naive baseline: predict sign(return_t) = sign(return_{t-1})."""
    return np.sign(prior_returns)


def fit_arima_order(train_returns: pd.Series, max_p: int = 3, max_q: int = 3) -> Tuple[int, int, int]:
    """
    Select ARIMA(p, 0, q) order on the (already-stationary) return series by
    BIC grid search over p in [0..max_p], q in [0..max_q]. d=0 because the
    input is already a return series (the differenced, stationary transform
    of price) -- an ARIMA(p, 1, 0) on PRICE LEVELS is mathematically
    identical to an AR(p) on RETURNS, so fitting on returns with d=0 loses
    no generality for the AR part while keeping the MA(q) component that a
    pure AutoReg baseline would lack.
    """
    best_bic = np.inf
    best_order = (1, 0, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    fitted = ARIMA(train_returns, order=(p, 0, q), trend='c').fit()
                    if fitted.bic < best_bic:
                        best_bic = fitted.bic
                        best_order = (p, 0, q)
                except Exception:
                    continue
    logger.info(f"ARIMA order selected by BIC grid search: {best_order} (BIC={best_bic:.2f})")
    return best_order


def rolling_arima_forecast(
    train_returns: pd.Series,
    test_returns: pd.Series,
    order: Tuple[int, int, int],
    warmup_returns: pd.Series = None,
) -> np.ndarray:
    """
    Genuine rolling one-step-ahead ARIMA forecast: fit once on the training
    window, then for each test-period step, forecast 1 step ahead and
    incorporate the realised actual via `.append(refit=False)` (a fast
    Kalman-filter state update -- no re-estimation of parameters, but the
    forecast at each step still only uses information up to and including
    the previous step, exactly matching how the neural model is evaluated).

    Args:
        warmup_returns: optional returns that occur chronologically before
            `test_returns` within the same fold but are NOT scored (e.g.
            the first `seq_len` days of a test fold, which the neural model
            also cannot produce a prediction for since it needs a full
            lookback window). These are appended to the ARIMA's state via
            `.append(refit=False)` so both models have "seen" the same
            amount of test-period history before the scored comparison
            begins -- otherwise the ARIMA would have an unfair information
            deficit relative to the neural model at the start of each fold.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        state = ARIMA(train_returns, order=order, trend='c').fit()

        if warmup_returns is not None and len(warmup_returns) > 0:
            state = state.append(warmup_returns, refit=False)

        preds = np.zeros(len(test_returns))
        for i in range(len(test_returns)):
            fc = state.forecast(steps=1)
            preds[i] = float(fc.iloc[0])
            state = state.append(test_returns.iloc[i:i + 1], refit=False)

    return preds
