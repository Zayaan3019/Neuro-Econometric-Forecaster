"""The served model must see the same inputs the evaluated model saw.

`serve_api.model_inputs` builds the hybrid's inputs for the newest row. The
model was trained and walk-forward-evaluated on `run_pipeline.MarketDataset`.
If the two drift apart, the live forecast comes from a different model than
the one MODEL_EVALUATION_REPORT.md describes, and nothing else would notice.
So: build both on the same history and require the same tensors.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from run_pipeline import MarketDataset, precompute_ardl_predictions
from serve_api import model_inputs

SEQ = 60
COLS = ["ret_1d", "ATR", "BB_Width", "rsi", "ADX", "vol_21d", "intraday_range"]


@pytest.fixture
def history():
    rng = np.random.default_rng(7)
    n = 260
    close = 4000 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    df = pd.DataFrame(rng.normal(size=(n, len(COLS))), columns=COLS,
                      index=pd.bdate_range("2025-01-01", periods=n))
    df["Close"] = close
    scaler = StandardScaler().fit(df[COLS].values)
    return df, scaler


def _dataset_item(df, scaler, t):
    """MarketDataset's (features, ardl_ret, vol) for row t of df."""
    scaled = df.copy()
    scaled[COLS] = scaler.transform(df[COLS].values)
    ardl = precompute_ardl_predictions(df["Close"])
    ds = MarketDataset(scaled, df["Close"], ardl, feature_cols=COLS, seq_len=SEQ)
    idx = ds.valid.index(t)
    features, ardl_ret, vol, _sentiment, _target = ds[idx]
    return features.numpy(), float(ardl_ret.item()), vol.numpy()


def test_serving_inputs_match_training_inputs(history):
    df, scaler = history
    t = len(df) - 2  # the last row MarketDataset can score (it needs close[t+1])
    want_feat, want_ardl, want_vol = _dataset_item(df, scaler, t)

    # Serving sees history only up to row t, which is then its newest row.
    got_feat, got_ardl, got_vol = model_inputs(df.iloc[: t + 1], COLS, scaler, seq_len=SEQ)

    np.testing.assert_allclose(got_feat, want_feat, atol=1e-6)
    assert got_ardl == pytest.approx(want_ardl, abs=1e-6)
    np.testing.assert_allclose(got_vol, want_vol, atol=1e-6)


def test_econometric_input_is_a_real_forecast_not_a_default(history):
    df, scaler = history
    _, ardl_ret, _ = model_inputs(df, COLS, scaler, seq_len=SEQ)
    assert ardl_ret != 0.0  # the old serving path always passed exactly 0.0


def test_ar_branch_is_a_fitted_model_not_the_naive_fallback(history):
    # The defect this pins: AutoReg.fit(disp=False) raised TypeError on every
    # call, a bare except swallowed it, and every "forecast" was the last close.
    df, _ = history
    preds = precompute_ardl_predictions(df["Close"])
    valid = preds.dropna()
    assert len(valid) > 100
    naive = df["Close"].loc[valid.index]
    assert (valid - naive).abs().max() > 1e-6


def test_ar_forecast_for_t_uses_nothing_after_t(history):
    df, _ = history
    t = 200
    before = precompute_ardl_predictions(df["Close"]).iloc[t]
    future_changed = df["Close"].copy()
    future_changed.iloc[t + 1 :] *= 1.5
    after = precompute_ardl_predictions(future_changed).iloc[t]
    assert after == pytest.approx(before, abs=1e-9)


def test_ar_forecast_for_t_targets_the_next_close(history):
    # preds[t] must move when close[t] moves: it forecasts close[t+1] from
    # data through t (the old alignment forecast close[t] from data through t-1).
    df, _ = history
    t = 200
    before = precompute_ardl_predictions(df["Close"]).iloc[t]
    today_changed = df["Close"].copy()
    today_changed.iloc[t] *= 1.02
    assert precompute_ardl_predictions(today_changed).iloc[t] != pytest.approx(before, abs=1e-6)


def test_sequence_ends_before_the_newest_row(history):
    df, scaler = history
    features, _, _ = model_inputs(df, COLS, scaler, seq_len=SEQ)
    scaled_last = scaler.transform(df[COLS].values)[-1]
    scaled_prev = scaler.transform(df[COLS].values)[-2]
    np.testing.assert_allclose(features[-1], scaled_prev, atol=1e-6)
    assert not np.allclose(features[-1], scaled_last)
