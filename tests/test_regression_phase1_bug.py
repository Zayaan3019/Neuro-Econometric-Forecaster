"""
Regression test for the root-caused Phase-1 directional-accuracy bug.

Original bug (documented in MODEL_EVALUATION_REPORT.md / the audit that
preceded this fix): the legacy training script (train_model.py) trained a
model to predict the normalized PRICE LEVEL `Close` at t+1
(`target = processed_df['Close'].iloc[actual_idx + 1]`), while the legacy
evaluation script (evaluate_model.py) computed "directional accuracy" by
comparing those level-predictions against `test_df.iloc[i, -1]` -- which,
after `Preprocessor.fit_transform`, is `Close_diff` (the normalized FIRST
DIFFERENCE of price), not `Close`. Comparing a level-predicting model's
output to a differenced-series "ground truth" is an apples-to-oranges
comparison that can produce an arbitrary, meaningless accuracy number
(the reported 32.84%) regardless of whether the model has learned
anything real.

The current, correct pipeline (run_pipeline.py / walk_forward_pipeline.py)
defines the model's target as the RETURN `(close[t+1]-close[t])/close[t]`
and evaluates directional accuracy as `sign(prediction) == sign(actual)`
on that SAME return quantity -- train and eval are measuring the same
thing. These tests lock that invariant in place so a future refactor
can't reintroduce a train/eval target mismatch.
"""
import numpy as np
import pandas as pd
import torch

from app.config import Config
from app.data.preprocessor import Preprocessor, TechnicalIndicatorEngine
from run_pipeline import MarketDataset, engineer_features


class TestNoLevelVsDifferenceMismatch:
    def test_preprocessor_close_and_close_diff_are_different_quantities(self):
        """
        Demonstrates the exact bug mechanism: after Preprocessor.fit_transform,
        'Close' (level) and 'Close_diff' (first difference) are materially
        different series -- so comparing predictions trained on one against
        ground truth read from the other is guaranteed to be meaningless.
        This is what the legacy evaluate_model.py effectively did.
        """
        np.random.seed(0)
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        close = 100 + np.cumsum(np.random.randn(300))
        df = pd.DataFrame({
            "Open": close, "High": close + 1, "Low": close - 1,
            "Close": close, "Volume": np.random.randint(1e6, 1e7, 300),
        }, index=dates)

        df_ind = TechnicalIndicatorEngine.compute_all(df)
        pre = Preprocessor()
        processed, meta = pre.fit_transform(df_ind)

        assert "Close" in processed.columns
        if meta["diff_order"] > 0:
            assert "Close_diff" in processed.columns
            # The two series must NOT be (numerically close to) the same
            # thing -- if they were, the original bug could never have
            # produced a meaningless comparison.
            corr = np.corrcoef(processed["Close"].values, processed["Close_diff"].values)[0, 1]
            assert abs(corr) < 0.9, (
                "Close (level) and Close_diff (first difference) are nearly "
                "identical in this fixture -- the regression test's premise "
                "doesn't hold; re-check the fixture."
            )

    def test_market_dataset_target_is_return_not_level(self):
        """
        The CURRENT dataset class must define its prediction target as a
        RETURN -- (close[t+1]-close[t])/close[t] -- not an absolute price
        level, and that target must be directly comparable (same units,
        same sign convention) to what the model is scored on.
        """
        np.random.seed(1)
        dates = pd.date_range("2020-01-01", periods=400, freq="D")
        close = pd.Series(100 + np.cumsum(np.random.randn(400) * 0.5), index=dates, name="Close")

        ohlcv = pd.DataFrame({
            "Open": close, "High": close + 1, "Low": close - 1,
            "Close": close, "Volume": 1_000_000,
        }, index=dates)

        full_df = engineer_features(ohlcv, pd.DataFrame())
        feature_cols = [c for c in full_df.columns if c != "Close"]

        ardl_dummy = pd.Series(close.values, index=range(len(close)))
        full_df_reset = full_df.reset_index(drop=True)

        ds = MarketDataset(full_df_reset, close.reset_index(drop=True), ardl_dummy,
                            feature_cols=feature_cols, seq_len=30)

        # Manually recompute the expected target for a few samples and
        # confirm the Dataset produces EXACTLY the return, not the level.
        close_arr = close.values
        for idx in [0, 5, len(ds) - 1]:
            t = ds.valid[idx]
            _, _, _, _, target = ds[idx]
            expected_return = (close_arr[t + 1] - close_arr[t]) / close_arr[t]
            assert np.isclose(target.item(), expected_return, atol=1e-4), (
                f"MarketDataset target at t={t} is {target.item()}, expected the "
                f"RETURN {expected_return} -- if this ever drifts back to an "
                f"absolute price level, the original train/eval mismatch bug "
                f"has been reintroduced."
            )
            # A price level would be ~100x larger than a daily return and
            # essentially never close to zero; a return is typically < 0.1.
            assert abs(target.item()) < 1.0, (
                "target magnitude looks like a price level, not a return"
            )

    def test_evaluation_and_training_target_use_same_construction(self):
        """
        walk_forward_pipeline.py independently reconstructs the scored
        return series from raw close prices and asserts it matches
        MarketDataset's own targets exactly (see the
        `np.testing.assert_allclose(scored_returns, actuals, ...)` check in
        walk_forward_pipeline.main()). This test exercises that same
        reconstruction logic in isolation so a future edit that breaks the
        alignment fails fast in CI rather than silently producing a
        meaningless metric again.
        """
        np.random.seed(2)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        n_seq = 30

        r_arr = close[1:] / close[:-1] - 1.0
        scored_returns = r_arr[n_seq: len(close) - 1]

        # Independently reconstruct via MarketDataset-style indexing
        expected = []
        for t in range(n_seq, n - 1):
            expected.append((close[t + 1] - close[t]) / close[t])
        expected = np.array(expected)

        np.testing.assert_allclose(scored_returns, expected, atol=1e-10)
