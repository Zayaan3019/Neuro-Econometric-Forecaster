"""
Every engineered feature at row t must be computable from data up to row t.

The test changes the FUTURE (every price and macro value from row `cut` on)
and requires the features of every earlier row to be unchanged, to the bit.
It catches any look-ahead in feature construction: a backward fill, a centred
window, a negative shift, a scaler or statistic fit on the whole series.

History: until 2026-11-24 `engineer_features` ended with `ffill().bfill()`.
The backward fill copied the first valid value of each warm-up column back
over its leading NaNs, so the first 199 rows of `sma200_ratio` held the value
from 2010-10-18. It touched only early training rows (never a test row or a
served input), but it was look-ahead, and this test would have caught it.
"""
import numpy as np
import pandas as pd
import pytest

from app.config import Config
from run_pipeline import engineer_features, load_macro, load_ohlcv


@pytest.fixture(scope="module")
def history():
    ohlcv = load_ohlcv().iloc[:600]
    macro = load_macro(Config.START_DATE, Config.END_DATE)
    return ohlcv, macro.loc[: ohlcv.index[-1]]


def _perturb_from(df: pd.DataFrame, cut: int, seed: int) -> pd.DataFrame:
    """Scale every numeric value from row `cut` on by a random factor in [0.8, 1.2]."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    num = out.select_dtypes(include=[np.number]).columns
    out[num] = out[num].astype(float)
    factors = rng.uniform(0.8, 1.2, size=(len(out) - cut, len(num)))
    out.loc[out.index[cut:], num] = out.loc[out.index[cut:], num].values * factors
    return out


# Cuts inside every warm-up window (SMA10 … SMA200) and well past them.
@pytest.mark.parametrize("cut", [5, 30, 150, 198, 260, 450])
def test_features_before_cut_ignore_the_future(history, cut):
    ohlcv, macro = history
    base = engineer_features(ohlcv, macro)

    changed_ohlcv = _perturb_from(ohlcv, cut, seed=cut)
    macro_cut = macro.index.searchsorted(ohlcv.index[cut])
    changed_macro = _perturb_from(macro, macro_cut, seed=cut + 1)
    changed = engineer_features(changed_ohlcv, changed_macro)

    assert list(base.columns) == list(changed.columns)
    leaked = [
        c for c in base.columns
        if not np.array_equal(base[c].values[:cut], changed[c].values[:cut], equal_nan=True)
    ]
    assert not leaked, f"features before row {cut} changed when only the future changed: {leaked}"


def test_no_feature_is_nan_after_engineering(history):
    """Warm-up rows are filled without looking ahead, never left as NaN."""
    ohlcv, macro = history
    feats = engineer_features(ohlcv, macro)
    assert not feats.isna().any().any()


def test_legacy_preprocessor_fits_on_training_rows_only(history):
    """Changing the test rows must not change anything fitted (audit defect 2)."""
    from app.data.preprocessor import Preprocessor

    ohlcv, _ = history
    fit_rows = 400
    a, meta_a = Preprocessor().fit_transform(ohlcv, fit_rows=fit_rows)
    b, meta_b = Preprocessor().fit_transform(_perturb_from(ohlcv, fit_rows, seed=7), fit_rows=fit_rows)

    np.testing.assert_array_equal(meta_a["scaler"].mean_, meta_b["scaler"].mean_)
    np.testing.assert_array_equal(meta_a["scaler"].scale_, meta_b["scaler"].scale_)
    assert meta_a["diff_order"] == meta_b["diff_order"]
    pd.testing.assert_frame_equal(a.iloc[:fit_rows], b.iloc[:fit_rows])


def test_legacy_preprocessor_requires_a_training_boundary(history):
    from app.data.preprocessor import Preprocessor

    ohlcv, _ = history
    with pytest.raises(TypeError):
        Preprocessor().fit_transform(ohlcv)  # no fit_rows: refuses to guess
    with pytest.raises(ValueError):
        Preprocessor().fit_transform(ohlcv, fit_rows=len(ohlcv) + 1)
