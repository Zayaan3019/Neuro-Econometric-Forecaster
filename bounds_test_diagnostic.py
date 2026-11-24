"""
Reproduce the ARDL bounds-test diagnostic in MODEL_EVALUATION_REPORT.md.

Question: is there a long-run level relationship between the S&P 500 close
and the VIX? The Pesaran-Shin-Smith bounds test answers it without first
deciding whether each series is I(0) or I(1). This is a DIAGNOSTIC: nothing
here feeds the forecasting model.

Until 2026-11-24 the report quoted F = 7.035 with no script in the repository
that produced it. This script is that script. It writes
models_saved/bounds_test.json, the source of the numbers in the report.

    python bounds_test_diagnostic.py
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import Config
from app.models.econometrics import ARDLBoundsTest
from run_pipeline import load_macro, load_ohlcv

logging.basicConfig(level=logging.WARNING)

#: The sample the report describes: the last 800 sessions through 2024-12-31.
SAMPLES = {
    "report_sample_through_2024": ("2024-12-31", 800),
    "latest_800_sessions": (Config.END_DATE, 800),
}
MAX_LAGS = 5
CASE = 3  # unrestricted intercept, no trend
OUTPUT = Path("models_saved/bounds_test.json")


def _clean(obj):
    """Make statsmodels/numpy output JSON-serialisable."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run_sample(close: pd.Series, vix: pd.Series, end: str, n: int) -> dict:
    data = pd.concat([close.rename("Close"), vix.rename("VIX")], axis=1).loc[:end].dropna()
    data = data.iloc[-n:]
    decision, f_stat, info = ARDLBoundsTest.bounds_f_test(
        data["Close"], data[["VIX"]], max_lags=MAX_LAGS, case=CASE, ic="bic"
    )
    crit = info["critical_values"]
    return {
        "first_date": str(data.index[0].date()),
        "last_date": str(data.index[-1].date()),
        "n": len(data),
        "f_statistic": f_stat,
        "selected_order": info["selected_order"],
        "upper_bound_5pct": crit["upper"][95.0],
        "upper_bound_1pct": crit["upper"][99.0],
        "lower_bound_5pct": crit["lower"][95.0],
        "decision_at_5pct": info["decision_at_5pct"],
        "valid": info["valid"],
        "integration_orders": info["integration_orders"],
        "error_correction_speed": info["error_correction_speed"],
        "long_run_coefficients": info["long_run_coefficients"],
    }


def main() -> None:
    close = load_ohlcv()["Close"]
    vix = load_macro(Config.START_DATE, Config.END_DATE)["VIX"]
    results = {name: run_sample(close, vix, end, n) for name, (end, n) in SAMPLES.items()}
    results["settings"] = {"max_lags": MAX_LAGS, "case": CASE, "ic": "bic",
                           "endog": "S&P 500 close (level)", "exog": "VIX (level)"}
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(_clean(results), indent=2))
    for name in SAMPLES:
        r = results[name]
        print(f"{name}: {r['first_date']} -> {r['last_date']} (n={r['n']})  "
              f"F={r['f_statistic']:.3f}  order={r['selected_order']}  "
              f"I(1) bounds 5%/1% = {r['upper_bound_5pct']:.2f}/{r['upper_bound_1pct']:.2f}  "
              f"-> {r['decision_at_5pct']}")
    print(f"Written: {OUTPUT}")


if __name__ == "__main__":
    main()
