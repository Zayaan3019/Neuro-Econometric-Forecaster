"""
Scenario-Based Market Condition Testing.

Evaluates the trained model's directional accuracy separately on each
calendar quarter of its HELD-OUT test period, using REAL S&P 500 data and the
CURRENT checkpoint (models_saved/best_model_v2.pt, produced by run_pipeline.py).

Only rows the checkpoint never saw are scored: rows >= val_end of
run_pipeline.split_bounds. Until 2026-11-24 this script scored the four
calendar quarters of 2024, but the checkpoint's validation split (used for
early stopping, i.e. for choosing the checkpoint) ran to 2024-02-07, so most
of "Q1 2024" was not out-of-sample. Quarters are now derived from the split,
and a quarter is clipped to its held-out rows.

A prior version of this file loaded the OLD, architecturally incompatible
checkpoint (models_saved/best_model.pt) and crashed with a state_dict
mismatch on every run -- it was never actually exercising the current
model. It has been rewritten to use the current checkpoint format,
feature set, and (critically) the correct RETURN-based target definition
that matches how the model is trained (see tests/test_regression_phase1_bug.py
for why the level-vs-difference mismatch matters).
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import pytest
from sklearn.preprocessing import StandardScaler

from app.config import Config
from run_pipeline import (
    load_ohlcv, load_macro, engineer_features, precompute_ardl_predictions, MarketDataset, split_bounds,
)
from app.models.fusion import NeuroEconometricNet

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = Path("models_saved/best_model_v2.pt")


def load_current_model(checkpoint_path: Path = CHECKPOINT_PATH):
    """Load the current-architecture checkpoint (produced by run_pipeline.py)."""
    ckpt = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)
    model = NeuroEconometricNet(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["config"]["hidden_dim"],
        nhead=ckpt["config"]["nhead"],
        num_lstm_layers=ckpt["config"]["num_layers"],
        dropout=ckpt["config"]["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(Config.DEVICE)
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = np.array(ckpt["scaler_mean"])
    scaler.scale_ = np.array(ckpt["scaler_scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(ckpt["feature_cols"])

    return model, scaler, ckpt["feature_cols"]


def evaluate_on_period(model, scaler, feature_cols, full_df, ohlcv, ardl_preds, start_idx, end_idx, period_name):
    """
    Evaluate directional accuracy on full_df.iloc[start_idx:end_idx], using
    `Config.SEQ_LENGTH` rows immediately BEFORE start_idx as lookback context
    (so the first prediction in the window still has a full causal history)
    and the model's actual RETURN target -- not a raw price level.
    """
    n_seq = Config.SEQ_LENGTH
    ctx_start = max(0, start_idx - n_seq)
    # One row past the period so its last day has a next-day return to score.
    stop = min(end_idx + 1, len(full_df))
    window_df = full_df.iloc[ctx_start:stop].copy()
    window_df[feature_cols] = scaler.transform(window_df[feature_cols].values)
    window_close = ohlcv["Close"].iloc[ctx_start:stop]
    window_ardl = ardl_preds.iloc[ctx_start:stop]

    ds = MarketDataset(window_df, window_close, window_ardl, feature_cols=feature_cols, seq_len=n_seq)
    if len(ds) == 0:
        return None

    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
    preds, actuals = [], []
    with torch.no_grad():
        for feat, ardl, vol, sent, tgt in loader:
            pred, alpha, _ = model(
                feat.to(Config.DEVICE), ardl.to(Config.DEVICE),
                vol.to(Config.DEVICE), sent.to(Config.DEVICE),
            )
            preds.append(pred.cpu().numpy())
            actuals.append(tgt.numpy())

    preds = np.concatenate(preds).flatten()
    actuals = np.concatenate(actuals).flatten()

    mse = float(np.mean((preds - actuals) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - actuals)))
    dir_accuracy = float(np.mean(np.sign(preds) == np.sign(actuals))) * 100

    logger.info(f"{period_name}: n={len(preds)} RMSE={rmse:.6f} MAE={mae:.6f} DirAcc={dir_accuracy:.2f}%")

    return {
        "period": period_name, "n_samples": len(preds),
        "rmse": rmse, "mae": mae, "directional_accuracy": dir_accuracy,
    }


@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="best_model_v2.pt not present (run run_pipeline.py first)")
def test_different_market_conditions():
    """Evaluate the checkpoint on each calendar quarter of its held-out test split."""
    model, scaler, feature_cols = load_current_model()

    ohlcv = load_ohlcv()
    macro = load_macro(Config.START_DATE, Config.END_DATE)
    full_df = engineer_features(ohlcv, macro)
    missing = [c for c in feature_cols if c not in full_df.columns]
    if missing:
        pytest.skip(f"Checkpoint feature set doesn't match current engineer_features() output "
                     f"(missing {missing}) -- likely trained before a feature-engineering change.")

    ardl_preds = precompute_ardl_predictions(full_df["Close"], window=Config.ARDL_WINDOW, lags=Config.ARDL_LAGS)

    _, val_end = split_bounds(len(full_df))
    held_out = full_df.index[val_end:]
    logger.info(f"Held-out test period: {held_out[0].date()} -> {held_out[-1].date()} ({len(held_out)} rows)")

    results = []
    for quarter, rows in pd.Series(np.arange(val_end, len(full_df)), index=held_out).groupby(held_out.to_period("Q")):
        if len(rows) < 10:
            logger.warning(f"Skipping {quarter}: only {len(rows)} held-out rows")
            continue
        result = evaluate_on_period(
            model, scaler, feature_cols, full_df, ohlcv, ardl_preds,
            int(rows.iloc[0]), int(rows.iloc[-1]) + 1, str(quarter),
        )
        if result:
            result["first_date"] = str(held_out[rows.iloc[0] - val_end].date())
            result["last_date"] = str(held_out[rows.iloc[-1] - val_end].date())
            results.append(result)

    assert len(results) > 0, "No quarters produced results -- data or checkpoint mismatch"
    for r in results:
        assert 0 <= r["directional_accuracy"] <= 100
        assert r["rmse"] >= 0
        assert r["n_samples"] > 0

    avg_dir_accuracy = float(np.mean([r["directional_accuracy"] for r in results]))
    std_dir_accuracy = float(np.std([r["directional_accuracy"] for r in results]))
    logger.info(f"Average Directional Accuracy across quarters: {avg_dir_accuracy:.2f}% (std={std_dir_accuracy:.2f}%)")

    output = {
        "timestamp": datetime.now().isoformat(),
        "held_out_from": str(held_out[0].date()),
        "periods": results,
        "summary": {
            "avg_directional_accuracy": avg_dir_accuracy,
            "std_directional_accuracy": std_dir_accuracy,
        },
    }
    Path("models_saved").mkdir(exist_ok=True)
    with open("models_saved/market_condition_test.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to: models_saved/market_condition_test.json")


if __name__ == "__main__":
    test_different_market_conditions()
