"""
True Rolling Walk-Forward Evaluation Harness
=============================================

Replaces the single 70/15/15 chronological split in run_pipeline.py with a
genuine rolling-origin walk-forward evaluation: an expanding training
window is retrained from scratch each fold, evaluated on the immediately
following, never-before-seen block of time, and the window then rolls
forward. No k-fold, no random split, anywhere.

Per fold, this script:
  1. Fits a causal regime detector (forward-filtered HMM posteriors,
     app/models/regime.py) on the fold's TRAINING window only, and adds
     the resulting P(high-vol regime) as an additional causal feature.
  2. Fits the feature scaler on the TRAINING window only.
  3. Trains NeuroEconometricNet from scratch on the training window (with
     a chronological validation slice carved from the end of that window
     for early stopping).
  4. Evaluates on the held-out test block: hybrid model (deterministic +
     MC-dropout uncertainty), a zero-return persistence baseline, and a
     BIC-selected rolling ARIMA(p,0,q) baseline -- all on the identical
     scored timestamps.
  5. Computes PSI on the input feature distributions (training window vs
     this test block).
  6. Runs a Diebold-Mariano test (hybrid vs ARIMA, hybrid vs persistence).

All fold results plus a pooled (all-folds-concatenated) summary are written
to models_saved/walk_forward_report.json. Every number in that file is
produced by this script -- nothing is hardcoded or assumed.
"""
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy import stats

from app.config import Config
from app.models.fusion import NeuroEconometricNet
from app.models.regime import CausalRegimeDetector
from app.engine.uncertainty import mc_dropout_predict, calibration_report
from app.engine.statistical_tests import diebold_mariano_test
from app.engine.baselines import persistence_forecast, fit_arima_order, rolling_arima_forecast
from app.monitoring.psi import psi_report

from run_pipeline import (
    load_ohlcv, load_macro, engineer_features, precompute_ardl_predictions,
    fit_scaler, scale_df, MarketDataset, ProductionLoss, build_optimizer, run_epoch,
)

Config.create_directories()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Config.LOGS_DIR / "walk_forward.log", mode="w", encoding="utf-8"),
    ],
    force=True,
)
log = logging.getLogger(__name__)
Config.set_seed()

device = Config.DEVICE

# ── Walk-forward configuration ──────────────────────────────────────────
# Compute-budget tradeoff (stated explicitly, not hidden): full retraining
# per fold costs ~5-15 min on CPU. Config.WALK_FORWARD_STEP (21 days) would
# require >100 folds -- intractable here. 5 folds with an expanding window,
# each covering roughly a year of out-of-sample test data, is the coarser
# but still genuinely walk-forward (no shuffling, strictly chronological,
# retrained-from-scratch-per-fold) scheme used below.
N_FOLDS = 5
INITIAL_TRAIN_FRAC = 0.55
VAL_FRAC_OF_TRAIN = 0.15
MC_SAMPLES = 30


def build_fold_boundaries(n_rows: int, n_folds: int = N_FOLDS, initial_frac: float = INITIAL_TRAIN_FRAC):
    initial_train_end = int(n_rows * initial_frac)
    remaining = n_rows - initial_train_end
    fold_size = remaining // n_folds
    boundaries = []
    train_end = initial_train_end
    for k in range(n_folds):
        test_start = train_end
        test_end = test_start + fold_size if k < n_folds - 1 else n_rows
        boundaries.append((train_end, test_start, test_end))
        train_end = test_end
    return boundaries


def compute_regime_feature(full_df: pd.DataFrame, train_end: int, test_end: int):
    """Causal regime probability: HMM fit on train only, forward-filtered over train+test."""
    reg_input_full = full_df[["ret_1d", "vol_10d"]].fillna(0.0).values[:test_end]
    det = CausalRegimeDetector(n_states=2, random_state=Config.SEED)
    det.fit(reg_input_full[:train_end])
    filtered = det.filtered_posteriors(reg_input_full)
    high_vol_prob = filtered[:, det.high_vol_state_]
    return high_vol_prob, det


def train_fold_model(train_sc, train_close, train_ardl, val_sc, val_close, val_ardl,
                      feature_cols, input_dim, max_epochs, patience, fold_idx):
    model = NeuroEconometricNet(
        input_dim=input_dim, hidden_dim=Config.HIDDEN_DIM, nhead=Config.NHEAD,
        num_lstm_layers=Config.NUM_LAYERS, dropout=Config.DROPOUT,
    ).to(device)
    crit = ProductionLoss()
    opt = build_optimizer(model, Config.LEARNING_RATE)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    kw = dict(feature_cols=feature_cols, seq_len=Config.SEQ_LENGTH)
    ds_train = MarketDataset(train_sc, train_close, train_ardl, **kw)
    ds_val = MarketDataset(val_sc, val_close, val_ardl, **kw)
    dl_train = DataLoader(ds_train, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)

    log.info(f"  [fold {fold_idx}] train samples={len(ds_train)}  val samples={len(ds_val)}")

    best_val = float("inf")
    best_state = None
    patience_count = 0
    for epoch in range(1, max_epochs + 1):
        tr = run_epoch(model, dl_train, crit, opt, device, training=True)
        va = run_epoch(model, dl_val, crit, None, device, training=False)
        sched.step(va["total"])
        is_best = va["total"] < best_val
        if is_best:
            best_val = va["total"]
            patience_count = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
        log.info(
            f"  [fold {fold_idx}] epoch {epoch:3d} {'*' if is_best else ' '} "
            f"train={tr['total']:.4f} val={va['total']:.4f} "
            f"dir_acc={va['dir_accuracy']:.3f} alpha_mean={va['alpha_mean']:.3f} alpha_std={va['alpha_std']:.4f}"
        )
        if patience_count >= patience:
            log.info(f"  [fold {fold_idx}] early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate_fold(model, test_sc, test_close, test_ardl, feature_cols):
    ds_test = MarketDataset(test_sc, test_close, test_ardl, feature_cols=feature_cols, seq_len=Config.SEQ_LENGTH)
    dl_test = DataLoader(ds_test, batch_size=256, shuffle=False, num_workers=0)

    all_pred, all_tgt, all_alpha = [], [], []
    all_feat, all_ardl, all_vol, all_sent = [], [], [], []
    model.eval()
    with torch.no_grad():
        for feat, ardl, vol, sent, tgt in dl_test:
            pred, alpha, _ = model(feat.to(device), ardl.to(device), vol.to(device), sent.to(device))
            all_pred.append(pred.cpu().numpy())
            all_tgt.append(tgt.numpy())
            all_alpha.append(alpha.cpu().numpy())
            all_feat.append(feat)
            all_ardl.append(ardl)
            all_vol.append(vol)
            all_sent.append(sent)

    preds = np.concatenate(all_pred).flatten()
    actuals = np.concatenate(all_tgt).flatten()
    alphas = np.concatenate(all_alpha).flatten()

    mc_chunks = []
    for feat, ardl, vol, sent in zip(all_feat, all_ardl, all_vol, all_sent):
        s = mc_dropout_predict(model, feat.to(device), ardl.to(device), vol.to(device), sent.to(device), n_samples=MC_SAMPLES)
        mc_chunks.append(s)
    mc_samples = np.concatenate(mc_chunks, axis=1)  # (MC_SAMPLES, N)

    return ds_test, preds, actuals, alphas, mc_samples


def compute_metrics(preds: np.ndarray, actuals: np.ndarray) -> dict:
    mse = float(np.mean((preds - actuals) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - actuals)))
    mape = float(np.mean(np.abs((preds - actuals) / (np.abs(actuals) + 1e-8)))) * 100
    dir_acc = float(np.mean(np.sign(preds) == np.sign(actuals)))
    if preds.std() > 1e-9 and actuals.std() > 1e-9:
        ic = float(np.corrcoef(preds, actuals)[0, 1])
    else:
        ic = 0.0
    signals = np.sign(preds)
    strat_rets = signals * actuals
    if strat_rets.std() > 1e-9:
        sharpe = float(np.sqrt(252) * strat_rets.mean() / strat_rets.std())
    else:
        sharpe = 0.0
    return {
        "MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE_pct": mape,
        "Directional_Accuracy": dir_acc, "IC": ic, "Sharpe_naive_signal_strategy": sharpe,
        "N": int(len(actuals)),
    }


def main():
    log.info("=" * 78)
    log.info("WALK-FORWARD EVALUATION — real S&P 500 data, expanding-window, retrain-per-fold")
    log.info("=" * 78)

    log.info("[1/4] Loading real OHLCV + macro data...")
    ohlcv = load_ohlcv()
    macro = load_macro(Config.START_DATE, Config.END_DATE)
    full_df = engineer_features(ohlcv, macro)
    feature_cols_base = [c for c in full_df.columns if c != "Close"]
    log.info(f"  Base feature columns ({len(feature_cols_base)}): {feature_cols_base}")

    log.info("[2/4] Pre-computing causal rolling ARDL predictions (whole series, window-based)...")
    ardl_preds = precompute_ardl_predictions(full_df["Close"], window=Config.ARDL_WINDOW, lags=Config.ARDL_LAGS)

    n = len(full_df)
    boundaries = build_fold_boundaries(n)
    log.info(f"[3/4] Walk-forward fold boundaries (train_end, test_start, test_end): {boundaries}")

    fold_results = []
    pooled = {"hybrid": [], "actual": [], "persistence": [], "arima": [], "mc": []}

    for k, (train_end, test_start, test_end) in enumerate(boundaries, start=1):
        log.info(f"\n{'=' * 70}\nFOLD {k}/{len(boundaries)}  train=[0:{train_end}]  test=[{test_start}:{test_end}]  (n_test={test_end - test_start})\n{'=' * 70}")

        high_vol_prob, regime_det = compute_regime_feature(full_df, train_end, test_end)

        fold_df = full_df.iloc[:test_end].copy()
        fold_df["regime_high_vol_prob"] = high_vol_prob
        feature_cols = feature_cols_base + ["regime_high_vol_prob"]

        val_start = int(train_end * (1 - VAL_FRAC_OF_TRAIN))
        actual_train_df = fold_df.iloc[:val_start]
        val_df = fold_df.iloc[val_start:train_end]
        test_df_fold = fold_df.iloc[test_start:test_end]

        scaler = fit_scaler(actual_train_df, feature_cols)
        actual_train_sc = scale_df(actual_train_df, feature_cols, scaler)
        val_sc = scale_df(val_df, feature_cols, scaler)
        test_sc = scale_df(test_df_fold, feature_cols, scaler)

        train_close = ohlcv["Close"].iloc[:val_start]
        val_close = ohlcv["Close"].iloc[val_start:train_end]
        test_close = ohlcv["Close"].iloc[test_start:test_end]

        train_ardl = ardl_preds.iloc[:val_start]
        val_ardl = ardl_preds.iloc[val_start:train_end]
        test_ardl = ardl_preds.iloc[test_start:test_end]

        input_dim = len(feature_cols)
        model = train_fold_model(
            actual_train_sc, train_close, train_ardl, val_sc, val_close, val_ardl,
            feature_cols, input_dim, max_epochs=Config.NUM_EPOCHS, patience=Config.EARLY_STOP_PATIENCE,
            fold_idx=k,
        )

        ds_test, preds, actuals, alphas, mc_samples = evaluate_fold(model, test_sc, test_close, test_ardl, feature_cols)

        # ---- Baselines on the identical scored timestamps ----
        n_seq = Config.SEQ_LENGTH
        test_close_arr = test_close.values.astype(float)
        r_arr = test_close_arr[1:] / test_close_arr[:-1] - 1.0
        scored_returns = r_arr[n_seq: len(test_close_arr) - 1]
        np.testing.assert_allclose(scored_returns, actuals, atol=1e-6,
                                    err_msg="Baseline/hybrid target alignment mismatch!")

        persistence_preds = persistence_forecast(len(actuals))

        # NOTE: .name must be unset and identical across train/warmup/scored --
        # statsmodels' ARIMA.append() concatenates these internally using the
        # Series name as the column label; a name mismatch (e.g. inherited
        # 'Close' from train_close.pct_change() vs an unnamed pd.Series())
        # raises "Columns must match to concatenate along rows."
        train_returns = train_close.pct_change().dropna().reset_index(drop=True)
        train_returns.name = None
        n_train_ret = len(train_returns)
        warmup_returns = pd.Series(r_arr[:n_seq], index=range(n_train_ret, n_train_ret + n_seq))
        scored_returns_s = pd.Series(
            scored_returns, index=range(n_train_ret + n_seq, n_train_ret + n_seq + len(scored_returns))
        )
        arima_order = fit_arima_order(train_returns)
        arima_preds = rolling_arima_forecast(train_returns, scored_returns_s, arima_order, warmup_returns=warmup_returns)

        # ---- PSI: fold's training feature distribution vs this test block ----
        psi_df = psi_report(actual_train_sc, test_sc, feature_cols)

        # ---- Metrics ----
        hybrid_metrics = compute_metrics(preds, actuals)
        persistence_metrics = compute_metrics(persistence_preds, actuals)
        arima_metrics = compute_metrics(arima_preds, actuals)

        dm_vs_arima = diebold_mariano_test(actuals, preds, arima_preds)
        dm_vs_persistence = diebold_mariano_test(actuals, preds, persistence_preds)

        calib = calibration_report(mc_samples, actuals)

        fold_result = {
            "fold": k,
            "train_end": int(train_end), "test_start": int(test_start), "test_end": int(test_end),
            "arima_order": list(arima_order),
            "hybrid_metrics": hybrid_metrics,
            "persistence_metrics": persistence_metrics,
            "arima_metrics": arima_metrics,
            "dm_hybrid_vs_arima": dm_vs_arima,
            "dm_hybrid_vs_persistence": dm_vs_persistence,
            "calibration": calib,
            "psi_n_alert": int((psi_df["status"] == "alert").sum()),
            "psi_n_moderate": int((psi_df["status"] == "moderate_drift").sum()),
            "psi_max": float(psi_df["psi"].max()),
            "psi_top5": psi_df.head(5).to_dict("records"),
            "alpha_mean": float(alphas.mean()),
            "alpha_std": float(alphas.std()),
        }
        fold_results.append(fold_result)
        log.info(f"FOLD {k} RESULT:\n" + json.dumps(fold_result, indent=2, default=str))

        pooled["hybrid"].append(preds)
        pooled["actual"].append(actuals)
        pooled["persistence"].append(persistence_preds)
        pooled["arima"].append(arima_preds)
        pooled["mc"].append(mc_samples)

    # ── Pooled aggregation ──────────────────────────────────────────────
    log.info("\n[4/4] Aggregating pooled walk-forward results across all folds...")
    pooled_actual = np.concatenate(pooled["actual"])
    pooled_hybrid = np.concatenate(pooled["hybrid"])
    pooled_persist = np.concatenate(pooled["persistence"])
    pooled_arima = np.concatenate(pooled["arima"])
    pooled_mc = np.concatenate(pooled["mc"], axis=1)

    pooled_hybrid_metrics = compute_metrics(pooled_hybrid, pooled_actual)
    pooled_persist_metrics = compute_metrics(pooled_persist, pooled_actual)
    pooled_arima_metrics = compute_metrics(pooled_arima, pooled_actual)

    pooled_dm_arima = diebold_mariano_test(pooled_actual, pooled_hybrid, pooled_arima)
    pooled_dm_persist = diebold_mariano_test(pooled_actual, pooled_hybrid, pooled_persist)
    pooled_calib = calibration_report(pooled_mc, pooled_actual)

    n_total = len(pooled_actual)
    n_correct = int(np.sum(np.sign(pooled_hybrid) == np.sign(pooled_actual)))
    binom_two = stats.binomtest(n_correct, n_total, p=0.5, alternative="two-sided")
    binom_gt = stats.binomtest(n_correct, n_total, p=0.5, alternative="greater")
    ci = binom_two.proportion_ci(confidence_level=0.95)

    final_report = {
        "timestamp": datetime.now().isoformat(),
        # Was a hardcoded literal ("...2010-01-04..2024-12-31") that silently
        # went stale the moment refresh_real_data.py extended the dataset --
        # exactly the class of bug this project's own audit exists to catch
        # (see MODEL_EVALUATION_REPORT.md's "Fabricated documentation" root
        # cause). Derived from Config's actual configured range instead.
        "data_source": f"real Yahoo Finance ^GSPC + macro (VIX/TNX/FVX/IRX/DXY/GLD/TLT), {Config.START_DATE}..{Config.END_DATE}",
        "n_folds": len(boundaries),
        "fold_boundaries": boundaries,
        "fold_results": fold_results,
        "pooled": {
            "n": n_total,
            "n_correct_direction": n_correct,
            "directional_accuracy": n_correct / n_total,
            "binomial_p_two_sided": binom_two.pvalue,
            "binomial_p_one_sided_beat_50pct": binom_gt.pvalue,
            "accuracy_95pct_ci": [ci.low, ci.high],
            "hybrid_metrics": pooled_hybrid_metrics,
            "persistence_metrics": pooled_persist_metrics,
            "arima_metrics": pooled_arima_metrics,
            "dm_hybrid_vs_arima": pooled_dm_arima,
            "dm_hybrid_vs_persistence": pooled_dm_persist,
            "calibration": pooled_calib,
        },
    }

    out_path = Config.MODEL_DIR / "walk_forward_report.json"
    with open(out_path, "w") as f:
        json.dump(final_report, f, indent=2, default=str)

    log.info(f"\nFINAL WALK-FORWARD REPORT saved to {out_path}")
    log.info("POOLED SUMMARY:\n" + json.dumps(final_report["pooled"], indent=2, default=str))

    print("\n" + "#" * 70)
    print("WALK-FORWARD FINAL SUMMARY (pooled across all folds)")
    print("#" * 70)
    print(f"N = {n_total}  |  Directional accuracy = {n_correct/n_total*100:.2f}%  "
          f"(binomial p vs 50%: two-sided={binom_two.pvalue:.4f}, one-sided-beat={binom_gt.pvalue:.4f})")
    print(f"Hybrid   RMSE={pooled_hybrid_metrics['RMSE']:.6f}  Sharpe={pooled_hybrid_metrics['Sharpe_naive_signal_strategy']:.3f}  IC={pooled_hybrid_metrics['IC']:.4f}")
    print(f"ARIMA    RMSE={pooled_arima_metrics['RMSE']:.6f}  Sharpe={pooled_arima_metrics['Sharpe_naive_signal_strategy']:.3f}  IC={pooled_arima_metrics['IC']:.4f}")
    print(f"Persist. RMSE={pooled_persist_metrics['RMSE']:.6f}  Sharpe={pooled_persist_metrics['Sharpe_naive_signal_strategy']:.3f}")
    print(f"DM (hybrid vs ARIMA): stat={pooled_dm_arima['dm_statistic']:.4f} p={pooled_dm_arima['p_value']:.4f} -> {pooled_dm_arima['interpretation']}")
    print(f"DM (hybrid vs persistence): stat={pooled_dm_persist['dm_statistic']:.4f} p={pooled_dm_persist['p_value']:.4f} -> {pooled_dm_persist['interpretation']}")
    print("#" * 70)

    return final_report


if __name__ == "__main__":
    main()
