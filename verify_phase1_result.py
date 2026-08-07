"""
Independent verification of run_pipeline.py's real-data test-set result.

Reloads best_model_v2.pt fresh, rebuilds the exact same test split from
data/GSPC_ohlcv.csv (real S&P 500 data) + data/macro_cache, and recomputes
directional accuracy, confusion matrix, and a binomial significance test
independently of run_pipeline.py's own evaluate() function.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
from scipy import stats

from app.config import Config
from app.models.fusion import NeuroEconometricNet
from run_pipeline import (
    load_ohlcv, load_macro, engineer_features, precompute_ardl_predictions,
    scale_df, MarketDataset
)

Config.set_seed()
device = Config.DEVICE

print("Loading real OHLCV + macro data...")
ohlcv = load_ohlcv()
macro = load_macro(Config.START_DATE, Config.END_DATE)
full_df = engineer_features(ohlcv, macro)

ckpt = torch.load(Config.MODEL_DIR / "best_model_v2.pt", map_location=device, weights_only=False)
feature_cols = ckpt["feature_cols"]
print(f"Checkpoint feature_cols ({len(feature_cols)}):", feature_cols)

assert all(c in full_df.columns for c in feature_cols), "Feature mismatch between checkpoint and freshly engineered data!"

ardl_preds = precompute_ardl_predictions(full_df["Close"], window=Config.ARDL_WINDOW, lags=Config.ARDL_LAGS)

n = len(full_df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

test_df = full_df.iloc[val_end:]
test_close = ohlcv["Close"].iloc[val_end:]
test_ardl = ardl_preds.iloc[val_end:]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.mean_ = np.array(ckpt["scaler_mean"])
scaler.scale_ = np.array(ckpt["scaler_scale"])
scaler.var_ = scaler.scale_ ** 2
scaler.n_features_in_ = len(feature_cols)

test_sc = test_df.copy()
test_sc[feature_cols] = scaler.transform(test_df[feature_cols].values)

ds_test = MarketDataset(test_sc, test_close, test_ardl, feature_cols=feature_cols, seq_len=Config.SEQ_LENGTH)
print(f"Test samples (independently rebuilt): {len(ds_test)}")

model = NeuroEconometricNet(
    input_dim=ckpt["input_dim"],
    hidden_dim=ckpt["config"]["hidden_dim"],
    nhead=ckpt["config"]["nhead"],
    num_lstm_layers=ckpt["config"]["num_layers"],
    dropout=ckpt["config"]["dropout"],
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

loader = torch.utils.data.DataLoader(ds_test, batch_size=256, shuffle=False)

preds, actuals = [], []
with torch.no_grad():
    for feat, ardl, vol, sent, tgt in loader:
        pred, alpha, _ = model(feat.to(device), ardl.to(device), vol.to(device), sent.to(device))
        preds.append(pred.cpu().numpy())
        actuals.append(tgt.numpy())

preds = np.concatenate(preds).flatten()
actuals = np.concatenate(actuals).flatten()

pred_dir = np.sign(preds)
act_dir = np.sign(actuals)

n_test = len(preds)
correct = int(np.sum(pred_dir == act_dir))
acc = correct / n_test

print("\n" + "="*60)
print("INDEPENDENT VERIFICATION — TEST SET (real S&P 500 data)")
print("="*60)
print(f"N = {n_test}")
print(f"Correct = {correct}")
print(f"Directional Accuracy = {acc*100:.2f}%")

# Confusion matrix: predicted up/down vs actual up/down
pred_up = pred_dir > 0
act_up = act_dir > 0
tp = int(np.sum(pred_up & act_up))    # predicted up, actual up
fp = int(np.sum(pred_up & ~act_up))   # predicted up, actual down
fn = int(np.sum(~pred_up & act_up))   # predicted down, actual up
tn = int(np.sum(~pred_up & ~act_up))  # predicted down, actual down
print(f"\nConfusion matrix (predicted vs actual):")
print(f"                 Actual Up   Actual Down")
print(f"  Predicted Up   {tp:>9}   {fp:>11}")
print(f"  Predicted Down {fn:>9}   {tn:>11}")

res_two = stats.binomtest(correct, n_test, p=0.5, alternative='two-sided')
res_gt = stats.binomtest(correct, n_test, p=0.5, alternative='greater')
ci = res_two.proportion_ci(confidence_level=0.95)
print(f"\nBinomial test vs 50% null:")
print(f"  p-value (two-sided): {res_two.pvalue:.4f}")
print(f"  p-value (one-sided, H1: acc>50%): {res_gt.pvalue:.4f}")
print(f"  95% CI on accuracy: [{ci.low*100:.2f}%, {ci.high*100:.2f}%]")

mse = float(np.mean((preds - actuals)**2))
mae = float(np.mean(np.abs(preds - actuals)))
rmse = float(np.sqrt(mse))
mape = float(np.mean(np.abs((preds - actuals) / (np.abs(actuals) + 1e-8)))) * 100
ic = float(np.corrcoef(preds, actuals)[0,1])
print(f"\nRegression metrics:")
print(f"  RMSE = {rmse:.6f}")
print(f"  MAE  = {mae:.6f}")
print(f"  MAPE = {mape:.2f}%")
print(f"  IC (Pearson corr, pred vs actual return) = {ic:.4f}")

# Persistence baseline: predict tomorrow's return = today's realized return (naive)
naive_pred_dir = np.sign(np.roll(actuals, 1))
naive_pred_dir[0] = 0
naive_correct = int(np.sum(naive_pred_dir[1:] == act_dir[1:]))
naive_acc = naive_correct / (n_test - 1)
print(f"\nPersistence baseline (predict sign(return_t) = sign(return_t-1)):")
print(f"  Accuracy = {naive_acc*100:.2f}%  ({naive_correct}/{n_test-1})")

print("\nDone.")
