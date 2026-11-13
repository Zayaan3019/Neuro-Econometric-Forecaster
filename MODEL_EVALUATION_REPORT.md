# Model Evaluation Report

**Generated:** 2026-08-07 · re-run on extended data 2026-08-10 · re-run 2026-11-12 after fixing the econometric branch · **re-run 2026-11-14 with cost-aware Sharpe and saved per-day predictions (current numbers)**
**Data:** Real S&P 500 (`^GSPC`) OHLCV + macro (VIX, 10Y/5Y/3M yields, DXY, GLD, TLT), 2010-01-01 → 2026-08-07, fetched directly from Yahoo Finance's public chart API (the `yfinance` library itself was IP-rate-limited in this environment; raw endpoint access was not).
**Evaluation methodology:** genuine rolling walk-forward, 5 expanding-window folds, model retrained from scratch each fold, evaluated only on the immediately following, never-before-seen block of time. No k-fold, no random split, anywhere in this pipeline.
**Reproduce with:** `python walk_forward_pipeline.py` (full retrain, ~35-45 min on CPU) → writes `models_saved/walk_forward_report.json`, the source of every number below, and `models_saved/walk_forward_predictions.csv`, every one of the 1,574 test-day predictions (fold, date, actual return, hybrid / ARIMA / persistence forecast, gate α), so any metric here can be recomputed without retraining. ARIMA and persistence reproduce exactly. **The hybrid does not reproduce bit-for-bit** (see the next section).

---

## What changed on 2026-11-12, and why every hybrid number moved

The hybrid's econometric branch is described as a rolling AR(5) forecast. **Until 2026-11-12 it never was one.** `precompute_ardl_predictions()` called `AutoReg(...).fit(disp=False)`; the installed statsmodels (0.14) rejects `disp` with a `TypeError`, and a bare `except:` caught it and substituted the last close for every forecast. So the "econometric forecast" fed to the model was the naive no-change forecast on every row, and every earlier hybrid number in this report's history was produced by that model. The forecast was also aligned one day early (it forecast close[t] from data through t-1 rather than close[t+1] from data through t).

It was found by a new test, `tests/test_serving_parity.py`, written to check that the live API feeds the model exactly what training fed it. That test also found the serving path built its inputs differently from training (the 60-row window included the newest row, the volatility input was unscaled, and the AR input was NaN and therefore always 0). Both are fixed:

- `ar_next_close()` is the single AR forecast used by training and serving; only `ValueError` / `LinAlgError` fall back to the naive forecast, and the run **raises if more than 1% of fits fall back**.
- `serve_api.model_inputs()` reproduces `MarketDataset`'s inputs for the newest row; six tests pin the parity, the alignment, and that the AR output is a fitted forecast rather than the fallback.
- The served checkpoint was retrained and the full walk-forward re-run. The numbers below are from that run. ARIMA and persistence are unaffected (they never used this code path), so their numbers are unchanged.

## Run-to-run variance of the hybrid (found 2026-11-14)

The 2026-11-14 re-run changed only reporting (cost-aware Sharpe, saved predictions), with the same seed, data and folds. ARIMA and persistence came out identical to the last digit. The hybrid did not: pooled direction accuracy **47.27% → 45.81%**, pooled Sharpe **+0.02 → −0.69**, and fold 2's Sharpe flipped from **+0.96 to −0.62**. `Config.set_seed()` seeds PyTorch, NumPy and Python, but the retrain is still not deterministic on this CPU setup; the exact source is not isolated.

What this means for reading the report:
- The hybrid's numbers carry run-to-run noise of a few points of direction accuracy and about ±1 of Sharpe per fold. Any single hybrid figure is one draw.
- The conclusion does not depend on the draw: in both runs the hybrid is below 50% and significantly less accurate than ARIMA and persistence by Diebold-Mariano.
- The ARIMA figures are exact and do not move.

The tables below are the 2026-11-14 run, the first with every per-day prediction saved; the 2026-11-12 figures are kept alongside.

## Headline result

**The hybrid Neuro-Econometric model (AR(5) branch + Transformer + LSTM + gated fusion) has no directional edge on real S&P 500 daily returns. With the AR branch working, its pooled direction accuracy is 45.81% (47.27% on the previous run), significantly *below* 50% (two-sided p = 0.00095), and it is significantly less accurate than both an order-selected ARIMA baseline and a predict-zero persistence baseline.**

| | N | Correct | Directional Accuracy | 95% CI | p (two-sided vs 50%) | p (one-sided, beats 50%) |
|---|---|---|---|---|---|---|
| **Hybrid model** | 1574 | 721 | **45.81%** | [43.32%, 48.31%] | 0.00095 | 0.9996 (fails) |
| Hybrid, previous run (2026-11-12) | 1574 | 744 | 47.27% | [44.78%, 49.77%] | 0.032 | 0.986 (fails) |
| ARIMA(p,0,q) baseline | 1574 | 841 | **53.43%** | — | — | — |
| Persistence (predict 0 return) | 1574 | 0 | 0.00%¹ | — | — | — |

¹ By construction: `sign(0)` never equals `sign(nonzero actual)`, so directional accuracy is undefined/zero for a model that always predicts exactly zero. Persistence is evaluated on RMSE/Sharpe instead (below), which is the metric it's meaningful for.

**Below 50% is not a usable signal in reverse.** It moves by about 1.5 points between two seeded runs, it is one of many statistics computed on this evaluation, and most folds are individually indistinguishable from 50%. "Flip the sign" chosen after seeing this result would be selection on the test data. The defensible reading is the same as before: no edge, and worse than the simple baseline.

### Per fold

| Fold | Test rows (N scored) | Hybrid direction | ARIMA direction | ARIMA order | Hybrid IC | ARIMA IC | Hybrid Sharpe | ARIMA Sharpe | DM vs ARIMA (p) | gate α mean / std |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2295-2670 (314) | 50.6% | 61.5% | (1,0,0) | 0.067 | 0.354 | 0.44 | 2.20 | 1.08 (0.28) | 0.76 / 0.21 |
| 2 | 2670-3045 (314) | 47.1% | 52.2% | (1,0,0) | 0.042 | 0.046 | -0.62 | 1.18 | 2.60 (0.0096) | 0.54 / 0.25 |
| 3 | 3045-3420 (314) | 46.5% | 46.2% | (2,0,0) | -0.023 | -0.013 | -0.90 | -0.00 | 2.67 (0.0079) | 0.73 / 0.19 |
| 4 | 3420-3795 (314) | 41.7% | 55.1% | (2,0,0) | -0.037 | -0.023 | -2.14 | 0.63 | 3.55 (4.5e-4) | 0.31 / 0.09 |
| 5 | 3795-4174 (318) | 43.1% | 52.2% | (2,0,0) | 0.061 | 0.039 | -2.09 | 0.75 | 7.13 (6.9e-12) | 0.82 / 0.02 |

ARIMA's columns are identical to the previous run; the hybrid's are not (previous run, hybrid direction by fold: 49.4, 49.0, 51.9, 43.0, 43.1%). In both runs the gap is largest in the two most recent folds, and only fold 1 leaves the hybrid indistinguishable from ARIMA by DM in this run (folds 1-3 did in the previous one).

### History of this result

| Run | Data | Methodology | Directional accuracy | p vs 50% |
|---|---|---|---|---|
| Single 70/15/15 split, original | Real, through 2024-12-31 | Leakage-checked, single split | 46.84% (237/506) | 0.929 (one-sided) |
| Quarterly slices, 2024 (`test_market_conditions.py`), original checkpoint | Real, 2024 only | Served model on each calendar quarter | 45.62% avg (std 3.10%) | — |
| 5-fold walk-forward, original | Real, through 2024-12-31 | Retrained per fold | 48.21% (672/1394) | 0.189 |
| 5-fold walk-forward, extended data (naive AR input) | Real, through 2026-08-07 | Retrained per fold | 49.24% (775/1574) | 0.562 |
| Single split, AR branch fixed | Real, through 2026-08-07 | Single split, test set | 44.52% | — |
| Quarterly slices, 2024, AR-fixed checkpoint | Real, 2024 only | Served model on each calendar quarter (60-63 days each) | 42.75% avg (std 1.82%) | — |
| 5-fold walk-forward, AR branch fixed (2026-11-12) | Real, through 2026-08-07 | Retrained per fold | 47.27% (744/1574) | 0.032 (below 50%) |
| **5-fold walk-forward, same code + cost metrics (2026-11-14)** | **Real, through 2026-08-07** | **Retrained per fold, same seed** | **45.81% (721/1574)** | **0.00095 (below 50%)** |

Every run before 2026-11-12 used the naive AR input described above. The conclusion has not changed in any of them; fixing the branch made the hybrid slightly worse, not better.

## Regression accuracy and strategy performance (pooled across all 5 folds)

| Metric | Hybrid (2026-11-14) | Hybrid (2026-11-12) | ARIMA(p,0,q) | Persistence |
|---|---|---|---|---|
| RMSE | 0.012948 | 0.012869 | **0.012073** | 0.012131 |
| MAE | 0.008728 | 0.008682 | 0.007864 | **0.007835** |
| IC (Pearson corr, pred vs actual return) | 0.0051 | 0.0238 | **0.0832** | 0.0000 |
| Sharpe of naive signal-following strategy, before costs | -0.694 | 0.023 | **1.039** | 0.000 |
| Sharpe after 1 bp per unit of turnover | -0.711 | — | **0.918** | 0.000 |
| Sharpe after 2 bps | -0.729 | — | **0.796** | 0.000 |
| Sharpe after 5 bps | -0.781 | — | **0.433** | 0.000 |
| Average daily turnover (Σ\|Δ position\|) | 0.13 | — | 0.92 | 0 |

**Transaction costs (added 2026-11-14).** The strategy holds `sign(forecast)` and is charged `bps × |position_t − position_{t−1}|` each day. ARIMA flips its position on most days (turnover 0.92/day), so costs bite: its Sharpe falls from 1.04 to 0.92 at 1 bp, 0.80 at 2 bps and 0.43 at 5 bps. At index-futures costs (around 1 bp) most of the edge survives; at retail costs it does not. These are still gross of slippage and financing, on one index, over one sample.

**Diebold-Mariano test (squared-error loss, HLN small-sample correction):**
- Hybrid vs ARIMA: DM = 4.381, **p = 1.3e-5** → ARIMA is significantly more accurate than the hybrid model (2026-11-12 run: 3.966, p = 7.6e-5).
- Hybrid vs persistence: DM = 3.860, **p = 1.2e-4** → persistence is significantly more accurate than the hybrid model (2026-11-12 run: 3.486, p = 5.0e-4).

**Interpretation:** the additional architectural complexity — a Transformer/LSTM encoder, a learned volatility-regime gate, an HMM regime detector — does not add predictive value on this data and hurts relative to a properly-selected univariate ARIMA. The only component with a statistically significant (if modest — IC≈0.08, ~53% direction) edge is the plain ARIMA baseline. This is consistent with near-efficiency at the one-day horizon for a heavily traded index, and with the tendency of flexible models to fit noise that a regularised linear model ignores.

## Calibrated uncertainty (MC-Dropout, 30 samples/prediction, pooled across folds)

| Nominal interval | Empirical coverage | Gap | Previous run |
|---|---|---|---|
| 50% | 4.1% | -45.9pp | 4.9% |
| 80% | 7.2% | -72.8pp | 9.0% |
| 90% | 8.9% | -81.1pp | 11.2% |

PIT histogram (10 bins): 522, 20, 20, 8, 12, 24, 8, 10, 18, 932 — U-shaped, i.e. outcomes fall outside the predicted distribution on both sides. Kolmogorov-Smirnov against Uniform(0,1) rejects calibration overwhelmingly (KS=0.561, p≈0, N=1574).

**The model's uncertainty estimates are severely overconfident** — its nominal "90% interval" contains the true outcome about 9-11% of the time. These MC-Dropout intervals must **not** be presented as calibrated uncertainty in any downstream use.

## Gating mechanism (fusion gate health)

The original audit found the gate frozen at α = 0.4650 ± 0.0002. In this run α_std per fold is 0.21, 0.25, 0.19, 0.09, 0.02 (2026-11-12 run: 0.23, 0.20, 0.20, 0.14, 0.02), so the gate is not frozen. α_mean is 0.76, 0.54, 0.73, 0.31, 0.82 (2026-11-12: 0.78, 0.62, 0.76, 0.71, 0.98): how much weight the gate gives the neural branch is itself one of the things that changes between seeded runs. (The frozen-gate symptom was tied to the legacy `HybridLoss`; `ProductionLoss` has an explicit gate-variance penalty.)

## Population Stability Index (input feature drift, per fold, reference = that fold's training window)

10-decile buckets, reference = fold training split, alert threshold 0.25, moderate-drift band [0.10, 0.25) (see `app/monitoring/psi.py` docstring for the full threshold justification). Every fold flagged 15-27 of 39 features as "alert"-level drift (fold-by-fold: 16, 15, 27, 19, 15) — expected and correct given this window now spans 2010-2026, including the 2020 COVID crash, the fastest Fed hiking cycle in decades, and whatever regime the most recent 400 trading days represent:

- **`OBV`** (On-Balance Volume, a cumulative running-sum indicator) shows PSI ≈ 8.28 in *every* fold (8.278–8.281) — mechanically expected for an unbounded cumulative series and not economically meaningful; it is a candidate for exclusion or re-scaling (e.g. differencing) in future work. Unchanged from the original report — this is a structural property of the feature, not something extending the data would move.
- This is PSI performing exactly as intended: flagging genuine, economically-explicable distribution shift, not sampling noise (see `tests/test_psi.py` for correctness verification against known synthetic shifts).

## ARDL bounds test (diagnostic; Pesaran-Shin-Smith)

A prior version of `ARDLBoundsTest.bounds_f_test()` was a hardcoded placeholder that always returned `(False, 0.0, {"message": "requires manual interpretation..."})` — it never ran an actual test. It has been replaced with a genuine implementation (`app/models/econometrics.py`, using `statsmodels`' `UECM`/bounds-test machinery), with AIC/BIC lag-order selection (not a fixed lag count) and real Pesaran-Shin-Smith asymptotic critical values.

As a diagnostic (not part of the trading model), tested on S&P 500 close level vs. VIX level (2010-2024, last 800 trading days): F-statistic = 7.035 (order selected: AR=1, DL=2 by BIC) exceeds the I(1) upper bound at 5% (4.81) and 1% (6.32) — **reject H0 of no long-run relationship** between the index level and VIX level. Integration-order check confirms the standard justification for using bounds testing here: Close is I(1) (ADF level p=0.95, first-difference p≈0), VIX is I(0) (ADF level p=0.003) — a genuinely mixed-order case, which Engle-Granger cointegration testing cannot handle but PSS bounds testing can. Note: the long-run coefficient on VIX in the cointegrating vector, while economically sensible in sign, was not itself individually significant at conventional levels (t-test p≈0.38) in this sample — the joint F-test result should be read as suggestive evidence of a level relationship, not a precisely-estimated one.

## Root causes found and fixed (Phase 1)

1. **Train/eval target mismatch (the actual cause of the original 32.84% figure).** The legacy `train_model.py` trained on the normalized **price level** (`Close` at t+1); the legacy `evaluate_model.py` read `test_df.iloc[i, -1]` as "ground truth," which — confirmed by running the real preprocessing pipeline — is `Close_diff`, the normalized **first difference**. Comparing a level-predicting model against a differenced-series target is an apples-to-oranges comparison that can produce an arbitrary, meaningless number regardless of whether the model learned anything. Independently, `evaluate_model.py` currently **cannot even load** `best_model.pt` against the current model code (`RuntimeError: Error(s) in loading state_dict` — architecturally incompatible, stale checkpoint) — reproduced by directly running it (see conversation record).
2. **Look-ahead leakage in feature normalization.** `Preprocessor.fit_transform()` fit `StandardScaler` on whatever DataFrame was passed in, with no train-only restriction; `train_model_production.py` called it on the full pre-split dataset. `run_pipeline.py` / `walk_forward_pipeline.py` fit the scaler on the training split only, per fold.
3. **Non-causal "directional accuracy" in the training loss.** `HybridLoss` (`app/engine/trainer.py`) computed direction as `sign(value - batch_mean)` — "above/below this batch's average" — not a true up/down signal. `ProductionLoss` (`run_pipeline.py`) uses IC / sign-of-return directly.
4. **Synthetic training data.** `data/GSPC_ohlcv.csv` was a geometric-Brownian-motion simulation (`generate_sample_data.py`, `np.random.seed(42)`) with regime blocks, not real market data — confirmed by its 2024-12-31 close ($19,139) vs. the real S&P 500 close that day (~$5,882). Replaced with real Yahoo Finance data (see header); the simulator and its output file have since been deleted from the repository. All results in this report are on real data.
5. **Fabricated documentation.** `README.md` claimed 68.7% directional accuracy / 1.87 Sharpe; `PROJECT_COMPLETION.md` claimed 52.93%; `FIX_SUMMARY.md` claimed "32.84% → 65-75%" as an achieved result. None of these numbers appeared in any evaluation artifact anywhere in the repository. These documents have been removed; this report and `README.md` are the only performance claims in the project now, and every number in both is reproducible by rerunning `walk_forward_pipeline.py`.
6. **The econometric branch was never fitted (found 2026-11-12).** `AutoReg.fit(disp=False)` raised `TypeError` on every call, a bare `except:` swallowed it, and every "AR forecast" was the last close. Fixed, with a 1% fallback cap that raises, and the forecast re-aligned to close[t+1]. See the top of this report.
7. **Training/serving skew (found 2026-11-12).** The live API built the model's inputs differently from training (window included the newest row, unscaled volatility input, AR input always 0). Fixed in `serve_api.model_inputs()` and pinned by `tests/test_serving_parity.py`.

## What's genuinely working

- Zero look-ahead bias, verified: `tests/test_walk_forward_boundaries.py` proves no train/test index overlap and no gaps across folds; `tests/test_regime.py` proves the HMM regime feature is causal (forward-filtered posteriors only) via future-perturbation tests; `tests/test_serving_parity.py` proves the AR forecast for row t uses nothing after t; scaler fit strictly on each fold's training split.
- Real, working Pesaran bounds test, PSI drift monitoring, MC-Dropout calibration checking, and Diebold-Mariano significance testing — all unit-tested against known-answer synthetic cases.
- The fusion gate is not frozen (α_std 0.02-0.25 across folds).
- The live API feeds the served model exactly the inputs the evaluated model was trained on (`tests/test_serving_parity.py`).

## Serving strategy (2026-08-10 update)

Per this report's own conclusion below, `serve_api.py`'s `/predict` endpoint no longer serves the hybrid model's prediction as the primary signal. `Config.PRIMARY_SIGNAL_MODEL` ("arima") selects which component drives the response's `predicted_price`/`signal`; the endpoint computes **both** the hybrid and a live ARIMA forecast (using `fit_arima_order` — the same order-selection function this evaluation used, not a fresh unvalidated variant) and returns both in every response, labeled, rather than picking a winner silently. This follows directly from the finding immediately below, not a separate decision: an evaluation that says "don't deploy X for signal generation" and then deploys X anyway would contradict this project's own evaluation discipline.

## Conclusion

An honest negative result, now reached with the econometric branch working as described. Fixing it moved the hybrid from 49.24% to 47.27% direction accuracy and from IC 0.087 to 0.024, and a same-seed re-run gave 45.81% and IC 0.005; none of it changed the conclusion. **Do not use the hybrid model for trading signals** — and `serve_api.py` does not: it serves ARIMA as the primary signal and returns the hybrid's forecast alongside, labelled. The one component with a small, statistically significant edge (ARIMA, ~53.4% direction, Sharpe ≈ 1.04 before costs, 0.92 after 1 bp and 0.43 after 5 bps) is worth isolated investigation, but that is a far more modest claim than the "68.7% accuracy" this repository once advertised.
