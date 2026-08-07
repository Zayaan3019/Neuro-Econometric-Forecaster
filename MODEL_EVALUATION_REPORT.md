# Model Evaluation Report

**Generated:** 2026-08-07
**Data:** Real S&P 500 (`^GSPC`) OHLCV + macro (VIX, 10Y/5Y/3M yields, DXY, GLD, TLT), 2010-01-04 → 2024-12-31, fetched directly from Yahoo Finance's public chart API (the `yfinance` library itself was IP-rate-limited in this environment; raw endpoint access was not).
**Evaluation methodology:** genuine rolling walk-forward, 5 expanding-window folds, model retrained from scratch each fold, evaluated only on the immediately following, never-before-seen block of time. No k-fold, no random split, anywhere in this pipeline.
**Reproduce with:** `python walk_forward_pipeline.py` (full retrain, ~40-90 min on CPU) → writes `models_saved/walk_forward_report.json`, the source of every number below.

---

## Headline result

**The hybrid Neuro-Econometric model (ARDL + Transformer + LSTM + gated fusion) shows no statistically significant directional edge on real S&P 500 daily returns, and is significantly *less* accurate than both a trivial persistence baseline and a simple AR(p) model.**

| | N | Correct | Directional Accuracy | 95% CI | p (two-sided vs 50%) | p (one-sided, beats 50%) |
|---|---|---|---|---|---|---|
| **Hybrid model** | 1394 | 672 | **48.21%** | [45.55%, 50.87%] | 0.189 | 0.914 (fails) |
| ARIMA(p,0,q) baseline | 1394 | 744 | **53.37%** | [50.71%, 56.02%] | **0.0127** | **0.0064** (significant) |
| Persistence (predict 0 return) | 1394 | 0 | 0.00%¹ | — | — | — |

¹ By construction: `sign(0)` never equals `sign(nonzero actual)`, so directional accuracy is undefined/zero for a model that always predicts exactly zero. Persistence is evaluated on RMSE/Sharpe instead (below), which is the metric it's meaningful for.

This is not a marginal result. It is the **same conclusion independently reached three separate times** in this audit, on three different data/methodology combinations:

| Run | Data | Methodology | Directional accuracy | p vs 50% |
|---|---|---|---|---|
| Single 70/15/15 split (`run_pipeline.py`) | Real | Leakage-checked, single split | 46.84% (237/506) | 0.929 (one-sided) |
| Quarterly slices, 2024 (`test_market_conditions.py`) | Real | Current model on each calendar quarter | 45.62% avg (std 3.10%) | — |
| **5-fold walk-forward (`walk_forward_pipeline.py`)** | **Real** | **Retrained per fold, expanding window** | **48.21% (672/1394)** | **0.189** |

## Regression accuracy and strategy performance (pooled across all 5 folds)

| Metric | Hybrid | ARIMA(p,0,q) | Persistence |
|---|---|---|---|
| RMSE | 0.013548 | **0.012607** | 0.012667 |
| MAE | 0.009038 | **0.008192** | 0.008160 |
| IC (Pearson corr, pred vs actual return) | 0.0069 | **0.0835** | 0.0000 |
| Sharpe of naive signal-following strategy | **-0.101** | **0.901** | 0.000 |

**Diebold-Mariano test (squared-error loss, HLN small-sample correction):**
- Hybrid vs ARIMA: DM = 4.886, **p = 1.15e-6** → ARIMA is significantly more accurate than the hybrid model.
- Hybrid vs persistence: DM = 4.247, **p = 2.31e-5** → persistence is significantly more accurate than the hybrid model.
- This held in 4 of 5 individual folds (fold 4 was the lone exception, DM p=0.14, not significant) and was significant pooled.

**Interpretation:** the additional architectural complexity — a Transformer/LSTM encoder, a learned volatility-regime gate, an HMM regime detector — does not add predictive value on this data. It actively *hurts* relative to a properly-selected univariate AR(p) model. The only component in the system that shows a genuine, statistically significant (if modest — IC≈0.08, ~53% direction) edge is the plain econometric ARIMA baseline. This is a legitimate, interesting finding in its own right: it is consistent with market micro-efficiency at the one-day horizon for a heavily-traded, large-cap index, and with the well-documented tendency of complex ML models to overfit weak/nonexistent signal that a correctly-regularized linear model does not.

## Calibrated uncertainty (MC-Dropout, 30 samples/prediction, pooled across folds)

| Nominal interval | Empirical coverage | Gap |
|---|---|---|
| 50% | 4.8% | -45.2pp |
| 80% | 8.3% | -71.7pp |
| 90% | 10.8% | -79.2pp |

PIT histogram is strongly U-shaped (519/1394 in the [0,0.1) bin, 753/1394 in the [0.9,1.0] bin, versus ~139/bin expected under uniformity); Kolmogorov-Smirnov test against Uniform(0,1) rejects calibration overwhelmingly (KS=0.489, p≈3.6e-307).

**The model's uncertainty estimates are severely overconfident** — its nominal "90% interval" contains the true outcome only ~11% of the time. Per this project's own rule (never claim an X% interval without checking empirical coverage), this model's MC-Dropout intervals should **not** be presented as calibrated uncertainty in any downstream use. This is itself a useful, honestly-reported finding: the point predictions are noisy enough, and the dropout-induced variance small enough relative to it, that MC-Dropout under-estimates true predictive uncertainty by roughly an order of magnitude here.

## Gating mechanism (fusion gate health)

The original audit found the gate frozen at α = 0.4650 ± 0.0002 (effectively constant). Across the 5 walk-forward folds on the current architecture: **α_std ranged from 0.050 to 0.250** (fold-by-fold: 0.050, 0.205, 0.250, 0.133, 0.083). The gate is demonstrably *not* frozen — it varies meaningfully within and across folds. This specific bug does not reproduce on the current architecture/training setup. (Root cause, per the audit: the original frozen-gate symptom was tied to the old architecture/loss combination in `app/engine/trainer.py`'s `HybridLoss`, which computes "directional accuracy" as `sign(value - batch_mean)` rather than a true sign-of-return — see Known Issues below. `run_pipeline.py` / `walk_forward_pipeline.py` use a different loss, `ProductionLoss`, with an explicit gate-variance penalty, which appears sufficient to prevent collapse.)

## Population Stability Index (input feature drift, per fold, reference = that fold's training window)

10-decile buckets, reference = fold training split, alert threshold 0.25, moderate-drift band [0.10, 0.25) (see `app/monitoring/psi.py` docstring for the full threshold justification). Every fold flagged 14-23 of 39 features as "alert"-level drift — expected and correct given this window spans 2010-2024, including the 2020 COVID crash and the fastest Fed hiking cycle in decades:

- **`OBV`** (On-Balance Volume, a cumulative running-sum indicator) shows PSI ≈ 8.27-8.28 in *every* fold — mechanically expected for an unbounded cumulative series and not economically meaningful; it is a candidate for exclusion or re-scaling (e.g. differencing) in future work.
- **`yield_spread`, `yield_3m`, `yield_10y`** show large drift in 3-4 of 5 folds — a real, correctly-detected regime shift (rates went from near-zero for most of the training window to >5% by 2023).
- This is PSI performing exactly as intended: flagging genuine, economically-explicable distribution shift, not sampling noise (see `tests/test_psi.py` for correctness verification against known synthetic shifts).

## ARDL bounds test (diagnostic; Pesaran-Shin-Smith)

A prior version of `ARDLBoundsTest.bounds_f_test()` was a hardcoded placeholder that always returned `(False, 0.0, {"message": "requires manual interpretation..."})` — it never ran an actual test. It has been replaced with a genuine implementation (`app/models/econometrics.py`, using `statsmodels`' `UECM`/bounds-test machinery), with AIC/BIC lag-order selection (not a fixed lag count) and real Pesaran-Shin-Smith asymptotic critical values.

As a diagnostic (not part of the trading model), tested on S&P 500 close level vs. VIX level (2010-2024, last 800 trading days): F-statistic = 7.035 (order selected: AR=1, DL=2 by BIC) exceeds the I(1) upper bound at 5% (4.81) and 1% (6.32) — **reject H0 of no long-run relationship** between the index level and VIX level. Integration-order check confirms the standard justification for using bounds testing here: Close is I(1) (ADF level p=0.95, first-difference p≈0), VIX is I(0) (ADF level p=0.003) — a genuinely mixed-order case, which Engle-Granger cointegration testing cannot handle but PSS bounds testing can. Note: the long-run coefficient on VIX in the cointegrating vector, while economically sensible in sign, was not itself individually significant at conventional levels (t-test p≈0.38) in this sample — the joint F-test result should be read as suggestive evidence of a level relationship, not a precisely-estimated one.

## Root causes found and fixed (Phase 1)

1. **Train/eval target mismatch (the actual cause of the original 32.84% figure).** The legacy `train_model.py` trained on the normalized **price level** (`Close` at t+1); the legacy `evaluate_model.py` read `test_df.iloc[i, -1]` as "ground truth," which — confirmed by running the real preprocessing pipeline — is `Close_diff`, the normalized **first difference**. Comparing a level-predicting model against a differenced-series target is an apples-to-oranges comparison that can produce an arbitrary, meaningless number regardless of whether the model learned anything. Independently, `evaluate_model.py` currently **cannot even load** `best_model.pt` against the current model code (`RuntimeError: Error(s) in loading state_dict` — architecturally incompatible, stale checkpoint) — reproduced by directly running it (see conversation record).
2. **Look-ahead leakage in feature normalization.** `Preprocessor.fit_transform()` fit `StandardScaler` on whatever DataFrame was passed in, with no train-only restriction; `train_model_production.py` called it on the full pre-split dataset. `run_pipeline.py` / `walk_forward_pipeline.py` fit the scaler on the training split only, per fold.
3. **Non-causal "directional accuracy" in the training loss.** `HybridLoss` (`app/engine/trainer.py`) computed direction as `sign(value - batch_mean)` — "above/below this batch's average" — not a true up/down signal. `ProductionLoss` (`run_pipeline.py`) uses IC / sign-of-return directly.
4. **Synthetic training data.** `data/GSPC_ohlcv.csv` was a geometric-Brownian-motion simulation (`generate_sample_data.py`, `np.random.seed(42)`) with regime blocks, not real market data — confirmed by its 2024-12-31 close ($19,139) vs. the real S&P 500 close that day (~$5,882). Replaced with real Yahoo Finance data (see header). All results in this report are on real data.
5. **Fabricated documentation.** `README.md` claimed 68.7% directional accuracy / 1.87 Sharpe; `PROJECT_COMPLETION.md` claimed 52.93%; `FIX_SUMMARY.md` claimed "32.84% → 65-75%" as an achieved result. None of these numbers appeared in any evaluation artifact anywhere in the repository. These documents have been removed; this report and `README.md` are the only performance claims in the project now, and every number in both is reproducible by rerunning `walk_forward_pipeline.py`.

## What's genuinely working

- Zero look-ahead bias, verified: `tests/test_walk_forward_boundaries.py` proves no train/test index overlap and no gaps across folds; `tests/test_regime.py` proves the HMM regime-detection feature is causal (forward-filtered posteriors only — not Viterbi, not smoothed forward-backward) via direct perturbation tests; scaler fit strictly on each fold's training split.
- Real, working Pesaran bounds test, PSI drift monitoring, MC-Dropout calibration checking, and Diebold-Mariano significance testing — all independently unit-tested (`tests/test_psi.py`, `tests/test_statistical_tests.py`, `tests/test_uncertainty.py`, `tests/test_regime.py`) against known-answer synthetic cases, not just run once and trusted.
- The fusion gate is demonstrably not frozen (α_std 0.05-0.25 across folds).
- The serving API (`serve_api.py`) works end-to-end against the current model and real data (verified via `TestClient`: `/health`, `/model/info`, `/predict` all return correct values).

## Conclusion

This is an honest negative result, arrived at independently three times with consistent effect size, and confirmed via a proper significance test (Diebold-Mariano) that the added model complexity is not just "no better than," but **measurably worse than**, a simple econometric baseline. Per this project's own evaluation discipline: **do not deploy the hybrid model for live trading signal generation.** The one component that does show a small, statistically significant edge (ARIMA, IC≈0.08, ~53.4% direction, p=0.013) is worth further isolated investigation, but that is a materially smaller and more modest claim than the original "68.7% accuracy" marketing copy this repository used to contain.
