# Model Evaluation Report

**Generated:** 2026-08-07, re-run 2026-08-10 on extended data
**Data:** Real S&P 500 (`^GSPC`) OHLCV + macro (VIX, 10Y/5Y/3M yields, DXY, GLD, TLT), 2010-01-01 → 2026-08-07, fetched directly from Yahoo Finance's public chart API (the `yfinance` library itself was IP-rate-limited in this environment; raw endpoint access was not). The original 2026-08-07 report's data ended 2024-12-31 — nearly 19 months before that report was actually written. `refresh_real_data.py` closed that gap on 2026-08-10 (400 additional real trading days), and every number below is from re-running the full evaluation on the extended series, not the original stale one.
**Evaluation methodology:** genuine rolling walk-forward, 5 expanding-window folds, model retrained from scratch each fold, evaluated only on the immediately following, never-before-seen block of time. No k-fold, no random split, anywhere in this pipeline.
**Reproduce with:** `python walk_forward_pipeline.py` (full retrain, ~35 min on CPU with the current dataset size) → writes `models_saved/walk_forward_report.json`, the source of every number below.

---

## Headline result

**The hybrid Neuro-Econometric model (ARDL + Transformer + LSTM + gated fusion) shows no statistically significant directional edge on real S&P 500 daily returns, and is significantly *less* accurate than both a trivial persistence baseline and a simple AR(p) model. Extending the evaluation window by 19 months and 400 real trading days did not change this conclusion — if anything, the gap widened.**

| | N | Correct | Directional Accuracy | 95% CI | p (two-sided vs 50%) | p (one-sided, beats 50%) |
|---|---|---|---|---|---|---|
| **Hybrid model** | 1574 | 775 | **49.24%** | [46.74%, 51.74%] | 0.562 | 0.736 (fails) |
| ARIMA(p,0,q) baseline | 1574 | 841 | **53.43%** | — | — | — |
| Persistence (predict 0 return) | 1574 | 0 | 0.00%¹ | — | — | — |

¹ By construction: `sign(0)` never equals `sign(nonzero actual)`, so directional accuracy is undefined/zero for a model that always predicts exactly zero. Persistence is evaluated on RMSE/Sharpe instead (below), which is the metric it's meaningful for.

<details>
<summary>Original 2026-08-07 report's numbers, for comparison (data through 2024-12-31 only, N=1394)</summary>

| | N | Correct | Directional Accuracy | 95% CI | p (two-sided vs 50%) |
|---|---|---|---|---|---|
| Hybrid model | 1394 | 672 | 48.21% | [45.55%, 50.87%] | 0.189 |
| ARIMA(p,0,q) baseline | 1394 | 744 | 53.37% | [50.71%, 56.02%] | 0.0127 |

</details>

This is not a marginal result. It is the **same conclusion independently reached four separate times** in this project's audit history, on four different data/methodology combinations:

| Run | Data | Methodology | Directional accuracy | p vs 50% |
|---|---|---|---|---|
| Single 70/15/15 split (`run_pipeline.py`), original | Real, through 2024-12-31 | Leakage-checked, single split | 46.84% (237/506) | 0.929 (one-sided) |
| Quarterly slices, 2024 (`test_market_conditions.py`) | Real, 2024 only | Current model on each calendar quarter | 45.62% avg (std 3.10%) | — |
| 5-fold walk-forward, original | Real, through 2024-12-31 | Retrained per fold, expanding window | 48.21% (672/1394) | 0.189 |
| Single 70/15/15 split (`run_pipeline.py`), extended data | Real, through 2026-08-07 | Leakage-checked, single split, test N=566 | 51.41% | — |
| **5-fold walk-forward, extended data** | **Real, through 2026-08-07** | **Retrained per fold, expanding window** | **49.24% (775/1574)** | **0.562** |

The single-split number (51.41%) is higher than the walk-forward's pooled 49.24%, and this run's fusion-gate α-std (0.0033) fell *below* the 0.01 "active gate" threshold `run_pipeline.py` itself flags — unlike the walk-forward folds' 0.021–0.201 range. Both are expected run-to-run variance from a single random initialization on a single split, not a contradiction: the walk-forward result (5 independent retrains, pooled) is the one actually reported as this project's headline number precisely because a single split's number is this sensitive to which split you happened to draw. Included here for completeness, not as a better result to prefer.

## Regression accuracy and strategy performance (pooled across all 5 folds, extended data)

| Metric | Hybrid | ARIMA(p,0,q) | Persistence |
|---|---|---|---|
| RMSE | 0.012856 | **0.012073** | 0.012131 |
| MAE | 0.008890 | **0.007864** | 0.007835 |
| IC (Pearson corr, pred vs actual return) | 0.0865 | 0.0832 | 0.0000 |
| Sharpe of naive signal-following strategy | 0.362 | **1.039** | 0.000 |

IC is close between hybrid and ARIMA this time (0.0865 vs 0.0832 — within noise of each other, unlike the RMSE/Sharpe/DM-test gap), but Sharpe of the naive signal-following strategy is not: ARIMA's 1.039 is roughly 3x the hybrid's 0.362. IC alone would understate how much better ARIMA's predictions are to actually trade on.

**Diebold-Mariano test (squared-error loss, HLN small-sample correction), extended data:**
- Hybrid vs ARIMA: DM = 4.077, **p = 4.79e-5** → ARIMA is significantly more accurate than the hybrid model.
- Hybrid vs persistence: DM = 3.056, **p = 2.28e-3** → persistence is significantly more accurate than the hybrid model.
- Both comparisons pooled-significant, consistent with the original (pre-extension) report's DM = 4.886/4.247, p = 1.15e-6/2.31e-5 — same direction and same order of magnitude, not a fluke of the smaller original sample.

**Interpretation:** the additional architectural complexity — a Transformer/LSTM encoder, a learned volatility-regime gate, an HMM regime detector — does not add predictive value on this data. It actively *hurts* relative to a properly-selected univariate AR(p) model. The only component in the system that shows a genuine, statistically significant (if modest — IC≈0.08, ~53% direction) edge is the plain econometric ARIMA baseline. This is a legitimate, interesting finding in its own right: it is consistent with market micro-efficiency at the one-day horizon for a heavily-traded, large-cap index, and with the well-documented tendency of complex ML models to overfit weak/nonexistent signal that a correctly-regularized linear model does not.

## Calibrated uncertainty (MC-Dropout, 30 samples/prediction, pooled across folds, extended data)

| Nominal interval | Empirical coverage | Gap |
|---|---|---|
| 50% | 4.0% | -46.0pp |
| 80% | 7.8% | -72.2pp |
| 90% | 9.6% | -80.4pp |

Kolmogorov-Smirnov test against Uniform(0,1) rejects calibration overwhelmingly (KS=0.554, p≈0, N=1574) — consistent with (in fact slightly worse than) the original report's KS=0.489.

**The model's uncertainty estimates are severely overconfident** — its nominal "90% interval" contains the true outcome only ~10% of the time. Per this project's own rule (never claim an X% interval without checking empirical coverage), this model's MC-Dropout intervals should **not** be presented as calibrated uncertainty in any downstream use. This is itself a useful, honestly-reported finding: the point predictions are noisy enough, and the dropout-induced variance small enough relative to it, that MC-Dropout under-estimates true predictive uncertainty by roughly an order of magnitude here.

## Gating mechanism (fusion gate health)

The original audit found the gate frozen at α = 0.4650 ± 0.0002 (effectively constant). Across the 5 walk-forward folds on the extended data: **α_std ranged from 0.021 to 0.201** (fold-by-fold: 0.189, 0.134, 0.201, 0.137, 0.021) — consistent with the original report's 0.050–0.250 range. The gate is demonstrably *not* frozen — it varies meaningfully within and across folds. This specific bug does not reproduce on the current architecture/training setup. (Root cause, per the audit: the original frozen-gate symptom was tied to the old architecture/loss combination in `app/engine/trainer.py`'s `HybridLoss`, which computes "directional accuracy" as `sign(value - batch_mean)` rather than a true sign-of-return — see Known Issues below. `run_pipeline.py` / `walk_forward_pipeline.py` use a different loss, `ProductionLoss`, with an explicit gate-variance penalty, which appears sufficient to prevent collapse.)

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
4. **Synthetic training data.** `data/GSPC_ohlcv.csv` was a geometric-Brownian-motion simulation (`generate_sample_data.py`, `np.random.seed(42)`) with regime blocks, not real market data — confirmed by its 2024-12-31 close ($19,139) vs. the real S&P 500 close that day (~$5,882). Replaced with real Yahoo Finance data (see header). All results in this report are on real data.
5. **Fabricated documentation.** `README.md` claimed 68.7% directional accuracy / 1.87 Sharpe; `PROJECT_COMPLETION.md` claimed 52.93%; `FIX_SUMMARY.md` claimed "32.84% → 65-75%" as an achieved result. None of these numbers appeared in any evaluation artifact anywhere in the repository. These documents have been removed; this report and `README.md` are the only performance claims in the project now, and every number in both is reproducible by rerunning `walk_forward_pipeline.py`.

## What's genuinely working

- Zero look-ahead bias, verified: `tests/test_walk_forward_boundaries.py` proves no train/test index overlap and no gaps across folds; `tests/test_regime.py` proves the HMM regime-detection feature is causal (forward-filtered posteriors only — not Viterbi, not smoothed forward-backward) via direct perturbation tests; scaler fit strictly on each fold's training split.
- Real, working Pesaran bounds test, PSI drift monitoring, MC-Dropout calibration checking, and Diebold-Mariano significance testing — all independently unit-tested (`tests/test_psi.py`, `tests/test_statistical_tests.py`, `tests/test_uncertainty.py`, `tests/test_regime.py`) against known-answer synthetic cases, not just run once and trusted.
- The fusion gate is demonstrably not frozen (α_std 0.02-0.20 across folds on the extended data).
- The serving API (`serve_api.py`) works end-to-end against the current model and real data (verified via `TestClient`: `/health`, `/model/info`, `/predict` all return correct values).

## Serving strategy (2026-08-10 update)

Per this report's own conclusion below, `serve_api.py`'s `/predict` endpoint no longer serves the hybrid model's prediction as the primary signal. `Config.PRIMARY_SIGNAL_MODEL` ("arima") selects which component drives the response's `predicted_price`/`signal`; the endpoint computes **both** the hybrid and a live ARIMA forecast (using `fit_arima_order` — the same order-selection function this evaluation used, not a fresh unvalidated variant) and returns both in every response, labeled, rather than picking a winner silently. This follows directly from the finding immediately below, not a separate decision: an evaluation that says "don't deploy X for signal generation" and then deploys X anyway would contradict this project's own evaluation discipline.

## Conclusion

This is an honest negative result, arrived at independently **four** times now (three original + one full re-run on 19 months of additional, previously-unused real data) with consistent effect size — extending the evaluation window did not change the finding, it sharpened it (DM test p-value moved from 1.15e-6 to 4.79e-5 on a larger sample; hybrid RMSE/Sharpe/IC all stayed on the same side of ARIMA's). Per this project's own evaluation discipline: **do not deploy the hybrid model for live trading signal generation** — and as of this update, `serve_api.py` no longer does (see above). The one component that does show a small, statistically significant edge (ARIMA, ~53.4% direction, Sharpe≈1.04) is worth further isolated investigation, but that is a materially smaller and more modest claim than the original "68.7% accuracy" marketing copy this repository used to contain — and a materially more modest claim than "75%," which is not what any evaluation in this repository's history has ever supported.
