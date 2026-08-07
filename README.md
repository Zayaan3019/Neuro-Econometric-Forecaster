# Neuro-Econometric Market Alpha Engine

A hybrid forecasting research harness that fuses econometrics (ARDL / bounds-testing, AR baselines) with deep learning (causally-masked Transformer + LSTM) behind a learned gating mechanism, evaluated under a genuine walk-forward protocol with calibration checking, drift monitoring, and significance testing throughout.

**This README makes no performance claim that isn't backed by a number in [`MODEL_EVALUATION_REPORT.md`](MODEL_EVALUATION_REPORT.md), which is reproducible end-to-end by rerunning `python walk_forward_pipeline.py`.** The honest headline: on real S&P 500 data under rolling walk-forward evaluation, the hybrid model shows **no statistically significant directional edge** (48.21% accuracy, N=1394, p=0.19 vs. a 50% null) and is **significantly less accurate** (Diebold-Mariano p<0.0001) than a simple AR(p) baseline. That negative result, and how it was reached, is the actual deliverable here — see the report for the full picture, including the one component that *does* show a modest, significant edge.

---

## What this project actually demonstrates

Not "68.7% directional accuracy" (an earlier version of this README claimed that; no artifact anywhere in the repo ever produced it — see [Audit trail](#audit-trail-what-was-wrong-and-what-was-fixed)). What it does demonstrate, all independently verified and unit-tested:

- **A causally-correct HMM regime detector.** Forward-filtered posteriors only (`P(state_t | obs_1..t)`) — never Viterbi decoding or forward-backward smoothing, both of which leak future information into a "what regime were we in at time t" feature. Proven, not assumed: `tests/test_regime.py` perturbs future observations and asserts the filtered posterior for the past is bit-for-bit unchanged.
- **A genuine Pesaran-Shin-Smith ARDL bounds test**, with AIC/BIC-selected lag order (not a fixed constant) and real asymptotic critical values, replacing a placeholder that previously always returned `(False, 0.0, "requires manual interpretation")` without running anything.
- **MC-Dropout uncertainty that is actually calibration-checked**, not assumed calibrated. On this model, it isn't (nominal 90% intervals cover ~11% of outcomes) — that's reported directly rather than hidden.
- **Population Stability Index drift monitoring** on input features with an explicitly justified bucket count and alert threshold (see `app/monitoring/psi.py` docstring), not a copied "0.2" convention.
- **A true rolling walk-forward evaluation**: 5 expanding-window folds, the model retrained from scratch each fold, evaluated only on the immediately following unseen block. Zero train/test index overlap, verified by `tests/test_walk_forward_boundaries.py`.
- **A Diebold-Mariano significance test** comparing the hybrid model against an ARIMA baseline and a persistence baseline on identical scored timestamps — not just eyeballed RMSE.

---

## Architecture

```
                    ┌─────────────────────────────┐
   Price/Volume ───▶│  Technical indicators (RSI,  │
   Macro (VIX,      │  MACD, ATR, ADX, BBands...)  │
   yields, DXY,      │  + multi-timeframe returns/   │──┐
   Gold, TLT)        │  volatility + calendar feats  │  │
                    └─────────────────────────────┘  │
                                                        ▼
                    ┌──────────────────────────────────────────┐
                    │  Causal HMM regime detector                │
                    │  (forward-filtered P(high-vol) as feature) │
                    └──────────────────┬───────────────────────┘
                                        ▼
        ┌───────────────────────────────────────────────────────┐
        │            NeuroEconometricNet                          │
        │  ┌─────────────────────┐      ┌──────────────────────┐ │
        │  │   Neural branch      │      │   Econometric branch  │ │
        │  │  causal-masked        │      │  rolling AutoReg(p)   │ │
        │  │  Transformer → LSTM   │      │  (1-step return fcst) │ │
        │  │  → latent state h     │      │                        │ │
        │  └──────────┬───────────┘      └───────────┬───────────┘ │
        │             │                                │            │
        │             ▼                                │            │
        │  ┌──────────────────────────┐                │            │
        │  │ Gated fusion: α = σ(W·[h,  │                │            │
        │  │  vol_regime, sentiment])   │                │            │
        │  │ ŷ = α·y_neural + (1-α)·y_ardl               │            │
        │  └──────────────────────────┘◀───────────────┘            │
        └───────────────────────────────────────────────────────────┘
                                        ▼
                          predicted 1-day-ahead return
```

**Key files:**

| File | Role |
|---|---|
| `app/models/neural.py` | Causal-masked Transformer + LSTM encoder (`HybridNeuralEncoder`), volatility-regime MLP |
| `app/models/fusion.py` | `NeuroEconometricNet` — the full hybrid model with gated fusion (`GatedFusionMechanism`) |
| `app/models/econometrics.py` | ARDL model with AIC/BIC lag selection, real Pesaran bounds test, Engle-Granger cointegration |
| `app/models/regime.py` | Causal HMM regime detector — forward-filtered posteriors only |
| `app/data/loader.py`, `app/data/preprocessor.py` | OHLCV/macro loading, technical indicators, stationarity testing |
| `app/engine/uncertainty.py` | MC-Dropout sampling + PIT histogram + reliability-diagram calibration checking |
| `app/engine/statistical_tests.py` | Diebold-Mariano test |
| `app/engine/baselines.py` | Persistence + rolling ARIMA baselines |
| `app/monitoring/psi.py` | Population Stability Index drift monitoring |
| `run_pipeline.py` | Single 70/15/15 chronological-split training run (fast baseline, no leakage) |
| `walk_forward_pipeline.py` | **The real evaluation**: 5-fold rolling walk-forward, retrains per fold, all baselines + DM test + PSI + calibration per fold |
| `serve_api.py` | FastAPI inference service (real-time 1-day-ahead prediction) |
| `tests/` | pytest suite — causality, correctness, and regression tests (see [Testing](#testing)) |

---

## Installation

```bash
git clone https://github.com/Zayaan3019/Neuro-Econometric-Forecaster.git
cd Neuro-Econometric-Forecaster
pip install -r requirements.txt
```

TA-Lib is optional — `app/data/preprocessor.py` falls back to pure-Python technical indicator implementations if the `talib` binary isn't installed (this is what CI/Docker actually run on).

**Data:** `data/GSPC_ohlcv.csv` contains real S&P 500 OHLCV (2010-01-04 → 2024-12-31), fetched directly from Yahoo Finance's public chart API — the `yfinance` *library* is IP-rate-limited in some sandboxed/cloud environments, but the underlying `query1.finance.yahoo.com` endpoint is not; `app/data/loader.py`'s `OHLCVLoader` will attempt a live `yfinance` fetch first and fall back to this local CSV. Macro series (VIX, yields, DXY, GLD, TLT) are similarly cached in `data/macro_cache/`. `data/GSPC_ohlcv_SYNTHETIC_backup.csv` is preserved for reference — it's the **synthetic**, geometric-Brownian-motion data (`generate_sample_data.py`) this project used to train and evaluate on before this audit; it is no longer used anywhere in the pipeline.

---

## Usage

### Train + evaluate (fast, single split)
```bash
python run_pipeline.py
```
Trains once on a 70/15/15 chronological split (scaler fit on train only), prints a validation + test report, saves `models_saved/best_model_v2.pt` and `models_saved/evaluation_report_v2.json`.

### Train + evaluate (the real evaluation: rolling walk-forward)
```bash
python walk_forward_pipeline.py
```
~40-90 minutes on CPU (5 folds, retrained from scratch each fold). Produces `models_saved/walk_forward_report.json` — the source of every number in `MODEL_EVALUATION_REPORT.md`.

### Serve predictions
```bash
python serve_api.py
# or: docker-compose up -d
```
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"ticker": "^GSPC", "horizon": 1}'
```
Requires `models_saved/best_model_v2.pt` to exist (run `run_pipeline.py` first). The `/predict` response includes an explicit disclaimer linking to the evaluation report — this is a reference implementation of a serving pipeline, not a trading-signal product.

### Testing
```bash
pytest tests/ test_market_conditions.py -v
```

---

## Testing

58 pytest tests across 7 files, all currently passing (`pytest tests/ test_market_conditions.py -q` → `58 passed`):

| File | Covers |
|---|---|
| `tests/test_models.py` | Model architecture shapes/forward-passes, ARDL fitting + lag selection, the real bounds test, preprocessing |
| `tests/test_regime.py` | **Causality** of the HMM regime detector — filtered ≠ smoothed, future-perturbation invariance |
| `tests/test_psi.py` | PSI correctness against known synthetic distribution shifts |
| `tests/test_statistical_tests.py` | Diebold-Mariano test correctly identifies which of two forecasts is more accurate |
| `tests/test_uncertainty.py` | MC-Dropout sampling behavior, coverage/PIT computation correctness on synthetic well- and mis-calibrated cases |
| `tests/test_walk_forward_boundaries.py` | No train/test index overlap or gaps across any fold, anywhere |
| `tests/test_regression_phase1_bug.py` | Locks in the fix for the original train/eval target-mismatch bug so it can't silently regress |
| `test_market_conditions.py` | Real 2024-quarter-by-quarter evaluation on the current model/checkpoint |

`test_model_robustness.py` is a legacy diagnostic script (prediction consistency / noise robustness / alpha-gating checks) whose checkpoint loading was fixed to the current format but which has not been fully migrated to the current returns-based methodology or converted into pytest tests — treat it as informational, not part of the verified suite above.

---

## Known limitations (stated, not hidden)

- **No statistically significant directional edge** for the hybrid model on daily S&P 500 returns (see headline result above). Do not use this for live trading signal generation.
- **MC-Dropout uncertainty on this model is not calibrated** (empirical coverage far below nominal) — don't present its intervals as real confidence bounds.
- **ARDL long-run coefficient significance is weak** even where the joint bounds-test F-statistic rejects H0 — read as suggestive, not precise.
- **`OBV` (On-Balance Volume) shows large PSI drift in every walk-forward fold** as an artifact of being an unbounded cumulative series, not a real signal change — a candidate for differencing or removal.
- Only 1-day-ahead prediction is supported end-to-end; the serving API rejects other horizons.
- Sentiment (FinBERT) scaffolding exists (`app/data/sentiment.py`) but isn't wired into the trained pipeline — `sentiment_score` is a constant 0.0 placeholder throughout.

---

## Audit trail: what was wrong, and what was fixed

This repository was audited end-to-end (see conversation record / `verify_phase1_result.py` for the reproduction steps). Summary of what was found and fixed — full detail in `MODEL_EVALUATION_REPORT.md`:

1. **The original "32.84% directional accuracy" was not a real measurement of model quality.** Root cause: the legacy evaluation script compared a model trained to predict a normalized *price level* against ground truth read from a *differenced* series — two different statistical objects. Independently, the legacy checkpoint is architecturally incompatible with the current model code and cannot even be loaded anymore (`RuntimeError` on `load_state_dict`).
2. **Training data was synthetic**, not real market data (`generate_sample_data.py`, a GBM simulator) — despite being loaded ahead of any live API call for every experiment previously run in this repo. Replaced with real Yahoo Finance data.
3. **Look-ahead leakage** in feature normalization (scaler fit on the full pre-split dataset in one training script) — fixed by fitting strictly on each split's training window.
4. **A non-causal "directional accuracy" loss term** (`sign(value - batch_mean)` instead of a true sign-of-return) in the legacy training loop — the current loss (`ProductionLoss` in `run_pipeline.py`) uses IC/sign-of-return directly.
5. **A placeholder ARDL bounds test** that always returned a hardcoded "not implemented" result — replaced with a real Pesaran-Shin-Smith test.
6. **A broken serving API** (loaded the incompatible legacy checkpoint, called the model as if it returned one tensor instead of a 3-tuple) — fixed and verified end-to-end.
7. **Multiple documents with fabricated, unreproduced performance numbers** (68.7% accuracy / 1.87 Sharpe in the old README; 52.93% in a since-removed `PROJECT_COMPLETION.md`; "32.84% → 65-75%" in a since-removed `FIX_SUMMARY.md`) — removed. Every number in this README and in `MODEL_EVALUATION_REPORT.md` is reproducible by rerunning `walk_forward_pipeline.py`.

---

## References

1. Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). "Bounds testing approaches to the analysis of level relationships." *Journal of Applied Econometrics*, 16(3), 289-326.
2. Vaswani et al. (2017). "Attention is All You Need." *NeurIPS*.
3. Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *ICML*.
4. Diebold, F. X. & Mariano, R. S. (1995). "Comparing Predictive Accuracy." *Journal of Business & Economic Statistics*, 13(3), 253-263.
5. Harvey, D., Leybourne, S., & Newbold, P. (1997). "Testing the equality of prediction mean squared errors." *International Journal of Forecasting*, 13(2), 281-291.

---

## Risk disclaimer

This software is for educational and research purposes. It is not financial advice, carries no performance guarantee, and — per its own evaluation results above — does not currently demonstrate a statistically significant trading edge. Do not use it to make investment decisions.

## License

MIT — see [LICENSE](LICENSE).
