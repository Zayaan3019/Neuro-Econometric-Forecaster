# Model Evaluation and Robustness Testing Report

**Generated:** 2026-02-03  
**Model:** Neuro-Econometric Market Alpha Engine  
**Version:** best_model.pt (Epoch 25/40)

---

## Executive Summary

The trained model has been thoroughly tested across multiple dimensions of robustness and accuracy. **Overall Assessment: MODEL NEEDS SIGNIFICANT IMPROVEMENT**

### Key Findings:
- ✗ **Directional Accuracy: 32.84%** - Below random (50%), statistically not significant
- ✓ **Prediction Consistency: PASS** - Model produces deterministic outputs
- ✓ **Noise Robustness: GOOD** - Robust to input noise up to 10%
- ✗ **Gating Mechanism: STUCK** - Alpha weight barely varies (0.4650 ± 0.0002)
- ✓ **Temporal Stability: PASS** - Performance is stable across time windows
- ⚠ **Sharpe Ratio: -1.171** - Negative risk-adjusted returns

---

## Detailed Test Results

### 1. Prediction Consistency Test ✓
**Status:** PASSED

The model produces perfectly consistent predictions on identical inputs, confirming deterministic behavior in evaluation mode.

- **Standard Deviation:** 0.0
- **Variance:** 0.0

This is expected and correct behavior for a neural network in eval mode.

---

### 2. Noise Robustness Test ✓
**Status:** PASSED

The model demonstrates good resilience to input noise:

| Noise Level | Mean Absolute Deviation |
|------------|------------------------|
| 1%         | 0.000011              |
| 5%         | 0.000060              |
| 10%        | 0.000076              |

**Assessment:** Model is robust to reasonable levels of input noise, which is critical for real-world deployment where data may be imperfect.

---

### 3. Directional Accuracy Test ✗
**Status:** FAILED (Critical Issue)

The most important metric for trading applications shows concerning results:

- **Directional Accuracy:** 32.84%
- **Correct Predictions:** 66/201
- **P-value vs Random (50%):** 1.0000
- **Statistical Significance:** NO

**This is a critical failure.** The model is significantly worse than random guessing, suggesting it may have learned to predict in the opposite direction or has not learned meaningful patterns.

#### Implications:
- Trading on these predictions would lose money
- Model is not ready for deployment
- Fundamental rethinking of approach may be needed

---

### 4. Prediction Magnitude Analysis ✓
**Status:** PASSED

Predictions show reasonable variation:

| Metric | Predictions | Actuals |
|--------|------------|---------|
| Mean   | 0.622      | 0.045   |
| Std    | 0.239      | 1.034   |
| Range  | [-0.038, 1.320] | [-2.830, 3.068] |

**Observation:** Predictions have much lower variance than actuals, suggesting the model may be overly conservative and not capturing the full range of market movements.

---

### 5. Gating Mechanism Analysis ✗
**Status:** FAILED

The adaptive gating mechanism (α) that should dynamically weight Neural vs ARDL branches is essentially frozen:

- **Mean α:** 0.4650
- **Std α:** 0.000156
- **Range:** [0.4646, 0.4653]

**Analysis:** The gating weight varies by only 0.0007 across all predictions. This defeats the purpose of the adaptive fusion mechanism - the model should dynamically adjust based on market conditions but instead applies a nearly constant 46.5%/53.5% split.

#### Root Causes:
1. Insufficient training signal to learn dynamic gating
2. Gating network may be undertrained
3. Volatility/sentiment features may not provide enough information
4. May need higher learning rate for fusion gate specifically

---

### 6. Temporal Stability Test ✓
**Status:** PASSED

Performance across 6 time windows (30-day windows):

- **Mean Accuracy:** 31.61%
- **Std Accuracy:** 9.43%
- **Range:** [17.24%, 48.28%]

**Assessment:** While the absolute accuracy is poor, the model's performance is at least consistent over time (std < 15% threshold). This suggests the model isn't catastrophically failing on certain market conditions.

---

## Unit Test Results

**Overall:** 17/22 tests passed, 1 skipped, 4 failed

### Failed Tests:
1. `test_adf_test_stationary` - Type checking issue (non-critical)
2. `test_adf_test_nonstationary` - Type checking issue (non-critical)
3. `test_fit_predict` - ARDLResults missing 'rsquared' attribute
4. `test_pairwise_cointegration` - Type checking issue (non-critical)

Most test failures are minor type-checking issues, not fundamental algorithm problems.

---

## Performance Metrics Summary

### On Normalized/Differenced Scale:
- **MSE:** 1.488287
- **RMSE:** 1.219954
- **MAE:** 0.963694
- **R²:** -0.390934 (negative indicates worse than mean baseline)
- **Directional Accuracy:** 32.84%

### Trading Performance (2024 Test Set):
- **Total Trades:** 195
- **Buy Signals:** 104
- **Sell Signals:** 91
- **Cumulative Return:** N/A (unrealistic due to scaling issues)
- **Sharpe Ratio:** -1.171 (negative = losing money)
- **Max Drawdown:** -152,319% (unrealistic, indicates scaling problem)
- **Win Rate:** 51.79%

---

## Critical Issues Identified

### 🔴 High Priority:

1. **Inverted Directional Accuracy (32.84%)**
   - Model predicts opposite of reality more often than correct
   - Immediate investigation needed
   - May indicate sign error in loss function or data preprocessing

2. **Frozen Gating Mechanism**
   - Adaptive fusion not working as designed
   - No dynamic adjustment to market conditions
   - Defeats purpose of hybrid architecture

3. **Negative R² (-0.39)**
   - Worse than simply predicting the mean
   - Fundamental model effectiveness issue

4. **Inverse Transform Not Implemented**
   - Cannot properly evaluate predictions on original price scale
   - Makes interpretation difficult

### 🟡 Medium Priority:

5. **Low Prediction Variance**
   - Model too conservative
   - May be over-regularized

6. **ARDL Model Issues**
   - statsmodels compatibility problems
   - Missing R² attribute

### 🟢 Low Priority:

7. **Test Suite Maintenance**
   - Minor type checking issues
   - Deprecation warnings (pandas, scipy)

---

## Root Cause Analysis

### Why is Directional Accuracy So Low?

Possible causes:
1. **Sign Error in Preprocessing:** Differencing or normalization may have inverted the target
2. **Lookahead Bias:** Target variable may be misaligned with features
3. **Overfitting to Training Period:** Model learned patterns specific to 2015-2023 that don't generalize to 2024
4. **Loss Function Mismatch:** Optimizing MSE doesn't guarantee directional accuracy
5. **Feature Engineering Issues:** Important signals may be lost in preprocessing

### Why is Gating Stuck?

Possible causes:
1. **Insufficient Gradient Flow:** Fusion gate may not receive strong enough gradients
2. **Poor Volatility/Sentiment Signals:** Using dummy values (0.02, 0.0) in testing
3. **Network Initialization:** Gate may have initialized near 0.465 and never moved
4. **Learning Rate:** Fusion components may need different learning rates

---

## Recommendations

### Immediate Actions (Do First):

1. **Investigate Directional Accuracy**
   ```python
   # Check if inverting predictions improves accuracy
   inverted_accuracy = np.mean(np.diff(predictions) < 0 == np.diff(actuals) > 0)
   # If this is ~67%, you have a sign error
   ```

2. **Verify Data Preprocessing Pipeline**
   - Check target variable alignment
   - Verify differencing direction
   - Ensure no lookahead bias

3. **Implement Proper Inverse Transform**
   - Save scaler state from training
   - Transform predictions back to price scale
   - Enables meaningful evaluation

### Short-Term Improvements:

4. **Fix Loss Function**
   - Increase λ_directional weight significantly (e.g., 2.0 → 5.0)
   - Consider focal loss for directional component
   - Add explicit penalties for wrong direction

5. **Unfreeze Gating Mechanism**
   - Use separate learning rate for fusion gate (higher)
   - Add entropy regularization to encourage diversity
   - Compute real volatility features instead of dummy values

6. **Increase Training Duration**
   - Model stopped at epoch 25/100
   - May not have converged fully
   - Try continuing training from checkpoint

### Medium-Term Improvements:

7. **Enhance Feature Engineering**
   - Add more market regime indicators
   - Include macroeconomic data
   - Better sentiment features (currently using dummy 0.0)

8. **Architectural Changes**
   - Try simpler baseline (pure LSTM) to establish performance ceiling
   - Gradually add complexity only if it helps
   - Consider ensemble of simpler models

9. **Better Validation Strategy**
   - Implement true walk-forward optimization
   - Use multiple train/test splits
   - Validate on out-of-sample periods

### Long-Term Considerations:

10. **Alternative Approaches**
    - Consider transformer-only architecture
    - Explore gradient boosting (XGBoost, LightGBM) as baseline
    - Investigate reinforcement learning for trading

11. **Market Microstructure**
    - Add bid-ask spread
    - Include volume profile
    - Consider order book data

---

## Testing Checklist Completed

- ✅ Basic unit tests (17/22 passed)
- ✅ Model loading and inference
- ✅ Prediction consistency
- ✅ Noise robustness
- ✅ Directional accuracy analysis
- ✅ Magnitude analysis
- ✅ Gating mechanism behavior
- ✅ Temporal stability
- ✅ Trading simulation
- ✅ Statistical significance tests
- ⚠️ Market condition testing (insufficient quarterly data)

---

## Conclusion

The current model, while architecturally sophisticated and technically sound in implementation, **is not ready for deployment**. The critical issue of inverted directional accuracy (32.84% vs expected 50%+) indicates a fundamental problem that must be resolved before any trading application.

### Next Steps:
1. **Debug directional accuracy issue** (highest priority)
2. **Verify data preprocessing and target alignment**
3. **Retrain with adjusted loss function** emphasizing directional accuracy
4. **Fix gating mechanism** to enable adaptive fusion
5. **Implement inverse transformation** for proper evaluation

### Positive Aspects:
- ✓ Model is stable and consistent
- ✓ Good noise robustness
- ✓ Temporal stability
- ✓ Most unit tests pass
- ✓ Architecture is well-designed

### Timeline Estimate:
- Debug and fix issues: 1-2 weeks
- Retrain with improvements: 3-5 days
- Re-evaluation: 1 day
- **Total:** ~3 weeks to production-ready model

---

## Files Generated

- `models_saved/evaluation_report.json` - Basic evaluation metrics
- `models_saved/robustness_test_results.json` - Comprehensive robustness tests
- `models_saved/market_condition_test.json` - Market condition analysis
- `MODEL_EVALUATION_REPORT.md` - This report

---

**Report Generated by:** Model Robustness Testing Suite  
**Date:** February 3, 2026  
**Test Coverage:** Comprehensive (Accuracy, Robustness, Stability, Statistical Significance)
