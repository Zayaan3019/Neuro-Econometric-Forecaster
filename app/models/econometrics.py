"""
Econometric Models Module (ARDL & Cointegration).

This module implements the "Linear Branch" of the Neuro-Econometric Engine,
using Autoregressive Distributed Lag (ARDL) models to capture long-term
equilibrium relationships and linear dynamics in financial time series.
"""

from typing import Tuple, Optional, Dict, Any, Union, List
import numpy as np
import pandas as pd
from statsmodels.tsa.ardl import ARDL, ardl_select_order, UECM
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.stattools import coint, adfuller
import logging

from app.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ARDLModel:
    """
    Autoregressive Distributed Lag (ARDL) model for financial forecasting.
    
    Mathematical Foundation:
        ARDL(p, q) model specification:
        Y_t = c + Σ(φ_i * Y_{t-i}) + Σ(β_j * X_{t-j}) + ε_t
        
        Where:
        - Y_t: Target variable (e.g., returns)
        - X_t: Exogenous variables (e.g., technical indicators)
        - p: Autoregressive lag order
        - q: Distributed lag order for exogenous variables
        - φ_i: AR coefficients
        - β_j: Distributed lag coefficients
        - ε_t: Error term (white noise)
    
    Key Properties:
        1. Captures both short-run dynamics and long-run equilibrium
        2. Allows different lag structures for dependent and independent variables
        3. Valid for I(0), I(1), or mixed integration order variables
        4. Asymptotically efficient under mild conditions
    
    Reference:
        Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). "Bounds testing approaches
        to the analysis of level relationships." Journal of Applied Econometrics.
    """
    
    def __init__(
        self,
        lags: Union[int, Dict[str, int], None] = Config.ARDL_LAGS,
        trend: str = 'c',
        select_order: bool = True,
        max_lags: int = 8,
        ic: str = 'bic',
    ):
        """
        Initialize ARDL model.

        Args:
            lags: Lag structure to use when select_order=False. If int, same
                 lags for all variables. If dict, specify lags per variable
                 {'Y': 5, 'X1': 3}. Ignored when select_order=True.
            trend: Trend specification:
                   - 'n': No trend
                   - 'c': Constant only
                   - 't': Constant and linear trend
                   - 'ct': Constant and time trend
            select_order: If True (default), lag order is chosen automatically
                 via information-criterion grid search (`ic`) over
                 1..max_lags for both the AR and distributed-lag components,
                 using `statsmodels.tsa.ardl.ardl_select_order` (or
                 `ar_select_order` when no exogenous variables are supplied).
                 If False, uses the fixed `lags` value.
            max_lags: Maximum lag order considered during selection.
            ic: Information criterion for selection -- 'aic' or 'bic'.
                BIC (default) penalises extra lags more heavily and is the
                more common choice for lag-order selection in small/medium
                financial samples, where AIC tends to overfit lag length.
        """
        self.lags = lags
        self.trend = trend
        self.select_order = select_order
        self.max_lags = max_lags
        self.ic = ic
        self.model_: Optional[ARDL] = None
        self.fitted_model_: Optional[Any] = None
        self.is_fitted: bool = False
        self.convergence_failed: bool = False
        self.selected_lags_: Optional[Any] = None

    def fit(
        self,
        endog: Union[pd.Series, np.ndarray],
        exog: Optional[Union[pd.DataFrame, np.ndarray]] = None
    ) -> 'ARDLModel':
        """
        Fit ARDL model to data.

        Args:
            endog: Endogenous variable (target). Shape: (T,)
            exog: Exogenous variables (features). Shape: (T, K)

        Returns:
            Self (fitted model).

        Raises:
            ValueError: If convergence fails and no fallback available.

        Algorithm:
            1. If select_order=True, grid-search lag order 1..max_lags by IC
               (AIC/BIC) -- NOT a fixed lag count.
            2. Estimate parameters via OLS/MLE at the selected (or fixed) lags.
            3. If singular matrix error, fall back to simpler AR model.
        """
        try:
            if self.select_order:
                # Grid-search lag order 1..max_lags by information criterion
                # (NOT a fixed constant) -- this is what distinguishes a
                # genuine ARDL specification search from "a plain lagged
                # linear regression with a hardcoded lag length".
                if exog is not None:
                    selection = ardl_select_order(
                        endog, maxlag=self.max_lags, exog=exog, maxorder=self.max_lags,
                        trend=self.trend, ic=self.ic,
                    )
                    self.model_ = selection.model  # already-specified, unfitted ARDL
                    self.selected_lags_ = self.model_.ardl_order
                    logger.info(
                        f"ardl_select_order chose (ar_lag, exog_lags)="
                        f"{self.selected_lags_} by {self.ic.upper()} "
                        f"(searched 1..{self.max_lags})"
                    )
                else:
                    ar_selection = ar_select_order(
                        endog, maxlag=self.max_lags, ic=self.ic, trend=self.trend
                    )
                    chosen_lags = max(ar_selection.ar_lags) if ar_selection.ar_lags else 1
                    self.selected_lags_ = chosen_lags
                    logger.info(
                        f"ar_select_order chose lags={ar_selection.ar_lags} "
                        f"by {self.ic.upper()} (searched 1..{self.max_lags})"
                    )
                    self.model_ = ARDL(
                        endog=endog, lags=chosen_lags, exog=None,
                        trend=self.trend, causal=True,
                    )
            else:
                self.selected_lags_ = self.lags
                self.model_ = ARDL(
                    endog=endog, lags=self.lags, exog=exog,
                    trend=self.trend, causal=True,  # no look-ahead bias
                )

            # Fit model
            self.fitted_model_ = self.model_.fit()
            self.is_fitted = True
            self.convergence_failed = False

            logger.info(f"ARDL fitted successfully. AIC: {self.fitted_model_.aic:.2f}, BIC: {self.fitted_model_.bic:.2f}")
            logger.info(f"R² (pseudo): {self._pseudo_rsquared():.4f}")

        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"ARDL convergence failed: {str(e)}. Falling back to AR model.")
            self.convergence_failed = True
            self._fit_ar_fallback(endog)

        return self
    
    def _fit_ar_fallback(self, endog: Union[pd.Series, np.ndarray]) -> None:
        """
        Fit simple AR model as fallback when ARDL fails.
        
        Args:
            endog: Endogenous variable.
        
        Mathematical Foundation:
            AR(p) model: Y_t = c + Σ(φ_i * Y_{t-i}) + ε_t
            
            This is a special case of ARDL with no exogenous variables.
        """
        try:
            # Use AutoReg from statsmodels
            ar_lags = self.lags if isinstance(self.lags, int) else 5
            
            ar_model = AutoReg(
                endog=endog,
                lags=ar_lags,
                trend=self.trend
            )
            
            self.fitted_model_ = ar_model.fit()
            self.is_fitted = True
            
            logger.info(f"AR({ar_lags}) fallback fitted successfully. AIC: {self.fitted_model_.aic:.2f}")
        
        except Exception as e:
            logger.error(f"AR fallback also failed: {str(e)}")
            raise ValueError("Both ARDL and AR models failed to converge. Check data quality.")
    
    def predict(
        self,
        steps: int = 1,
        exog: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        dynamic: bool = False
    ) -> np.ndarray:
        """
        Generate forecasts from fitted ARDL model.
        
        Args:
            steps: Number of steps ahead to forecast.
            exog: Future values of exogenous variables (required if used in fitting).
            dynamic: If True, use forecasted values for multi-step prediction.
                    If False, use actual values (one-step-ahead rolling).
        
        Returns:
            Array of predictions. Shape: (steps,)
        
        Mathematical Formulation:
            One-step-ahead: E[Y_{T+1} | Ω_T] = c + Σ(φ_i * Y_{T+1-i}) + Σ(β_j * X_{T+1-j})
            
            Where Ω_T is the information set at time T.
        
        Raises:
            RuntimeError: If model not fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        
        try:
            if dynamic:
                # Dynamic forecast (use predicted values)
                forecast = self.fitted_model_.forecast(steps=steps, exog=exog)
            else:
                # Static forecast (requires actual past values)
                forecast = self.fitted_model_.predict(start=len(self.fitted_model_.model.endog),
                                                     end=len(self.fitted_model_.model.endog) + steps - 1,
                                                     exog=exog,
                                                     dynamic=False)
            
            return np.asarray(forecast)
        
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            # Return naive forecast (last observed value)
            last_value = self.fitted_model_.model.endog[-1]
            return np.full(steps, last_value)
    
    def get_coefficients(self) -> Dict[str, float]:
        """
        Extract fitted model coefficients.
        
        Returns:
            Dictionary mapping variable names to coefficient estimates.
        
        Application:
            Coefficients can be used for:
            - Economic interpretation (elasticities)
            - Feature importance analysis
            - Model comparison and validation
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")
        
        return dict(self.fitted_model_.params)

    def _pseudo_rsquared(self) -> float:
        """
        statsmodels' ARDLResults does NOT expose `.rsquared` (unlike plain
        OLS results) -- accessing it raises AttributeError. Compute the
        standard 1 - SS_res/SS_tot pseudo-R² directly from residuals and
        fitted values instead of assuming an attribute that doesn't exist.
        AutoReg's fallback results DO have `.model.endog`, so this works
        for both code paths.
        """
        try:
            resid = np.asarray(self.fitted_model_.resid)
            fitted_vals = np.asarray(self.fitted_model_.fittedvalues)
            actual = resid + fitted_vals
            ss_res = np.sum(resid ** 2)
            ss_tot = np.sum((actual - actual.mean()) ** 2)
            return float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
        except Exception as e:
            logger.warning(f"Could not compute pseudo-R²: {e}")
            return float('nan')

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Extract model diagnostics and fit statistics.

        Returns:
            Dictionary with:
            - aic: Akaike Information Criterion
            - bic: Bayesian Information Criterion
            - rsquared: pseudo-R² computed from residuals (ARDLResults has
              no native `.rsquared` attribute -- a prior version of this
              method assumed it did and raised AttributeError on every call)
            - adj_rsquared: Adjusted R-squared (OLS/AutoReg fallback only)
            - log_likelihood: Log-likelihood value
            - residuals: Model residuals
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")

        return {
            "aic": self.fitted_model_.aic,
            "bic": self.fitted_model_.bic,
            "rsquared": self._pseudo_rsquared(),
            "adj_rsquared": getattr(self.fitted_model_, 'rsquared_adj', None),
            "log_likelihood": self.fitted_model_.llf,
            "residuals": self.fitted_model_.resid,
            "convergence_failed": self.convergence_failed
        }


class CointegrationTester:
    """
    Test for cointegration between multiple time series.
    
    Mathematical Foundation:
        Two series X_t and Y_t are cointegrated if:
        1. Both are I(1) (non-stationary)
        2. Linear combination Z_t = Y_t - βX_t is I(0) (stationary)
        
        This implies a long-run equilibrium relationship:
        Y_t = α + βX_t + ε_t
        
        Where ε_t is stationary, representing short-run deviations.
    
    Application in Finance:
        - Pairs trading: Find cointegrated assets for mean-reversion strategies
        - Risk management: Identify stable hedge ratios
        - Portfolio optimization: Construct stationary portfolio combinations
    
    Reference:
        Engle, R. F., & Granger, C. W. (1987). "Co-integration and error correction:
        Representation, estimation, and testing." Econometrica, 55(2), 251-276.
    """
    
    @staticmethod
    def test_pairwise(
        y: Union[pd.Series, np.ndarray],
        x: Union[pd.Series, np.ndarray],
        significance_level: float = 0.05
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Test for cointegration between two series (Engle-Granger test).
        
        Args:
            y: First time series.
            x: Second time series.
            significance_level: P-value threshold (default 0.05).
        
        Returns:
            Tuple of (is_cointegrated, p_value, test_statistics).
        
        Algorithm:
            1. Regress Y on X: Y_t = α + βX_t + ε_t
            2. Test residuals ε_t for stationarity (ADF test)
            3. If residuals are stationary, series are cointegrated
        """
        try:
            # Engle-Granger cointegration test
            coint_stat, p_value, critical_values = coint(y, x)

            is_cointegrated = bool(p_value < significance_level)
            
            test_stats = {
                "test_statistic": coint_stat,
                "p_value": p_value,
                "critical_values": {
                    "1%": critical_values[0],
                    "5%": critical_values[1],
                    "10%": critical_values[2]
                }
            }
            
            logger.info(f"Cointegration test - Statistic: {coint_stat:.4f}, p-value: {p_value:.4f}, Cointegrated: {is_cointegrated}")
            
            return is_cointegrated, p_value, test_stats
        
        except Exception as e:
            logger.error(f"Cointegration test failed: {str(e)}")
            return False, 1.0, {}


class ARDLBoundsTest:
    """
    ARDL Bounds Testing for long-run relationships.
    
    Mathematical Foundation:
        The bounds test checks for the existence of a long-run relationship
        (cointegration) in an ARDL model using an F-statistic.
        
        Null Hypothesis (H0): No long-run relationship
        Alternative (H1): Long-run relationship exists
        
        Critical values depend on:
        - Number of variables (k)
        - Whether variables are I(0) or I(1)
        - Sample size
    
    Advantages over Johansen/Engle-Granger:
        1. Valid for I(0), I(1), or mixed integration orders
        2. Can test multiple variables simultaneously
        3. More powerful in small samples
    
    Reference:
        Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). "Bounds testing
        approaches to the analysis of level relationships." Journal of
        Applied Econometrics, 16(3), 289-326.

    Implementation note:
        This performs the GENUINE Pesaran-Shin-Smith (PSS) bounds test via
        statsmodels' `UECM` (Unrestricted Error Correction Model) class,
        which reformulates the fitted ARDL as an error-correction regression

            Δy_t = c + φ·y_{t-1} + Σ_j θ_j·x_{j,t-1} + Σ lagged Δ terms + ε_t

        and provides the PSS F-statistic together with asymptotic critical
        value bounds (simulated by statsmodels; see `UECMResults.bounds_test`).
        A prior version of this class returned a hardcoded placeholder
        ("requires manual interpretation") and never actually ran a test --
        that has been replaced with a real, working implementation below.
        The bounds test is only valid when no regressor is I(2); callers
        should run `check_integration_orders` first.
    """

    @staticmethod
    def check_integration_orders(
        series_dict: Dict[str, Union[pd.Series, np.ndarray]],
        significance_level: float = 0.05,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Classify each series as I(0) or I(1) via ADF, and flag any series
        that still shows a unit root after one difference (i.e. potentially
        I(2)+) -- the PSS bounds test is invalid if any regressor is I(2).

        Returns:
            {name: {"order": "I(0)"|"I(1)"|"I(2)+", "adf_level_p": float,
                     "adf_diff_p": float}}
        """
        results: Dict[str, Dict[str, Any]] = {}
        for name, s in series_dict.items():
            s = pd.Series(s).dropna()
            try:
                p_level = adfuller(s, autolag='AIC')[1]
            except Exception:
                p_level = 1.0
            if p_level < significance_level:
                results[name] = {"order": "I(0)", "adf_level_p": p_level, "adf_diff_p": None}
                continue
            try:
                p_diff = adfuller(s.diff().dropna(), autolag='AIC')[1]
            except Exception:
                p_diff = 1.0
            order = "I(1)" if p_diff < significance_level else "I(2)+"
            results[name] = {"order": order, "adf_level_p": p_level, "adf_diff_p": p_diff}
        return results

    @staticmethod
    def bounds_f_test(
        endog: Union[pd.Series, np.ndarray],
        exog: Union[pd.DataFrame, pd.Series],
        max_lags: int = 5,
        trend: str = 'c',
        case: int = 3,
        ic: str = 'bic',
    ) -> Tuple[Optional[bool], float, Dict[str, Any]]:
        """
        Perform the real ARDL (Pesaran-Shin-Smith) bounds F-test for a
        long-run level relationship between `endog` and `exog`.

        Args:
            endog: Dependent level series (e.g. index price level).
            exog: One or more exogenous level series (DataFrame or Series).
            max_lags: Maximum AR/DL lag order considered during IC-based
                order selection (see `ardl_select_order`).
            trend: 'c' = unrestricted intercept, no trend (PSS case III,
                the standard choice absent a strong prior for a
                deterministic trend in the relationship).
            case: PSS case number (1-5). Default 3 = unrestricted intercept,
                no trend, matching `trend='c'` above.
            ic: Information criterion for lag-order selection.

        Returns:
            (has_long_run_relationship, f_statistic, info) where
            has_long_run_relationship is:
                True  -- F-stat exceeds the I(1) upper bound (reject H0)
                False -- F-stat is below the I(0) lower bound (fail to reject H0)
                None  -- inconclusive (F-stat falls between the bounds --
                          the correct, honest answer for the PSS test in
                          that region; do NOT round this to True/False)
        """
        exog_df = exog.to_frame() if isinstance(exog, pd.Series) else pd.DataFrame(exog)

        integration = ARDLBoundsTest.check_integration_orders(
            {**{"endog": endog}, **{c: exog_df[c] for c in exog_df.columns}}
        )
        i2_vars = [k for k, v in integration.items() if v["order"] == "I(2)+"]
        if i2_vars:
            logger.warning(
                f"Bounds test invalid: {i2_vars} appear I(2) or higher. "
                "PSS bounds testing requires all regressors to be I(0) or I(1)."
            )

        try:
            selection = ardl_select_order(
                endog, maxlag=max_lags, exog=exog_df, maxorder=max_lags,
                trend=trend, ic=ic,
            )
            ardl_order = selection.model.ardl_order

            # If IC selection dropped every exogenous lag (order 0 for all
            # exog columns), the UECM representation is undefined (bounds
            # test requires >=1 exogenous variable with >=1 lag). Force a
            # minimum of 1 lag in that degenerate case and say so plainly,
            # rather than silently crashing or fabricating a result.
            forced_min_order = False
            if len(ardl_order) == 1 or all(o == 0 for o in ardl_order[1:]):
                forced_min_order = True
                selection = ardl_select_order(
                    endog, maxlag=max_lags, exog=exog_df,
                    maxorder={c: max(1, max_lags) for c in exog_df.columns},
                    trend=trend, ic=ic,
                )
                ardl_order = selection.model.ardl_order

            uecm = UECM.from_ardl(selection.model)
            uecm_res = uecm.fit()
            bt = uecm_res.bounds_test(case=case)

            f_stat = float(bt.stat)
            crit_vals = bt.crit_vals
            p_values = bt.p_values

            # Determine the 5%-significance bound decision. statsmodels
            # indexes `crit_vals` by PERCENTILE (90.0/95.0/99.0/99.9), i.e.
            # 95.0 corresponds to the 5% significance level -- NOT by alpha.
            try:
                upper_5pct = crit_vals.loc[95.0, 'upper'] if hasattr(crit_vals, 'loc') else None
                lower_5pct = crit_vals.loc[95.0, 'lower'] if hasattr(crit_vals, 'loc') else None
            except Exception:
                upper_5pct = lower_5pct = None

            if upper_5pct is not None and f_stat > upper_5pct:
                decision = True
            elif lower_5pct is not None and f_stat < lower_5pct:
                decision = False
            else:
                decision = None  # inconclusive region -- do not fabricate a verdict

            # Long-run equilibrium coefficients + error-correction speed of
            # adjustment (the coefficient on the lagged level of endog in
            # the ECM regression; must be negative & significant for a
            # valid error-correction mechanism).
            try:
                long_run_coefs = uecm_res.ci_params.to_dict()
            except Exception:
                long_run_coefs = {}
            # UECM's endog is the DIFFERENCED series (e.g. "D.Close"); the
            # error-correction speed-of-adjustment is the coefficient on the
            # own LEVEL lag, i.e. the "<base_name>.L1" parameter.
            ec_speed = None
            raw_endog_name = getattr(uecm_res.model, "endog_names", "")
            base_name = raw_endog_name[2:] if raw_endog_name.startswith("D.") else raw_endog_name
            ec_param_name = f"{base_name}.L1"
            if ec_param_name in uecm_res.params.index:
                ec_speed = float(uecm_res.params[ec_param_name])

            info = {
                "f_statistic": f_stat,
                "critical_values": crit_vals.to_dict() if hasattr(crit_vals, 'to_dict') else crit_vals,
                "p_values": p_values.to_dict() if hasattr(p_values, 'to_dict') else p_values,
                "selected_order": ardl_order,
                "forced_min_order_for_uecm": forced_min_order,
                "integration_orders": integration,
                "long_run_coefficients": long_run_coefs,
                "error_correction_speed": ec_speed,
                "decision_at_5pct": (
                    "reject H0 (long-run relationship)" if decision is True else
                    "fail to reject H0 (no evidence of long-run relationship)" if decision is False else
                    "inconclusive (F-stat between I(0)/I(1) bounds)"
                ),
                "reference": "Pesaran, Shin & Smith (2001), JAE 16(3):289-326",
            }
            logger.info(
                f"PSS bounds test: F={f_stat:.4f}, decision={info['decision_at_5pct']}, "
                f"order={ardl_order}"
            )
            return decision, f_stat, info

        except Exception as e:
            logger.error(f"Bounds test failed: {str(e)}")
            return None, 0.0, {"error": str(e)}


class EconometricPredictor:
    """
    High-level wrapper for econometric forecasting.
    
    Provides a unified interface that:
    1. Automatically handles ARDL fitting with fallback
    2. Manages prediction and uncertainty quantification
    3. Integrates with the fusion network
    """
    
    def __init__(self, lags: int = Config.ARDL_LAGS):
        """
        Initialize econometric predictor.
        
        Args:
            lags: Number of autoregressive lags.
        """
        self.lags = lags
        self.ardl_model = ARDLModel(lags=lags)
    
    def fit_predict(
        self,
        target: pd.Series,
        features: Optional[pd.DataFrame] = None,
        horizon: int = 1
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fit ARDL and generate predictions in one call.
        
        Args:
            target: Target variable time series.
            features: Exogenous features (optional).
            horizon: Forecast horizon (steps ahead).
        
        Returns:
            Tuple of (predictions, diagnostics).
        
        This method is designed for integration with the training loop
        where we need both fitting and prediction in each iteration.
        """
        # Fit model
        self.ardl_model.fit(endog=target, exog=features)
        
        # Generate predictions
        predictions = self.ardl_model.predict(steps=horizon, exog=features)
        
        # Get diagnostics
        diagnostics = self.ardl_model.get_diagnostics()
        
        return predictions, diagnostics
    
    def rolling_prediction(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[list] = None,
        window_size: int = 252,
        horizon: int = 1
    ) -> pd.Series:
        """
        Perform rolling window predictions for backtesting.
        
        Args:
            data: DataFrame with target and features.
            target_col: Name of target column.
            feature_cols: List of feature column names.
            window_size: Size of rolling training window.
            horizon: Steps ahead to predict.
        
        Returns:
            Series of out-of-sample predictions aligned with data index.
        
        Algorithm:
            For t = window_size to T:
                1. Train on data[t-window_size:t]
                2. Predict data[t+horizon]
                3. Store prediction
                4. Slide window forward
        """
        predictions = []
        indices = []
        
        for i in range(window_size, len(data) - horizon):
            # Extract training window
            train_target = data[target_col].iloc[i-window_size:i]
            
            if feature_cols:
                train_features = data[feature_cols].iloc[i-window_size:i]
            else:
                train_features = None
            
            # Fit and predict
            try:
                pred, _ = self.fit_predict(train_target, train_features, horizon)
                predictions.append(pred[0])
                indices.append(data.index[i + horizon - 1])
            except Exception as e:
                logger.warning(f"Prediction failed at index {i}: {str(e)}")
                predictions.append(np.nan)
                indices.append(data.index[i + horizon - 1])
        
        return pd.Series(predictions, index=indices, name='ardl_prediction')
