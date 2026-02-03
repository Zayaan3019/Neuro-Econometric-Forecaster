"""
Econometric Models Module (ARDL & Cointegration).

This module implements the "Linear Branch" of the Neuro-Econometric Engine,
using Autoregressive Distributed Lag (ARDL) models to capture long-term
equilibrium relationships and linear dynamics in financial time series.
"""

from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import coint
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
        lags: Union[int, Dict[str, int]] = Config.ARDL_LAGS,
        trend: str = 'c'
    ):
        """
        Initialize ARDL model.
        
        Args:
            lags: Lag structure. If int, same lags for all variables.
                 If dict, specify lags per variable {'Y': 5, 'X1': 3}.
            trend: Trend specification:
                   - 'n': No trend
                   - 'c': Constant only
                   - 't': Constant and linear trend
                   - 'ct': Constant and time trend
        """
        self.lags = lags
        self.trend = trend
        self.model_: Optional[ARDL] = None
        self.fitted_model_: Optional[Any] = None
        self.is_fitted: bool = False
        self.convergence_failed: bool = False
    
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
            1. Construct ARDL model with specified lags
            2. Estimate parameters via Maximum Likelihood (MLE)
            3. If singular matrix error, fall back to simpler AR model
        """
        logger.info(f"Fitting ARDL model with lags={self.lags}, trend={self.trend}")
        
        try:
            # Create ARDL model
            self.model_ = ARDL(
                endog=endog,
                lags=self.lags,
                exog=exog,
                trend=self.trend,
                causal=True  # Enforce causality (no look-ahead bias)
            )
            
            # Fit model
            self.fitted_model_ = self.model_.fit()
            self.is_fitted = True
            self.convergence_failed = False
            
            # Log model diagnostics
            logger.info(f"ARDL fitted successfully. AIC: {self.fitted_model_.aic:.2f}, BIC: {self.fitted_model_.bic:.2f}")
            logger.info(f"R²: {self.fitted_model_.rsquared:.4f}")
            
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
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Extract model diagnostics and fit statistics.
        
        Returns:
            Dictionary with:
            - aic: Akaike Information Criterion
            - bic: Bayesian Information Criterion
            - rsquared: R-squared (goodness of fit)
            - adj_rsquared: Adjusted R-squared
            - log_likelihood: Log-likelihood value
            - residuals: Model residuals
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")
        
        return {
            "aic": self.fitted_model_.aic,
            "bic": self.fitted_model_.bic,
            "rsquared": self.fitted_model_.rsquared,
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
            
            is_cointegrated = p_value < significance_level
            
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
        Pesaran et al. (2001). "Bounds testing approaches to the analysis
        of level relationships." Journal of Applied Econometrics.
    """
    
    @staticmethod
    def bounds_f_test(
        ardl_model: ARDLModel,
        alpha: float = 0.05
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Perform ARDL bounds F-test for cointegration.
        
        Args:
            ardl_model: Fitted ARDL model.
            alpha: Significance level.
        
        Returns:
            Tuple of (has_long_run_relationship, f_statistic, test_info).
        
        Note:
            This is a simplified implementation. Production systems should use
            critical values from Pesaran et al. (2001) Table CI.
        """
        if not ardl_model.is_fitted:
            raise ValueError("ARDL model must be fitted first.")
        
        try:
            # Get F-statistic from model
            # Note: statsmodels ARDL may not have direct bounds test
            # This is a placeholder for the concept
            
            logger.warning("Bounds test implementation requires custom critical values.")
            
            # Placeholder return
            return False, 0.0, {
                "message": "Bounds test requires manual interpretation with Pesaran tables",
                "reference": "Pesaran et al. (2001) Table CI"
            }
        
        except Exception as e:
            logger.error(f"Bounds test failed: {str(e)}")
            return False, 0.0, {}


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
