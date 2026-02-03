"""
Data Preprocessing Module for Neuro-Econometric Engine.

This module implements critical stationarity checks (Augmented Dickey-Fuller Test),
differencing transformations, and normalization required for ARDL modeling and
neural network training.
"""

from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

# Try to import TA-Lib, fall back to pure Python implementations if not available
try:
    import talib as ta
    TALIB_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("TA-Lib loaded successfully (using optimized C library)")
except ImportError:
    TALIB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TA-Lib not available. Using pure Python implementations (slower but functional).")

from app.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StationarityTester:
    """
    Statistical tests for time-series stationarity.
    
    Implements Augmented Dickey-Fuller (ADF) and KPSS tests to determine if
    a time series requires differencing for ARDL modeling.
    
    Mathematical Foundation:
        ADF Test: H0 = Unit root exists (non-stationary)
        KPSS Test: H0 = Series is stationary
        
        A series is considered stationary if:
        - ADF p-value < 0.05 (reject H0, no unit root)
        - KPSS p-value > 0.05 (fail to reject H0, is stationary)
    """
    
    def __init__(self, significance_level: float = Config.ADF_PVALUE_THRESHOLD):
        """
        Initialize stationarity tester.
        
        Args:
            significance_level: Threshold for p-value interpretation (default 0.05).
        """
        self.significance_level = significance_level
    
    def adf_test(self, series: pd.Series) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Perform Augmented Dickey-Fuller test for unit root.
        
        Args:
            series: Time-series data to test (pandas Series).
        
        Returns:
            Tuple of (is_stationary, p_value, test_statistics).
        
        Mathematical Formulation:
            ΔY_t = α + βt + γY_{t-1} + Σ(δ_i ΔY_{t-i}) + ε_t
            
            Where:
            - γ: Coefficient to test (if γ = 0, unit root exists)
            - Test statistic compared against critical values
        """
        # Remove NaN values
        series_clean = series.dropna()
        
        if len(series_clean) < 20:
            logger.warning("Series too short for reliable ADF test (<20 observations)")
            return False, 1.0, {}
        
        try:
            result = adfuller(series_clean, autolag='AIC')
            
            test_stats = {
                "adf_statistic": result[0],
                "p_value": result[1],
                "usedlag": result[2],
                "nobs": result[3],
                "critical_values": result[4],
                "icbest": result[5]
            }
            
            is_stationary = result[1] < self.significance_level
            
            logger.info(f"ADF Test - Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}, Stationary: {is_stationary}")
            
            return is_stationary, result[1], test_stats
        
        except Exception as e:
            logger.error(f"ADF test failed: {str(e)}")
            return False, 1.0, {}
    
    def kpss_test(self, series: pd.Series, regression: str = 'c') -> Tuple[bool, float, Dict[str, Any]]:
        """
        Perform KPSS test for stationarity.
        
        Args:
            series: Time-series data to test.
            regression: Type of regression ('c' for constant, 'ct' for constant+trend).
        
        Returns:
            Tuple of (is_stationary, p_value, test_statistics).
        
        Note:
            KPSS has opposite null hypothesis to ADF:
            - ADF H0: Non-stationary
            - KPSS H0: Stationary
        """
        series_clean = series.dropna()
        
        if len(series_clean) < 20:
            logger.warning("Series too short for reliable KPSS test")
            return False, 0.0, {}
        
        try:
            result = kpss(series_clean, regression=regression, nlags='auto')
            
            test_stats = {
                "kpss_statistic": result[0],
                "p_value": result[1],
                "lags": result[2],
                "critical_values": result[3]
            }
            
            # For KPSS, p-value > 0.05 means stationary
            is_stationary = result[1] > self.significance_level
            
            logger.info(f"KPSS Test - Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}, Stationary: {is_stationary}")
            
            return is_stationary, result[1], test_stats
        
        except Exception as e:
            logger.error(f"KPSS test failed: {str(e)}")
            return False, 0.0, {}
    
    def comprehensive_test(self, series: pd.Series) -> Tuple[bool, Dict[str, Any]]:
        """
        Run both ADF and KPSS tests for robust stationarity assessment.
        
        Args:
            series: Time-series data to test.
        
        Returns:
            Tuple of (is_stationary, combined_statistics).
        
        Decision Logic:
            - Both tests agree (stationary): Return True
            - Both tests agree (non-stationary): Return False
            - Tests disagree: Return False (conservative approach)
        """
        adf_stationary, adf_pval, adf_stats = self.adf_test(series)
        kpss_stationary, kpss_pval, kpss_stats = self.kpss_test(series)
        
        # Conservative approach: Both tests must agree
        is_stationary = adf_stationary and kpss_stationary
        
        combined_stats = {
            "adf": adf_stats,
            "kpss": kpss_stats,
            "consensus_stationary": is_stationary
        }
        
        return is_stationary, combined_stats


class DifferencingTransformer:
    """
    Applies differencing transformations to achieve stationarity.
    
    Mathematical Foundation:
        First Difference: ΔY_t = Y_t - Y_{t-1}
        Second Difference: Δ²Y_t = ΔY_t - ΔY_{t-1}
        
        Most economic series become stationary after 1-2 differences.
    """
    
    def __init__(self, max_order: int = Config.ARDL_MAX_DIFF_ORDER):
        """
        Initialize differencing transformer.
        
        Args:
            max_order: Maximum differencing order (typically 1 or 2).
        """
        self.max_order = max_order
        self.tester = StationarityTester()
        self.diff_order: int = 0
        self.original_values: Optional[np.ndarray] = None
    
    def fit_transform(self, series: pd.Series) -> Tuple[pd.Series, int]:
        """
        Automatically difference series until stationary.
        
        Args:
            series: Original time-series.
        
        Returns:
            Tuple of (differenced_series, order_of_differencing).
        
        Algorithm:
            1. Test stationarity of original series
            2. If non-stationary, apply first difference
            3. Re-test, if still non-stationary, apply second difference
            4. Stop at max_order or when stationary
        """
        current_series = series.copy()
        self.diff_order = 0
        
        # Store original values for potential inversion
        self.original_values = series.values
        
        # Test original series
        is_stationary, _ = self.tester.comprehensive_test(current_series)
        
        if is_stationary:
            logger.info("Series is already stationary. No differencing required.")
            return current_series, 0
        
        # Apply differencing iteratively
        for order in range(1, self.max_order + 1):
            logger.info(f"Applying differencing of order {order}")
            current_series = current_series.diff().dropna()
            
            is_stationary, _ = self.tester.comprehensive_test(current_series)
            
            if is_stationary:
                self.diff_order = order
                logger.info(f"Series is stationary after {order} difference(s)")
                return current_series, order
        
        # If still not stationary after max_order
        logger.warning(f"Series is not stationary after {self.max_order} differences. Proceeding anyway.")
        self.diff_order = self.max_order
        return current_series, self.max_order
    
    def inverse_transform(self, differenced_series: pd.Series, original_level: float) -> pd.Series:
        """
        Reconstruct original scale from differenced predictions.
        
        Args:
            differenced_series: Differenced predictions.
            original_level: Last known value of the original series.
        
        Returns:
            Series in original scale.
        
        Mathematical Formulation:
            If ΔY_t = Y_t - Y_{t-1}, then:
            Y_t = Y_{t-1} + ΔY_t
        """
        if self.diff_order == 0:
            return differenced_series
        
        reconstructed = [original_level]
        
        for diff_value in differenced_series:
            reconstructed.append(reconstructed[-1] + diff_value)
        
        return pd.Series(reconstructed[1:], index=differenced_series.index)


class TechnicalIndicatorEngine:
    """
    Computes technical indicators using TA-Lib or pure Python fallbacks.
    
    Provides feature engineering for neural network input, capturing
    momentum, volatility, and trend information.
    """
    
    @staticmethod
    def _compute_rsi_python(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Pure Python implementation of RSI.
        
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        """
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    @staticmethod
    def _compute_sma_python(prices: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average."""
        return pd.Series(prices).rolling(window=period, min_periods=1).mean().values
    
    @staticmethod
    def _compute_ema_python(prices: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average."""
        return pd.Series(prices).ewm(span=period, adjust=False).mean().values
    
    @staticmethod
    def _compute_macd_python(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MACD implementation.
        
        MACD = EMA(fast) - EMA(slow)
        Signal = EMA(MACD, signal)
        Histogram = MACD - Signal
        """
        ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean()
        ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return macd.values, signal_line.values, histogram.values
    
    @staticmethod
    def _compute_bollinger_bands_python(prices: np.ndarray, period: int = 20, std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bollinger Bands implementation.
        
        Middle = SMA(period)
        Upper = Middle + (std * StdDev)
        Lower = Middle - (std * StdDev)
        """
        series = pd.Series(prices)
        middle = series.rolling(window=period, min_periods=1).mean()
        std_dev = series.rolling(window=period, min_periods=1).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper.values, middle.values, lower.values
    
    @staticmethod
    def _compute_atr_python(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Average True Range implementation.
        
        TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        ATR = EMA(TR, period)
        """
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        
        return atr
    
    @staticmethod
    def _compute_adx_python(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Average Directional Index (simplified implementation).
        
        Measures trend strength (not direction).
        """
        # Simplified ADX calculation
        # In production, full +DI, -DI, and DX calculation would be needed
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Pad to original length
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smooth with EMA
        plus_di = pd.Series(plus_dm).ewm(span=period, adjust=False).mean()
        minus_di = pd.Series(minus_dm).ewm(span=period, adjust=False).mean()
        
        # Calculate DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        
        # ADX is smoothed DX
        adx = dx.ewm(span=period, adjust=False).mean().values
        
        return adx
    
    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators specified in Config.
        
        Args:
            df: OHLCV DataFrame with columns [Open, High, Low, Close, Volume].
        
        Returns:
            DataFrame with original data + technical indicators.
        
        Indicators Computed:
            - RSI: Relative Strength Index (momentum oscillator)
            - MACD: Moving Average Convergence Divergence (trend)
            - Bollinger Bands: Volatility bands
            - ATR: Average True Range (volatility)
            - ADX: Average Directional Index (trend strength)
        """
        df = df.copy()
        
        # Extract price arrays
        close = df['Close'].values.astype(float)
        high = df['High'].values.astype(float)
        low = df['Low'].values.astype(float)
        volume = df['Volume'].values.astype(float)
        
        try:
            if TALIB_AVAILABLE:
                # Use TA-Lib (faster, more accurate)
                # RSI - Relative Strength Index
                rsi_period = Config.TECHNICAL_INDICATORS['RSI']['period']
                df['RSI'] = ta.RSI(close, timeperiod=rsi_period)
                
                # MACD - Moving Average Convergence Divergence
                macd_config = Config.TECHNICAL_INDICATORS['MACD']
                macd, macd_signal, macd_hist = ta.MACD(
                    close,
                    fastperiod=macd_config['fast'],
                    slowperiod=macd_config['slow'],
                    signalperiod=macd_config['signal']
                )
                df['MACD'] = macd
                df['MACD_Signal'] = macd_signal
                df['MACD_Hist'] = macd_hist
                
                # Bollinger Bands
                bb_config = Config.TECHNICAL_INDICATORS['BBANDS']
                upper, middle, lower = ta.BBANDS(
                    close,
                    timeperiod=bb_config['period'],
                    nbdevup=bb_config['std'],
                    nbdevdn=bb_config['std']
                )
                df['BB_Upper'] = upper
                df['BB_Middle'] = middle
                df['BB_Lower'] = lower
                df['BB_Width'] = (upper - lower) / (middle + 1e-10)  # Normalized width
                
                # ATR - Average True Range (Volatility)
                atr_period = Config.TECHNICAL_INDICATORS['ATR']['period']
                df['ATR'] = ta.ATR(high, low, close, timeperiod=atr_period)
                
                # ADX - Average Directional Index (Trend Strength)
                adx_period = Config.TECHNICAL_INDICATORS['ADX']['period']
                df['ADX'] = ta.ADX(high, low, close, timeperiod=adx_period)
                
                # Additional momentum indicators
                df['MOM'] = ta.MOM(close, timeperiod=10)  # Momentum
                df['ROC'] = ta.ROC(close, timeperiod=10)  # Rate of Change
                
                # Volume indicators
                df['OBV'] = ta.OBV(close, volume)  # On-Balance Volume
                
            else:
                # Use pure Python implementations
                logger.info("Computing technical indicators with pure Python...")
                
                # RSI
                rsi_period = Config.TECHNICAL_INDICATORS['RSI']['period']
                df['RSI'] = TechnicalIndicatorEngine._compute_rsi_python(close, rsi_period)
                
                # MACD
                macd_config = Config.TECHNICAL_INDICATORS['MACD']
                macd, macd_signal, macd_hist = TechnicalIndicatorEngine._compute_macd_python(
                    close, 
                    macd_config['fast'], 
                    macd_config['slow'], 
                    macd_config['signal']
                )
                df['MACD'] = macd
                df['MACD_Signal'] = macd_signal
                df['MACD_Hist'] = macd_hist
                
                # Bollinger Bands
                bb_config = Config.TECHNICAL_INDICATORS['BBANDS']
                upper, middle, lower = TechnicalIndicatorEngine._compute_bollinger_bands_python(
                    close, 
                    bb_config['period'], 
                    bb_config['std']
                )
                df['BB_Upper'] = upper
                df['BB_Middle'] = middle
                df['BB_Lower'] = lower
                df['BB_Width'] = (upper - lower) / (middle + 1e-10)
                
                # ATR
                atr_period = Config.TECHNICAL_INDICATORS['ATR']['period']
                df['ATR'] = TechnicalIndicatorEngine._compute_atr_python(high, low, close, atr_period)
                
                # ADX
                adx_period = Config.TECHNICAL_INDICATORS['ADX']['period']
                df['ADX'] = TechnicalIndicatorEngine._compute_adx_python(high, low, close, adx_period)
                
                # Momentum
                df['MOM'] = pd.Series(close).diff(10).values
                
                # Rate of Change
                df['ROC'] = pd.Series(close).pct_change(10).values * 100
                
                # On-Balance Volume
                obv = np.zeros(len(close))
                obv[0] = volume[0]
                for i in range(1, len(close)):
                    if close[i] > close[i-1]:
                        obv[i] = obv[i-1] + volume[i]
                    elif close[i] < close[i-1]:
                        obv[i] = obv[i-1] - volume[i]
                    else:
                        obv[i] = obv[i-1]
                df['OBV'] = obv
            
            logger.info(f"Computed {len([c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']])} technical indicators")
        
        except Exception as e:
            logger.error(f"Error computing technical indicators: {str(e)}")
            # Return DataFrame with NaN indicators rather than failing
            for col in ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Middle', 
                       'BB_Lower', 'BB_Width', 'ATR', 'ADX', 'MOM', 'ROC', 'OBV']:
                if col not in df.columns:
                    df[col] = np.nan
        
        return df


class Preprocessor:
    """
    Master preprocessing pipeline combining all transformations.
    
    Orchestrates:
    1. Technical indicator computation
    2. Stationarity checks and differencing
    3. Normalization/Standardization
    4. Missing value handling
    """
    
    def __init__(self, scaling_method: str = 'standard'):
        """
        Initialize preprocessor.
        
        Args:
            scaling_method: 'standard' (z-score) or 'minmax' (0-1 range).
        """
        self.scaling_method = scaling_method
        self.scaler: Optional[Any] = None
        self.diff_transformer = DifferencingTransformer()
        self.feature_names: Optional[list] = None
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = 'Close',
        compute_indicators: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Full preprocessing pipeline.
        
        Args:
            df: Raw OHLCV DataFrame.
            target_col: Column to test for stationarity (typically 'Close').
            compute_indicators: Whether to compute technical indicators.
        
        Returns:
            Tuple of (processed_df, metadata).
        
        Pipeline Steps:
            1. Compute technical indicators
            2. Check stationarity and difference if needed
            3. Handle missing values (forward-fill)
            4. Normalize features
        """
        df_processed = df.copy()
        metadata = {}
        
        # Step 1: Compute technical indicators
        if compute_indicators:
            df_processed = TechnicalIndicatorEngine.compute_all(df_processed)
        
        # Step 2: Stationarity check on target
        if target_col in df_processed.columns:
            target_series = df_processed[target_col]
            stationary_target, diff_order = self.diff_transformer.fit_transform(target_series)
            
            metadata['diff_order'] = diff_order
            metadata['original_target_last'] = target_series.iloc[-1]
            
            # Replace target with differenced version if needed
            if diff_order > 0:
                df_processed[f'{target_col}_diff'] = stationary_target
        
        # Step 3: Handle missing values
        df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
        
        # Step 4: Normalization
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_names = numeric_cols
        
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        df_processed[numeric_cols] = self.scaler.fit_transform(df_processed[numeric_cols])
        
        metadata['feature_names'] = self.feature_names
        metadata['scaler'] = self.scaler
        
        logger.info(f"Preprocessing complete. Shape: {df_processed.shape}")
        
        return df_processed, metadata
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted transformations to new data (for inference).
        
        Args:
            df: New OHLCV data.
        
        Returns:
            Transformed DataFrame.
        """
        if self.scaler is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform() first.")
        
        df_processed = TechnicalIndicatorEngine.compute_all(df)
        df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
        
        # Apply same normalization
        df_processed[self.feature_names] = self.scaler.transform(df_processed[self.feature_names])
        
        return df_processed
