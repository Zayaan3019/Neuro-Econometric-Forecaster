"""
Walk-Forward Backtesting Engine.

This module implements rigorous walk-forward validation for the Neuro-Econometric
Engine, ensuring zero look-ahead bias and realistic performance estimation.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from tqdm import tqdm

from app.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """Container for backtest performance metrics."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    directional_accuracy: float
    num_trades: int
    avg_trade_return: float
    trades_per_year: float


class PerformanceAnalyzer:
    """
    Compute comprehensive trading performance metrics.
    
    Implements industry-standard metrics for quantitative strategy evaluation:
    - Sharpe Ratio: Risk-adjusted return
    - Sortino Ratio: Downside risk-adjusted return
    - Maximum Drawdown: Largest peak-to-trough decline
    - Win Rate: Percentage of profitable trades
    - Profit Factor: Gross profits / Gross losses
    """
    
    @staticmethod
    def compute_returns(
        prices: pd.Series,
        signals: pd.Series
    ) -> pd.Series:
        """
        Compute strategy returns from prices and signals.
        
        Args:
            prices: Series of asset prices.
            signals: Series of trading signals (1=Long, -1=Short, 0=Flat).
        
        Returns:
            Series of strategy returns.
        
        Mathematical Formulation:
            r_t = signal_{t-1} · (price_t - price_{t-1}) / price_{t-1}
            
            Note: We use lagged signals to avoid look-ahead bias.
        """
        # Compute price returns
        price_returns = prices.pct_change()
        
        # Lag signals by 1 period (can only trade on signal from previous period)
        lagged_signals = signals.shift(1)
        
        # Strategy returns
        strategy_returns = lagged_signals * price_returns
        
        return strategy_returns.fillna(0)
    
    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Compute annualized Sharpe ratio.
        
        Args:
            returns: Series of strategy returns.
            risk_free_rate: Annual risk-free rate (default 2%).
            periods_per_year: Number of trading periods per year.
        
        Returns:
            Annualized Sharpe ratio.
        
        Mathematical Foundation:
            Sharpe = (E[R_strategy] - R_f) / σ(R_strategy) * √(periods_per_year)
            
            Where:
            - E[R_strategy]: Expected strategy return
            - R_f: Risk-free rate
            - σ(R_strategy): Standard deviation of strategy returns
        
        Interpretation:
            - Sharpe > 1: Good
            - Sharpe > 2: Very good
            - Sharpe > 3: Excellent
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
        
        return float(sharpe)
    
    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Compute annualized Sortino ratio.
        
        Args:
            returns: Series of strategy returns.
            risk_free_rate: Annual risk-free rate.
            periods_per_year: Number of trading periods per year.
        
        Returns:
            Annualized Sortino ratio.
        
        Mathematical Foundation:
            Sortino = (E[R_strategy] - R_f) / σ_downside * √(periods_per_year)
            
            Where σ_downside = √(E[min(R - R_f, 0)²])
        
        Advantage over Sharpe:
            Only penalizes downside volatility, not upside volatility.
            More appropriate for asymmetric return distributions.
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()
        
        return float(sortino)
    
    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> Tuple[float, int]:
        """
        Compute maximum drawdown and its duration.
        
        Args:
            equity_curve: Series of cumulative portfolio values.
        
        Returns:
            Tuple of (max_drawdown_pct, duration_in_days).
        
        Mathematical Foundation:
            Drawdown_t = (Peak_t - Value_t) / Peak_t
            MaxDrawdown = max_t(Drawdown_t)
            
            Where Peak_t = max(Value_s for s ≤ t)
        
        Application:
            Critical risk metric. Represents largest capital loss from peak.
            Institutional investors often have max drawdown limits (e.g., 20%).
        """
        if len(equity_curve) == 0:
            return 0.0, 0
        
        # Compute running maximum
        running_max = equity_curve.expanding().max()
        
        # Compute drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Maximum drawdown
        max_dd = drawdown.min()
        
        # Drawdown duration
        # Find the longest period below previous peak
        in_drawdown = drawdown < 0
        drawdown_periods = in_drawdown.astype(int).groupby(
            (in_drawdown != in_drawdown.shift()).cumsum()
        ).sum()
        max_duration = int(drawdown_periods.max()) if len(drawdown_periods) > 0 else 0
        
        return float(max_dd), max_duration
    
    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        """
        Compute win rate (percentage of profitable periods).
        
        Args:
            returns: Series of strategy returns.
        
        Returns:
            Win rate as percentage (0-100).
        """
        if len(returns) == 0:
            return 0.0
        
        winning_periods = (returns > 0).sum()
        total_periods = len(returns[returns != 0])
        
        if total_periods == 0:
            return 0.0
        
        return float(100 * winning_periods / total_periods)
    
    @staticmethod
    def profit_factor(returns: pd.Series) -> float:
        """
        Compute profit factor.
        
        Args:
            returns: Series of strategy returns.
        
        Returns:
            Profit factor (gross profits / gross losses).
        
        Mathematical Foundation:
            PF = Σ(returns where returns > 0) / |Σ(returns where returns < 0)|
        
        Interpretation:
            - PF < 1: Losing strategy
            - PF = 1: Break-even
            - PF > 1: Profitable strategy
            - PF > 2: Strong strategy
        """
        gross_profits = returns[returns > 0].sum()
        gross_losses = abs(returns[returns < 0].sum())
        
        if gross_losses == 0:
            return np.inf if gross_profits > 0 else 1.0
        
        return float(gross_profits / gross_losses)
    
    @staticmethod
    def directional_accuracy(
        predictions: pd.Series,
        actuals: pd.Series
    ) -> float:
        """
        Compute directional accuracy (correct trend prediction).
        
        Args:
            predictions: Predicted price changes.
            actuals: Actual price changes.
        
        Returns:
            Directional accuracy as percentage (0-100).
        """
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        
        correct = (pred_direction == actual_direction).sum()
        total = len(predictions)
        
        if total == 0:
            return 0.0
        
        return float(100 * correct / total)


class WalkForwardBacktester:
    """
    Walk-Forward backtesting engine with strict temporal validation.
    
    Algorithm:
        For each time window:
            1. Train on window [t - train_size : t]
            2. Test on window [t : t + test_size]
            3. Record predictions and performance
            4. Slide window forward by step_size
            5. Repeat until end of data
    
    Key Properties:
        - Zero look-ahead bias
        - Realistic out-of-sample performance estimation
        - Accounts for regime changes over time
        - Mimics real-world deployment
    """
    
    def __init__(
        self,
        train_window_size: int = Config.TRAIN_WINDOW_SIZE,
        test_window_size: int = Config.TEST_WINDOW_SIZE,
        step_size: int = Config.WALK_FORWARD_STEP
    ):
        """
        Initialize walk-forward backtester.
        
        Args:
            train_window_size: Number of periods for training.
            test_window_size: Number of periods for testing.
            step_size: Number of periods to slide forward.
        """
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        self.step_size = step_size
        
        self.performance_analyzer = PerformanceAnalyzer()
    
    def generate_splits(
        self,
        data_length: int
    ) -> List[Tuple[range, range]]:
        """
        Generate train/test split indices for walk-forward validation.
        
        Args:
            data_length: Total length of dataset.
        
        Returns:
            List of (train_indices, test_indices) tuples.
        """
        splits = []
        
        start = 0
        while start + self.train_window_size + self.test_window_size <= data_length:
            train_end = start + self.train_window_size
            test_end = min(train_end + self.test_window_size, data_length)
            
            train_indices = range(start, train_end)
            test_indices = range(train_end, test_end)
            
            splits.append((train_indices, test_indices))
            
            start += self.step_size
        
        logger.info(f"Generated {len(splits)} walk-forward splits")
        return splits
    
    def backtest(
        self,
        data: pd.DataFrame,
        model: Any,
        target_col: str = 'Close',
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, BacktestMetrics]:
        """
        Run walk-forward backtest.
        
        Args:
            data: DataFrame with prices and features.
            model: Trained model with fit() and predict() methods.
            target_col: Name of target column.
            feature_cols: List of feature column names.
        
        Returns:
            Tuple of (results_df, metrics).
        """
        splits = self.generate_splits(len(data))
        
        all_predictions = []
        all_actuals = []
        all_dates = []
        all_signals = []
        
        logger.info("Starting walk-forward backtest")
        
        for train_idx, test_idx in tqdm(splits, desc="Backtesting"):
            # Extract train/test data
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Train model
            try:
                if feature_cols:
                    model.fit(train_data[target_col], train_data[feature_cols])
                else:
                    model.fit(train_data[target_col])
                
                # Predict on test set
                if feature_cols:
                    predictions = model.predict(test_data[feature_cols])
                else:
                    predictions = model.predict(len(test_data))
                
                # Store results
                actuals = test_data[target_col].values
                dates = test_data.index
                
                all_predictions.extend(predictions)
                all_actuals.extend(actuals)
                all_dates.extend(dates)
                
                # Generate trading signals
                # Simple strategy: Buy if predicted increase, Sell if predicted decrease
                signals = np.sign(np.diff(predictions, prepend=predictions[0]))
                all_signals.extend(signals)
            
            except Exception as e:
                logger.warning(f"Backtest iteration failed: {str(e)}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'date': all_dates,
            'actual': all_actuals,
            'predicted': all_predictions,
            'signal': all_signals
        })
        results_df.set_index('date', inplace=True)
        
        # Compute performance metrics
        metrics = self._compute_metrics(results_df, data[target_col])
        
        return results_df, metrics
    
    def _compute_metrics(
        self,
        results_df: pd.DataFrame,
        price_series: pd.Series
    ) -> BacktestMetrics:
        """
        Compute comprehensive backtest metrics.
        
        Args:
            results_df: DataFrame with predictions and signals.
            price_series: Original price series.
        
        Returns:
            BacktestMetrics object.
        """
        # Align price series with results
        prices = price_series.loc[results_df.index]
        signals = results_df['signal']
        
        # Compute strategy returns
        strategy_returns = self.performance_analyzer.compute_returns(prices, signals)
        
        # Equity curve
        equity_curve = (1 + strategy_returns).cumprod()
        
        # Compute metrics
        total_return = float(equity_curve.iloc[-1] - 1) if len(equity_curve) > 0 else 0.0
        
        # Annualized return
        trading_days = len(strategy_returns)
        years = trading_days / 252
        annualized_return = float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0
        
        # Risk metrics
        sharpe = self.performance_analyzer.sharpe_ratio(strategy_returns)
        sortino = self.performance_analyzer.sortino_ratio(strategy_returns)
        max_dd, dd_duration = self.performance_analyzer.max_drawdown(equity_curve)
        
        # Trade metrics
        win_rate = self.performance_analyzer.win_rate(strategy_returns)
        profit_factor = self.performance_analyzer.profit_factor(strategy_returns)
        
        # Directional accuracy
        directional_acc = self.performance_analyzer.directional_accuracy(
            results_df['predicted'].diff(),
            results_df['actual'].diff()
        )
        
        # Trade statistics
        num_trades = int((signals.diff() != 0).sum())
        avg_trade_return = float(strategy_returns[strategy_returns != 0].mean()) if num_trades > 0 else 0.0
        trades_per_year = num_trades / years if years > 0 else 0.0
        
        metrics = BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_duration=dd_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            directional_accuracy=directional_acc,
            num_trades=num_trades,
            avg_trade_return=avg_trade_return,
            trades_per_year=trades_per_year
        )
        
        return metrics
    
    def print_metrics(self, metrics: BacktestMetrics) -> None:
        """
        Print formatted backtest metrics.
        
        Args:
            metrics: BacktestMetrics object.
        """
        print("\n" + "="*60)
        print("BACKTEST PERFORMANCE METRICS")
        print("="*60)
        print(f"Total Return:              {metrics.total_return*100:>10.2f}%")
        print(f"Annualized Return:         {metrics.annualized_return*100:>10.2f}%")
        print(f"Sharpe Ratio:              {metrics.sharpe_ratio:>10.2f}")
        print(f"Sortino Ratio:             {metrics.sortino_ratio:>10.2f}")
        print(f"Max Drawdown:              {metrics.max_drawdown*100:>10.2f}%")
        print(f"Max Drawdown Duration:     {metrics.max_drawdown_duration:>10} days")
        print(f"Win Rate:                  {metrics.win_rate:>10.2f}%")
        print(f"Profit Factor:             {metrics.profit_factor:>10.2f}")
        print(f"Directional Accuracy:      {metrics.directional_accuracy:>10.2f}%")
        print(f"Number of Trades:          {metrics.num_trades:>10}")
        print(f"Avg Trade Return:          {metrics.avg_trade_return*100:>10.4f}%")
        print(f"Trades Per Year:           {metrics.trades_per_year:>10.1f}")
        print("="*60 + "\n")
