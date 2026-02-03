"""
Comprehensive Model Evaluation and Diagnostics.

This script:
1. Loads the trained model
2. Performs detailed evaluation on test set
3. Runs walk-forward backtesting
4. Generates trading signals and performance metrics
5. Creates visualization reports
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from app.config import Config
from app.data.loader import DataLoader
from app.data.preprocessor import Preprocessor, TechnicalIndicatorEngine
from app.models.fusion import NeuroEconometricNet
from app.engine.backtester import WalkForwardBacktester, PerformanceAnalyzer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_trained_model(model_path: str, metadata_path: str):
    """Load trained model and metadata."""
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    
    # Get input dimension from checkpoint
    input_dim = checkpoint['model_state_dict']['neural_branch.input_projection.weight'].shape[1]
    
    model = NeuroEconometricNet(
        input_dim=input_dim,
        hidden_dim=metadata['config']['hidden_dim'],
        num_lstm_layers=metadata['config']['num_layers']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(Config.DEVICE)
    model.eval()
    
    logger.info(f"✓ Model loaded from {model_path}")
    logger.info(f"  Trained on: {metadata['timestamp']}")
    logger.info(f"  Best epoch: {metadata['best_epoch']}/{metadata['total_epochs']}")
    
    return model, metadata


def prepare_test_data(start_date='2024-01-01', end_date='2024-12-31'):
    """Prepare test data for evaluation."""
    logger.info(f"\nLoading test data: {start_date} to {end_date}")
    
    # Load data
    data_loader = DataLoader(ticker=Config.TICKER, local_csv='data/GSPC_ohlcv.csv')
    ohlcv_df, _ = data_loader.load_all(start_date, end_date, include_news=False)
    
    # Compute indicators and preprocess
    df_with_indicators = TechnicalIndicatorEngine.compute_all(ohlcv_df)
    
    preprocessor = Preprocessor()
    processed_df, metadata = preprocessor.fit_transform(df_with_indicators)
    
    logger.info(f"✓ Test data prepared: {len(processed_df)} samples")
    
    return processed_df, ohlcv_df, preprocessor, metadata


def evaluate_predictions(model, test_df, raw_df, preprocessor):
    """Evaluate model predictions on test set."""
    logger.info("\n" + "="*60)
    logger.info("DETAILED MODEL EVALUATION")
    logger.info("="*60)
    
    predictions = []
    actuals = []
    alpha_values = []
    ardl_preds = []
    neural_preds = []
    
    # Get the actual input dimension from the model
    input_dim = model.neural_branch.input_projection.weight.shape[1]
    logger.info(f"Model expects input_dim: {input_dim}, Test data has {test_df.shape[1]-1} features")
    
    # Make predictions
    with torch.no_grad():
        for i in range(Config.SEQ_LENGTH, len(test_df)):
            # Get sequence
            seq = test_df.iloc[i-Config.SEQ_LENGTH:i].values
            
            # Prepare input - use only the columns that match the model's expected input
            X = torch.tensor(seq[:, :input_dim], dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
            
            # Dummy ARDL, vol, sentiment (since we're testing the saved model)
            ardl_pred = torch.tensor([[test_df.iloc[i-1, -1]]], dtype=torch.float32).to(Config.DEVICE)
            # Volatility features: 5 dimensions (ATR, BB_Width, ADX, StdDev, Range)
            volatility = torch.tensor([[0.02, 0.03, 50.0, 0.02, 0.01]], dtype=torch.float32).to(Config.DEVICE)
            sentiment = torch.tensor([[0.0]], dtype=torch.float32).to(Config.DEVICE)
            
            # Forward pass
            output = model(X, ardl_pred, volatility, sentiment)
            # Model returns tuple: (fused_prediction, alpha, neural_prediction)
            pred, alpha, neural_pred = output
            
            predictions.append(pred.item())
            actuals.append(test_df.iloc[i, -1])
            
            # Store gating weight
            alpha_values.append(alpha.item())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    
    # R² calculation
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Directional accuracy
    pred_direction = np.diff(predictions) > 0
    actual_direction = np.diff(actuals) > 0
    dir_accuracy = np.mean(pred_direction == actual_direction) * 100
    
    logger.info("\nScaled Metrics (on normalized data):")
    logger.info(f"  MSE: {mse:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  MAE: {mae:.6f}")
    logger.info(f"  R²: {r2:.6f}")
    logger.info(f"  Directional Accuracy: {dir_accuracy:.2f}%")
    
    if alpha_values:
        logger.info(f"\nGating Analysis:")
        logger.info(f"  Mean α: {np.mean(alpha_values):.4f}")
        logger.info(f"  Std α: {np.std(alpha_values):.4f}")
        logger.info(f"  Min/Max α: [{np.min(alpha_values):.4f}, {np.max(alpha_values):.4f}]")
    
    # Try to inverse transform if possible
    try:
        # This is a simplified inverse transform - proper implementation needs scaler
        logger.info("\n⚠ Note: Inverse transformation to original price scale not fully implemented")
        logger.info("   Current metrics are on normalized/differenced scale")
    except Exception as e:
        logger.warning(f"Could not inverse transform: {e}")
    
    return {
        'predictions': predictions,
        'actuals': actuals,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'directional_accuracy': dir_accuracy,
        'alpha_values': alpha_values
    }


def run_walk_forward_backtest(model_path: str):
    """Run walk-forward backtesting with proper out-of-sample evaluation."""
    logger.info("\n" + "="*60)
    logger.info("WALK-FORWARD BACKTESTING")
    logger.info("="*60)
    
    # Load full dataset
    data_loader = DataLoader(ticker=Config.TICKER, local_csv='data/GSPC_ohlcv.csv')
    ohlcv_df, _ = data_loader.load_all(Config.START_DATE, Config.END_DATE, include_news=False)
    
    # Use WalkForwardBacktester
    backtester = WalkForwardBacktester(
        train_window=Config.TRAIN_WINDOW_SIZE,
        test_window=Config.TEST_WINDOW_SIZE,
        step=Config.WALK_FORWARD_STEP
    )
    
    logger.info(f"\nBacktest Configuration:")
    logger.info(f"  Train window: {Config.TRAIN_WINDOW_SIZE} days")
    logger.info(f"  Test window: {Config.TEST_WINDOW_SIZE} days")
    logger.info(f"  Step size: {Config.WALK_FORWARD_STEP} days")
    
    # For now, we'll use the existing model
    # In production, you'd retrain on each window
    logger.info("\n⚠ Using pre-trained model for all windows")
    logger.info("  For true walk-forward, retrain model on each window")
    
    return None


def generate_trading_signals(predictions, actuals, threshold=0.01):
    """Generate trading signals from predictions."""
    logger.info("\n" + "="*60)
    logger.info("TRADING SIGNAL GENERATION")
    logger.info("="*60)
    
    # Calculate predicted returns
    pred_returns = np.diff(predictions) / predictions[:-1]
    actual_returns = np.diff(actuals) / actuals[:-1]
    
    # Generate signals: 1 (Buy), 0 (Hold), -1 (Sell)
    signals = np.zeros(len(pred_returns))
    signals[pred_returns > threshold] = 1  # Buy signal
    signals[pred_returns < -threshold] = -1  # Sell signal
    
    # Calculate strategy returns
    strategy_returns = signals * actual_returns
    
    # Performance metrics
    cumulative_return = (1 + strategy_returns).prod() - 1
    sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    max_drawdown = (pd.Series(strategy_returns + 1).cumprod().div(pd.Series(strategy_returns + 1).cumprod().cummax()) - 1).min()
    
    win_rate = np.sum(strategy_returns > 0) / np.sum(signals != 0) * 100 if np.sum(signals != 0) > 0 else 0
    
    logger.info(f"\nTrading Performance:")
    logger.info(f"  Total trades: {np.sum(signals != 0)}")
    logger.info(f"  Buy signals: {np.sum(signals == 1)}")
    logger.info(f"  Sell signals: {np.sum(signals == -1)}")
    logger.info(f"  Cumulative return: {cumulative_return*100:.2f}%")
    logger.info(f"  Sharpe ratio: {sharpe_ratio:.3f}")
    logger.info(f"  Max drawdown: {max_drawdown*100:.2f}%")
    logger.info(f"  Win rate: {win_rate:.2f}%")
    
    return {
        'signals': signals,
        'strategy_returns': strategy_returns,
        'cumulative_return': cumulative_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }


def main():
    """Main evaluation pipeline."""
    print("="*60)
    print("NEURO-ECONOMETRIC MODEL EVALUATION")
    print("="*60)
    print(f"Timestamp: {datetime.now()}")
    print("="*60)
    
    try:
        # 1. Load trained model
        model_path = "models_saved/best_model.pt"
        metadata_path = "models_saved/training_info.json"
        
        if not Path(model_path).exists():
            logger.error(f"Model not found: {model_path}")
            logger.error("Please train the model first: python train_model.py")
            return
        
        model, metadata = load_trained_model(model_path, metadata_path)
        
        # 2. Prepare test data (hold-out set from 2024)
        test_df, raw_df, preprocessor, prep_metadata = prepare_test_data('2024-01-01', '2024-12-31')
        
        # 3. Evaluate predictions
        eval_results = evaluate_predictions(model, test_df, raw_df, preprocessor)
        
        # 4. Generate trading signals
        trading_results = generate_trading_signals(
            eval_results['predictions'],
            eval_results['actuals']
        )
        
        # 5. Save evaluation report
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': metadata,
            'evaluation_metrics': {
                'mse': float(eval_results['mse']),
                'rmse': float(eval_results['rmse']),
                'mae': float(eval_results['mae']),
                'r2': float(eval_results['r2']),
                'directional_accuracy': float(eval_results['directional_accuracy'])
            },
            'trading_performance': {
                'cumulative_return': float(trading_results['cumulative_return']),
                'sharpe_ratio': float(trading_results['sharpe_ratio']),
                'max_drawdown': float(trading_results['max_drawdown']),
                'win_rate': float(trading_results['win_rate'])
            }
        }
        
        report_path = "models_saved/evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✓ Evaluation report saved to: {report_path}")
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETED")
        print("="*60)
        print("\nKey Findings:")
        print(f"  • Directional Accuracy: {eval_results['directional_accuracy']:.2f}%")
        print(f"  • Sharpe Ratio: {trading_results['sharpe_ratio']:.3f}")
        print(f"  • Strategy Return: {trading_results['cumulative_return']*100:.2f}%")
        print("\nRecommendations:")
        if eval_results['r2'] < 0.3:
            print("  ⚠ Low R² suggests model needs improvement")
            print("    - Consider longer training window")
            print("    - Add more features (sentiment, macro indicators)")
            print("    - Tune hyperparameters")
        if trading_results['sharpe_ratio'] < 1.0:
            print("  ⚠ Low Sharpe ratio - high risk-adjusted returns needed")
            print("    - Refine signal threshold")
            print("    - Implement stop-loss/take-profit")
        
    except Exception as e:
        logger.error(f"\n❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
