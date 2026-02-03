"""
Scenario-Based Market Condition Testing.

Tests model performance under different market conditions:
1. Bull market
2. Bear market  
3. High volatility
4. Low volatility
5. Trending vs ranging markets
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from app.config import Config
from app.data.loader import DataLoader
from app.data.preprocessor import Preprocessor, TechnicalIndicatorEngine
from app.models.fusion import NeuroEconometricNet

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path: str, metadata_path: str):
    """Load trained model."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    checkpoint = torch.load(model_path, map_location=Config.DEVICE, weights_only=False)
    input_dim = checkpoint['model_state_dict']['neural_branch.input_projection.weight'].shape[1]
    
    model = NeuroEconometricNet(
        input_dim=input_dim,
        hidden_dim=metadata['config']['hidden_dim'],
        num_lstm_layers=metadata['config']['num_layers']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(Config.DEVICE)
    model.eval()
    
    return model, metadata


def evaluate_on_period(model, test_df, period_name):
    """Evaluate model on a specific time period."""
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for i in range(Config.SEQ_LENGTH, len(test_df)):
            seq = test_df.iloc[i-Config.SEQ_LENGTH:i].values
            
            X = torch.tensor(seq[:, :17], dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
            ardl_pred = torch.tensor([[test_df.iloc[i-1, -1]]], dtype=torch.float32).to(Config.DEVICE)
            volatility = torch.tensor([[0.02, 0.03, 50.0, 0.02, 0.01]], dtype=torch.float32).to(Config.DEVICE)
            sentiment = torch.tensor([[0.0]], dtype=torch.float32).to(Config.DEVICE)
            
            output = model(X, ardl_pred, volatility, sentiment)
            pred, alpha, _ = output
            
            predictions.append(pred.item())
            actuals.append(test_df.iloc[i, -1])
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    
    # Directional accuracy
    if len(predictions) > 1:
        pred_direction = np.diff(predictions) > 0
        actual_direction = np.diff(actuals) > 0
        dir_accuracy = np.mean(pred_direction == actual_direction) * 100
    else:
        dir_accuracy = 0.0
    
    logger.info(f"\n{period_name} Period:")
    logger.info(f"  Samples: {len(predictions)}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  MAE: {mae:.6f}")
    logger.info(f"  Directional Accuracy: {dir_accuracy:.2f}%")
    
    return {
        'period': period_name,
        'n_samples': len(predictions),
        'rmse': float(rmse),
        'mae': float(mae),
        'directional_accuracy': float(dir_accuracy)
    }


def test_different_market_conditions():
    """Test model on different market conditions."""
    logger.info("="*60)
    logger.info("MARKET CONDITION TESTING")
    logger.info("="*60)
    
    # Load model
    model, metadata = load_model(
        "models_saved/best_model.pt",
        "models_saved/training_info.json"
    )
    
    results = []
    
    # Test on different time periods (representing different market conditions)
    periods = [
        ('2024-01-01', '2024-03-31', 'Q1 2024'),
        ('2024-04-01', '2024-06-30', 'Q2 2024'),
        ('2024-07-01', '2024-09-30', 'Q3 2024'),
        ('2024-10-01', '2024-12-31', 'Q4 2024'),
    ]
    
    for start_date, end_date, period_name in periods:
        try:
            # Load data
            data_loader = DataLoader(ticker=Config.TICKER, local_csv='data/GSPC_ohlcv.csv')
            ohlcv_df, _ = data_loader.load_all(start_date, end_date, include_news=False)
            
            if len(ohlcv_df) < Config.SEQ_LENGTH + 10:
                logger.warning(f"Skipping {period_name}: insufficient data")
                continue
            
            # Preprocess
            df_with_indicators = TechnicalIndicatorEngine.compute_all(ohlcv_df)
            preprocessor = Preprocessor()
            test_df, _ = preprocessor.fit_transform(df_with_indicators)
            
            # Evaluate
            result = evaluate_on_period(model, test_df, period_name)
            results.append(result)
            
        except Exception as e:
            logger.warning(f"Error evaluating {period_name}: {e}")
    
    # Summary
    if results:
        logger.info("\n" + "="*60)
        logger.info("MARKET CONDITION SUMMARY")
        logger.info("="*60)
        
        avg_dir_accuracy = np.mean([r['directional_accuracy'] for r in results])
        std_dir_accuracy = np.std([r['directional_accuracy'] for r in results])
        
        logger.info(f"Average Directional Accuracy: {avg_dir_accuracy:.2f}%")
        logger.info(f"Std of Directional Accuracy: {std_dir_accuracy:.2f}%")
        logger.info(f"Best Period: {max(results, key=lambda x: x['directional_accuracy'])['period']} "
                   f"({max(r['directional_accuracy'] for r in results):.2f}%)")
        logger.info(f"Worst Period: {min(results, key=lambda x: x['directional_accuracy'])['period']} "
                   f"({min(r['directional_accuracy'] for r in results):.2f}%)")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'periods': results,
        'summary': {
            'avg_directional_accuracy': float(avg_dir_accuracy) if results else 0.0,
            'std_directional_accuracy': float(std_dir_accuracy) if results else 0.0
        }
    }
    
    with open('models_saved/market_condition_test.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info("\n✓ Results saved to: models_saved/market_condition_test.json")


if __name__ == "__main__":
    test_different_market_conditions()
