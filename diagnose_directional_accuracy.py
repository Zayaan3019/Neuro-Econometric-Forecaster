"""
Quick Diagnostic Script to Investigate Directional Accuracy Issue.

This script helps identify the root cause of poor directional accuracy.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import json

from app.config import Config
from app.data.loader import DataLoader
from app.data.preprocessor import Preprocessor, TechnicalIndicatorEngine
from app.models.fusion import NeuroEconometricNet

print("="*60)
print("DIRECTIONAL ACCURACY DIAGNOSTIC")
print("="*60)

# Load model
with open("models_saved/training_info.json", 'r') as f:
    metadata = json.load(f)

checkpoint = torch.load("models_saved/best_model.pt", map_location=Config.DEVICE, weights_only=False)
input_dim = checkpoint['model_state_dict']['neural_branch.input_projection.weight'].shape[1]

model = NeuroEconometricNet(
    input_dim=input_dim,
    hidden_dim=metadata['config']['hidden_dim'],
    num_lstm_layers=metadata['config']['num_layers']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(Config.DEVICE)
model.eval()

print("✓ Model loaded")

# Load test data
data_loader = DataLoader(ticker=Config.TICKER, local_csv='data/GSPC_ohlcv.csv')
ohlcv_df, _ = data_loader.load_all('2024-01-01', '2024-12-31', include_news=False)

df_with_indicators = TechnicalIndicatorEngine.compute_all(ohlcv_df)
preprocessor = Preprocessor()
test_df, _ = preprocessor.fit_transform(df_with_indicators)

print(f"✓ Test data loaded: {len(test_df)} samples")
print(f"✓ Test data shape: {test_df.shape}")
print(f"\nTest data columns: {list(test_df.columns)}")

# Make predictions
predictions = []
actuals = []

with torch.no_grad():
    for i in range(Config.SEQ_LENGTH, min(Config.SEQ_LENGTH + 20, len(test_df))):  # First 20 predictions
        seq = test_df.iloc[i-Config.SEQ_LENGTH:i].values
        
        X = torch.tensor(seq[:, :17], dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
        ardl_pred = torch.tensor([[test_df.iloc[i-1, -1]]], dtype=torch.float32).to(Config.DEVICE)
        volatility = torch.tensor([[0.02, 0.03, 50.0, 0.02, 0.01]], dtype=torch.float32).to(Config.DEVICE)
        sentiment = torch.tensor([[0.0]], dtype=torch.float32).to(Config.DEVICE)
        
        output = model(X, ardl_pred, volatility, sentiment)
        pred, alpha, neural_pred = output
        
        predictions.append(pred.item())
        actuals.append(test_df.iloc[i, -1])

predictions = np.array(predictions)
actuals = np.array(actuals)

# Analysis
print("\n" + "="*60)
print("SAMPLE PREDICTIONS vs ACTUALS")
print("="*60)
print(f"{'Index':<8} {'Prediction':<15} {'Actual':<15} {'Direction':<10}")
print("-"*60)
for i in range(len(predictions)):
    pred_dir = "UP" if i > 0 and predictions[i] > predictions[i-1] else "DOWN"
    actual_dir = "UP" if i > 0 and actuals[i] > actuals[i-1] else "DOWN"
    match = "✓" if pred_dir == actual_dir else "✗"
    print(f"{i:<8} {predictions[i]:<15.6f} {actuals[i]:<15.6f} {match} {pred_dir}/{actual_dir}")

# Calculate directional accuracy
pred_direction = np.diff(predictions) > 0
actual_direction = np.diff(actuals) > 0
dir_accuracy = np.mean(pred_direction == actual_direction) * 100

print("\n" + "="*60)
print("DIRECTIONAL ANALYSIS")
print("="*60)
print(f"Directional Accuracy: {dir_accuracy:.2f}%")
print(f"Correct: {np.sum(pred_direction == actual_direction)}/{len(pred_direction)}")

# Test if inverting helps
inverted_accuracy = np.mean(pred_direction != actual_direction) * 100
print(f"\nInverted Accuracy (flipping all predictions): {inverted_accuracy:.2f}%")

if inverted_accuracy > 60:
    print("\n⚠️ WARNING: Inverting predictions would give >60% accuracy!")
    print("   This suggests a SIGN ERROR in the model or preprocessing.")
    print("   Consider:")
    print("   1. Check if target variable was inverted during preprocessing")
    print("   2. Verify loss function signs")
    print("   3. Check if differencing direction is correct")

# Statistical checks
print("\n" + "="*60)
print("STATISTICAL CHECKS")
print("="*60)
print(f"Predictions Mean: {np.mean(predictions):.6f}")
print(f"Predictions Std: {np.std(predictions):.6f}")
print(f"Predictions Range: [{np.min(predictions):.6f}, {np.max(predictions):.6f}]")
print()
print(f"Actuals Mean: {np.mean(actuals):.6f}")
print(f"Actuals Std: {np.std(actuals):.6f}")
print(f"Actuals Range: [{np.min(actuals):.6f}, {np.max(actuals):.6f}]")
print()
print(f"Correlation: {np.corrcoef(predictions, actuals)[0,1]:.4f}")

# Correlation analysis
if len(predictions) > 2:
    # Correlation between actual and predicted changes
    pred_changes = np.diff(predictions)
    actual_changes = np.diff(actuals)
    change_corr = np.corrcoef(pred_changes, actual_changes)[0,1]
    print(f"Change Correlation: {change_corr:.4f}")
    
    if change_corr < -0.3:
        print("\n⚠️ NEGATIVE CORRELATION detected!")
        print("   Model is predicting opposite direction systematically.")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
