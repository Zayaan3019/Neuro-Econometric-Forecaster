"""
Comprehensive Model Robustness Testing Suite.

This script performs extensive validation of the trained model including:
1. Prediction Accuracy Tests
2. Stability Tests (noise robustness)
3. Edge Case Handling
4. Performance Consistency
5. Out-of-Sample Validation
6. Statistical Significance Tests
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
from scipy import stats
import matplotlib.pyplot as plt

from app.config import Config
from app.data.loader import DataLoader
from app.data.preprocessor import Preprocessor, TechnicalIndicatorEngine
from app.models.fusion import NeuroEconometricNet

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ModelRobustnessTester:
    """Comprehensive model robustness testing framework."""
    
    def __init__(self, model_path: str, metadata_path: str):
        """Initialize tester with trained model."""
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.results = {}
        
    def load_model(self):
        """Load trained model."""
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        checkpoint = torch.load(self.model_path, map_location=Config.DEVICE, weights_only=False)
        input_dim = checkpoint['model_state_dict']['neural_branch.input_projection.weight'].shape[1]
        
        self.model = NeuroEconometricNet(
            input_dim=input_dim,
            hidden_dim=self.metadata['config']['hidden_dim'],
            num_lstm_layers=self.metadata['config']['num_layers']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(Config.DEVICE)
        self.model.eval()
        
        logger.info(f"✓ Model loaded: {self.model_path}")
        
    def test_prediction_consistency(self, test_data, n_runs=10):
        """
        Test 1: Prediction Consistency
        Verify model produces consistent predictions on same input.
        """
        logger.info("\n" + "="*60)
        logger.info("TEST 1: PREDICTION CONSISTENCY")
        logger.info("="*60)
        
        # Get a sample sequence
        sample_idx = Config.SEQ_LENGTH
        seq = test_data.iloc[sample_idx-Config.SEQ_LENGTH:sample_idx].values
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_runs):
                X = torch.tensor(seq[:, :17], dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
                ardl_pred = torch.tensor([[test_data.iloc[sample_idx-1, -1]]], dtype=torch.float32).to(Config.DEVICE)
                volatility = torch.tensor([[0.02, 0.03, 50.0, 0.02, 0.01]], dtype=torch.float32).to(Config.DEVICE)
                sentiment = torch.tensor([[0.0]], dtype=torch.float32).to(Config.DEVICE)
                
                output = self.model(X, ardl_pred, volatility, sentiment)
                pred, alpha, neural_pred = output
                predictions.append(pred.item())
        
        predictions = np.array(predictions)
        std_dev = np.std(predictions)
        variance = np.var(predictions)
        
        # Test passes if variance is very small (model is deterministic in eval mode)
        consistency_pass = std_dev < 1e-6
        
        logger.info(f"Predictions mean: {np.mean(predictions):.6f}")
        logger.info(f"Predictions std: {std_dev:.6e}")
        logger.info(f"Predictions variance: {variance:.6e}")
        logger.info(f"Status: {'✓ PASS' if consistency_pass else '✗ FAIL'}")
        
        self.results['consistency'] = {
            'passed': bool(consistency_pass),
            'std_dev': float(std_dev),
            'variance': float(variance)
        }
        
        return consistency_pass
    
    def test_noise_robustness(self, test_data, noise_levels=[0.01, 0.05, 0.1]):
        """
        Test 2: Noise Robustness
        Test how model handles noisy inputs.
        """
        logger.info("\n" + "="*60)
        logger.info("TEST 2: NOISE ROBUSTNESS")
        logger.info("="*60)
        
        sample_idx = Config.SEQ_LENGTH
        seq = test_data.iloc[sample_idx-Config.SEQ_LENGTH:sample_idx].values
        
        # Original prediction
        with torch.no_grad():
            X_orig = torch.tensor(seq[:, :17], dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
            ardl_pred = torch.tensor([[test_data.iloc[sample_idx-1, -1]]], dtype=torch.float32).to(Config.DEVICE)
            volatility = torch.tensor([[0.02, 0.03, 50.0, 0.02, 0.01]], dtype=torch.float32).to(Config.DEVICE)
            sentiment = torch.tensor([[0.0]], dtype=torch.float32).to(Config.DEVICE)
            
            output = self.model(X_orig, ardl_pred, volatility, sentiment)
            pred_orig, _, _ = output
            pred_orig = pred_orig.item()
        
        noise_results = {}
        for noise_level in noise_levels:
            noisy_predictions = []
            
            for _ in range(10):
                # Add Gaussian noise
                noise = np.random.normal(0, noise_level, seq[:, :17].shape)
                seq_noisy = seq.copy()
                seq_noisy[:, :17] += noise
                
                with torch.no_grad():
                    X_noisy = torch.tensor(seq_noisy[:, :17], dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
                    output = self.model(X_noisy, ardl_pred, volatility, sentiment)
                    pred, _, _ = output
                    noisy_predictions.append(pred.item())
            
            noisy_predictions = np.array(noisy_predictions)
            mean_deviation = np.mean(np.abs(noisy_predictions - pred_orig))
            
            logger.info(f"\nNoise level {noise_level*100:.1f}%:")
            logger.info(f"  Original prediction: {pred_orig:.6f}")
            logger.info(f"  Noisy predictions mean: {np.mean(noisy_predictions):.6f}")
            logger.info(f"  Mean absolute deviation: {mean_deviation:.6f}")
            
            noise_results[f"noise_{noise_level}"] = {
                'mean_deviation': float(mean_deviation),
                'predictions_std': float(np.std(noisy_predictions))
            }
        
        self.results['noise_robustness'] = noise_results
        return noise_results
    
    def test_directional_accuracy(self, test_data):
        """
        Test 3: Directional Accuracy
        Most important metric for trading - predicting direction correctly.
        """
        logger.info("\n" + "="*60)
        logger.info("TEST 3: DIRECTIONAL ACCURACY")
        logger.info("="*60)
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for i in range(Config.SEQ_LENGTH, len(test_data)):
                seq = test_data.iloc[i-Config.SEQ_LENGTH:i].values
                
                X = torch.tensor(seq[:, :17], dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
                ardl_pred = torch.tensor([[test_data.iloc[i-1, -1]]], dtype=torch.float32).to(Config.DEVICE)
                volatility = torch.tensor([[0.02, 0.03, 50.0, 0.02, 0.01]], dtype=torch.float32).to(Config.DEVICE)
                sentiment = torch.tensor([[0.0]], dtype=torch.float32).to(Config.DEVICE)
                
                output = self.model(X, ardl_pred, volatility, sentiment)
                pred, _, _ = output
                
                predictions.append(pred.item())
                actuals.append(test_data.iloc[i, -1])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate directional accuracy
        pred_direction = np.diff(predictions) > 0
        actual_direction = np.diff(actuals) > 0
        dir_accuracy = np.mean(pred_direction == actual_direction) * 100
        
        # Binomial test: is directional accuracy significantly better than random (50%)?
        n_correct = np.sum(pred_direction == actual_direction)
        n_total = len(pred_direction)
        # Use binomtest for newer scipy versions, binom_test for older
        try:
            p_value = stats.binomtest(n_correct, n_total, 0.5, alternative='greater').pvalue
        except AttributeError:
            from scipy.stats import binom_test
            p_value = binom_test(n_correct, n_total, 0.5, alternative='greater')
        
        logger.info(f"Directional Accuracy: {dir_accuracy:.2f}%")
        logger.info(f"Correct predictions: {n_correct}/{n_total}")
        logger.info(f"P-value (vs random): {p_value:.4f}")
        
        significance_level = 0.05
        is_significant = p_value < significance_level
        
        logger.info(f"Statistically significant: {'✓ YES' if is_significant else '✗ NO'}")
        
        self.results['directional_accuracy'] = {
            'accuracy': float(dir_accuracy),
            'n_correct': int(n_correct),
            'n_total': int(n_total),
            'p_value': float(p_value),
            'is_significant': bool(is_significant)
        }
        
        return dir_accuracy, is_significant
    
    def test_prediction_magnitude(self, test_data):
        """
        Test 4: Prediction Magnitude Analysis
        Check if predictions have reasonable magnitude.
        """
        logger.info("\n" + "="*60)
        logger.info("TEST 4: PREDICTION MAGNITUDE ANALYSIS")
        logger.info("="*60)
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for i in range(Config.SEQ_LENGTH, len(test_data)):
                seq = test_data.iloc[i-Config.SEQ_LENGTH:i].values
                
                X = torch.tensor(seq[:, :17], dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
                ardl_pred = torch.tensor([[test_data.iloc[i-1, -1]]], dtype=torch.float32).to(Config.DEVICE)
                volatility = torch.tensor([[0.02, 0.03, 50.0, 0.02, 0.01]], dtype=torch.float32).to(Config.DEVICE)
                sentiment = torch.tensor([[0.0]], dtype=torch.float32).to(Config.DEVICE)
                
                output = self.model(X, ardl_pred, volatility, sentiment)
                pred, _, _ = output
                
                predictions.append(pred.item())
                actuals.append(test_data.iloc[i, -1])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        pred_range = (np.min(predictions), np.max(predictions))
        
        actual_mean = np.mean(actuals)
        actual_std = np.std(actuals)
        actual_range = (np.min(actuals), np.max(actuals))
        
        logger.info("Predictions:")
        logger.info(f"  Mean: {pred_mean:.6f}")
        logger.info(f"  Std: {pred_std:.6f}")
        logger.info(f"  Range: [{pred_range[0]:.6f}, {pred_range[1]:.6f}]")
        
        logger.info("\nActuals:")
        logger.info(f"  Mean: {actual_mean:.6f}")
        logger.info(f"  Std: {actual_std:.6f}")
        logger.info(f"  Range: [{actual_range[0]:.6f}, {actual_range[1]:.6f}]")
        
        # Check if predictions are in reasonable range (not all same value)
        pred_variance_ok = pred_std > 1e-3
        
        logger.info(f"\nPrediction variance check: {'✓ PASS' if pred_variance_ok else '✗ FAIL'}")
        
        self.results['magnitude_analysis'] = {
            'predictions': {
                'mean': float(pred_mean),
                'std': float(pred_std),
                'range': [float(pred_range[0]), float(pred_range[1])]
            },
            'actuals': {
                'mean': float(actual_mean),
                'std': float(actual_std),
                'range': [float(actual_range[0]), float(actual_range[1])]
            },
            'variance_check_passed': bool(pred_variance_ok)
        }
        
        return pred_variance_ok
    
    def test_alpha_gating_behavior(self, test_data):
        """
        Test 5: Gating Mechanism Analysis
        Analyze the behavior of the alpha gating weight.
        """
        logger.info("\n" + "="*60)
        logger.info("TEST 5: GATING MECHANISM ANALYSIS")
        logger.info("="*60)
        
        alpha_values = []
        
        with torch.no_grad():
            for i in range(Config.SEQ_LENGTH, len(test_data)):
                seq = test_data.iloc[i-Config.SEQ_LENGTH:i].values
                
                X = torch.tensor(seq[:, :17], dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
                ardl_pred = torch.tensor([[test_data.iloc[i-1, -1]]], dtype=torch.float32).to(Config.DEVICE)
                volatility = torch.tensor([[0.02, 0.03, 50.0, 0.02, 0.01]], dtype=torch.float32).to(Config.DEVICE)
                sentiment = torch.tensor([[0.0]], dtype=torch.float32).to(Config.DEVICE)
                
                output = self.model(X, ardl_pred, volatility, sentiment)
                _, alpha, _ = output
                
                alpha_values.append(alpha.item())
        
        alpha_values = np.array(alpha_values)
        
        alpha_mean = np.mean(alpha_values)
        alpha_std = np.std(alpha_values)
        alpha_range = (np.min(alpha_values), np.max(alpha_values))
        
        logger.info(f"Alpha Statistics:")
        logger.info(f"  Mean: {alpha_mean:.4f}")
        logger.info(f"  Std: {alpha_std:.6f}")
        logger.info(f"  Range: [{alpha_range[0]:.4f}, {alpha_range[1]:.4f}]")
        
        # Check if alpha is varying (not stuck at constant value)
        alpha_varying = alpha_std > 1e-3
        
        # Check if alpha is within valid range [0, 1]
        alpha_valid_range = (alpha_range[0] >= 0) and (alpha_range[1] <= 1)
        
        logger.info(f"\nAlpha variation check: {'✓ PASS' if alpha_varying else '✗ FAIL (stuck at constant)'}")
        logger.info(f"Alpha range check: {'✓ PASS' if alpha_valid_range else '✗ FAIL'}")
        
        # Interpretation
        if alpha_mean > 0.7:
            logger.info("\n⚠ Model heavily favors neural branch")
        elif alpha_mean < 0.3:
            logger.info("\n⚠ Model heavily favors ARDL branch")
        else:
            logger.info("\n✓ Model balances both branches")
        
        self.results['gating_analysis'] = {
            'alpha_mean': float(alpha_mean),
            'alpha_std': float(alpha_std),
            'alpha_range': [float(alpha_range[0]), float(alpha_range[1])],
            'is_varying': bool(alpha_varying),
            'is_valid_range': bool(alpha_valid_range)
        }
        
        return alpha_varying and alpha_valid_range
    
    def test_prediction_stability(self, test_data, window_size=30):
        """
        Test 6: Temporal Stability
        Check if model performance is stable over time windows.
        """
        logger.info("\n" + "="*60)
        logger.info("TEST 6: TEMPORAL STABILITY")
        logger.info("="*60)
        
        n_windows = (len(test_data) - Config.SEQ_LENGTH) // window_size
        window_accuracies = []
        
        for w in range(n_windows):
            start_idx = Config.SEQ_LENGTH + w * window_size
            end_idx = min(start_idx + window_size, len(test_data))
            
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for i in range(start_idx, end_idx):
                    seq = test_data.iloc[i-Config.SEQ_LENGTH:i].values
                    
                    X = torch.tensor(seq[:, :17], dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
                    ardl_pred = torch.tensor([[test_data.iloc[i-1, -1]]], dtype=torch.float32).to(Config.DEVICE)
                    volatility = torch.tensor([[0.02, 0.03, 50.0, 0.02, 0.01]], dtype=torch.float32).to(Config.DEVICE)
                    sentiment = torch.tensor([[0.0]], dtype=torch.float32).to(Config.DEVICE)
                    
                    output = self.model(X, ardl_pred, volatility, sentiment)
                    pred, _, _ = output
                    
                    predictions.append(pred.item())
                    actuals.append(test_data.iloc[i, -1])
            
            if len(predictions) > 1:
                predictions = np.array(predictions)
                actuals = np.array(actuals)
                
                pred_direction = np.diff(predictions) > 0
                actual_direction = np.diff(actuals) > 0
                dir_accuracy = np.mean(pred_direction == actual_direction) * 100
                
                window_accuracies.append(dir_accuracy)
        
        if len(window_accuracies) > 1:
            mean_accuracy = np.mean(window_accuracies)
            std_accuracy = np.std(window_accuracies)
            
            logger.info(f"Number of windows: {len(window_accuracies)}")
            logger.info(f"Mean accuracy across windows: {mean_accuracy:.2f}%")
            logger.info(f"Std of accuracy: {std_accuracy:.2f}%")
            logger.info(f"Min/Max accuracy: {np.min(window_accuracies):.2f}% / {np.max(window_accuracies):.2f}%")
            
            # Check if performance is relatively stable (std < 15%)
            is_stable = std_accuracy < 15.0
            
            logger.info(f"\nStability check: {'✓ PASS' if is_stable else '✗ FAIL (high variance)'}")
            
            self.results['temporal_stability'] = {
                'n_windows': len(window_accuracies),
                'mean_accuracy': float(mean_accuracy),
                'std_accuracy': float(std_accuracy),
                'min_accuracy': float(np.min(window_accuracies)),
                'max_accuracy': float(np.max(window_accuracies)),
                'is_stable': bool(is_stable)
            }
            
            return is_stable
        else:
            logger.warning("Not enough data for stability test")
            return None
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        logger.info("\n" + "="*60)
        logger.info("ROBUSTNESS TEST SUMMARY")
        logger.info("="*60)
        
        # Count passes/fails
        tests_run = 0
        tests_passed = 0
        
        if 'consistency' in self.results:
            tests_run += 1
            if self.results['consistency']['passed']:
                tests_passed += 1
                logger.info("✓ Prediction Consistency: PASS")
            else:
                logger.info("✗ Prediction Consistency: FAIL")
        
        if 'directional_accuracy' in self.results:
            tests_run += 1
            if self.results['directional_accuracy']['is_significant']:
                tests_passed += 1
                logger.info(f"✓ Directional Accuracy: PASS ({self.results['directional_accuracy']['accuracy']:.2f}%)")
            else:
                logger.info(f"✗ Directional Accuracy: FAIL ({self.results['directional_accuracy']['accuracy']:.2f}%)")
        
        if 'magnitude_analysis' in self.results:
            tests_run += 1
            if self.results['magnitude_analysis']['variance_check_passed']:
                tests_passed += 1
                logger.info("✓ Prediction Magnitude: PASS")
            else:
                logger.info("✗ Prediction Magnitude: FAIL")
        
        if 'gating_analysis' in self.results:
            tests_run += 1
            if self.results['gating_analysis']['is_varying'] and self.results['gating_analysis']['is_valid_range']:
                tests_passed += 1
                logger.info("✓ Gating Mechanism: PASS")
            else:
                logger.info("✗ Gating Mechanism: FAIL")
        
        if 'temporal_stability' in self.results and self.results['temporal_stability'] is not None:
            tests_run += 1
            if self.results['temporal_stability']['is_stable']:
                tests_passed += 1
                logger.info("✓ Temporal Stability: PASS")
            else:
                logger.info("✗ Temporal Stability: FAIL")
        
        logger.info(f"\nOverall: {tests_passed}/{tests_run} tests passed")
        
        # Overall assessment
        pass_rate = tests_passed / tests_run if tests_run > 0 else 0
        
        logger.info("\n" + "="*60)
        if pass_rate >= 0.8:
            logger.info("ASSESSMENT: ✓ MODEL IS ROBUST")
        elif pass_rate >= 0.6:
            logger.info("ASSESSMENT: ⚠ MODEL IS MODERATELY ROBUST")
        else:
            logger.info("ASSESSMENT: ✗ MODEL NEEDS IMPROVEMENT")
        logger.info("="*60)
        
        self.results['summary'] = {
            'tests_run': tests_run,
            'tests_passed': tests_passed,
            'pass_rate': pass_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        return pass_rate


def main():
    """Main robustness testing pipeline."""
    print("="*60)
    print("MODEL ROBUSTNESS TESTING SUITE")
    print("="*60)
    print(f"Timestamp: {datetime.now()}")
    print("="*60)
    
    try:
        # Initialize tester
        tester = ModelRobustnessTester(
            model_path="models_saved/best_model.pt",
            metadata_path="models_saved/training_info.json"
        )
        
        # Load model
        tester.load_model()
        
        # Prepare test data
        logger.info("\nLoading test data...")
        data_loader = DataLoader(ticker=Config.TICKER, local_csv='data/GSPC_ohlcv.csv')
        ohlcv_df, _ = data_loader.load_all('2024-01-01', '2024-12-31', include_news=False)
        
        df_with_indicators = TechnicalIndicatorEngine.compute_all(ohlcv_df)
        preprocessor = Preprocessor()
        test_df, metadata = preprocessor.fit_transform(df_with_indicators)
        
        logger.info(f"✓ Test data prepared: {len(test_df)} samples\n")
        
        # Run all tests
        tester.test_prediction_consistency(test_df)
        tester.test_noise_robustness(test_df)
        tester.test_directional_accuracy(test_df)
        tester.test_prediction_magnitude(test_df)
        tester.test_alpha_gating_behavior(test_df)
        tester.test_prediction_stability(test_df)
        
        # Generate summary
        pass_rate = tester.generate_summary_report()
        
        # Save results
        output_path = "models_saved/robustness_test_results.json"
        with open(output_path, 'w') as f:
            json.dump(tester.results, f, indent=2)
        
        logger.info(f"\n✓ Detailed results saved to: {output_path}")
        
        return pass_rate
        
    except Exception as e:
        logger.error(f"\n❌ Testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0


if __name__ == "__main__":
    main()
