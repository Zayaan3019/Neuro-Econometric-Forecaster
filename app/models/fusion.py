"""
Fusion Network: The Core of the Neuro-Econometric Engine.

This module implements the "Gated Fusion Mechanism" that dynamically combines
predictions from the Linear Branch (ARDL) and Non-Linear Branch (Neural Network)
based on market volatility regimes.
"""

from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging

from app.config import Config
from app.models.neural import HybridNeuralEncoder, VolatilityRegimeDetector
from app.models.econometrics import ARDLModel, EconometricPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GatedFusionMechanism(nn.Module):
    """
    Learnable gating mechanism for fusion.
    
    Mathematical Foundation:
        α = σ(W · [State, Volatility, Sentiment] + b)
        
        Final_Prediction = α · Y_neural + (1 - α) · Y_ardl
    
    Where:
        - α ∈ [0, 1]: Gating weight (learned)
        - σ: Sigmoid activation (ensures α ∈ [0, 1])
        - State: Market state representation from neural network
        - Volatility: Current volatility regime
        - Sentiment: Aggregate sentiment score
    
    Intuition:
        - When α → 1: Trust neural network (non-linear regime)
        - When α → 0: Trust ARDL (linear regime)
        - α is learned from data, not hand-crafted
    
    Key Innovation:
        Unlike simple averaging, the gating is context-dependent:
        - High volatility + extreme sentiment → Higher α
        - Low volatility + neutral sentiment → Lower α
    """
    
    def __init__(
        self,
        state_dim: int = Config.HIDDEN_DIM,
        hidden_dim: int = Config.FUSION_HIDDEN_DIM
    ):
        """
        Initialize gated fusion mechanism.
        
        Args:
            state_dim: Dimension of market state representation.
            hidden_dim: Hidden dimension for gating network.
        """
        super(GatedFusionMechanism, self).__init__()
        
        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(state_dim + 2, hidden_dim),  # +2 for volatility and sentiment
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output α ∈ [0, 1]
        )
        
        logger.info(f"Initialized GatedFusionMechanism with state_dim={state_dim}")
    
    def forward(
        self,
        market_state: torch.Tensor,
        volatility: torch.Tensor,
        sentiment: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gating weight α.
        
        Args:
            market_state: Latent market representation. Shape: (batch, state_dim)
            volatility: Volatility measure. Shape: (batch, 1)
            sentiment: Sentiment score. Shape: (batch, 1)
        
        Returns:
            Gating weight α. Shape: (batch, 1)
        """
        # Concatenate inputs
        gate_input = torch.cat([market_state, volatility, sentiment], dim=1)
        
        # Compute gating weight
        alpha = self.gate_network(gate_input)
        
        return alpha


class NeuroEconometricNet(nn.Module):
    """
    The complete Neuro-Econometric Market Alpha Engine.
    
    Architecture:
        ┌─────────────────────────────────────────────────────┐
        │                   Input Data                         │
        │  (Price OHLCV + Technical Indicators + Sentiment)    │
        └────────────────┬────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            │                         │
    ┌───────▼──────┐         ┌────────▼─────────┐
    │ ARDL Branch  │         │  Neural Branch   │
    │  (Linear)    │         │  (Non-Linear)    │
    │              │         │  Transformer +   │
    │ Econometric  │         │  LSTM            │
    │ Forecasting  │         │                  │
    └───────┬──────┘         └────────┬─────────┘
            │                         │
            │  Y_ardl                 │  Y_neural
            │                         │
            └────────────┬────────────┘
                         │
                ┌────────▼────────┐
                │ Gated Fusion    │
                │ α = f(State)    │
                │ Y = α·Y_neural  │
                │   + (1-α)·Y_ardl│
                └────────┬────────┘
                         │
                    ┌────▼─────┐
                    │ Output:  │
                    │ Alpha    │
                    │ Signal   │
                    └──────────┘
    
    Training Strategy:
        End-to-end training with backpropagation through both branches.
        The gating mechanism learns to weight branches based on performance.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = Config.HIDDEN_DIM,
        nhead: int = Config.NHEAD,
        num_lstm_layers: int = Config.NUM_LAYERS,
        dropout: float = Config.DROPOUT,
        ardl_lags: int = Config.ARDL_LAGS
    ):
        """
        Initialize Neuro-Econometric Network.
        
        Args:
            input_dim: Dimension of input features.
            hidden_dim: Hidden dimension for neural networks.
            nhead: Number of attention heads.
            num_lstm_layers: Number of LSTM layers.
            dropout: Dropout rate.
            ardl_lags: Number of ARDL lags.
        """
        super(NeuroEconometricNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Neural Branch: Hybrid Transformer + LSTM
        self.neural_branch = HybridNeuralEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout
        )
        
        # ARDL Branch (initialized later with data)
        self.ardl_predictor = EconometricPredictor(lags=ardl_lags)
        self.ardl_lags = ardl_lags
        
        # Volatility Regime Detector
        self.volatility_detector = VolatilityRegimeDetector(
            input_dim=5,  # ATR, BB_Width, ADX, StdDev, Range
            hidden_dim=32
        )
        
        # Gated Fusion Mechanism
        self.fusion_gate = GatedFusionMechanism(
            state_dim=hidden_dim,
            hidden_dim=Config.FUSION_HIDDEN_DIM
        )
        
        # Final output layer
        self.output_layer = nn.Linear(1, 1)
        
        logger.info("Initialized NeuroEconometricNet")
    
    def forward(
        self,
        x: torch.Tensor,
        ardl_predictions: torch.Tensor,
        volatility_features: torch.Tensor,
        sentiment_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through complete Neuro-Econometric Network.
        
        Args:
            x: Input features for neural branch. Shape: (batch, seq_len, input_dim)
            ardl_predictions: Pre-computed ARDL predictions. Shape: (batch, 1)
            volatility_features: Features for volatility detection. Shape: (batch, 5)
            sentiment_scores: Sentiment scores. Shape: (batch, 1)
        
        Returns:
            Tuple of (fused_prediction, alpha, neural_prediction).
            - fused_prediction: Final output. Shape: (batch, 1)
            - alpha: Gating weight. Shape: (batch, 1)
            - neural_prediction: Neural branch output. Shape: (batch, 1)
        
        Mathematical Flow:
            1. Y_neural = NeuralBranch(x)
            2. Y_ardl = ardl_predictions (pre-computed)
            3. State = NeuralBranch.extract_representation(x)
            4. Volatility = VolatilityDetector(volatility_features)
            5. α = FusionGate(State, Volatility, Sentiment)
            6. Y_fused = α · Y_neural + (1 - α) · Y_ardl
        """
        batch_size = x.size(0)
        
        # Step 1: Neural Branch Prediction
        neural_prediction = self.neural_branch(x)  # (batch, 1)
        
        # Step 2: Extract market state representation
        market_state = self.neural_branch.extract_representation(x)  # (batch, hidden_dim)
        
        # Step 3: Detect volatility regime
        volatility_regime = self.volatility_detector(volatility_features)  # (batch, 1)
        
        # Step 4: Compute gating weight α
        alpha = self.fusion_gate(market_state, volatility_regime, sentiment_scores)  # (batch, 1)
        
        # Step 5: Fuse predictions
        fused_prediction = alpha * neural_prediction + (1 - alpha) * ardl_predictions
        
        # Step 6: Final output layer (optional refinement)
        fused_prediction = self.output_layer(fused_prediction)
        
        return fused_prediction, alpha, neural_prediction
    
    def predict_alpha_signal(
        self,
        prediction: torch.Tensor,
        current_price: float,
        threshold_buy: float = 0.02,
        threshold_sell: float = -0.02
    ) -> torch.Tensor:
        """
        Convert price prediction to alpha signal (Buy/Sell/Hold).
        
        Args:
            prediction: Predicted price change. Shape: (batch, 1)
            current_price: Current market price.
            threshold_buy: Minimum return to trigger Buy signal (e.g., 2%).
            threshold_sell: Maximum return to trigger Sell signal (e.g., -2%).
        
        Returns:
            Alpha signal. Shape: (batch, 1)
            - 1.0: Buy (bullish)
            - 0.0: Hold (neutral)
            - -1.0: Sell (bearish)
        
        Mathematical Formulation:
            Expected_Return = (Prediction - Current_Price) / Current_Price
            
            Signal = {  1 if Expected_Return > threshold_buy
                       -1 if Expected_Return < threshold_sell
                        0 otherwise
        """
        # Compute expected return
        expected_return = (prediction - current_price) / current_price
        
        # Generate signals
        signals = torch.zeros_like(expected_return)
        signals[expected_return > threshold_buy] = 1.0
        signals[expected_return < threshold_sell] = -1.0
        
        return signals


class HybridTrainingWrapper:
    """
    Wrapper for training the Neuro-Econometric Network.
    
    Handles:
    1. ARDL fitting on each batch
    2. Neural network forward pass
    3. Loss computation and backpropagation
    4. Logging and metrics
    """
    
    def __init__(
        self,
        model: NeuroEconometricNet,
        device: torch.device = Config.DEVICE
    ):
        """
        Initialize training wrapper.
        
        Args:
            model: NeuroEconometricNet instance.
            device: Computation device.
        """
        self.model = model.to(device)
        self.device = device
    
    def compute_ardl_predictions(
        self,
        target_series: pd.Series,
        features: Optional[pd.DataFrame] = None,
        horizon: int = 1
    ) -> torch.Tensor:
        """
        Compute ARDL predictions for a batch.
        
        Args:
            target_series: Target time series.
            features: Exogenous features (optional).
            horizon: Forecast horizon.
        
        Returns:
            ARDL predictions as PyTorch tensor.
        """
        try:
            ardl_preds, _ = self.model.ardl_predictor.fit_predict(
                target=target_series,
                features=features,
                horizon=horizon
            )
            
            # Convert to tensor
            ardl_tensor = torch.tensor(ardl_preds, dtype=torch.float32).unsqueeze(1)
            return ardl_tensor.to(self.device)
        
        except Exception as e:
            logger.warning(f"ARDL prediction failed: {str(e)}. Using zero prediction.")
            return torch.zeros(1, 1, device=self.device)
    
    def prepare_batch(
        self,
        batch_data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare a training batch with all required inputs.
        
        Args:
            batch_data: Dictionary containing:
                - 'features': Neural network input features
                - 'target': Target values
                - 'volatility_features': Volatility indicators
                - 'sentiment': Sentiment scores
                - 'price_history': Historical prices for ARDL
        
        Returns:
            Tuple of (features, ardl_preds, volatility_features, sentiment, targets).
        """
        features = batch_data['features'].to(self.device)
        targets = batch_data['target'].to(self.device)
        volatility_features = batch_data['volatility_features'].to(self.device)
        sentiment = batch_data['sentiment'].to(self.device)
        
        # Compute ARDL predictions
        price_history = batch_data['price_history']
        ardl_preds = self.compute_ardl_predictions(price_history)
        
        return features, ardl_preds, volatility_features, sentiment, targets


class EnsembleUncertaintyEstimator:
    """
    Estimate prediction uncertainty using ensemble methods.
    
    Application:
        Uncertainty quantification helps with:
        - Risk management (avoid trades with high uncertainty)
        - Position sizing (larger positions when confident)
        - Model validation (detect distribution shift)
    
    Method:
        Train multiple instances with different initializations,
        compute prediction variance across ensemble.
    """
    
    def __init__(self, num_models: int = 5):
        """
        Initialize ensemble estimator.
        
        Args:
            num_models: Number of models in ensemble.
        """
        self.num_models = num_models
        self.models: list = []
    
    def fit_ensemble(
        self,
        model_config: Dict[str, Any],
        training_data: Any,
        training_fn: callable
    ) -> None:
        """
        Train ensemble of models.
        
        Args:
            model_config: Configuration for model initialization.
            training_data: Data for training.
            training_fn: Function that trains a single model.
        """
        for i in range(self.num_models):
            logger.info(f"Training ensemble model {i+1}/{self.num_models}")
            
            # Initialize model with different seed
            Config.SEED = Config.SEED + i
            Config.set_seed()
            
            model = NeuroEconometricNet(**model_config)
            trained_model = training_fn(model, training_data)
            self.models.append(trained_model)
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation.
        
        Args:
            x: Input features.
            **kwargs: Additional arguments for forward pass.
        
        Returns:
            Tuple of (mean_prediction, std_prediction).
        """
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                pred, _, _ = model(x, **kwargs)
                predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (num_models, batch, 1)
        
        # Compute mean and std
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred
