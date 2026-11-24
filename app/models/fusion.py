"""
Fusion Network: gated blend of the AR(5) econometric branch and the neural branch.

The econometric input is a rolling AR(5) forecast of the next close, computed
outside the network (run_pipeline.precompute_ardl_predictions; "ardl" in that
name and in `ardl_predictions` below is historical). ARIMA is a separate
baseline and the ARDL bounds test a separate diagnostic; neither feeds the gate.

Critical fix applied: the previous code ran neural_branch.forward() AND
neural_branch.extract_representation() as two separate calls, effectively
running the encoder twice per sample.  Now forward() returns
(prediction, latent) together so the fusion gate receives the latent
representation without any duplicate computation.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging

from app.config import Config
from app.models.neural import HybridNeuralEncoder, VolatilityRegimeDetector

logger = logging.getLogger(__name__)


class GatedFusionMechanism(nn.Module):
    """
    Learnable context-dependent gate.

        α = σ(W · [latent_state ‖ volatility_regime ‖ sentiment] + b)

        final = α · Y_neural + (1-α) · Y_AR

    α → 1 : trust the neural branch  (non-linear / turbulent regime)
    α → 0 : trust the AR(5) branch   (linear / quiet regime)

    The gate is initialised so that α ≈ 0.5 at the start of training,
    which prevents the gate from being frozen near 0 or 1.
    """

    def __init__(self, state_dim: int, hidden_dim: int = Config.FUSION_HIDDEN_DIM):
        super().__init__()
        # input = latent(state_dim) + volatility(1) + sentiment(1)
        in_dim = state_dim + 2
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        # Initialise the final bias to 0 so sigmoid(0) = 0.5
        nn.init.zeros_(self.gate[-2].bias)
        logger.info(f"GatedFusionMechanism: state_dim={state_dim}, hidden={hidden_dim}")

    def forward(
        self,
        market_state: torch.Tensor,   # (B, state_dim)
        volatility: torch.Tensor,      # (B, 1)
        sentiment: torch.Tensor,       # (B, 1)
    ) -> torch.Tensor:                 # (B, 1)
        gate_input = torch.cat([market_state, volatility, sentiment], dim=1)
        return self.gate(gate_input)


class NeuroEconometricNet(nn.Module):
    """
    Complete Neuro-Econometric Market Alpha Engine.

    Forward pass (single encoding pass, no duplication):
        1. latent, Y_neural = NeuralBranch(x)          ← one pass only
        2. vol_regime       = VolatilityDetector(v)
        3. α                = FusionGate(latent, vol, sent)
        4. Y_fused          = α·Y_neural + (1-α)·Y_AR   (AR(5) forecast, an input)
        5. Y_out            = OutputLayer(Y_fused)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = Config.HIDDEN_DIM,
        nhead: int = Config.NHEAD,
        num_lstm_layers: int = Config.NUM_LAYERS,
        dropout: float = Config.DROPOUT,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.neural_branch = HybridNeuralEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
        )

        self.volatility_detector = VolatilityRegimeDetector(input_dim=5, hidden_dim=32)

        self.fusion_gate = GatedFusionMechanism(
            state_dim=hidden_dim,
            hidden_dim=Config.FUSION_HIDDEN_DIM,
        )

        # Lightweight refinement after fusion
        self.output_layer = nn.Linear(1, 1)
        nn.init.ones_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        x: torch.Tensor,               # (B, seq_len, input_dim)
        ardl_predictions: torch.Tensor, # (B, 1) the rolling AR(5) forecast
        volatility_features: torch.Tensor, # (B, 5)
        sentiment_scores: torch.Tensor, # (B, 1)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (fused_prediction, alpha, neural_prediction)
        """
        # ── single neural-branch pass ──────────────────────────────
        neural_prediction, market_state = self.neural_branch(x)   # (B,1), (B,H)

        # ── volatility regime ──────────────────────────────────────
        vol_regime = self.volatility_detector(volatility_features)  # (B,1)

        # ── gating weight ──────────────────────────────────────────
        alpha = self.fusion_gate(market_state, vol_regime, sentiment_scores)  # (B,1)

        # ── fusion ─────────────────────────────────────────────────
        fused = alpha * neural_prediction + (1.0 - alpha) * ardl_predictions
        fused = self.output_layer(fused)

        return fused, alpha, neural_prediction
