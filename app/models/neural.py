"""
Neural Network Architectures for Non-Linear Market Modeling.

Implements the Neural Branch of the Neuro-Econometric Engine combining
stacked Transformers with causal masking and LSTM for temporal modeling.

Key fix: extract_representation() now shares computation with forward()
by caching the latent state — eliminates the previous double-forward-pass bug.
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from app.config import Config

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)          # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block with causal (autoregressive) masking.

    Causal masking is critical for financial forecasting: the model must
    not attend to future time steps, which would introduce look-ahead bias.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask so position t can only attend to 0..t."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        return mask  # True = ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        mask = self._causal_mask(seq_len, x.device)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class HybridNeuralEncoder(nn.Module):
    """
    Hybrid Transformer → LSTM encoder.

    Architecture:
        features → linear projection → positional encoding
               → stacked TransformerBlocks (causal)
               → LSTM
               → latent state h (batch, hidden_dim)
               → output projection → scalar prediction

    IMPORTANT: forward() returns (prediction, latent_state) so that
    the fusion network can reuse the latent state without a second
    forward pass.  extract_representation() is kept for backwards
    compatibility but simply calls forward().
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = Config.HIDDEN_DIM,
        nhead: int = Config.NHEAD,
        num_lstm_layers: int = Config.NUM_LAYERS,
        dropout: float = Config.DROPOUT,
        num_transformer_layers: int = Config.TRANSFORMER_LAYERS,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, dropout=dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
            )
            for _ in range(num_transformer_layers)
        ])

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.lstm_dropout = nn.Dropout(dropout)

        self.output_proj = nn.Linear(hidden_dim, 1)
        # Zero-init output so model starts predicting ~0 returns (correct prior)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self._init_weights()
        logger.info(
            f"HybridNeuralEncoder: input={input_dim}, hidden={hidden_dim}, "
            f"nhead={nhead}, tf_layers={num_transformer_layers}, lstm_layers={num_lstm_layers}"
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.output_proj:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared encoding path → latent state (batch, hidden_dim)."""
        x = self.input_proj(x)           # (B, T, H)
        x = self.pos_enc(x)
        for block in self.transformer_blocks:
            x = block(x)
        _, (h_n, _) = self.lstm(x)
        latent = self.lstm_dropout(h_n[-1])  # (B, H)
        return latent

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            prediction:  (batch, 1)
            latent:      (batch, hidden_dim)  — reused by fusion gate
        """
        latent = self._encode(x)
        return self.output_proj(latent), latent

    def extract_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Returns latent state without re-running the full network."""
        return self._encode(x)


class VolatilityRegimeDetector(nn.Module):
    """
    Small MLP that maps volatility features → regime probability ∈ [0,1].

    0 = low-vol / trending, 1 = high-vol / chaotic.
    Used by the fusion gate to tilt toward the neural branch in turbulent regimes.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
