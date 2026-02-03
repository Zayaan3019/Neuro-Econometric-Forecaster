"""
Neural Network Architectures for Non-Linear Market Modeling.

This module implements the "Neural Branch" of the Neuro-Econometric Engine,
combining Transformers and LSTMs to capture complex non-linear patterns
in multi-modal financial data (price + sentiment).
"""

from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from app.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer architecture.
    
    Mathematical Foundation:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        Where:
        - pos: Position in sequence
        - i: Dimension index
        - d_model: Embedding dimension
    
    Rationale:
        Transformers have no inherent sense of sequence order. Positional
        encodings inject temporal information, allowing the model to
        distinguish between Y_t and Y_{t-1}.
    
    Reference:
        Vaswani et al. (2017). "Attention is All You Need." NeurIPS.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Dimension of embeddings.
            max_len: Maximum sequence length.
            dropout: Dropout rate for regularization.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor. Shape: (seq_len, batch_size, d_model)
        
        Returns:
            Tensor with positional encoding added. Shape: (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism for Transformers.
    
    Mathematical Foundation:
        Attention(Q, K, V) = softmax(QK^T / √d_k) V
        
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        Where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    Intuition:
        Attention allows the model to focus on relevant past time steps
        when making predictions. Multiple heads capture different types
        of dependencies (e.g., trend, seasonality, volatility clusters).
    
    Application in Finance:
        - Head 1 might focus on recent price momentum
        - Head 2 might focus on volatility regimes
        - Head 3 might correlate sentiment with price movements
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Dimension of model embeddings.
            nhead: Number of attention heads.
            dropout: Dropout rate.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.
        
        Args:
            query: Query tensor. Shape: (batch, seq_len, d_model)
            key: Key tensor. Shape: (batch, seq_len, d_model)
            value: Value tensor. Shape: (batch, seq_len, d_model)
            mask: Attention mask (optional). Shape: (batch, seq_len, seq_len)
        
        Returns:
            Tuple of (attention_output, attention_weights).
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply final linear
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output, attention_weights


class TransformerEncoder(nn.Module):
    """
    Transformer encoder block for sequence modeling.
    
    Architecture:
        Input → Multi-Head Attention → Add & Norm → Feed-Forward → Add & Norm → Output
    
    The encoder captures complex temporal dependencies without the
    vanishing gradient problems of vanilla RNNs.
    """
    
    def __init__(
        self,
        d_model: int = Config.HIDDEN_DIM,
        nhead: int = Config.NHEAD,
        dim_feedforward: int = 512,
        dropout: float = Config.DROPOUT
    ):
        """
        Initialize Transformer encoder.
        
        Args:
            d_model: Dimension of model embeddings.
            nhead: Number of attention heads.
            dim_feedforward: Dimension of feed-forward network.
            dropout: Dropout rate.
        """
        super(TransformerEncoder, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Transformer encoder.
        
        Args:
            x: Input tensor. Shape: (batch, seq_len, d_model)
            mask: Attention mask (optional).
        
        Returns:
            Encoded tensor. Shape: (batch, seq_len, d_model)
        """
        # Multi-head attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class LSTMEncoder(nn.Module):
    """
    LSTM encoder for sequential modeling.
    
    Mathematical Foundation:
        LSTM Cell Equations:
        f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     # Forget gate
        i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     # Input gate
        C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  # Candidate cell state
        C_t = f_t * C_{t-1} + i_t * C̃_t         # Cell state update
        o_t = σ(W_o · [h_{t-1}, x_t] + b_o)     # Output gate
        h_t = o_t * tanh(C_t)                   # Hidden state
    
    Advantages:
        - Mitigates vanishing gradient problem
        - Captures long-term dependencies
        - Proven effectiveness in financial time series
    
    Reference:
        Hochreiter & Schmidhuber (1997). "Long Short-Term Memory."
        Neural Computation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = Config.HIDDEN_DIM,
        num_layers: int = Config.NUM_LAYERS,
        dropout: float = Config.DROPOUT,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM encoder.
        
        Args:
            input_dim: Dimension of input features.
            hidden_dim: Dimension of LSTM hidden state.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout rate (applied between layers).
            bidirectional: Whether to use bidirectional LSTM.
        """
        super(LSTMEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM.
        
        Args:
            x: Input tensor. Shape: (batch, seq_len, input_dim)
            hidden: Initial hidden state (optional).
        
        Returns:
            Tuple of (output, (h_n, c_n)).
            - output: All hidden states. Shape: (batch, seq_len, hidden_dim * directions)
            - h_n: Final hidden state. Shape: (num_layers * directions, batch, hidden_dim)
            - c_n: Final cell state. Shape: (num_layers * directions, batch, hidden_dim)
        """
        output, (h_n, c_n) = self.lstm(x, hidden)
        output = self.dropout(output)
        
        return output, (h_n, c_n)


class HybridNeuralEncoder(nn.Module):
    """
    Hybrid architecture combining Transformer and LSTM.
    
    Architecture Flow:
        Input Features → Projection → Transformer → LSTM → Output Representation
    
    Rationale:
        - Transformer: Captures global dependencies and parallel patterns
        - LSTM: Refines sequential dependencies and handles noise
        - Combination: Best of both worlds - global + local temporal modeling
    
    This is the "Neural Branch" that will be fused with the ARDL branch.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = Config.HIDDEN_DIM,
        nhead: int = Config.NHEAD,
        num_lstm_layers: int = Config.NUM_LAYERS,
        dropout: float = Config.DROPOUT
    ):
        """
        Initialize hybrid neural encoder.
        
        Args:
            input_dim: Dimension of input features (technical indicators + sentiment).
            hidden_dim: Hidden dimension for both Transformer and LSTM.
            nhead: Number of attention heads in Transformer.
            num_lstm_layers: Number of LSTM layers.
            dropout: Dropout rate.
        """
        super(HybridNeuralEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection to match Transformer dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )
        
        # LSTM encoder
        self.lstm = LSTMEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout,
            bidirectional=False
        )
        
        # Output layer
        self.output_projection = nn.Linear(hidden_dim, 1)  # Single output (price prediction)
        
        logger.info(f"Initialized HybridNeuralEncoder with input_dim={input_dim}, hidden_dim={hidden_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid encoder.
        
        Args:
            x: Input tensor. Shape: (batch, seq_len, input_dim)
        
        Returns:
            Prediction tensor. Shape: (batch, 1)
        
        Architecture Flow:
            1. Project input to hidden dimension
            2. Add positional encoding
            3. Pass through Transformer (global patterns)
            4. Pass through LSTM (sequential refinement)
            5. Extract final hidden state
            6. Project to single output
        """
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, hidden_dim)
        
        # Add positional encoding (requires seq_len first)
        x = x.transpose(0, 1)  # (seq_len, batch, hidden_dim)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch, seq_len, hidden_dim)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len, hidden_dim)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_dim)
        
        # Use final hidden state for prediction
        final_hidden = h_n[-1]  # (batch, hidden_dim)
        
        # Output projection
        output = self.output_projection(final_hidden)  # (batch, 1)
        
        return output
    
    def extract_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract latent representation without final projection.
        
        Args:
            x: Input tensor. Shape: (batch, seq_len, input_dim)
        
        Returns:
            Latent representation. Shape: (batch, hidden_dim)
        
        Application:
            This representation is used by the fusion network to compute
            gating weights and combine with ARDL predictions.
        """
        batch_size = x.size(0)
        
        # Same as forward, but stop before final projection
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Return final hidden state
        return h_n[-1]  # (batch, hidden_dim)


class VolatilityRegimeDetector(nn.Module):
    """
    Neural network for detecting market volatility regimes.
    
    Application:
        Used by the fusion network to adjust gating weights.
        High volatility → Trust neural network more (non-linear patterns)
        Low volatility → Trust ARDL more (stable linear relationships)
    
    Architecture:
        Input Features → Dense → ReLU → Dense → Sigmoid → Regime Probability
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Initialize volatility regime detector.
        
        Args:
            input_dim: Dimension of input features (typically includes ATR, BB_Width).
            hidden_dim: Hidden dimension for dense layers.
        """
        super(VolatilityRegimeDetector, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict volatility regime.
        
        Args:
            x: Input features. Shape: (batch, input_dim)
        
        Returns:
            Regime probability. Shape: (batch, 1)
            - 0.0: Low volatility (stable regime)
            - 1.0: High volatility (chaotic regime)
        """
        return self.network(x)
