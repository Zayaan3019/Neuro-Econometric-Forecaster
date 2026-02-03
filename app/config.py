"""
Configuration Module for Neuro-Econometric Market Alpha Engine.

This module defines all hyperparameters, paths, and global settings required
for reproducible experimentation and production deployment.
"""

from pathlib import Path
from typing import Dict, Any
import torch
import numpy as np
import random


class Config:
    """
    Global configuration class with strict type hints.
    
    Attributes:
        SEED: Random seed for reproducibility across PyTorch, NumPy, and Python random.
        DEVICE: Computation device (CUDA if available, else CPU).
        DATA_DIR: Root directory for storing raw and processed data.
        MODEL_DIR: Directory for saving trained model checkpoints.
    """
    
    # ============================================================
    # REPRODUCIBILITY
    # ============================================================
    SEED: int = 42
    
    # ============================================================
    # COMPUTE RESOURCES
    # ============================================================
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS: int = 4
    
    # ============================================================
    # PATHS
    # ============================================================
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODEL_DIR: Path = PROJECT_ROOT / "models_saved"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    
    # ============================================================
    # DATA CONFIGURATION
    # ============================================================
    TICKER: str = "^GSPC"  # S&P 500 Index (Change to "BTC-USD" for Bitcoin)
    START_DATE: str = "2015-01-01"
    END_DATE: str = "2024-12-31"
    NEWS_API_KEY: str = "YOUR_NEWS_API_KEY_HERE"  # Replace with actual key
    
    # ============================================================
    # ECONOMETRIC PARAMETERS (ARDL)
    # ============================================================
    ARDL_LAGS: int = 5  # Autoregressive lags
    ARDL_MAX_DIFF_ORDER: int = 2  # Maximum differencing order for stationarity
    ADF_PVALUE_THRESHOLD: float = 0.05  # Significance level for ADF test
    
    # ============================================================
    # NEURAL NETWORK ARCHITECTURE
    # ============================================================
    HIDDEN_DIM: int = 128  # LSTM hidden dimension
    NUM_LAYERS: int = 2  # Number of LSTM layers
    NHEAD: int = 8  # Number of attention heads in Transformer
    DROPOUT: float = 0.3  # Dropout rate for regularization
    SEQ_LENGTH: int = 60  # Lookback window (e.g., 60 days)
    
    # ============================================================
    # FUSION NETWORK PARAMETERS
    # ============================================================
    FUSION_HIDDEN_DIM: int = 64  # Hidden dimension for gating mechanism
    ALPHA_INIT: float = 0.5  # Initial weight for neural vs econometric blend
    
    # ============================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 100
    EARLY_STOP_PATIENCE: int = 15
    WEIGHT_DECAY: float = 1e-5  # L2 regularization
    
    # ============================================================
    # BACKTESTING PARAMETERS
    # ============================================================
    TRAIN_WINDOW_SIZE: int = 252 * 3  # 3 years of trading days
    TEST_WINDOW_SIZE: int = 21  # 1 month ahead prediction
    WALK_FORWARD_STEP: int = 21  # Re-train every month
    
    # ============================================================
    # SENTIMENT ANALYSIS
    # ============================================================
    FINBERT_MODEL: str = "ProsusAI/finbert"
    SENTIMENT_BATCH_SIZE: int = 16
    MAX_NEWS_PER_DAY: int = 10  # Max headlines to aggregate per day
    
    # ============================================================
    # TECHNICAL INDICATORS
    # ============================================================
    TECHNICAL_INDICATORS: Dict[str, Any] = {
        "RSI": {"period": 14},
        "MACD": {"fast": 12, "slow": 26, "signal": 9},
        "BBANDS": {"period": 20, "std": 2},
        "ATR": {"period": 14},
        "ADX": {"period": 14}
    }
    
    # ============================================================
    # API CONFIGURATION
    # ============================================================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = False  # Set to False in production
    
    @classmethod
    def set_seed(cls) -> None:
        """
        Set global random seed for reproducibility across all libraries.
        
        This ensures deterministic behavior in:
        - PyTorch (CPU and CUDA operations)
        - NumPy random number generation
        - Python's built-in random module
        """
        torch.manual_seed(cls.SEED)
        torch.cuda.manual_seed_all(cls.SEED)
        np.random.seed(cls.SEED)
        random.seed(cls.SEED)
        
        # Ensure deterministic behavior in PyTorch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)


# Initialize on import
Config.set_seed()
Config.create_directories()
