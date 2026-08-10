"""
Configuration Module for Neuro-Econometric Market Alpha Engine.

This module defines all hyperparameters, paths, and global settings required
for reproducible experimentation and production deployment.
"""

from pathlib import Path
from typing import Dict, Any, List
import os
import torch
import numpy as np
import random


class Config:
    """
    Global configuration class with strict type hints.
    """

    # ============================================================
    # REPRODUCIBILITY
    # ============================================================
    SEED: int = 42

    # ============================================================
    # COMPUTE RESOURCES
    # ============================================================
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS: int = 0  # 0 for Windows compatibility

    # ============================================================
    # PATHS
    # ============================================================
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    # DATA_DIR/MODEL_DIR stay under PROJECT_ROOT -- they're read sources
    # (Dockerfile.lambda COPYs real content into them at build time), not
    # runtime write targets, so Lambda's read-only /var/task is fine for them.
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODEL_DIR: Path = PROJECT_ROOT / "models_saved"
    # LOGS_DIR IS a runtime write target (create_directories() below actually
    # creates it, run_pipeline.py/walk_forward_pipeline.py open FileHandlers
    # under it) -- PROJECT_ROOT/logs fails on Lambda specifically, verified
    # empirically via a real deploy: "OSError: [Errno 30] Read-only file
    # system: '/var/task/logs'". AWS_LAMBDA_FUNCTION_NAME is set automatically
    # in every Lambda execution environment and never set elsewhere, so this
    # doesn't change local/EC2/Docker-compose behavior at all -- /tmp is the
    # one writable path Lambda guarantees.
    LOGS_DIR: Path = (
        Path("/tmp/logs") if os.environ.get("AWS_LAMBDA_FUNCTION_NAME") else PROJECT_ROOT / "logs"
    )

    # ============================================================
    # DATA CONFIGURATION
    # ============================================================
    TICKER: str = "^GSPC"
    START_DATE: str = "2010-01-01"   # Extended to 14 years for richer training
    # Extended 2024-12-31 -> 2026-08-07 via refresh_real_data.py (2026-08-10):
    # MODEL_EVALUATION_REPORT.md's data ended nearly 19 months before that
    # report was actually written -- this closes the gap with 400 more real
    # trading days rather than leaving the evaluation stale.
    END_DATE: str = "2026-08-07"
    NEWS_API_KEY: str = "YOUR_NEWS_API_KEY_HERE"

    # Which component serve_api.py's /predict actually uses for
    # predicted_price/signal -- "hybrid" or "arima". Set from
    # MODEL_EVALUATION_REPORT.md's real walk-forward result, not defaulted to
    # the fancier model: that report found the hybrid model's directional
    # accuracy was not statistically distinguishable from 50% and
    # significantly worse (Diebold-Mariano) than the plain ARIMA baseline, so
    # "arima" is the honest default here pending the extended-data re-run
    # (see refresh_real_data.py) confirming whether this still holds.
    PRIMARY_SIGNAL_MODEL: str = "arima"

    # ============================================================
    # MACRO DATA (free via yfinance)
    # ============================================================
    MACRO_TICKERS: Dict[str, str] = {
        "VIX":  "^VIX",        # CBOE Volatility Index
        "TNX":  "^TNX",        # 10-Year Treasury Yield
        "FVX":  "^FVX",        # 5-Year Treasury Yield
        "IRX":  "^IRX",        # 13-Week T-Bill
        "DXY":  "DX-Y.NYB",   # US Dollar Index
        "GLD":  "GLD",         # Gold ETF (risk-off proxy)
        "TLT":  "TLT",         # 20+ Year Treasury ETF
    }

    # ============================================================
    # ECONOMETRIC PARAMETERS (ARDL)
    # ============================================================
    ARDL_LAGS: int = 5
    ARDL_WINDOW: int = 120          # Rolling window for ARDL fitting (trading days)
    ARDL_MAX_DIFF_ORDER: int = 2
    ADF_PVALUE_THRESHOLD: float = 0.05

    # ============================================================
    # NEURAL NETWORK ARCHITECTURE
    # ============================================================
    HIDDEN_DIM: int = 128
    NUM_LAYERS: int = 2
    NHEAD: int = 8
    DROPOUT: float = 0.2            # Reduced from 0.3 for better gradient flow
    SEQ_LENGTH: int = 60
    TRANSFORMER_LAYERS: int = 2     # Stack two transformer blocks

    # ============================================================
    # FUSION NETWORK PARAMETERS
    # ============================================================
    FUSION_HIDDEN_DIM: int = 64
    ALPHA_INIT: float = 0.5

    # ============================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-4     # Stable LR for complex hybrid model
    NUM_EPOCHS: int = 80
    EARLY_STOP_PATIENCE: int = 15
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP_NORM: float = 0.5     # Tighter clip for stability

    # Loss weights
    LAMBDA_MSE: float = 1.0
    LAMBDA_DIRECTIONAL: float = 2.5  # Strong directional signal
    LAMBDA_AGREEMENT: float = 0.15
    LAMBDA_REG: float = 0.05

    # ============================================================
    # BACKTESTING PARAMETERS
    # ============================================================
    TRAIN_WINDOW_SIZE: int = 252 * 3
    TEST_WINDOW_SIZE: int = 21
    WALK_FORWARD_STEP: int = 21
    TRANSACTION_COST_BPS: float = 10.0  # 10 bps one-way (institutional median)

    # ============================================================
    # SENTIMENT ANALYSIS
    # ============================================================
    FINBERT_MODEL: str = "ProsusAI/finbert"
    SENTIMENT_BATCH_SIZE: int = 16
    MAX_NEWS_PER_DAY: int = 10

    # ============================================================
    # TECHNICAL INDICATORS
    # ============================================================
    TECHNICAL_INDICATORS: Dict[str, Any] = {
        "RSI":    {"period": 14},
        "MACD":   {"fast": 12, "slow": 26, "signal": 9},
        "BBANDS": {"period": 20, "std": 2},
        "ATR":    {"period": 14},
        "ADX":    {"period": 14},
    }

    # ============================================================
    # API CONFIGURATION
    # ============================================================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = False
    
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
