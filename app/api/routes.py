"""
FastAPI Routes for Real-Time Inference.

This module provides async REST API endpoints for serving predictions
from the trained Neuro-Econometric Market Alpha Engine.
"""

from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from app.config import Config
from app.models.fusion import NeuroEconometricNet
from app.data.loader import DataLoader
from app.data.preprocessor import Preprocessor
from app.data.sentiment import FinBERTSentimentAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Neuro-Econometric Market Alpha Engine API",
    description="Real-time market prediction and alpha signal generation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class PredictionRequest(BaseModel):
    """Request model for market prediction."""
    ticker: str = Field(..., example="^GSPC", description="Stock/Index ticker symbol")
    lookback_days: int = Field(60, ge=1, le=365, description="Number of historical days to use")
    include_news: bool = Field(True, description="Whether to include sentiment analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "ticker": "^GSPC",
                "lookback_days": 60,
                "include_news": True
            }
        }


class AlphaSignal(BaseModel):
    """Alpha signal model."""
    signal: str = Field(..., description="Trading signal: BUY, SELL, or HOLD")
    confidence: float = Field(..., ge=0, le=1, description="Signal confidence (0-1)")
    predicted_return: float = Field(..., description="Expected return (%)")
    gating_weight: float = Field(..., ge=0, le=1, description="Neural vs ARDL weight")
    volatility_regime: str = Field(..., description="Current volatility regime")


class PredictionResponse(BaseModel):
    """Response model for market prediction."""
    timestamp: datetime
    ticker: str
    current_price: float
    predicted_price: float
    alpha_signal: AlphaSignal
    technical_indicators: Dict[str, float]
    sentiment_score: Optional[float]
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    version: str


# ============================================================
# GLOBAL STATE
# ============================================================

class ModelState:
    """Global model state container."""
    model: Optional[NeuroEconometricNet] = None
    preprocessor: Optional[Preprocessor] = None
    sentiment_analyzer: Optional[FinBERTSentimentAnalyzer] = None
    data_loader: Optional[DataLoader] = None
    is_loaded: bool = False


state = ModelState()


# ============================================================
# STARTUP/SHUTDOWN
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Load model and initialize services on startup."""
    logger.info("Starting API server...")
    
    try:
        # Initialize data loader
        state.data_loader = DataLoader()
        
        # Initialize preprocessor
        state.preprocessor = Preprocessor()
        
        # Initialize sentiment analyzer
        state.sentiment_analyzer = FinBERTSentimentAnalyzer()
        
        # Load trained model
        model_path = Config.MODEL_DIR / "best_model.pt"
        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            
            # Initialize model architecture
            state.model = NeuroEconometricNet(
                input_dim=50,  # Update based on your feature dimensions
                hidden_dim=Config.HIDDEN_DIM
            )
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=Config.DEVICE)
            state.model.load_state_dict(checkpoint['model_state_dict'])
            state.model.to(Config.DEVICE)
            state.model.eval()
            
            state.is_loaded = True
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model file not found at {model_path}. API will run in demo mode.")
            state.is_loaded = False
    
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        state.is_loaded = False


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API server...")


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Neuro-Econometric Market Alpha Engine API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if state.is_loaded else "model_not_loaded",
        model_loaded=state.is_loaded,
        device=str(Config.DEVICE),
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate market prediction and alpha signal.
    
    Args:
        request: Prediction request with ticker and parameters.
    
    Returns:
        PredictionResponse with alpha signal and metadata.
    
    Raises:
        HTTPException: If prediction fails.
    """
    if not state.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model checkpoint exists."
        )
    
    try:
        logger.info(f"Prediction request for {request.ticker}")
        
        # Fetch data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - pd.Timedelta(days=request.lookback_days + 100)).strftime("%Y-%m-%d")
        
        ohlcv_df, news_df = state.data_loader.load_all(
            start_date=start_date,
            end_date=end_date,
            include_news=request.include_news
        )
        
        if len(ohlcv_df) < request.lookback_days:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data. Required: {request.lookback_days}, Available: {len(ohlcv_df)}"
            )
        
        # Preprocess data
        processed_df, metadata = state.preprocessor.fit_transform(ohlcv_df)
        
        # Get current price
        current_price = float(ohlcv_df['Close'].iloc[-1])
        
        # Extract features for prediction
        recent_data = processed_df.iloc[-request.lookback_days:]
        
        # Compute sentiment
        sentiment_score = 0.0
        if request.include_news and not news_df.empty:
            recent_news = news_df.tail(10)
            if 'headlines' in recent_news.columns:
                sentiment_score = state.sentiment_analyzer.compute_sentiment_score(
                    recent_news['headlines'].tolist()
                )
        
        # Prepare model inputs (simplified for demonstration)
        # In production, this should match your exact training data format
        features = torch.tensor(
            recent_data.values,
            dtype=torch.float32,
            device=Config.DEVICE
        ).unsqueeze(0)  # Add batch dimension
        
        # Dummy inputs for ARDL, volatility, sentiment
        # In production, compute these properly
        ardl_pred = torch.tensor([[current_price]], device=Config.DEVICE)
        volatility_features = torch.randn(1, 5, device=Config.DEVICE)
        sentiment_tensor = torch.tensor([[sentiment_score]], device=Config.DEVICE)
        
        # Model inference
        with torch.no_grad():
            prediction, alpha, neural_pred = state.model(
                features,
                ardl_pred,
                volatility_features,
                sentiment_tensor
            )
        
        predicted_price = float(prediction.cpu().item())
        gating_weight = float(alpha.cpu().item())
        
        # Compute expected return
        expected_return = ((predicted_price - current_price) / current_price) * 100
        
        # Generate alpha signal
        if expected_return > 2.0:
            signal = "BUY"
            confidence = min(abs(expected_return) / 10, 1.0)
        elif expected_return < -2.0:
            signal = "SELL"
            confidence = min(abs(expected_return) / 10, 1.0)
        else:
            signal = "HOLD"
            confidence = 1.0 - min(abs(expected_return) / 2, 1.0)
        
        # Determine volatility regime
        volatility_prob = float(volatility_features.mean().cpu().item())
        if volatility_prob > 0.7:
            volatility_regime = "HIGH"
        elif volatility_prob < 0.3:
            volatility_regime = "LOW"
        else:
            volatility_regime = "MODERATE"
        
        # Extract technical indicators
        technical_indicators = {
            "RSI": float(ohlcv_df['RSI'].iloc[-1]) if 'RSI' in ohlcv_df.columns else 50.0,
            "MACD": float(ohlcv_df['MACD'].iloc[-1]) if 'MACD' in ohlcv_df.columns else 0.0,
            "ATR": float(ohlcv_df['ATR'].iloc[-1]) if 'ATR' in ohlcv_df.columns else 0.0,
            "ADX": float(ohlcv_df['ADX'].iloc[-1]) if 'ADX' in ohlcv_df.columns else 25.0
        }
        
        # Build response
        response = PredictionResponse(
            timestamp=datetime.now(),
            ticker=request.ticker,
            current_price=current_price,
            predicted_price=predicted_price,
            alpha_signal=AlphaSignal(
                signal=signal,
                confidence=confidence,
                predicted_return=expected_return,
                gating_weight=gating_weight,
                volatility_regime=volatility_regime
            ),
            technical_indicators=technical_indicators,
            sentiment_score=sentiment_score if request.include_news else None,
            metadata={
                "lookback_days": request.lookback_days,
                "data_points": len(recent_data),
                "model_version": "1.0.0"
            }
        )
        
        logger.info(f"Prediction completed: {signal} with {confidence:.2f} confidence")
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/supported_tickers", response_model=List[str])
async def get_supported_tickers():
    """Get list of supported ticker symbols."""
    return [
        "^GSPC",  # S&P 500
        "^DJI",   # Dow Jones
        "^IXIC",  # NASDAQ
        "BTC-USD",  # Bitcoin
        "ETH-USD",  # Ethereum
        "GC=F",   # Gold Futures
        "CL=F",   # Crude Oil
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "GOOGL"   # Google
    ]


@app.get("/model_info", response_model=Dict[str, Any])
async def get_model_info():
    """Get information about the loaded model."""
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "architecture": "Neuro-Econometric Hybrid (ARDL + Transformer + LSTM)",
        "input_features": ["OHLCV", "Technical Indicators", "Sentiment Scores"],
        "output": "Price Prediction + Alpha Signal",
        "training_data": {
            "start_date": Config.START_DATE,
            "end_date": Config.END_DATE,
            "ticker": Config.TICKER
        },
        "hyperparameters": {
            "hidden_dim": Config.HIDDEN_DIM,
            "num_layers": Config.NUM_LAYERS,
            "dropout": Config.DROPOUT,
            "ardl_lags": Config.ARDL_LAGS
        },
        "device": str(Config.DEVICE),
        "version": "1.0.0"
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "routes:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.API_RELOAD,
        log_level="info"
    )
