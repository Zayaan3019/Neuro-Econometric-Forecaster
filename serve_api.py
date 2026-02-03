"""
Production Inference API for Neuro-Econometric Market Alpha Engine.

FastAPI server with:
- Real-time prediction endpoint
- Model health monitoring  
- Proper error handling
- Request/response validation
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import torch
import json
import logging
from pathlib import Path

from app.config import Config
from app.data.loader import DataLoader
from app.data.preprocessor import Preprocessor, TechnicalIndicatorEngine
from app.models.fusion import NeuroEconometricNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Neuro-Econometric Market Alpha Engine",
    description="Production-grade hybrid ML system for market forecasting",
    version="1.0.0"
)

# Global state
model: Optional[NeuroEconometricNet] = None
model_metadata: Optional[Dict] = None
preprocessor: Optional[Preprocessor] = None


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    ticker: str = Field(default="^GSPC", description="Stock/Index ticker symbol")
    horizon: int = Field(default=1, ge=1, le=30, description="Prediction horizon in days")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "^GSPC",
                "horizon": 1
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    ticker: str
    timestamp: str
    current_price: float
    predictions: List[Dict[str, Any]]
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float
    model_version: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str
    device: str


def load_model():
    """Load trained model at startup."""
    global model, model_metadata, preprocessor
    
    try:
        model_path = Path("models_saved/best_model.pt")
        metadata_path = Path("models_saved/training_info.json")
        
        if not model_path.exists():
            logger.warning("Model not found. Please train first: python train_model.py")
            return False
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            model_metadata = json.load(f)
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=Config.DEVICE)
        
        # Get input dimension from checkpoint
        input_dim = checkpoint['model_state_dict']['neural_branch.input_projection.weight'].shape[1]
        
        # Initialize model
        model = NeuroEconometricNet(
            input_dim=input_dim,
            hidden_dim=model_metadata['config']['hidden_dim'],
            num_lstm_layers=model_metadata['config']['num_layers']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(Config.DEVICE)
        model.eval()
        
        # Initialize preprocessor
        preprocessor = Preprocessor()
        
        logger.info("✓ Model loaded successfully")
        logger.info(f"  Input dim: {input_dim}")
        logger.info(f"  Device: {Config.DEVICE}")
        logger.info(f"  Trained on: {model_metadata['timestamp']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    logger.info("Starting Neuro-Econometric API...")
    success = load_model()
    if success:
        logger.info("✓ API ready for inference")
    else:
        logger.warning("⚠ API started but model not loaded")


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Neuro-Econometric Market Alpha Engine",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "model_info": "/model/info"
        },
        "documentation": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
        device=str(Config.DEVICE)
    )


@app.get("/model/info")
async def model_info():
    """Get model information and training metrics."""
    if model is None or model_metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_architecture": "Neuro-Econometric Hybrid (ARDL + Transformer + LSTM)",
        "training_info": model_metadata,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(Config.DEVICE)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate market predictions and trading signals.
    
    This endpoint:
    1. Fetches recent market data
    2. Runs preprocessing pipeline
    3. Generates predictions using hybrid model
    4. Returns actionable trading signals
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        logger.info(f"Prediction request for {request.ticker}, horizon={request.horizon}")
        
        # 1. Fetch recent data
        data_loader = DataLoader(ticker=request.ticker, local_csv='data/GSPC_ohlcv.csv')
        
        # Get last 6 months of data for context
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
        
        ohlcv_df, _ = data_loader.load_all(start_date, end_date, include_news=False)
        
        if len(ohlcv_df) < Config.SEQ_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: need at least {Config.SEQ_LENGTH} days"
            )
        
        current_price = float(ohlcv_df['Close'].iloc[-1])
        
        # 2. Preprocess
        df_with_indicators = TechnicalIndicatorEngine.compute_all(ohlcv_df)
        processed_df, _ = preprocessor.fit_transform(df_with_indicators)
        
        # 3. Make prediction
        with torch.no_grad():
            # Get last sequence
            seq = processed_df.iloc[-Config.SEQ_LENGTH:].values
            X = torch.tensor(seq[:, :-1], dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
            
            # Dummy inputs for ARDL, volatility, sentiment
            ardl_pred = torch.tensor([processed_df.iloc[-1, -1]], dtype=torch.float32).to(Config.DEVICE)
            volatility = torch.tensor([0.02], dtype=torch.float32).to(Config.DEVICE)
            sentiment = torch.tensor([0.0], dtype=torch.float32).to(Config.DEVICE)
            
            # Forward pass
            prediction = model(X, ardl_pred, volatility, sentiment)
            pred_value = prediction.item()
        
        # 4. Generate trading signal
        # This is simplified - in production, use more sophisticated logic
        predicted_change = pred_value - processed_df.iloc[-1, -1]
        
        if abs(predicted_change) < 0.01:  # Threshold
            signal = "HOLD"
            confidence = 0.5
        elif predicted_change > 0:
            signal = "BUY"
            confidence = min(0.9, 0.5 + abs(predicted_change) * 10)
        else:
            signal = "SELL"
            confidence = min(0.9, 0.5 + abs(predicted_change) * 10)
        
        # 5. Format response
        predictions = [
            {
                "day": i + 1,
                "predicted_value": pred_value,  # Note: this is on scaled/differenced scale
                "note": "Value is on preprocessed scale - not raw price"
            }
            for i in range(request.horizon)
        ]
        
        return PredictionResponse(
            ticker=request.ticker,
            timestamp=datetime.now().isoformat(),
            current_price=current_price,
            predictions=predictions,
            signal=signal,
            confidence=confidence,
            model_version=model_metadata['timestamp'] if model_metadata else "unknown"
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest")
async def run_backtest(start_date: str, end_date: str):
    """Run historical backtest on date range."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # TODO: Implement backtesting logic
    return {
        "message": "Backtesting endpoint - implementation pending",
        "start_date": start_date,
        "end_date": end_date
    }


if __name__ == "__main__":
    import uvicorn
    import pandas as pd
    
    logger.info("="*60)
    logger.info("NEURO-ECONOMETRIC MARKET ALPHA ENGINE - API SERVER")
    logger.info("="*60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
