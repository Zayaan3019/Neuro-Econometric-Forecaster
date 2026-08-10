"""
Inference API for the Neuro-Econometric Market Alpha Engine.

FastAPI server with a real-time prediction endpoint, health check, and
model info endpoint. Loads the CURRENT checkpoint format
(models_saved/best_model_v2.pt, produced by run_pipeline.py /
walk_forward_pipeline.py) -- a prior version of this file loaded an
architecturally incompatible legacy checkpoint (best_model.pt /
'model_state_dict' / 'neural_branch.input_projection...') and would raise
a RuntimeError on every startup once the model architecture was refactored.
It also called `model(...)` as if it returned a single tensor, when
NeuroEconometricNet.forward() returns a (prediction, alpha, neural_pred)
tuple -- that bug has been fixed too.

IMPORTANT HONESTY NOTE: the walk-forward evaluation in this repository
(see MODEL_EVALUATION_REPORT.md / models_saved/walk_forward_report.json)
found NO statistically significant directional edge for this model on
real S&P 500 data. This API is provided as a working reference
implementation of a real-time inference service, not as a claim that its
predictions are profitable trading signals.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler

from app.config import Config
from run_pipeline import load_ohlcv, load_macro, engineer_features, precompute_ardl_predictions
from app.models.fusion import NeuroEconometricNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Neuro-Econometric Market Alpha Engine",
    description="Reference real-time inference service for the hybrid ARDL/Transformer/LSTM forecaster.",
    version="2.0.0",
)

CHECKPOINT_PATH = Path("models_saved/best_model_v2.pt")

model: Optional[NeuroEconometricNet] = None
scaler: Optional[StandardScaler] = None
feature_cols: Optional[List[str]] = None
checkpoint_meta: Optional[Dict[str, Any]] = None


class PredictionRequest(BaseModel):
    ticker: str = Field(default="^GSPC", description="Stock/Index ticker symbol")
    horizon: int = Field(default=1, ge=1, le=1, description="Prediction horizon in days (only 1-day-ahead is supported)")

    class Config:
        json_schema_extra = {"example": {"ticker": "^GSPC", "horizon": 1}}


class PredictionResponse(BaseModel):
    ticker: str
    timestamp: str
    current_price: float
    predicted_return_pct: float
    predicted_price: float
    alpha_gate: float
    signal: str
    model_version: str
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    device: str


def load_model() -> bool:
    """Load the current checkpoint (best_model_v2.pt) at startup."""
    global model, scaler, feature_cols, checkpoint_meta

    if not CHECKPOINT_PATH.exists():
        logger.warning(f"Checkpoint not found at {CHECKPOINT_PATH}. Run run_pipeline.py or "
                        f"walk_forward_pipeline.py first. API will run in demo mode.")
        return False

    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=Config.DEVICE, weights_only=False)

        model = NeuroEconometricNet(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt["config"]["hidden_dim"],
            nhead=ckpt["config"]["nhead"],
            num_lstm_layers=ckpt["config"]["num_layers"],
            dropout=ckpt["config"]["dropout"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(Config.DEVICE)
        model.eval()

        scaler = StandardScaler()
        scaler.mean_ = np.array(ckpt["scaler_mean"])
        scaler.scale_ = np.array(ckpt["scaler_scale"])
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(ckpt["feature_cols"])

        feature_cols = ckpt["feature_cols"]
        checkpoint_meta = {
            "epoch": ckpt.get("epoch"),
            "val_loss": ckpt.get("val_loss"),
            "val_dir_acc": ckpt.get("val_dir_acc"),
            "alpha_std": ckpt.get("alpha_std"),
            "n_features": len(feature_cols),
        }

        logger.info(f"Model loaded from {CHECKPOINT_PATH}: {checkpoint_meta}")
        return True

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Neuro-Econometric API...")
    if load_model():
        logger.info("API ready for inference.")
    else:
        logger.warning("API started but model not loaded (endpoints will return 503).")


@app.get("/", response_class=JSONResponse)
async def root():
    return {
        "name": "Neuro-Econometric Market Alpha Engine",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {"predict": "/predict", "health": "/health", "model_info": "/model/info"},
        "documentation": "/docs",
        "note": "See MODEL_EVALUATION_REPORT.md for real, walk-forward-validated performance numbers.",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
        device=str(Config.DEVICE),
    )


@app.get("/model/info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_architecture": "Neuro-Econometric Hybrid (ARDL/AutoReg + Transformer + LSTM, gated fusion)",
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_metadata": checkpoint_meta,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(Config.DEVICE),
        "walk_forward_validated": "see models_saved/walk_forward_report.json",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate a 1-day-ahead return prediction and gating weight for the
    requested ticker using the most recent available data.
    """
    if model is None or scaler is None or feature_cols is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first via run_pipeline.py.")

    try:
        logger.info(f"Prediction request for {request.ticker}")

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")

        ohlcv = load_ohlcv() if request.ticker == Config.TICKER else None
        if ohlcv is None:
            raise HTTPException(
                status_code=400,
                detail=f"Only ticker '{Config.TICKER}' is supported by this reference deployment "
                       f"(uses the locally cached OHLCV/macro data the model was trained on).",
            )

        macro = load_macro(Config.START_DATE, Config.END_DATE)
        full_df = engineer_features(ohlcv, macro)

        missing = [c for c in feature_cols if c not in full_df.columns]
        if missing:
            raise HTTPException(status_code=500, detail=f"Feature mismatch vs checkpoint: missing {missing}")

        if len(full_df) < Config.SEQ_LENGTH + 1:
            raise HTTPException(status_code=400, detail=f"Insufficient data: need at least {Config.SEQ_LENGTH + 1} rows")

        current_price = float(ohlcv["Close"].iloc[-1])

        ardl_preds = precompute_ardl_predictions(full_df["Close"], window=Config.ARDL_WINDOW, lags=Config.ARDL_LAGS)

        window_df = full_df.iloc[-Config.SEQ_LENGTH:].copy()
        window_df[feature_cols] = scaler.transform(window_df[feature_cols].values)

        feat = torch.tensor(window_df[feature_cols].values, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)

        last_ardl = ardl_preds.iloc[-1]
        ardl_ret = float((last_ardl - current_price) / current_price) if pd.notna(last_ardl) and current_price > 0 else 0.0
        ardl_tensor = torch.tensor([[ardl_ret]], dtype=torch.float32).to(Config.DEVICE)

        vol_names = ["ATR", "BB_Width", "ADX", "vol_21d", "intraday_range"]
        vol_vals = [float(np.clip(full_df[c].iloc[-1], 0, 1)) if c in full_df.columns else 0.5 for c in vol_names]
        vol_tensor = torch.tensor([vol_vals], dtype=torch.float32).to(Config.DEVICE)

        sentiment_tensor = torch.zeros(1, 1, dtype=torch.float32).to(Config.DEVICE)

        with torch.no_grad():
            pred, alpha, _ = model(feat, ardl_tensor, vol_tensor, sentiment_tensor)
            predicted_return = float(pred.item())
            alpha_value = float(alpha.item())

        predicted_price = current_price * (1 + predicted_return)

        if abs(predicted_return) < 0.002:
            signal = "HOLD"
        elif predicted_return > 0:
            signal = "BUY"
        else:
            signal = "SELL"

        return PredictionResponse(
            ticker=request.ticker,
            timestamp=datetime.now().isoformat(),
            current_price=current_price,
            predicted_return_pct=predicted_return * 100,
            predicted_price=predicted_price,
            alpha_gate=alpha_value,
            signal=signal,
            model_version=str(checkpoint_meta.get("epoch")) if checkpoint_meta else "unknown",
            disclaimer=(
                "This model's directional accuracy on real S&P 500 walk-forward "
                "evaluation was not statistically distinguishable from 50% "
                "(see MODEL_EVALUATION_REPORT.md). This output is for "
                "demonstration of the serving pipeline, not a trading signal."
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Lambda entrypoint (Dockerfile.lambda's CMD is "serve_api.handler"). Unused
# outside Lambda -- local/Docker-Compose runs still go through __main__ below.
# Mangum's default lifespan="auto" runs the @app.on_event("startup") model
# load once per cold start (once per execution environment), then reuses the
# warm environment for subsequent invocations, same as a long-lived process.
from mangum import Mangum  # noqa: E402

handler = Mangum(app)


if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 60)
    logger.info("NEURO-ECONOMETRIC MARKET ALPHA ENGINE - API SERVER")
    logger.info("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
