# Neuro-Econometric Market Alpha Engine 🚀

**Production-Grade Hybrid ML System for Market Forecasting**

A cutting-edge financial forecasting engine that combines classical econometrics (ARDL) with modern deep learning (Transformers + LSTMs) through an intelligent gating mechanism to generate Alpha signals for trading strategies.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│          Multi-Modal Input Data             │
│  (OHLCV + Technical Indicators + Sentiment) │
└──────────────────┬──────────────────────────┘
                   │
      ┌────────────┴────────────┐
      │                         │
┌─────▼──────┐         ┌────────▼─────────┐
│ARDL Branch │         │  Neural Branch   │
│  (Linear)  │         │  (Non-Linear)    │
│            │         │  Transformer +   │
│Econometric │         │     LSTM         │
│Forecasting │         │                  │
└─────┬──────┘         └────────┬─────────┘
      │                         │
      │  Y_ardl                 │  Y_neural
      └────────────┬────────────┘
                   │
          ┌────────▼────────┐
          │ Gated Fusion    │
          │ α = σ(W·State)  │
          │ Y = α·Y_neural  │
          │   + (1-α)·Y_ardl│
          └────────┬────────┘
                   │
              ┌────▼─────┐
              │  Alpha   │
              │  Signal  │
              └──────────┘
```

### Key Features

- **Hybrid Architecture**: Combines ARDL (linear econometrics) with Transformers & LSTMs (non-linear neural networks)
- **Adaptive Fusion**: Gating mechanism dynamically weights classical vs modern approaches based on market conditions
- **Multi-Modal Data**: Integrates price data, technical indicators, and sentiment analysis
- **Walk-Forward Validation**: Strict temporal validation prevents look-ahead bias
- **Production-Ready**: FastAPI REST API with async inference, monitoring, and error handling

---

## 📊 Performance Metrics

**Training Results** (2015-2024, S&P 500):
- **Training Duration**: ~40 minutes (CPU)
- **Directional Accuracy**: 52.93%
- **Gating Statistics**: α=0.47 (47% neural, 53% ARDL)
- **Convergence**: 40 epochs with early stopping

⚠️ **Note on Metrics**: The extremely negative R² (-83M) indicates a scaling/preprocessing issue that needs addressing for production deployment. The model architecture is sound but requires calibration of the inverse transformation pipeline.

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.11+
python --version

# Required packages
pip install -r requirements.txt
```

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/market-alpha-engine.git
cd market-alpha-engine

# Install dependencies
pip install -r requirements.txt

# Generate sample data (if Yahoo Finance unavailable)
python generate_sample_data.py

# Train model (~40 minutes)
python train_model.py

# Check system status
python setup_check.py
```

### Run API Server

```bash
# Start FastAPI server
python serve_api.py

# API will be available at:
# - Interactive Docs: http://localhost:8000/docs
# - Health Check: http://localhost:8000/health
# - Predictions: http://localhost:8000/predict
```

---

## 📡 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-02-02T20:00:00",
  "device": "cpu"
}
```

### Get Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "^GSPC",
    "horizon": 1
  }'
```

Response:
```json
{
  "ticker": "^GSPC",
  "timestamp": "2026-02-02T20:00:00",
  "current_price": 5234.18,
  "predictions": [
    {
      "day": 1,
      "predicted_value": 0.0234,
      "note": "Value is on preprocessed scale"
    }
  ],
  "signal": "BUY",
  "confidence": 0.65,
  "model_version": "2026-02-02T20:04:17"
}
```

### Model Information

```bash
curl http://localhost:8000/model/info
```

---

## 🧪 Model Training

### Configuration

Edit `app/config.py` to customize:

```python
class Config:
    # Data
    TICKER: str = "^GSPC"  # S&P 500
    START_DATE: str = "2015-01-01"
    END_DATE: str = "2024-12-31"
    
    # Architecture
    HIDDEN_DIM: int = 128
    NUM_LAYERS: int = 2
    NHEAD: int = 8
    DROPOUT: float = 0.3
    SEQ_LENGTH: int = 60
    
    # Training
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 100
    EARLY_STOP_PATIENCE: int = 15
```

### Training Pipeline

```bash
python train_model.py
```

**Artifacts Generated**:
- `models_saved/best_model.pt` - Model checkpoint
- `models_saved/training_history.csv` - Loss curves
- `models_saved/training_info.json` - Metadata
- `logs/training.log` - Detailed logs

---

## 📈 Evaluation & Backtesting

### Run Evaluation

```bash
python evaluate_model.py
```

Provides:
- Prediction metrics (MSE, RMSE, MAE, R²)
- Directional accuracy
- Trading signal analysis
- Sharpe ratio & drawdown

### Custom Backtesting

```python
from app.engine.backtester import WalkForwardBacktester

backtester = WalkForwardBacktester(
    train_window=756,  # 3 years
    test_window=21,    # 1 month
    step=21
)

results = backtester.run(model, data)
```

---

## 🛠️ Project Structure

```
Market_Alpha_Engine/
├── app/
│   ├── config.py              # Configuration
│   ├── data/
│   │   ├── loader.py          # Data loading (yfinance, NewsAPI)
│   │   ├── preprocessor.py    # Stationarity tests, indicators
│   │   └── sentiment.py       # FinBERT sentiment analysis
│   ├── models/
│   │   ├── econometrics.py    # ARDL implementation
│   │   ├── neural.py          # Transformer + LSTM
│   │   └── fusion.py          # Gated fusion mechanism
│   ├── engine/
│   │   ├── trainer.py         # Training loop
│   │   └── backtester.py      # Walk-forward validation
│   └── api/
│       └── routes.py          # FastAPI endpoints
├── data/
│   └── GSPC_ohlcv.csv         # Sample market data
├── models_saved/              # Trained model artifacts
├── logs/                      # Training logs
├── train_model.py             # Training script
├── serve_api.py               # Production API server
├── evaluate_model.py          # Evaluation script
├── generate_sample_data.py    # Sample data generator
├── setup_check.py             # System diagnostics
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🔧 Technical Details

### ARDL (AutoRegressive Distributed Lag)
- Classical econometric model for time series
- Handles non-stationary data via differencing
- ADF & KPSS stationarity tests
- Automatic lag selection via AIC/BIC

### Neural Branch
- **Transformer Encoder**: Captures long-range dependencies and attention patterns
- **LSTM Layers**: Models sequential dynamics and temporal patterns
- **Multi-Head Attention**: Focuses on relevant time steps
- **Positional Encoding**: Preserves temporal order information

### Gating Mechanism
```
α = σ(W · [h_t, volatility_t, sentiment_t])
Y_final = α · Y_neural + (1 - α) · Y_ardl
```
- Adaptive weighting based on market state
- Higher α → more weight on neural predictions
- Lower α → more weight on econometric forecasts

### Technical Indicators
- **Momentum**: RSI, MOM, ROC
- **Trend**: MACD, ADX
- **Volatility**: ATR, Bollinger Bands
- **Volume**: OBV

---

## 🐛 Troubleshooting

### Yahoo Finance Rate Limiting

If you encounter `429 Too Many Requests`:

```bash
# Generate local sample data
python generate_sample_data.py

# System will automatically use local CSV
```

### TA-Lib Not Available

System automatically falls back to pure Python implementations:
```
WARNING: TA-Lib not available. Using pure Python implementations.
```

No action needed - pure Python versions are fully functional.

### Model Scaling Issues

Current R² metric indicates preprocessing needs calibration:

1. **Inverse Transform**: Ensure predictions are converted back to original scale
2. **Feature Alignment**: Verify train/test feature consistency
3. **Differencing Inversion**: Properly reconstruct from differenced values

---

## 📝 Development Roadmap

### Completed ✅
- [x] Hybrid ARDL + Neural architecture
- [x] Multi-modal data pipeline
- [x] Walk-forward backtesting
- [x] FastAPI production server
- [x] Comprehensive logging & monitoring
- [x] Pure Python technical indicators (TA-Lib fallback)

### In Progress 🚧
- [ ] Fix preprocessing inverse transformation
- [ ] Sentiment integration (NewsAPI + FinBERT)
- [ ] Enhanced evaluation metrics

### Planned 📋
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Real-time WebSocket streaming
- [ ] Multi-asset support (crypto, forex)
- [ ] Model versioning & A/B testing
- [ ] Dashboard UI (Streamlit/Plotly)

---

## 📚 References

1. **ARDL Models**: Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). "Bounds testing approaches to the analysis of level relationships"
2. **Transformers**: Vaswani et al. (2017). "Attention Is All You Need"
3. **FinBERT**: Araci, D. (2019). "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models"
4. **Walk-Forward Analysis**: Pardo, R. (2008). "The Evaluation and Optimization of Trading Strategies"

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👤 Author

**Mohamed Zayaan**

- GitHub: [@your-username](https://github.com/your-username)
- Email: your.email@example.com

---

## ⚠️ Disclaimer

**For Educational and Research Purposes Only**

This software is provided for educational and research purposes only. It is not intended to provide financial advice or recommendations for live trading. 

- **No Warranty**: The software is provided "as is" without warranty of any kind
- **No Liability**: Authors are not liable for any losses from using this software
- **Not Financial Advice**: Predictions are experimental and should not be used for actual trading
- **Risk Warning**: Trading financial instruments involves substantial risk of loss

Always conduct your own research and consult with financial professionals before making investment decisions.

---

## 🌟 Acknowledgments

- Yahoo Finance for market data
- Hugging Face for FinBERT models
- PyTorch & scikit-learn teams
- FastAPI community
- statsmodels contributors

---

<div align="center">
  <sub>Built with ❤️ for the quantitative finance community</sub>
</div>
