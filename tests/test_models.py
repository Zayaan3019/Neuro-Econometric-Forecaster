"""
Test Suite for Neuro-Econometric Market Alpha Engine.

Validates data integrity, model shapes, and core functionality.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

from app.config import Config
from app.data.loader import OHLCVLoader, NewsLoader, DataLoader
from app.data.preprocessor import (
    StationarityTester,
    DifferencingTransformer,
    TechnicalIndicatorEngine,
    Preprocessor
)
from app.data.sentiment import FinBERTSentimentAnalyzer, SentimentFeatureEngineer
from app.models.econometrics import ARDLModel, CointegrationTester, EconometricPredictor
from app.models.neural import (
    PositionalEncoding,
    MultiHeadAttention,
    TransformerEncoder,
    LSTMEncoder,
    HybridNeuralEncoder
)
from app.models.fusion import (
    GatedFusionMechanism,
    NeuroEconometricNet
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)
    
    data = {
        'Open': np.random.randn(len(dates)).cumsum() + 100,
        'High': np.random.randn(len(dates)).cumsum() + 102,
        'Low': np.random.randn(len(dates)).cumsum() + 98,
        'Close': np.random.randn(len(dates)).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
    }
    
    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def sample_news_data():
    """Generate sample news data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    
    headlines = [
        "Market rallies on positive economic data",
        "Stocks decline amid trade concerns",
        "Tech sector shows strong growth",
    ]
    
    data = {
        'date': dates[:len(headlines)],
        'title': headlines,
        'description': headlines,
        'source': ['Reuters'] * len(headlines),
        'url': ['http://example.com'] * len(headlines)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_time_series():
    """Generate sample time series for econometric tests."""
    np.random.seed(42)
    return pd.Series(np.random.randn(252).cumsum() + 100)


# ============================================================
# DATA LOADER TESTS
# ============================================================

class TestOHLCVLoader:
    """Test OHLCV data loader."""
    
    def test_initialization(self):
        """Test loader initialization."""
        loader = OHLCVLoader(ticker="^GSPC")
        assert loader.ticker == "^GSPC"
        assert loader.session is not None
    
    def test_validate_and_clean(self, sample_ohlcv_data):
        """Test data validation and cleaning."""
        loader = OHLCVLoader()
        
        # Add some missing values
        sample_ohlcv_data.loc[sample_ohlcv_data.index[10], 'Close'] = np.nan
        
        cleaned = loader._validate_and_clean(sample_ohlcv_data)
        
        assert not cleaned.isna().any().any()
        assert (cleaned['High'] >= cleaned['Low']).all()


class TestNewsLoader:
    """Test news data loader."""
    
    def test_initialization(self):
        """Test loader initialization."""
        loader = NewsLoader(api_key="test_key")
        assert loader.api_key == "test_key"
        assert loader.base_url is not None
    
    def test_aggregate_by_day(self, sample_news_data):
        """Test news aggregation by day."""
        loader = NewsLoader()
        aggregated = loader.aggregate_by_day(sample_news_data)
        
        assert 'date' in aggregated.columns
        assert 'headlines' in aggregated.columns
        assert len(aggregated) <= len(sample_news_data)


# ============================================================
# PREPROCESSOR TESTS
# ============================================================

class TestStationarityTester:
    """Test stationarity testing."""
    
    def test_adf_test_stationary(self):
        """Test ADF test on stationary series."""
        tester = StationarityTester()
        
        # Generate stationary series (white noise)
        np.random.seed(42)
        stationary_series = pd.Series(np.random.randn(252))
        
        is_stationary, p_value, stats = tester.adf_test(stationary_series)
        
        assert isinstance(is_stationary, bool)
        assert 0 <= p_value <= 1
        assert 'adf_statistic' in stats
    
    def test_adf_test_nonstationary(self, sample_time_series):
        """Test ADF test on non-stationary series."""
        tester = StationarityTester()
        
        is_stationary, p_value, stats = tester.adf_test(sample_time_series)
        
        # Random walk should typically be non-stationary
        assert isinstance(is_stationary, bool)
        assert 'p_value' in stats


class TestDifferencingTransformer:
    """Test differencing transformer."""
    
    def test_fit_transform(self, sample_time_series):
        """Test differencing transformation."""
        transformer = DifferencingTransformer(max_order=2)
        
        differenced, order = transformer.fit_transform(sample_time_series)
        
        assert isinstance(order, int)
        assert 0 <= order <= 2
        assert len(differenced) <= len(sample_time_series)


class TestTechnicalIndicatorEngine:
    """Test technical indicator computation."""
    
    def test_compute_all(self, sample_ohlcv_data):
        """Test computation of all technical indicators."""
        df_with_indicators = TechnicalIndicatorEngine.compute_all(sample_ohlcv_data)
        
        # Check that indicators were added
        expected_indicators = ['RSI', 'MACD', 'ATR', 'ADX']
        for indicator in expected_indicators:
            assert indicator in df_with_indicators.columns
        
        # Check no NaN in recent data (after warmup)
        recent = df_with_indicators.iloc[-100:]
        assert not recent[expected_indicators].isna().all().any()


# ============================================================
# SENTIMENT ANALYSIS TESTS
# ============================================================

class TestSentimentAnalyzer:
    """Test sentiment analysis."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_initialization(self):
        """Test FinBERT initialization."""
        analyzer = FinBERTSentimentAnalyzer()
        assert analyzer.model is not None
        assert analyzer.tokenizer is not None
    
    def test_sentiment_score_range(self):
        """Test that sentiment scores are in valid range."""
        analyzer = FinBERTSentimentAnalyzer()
        
        texts = [
            "The market is performing exceptionally well",
            "Significant losses reported in tech sector",
            "Trading remains flat with no clear direction"
        ]
        
        score = analyzer.compute_sentiment_score(texts)
        
        assert -1 <= score <= 1


# ============================================================
# ECONOMETRIC MODEL TESTS
# ============================================================

class TestARDLModel:
    """Test ARDL model."""
    
    def test_initialization(self):
        """Test ARDL initialization."""
        model = ARDLModel(lags=5)
        assert model.lags == 5
        assert not model.is_fitted
    
    def test_fit_predict(self, sample_time_series):
        """Test ARDL fitting and prediction."""
        model = ARDLModel(lags=5)
        
        # Fit model
        model.fit(endog=sample_time_series)
        
        assert model.is_fitted
        
        # Predict
        predictions = model.predict(steps=5)
        
        assert len(predictions) == 5
        assert not np.isnan(predictions).any()


class TestCointegrationTester:
    """Test cointegration testing."""
    
    def test_pairwise_cointegration(self):
        """Test pairwise cointegration test."""
        np.random.seed(42)
        
        # Generate cointegrated series
        x = pd.Series(np.random.randn(252).cumsum())
        y = 2 * x + np.random.randn(252) * 0.1  # y is closely related to x
        
        is_coint, p_value, stats = CointegrationTester.test_pairwise(y, x)
        
        assert isinstance(is_coint, bool)
        assert 0 <= p_value <= 1
        assert 'test_statistic' in stats


# ============================================================
# NEURAL NETWORK TESTS
# ============================================================

class TestPositionalEncoding:
    """Test positional encoding."""
    
    def test_forward_pass(self):
        """Test positional encoding forward pass."""
        d_model = 128
        seq_len = 60
        batch_size = 32
        
        pos_encoder = PositionalEncoding(d_model=d_model)
        x = torch.randn(seq_len, batch_size, d_model)
        
        output = pos_encoder(x)
        
        assert output.shape == (seq_len, batch_size, d_model)
        assert not torch.isnan(output).any()


class TestMultiHeadAttention:
    """Test multi-head attention."""
    
    def test_forward_pass(self):
        """Test attention forward pass."""
        d_model = 128
        nhead = 8
        seq_len = 60
        batch_size = 32
        
        attention = MultiHeadAttention(d_model=d_model, nhead=nhead)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, weights = attention(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, nhead, seq_len, seq_len)
        assert not torch.isnan(output).any()


class TestHybridNeuralEncoder:
    """Test hybrid neural encoder."""
    
    def test_initialization(self):
        """Test encoder initialization."""
        encoder = HybridNeuralEncoder(input_dim=50, hidden_dim=128)
        
        assert encoder.input_dim == 50
        assert encoder.hidden_dim == 128
    
    def test_forward_pass(self):
        """Test encoder forward pass."""
        batch_size = 32
        seq_len = 60
        input_dim = 50
        
        encoder = HybridNeuralEncoder(input_dim=input_dim, hidden_dim=128)
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output = encoder(x)
        
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()
    
    def test_extract_representation(self):
        """Test representation extraction."""
        batch_size = 32
        seq_len = 60
        input_dim = 50
        hidden_dim = 128
        
        encoder = HybridNeuralEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        x = torch.randn(batch_size, seq_len, input_dim)
        
        representation = encoder.extract_representation(x)
        
        assert representation.shape == (batch_size, hidden_dim)
        assert not torch.isnan(representation).any()


# ============================================================
# FUSION NETWORK TESTS
# ============================================================

class TestGatedFusionMechanism:
    """Test gated fusion mechanism."""
    
    def test_forward_pass(self):
        """Test fusion gate forward pass."""
        batch_size = 32
        state_dim = 128
        
        fusion = GatedFusionMechanism(state_dim=state_dim)
        
        market_state = torch.randn(batch_size, state_dim)
        volatility = torch.randn(batch_size, 1)
        sentiment = torch.randn(batch_size, 1)
        
        alpha = fusion(market_state, volatility, sentiment)
        
        assert alpha.shape == (batch_size, 1)
        assert (alpha >= 0).all() and (alpha <= 1).all()


class TestNeuroEconometricNet:
    """Test complete Neuro-Econometric Network."""
    
    def test_initialization(self):
        """Test network initialization."""
        model = NeuroEconometricNet(input_dim=50, hidden_dim=128)
        
        assert model.input_dim == 50
        assert model.hidden_dim == 128
    
    def test_forward_pass(self):
        """Test network forward pass."""
        batch_size = 32
        seq_len = 60
        input_dim = 50
        
        model = NeuroEconometricNet(input_dim=input_dim, hidden_dim=128)
        
        x = torch.randn(batch_size, seq_len, input_dim)
        ardl_preds = torch.randn(batch_size, 1)
        volatility_features = torch.randn(batch_size, 5)
        sentiment = torch.randn(batch_size, 1)
        
        fused_pred, alpha, neural_pred = model(x, ardl_preds, volatility_features, sentiment)
        
        assert fused_pred.shape == (batch_size, 1)
        assert alpha.shape == (batch_size, 1)
        assert neural_pred.shape == (batch_size, 1)
        
        assert not torch.isnan(fused_pred).any()
        assert (alpha >= 0).all() and (alpha <= 1).all()


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestIntegration:
    """Integration tests for complete pipeline."""
    
    def test_data_to_model_pipeline(self, sample_ohlcv_data):
        """Test complete pipeline from data to model."""
        # Preprocess
        preprocessor = Preprocessor()
        processed_df, metadata = preprocessor.fit_transform(sample_ohlcv_data)
        
        assert not processed_df.isna().any().any()
        
        # Prepare model input
        seq_len = 60
        input_dim = len(metadata['feature_names'])
        
        # Create dummy batch
        x = torch.randn(1, seq_len, input_dim)
        ardl_preds = torch.randn(1, 1)
        volatility_features = torch.randn(1, 5)
        sentiment = torch.randn(1, 1)
        
        # Forward pass through model
        model = NeuroEconometricNet(input_dim=input_dim, hidden_dim=64)
        fused_pred, alpha, neural_pred = model(x, ardl_preds, volatility_features, sentiment)
        
        assert fused_pred.shape == (1, 1)
        assert not torch.isnan(fused_pred).any()


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
