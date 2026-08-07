"""
Test Suite for Neuro-Econometric Market Alpha Engine.

Validates data integrity, model shapes, and core functionality against the
CURRENT model architecture (app/models/neural.py, app/models/fusion.py).

NOTE: this file previously imported classes (`MultiHeadAttention`,
`TransformerEncoder`, `LSTMEncoder`) that no longer exist -- they were
written against an earlier, since-refactored architecture, and the whole
module failed to even import (`ImportError` at collection time), meaning
none of these tests were actually running despite being reported as
"17/22 passed" in a prior evaluation report. This file has been rewritten
against the current classes so it actually exercises the running code.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from app.config import Config
from app.data.loader import OHLCVLoader, NewsLoader
from app.data.preprocessor import (
    StationarityTester,
    DifferencingTransformer,
    TechnicalIndicatorEngine,
    Preprocessor,
)
from app.models.econometrics import ARDLModel, CointegrationTester, ARDLBoundsTest, EconometricPredictor
from app.models.neural import (
    PositionalEncoding,
    TransformerBlock,
    HybridNeuralEncoder,
    VolatilityRegimeDetector,
)
from app.models.fusion import GatedFusionMechanism, NeuroEconometricNet


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)

    close = 100 + np.cumsum(np.random.randn(len(dates)))
    data = {
        'Open': close + np.random.randn(len(dates)) * 0.5,
        'High': close + np.abs(np.random.randn(len(dates))) + 1,
        'Low': close - np.abs(np.random.randn(len(dates))) - 1,
        'Close': close,
        'Volume': np.random.randint(1_000_000, 10_000_000, len(dates)),
    }
    df = pd.DataFrame(data, index=dates)
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    return df


@pytest.fixture
def sample_news_data():
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
        'url': ['http://example.com'] * len(headlines),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_time_series():
    """Non-stationary random-walk series for econometric tests."""
    np.random.seed(42)
    return pd.Series(np.random.randn(252).cumsum() + 100)


@pytest.fixture
def sample_stationary_series():
    np.random.seed(42)
    return pd.Series(np.random.randn(300))


# ============================================================
# DATA LOADER TESTS
# ============================================================

class TestOHLCVLoader:
    def test_initialization(self):
        loader = OHLCVLoader(ticker="^GSPC")
        assert loader.ticker == "^GSPC"
        assert loader.session is not None

    def test_validate_and_clean(self, sample_ohlcv_data):
        loader = OHLCVLoader()
        sample_ohlcv_data.loc[sample_ohlcv_data.index[10], 'Close'] = np.nan
        cleaned = loader._validate_and_clean(sample_ohlcv_data)
        assert not cleaned.isna().any().any()
        assert (cleaned['High'] >= cleaned['Low']).all()


class TestNewsLoader:
    def test_initialization(self):
        loader = NewsLoader(api_key="test_key")
        assert loader.api_key == "test_key"
        assert loader.base_url is not None

    def test_aggregate_by_day(self, sample_news_data):
        loader = NewsLoader()
        aggregated = loader.aggregate_by_day(sample_news_data)
        assert 'date' in aggregated.columns
        assert 'headlines' in aggregated.columns
        assert len(aggregated) <= len(sample_news_data)


# ============================================================
# PREPROCESSOR TESTS
# ============================================================

class TestStationarityTester:
    def test_adf_test_stationary(self, sample_stationary_series):
        tester = StationarityTester()
        is_stationary, p_value, stats = tester.adf_test(sample_stationary_series)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
        assert 'adf_statistic' in stats

    def test_adf_test_nonstationary(self, sample_time_series):
        tester = StationarityTester()
        is_stationary, p_value, stats = tester.adf_test(sample_time_series)
        assert isinstance(p_value, float)
        assert 'p_value' in stats


class TestDifferencingTransformer:
    def test_fit_transform(self, sample_time_series):
        transformer = DifferencingTransformer(max_order=2)
        differenced, order = transformer.fit_transform(sample_time_series)
        assert isinstance(order, int)
        assert 0 <= order <= 2
        assert len(differenced) <= len(sample_time_series)


class TestTechnicalIndicatorEngine:
    def test_compute_all(self, sample_ohlcv_data):
        df_with_indicators = TechnicalIndicatorEngine.compute_all(sample_ohlcv_data)
        expected_indicators = ['RSI', 'MACD', 'ATR', 'ADX']
        for indicator in expected_indicators:
            assert indicator in df_with_indicators.columns
        recent = df_with_indicators.iloc[-100:]
        assert not recent[expected_indicators].isna().all().any()


# ============================================================
# ECONOMETRIC MODEL TESTS
# ============================================================

class TestARDLModel:
    def test_initialization(self):
        model = ARDLModel(lags=5, select_order=False)
        assert model.lags == 5
        assert not model.is_fitted

    def test_fit_predict_fixed_lags(self, sample_time_series):
        model = ARDLModel(lags=5, select_order=False)
        model.fit(endog=sample_time_series)
        assert model.is_fitted
        predictions = model.predict(steps=5)
        assert len(predictions) == 5
        assert not np.isnan(predictions).any()

    def test_fit_selects_lag_order_by_ic(self, sample_time_series):
        """AIC/BIC-based selection should not just fall back to Config.ARDL_LAGS."""
        model = ARDLModel(select_order=True, max_lags=8, ic='bic')
        model.fit(endog=sample_time_series)
        assert model.is_fitted
        assert model.selected_lags_ is not None


class TestARDLBoundsTest:
    def test_bounds_test_runs_and_returns_real_stats(self):
        """
        Regression test: this previously returned a hardcoded placeholder
        (False, 0.0, {"message": "requires manual interpretation..."}) and
        never ran an actual test. Verify it now performs a genuine PSS
        bounds test with real critical values.
        """
        np.random.seed(0)
        n = 300
        x = pd.Series(np.random.randn(n).cumsum() + 100, name='x')
        y = (0.5 * x + np.random.randn(n).cumsum() * 0.3 + 50)
        y.name = 'y'

        decision, f_stat, info = ARDLBoundsTest.bounds_f_test(y, x, max_lags=4, case=3)

        assert isinstance(f_stat, float)
        assert f_stat >= 0
        assert "critical_values" in info
        assert "lower" in info["critical_values"] and "upper" in info["critical_values"]
        assert decision in (True, False, None)  # never a fabricated verdict outside {reject, fail-to-reject, inconclusive}
        assert "message" not in info or "manual interpretation" not in info.get("message", "")

    def test_integration_order_detection(self):
        """A random walk should be classified I(1); white noise should be I(0)."""
        np.random.seed(1)
        rw = pd.Series(np.random.randn(400).cumsum())
        wn = pd.Series(np.random.randn(400))
        orders = ARDLBoundsTest.check_integration_orders({"rw": rw, "wn": wn})
        assert orders["wn"]["order"] == "I(0)"
        assert orders["rw"]["order"] in ("I(1)", "I(2)+")


class TestCointegrationTester:
    def test_pairwise_cointegration(self):
        np.random.seed(42)
        x = pd.Series(np.random.randn(252).cumsum())
        y = 2 * x + np.random.randn(252) * 0.1
        is_coint, p_value, stats = CointegrationTester.test_pairwise(y, x)
        assert isinstance(is_coint, bool)
        assert 0 <= p_value <= 1
        assert 'test_statistic' in stats


# ============================================================
# NEURAL NETWORK TESTS
# ============================================================

class TestPositionalEncoding:
    def test_forward_pass(self):
        d_model = 128
        seq_len = 60
        batch_size = 32
        pos_encoder = PositionalEncoding(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model)  # batch_first
        output = pos_encoder(x)
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()


class TestTransformerBlock:
    def test_forward_pass_and_causal_mask_shape(self):
        d_model, nhead, seq_len, batch_size = 128, 8, 60, 16
        block = TransformerBlock(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.1)
        x = torch.randn(batch_size, seq_len, d_model)
        out = block(x)
        assert out.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(out).any()

    def test_causal_mask_blocks_future(self):
        """The causal mask must be strictly upper-triangular (True = ignore future)."""
        mask = TransformerBlock._causal_mask(5, torch.device('cpu'))
        assert mask.shape == (5, 5)
        assert mask[0, 1].item() is True  # position 0 must not attend to position 1 (future)
        assert mask[4, 0].item() is False  # position 4 may attend to position 0 (past)
        assert not mask.diagonal().any()  # a position may always attend to itself


class TestHybridNeuralEncoder:
    def test_forward_pass_returns_prediction_and_latent(self):
        batch_size, seq_len, input_dim, hidden_dim = 16, 60, 40, 64
        encoder = HybridNeuralEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        x = torch.randn(batch_size, seq_len, input_dim)
        prediction, latent = encoder(x)
        assert prediction.shape == (batch_size, 1)
        assert latent.shape == (batch_size, hidden_dim)
        assert not torch.isnan(prediction).any()

    def test_extract_representation_matches_forward_latent(self):
        """extract_representation() must share the encoder path with forward(), not double-compute."""
        torch.manual_seed(0)
        encoder = HybridNeuralEncoder(input_dim=20, hidden_dim=32)
        encoder.eval()
        x = torch.randn(4, 30, 20)
        with torch.no_grad():
            _, latent_from_forward = encoder(x)
            latent_from_extract = encoder.extract_representation(x)
        assert torch.allclose(latent_from_forward, latent_from_extract, atol=1e-6)


class TestVolatilityRegimeDetector:
    def test_output_in_unit_interval(self):
        detector = VolatilityRegimeDetector(input_dim=5, hidden_dim=16)
        x = torch.randn(10, 5)
        out = detector(x)
        assert out.shape == (10, 1)
        assert (out >= 0).all() and (out <= 1).all()


# ============================================================
# FUSION NETWORK TESTS
# ============================================================

class TestGatedFusionMechanism:
    def test_forward_pass(self):
        batch_size, state_dim = 32, 128
        fusion = GatedFusionMechanism(state_dim=state_dim)
        market_state = torch.randn(batch_size, state_dim)
        volatility = torch.randn(batch_size, 1)
        sentiment = torch.randn(batch_size, 1)
        alpha = fusion(market_state, volatility, sentiment)
        assert alpha.shape == (batch_size, 1)
        assert (alpha >= 0).all() and (alpha <= 1).all()


class TestNeuroEconometricNet:
    def test_forward_pass(self):
        batch_size, seq_len, input_dim = 16, 60, 40
        model = NeuroEconometricNet(input_dim=input_dim, hidden_dim=64)
        x = torch.randn(batch_size, seq_len, input_dim)
        ardl_preds = torch.randn(batch_size, 1) * 0.01
        volatility_features = torch.rand(batch_size, 5)
        sentiment = torch.zeros(batch_size, 1)

        fused_pred, alpha, neural_pred = model(x, ardl_preds, volatility_features, sentiment)

        assert fused_pred.shape == (batch_size, 1)
        assert alpha.shape == (batch_size, 1)
        assert neural_pred.shape == (batch_size, 1)
        assert not torch.isnan(fused_pred).any()
        assert (alpha >= 0).all() and (alpha <= 1).all()

    def test_fusion_is_convex_combination(self):
        """fused = alpha*neural + (1-alpha)*ardl BEFORE the output_layer refinement --
        verify the raw gate math, since this is the core architectural claim."""
        torch.manual_seed(0)
        model = NeuroEconometricNet(input_dim=10, hidden_dim=16)
        model.eval()
        x = torch.randn(4, 20, 10)
        ardl = torch.randn(4, 1) * 0.01
        vol = torch.rand(4, 5)
        sent = torch.zeros(4, 1)
        with torch.no_grad():
            neural_pred, latent = model.neural_branch(x)
            vol_regime = model.volatility_detector(vol)
            alpha = model.fusion_gate(latent, vol_regime, sent)
            expected_fused_pre_output = alpha * neural_pred + (1.0 - alpha) * ardl
            expected_final = model.output_layer(expected_fused_pre_output)
            fused, alpha2, neural_pred2 = model(x, ardl, vol, sent)
        assert torch.allclose(alpha, alpha2, atol=1e-6)
        assert torch.allclose(fused, expected_final, atol=1e-5)


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestIntegration:
    def test_data_to_model_pipeline(self, sample_ohlcv_data):
        preprocessor = Preprocessor()
        processed_df, metadata = preprocessor.fit_transform(sample_ohlcv_data)
        assert not processed_df.isna().any().any()

        seq_len = 60
        input_dim = len(metadata['feature_names'])

        x = torch.randn(1, seq_len, input_dim)
        ardl_preds = torch.randn(1, 1) * 0.01
        volatility_features = torch.rand(1, 5)
        sentiment = torch.zeros(1, 1)

        model = NeuroEconometricNet(input_dim=input_dim, hidden_dim=64)
        fused_pred, alpha, neural_pred = model(x, ardl_preds, volatility_features, sentiment)

        assert fused_pred.shape == (1, 1)
        assert not torch.isnan(fused_pred).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
