"""
A computation that fails must raise, not return a plausible-looking default.

Audit defect 8: a bare `except` turned every AR fit into "yesterday's close",
and the hybrid trained on a naive lag for its whole history. Defect 5: a
bounds test returned a fixed answer without computing anything. These tests
pin the rule for the econometric code that remains.
"""
import numpy as np
import pandas as pd
import pytest

from app.data.preprocessor import Preprocessor
from app.models import econometrics
from app.models.econometrics import ARDLBoundsTest, ARDLModel, CointegrationTester
from app.models.fusion import NeuroEconometricNet


@pytest.fixture
def random_walk():
    rng = np.random.default_rng(0)
    return pd.Series(100 + np.cumsum(rng.normal(size=300)))


def test_ardl_forecast_failure_raises_instead_of_returning_last_value(random_walk, monkeypatch):
    model = ARDLModel(lags=3, select_order=False).fit(endog=random_walk)

    def broken(*args, **kwargs):
        raise ValueError("simulated numerical failure")

    monkeypatch.setattr(model.fitted_model_, "predict", broken)
    with pytest.raises(RuntimeError, match="ARDL forecast failed"):
        model.predict(steps=1)


def test_ardl_forecast_is_a_real_forecast_not_the_last_value(random_walk):
    model = ARDLModel(lags=3, select_order=False).fit(endog=random_walk)
    forecast = model.predict(steps=1)
    assert forecast.shape == (1,)
    assert forecast[0] != pytest.approx(random_walk.iloc[-1], abs=1e-12)


def test_bounds_test_failure_raises_instead_of_returning_a_verdict(random_walk, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("simulated failure")

    monkeypatch.setattr(econometrics, "ardl_select_order", broken)
    exog = pd.DataFrame({"x": random_walk.values[::-1].copy()})
    with pytest.raises(RuntimeError, match="Bounds test could not be computed"):
        ARDLBoundsTest.bounds_f_test(random_walk, exog, max_lags=2)


def test_failed_adf_raises_instead_of_reporting_a_unit_root(monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("simulated failure")

    monkeypatch.setattr(econometrics, "adfuller", broken)
    with pytest.raises(ValueError, match="ADF test failed"):
        ARDLBoundsTest.check_integration_orders({"s": np.arange(50.0)})


def test_failed_cointegration_raises_instead_of_reporting_none(random_walk, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("simulated failure")

    monkeypatch.setattr(econometrics, "coint", broken)
    with pytest.raises(RuntimeError, match="cointegration test could not be computed"):
        CointegrationTester.test_pairwise(random_walk, random_walk)


def test_undifferencing_without_an_anchor_raises(random_walk):
    df = pd.DataFrame({
        "Open": random_walk, "High": random_walk + 1, "Low": random_walk - 1,
        "Close": random_walk, "Volume": 1e6,
    })
    pre = Preprocessor()
    _, meta = pre.fit_transform(df, fit_rows=200)
    if meta["diff_order"] == 0:
        pytest.skip("fixture came out stationary; nothing to undifference")
    with pytest.raises(ValueError, match="last_actual_value"):
        pre.inverse_transform_predictions(np.zeros(3))


def test_hybrid_network_has_no_hidden_econometric_model():
    """The AR forecast is an INPUT; the network must not carry its own fallback-prone fitter."""
    net = NeuroEconometricNet(input_dim=8, hidden_dim=32)
    assert not hasattr(net, "ardl_predictor")
