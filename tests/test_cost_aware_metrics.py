"""
The sign-following Sharpe, gross and net of trading costs.

A strategy that flips every day pays two units of turnover per flip; one that
holds its position pays once on entry. The net figures must reflect exactly
that, so a gross Sharpe can never be quoted without its cost.
"""

import numpy as np
import pytest

from walk_forward_pipeline import COST_BPS_GRID, compute_metrics


def _sharpe(r):
    return np.sqrt(252) * r.mean() / r.std()


def test_buy_and_hold_pays_once():
    rng = np.random.default_rng(0)
    actuals = rng.normal(0.0005, 0.01, 500)
    m = compute_metrics(np.ones(500), actuals)
    assert m["avg_daily_turnover"] == pytest.approx(1 / 500)
    expected = actuals.copy()
    expected[0] -= 5 / 1e4
    assert m["Sharpe_net_of_costs"]["5bps"] == pytest.approx(_sharpe(expected))


def test_daily_flipping_pays_two_units_per_flip():
    rng = np.random.default_rng(1)
    actuals = rng.normal(0.0, 0.01, 400)
    preds = np.where(np.arange(400) % 2 == 0, 1.0, -1.0)
    m = compute_metrics(preds, actuals)
    # 1 unit on entry, then 2 on each of the 399 flips.
    assert m["avg_daily_turnover"] == pytest.approx((1 + 2 * 399) / 400)
    turnover = np.full(400, 2.0)
    turnover[0] = 1.0
    gross = np.sign(preds) * actuals
    for bps in COST_BPS_GRID:
        assert m["Sharpe_net_of_costs"][f"{bps:g}bps"] == pytest.approx(
            _sharpe(gross - turnover * bps / 1e4)
        )


def test_costs_only_lower_the_sharpe():
    rng = np.random.default_rng(2)
    actuals = rng.normal(0.0004, 0.01, 1000)
    preds = actuals + rng.normal(0, 0.01, 1000)     # some skill, frequent flips
    m = compute_metrics(preds, actuals)
    nets = [m["Sharpe_net_of_costs"][f"{b:g}bps"] for b in sorted(COST_BPS_GRID)]
    assert m["Sharpe_naive_signal_strategy"] > nets[0] > nets[1] > nets[2]


def test_flat_forecast_trades_nothing():
    m = compute_metrics(np.zeros(100), np.random.default_rng(3).normal(0, 0.01, 100))
    assert m["avg_daily_turnover"] == 0.0
    assert all(v == 0.0 for v in m["Sharpe_net_of_costs"].values())
