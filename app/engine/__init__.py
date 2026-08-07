"""
Engine module: uncertainty quantification, statistical significance tests,
and baseline forecasters used by the walk-forward evaluation harness
(walk_forward_pipeline.py, at the repository root).

This previously also exported `Trainer`/`HybridLoss`/`EarlyStopping` (from
trainer.py) and `WalkForwardBacktester`/`PerformanceAnalyzer` (from
backtester.py). Both files were only used by the now-removed
train_model_production.py / evaluate_model_production.py scripts, which
were superseded by run_pipeline.py / walk_forward_pipeline.py and had two
unfixed bugs of their own (pre-split scaler leakage; a directional-accuracy
loss term defined as "above/below this batch's mean" rather than a true
sign-of-return). Rather than fix bugs in code nothing exercises anymore,
they were removed along with the scripts that used them.
"""

from app.engine.uncertainty import (
    enable_mc_dropout, mc_dropout_predict, prediction_intervals,
    empirical_coverage, pit_values, pit_uniformity_test, calibration_report,
)
from app.engine.statistical_tests import diebold_mariano_test
from app.engine.baselines import (
    persistence_forecast, sign_persistence_forecast,
    fit_arima_order, rolling_arima_forecast,
)

__all__ = [
    'enable_mc_dropout', 'mc_dropout_predict', 'prediction_intervals',
    'empirical_coverage', 'pit_values', 'pit_uniformity_test', 'calibration_report',
    'diebold_mariano_test',
    'persistence_forecast', 'sign_persistence_forecast',
    'fit_arima_order', 'rolling_arima_forecast',
]
