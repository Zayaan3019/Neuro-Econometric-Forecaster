"""
Population Stability Index (PSI) drift monitoring on INPUT feature
distributions.

PSI is a standard credit-risk/ML-monitoring statistic (its use predates
modern MLOps -- see Fannie Mae / SAS scorecard-monitoring documentation)
that quantifies how much a feature's distribution has shifted between a
reference window and a current window:

    PSI = Sum_i ( (current_pct_i - ref_pct_i) * ln(current_pct_i / ref_pct_i) )

where i indexes a fixed set of bins (buckets) defined on the REFERENCE
window, and current_pct_i / ref_pct_i are the fraction of observations
falling in bucket i for each window.

Design choices made explicit here (per the project's own rule: state and
justify parameters rather than copying conventions blindly):

* Bucket count = 10 (deciles of the reference/training distribution).
  Rationale: with 10 roughly-equal-mass reference bins, each bin holds
  ~10% of the training data, so a PSI contribution from any single bin
  reflects a real, non-trivial reallocation of probability mass (moving
  ~1% of mass between two decile bins changes PSI by a small fraction of
  a percent -- the statistic isn't hypersensitive to sampling noise at
  this granularity). Finer bucketing (20-50 bins) is common with very
  large populations; 10 is the standard choice for the sample sizes
  (hundreds to low thousands of rows per walk-forward fold) used here.

* Reference window = the TRAINING split of each walk-forward fold (never
  the fold's own test window) -- this mirrors how PSI is used in
  production: reference = the population the model was fit on, current =
  the population it is scoring now.

* Alert threshold = 0.25, not the commonly quoted "0.2" default.
  Rationale: with 10 discrete bins, a PSI of 0.25 corresponds to a
  distribution shift where roughly a quarter of the population has moved
  into different decile buckets than in training -- e.g. multiple bins
  each seeing several percentage points of mass reallocation, not one
  noisy bin. Given daily financial features are inherently non-stationary
  (volatility regimes shift over multi-year training windows), a stricter
  0.10 "no shift" threshold flags on essentially every fold and stops
  being informative; 0.25 keeps the flag reserved for shifts large enough
  to plausibly matter for the model's validity, while 0.10-0.25 is
  reported as "moderate drift" rather than silently dropped.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

PSI_BUCKETS = 10
PSI_ALERT_THRESHOLD = 0.25
PSI_MODERATE_THRESHOLD = 0.10


def _bucket_edges(reference: np.ndarray, n_buckets: int = PSI_BUCKETS) -> np.ndarray:
    """Quantile bin edges from the reference distribution (deciles by default)."""
    reference = reference[np.isfinite(reference)]
    quantiles = np.linspace(0, 1, n_buckets + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        # Degenerate (near-constant) feature -- widen artificially so PSI is
        # still computable rather than raising.
        lo, hi = reference.min(), reference.max()
        pad = max(abs(hi - lo), 1e-6)
        edges = np.array([lo - pad, lo, hi, hi + pad])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_buckets: int = PSI_BUCKETS,
    epsilon: float = 1e-4,
) -> float:
    """
    Compute PSI for a single feature between a reference and current sample.

    Bin edges are the quantiles of `reference` (not `current`) -- PSI always
    measures how `current` has drifted relative to the population the model
    was built on.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    edges = _bucket_edges(reference, n_buckets)

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    # Additive smoothing so empty bins don't produce log(0)/division by zero.
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)
    ref_pct /= ref_pct.sum()
    cur_pct /= cur_pct.sum()

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def psi_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: List[str],
    n_buckets: int = PSI_BUCKETS,
) -> pd.DataFrame:
    """
    Compute PSI for every feature in `feature_cols` between reference
    (training split) and current (a walk-forward test fold, or live scoring
    batch) windows.

    Returns a DataFrame with columns [feature, psi, status] sorted by PSI
    descending, where status in {"stable", "moderate_drift", "alert"}.
    """
    rows = []
    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        psi = compute_psi(
            reference_df[col].values, current_df[col].values, n_buckets=n_buckets
        )
        if psi >= PSI_ALERT_THRESHOLD:
            status = "alert"
        elif psi >= PSI_MODERATE_THRESHOLD:
            status = "moderate_drift"
        else:
            status = "stable"
        rows.append({"feature": col, "psi": psi, "status": status})

    report = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    n_alert = int((report["status"] == "alert").sum())
    n_moderate = int((report["status"] == "moderate_drift").sum())
    logger.info(
        f"PSI report: {len(report)} features, {n_alert} alert (PSI>={PSI_ALERT_THRESHOLD}), "
        f"{n_moderate} moderate drift (PSI in [{PSI_MODERATE_THRESHOLD},{PSI_ALERT_THRESHOLD}))"
    )
    return report
