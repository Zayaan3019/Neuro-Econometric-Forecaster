"""
Regime Detection Module.

Implements a Gaussian Hidden Markov Model over causal market
return/volatility features to detect latent market regimes (e.g.
low-volatility/trending vs high-volatility/turbulent), used to gate the
Neuro-Econometric fusion mechanism.

CAUSALITY IS THE ENTIRE POINT OF THIS MODULE
----------------------------------------------
A regime label used as a *historical training feature* must only depend on
information available up to that point in time. There are three distinct
ways to decode a fitted HMM, and two of them are non-causal:

1. **Filtered posteriors** P(state_t | obs_1..t)   -- forward pass ONLY.
   Causal. This is the one we use for feature construction.
2. **Smoothed posteriors** P(state_t | obs_1..T)   -- forward-backward.
   Uses every observation in the sequence, including t+1..T. hmmlearn's
   `predict_proba` / `score_samples` return THIS. Non-causal if used as a
   historical feature.
3. **Viterbi path**        argmax P(states_1..T | obs_1..T) -- global
   decode. Also uses the whole sequence (the MAP path at time t can change
   depending on what happens after t). hmmlearn's `decode(algorithm="viterbi")`
   returns this. Non-causal for the same reason.

A documented bug in a sibling project in this portfolio used Viterbi-decoded
regime history as a model input, which leaks future information into the
training set. This module exists specifically to avoid repeating that bug:
`filtered_posteriors` / `high_vol_probability` are the ONLY methods that are
safe to use for historical feature construction. `smoothed_posteriors` and
`viterbi_states` are kept only for offline diagnostics/visualization and are
clearly labeled as non-causal.
"""

from typing import Optional
import numpy as np
from scipy.special import logsumexp
from hmmlearn.hmm import GaussianHMM
import logging

logger = logging.getLogger(__name__)


class CausalRegimeDetector:
    """
    2-state (default) Gaussian HMM regime detector with a causal forward-filter
    decoder for feature construction.

    Fitting (EM over a training window) is not a causality concern in itself --
    it is exactly analogous to fitting a scaler or a neural network's weights on
    a training set. The causality concern is specific to *decoding*: whichever
    decode algorithm is used to turn the fitted model + an observation sequence
    into per-timestep state probabilities must not look into the future when
    those probabilities are used as an "as of time t" feature.
    """

    def __init__(self, n_states: int = 2, random_state: int = 42, n_iter: int = 200):
        self.n_states = n_states
        self.random_state = random_state
        self.n_iter = n_iter
        self.model: Optional[GaussianHMM] = None
        self.high_vol_state_: Optional[int] = None

    def fit(self, features: np.ndarray) -> "CausalRegimeDetector":
        """
        Fit HMM parameters via EM on a TRAINING WINDOW ONLY.

        Args:
            features: (T, F) array of causal features (e.g. daily return,
                rolling realised volatility) computed only from data up to
                each row's own timestamp. Must be the TRAIN split only --
                never fit on validation/test data.
        """
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=self.n_iter,
            random_state=self.random_state,
            tol=1e-3,
        )
        self.model.fit(features)

        # Identify the "high volatility" state as the one with larger variance
        # on the first feature column (by convention: callers should pass
        # realised volatility or |return| as feature column 0).
        variances = self.model.covars_[:, 0, 0]
        self.high_vol_state_ = int(np.argmax(variances))
        logger.info(
            f"CausalRegimeDetector fitted: n_states={self.n_states}, "
            f"high_vol_state={self.high_vol_state_}, "
            f"state_variances={variances.tolist()}"
        )
        return self

    def filtered_posteriors(self, features: np.ndarray) -> np.ndarray:
        """
        Forward-only filtered posterior P(state_t | obs_1..t) for every t.

        Implements the HMM forward (alpha) recursion manually in log-space:
            log_alpha_1(i) = log(pi_i) + log(b_i(obs_1))
            log_alpha_t(i) = logsumexp_j[ log_alpha_{t-1}(j) + log(A_ji) ] + log(b_i(obs_t))
            P(state_t=i | obs_1..t) = normalize_i( exp(log_alpha_t(i)) )

        Deliberately does NOT call hmmlearn's `predict_proba`/`score_samples`,
        which run the forward-BACKWARD algorithm (smoothed, non-causal).

        Args:
            features: (T, F) observation sequence. May extend beyond the
                training window (e.g. train+test concatenated) -- the
                recursion is run once, sequentially, so filtered_posteriors[t]
                is guaranteed to depend only on features[0..t], regardless of
                how far the sequence extends beyond t.
        """
        if self.model is None:
            raise ValueError("Detector not fitted. Call fit() first.")
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        n_t = len(features)
        n_s = self.n_states
        log_frameprob = self.model._compute_log_likelihood(features)  # (T, n_states)
        log_startprob = np.log(self.model.startprob_ + 1e-300)
        log_transmat = np.log(self.model.transmat_ + 1e-300)

        log_alpha = np.zeros((n_t, n_s))
        log_alpha[0] = log_startprob + log_frameprob[0]
        for t in range(1, n_t):
            for j in range(n_s):
                log_alpha[t, j] = (
                    logsumexp(log_alpha[t - 1] + log_transmat[:, j]) + log_frameprob[t, j]
                )

        # Row-wise normalize to get the actual posterior probabilities.
        log_alpha_norm = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
        posteriors = np.exp(log_alpha_norm)
        return posteriors  # (T, n_states)

    def high_vol_probability(self, features: np.ndarray) -> np.ndarray:
        """Causal P(high-volatility regime at t | obs up to t). Safe as a feature."""
        posteriors = self.filtered_posteriors(features)
        return posteriors[:, self.high_vol_state_]

    # ------------------------------------------------------------------
    # Diagnostics only -- NON-CAUSAL. Never feed these into model features.
    # ------------------------------------------------------------------
    def smoothed_posteriors(self, features: np.ndarray) -> np.ndarray:
        """P(state_t | entire sequence). Forward-backward. NON-CAUSAL -- diagnostics only."""
        if self.model is None:
            raise ValueError("Detector not fitted.")
        _, posteriors = self.model.score_samples(features)
        return posteriors

    def viterbi_states(self, features: np.ndarray) -> np.ndarray:
        """Global MAP state path. NON-CAUSAL -- diagnostics only."""
        if self.model is None:
            raise ValueError("Detector not fitted.")
        _, states = self.model.decode(features, algorithm="viterbi")
        return states
