"""
Rényi Differential Privacy (RDP) accountant for the subsampled Gaussian mechanism.

References:
  [1] Mironov (2017), "Rényi Differential Privacy."  NeurIPS.
  [2] Wang et al. (2019), "Subsampled Rényi DP and Analytical Moments Accountant."
  [3] Balle et al. (2020), "Hypothesis Testing Interpretations and Renormalization of DP."
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

# Standard RDP orders (α values) used for optimisation
_ORDERS: List[float] = (
    [1 + x / 10.0 for x in range(1, 100)]
    + list(range(12, 65))
    + [1e6]
)


def _rdp_gaussian(alpha: float, sigma: float) -> float:
    """RDP of the Gaussian mechanism (sensitivity = 1, noise std = sigma). [1, Prop 3]"""
    if np.isinf(alpha):
        return 0.5 / sigma**2
    return alpha / (2.0 * sigma**2)


def _rdp_subsampled_gaussian(alpha: float, sigma: float, q: float) -> float:
    """
    RDP of Poisson-subsampled Gaussian mechanism.
    Tight bound for integer α ≥ 2 via the exact log-moment formula [2, Thm 1].
    Falls back to the q²-scaling bound for non-integer or α < 2.
    """
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return _rdp_gaussian(alpha, sigma)

    if alpha >= 2 and alpha == int(alpha):
        a = int(alpha)
        log_sum = -np.inf
        for k in range(a + 1):
            # log C(a,k) + k*log(q) + (a-k)*log(1-q)
            log_binom = math.lgamma(a + 1) - math.lgamma(k + 1) - math.lgamma(a - k + 1)
            log_coeff = log_binom + k * math.log(q) + (a - k) * math.log(1 - q + 1e-300)
            # Gaussian MGF: E[exp((k-1)*privacy_loss)] = exp(k*(k-1)/(2σ²))
            log_mgf = k * (k - 1) / (2.0 * sigma**2) if k >= 2 else 0.0
            log_term = log_coeff + log_mgf
            # numerically stable log-sum-exp
            if log_sum == -np.inf:
                log_sum = log_term
            else:
                m = max(log_sum, log_term)
                log_sum = m + math.log(math.exp(log_sum - m) + math.exp(log_term - m))
        return log_sum / (alpha - 1)

    # Conservative upper bound for non-integer α (safe, slightly loose)
    return min(
        q * _rdp_gaussian(alpha, sigma),
        _rdp_gaussian(alpha, sigma),
    )


def _rdp_to_dp(rdp: float, alpha: float, delta: float) -> float:
    """Convert RDP guarantee to (ε, δ)-DP. [3, Prop 12] — tight conversion."""
    if alpha <= 1:
        return np.inf
    eps = rdp + (math.log(1.0 / delta) + math.log((alpha - 1) / alpha)) / (alpha - 1) - 1.0 / alpha
    return float(eps)


class RDPAccountant:
    """
    Accumulates per-step RDP costs and converts to (ε, δ)-DP on demand.

    Usage:
        acc = RDPAccountant()
        for round in ...:
            acc.accumulate(noise_multiplier=1.1, sample_rate=0.01, num_steps=5)
        eps = acc.get_epsilon(delta=1e-5)
    """

    def __init__(self):
        self._history: List[Tuple[float, float, int]] = []  # (q, sigma, steps)

    def accumulate(self, noise_multiplier: float, sample_rate: float, num_steps: int = 1):
        """Record privacy cost for `num_steps` identical DP-SGD steps."""
        self._history.append((sample_rate, noise_multiplier, num_steps))

    def _total_rdp(self, alpha: float) -> float:
        return sum(
            n * _rdp_subsampled_gaussian(alpha, sigma, q)
            for q, sigma, n in self._history
        )

    def get_epsilon(self, delta: float) -> float:
        """Return the minimum ε such that (ε, δ)-DP holds."""
        best = np.inf
        for alpha in _ORDERS:
            rdp = self._total_rdp(alpha)
            if np.isfinite(rdp):
                eps = _rdp_to_dp(rdp, alpha, delta)
                if np.isfinite(eps):
                    best = min(best, eps)
        return float(best)

    @property
    def total_steps(self) -> int:
        return sum(n for _, _, n in self._history)

    def reset(self):
        self._history.clear()
