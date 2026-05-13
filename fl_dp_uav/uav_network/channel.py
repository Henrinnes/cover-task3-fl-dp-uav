import numpy as np

from .node import UAVNode

_C = 3e8  # speed of light


class UAVChannel:
    """
    Air-to-air (A2A) channel: free-space path loss + Rician fading.
    Ref: ITU-R P.525; Al-Hourani et al., "Optimal LAP Altitude for Maximum Coverage."
    """

    def __init__(
        self,
        frequency_hz: float,
        path_loss_exp: float,
        rician_k_db: float,
        noise_power_dbm: float,
        bandwidth_hz: float,
        rng: np.random.Generator,
    ):
        self.freq = frequency_hz
        self.alpha = path_loss_exp
        self.K = 10 ** (rician_k_db / 10)        # linear K-factor
        self.noise_power_dbm = noise_power_dbm
        self.bandwidth_hz = bandwidth_hz
        self.rng = rng

    def _fspl_db(self, distance_m: float) -> float:
        d = max(distance_m, 1.0)
        return 20 * np.log10(4 * np.pi * d * self.freq / _C)

    def _rician_power_gain(self) -> float:
        """Sample one Rician fading power gain (non-central chi-squared, 2 DoF)."""
        sigma = 1.0 / np.sqrt(2 * (self.K + 1))
        mu_los = np.sqrt(self.K / (self.K + 1))
        r = self.rng.normal(mu_los, sigma) ** 2 + self.rng.normal(0.0, sigma) ** 2
        return float(r)

    def snr_db(self, tx: UAVNode, rx: UAVNode) -> float:
        dist = tx.distance_to(rx)
        pl = self._fspl_db(dist)
        fading_db = 10 * np.log10(self._rician_power_gain() + 1e-15)
        rx_power_dbm = tx.tx_power_dbm - pl + fading_db
        return rx_power_dbm - self.noise_power_dbm

    def shannon_rate_bps(self, snr_db: float) -> float:
        snr_lin = 10 ** (snr_db / 10)
        return self.bandwidth_hz * np.log2(1.0 + snr_lin)

    def can_communicate(self, tx: UAVNode, rx: UAVNode, min_snr_db: float) -> bool:
        return self.snr_db(tx, rx) >= min_snr_db
