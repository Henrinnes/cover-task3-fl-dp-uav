"""
MARL environment: state representation and reward function for UAV agents.

State vector for UAV i at round t  (dim = 6):
  [battery_i, snr_norm_i, data_quality_i, round_progress, global_acc, prev_reward_i]

Action → local_epochs mapping:
  0 → skip (0 epochs)
  1 → 1 epoch
  2 → 3 epochs
  3 → 5 epochs

Reward formula:
  R_i = 10 * Δacc * (contrib_i / Σcontrib)  −  0.005 * epochs_i
        with a −0.01 penalty for skipping (discourages passive free-riding)

Battery model:
  drains 0.02 per epoch trained, recovers +0.01 each round (trickle charge),
  clipped to [0.1, 1.0].
"""
from typing import Dict, List

import numpy as np

ACTION_EPOCHS: List[int] = [0, 1, 3, 5]
STATE_DIM: int = 6
_MAX_SNR_DB: float = 40.0


class MARLEnvironment:
    def __init__(self, num_uavs: int, num_rounds: int):
        self.num_uavs   = num_uavs
        self.num_rounds = num_rounds
        self._batteries    = np.ones(num_uavs, dtype=np.float32)
        self._prev_rewards = np.zeros(num_uavs, dtype=np.float32)
        self._prev_acc     = 0.0

    def reset(self):
        self._batteries    = np.random.uniform(0.6, 1.0, self.num_uavs).astype(np.float32)
        self._prev_rewards = np.zeros(self.num_uavs, dtype=np.float32)
        self._prev_acc     = 0.0

    def get_states(
        self,
        snr_db: Dict[int, float],
        data_qualities: np.ndarray,
        round_idx: int,
        global_acc: float,
    ) -> np.ndarray:
        """Returns [num_uavs, STATE_DIM] state matrix."""
        states = np.zeros((self.num_uavs, STATE_DIM), dtype=np.float32)
        for i in range(self.num_uavs):
            snr_norm = float(np.clip(snr_db.get(i, 0.0), 0.0, _MAX_SNR_DB) / _MAX_SNR_DB)
            states[i] = [
                self._batteries[i],
                snr_norm,
                float(data_qualities[i]),
                float(round_idx) / max(self.num_rounds - 1, 1),
                float(np.clip(global_acc, 0.0, 1.0)),
                float(np.clip(self._prev_rewards[i], -2.0, 2.0)),
            ]
        return states

    def compute_rewards(
        self,
        actions: List[int],
        acc_after: float,
        contributions: Dict[int, float],
    ) -> np.ndarray:
        """
        Compute per-UAV reward for the completed round.

        acc_after      : global test accuracy after aggregation
        contributions  : {uav_id: cosine-sim contribution score (≥ 0)}
        """
        acc_gain      = max(acc_after - self._prev_acc, 0.0)
        total_contrib = sum(contributions.values()) + 1e-8
        rewards       = np.zeros(self.num_uavs, dtype=np.float32)

        for i, action in enumerate(actions):
            epochs = ACTION_EPOCHS[action]
            if epochs == 0:
                rewards[i] = -0.01
            else:
                w = contributions.get(i, 0.0) / total_contrib
                rewards[i] = float(10.0 * acc_gain * w - 0.005 * epochs)

        # Battery dynamics
        for i, action in enumerate(actions):
            drain = 0.02 * ACTION_EPOCHS[action]
            self._batteries[i] = float(
                np.clip(self._batteries[i] - drain + 0.01, 0.1, 1.0)
            )

        self._prev_rewards = rewards.copy()
        self._prev_acc     = acc_after
        return rewards
