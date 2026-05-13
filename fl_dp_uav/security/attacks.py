"""
Attack models for Task 3.2 — Threat Analysis in UAV FL Networks.

Three attack classes are modelled:

1. SybilAttack
   A single adversary registers multiple fake UAV identities to claim
   disproportionate rewards from the smart contract.
   Threat level: HIGH for incentive layer, LOW for model quality.

2. ByzantineAttack
   Malicious UAVs submit corrupted gradient updates to degrade global model
   convergence (label-flipping or random noise injection).
   Threat level: HIGH for model quality.

3. FreeRiderAttack
   UAVs submit minimal/stale gradients to claim rewards without meaningful
   contribution to FL (classic "free-rider problem" in public goods games).
   Threat level: MEDIUM — degrades convergence slowly.

Refs:
  Fung et al. (2018), "Mitigating Sybils in FL with Foolsgold."
  Blanchard et al. (2017), "Machine Learning with Adversaries: Byzantine Tolerant SGD."
  Lin et al. (2019), "Free-rider Attacks on Model Aggregation in FL."
"""

import copy
from typing import Dict, List, Optional, Tuple

import torch


# ------------------------------------------------------------------
# 1. Sybil Attack
# ------------------------------------------------------------------

class SybilAttack:
    """
    Simulates one adversary controlling `num_fake_ids` phantom UAV identities.
    Each fake UAV submits a copy of a genuine participant's update to pass
    gradient-level checks, while flooding the incentive layer with reward claims.
    """

    def __init__(self, adversary_id: int, num_fake_ids: int = 3):
        self.adversary_id = adversary_id
        self.num_fake_ids = num_fake_ids
        self.fake_ids: List[int] = list(range(10000, 10000 + num_fake_ids))

    def inject_transactions(self, genuine_payload: dict, fl_round: int) -> List[dict]:
        """Return fake transaction payloads claiming the same PoRS proof."""
        fakes = []
        for fid in self.fake_ids:
            fake = copy.deepcopy(genuine_payload)
            fake["uav_id"] = fid
            fake["is_sybil"] = True
            fakes.append(fake)
        return fakes

    def amplified_reward_claim(self, legitimate_reward: float) -> float:
        """Total reward the adversary would unfairly claim."""
        return legitimate_reward * (1 + self.num_fake_ids)


# ------------------------------------------------------------------
# 2. Byzantine Attack
# ------------------------------------------------------------------

class ByzantineAttack:
    """
    Malicious UAVs that submit corrupted model updates.
    Modes:
      'random'       — replace gradients with Gaussian noise
      'sign_flip'    — negate all gradients (strong attack)
      'label_flip'   — scaled noise mimicking label-flipping effect
      'min_max'      — craft update to maximise deviation from honest mean
    """

    def __init__(self, mode: str = "sign_flip", scale: float = 1.0):
        if mode not in {"random", "sign_flip", "label_flip", "min_max"}:
            raise ValueError(f"Unknown Byzantine mode: {mode}")
        self.mode = mode
        self.scale = scale

    def corrupt(
        self,
        state_dict: Dict[str, torch.Tensor],
        honest_updates: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return a corrupted state dict. Integer tensors (e.g. BatchNorm tracking) are left intact."""
        corrupted = {}
        if self.mode == "random":
            for k, v in state_dict.items():
                corrupted[k] = torch.randn_like(v.float()) * self.scale if v.is_floating_point() else v.clone()
        elif self.mode == "sign_flip":
            for k, v in state_dict.items():
                corrupted[k] = -v.float() * self.scale if v.is_floating_point() else v.clone()
        elif self.mode == "label_flip":
            for k, v in state_dict.items():
                corrupted[k] = -v.float() * self.scale * 2.0 if v.is_floating_point() else v.clone()
        elif self.mode == "min_max" and honest_updates:
            mean = {}
            for k in state_dict:
                mean[k] = torch.stack([u[k].float() for u in honest_updates]).mean(0)
            for k, v in state_dict.items():
                if v.is_floating_point():
                    corrupted[k] = mean[k] - (v.float() - mean[k]) * self.scale
                else:
                    corrupted[k] = v.clone()
        else:
            corrupted = copy.deepcopy(state_dict)
        return corrupted


# ------------------------------------------------------------------
# 3. Free-Rider Attack
# ------------------------------------------------------------------

class FreeRiderAttack:
    """
    UAV submits a stale or near-zero gradient update to claim rewards
    while avoiding the computational cost of local training.
    Modes:
      'zero'   — submit all-zero update (most obvious, easily detected)
      'stale'  — resubmit last round's update (harder to detect)
      'noise'  — tiny noise to avoid zero-norm detection
    """

    def __init__(self, mode: str = "noise", noise_scale: float = 1e-4):
        self.mode = mode
        self.noise_scale = noise_scale
        self._last_update: Optional[Dict[str, torch.Tensor]] = None

    def free_ride(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self.mode == "zero":
            result = {k: torch.zeros_like(v) for k, v in state_dict.items()}
        elif self.mode == "stale" and self._last_update is not None:
            result = copy.deepcopy(self._last_update)
        else:  # noise — only perturb float tensors
            result = {
                k: torch.randn_like(v.float()) * self.noise_scale if v.is_floating_point() else v.clone()
                for k, v in state_dict.items()
            }
        self._last_update = copy.deepcopy(state_dict)
        return result
