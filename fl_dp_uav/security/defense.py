"""
Defence mechanisms against Sybil, Byzantine, and free-rider attacks.

1. SybilDefense      — stake-based identity + PoRS cross-check
2. ByzantineDefense  — gradient anomaly detection (cosine similarity + norm)
3. FreeRiderDetector — gradient norm threshold + cross-round consistency
4. AggregationRules  — robust aggregation alternatives to FedAvg

Refs:
  Fung et al. (2018), "Foolsgold: Countering Sybils in FL."
  Yin et al. (2018), "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates."
  El Mhamdi et al. (2018), "The Hidden Vulnerability of Distributed Learning in Byzantium."
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch


# ------------------------------------------------------------------
# 1. Sybil Defence
# ------------------------------------------------------------------

class SybilDefense:
    """
    Detects Sybil identities by requiring:
      (a) Minimum on-chain token stake before accepting transactions.
      (b) Gradient similarity check: Sybil clones share near-identical updates.

    FoolsGold variant: penalise clients with cosine similarity > threshold
    (genuine clients have diverse updates due to non-IID data).
    """

    def __init__(
        self,
        min_stake: float = 5.0,
        similarity_threshold: float = 0.95,
    ):
        self.min_stake = min_stake
        self.similarity_threshold = similarity_threshold

    def filter_by_stake(
        self, client_ids: List[int], balances: Dict[int, float]
    ) -> List[int]:
        return [cid for cid in client_ids if balances.get(cid, 0) >= self.min_stake]

    def _flatten(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([v.detach().cpu().float().flatten() for v in state_dict.values() if v.is_floating_point()])

    def detect_sybil_clones(
        self, updates: List[Tuple[int, Dict[str, torch.Tensor]]]
    ) -> Set[int]:
        """
        FoolsGold: flag pairs with cosine similarity above threshold as potential Sybils.
        Returns set of suspected client IDs (keeps the first occurrence, flags the rest).
        """
        if len(updates) < 2:
            return set()

        vecs = [(cid, self._flatten(sd)) for cid, sd in updates]
        flagged: Set[int] = set()
        seen_vecs: List[Tuple[int, torch.Tensor]] = []

        for cid, vec in vecs:
            for prev_cid, prev_vec in seen_vecs:
                cos_sim = torch.nn.functional.cosine_similarity(
                    vec.unsqueeze(0), prev_vec.unsqueeze(0)
                ).item()
                if cos_sim > self.similarity_threshold:
                    flagged.add(cid)
                    break
            seen_vecs.append((cid, vec))

        return flagged


# ------------------------------------------------------------------
# 2. Byzantine Defence
# ------------------------------------------------------------------

class ByzantineDefense:
    """
    Detects Byzantine updates via gradient norm + cosine anomaly scoring.
    Clients with Z-score outlier norms OR low mean cosine similarity are excluded.
    """

    def __init__(
        self,
        norm_z_threshold: float = 2.5,
        cosine_threshold: float = 0.0,
    ):
        self.norm_z_threshold = norm_z_threshold
        self.cosine_threshold = cosine_threshold

    def _flatten(self, sd: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([v.detach().cpu().float().flatten() for v in sd.values() if v.is_floating_point()])

    def detect(
        self, updates: List[Tuple[int, Dict[str, torch.Tensor]]]
    ) -> Tuple[List[int], List[int]]:
        """
        Returns (honest_ids, flagged_ids).
        Applies norm outlier detection + cosine deviation test.
        """
        if len(updates) < 3:
            return [cid for cid, _ in updates], []

        vecs = [(cid, self._flatten(sd)) for cid, sd in updates]
        norms = np.array([v.norm().item() for _, v in vecs])

        # Z-score norm filter
        mu, sigma = norms.mean(), norms.std() + 1e-8
        z_scores = np.abs((norms - mu) / sigma)

        # Cosine similarity to the mean gradient
        all_vecs = torch.stack([v for _, v in vecs])
        mean_vec = all_vecs.mean(0)
        mean_norm = mean_vec.norm() + 1e-8

        honest, flagged = [], []
        for i, (cid, vec) in enumerate(vecs):
            cos = torch.nn.functional.cosine_similarity(
                vec.unsqueeze(0), mean_vec.unsqueeze(0)
            ).item()
            if z_scores[i] > self.norm_z_threshold or cos < self.cosine_threshold:
                flagged.append(cid)
            else:
                honest.append(cid)

        return honest, flagged


# ------------------------------------------------------------------
# 3. Free-Rider Detection
# ------------------------------------------------------------------

class FreeRiderDetector:
    """
    Flags clients whose gradient norm falls below a dynamic threshold
    (set at fraction `threshold_frac` of the median norm).
    Also tracks inter-round update similarity to catch stale-update free-riders.
    """

    def __init__(self, threshold_frac: float = 0.05):
        self.threshold_frac = threshold_frac
        self._prev_updates: Dict[int, torch.Tensor] = {}

    def _flatten(self, sd: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([v.float().flatten() for v in sd.values()])

    def detect(
        self, updates: List[Tuple[int, Dict[str, torch.Tensor]]]
    ) -> Tuple[List[int], List[int]]:
        """Returns (genuine_ids, free_rider_ids)."""
        if not updates:
            return [], []

        vecs = {cid: self._flatten(sd) for cid, sd in updates}
        norms = np.array([vecs[cid].norm().item() for cid in vecs])
        median_norm = float(np.median(norms)) if len(norms) > 0 else 1.0
        threshold = self.threshold_frac * median_norm

        genuine, free_riders = [], []
        for cid, vec in vecs.items():
            norm = vec.norm().item()
            stale = False
            if cid in self._prev_updates:
                prev = self._prev_updates[cid].cpu()
                cos = torch.nn.functional.cosine_similarity(
                    vec.cpu().unsqueeze(0), prev.unsqueeze(0)
                ).item()
                stale = cos > 0.9999

            if norm < threshold or stale:
                free_riders.append(cid)
            else:
                genuine.append(cid)

        # Update history
        for cid, vec in vecs.items():
            self._prev_updates[cid] = vec.detach().clone()

        return genuine, free_riders


# ------------------------------------------------------------------
# 4. Robust Aggregation Rules
# ------------------------------------------------------------------

def _agg_key(updates, key, fn):
    """Apply fn to float tensors; for integer tensors (e.g. BatchNorm tracking) use majority vote."""
    sample = updates[0][0][key]
    if sample.is_floating_point():
        return fn(updates, key)
    # Integer tensor: return the value from the first honest update
    return sample.clone()


def fedavg(
    updates: List[Tuple[Dict[str, torch.Tensor], int]]
) -> Dict[str, torch.Tensor]:
    """Standard FedAvg — all tensors moved to CPU before averaging."""
    total = sum(n for _, n in updates)
    result = {}
    for key in updates[0][0]:
        sample = updates[0][0][key]
        if sample.is_floating_point():
            result[key] = sum(w[key].detach().cpu().float() * (n / total) for w, n in updates)
        else:
            result[key] = sample.detach().cpu().clone()
    return result


def coordinate_wise_median(
    updates: List[Tuple[Dict[str, torch.Tensor], int]]
) -> Dict[str, torch.Tensor]:
    """
    Coordinate-wise median aggregation (Yin et al., 2018).
    Tolerates up to ⌊(n-1)/2⌋ Byzantine clients.
    """
    result = {}
    for key in updates[0][0]:
        sample = updates[0][0][key]
        if sample.is_floating_point():
            stacked = torch.stack([w[key].detach().cpu().float() for w, _ in updates])
            result[key] = stacked.median(dim=0).values
        else:
            result[key] = sample.detach().cpu().clone()
    return result


def trimmed_mean(
    updates: List[Tuple[Dict[str, torch.Tensor], int]],
    trim_fraction: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Trimmed-mean aggregation: drop top and bottom `trim_fraction` per coordinate.
    """
    n = len(updates)
    k = max(1, int(n * trim_fraction))
    result = {}
    for key in updates[0][0]:
        sample = updates[0][0][key]
        if sample.is_floating_point():
            stacked = torch.stack([w[key].detach().cpu().float() for w, _ in updates])
            sorted_vals, _ = stacked.sort(dim=0)
            trimmed = sorted_vals[k : n - k]
            result[key] = trimmed.mean(dim=0)
        else:
            result[key] = sample.clone()
    return result
