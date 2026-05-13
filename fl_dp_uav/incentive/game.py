"""
Stackelberg Game-Theoretic Incentive Mechanism for FL in UAV Networks.

Model
-----
Leader  : Parameter server (PS) sets per-unit reward rate λ ≥ 0
Followers: N UAVs, each chooses effort e_i ∈ [0, e_max]

UAV utility (follower i):
    U_i(e_i; λ) = λ · e_i  −  (c_i / 2) · e_i²

  where c_i > 0 is UAV i's heterogeneous cost coefficient
  (captures energy, compute, and bandwidth costs).

Follower best response (BR):
    ∂U_i/∂e_i = λ − c_i · e_i = 0
    ⟹ e_i*(λ) = min(λ / c_i, e_max)

PS utility (leader):
    U_s(λ) = α · log(1 + E(λ))  −  λ · E(λ)

  where E(λ) = Σ e_i*(λ)  is the aggregate FL effort
  and  α > 0 controls how much the PS values model quality.
  (Logarithmic quality captures diminishing returns of more FL data.)

Stackelberg equilibrium (SE):
    λ* = argmax U_s(λ)   subject to  λ ≥ 0
    ∂U_s/∂λ = αH/(1 + λH) − H(2λ + 1/c̄) = 0   [see derivation below]

For the unconstrained case with e_i*(λ) = λ/c_i:
    E(λ) = λ · H   where H = Σ(1/c_i)
    U_s(λ) = α · log(1 + λH) − λ²H
    ∂U_s/∂λ = αH/(1 + λH) − 2λH = 0
    ⟹ 2λ²H + 2λ − α = 0
    ⟹ λ* = (−1 + √(1 + 2αH)) / (2H)   [positive root]

Refs:
  Yang et al. (2019), "Incentive Mechanism Design for Federated Learning."
  Zhan et al. (2020), "A Learning-Based Incentive Mechanism for FL." IEEE IoT J.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class UAVGamePlayer:
    uav_id: int
    cost_coeff: float        # c_i: higher → more expensive to contribute
    max_effort: float = 1.0  # e_max: physical upper bound on effort
    effort: float = 0.0      # current effort level e_i
    utility: float = 0.0     # current utility U_i

    def best_response(self, reward_rate: float) -> float:
        """e_i*(λ) = min(λ/c_i, e_max)"""
        return min(reward_rate / max(self.cost_coeff, 1e-8), self.max_effort)

    def compute_utility(self, reward_rate: float) -> float:
        return reward_rate * self.effort - (self.cost_coeff / 2.0) * self.effort**2


@dataclass
class StackelbergGame:
    """
    One-leader (PS) N-follower (UAV) Stackelberg game.
    Solves for the optimal reward rate λ* and the resulting Nash equilibrium.
    """

    players: List[UAVGamePlayer]
    quality_weight: float = 10.0   # α: PS valuation of FL model quality
    _history: List[dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Equilibrium computation
    # ------------------------------------------------------------------

    def _aggregate_effort(self, reward_rate: float) -> float:
        return sum(p.best_response(reward_rate) for p in self.players)

    def _ps_utility(self, reward_rate: float) -> float:
        E = self._aggregate_effort(reward_rate)
        return self.quality_weight * math.log1p(E) - reward_rate * E

    def optimal_reward_rate(self) -> float:
        """
        Analytical SE solution for the unconstrained case e_i* = λ/c_i.
        H = Σ(1/c_i), λ* = (−1 + √(1 + 2αH)) / (2H)
        Falls back to numerical optimisation when max_effort is binding.
        """
        # Check if any player is effort-constrained at the analytical λ*
        H = sum(1.0 / max(p.cost_coeff, 1e-8) for p in self.players)
        if H < 1e-8:
            return 0.0

        lambda_analytical = (-1.0 + math.sqrt(1.0 + 2.0 * self.quality_weight * H)) / (2.0 * H)

        all_unconstrained = all(
            lambda_analytical / p.cost_coeff <= p.max_effort for p in self.players
        )
        if all_unconstrained:
            return lambda_analytical

        # Numerical optimisation via golden-section search
        return self._golden_section_search(lo=0.0, hi=self.quality_weight, tol=1e-6)

    def _golden_section_search(self, lo: float, hi: float, tol: float) -> float:
        gr = (math.sqrt(5) + 1) / 2
        c = hi - (hi - lo) / gr
        d = lo + (hi - lo) / gr
        while abs(hi - lo) > tol:
            if self._ps_utility(c) > self._ps_utility(d):
                hi = d
            else:
                lo = c
            c = hi - (hi - lo) / gr
            d = lo + (hi - lo) / gr
        return (lo + hi) / 2

    # ------------------------------------------------------------------
    # Game execution
    # ------------------------------------------------------------------

    def solve(self, fl_round: int = 0) -> dict:
        """
        Compute Stackelberg equilibrium for this round.
        Returns a result dict suitable for logging.
        """
        lambda_star = self.optimal_reward_rate()

        total_effort = 0.0
        for p in self.players:
            p.effort = p.best_response(lambda_star)
            p.utility = p.compute_utility(lambda_star)
            total_effort += p.effort

        ps_utility = self._ps_utility(lambda_star)

        result = {
            "fl_round": fl_round,
            "lambda_star": lambda_star,
            "total_effort": total_effort,
            "ps_utility": ps_utility,
            "avg_uav_utility": np.mean([p.utility for p in self.players]),
            "efforts": {p.uav_id: p.effort for p in self.players},
        }
        self._history.append(result)
        return result

    # ------------------------------------------------------------------
    # Social welfare analysis
    # ------------------------------------------------------------------

    def social_welfare(self) -> float:
        """Sum of all players' utilities at current equilibrium."""
        if not self._history:
            return 0.0
        last = self._history[-1]
        return last["ps_utility"] + sum(p.utility for p in self.players)

    def participation_rate(self, effort_threshold: float = 0.05) -> float:
        """Fraction of UAVs with effort above threshold (willing to participate)."""
        active = sum(1 for p in self.players if p.effort >= effort_threshold)
        return active / max(len(self.players), 1)

    def free_rider_ids(self, effort_threshold: float = 0.05) -> List[int]:
        return [p.uav_id for p in self.players if p.effort < effort_threshold]

    def history_summary(self) -> dict:
        if not self._history:
            return {}
        lambdas = [h["lambda_star"] for h in self._history]
        efforts = [h["total_effort"] for h in self._history]
        return {
            "rounds": len(self._history),
            "lambda_mean": float(np.mean(lambdas)),
            "lambda_std": float(np.std(lambdas)),
            "effort_mean": float(np.mean(efforts)),
            "effort_std": float(np.std(efforts)),
        }


# ------------------------------------------------------------------
# Factory: build a game from UAV cost configuration
# ------------------------------------------------------------------

def build_game(
    num_uavs: int,
    cost_mode: str = "heterogeneous",
    quality_weight: float = 10.0,
    rng: Optional[np.random.Generator] = None,
) -> StackelbergGame:
    """
    Convenience factory. cost_mode:
      'homogeneous'   — all c_i = 1.0 (baseline)
      'heterogeneous' — c_i ~ Uniform(0.5, 3.0) (realistic: different UAV capabilities)
      'adversarial'   — some UAVs have very high cost (near-free-riders)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if cost_mode == "homogeneous":
        costs = [1.0] * num_uavs
    elif cost_mode == "heterogeneous":
        costs = rng.uniform(0.5, 3.0, size=num_uavs).tolist()
    elif cost_mode == "adversarial":
        costs = rng.uniform(0.5, 2.0, size=num_uavs).tolist()
        # 20% of UAVs are near-free-riders (high cost → low effort)
        n_adversarial = max(1, num_uavs // 5)
        for i in rng.choice(num_uavs, size=n_adversarial, replace=False):
            costs[i] = rng.uniform(8.0, 15.0)
    else:
        raise ValueError(f"Unknown cost_mode: {cost_mode}")

    players = [
        UAVGamePlayer(uav_id=i, cost_coeff=c) for i, c in enumerate(costs)
    ]
    return StackelbergGame(players=players, quality_weight=quality_weight)
