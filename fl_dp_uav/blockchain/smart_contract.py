"""
Resource-Sharing Smart Contract for UAV FL networks.

Responsibilities:
  1. Accept and verify Proof-of-Resource-Sharing (PoRS) submissions.
  2. Compute contribution scores and distribute token rewards.
  3. Update UAV reputation scores based on verified behaviour.
  4. Maintain on-chain ledger of token balances.

The contract is executed by the PS after each block is finalised.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .block import Transaction


@dataclass
class ContractState:
    """Persistent state stored on-chain (updated after each block)."""
    token_balances: Dict[int, float] = field(default_factory=dict)
    reputation_scores: Dict[int, float] = field(default_factory=dict)
    total_rewards_distributed: float = 0.0
    round_contributions: Dict[int, Dict[int, float]] = field(default_factory=dict)  # round → {uav_id: score}


class ResourceSharingContract:
    """
    Implements the incentive-compatible resource-sharing contract.

    Reward formula (Stackelberg-derived, see incentive/game.py):
      reward_i = (contribution_i / total_contribution) * round_budget
    where contribution_i is the verified effort score from PoRS.
    """

    INITIAL_BALANCE = 100.0      # tokens each UAV starts with
    STAKE_REQUIRED = 5.0         # tokens required to participate
    BASE_ROUND_BUDGET = 50.0     # tokens distributed per FL round
    REPUTATION_DECAY = 0.95      # reputation decay factor per round

    def __init__(self):
        self.state = ContractState()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def register_uav(self, uav_id: int):
        if uav_id not in self.state.token_balances:
            self.state.token_balances[uav_id] = self.INITIAL_BALANCE
            self.state.reputation_scores[uav_id] = 1.0

    # ------------------------------------------------------------------
    # PoRS verification and contribution scoring
    # ------------------------------------------------------------------

    def _verify_pors(self, proof: dict) -> Tuple[bool, float]:
        """
        Verify a Proof-of-Resource-Sharing and return (is_valid, effort_score).
        In simulation: checks that all required proof fields are present and sane.
        Real deployment: would use zero-knowledge proofs or trusted hardware attestation.
        """
        required = {"gradient_hash", "data_merkle_root", "compute_time_s", "bytes_transmitted"}
        if not required.issubset(proof.keys()):
            return False, 0.0

        compute_time = proof.get("compute_time_s", 0)
        bytes_tx = proof.get("bytes_transmitted", 0)
        num_samples = proof.get("num_samples", 0)

        if compute_time <= 0 or bytes_tx <= 0 or num_samples <= 0:
            return False, 0.0

        # Effort score: normalised combination of computation + data + bandwidth
        effort = (
            0.5 * min(compute_time / 60.0, 1.0)          # up to 60s normalised
            + 0.3 * min(num_samples / 500.0, 1.0)         # up to 500 samples
            + 0.2 * min(bytes_tx / (10 * 1024 * 1024), 1.0)  # up to 10 MB
        )
        return True, float(effort)

    # ------------------------------------------------------------------
    # Processing transactions from a finalised block
    # ------------------------------------------------------------------

    def process_block(self, transactions: List[Transaction], fl_round: int):
        """Execute all contract-relevant transactions from a finalised block."""
        # Decay reputations each round
        for uid in self.state.reputation_scores:
            self.state.reputation_scores[uid] *= self.REPUTATION_DECAY
            self.state.reputation_scores[uid] = max(
                self.state.reputation_scores[uid], 0.01
            )

        contributions: Dict[int, float] = {}

        for tx in transactions:
            if tx.tx_type == "DATA_SHARE":
                self._handle_data_share(tx, contributions)
            elif tx.tx_type == "RESOURCE_STAKE":
                self._handle_stake(tx)

        if contributions:
            self._distribute_rewards(contributions, fl_round)

        self.state.round_contributions[fl_round] = dict(contributions)

    def _handle_data_share(self, tx: Transaction, contributions: Dict[int, float]):
        uid = tx.sender_id
        self.register_uav(uid)

        if self.state.token_balances.get(uid, 0) < self.STAKE_REQUIRED:
            return  # insufficient stake — transaction invalid

        proof = tx.payload.get("pors_proof", {})
        valid, effort = self._verify_pors(proof)

        if valid:
            reputation = self.state.reputation_scores.get(uid, 1.0)
            contributions[uid] = effort * reputation
            # Boost reputation for valid contribution
            self.state.reputation_scores[uid] = min(
                self.state.reputation_scores[uid] + 0.05, 1.0
            )

    def _handle_stake(self, tx: Transaction):
        uid = tx.sender_id
        self.register_uav(uid)
        amount = tx.payload.get("amount", 0.0)
        if amount > 0:
            self.state.token_balances[uid] = (
                self.state.token_balances.get(uid, 0) + amount
            )

    def _distribute_rewards(self, contributions: Dict[int, float], fl_round: int):
        total = sum(contributions.values())
        if total <= 0:
            return

        # Scale budget by number of contributors (encourages participation)
        budget = self.BASE_ROUND_BUDGET * min(1.0, len(contributions) / 5.0)

        for uid, contrib in contributions.items():
            reward = (contrib / total) * budget
            self.state.token_balances[uid] = (
                self.state.token_balances.get(uid, 0) + reward
            )
            self.state.total_rewards_distributed += reward

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def balance(self, uav_id: int) -> float:
        return self.state.token_balances.get(uav_id, 0.0)

    def reputation(self, uav_id: int) -> float:
        return self.state.reputation_scores.get(uav_id, 0.0)

    def leaderboard(self) -> List[Tuple[int, float]]:
        return sorted(
            self.state.token_balances.items(), key=lambda x: x[1], reverse=True
        )
