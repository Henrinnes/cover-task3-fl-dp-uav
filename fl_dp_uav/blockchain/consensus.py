"""
Proof-of-Authority (PoA) consensus for UAV networks.

Rationale: PoW is energy-prohibitive on battery-powered UAVs; PoS requires
persistent stake which is impractical for ephemeral emergency deployments.
PoA uses a pre-authorised validator set (high-reputation UAVs) with round-robin
block proposal and simple majority sign-off.

Ref: Clique PoA spec (EIP-225); adapted for ad-hoc UAV settings.
"""

from typing import Dict, List, Optional

from .block import Block
from .chain import Blockchain


class PoAConsensus:
    """
    Manages the validator set, leader rotation, and block finalisation.
    Tolerates up to f Byzantine validators where 2f+1 ≤ |validators|.
    """

    def __init__(self, validator_ids: List[int], fault_tolerance: int = 1):
        if len(validator_ids) < 2 * fault_tolerance + 1:
            raise ValueError(
                f"Need at least {2*fault_tolerance+1} validators to tolerate {fault_tolerance} faults."
            )
        self.validators = list(validator_ids)
        self.f = fault_tolerance
        self._epoch = 0  # tracks proposal rotation

    @property
    def quorum(self) -> int:
        """Minimum signatures needed to finalise a block."""
        return len(self.validators) // 2 + 1

    def current_proposer(self, exclude: Optional[set] = None) -> Optional[int]:
        """Next eligible proposer in round-robin order, skipping excluded validators.

        Scanning forward from the current epoch (rather than indexing it
        directly) keeps a validator that is temporarily excluded — e.g. the
        round it is caught as Byzantine/free-riding — from being handed the
        proposer slot it can't fulfil.
        """
        exclude = exclude or set()
        n = len(self.validators)
        for offset in range(n):
            candidate = self.validators[(self._epoch + offset) % n]
            if candidate not in exclude:
                return candidate
        return None  # every validator excluded this round

    def propose_and_finalise(
        self,
        chains: Dict[int, Blockchain],
        fl_round: int,
        exclude_ids: Optional[List[int]] = None,
    ) -> Optional[Block]:
        """
        Full PoA round:
          1. Proposer collects mempools from all validators.
          2. Builds a candidate block.
          3. Validators inspect and sign (honest = sign if valid; Byzantine = abstain here).
          4. If quorum reached, block appended to all chains.

        Returns the finalised block, or None if quorum not met.
        """
        exclude = set(exclude_ids or [])
        proposer_id = self.current_proposer(exclude)
        if proposer_id is None:
            return None  # no eligible proposer this round

        # Gather transactions from all validator mempools
        all_txs = []
        for vid in self.validators:
            if vid in chains and vid not in exclude:
                all_txs.extend(chains[vid].flush_mempool())

        # Deduplicate by signature
        seen = set()
        unique_txs = []
        for tx in all_txs:
            if tx.signature not in seen:
                seen.add(tx.signature)
                unique_txs.append(tx)

        if not unique_txs and fl_round > 0:
            return None  # nothing to commit this round

        proposer_chain = chains.get(proposer_id)
        if proposer_chain is None:
            return None

        candidate = proposer_chain.propose_block(proposer_id, unique_txs)

        # Collect validator signatures
        sigs_collected = 0
        for vid in self.validators:
            if vid not in exclude and vid in chains:
                if chains[vid]._is_valid_block(candidate):
                    candidate.add_validator_sig(vid)
                    sigs_collected += 1

        if sigs_collected < self.quorum:
            self._epoch += 1  # rotate away from a stale/failing proposer
            return None  # consensus failed

        # Append to every replica — including excluded validators' — so no
        # honest node's local chain falls permanently out of sync with the
        # canonical chain (exclusion revokes this round's vote/proposal
        # rights only, not ledger replication).
        for vid in self.validators:
            if vid in chains:
                chains[vid].append_block(candidate)

        self._epoch += 1
        return candidate

    def add_validator(self, node_id: int):
        if node_id not in self.validators:
            self.validators.append(node_id)

    def remove_validator(self, node_id: int):
        if node_id in self.validators and len(self.validators) > self.quorum:
            self.validators.remove(node_id)
