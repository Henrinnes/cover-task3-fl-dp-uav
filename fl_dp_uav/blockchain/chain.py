"""
Distributed ledger: hash-linked chain of blocks.
Each UAV node holds a local replica; the class is used per-node.
"""

import time
from typing import List, Optional

from .block import Block, Transaction


class Blockchain:
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.chain: List[Block] = []
        self.mempool: List[Transaction] = []
        self._create_genesis()

    def _create_genesis(self):
        genesis = Block(
            index=0,
            transactions=[],
            previous_hash="0" * 64,
            proposer_id=-1,
            timestamp=0.0,
        ).finalize()
        self.chain.append(genesis)

    @property
    def latest_block(self) -> Block:
        return self.chain[-1]

    @property
    def height(self) -> int:
        return len(self.chain)

    # ------------------------------------------------------------------
    # Mempool management
    # ------------------------------------------------------------------

    def add_transaction(self, tx: Transaction) -> bool:
        """Validate and add transaction to mempool."""
        if not tx.is_valid():
            return False
        self.mempool.append(tx)
        return True

    def flush_mempool(self) -> List[Transaction]:
        txs, self.mempool = self.mempool[:], []
        return txs

    # ------------------------------------------------------------------
    # Block production (called by the PoA proposer)
    # ------------------------------------------------------------------

    def propose_block(self, proposer_id: int, transactions: List[Transaction]) -> Block:
        block = Block(
            index=self.height,
            transactions=transactions,
            previous_hash=self.latest_block.hash,
            proposer_id=proposer_id,
        ).finalize()
        return block

    # ------------------------------------------------------------------
    # Block acceptance (called after PoA consensus)
    # ------------------------------------------------------------------

    def append_block(self, block: Block) -> bool:
        if not self._is_valid_block(block):
            return False
        self.chain.append(block)
        return True

    def _is_valid_block(self, block: Block) -> bool:
        if block.previous_hash != self.latest_block.hash:
            return False
        if block._compute_hash() != block.hash:
            return False
        for tx in block.transactions:
            if not tx.is_valid():
                return False
        return True

    # ------------------------------------------------------------------
    # Chain validation (integrity check)
    # ------------------------------------------------------------------

    def is_valid_chain(self) -> bool:
        for i in range(1, len(self.chain)):
            cur = self.chain[i]
            prev = self.chain[i - 1]
            if cur.previous_hash != prev.hash:
                return False
            if cur._compute_hash() != cur.hash:
                return False
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def transactions_for_round(self, fl_round: int) -> List[Transaction]:
        txs = []
        for block in self.chain:
            txs.extend(tx for tx in block.transactions if tx.fl_round == fl_round)
        return txs

    def get_reward_history(self, node_id: int) -> List[float]:
        rewards = []
        for block in self.chain:
            for tx in block.transactions:
                if tx.tx_type == "REWARD" and tx.recipient_id == node_id:
                    rewards.append(tx.payload.get("amount", 0.0))
        return rewards
