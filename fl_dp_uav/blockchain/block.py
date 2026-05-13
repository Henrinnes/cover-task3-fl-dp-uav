"""
Core blockchain data structures: Transaction and Block.

Transaction types used in Task 3.2:
  DATA_SHARE      — UAV submits a gradient update proof for an FL round
  RESOURCE_STAKE  — UAV stakes compute/bandwidth to participate
  REWARD          — Smart contract distributes tokens to contributors
  REPUTATION      — Smart contract updates a UAV's trust score
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Transaction:
    tx_type: str          # DATA_SHARE | RESOURCE_STAKE | REWARD | REPUTATION
    sender_id: int
    recipient_id: int
    fl_round: int
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    signature: str = field(default="", init=False)

    def __post_init__(self):
        self.signature = self._sign()

    def _sign(self) -> str:
        """Simulated ECDSA: SHA-256 of (sender | type | payload | timestamp)."""
        raw = f"{self.sender_id}{self.tx_type}{json.dumps(self.payload, sort_keys=True)}{self.timestamp:.6f}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def is_valid(self) -> bool:
        return self.signature == self._sign()

    def to_dict(self) -> dict:
        return {
            "tx_type": self.tx_type,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "fl_round": self.fl_round,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "signature": self.signature,
        }


@dataclass
class Block:
    index: int
    transactions: List[Transaction]
    previous_hash: str
    proposer_id: int
    timestamp: float = field(default_factory=time.time)
    validator_signatures: List[str] = field(default_factory=list)
    _hash: str = field(default="", init=False, repr=False)

    def _compute_hash(self) -> str:
        content = json.dumps(
            {
                "index": self.index,
                "txs": [tx.to_dict() for tx in self.transactions],
                "previous_hash": self.previous_hash,
                "proposer_id": self.proposer_id,
                "timestamp": self.timestamp,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    @property
    def hash(self) -> str:
        if not self._hash:
            self._hash = self._compute_hash()
        return self._hash

    def finalize(self) -> "Block":
        self._hash = self._compute_hash()
        return self

    def add_validator_sig(self, validator_id: int):
        sig = hashlib.sha256(f"{validator_id}{self.hash}".encode()).hexdigest()[:16]
        self.validator_signatures.append(sig)

    @property
    def num_validators(self) -> int:
        return len(self.validator_signatures)
