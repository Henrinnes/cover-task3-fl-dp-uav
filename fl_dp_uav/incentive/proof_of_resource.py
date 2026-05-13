"""
Proof-of-Resource-Sharing (PoRS) protocol.

Each participating UAV constructs a PoRS proof that commits to:
  - gradient_hash     : SHA-256 of the model update (proves computation)
  - data_merkle_root  : Merkle root of training batches (proves data usage)
  - compute_time_s    : wall-clock training time (proves effort)
  - bytes_transmitted : gradient update size in bytes (proves bandwidth)
  - num_samples       : number of local training samples used

In a real deployment these would use trusted hardware attestation (TrustZone/SGX)
or zero-knowledge proofs. Here we simulate the protocol faithfully.
"""

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


def _gradient_hash(state_dict: Dict[str, torch.Tensor]) -> str:
    """SHA-256 of the serialised gradient update (truncated for efficiency)."""
    buf = []
    for key in sorted(state_dict.keys()):
        buf.append(state_dict[key].cpu().float().numpy().tobytes())
    raw = b"".join(buf)
    return hashlib.sha256(raw).hexdigest()


def _merkle_root(items: List[bytes]) -> str:
    """Binary Merkle tree root over a list of byte strings."""
    if not items:
        return hashlib.sha256(b"").hexdigest()

    layer = [hashlib.sha256(x).hexdigest() for x in items]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])  # duplicate last node for odd-length layers
        layer = [
            hashlib.sha256((layer[i] + layer[i + 1]).encode()).hexdigest()
            for i in range(0, len(layer), 2)
        ]
    return layer[0]


def _model_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    """Estimate bytes needed to transmit model update (float32)."""
    total = 0
    for t in state_dict.values():
        total += t.numel() * 4  # 4 bytes per float32
    return total


@dataclass
class PoRSProof:
    uav_id: int
    fl_round: int
    gradient_hash: str
    data_merkle_root: str
    compute_time_s: float
    bytes_transmitted: int
    num_samples: int
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uav_id": self.uav_id,
            "fl_round": self.fl_round,
            "gradient_hash": self.gradient_hash,
            "data_merkle_root": self.data_merkle_root,
            "compute_time_s": self.compute_time_s,
            "bytes_transmitted": self.bytes_transmitted,
            "num_samples": self.num_samples,
            "timestamp": self.timestamp,
        }


class PoRSProver:
    """Attached to each FL client; builds a PoRS proof after local training."""

    def build_proof(
        self,
        uav_id: int,
        fl_round: int,
        state_dict: Dict[str, torch.Tensor],
        num_samples: int,
        compute_time_s: float,
        batch_hashes: Optional[List[bytes]] = None,
    ) -> PoRSProof:
        grad_hash = _gradient_hash(state_dict)
        data_root = _merkle_root(batch_hashes or [str(num_samples).encode()])
        tx_bytes = _model_bytes(state_dict)

        return PoRSProof(
            uav_id=uav_id,
            fl_round=fl_round,
            gradient_hash=grad_hash,
            data_merkle_root=data_root,
            compute_time_s=compute_time_s,
            bytes_transmitted=tx_bytes,
            num_samples=num_samples,
            timestamp=time.time(),
        )


class PoRSVerifier:
    """Run by the smart contract / PS to validate incoming proofs."""

    def __init__(
        self,
        min_compute_time: float = 0.1,
        min_samples: int = 5,
        min_bytes: int = 1024,
    ):
        self.min_compute_time = min_compute_time
        self.min_samples = min_samples
        self.min_bytes = min_bytes

    def verify(self, proof: PoRSProof) -> bool:
        if proof.compute_time_s < self.min_compute_time:
            return False
        if proof.num_samples < self.min_samples:
            return False
        if proof.bytes_transmitted < self.min_bytes:
            return False
        if len(proof.gradient_hash) != 64:
            return False
        return True

    def effort_score(self, proof: PoRSProof) -> float:
        """Normalised effort ∈ (0, 1] for reward weighting."""
        s_compute = min(proof.compute_time_s / 30.0, 1.0)
        s_data = min(proof.num_samples / 500.0, 1.0)
        s_bw = min(proof.bytes_transmitted / (5 * 1024 * 1024), 1.0)
        return 0.5 * s_compute + 0.3 * s_data + 0.2 * s_bw
