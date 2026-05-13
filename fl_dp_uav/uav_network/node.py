import numpy as np
from dataclasses import dataclass, field


@dataclass
class UAVNode:
    node_id: int
    position: np.ndarray          # [x, y, z] metres
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tx_power_dbm: float = 23.0
    is_active: bool = True

    def distance_to(self, other: "UAVNode") -> float:
        return float(np.linalg.norm(self.position - other.position))
