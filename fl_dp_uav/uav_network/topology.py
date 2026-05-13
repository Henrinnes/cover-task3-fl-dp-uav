import numpy as np
from typing import List, Tuple

from .node import UAVNode


class UAVNetwork:
    def __init__(
        self,
        num_uavs: int,
        area_size: Tuple[float, float],
        altitude_range: Tuple[float, float],
        rng: np.random.Generator,
    ):
        self.num_uavs = num_uavs
        self.area_size = area_size
        self.altitude_range = altitude_range
        self.rng = rng
        self.nodes: List[UAVNode] = []
        self._init_nodes()

    def _init_nodes(self):
        for i in range(self.num_uavs):
            pos = np.array([
                self.rng.uniform(0, self.area_size[0]),
                self.rng.uniform(0, self.area_size[1]),
                self.rng.uniform(self.altitude_range[0], self.altitude_range[1]),
            ])
            self.nodes.append(UAVNode(node_id=i, position=pos))

    def active_nodes(self) -> List[UAVNode]:
        return [n for n in self.nodes if n.is_active]

    def distance_matrix(self) -> np.ndarray:
        n = len(self.nodes)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self.nodes[i].distance_to(self.nodes[j])
                D[i, j] = D[j, i] = d
        return D
