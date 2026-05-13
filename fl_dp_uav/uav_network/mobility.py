import numpy as np
from typing import Dict, List, Tuple

from .node import UAVNode


class RandomWaypointMobility:
    """3D random waypoint mobility model for UAVs."""

    def __init__(
        self,
        area_size: Tuple[float, float],
        altitude_range: Tuple[float, float],
        max_speed: float,
        pause_time: float,
        rng: np.random.Generator,
    ):
        self.area_size = area_size
        self.altitude_range = altitude_range
        self.max_speed = max_speed
        self.pause_time = pause_time
        self.rng = rng
        self._waypoints: Dict[int, np.ndarray] = {}
        self._pause_timers: Dict[int, float] = {}

    def _new_waypoint(self) -> np.ndarray:
        return np.array([
            self.rng.uniform(0, self.area_size[0]),
            self.rng.uniform(0, self.area_size[1]),
            self.rng.uniform(self.altitude_range[0], self.altitude_range[1]),
        ])

    def step(self, nodes: List[UAVNode], dt: float = 1.0):
        for node in nodes:
            nid = node.node_id
            if nid not in self._waypoints:
                self._waypoints[nid] = self._new_waypoint()
                self._pause_timers[nid] = 0.0

            if self._pause_timers[nid] > 0:
                self._pause_timers[nid] -= dt
                node.velocity = np.zeros(3)
                continue

            direction = self._waypoints[nid] - node.position
            dist = np.linalg.norm(direction)

            if dist < 2.0:
                self._waypoints[nid] = self._new_waypoint()
                self._pause_timers[nid] = self.pause_time
                node.velocity = np.zeros(3)
            else:
                speed = self.rng.uniform(1.0, self.max_speed)
                node.velocity = (direction / dist) * speed
                node.position = node.position + node.velocity * dt
                node.position[0] = np.clip(node.position[0], 0, self.area_size[0])
                node.position[1] = np.clip(node.position[1], 0, self.area_size[1])
                node.position[2] = np.clip(
                    node.position[2], self.altitude_range[0], self.altitude_range[1]
                )
