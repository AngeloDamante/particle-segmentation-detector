"""Define class Particle"""

from typing import Tuple


class Particle:
    def __init__(self, t: int, x: float, y: float, z: float) -> None:
        self.t = t
        self.x = x
        self.y = y
        self.z = z

    def get_coords(self) -> Tuple[int, float, float, float]:
        return self.t, self.x, self.y, self.z
