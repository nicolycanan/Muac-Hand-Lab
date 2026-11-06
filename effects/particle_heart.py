"""Aplica a transformação de uma geometria ao mesmo sistema de partículas."""

from __future__ import annotations

import numpy as np

from config import HEART_BASE_SCALE, MAX_SCALE, MIN_SCALE, PARTICLE_COUNT
from particles.particle_system import ParticleSystem
from particles.shapes import generate_heart_targets
from particles.transforms import rotate_points


class ParticleHeartEffect:
    def __init__(self, count: int = PARTICLE_COUNT):
        self.count = count
        self.system = ParticleSystem(count=count, initial_targets=generate_heart_targets(count, radius=0.9))
        self.center = np.array([0.0, 0.0], dtype=np.float32)
        self.scale = 1.0
        self.rotation = 0.0
        self.visible = True

    @property
    def base_targets(self) -> np.ndarray:
        return self.system.base_targets

    def set_transform(
        self, center: tuple[float, float], scale: float, rotation: float,
        base_targets: np.ndarray | None = None,
    ) -> None:
        self.center = np.asarray(center, dtype=np.float32)
        self.scale = float(np.clip(scale, MIN_SCALE, MAX_SCALE))
        self.rotation = float(rotation)
        base_scale = HEART_BASE_SCALE * self.scale
        geometry = self.base_targets if base_targets is None else base_targets
        transformed = rotate_points(geometry, self.rotation)
        transformed *= base_scale
        transformed += self.center
        self.system.set_targets(transformed)

    def update(self, dt: float) -> None:
        if self.visible:
            self.system.update(dt)

    def draw(self, frame) -> None:
        if self.visible:
            self.system.draw(frame)

    def explode(self) -> None:
        self.system.explode()

    def scatter(self) -> None:
        self.system.scatter()

    def toggle(self) -> None:
        self.visible = not self.visible
