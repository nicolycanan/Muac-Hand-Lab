"""Particle Heart effect wrapper around a particle system."""

from __future__ import annotations

import numpy as np

from config import PARTICLE_COUNT
from particles.particle_system import ParticleSystem
from particles.shapes import generate_heart_targets
from particles.transforms import rotate_points, scale_points, translate_points


class ParticleHeartEffect:
    def __init__(self, count: int = PARTICLE_COUNT):
        self.count = count
        self.base_targets = generate_heart_targets(count)
        self.system = ParticleSystem(count=count, initial_targets=self.base_targets.copy())
        self.center = np.array([0.0, 0.0], dtype=np.float32)
        self.scale = 1.0
        self.rotation = 0.0
        self.visible = True

    def set_transform(self, center: tuple[float, float], scale: float, rotation: float) -> None:
        self.center = np.asarray(center, dtype=np.float32)
        self.scale = float(scale)
        self.rotation = float(rotation)
        base_scale = 180.0 * max(0.35, self.scale)
        transformed = self.base_targets.copy() * base_scale
        transformed = rotate_points(transformed, self.rotation, center=(0.0, 0.0))
        transformed = translate_points(transformed, self.center[0], self.center[1])
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
