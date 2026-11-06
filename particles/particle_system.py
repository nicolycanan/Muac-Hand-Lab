"""Particle system with converging, scattering, and exploding behavior."""

from __future__ import annotations

import numpy as np

from config import DAMPING, PARTICLE_COUNT, SHAPE_FORCE
from particles.shapes import generate_heart_targets


class ParticleSystem:
    def __init__(self, count: int = PARTICLE_COUNT, initial_targets: np.ndarray | None = None):
        self.count = count
        self.positions = np.zeros((count, 2), dtype=np.float32)
        self.velocities = np.zeros((count, 2), dtype=np.float32)
        self.targets = np.zeros((count, 2), dtype=np.float32)
        self.max_speed = 2.5
        self.exploding = False
        self.scatter_t = 0.0

        if initial_targets is None:
            initial_targets = generate_heart_targets(count)
        self.set_targets(initial_targets)

    def set_targets(self, targets: np.ndarray) -> None:
        arr = np.asarray(targets, dtype=np.float32)
        if arr.shape != (self.count, 2):
            raise ValueError(f"Expected targets shape {(self.count, 2)} but got {arr.shape}")
        self.targets = arr.copy()
        self.positions = arr.copy()
        self.velocities.fill(0.0)

    def update(self, dt: float, force: float = SHAPE_FORCE, damping: float = DAMPING) -> None:
        if self.exploding:
            self.velocities *= 0.95
            self.positions += self.velocities * dt * 60.0
            self.scatter_t -= dt
            if self.scatter_t <= 0.0:
                self.exploding = False
            return

        delta = self.targets - self.positions
        acceleration = delta * force
        self.velocities += acceleration * dt * 60.0
        self.velocities *= damping

        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        capped = np.clip(speeds, 0.0, self.max_speed)
        self.velocities *= np.where(speeds > 1e-6, capped / speeds, 1.0)

        self.positions += self.velocities * dt * 60.0

    def scatter(self, radius: float = 18.0) -> None:
        rng = np.random.default_rng()
        noise = rng.normal(0.0, radius, size=self.positions.shape).astype(np.float32)
        self.targets = self.positions + noise
        self.scatter_t = 0.6

    def explode(self, strength: float = 22.0) -> None:
        rng = np.random.default_rng()
        self.velocities += rng.normal(0.0, strength, size=self.positions.shape).astype(np.float32)
        self.exploding = True
        self.scatter_t = 0.7

    def draw(self, frame: np.ndarray, color: tuple[int, int, int] = (0, 184, 255), radius: int = 2) -> None:
        for index, (x, y) in enumerate(self.positions):
            px = int(round(x))
            py = int(round(y))
            if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                r = 150 + int(60 * np.sin(index * 0.15))
                g = 60 + int(80 * np.cos(index * 0.1))
                b = 190 + int(50 * np.sin(index * 0.12 + 1.0))
                c = (int(np.clip(b, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(r, 0, 255)))
                cv2.circle(frame, (px, py), radius, c, -1, cv2.LINE_AA)
                if radius > 1:
                    cv2.circle(frame, (px, py), radius + 1, (255, 200, 210), 1, cv2.LINE_AA)


# Ensure cv2 is imported for circle drawing.
import cv2  # noqa: E402
