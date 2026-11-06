"""Particle system with converging, scattering, and exploding behavior."""

from __future__ import annotations

import numpy as np

from config import (
    DAMPING,
    PARTICLE_BASE_COLOR,
    PARTICLE_GLOW,
    PARTICLE_GLOW_COLOR,
    PARTICLE_GLOW_STRENGTH,
    PARTICLE_COUNT,
    PARTICLE_OUTLINE,
    PARTICLE_RADIUS,
    SHAPE_FORCE,
)
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
        self._scatter_restore_targets: np.ndarray | None = None

        if initial_targets is None:
            initial_targets = generate_heart_targets(count)
        self.initialize_particles(initial_targets)

    def _sanitize(self, array: np.ndarray) -> np.ndarray:
        array = np.asarray(array, dtype=np.float32)
        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    def initialize_particles(self, targets: np.ndarray) -> None:
        arr = self._sanitize(targets)
        if arr.shape != (self.count, 2):
            raise ValueError(f"Expected targets shape {(self.count, 2)} but got {arr.shape}")
        self.targets = arr.copy()
        self.positions = arr.copy()
        self.velocities = np.zeros_like(arr)
        self.exploding = False
        self.scatter_t = 0.0
        self._scatter_restore_targets = None

    def set_targets(self, targets: np.ndarray) -> None:
        arr = self._sanitize(targets)
        if arr.shape != (self.count, 2):
            raise ValueError(f"Expected targets shape {(self.count, 2)} but got {arr.shape}")
        self.targets = arr.copy()
        if self.positions.shape != self.targets.shape:
            self.positions = arr.copy()
            self.velocities = np.zeros_like(arr)
        else:
            self.positions = self._sanitize(self.positions)
            self.velocities = self._sanitize(self.velocities)

    def update(self, dt: float, force: float = SHAPE_FORCE, damping: float = DAMPING) -> None:
        self.positions = self._sanitize(self.positions)
        self.velocities = self._sanitize(self.velocities)
        self.targets = self._sanitize(self.targets)

        if self.scatter_t > 0.0:
            self.scatter_t -= dt
            if self.scatter_t <= 0.0:
                self.scatter_t = 0.0
                if self._scatter_restore_targets is not None:
                    self.targets = self._scatter_restore_targets.copy()
                    self._scatter_restore_targets = None

        if self.exploding:
            self.velocities *= 0.95
            self.positions += self.velocities * dt * 60.0
            self.scatter_t = max(0.0, self.scatter_t)
            self.positions = self._sanitize(self.positions)
            self.velocities = self._sanitize(self.velocities)
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
        safe_scale = np.zeros_like(capped)
        np.divide(capped, speeds, out=safe_scale, where=speeds > 1e-6)
        self.velocities *= safe_scale

        self.positions += self.velocities * dt * 60.0
        self.positions = self._sanitize(self.positions)
        self.velocities = self._sanitize(self.velocities)

    def scatter(self, radius: float = 18.0) -> None:
        if self._scatter_restore_targets is None:
            self._scatter_restore_targets = self.targets.copy()
        rng = np.random.default_rng()
        noise = rng.normal(0.0, radius, size=self.positions.shape).astype(np.float32)
        self.targets = self.positions + noise
        self.scatter_t = 0.6

    def explode(self, strength: float = 22.0) -> None:
        rng = np.random.default_rng()
        self.velocities += rng.normal(0.0, strength, size=self.positions.shape).astype(np.float32)
        self.exploding = True
        self.scatter_t = 0.7

    def draw(self, frame: np.ndarray, color: tuple[int, int, int] | None = None, radius: int | None = None) -> None:
        if color is None:
            color = PARTICLE_BASE_COLOR
        if radius is None:
            radius = PARTICLE_RADIUS

        for px, py in np.round(self.positions).astype(int):
            if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                cv2.circle(frame, (px, py), radius, color, -1, cv2.LINE_AA)
                if PARTICLE_GLOW and radius > 0:
                    glow_radius = max(radius + 1, 2)
                    cv2.circle(frame, (px, py), glow_radius, PARTICLE_GLOW_COLOR, 1, cv2.LINE_AA)
                elif PARTICLE_OUTLINE and radius > 0:
                    cv2.circle(frame, (px, py), radius, color, 1, cv2.LINE_AA)


# Ensure cv2 is imported for circle drawing.
import cv2  # noqa: E402
