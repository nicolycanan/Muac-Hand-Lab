"""Simulação vetorizada da atração, repulsão, captura e dispersão das partículas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
    DAMPING,
    GRAB_CLOUD_RADIUS,
    GRAB_FOLLOW_FORCE,
    GRAB_FORCE,
    GRAB_RADIUS,
    PARTICLE_BASE_COLOR,
    PARTICLE_COUNT,
    PARTICLE_RADIUS,
    REPEL_FORCE,
    REPEL_RADIUS,
    SHAPE_FORCE,
    SWIPE_FORCE,
)
from particles.shapes import generate_heart_targets


@dataclass
class GrabCloud:
    mask: np.ndarray
    offsets: np.ndarray
    center: np.ndarray
    point: np.ndarray
    motion: np.ndarray


class ParticleSystem:
    def __init__(self, count: int = PARTICLE_COUNT, initial_targets: np.ndarray | None = None):
        self.count = count
        self.positions = np.zeros((count, 2), dtype=np.float32)
        self.velocities = np.zeros((count, 2), dtype=np.float32)
        self.targets = np.zeros((count, 2), dtype=np.float32)
        self.colors = np.tile(np.array(PARTICLE_BASE_COLOR, dtype=np.uint8), (count, 1))
        self._acceleration = np.empty_like(self.positions)
        self._speed_scale = np.empty(count, dtype=np.float32)
        self.max_speed = 8.0
        self._break_remaining = 0.0
        self.scatter_t = 0.0
        self._scatter_targets: np.ndarray | None = None
        self._grab: GrabCloud | None = None

        if initial_targets is None:
            initial_targets = generate_heart_targets(count)
        self.base_targets = self._validated_targets(initial_targets)
        self.base_targets.setflags(write=False)
        self.initialize_particles(self.base_targets)

    @property
    def exploding(self) -> bool:
        return self._break_remaining > 0.0

    def _validated_targets(self, targets: np.ndarray) -> np.ndarray:
        arr = np.asarray(targets, dtype=np.float32)
        if arr.shape != (self.count, 2):
            raise ValueError(f"Expected targets shape {(self.count, 2)} but got {arr.shape}")
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def initialize_particles(self, targets: np.ndarray) -> None:
        self.set_targets(targets)
        self.end_grab()
        self.positions[:] = self.targets
        self.velocities.fill(0.0)
        self._break_remaining = 0.0
        self.scatter_t = 0.0
        self._scatter_targets = None

    def set_targets(self, targets: np.ndarray) -> None:
        self.targets[:] = self._validated_targets(targets)

    def update(self, dt: float, force: float = SHAPE_FORCE, damping: float = DAMPING) -> None:
        if not np.isfinite(dt) or dt <= 0.0:
            return

        # Divide quadros longos em subpassos para estabilizar a mola quando o FPS oscila.
        frames = min(dt, 0.05) * 60.0
        steps = int(np.ceil(frames))
        step = frames / steps
        for _ in range(steps):
            elapsed = step / 60.0
            if self.exploding:
                self.velocities *= 0.95 ** step
                self._break_remaining = max(0.0, self._break_remaining - elapsed)
            else:
                target = self.targets if self._scatter_targets is None else self._scatter_targets
                np.subtract(target, self.positions, out=self._acceleration)
                self._acceleration *= force * step
                self.velocities += self._acceleration
                self.velocities *= damping ** step

                np.einsum("ij,ij->i", self.velocities, self.velocities, out=self._speed_scale)
                np.sqrt(self._speed_scale, out=self._speed_scale)
                np.maximum(self._speed_scale, self.max_speed, out=self._speed_scale)
                np.divide(self.max_speed, self._speed_scale, out=self._speed_scale)
                self.velocities *= self._speed_scale[:, None]

            if self._grab is not None:
                self.velocities[self._grab.mask] = 0.0
            np.multiply(self.velocities, step, out=self._acceleration)
            self.positions += self._acceleration

            self.scatter_t = max(0.0, self.scatter_t - elapsed)
            if self.scatter_t == 0.0:
                self._scatter_targets = None

    def scatter(self, radius: float = 18.0) -> None:
        noise = np.random.default_rng().normal(0.0, radius, size=self.positions.shape).astype(np.float32)
        self._scatter_targets = self.positions + noise
        self.scatter_t = 0.6

    def explode(self, strength: float = 22.0) -> None:
        self.break_apart(self.positions.mean(axis=0), strength)

    def break_apart(self, point: np.ndarray, strength: float = 8.0) -> None:
        self.end_grab()
        self._scatter_targets = None
        self.scatter_t = 0.0
        rng = np.random.default_rng()
        delta = self.positions - point
        distances = np.linalg.norm(delta, axis=1, keepdims=True)
        directions = delta / np.maximum(distances, 1e-6)
        at_origin = distances[:, 0] <= 1e-6
        angles = rng.uniform(0, 2 * np.pi, at_origin.sum())
        directions[at_origin] = np.column_stack((np.cos(angles), np.sin(angles)))
        magnitude = rng.uniform(0.375 * strength, strength, size=(self.count, 1))
        self.velocities += directions * magnitude + rng.uniform(-1.0, 1.0, size=delta.shape)
        self._break_remaining = 0.22

    def apply_local_repulsion(
        self,
        point: np.ndarray | tuple[float, float],
        radius: float = REPEL_RADIUS,
        strength: float = REPEL_FORCE,
        dt: float = 1.0 / 60.0,
        motion: np.ndarray | None = None,
    ) -> None:
        delta = self.positions - point
        squared = np.einsum("ij,ij->i", delta, delta)
        active = (squared > 1e-6) & (squared < radius ** 2)
        distances = np.sqrt(squared[active, None])
        falloff = 1.0 - distances / radius
        push = strength * delta[active] / distances
        if motion is not None:
            push += SWIPE_FORCE * motion
        self.velocities[active] += push * falloff * min(max(dt, 0.0), 0.05) * 60.0

    def begin_or_update_grab(
        self,
        point: np.ndarray | tuple[float, float],
        radius: float = GRAB_RADIUS,
    ) -> None:
        current = np.asarray(point, dtype=np.float32)
        grab = self._grab
        if grab is None:
            delta = self.positions - current
            mask = np.einsum("ij,ij->i", delta, delta) <= radius ** 2
            if mask.any():
                captured = self.positions[mask]
                center = captured.mean(axis=0)
                self._grab = GrabCloud(mask, captured - center, center, current.copy(), np.zeros_like(current))
            return

        grab.motion = current - grab.point
        predicted = grab.center + grab.motion
        correction = current - predicted
        distance = float(np.linalg.norm(correction))
        # Limita a aproximação da nuvem à pinça e preserva a distância entre as partículas.
        pull = GRAB_FORCE * min(1.0, GRAB_CLOUD_RADIUS / max(distance, 1e-6))
        grab.center = predicted + pull * correction
        grab.point[:] = current
        self.positions[grab.mask] = grab.center + grab.offsets

    def end_grab(self) -> None:
        if self._grab is not None:
            self.velocities[self._grab.mask] = GRAB_FOLLOW_FORCE * self._grab.motion
            self._grab = None

    def draw(self, frame: np.ndarray, color: tuple[int, int, int] | None = None, radius: int | None = None) -> None:
        radius = PARTICLE_RADIUS if radius is None else radius
        pixels = np.rint(self.positions).astype(np.int32)
        x, y = pixels.T
        height, width = frame.shape[:2]
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                px, py = x + dx, y + dy
                visible = (px >= 0) & (px < width) & (py >= 0) & (py < height)
                frame[py[visible], px[visible]] = self.colors[visible] if color is None else color
