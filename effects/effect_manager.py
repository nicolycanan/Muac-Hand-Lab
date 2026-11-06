"""Effect manager and global state decisions."""

from __future__ import annotations

import numpy as np

from config import MAX_SCALE, MIN_SCALE, PARTICLE_COUNT, START_WITH_HEART
from effects.particle_heart import ParticleHeartEffect


class EffectManager:
    def __init__(self):
        self.heart = ParticleHeartEffect(PARTICLE_COUNT)
        self.heart_enabled = START_WITH_HEART
        self.active_effect = "PARTICLE_HEART" if self.heart_enabled else "IDLE"
        self.transform_center = np.array([0.0, 0.0], dtype=np.float32)
        self.transform_scale = 1.0
        self.transform_rotation = 0.0
        self.heart.set_transform((320.0, 240.0), 1.0, 0.0)

    def toggle_heart(self) -> None:
        self.heart_enabled = not self.heart_enabled
        self.active_effect = "PARTICLE_HEART" if self.heart_enabled else "IDLE"

    def set_transform(self, center: tuple[float, float], scale: float, rotation: float) -> None:
        self.transform_center = np.asarray(center, dtype=np.float32)
        self.transform_scale = float(max(MIN_SCALE, min(scale, MAX_SCALE)))
        self.transform_rotation = float(rotation)
        self.heart.set_transform(tuple(self.transform_center), self.transform_scale, self.transform_rotation)

    def update(self, frame, hands: list[dict], gesture: str, dt: float) -> str:
        self.active_effect = "IDLE"
        if not self.heart_enabled:
            return self.active_effect

        width = max(frame.shape[1], 1)
        height = max(frame.shape[0], 1)
        center = (width / 2.0, height / 2.0)
        default_scale = 1.0

        if gesture == "TRANSFORM":
            self.active_effect = "TRANSFORM"
            if len(hands) >= 2:
                left = np.asarray(hands[0]["center"], dtype=np.float32)
                right = np.asarray(hands[1]["center"], dtype=np.float32)
                center = (left + right) / 2.0
                scale = np.linalg.norm(right - left) / 180.0
                rotation = np.arctan2(right[1] - left[1], right[0] - left[0])
                self.set_transform((float(center[0]), float(center[1])), scale, rotation)
            else:
                self.set_transform((float(center[0]), float(center[1])), default_scale, self.transform_rotation)
            return self.active_effect

        self.active_effect = "PARTICLE_HEART"
        self.set_transform((float(center[0]), float(center[1])), default_scale, self.transform_rotation)
        if gesture == "OPEN_HAND":
            self.heart.system.scatter()
        elif gesture == "FIST":
            self.heart.system.explode()
        self.heart.update(dt)
        self.heart.draw(frame)
        return self.active_effect

    def draw(self, frame) -> None:
        if self.heart_enabled:
            self.heart.draw(frame)
