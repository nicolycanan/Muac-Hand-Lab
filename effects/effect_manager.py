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
        self.transform_reference = None
        self.transform_active = False
        self._last_open_hand_scatter = 0.0
        self.heart.set_transform((320.0, 240.0), 0.82, 0.0)

    def toggle_heart(self) -> None:
        self.heart_enabled = not self.heart_enabled
        self.active_effect = "PARTICLE_HEART" if self.heart_enabled else "IDLE"

    def set_transform(self, center: tuple[float, float], scale: float, rotation: float) -> None:
        self.transform_center = np.asarray(center, dtype=np.float32)
        self.transform_scale = float(max(MIN_SCALE, min(scale, MAX_SCALE)))
        self.transform_rotation = float(rotation)
        self.heart.set_transform(tuple(self.transform_center), self.transform_scale, self.transform_rotation)

    def _capture_transform_reference(self, hands: list[dict]) -> None:
        if len(hands) < 2:
            return
        left = hands[0]
        right = hands[1]
        left_pin = (np.asarray(left["landmarks"][4], dtype=np.float32) + np.asarray(left["landmarks"][8], dtype=np.float32)) / 2.0
        right_pin = (np.asarray(right["landmarks"][4], dtype=np.float32) + np.asarray(right["landmarks"][8], dtype=np.float32)) / 2.0
        center = (left_pin + right_pin) / 2.0
        distance = float(np.linalg.norm(right_pin - left_pin))
        angle = float(np.arctan2(right_pin[1] - left_pin[1], right_pin[0] - left_pin[0]))
        self.transform_reference = {
            "hand_center": center,
            "hand_distance": max(distance, 1e-6),
            "hand_angle": angle,
            "shape_center": np.array(self.transform_center, dtype=np.float32),
            "shape_scale": float(self.transform_scale),
            "shape_rotation": float(self.transform_rotation),
        }

    def update(self, frame, hands: list[dict], gesture: str, dt: float) -> str:
        if not self.heart_enabled:
            self.active_effect = "IDLE"
            return self.active_effect

        width = max(frame.shape[1], 1)
        height = max(frame.shape[0], 1)
        center = (width / 2.0, height / 2.0)
        default_scale = 1.0

        if gesture == "TRANSFORM":
            self.active_effect = "TRANSFORM"
            if len(hands) >= 2:
                left = hands[0]
                right = hands[1]
                left_pin = (np.asarray(left["landmarks"][4], dtype=np.float32) + np.asarray(left["landmarks"][8], dtype=np.float32)) / 2.0
                right_pin = (np.asarray(right["landmarks"][4], dtype=np.float32) + np.asarray(right["landmarks"][8], dtype=np.float32)) / 2.0
                current_center = (left_pin + right_pin) / 2.0
                current_distance = float(np.linalg.norm(right_pin - left_pin))
                current_angle = float(np.arctan2(right_pin[1] - left_pin[1], right_pin[0] - left_pin[0]))
                if self.transform_reference is None or not self.transform_active:
                    self._capture_transform_reference(hands)
                if self.transform_reference is not None:
                    center_delta = current_center - self.transform_reference["hand_center"]
                    base_center = self.transform_reference["shape_center"]
                    new_center = base_center + center_delta
                    scale_delta = current_distance / max(self.transform_reference["hand_distance"], 1e-6)
                    new_scale = self.transform_reference["shape_scale"] * scale_delta
                    angle_delta = current_angle - self.transform_reference["hand_angle"]
                    new_rotation = self.transform_reference["shape_rotation"] + angle_delta
                    self.set_transform((float(new_center[0]), float(new_center[1])), new_scale, new_rotation)
                    self.transform_active = True
                else:
                    self.set_transform((float(center[0]), float(center[1])), default_scale, self.transform_rotation)
            self.heart.update(dt)
            self.heart.draw(frame)
            return self.active_effect

        self.active_effect = "PARTICLE_HEART"
        self.transform_active = False
        self.set_transform((float(center[0]), float(center[1])), default_scale, self.transform_rotation)
        if gesture == "OPEN_HAND":
            if self.heart.system.scatter_t <= 0.05:
                self.heart.system.scatter(7.0)
        elif gesture == "FIST":
            self.heart.system.explode()
        self.heart.update(dt)
        self.heart.draw(frame)
        return self.active_effect

    def draw(self, frame) -> None:
        if self.heart_enabled:
            self.heart.draw(frame)
