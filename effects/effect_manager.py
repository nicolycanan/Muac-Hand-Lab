"""Aplica os gestos confirmados ao coração e coordena os modos de interação."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    MAX_SCALE,
    MIN_SCALE,
    PARTICLE_COUNT,
    START_WITH_HEART,
    TRANSFORM_MOVE_SMOOTHING,
    TRANSFORM_ROTATE_SMOOTHING,
    TRANSFORM_ROTATE_STEP_MAX,
    TRANSFORM_SCALE_SMOOTHING,
    TRANSFORM_SCALE_STEP_MAX,
    TRANSFORM_SCALE_STEP_MIN,
)
from core.gesture_controller import GestureController, GestureEvent, InteractionMode
from effects.particle_heart import ParticleHeartEffect


@dataclass
class TransformReference:
    pinch_center: np.ndarray
    pinch_distance: float
    shape_center: np.ndarray
    shape_scale: float
    shape_rotation: float
    previous_angle: float
    smooth_center: np.ndarray
    smooth_distance: float
    angle_delta: float = 0.0


class EffectManager:
    def __init__(self):
        self.heart = ParticleHeartEffect(PARTICLE_COUNT)
        self.heart_enabled = START_WITH_HEART
        self.active_effect = "PARTICLE_HEART" if self.heart_enabled else "IDLE"
        self.interaction_mode = InteractionMode.SHAPE if self.heart_enabled else InteractionMode.IDLE
        self.transform_center = np.array([CAMERA_WIDTH / 2, CAMERA_HEIGHT / 2], dtype=np.float32)
        self.transform_scale = 0.82
        self.transform_rotation = 0.0
        self.transform_reference: TransformReference | None = None
        self._grab_hand: str | None = None
        self.transform_active = False
        self.heart.set_transform(tuple(self.transform_center), self.transform_scale, 0.0)
        system = self.heart.system
        system.positions[:] = system.targets + np.random.default_rng(0).normal(0, 8, size=system.targets.shape)

    @property
    def current_shape(self) -> str | None:
        return "HEART" if self.heart_enabled else None

    def toggle_heart(self) -> None:
        self.heart_enabled = not self.heart_enabled
        self.active_effect = "PARTICLE_HEART" if self.heart_enabled else "IDLE"

    def set_transform(self, center: tuple[float, float], scale: float, rotation: float) -> None:
        self.transform_center = np.asarray(center, dtype=np.float32)
        self.transform_scale = float(max(MIN_SCALE, min(scale, MAX_SCALE)))
        self.transform_rotation = float(rotation)
        self.heart.set_transform(tuple(self.transform_center), self.transform_scale, self.transform_rotation)

    def _update_transform(self, first: np.ndarray, second: np.ndarray) -> None:
        center = (first + second) * 0.5
        delta = second - first
        distance = float(np.linalg.norm(delta))
        if distance < 8.0:
            # Pinças quase sobrepostas tornam a distância e o ângulo instáveis.
            self.transform_reference = None
            return

        angle = float(np.arctan2(delta[1], delta[0]))
        reference = self.transform_reference
        if reference is None:
            self.transform_reference = TransformReference(
                center, distance, self.transform_center.copy(), self.transform_scale,
                self.transform_rotation, angle, center.copy(), distance,
            )
            return

        reference.smooth_center += TRANSFORM_MOVE_SMOOTHING * (center - reference.smooth_center)
        reference.smooth_distance += TRANSFORM_SCALE_SMOOTHING * (distance - reference.smooth_distance)
        # A linha das pinças tem período de 180°. O menor arco evita saltos
        # quando as mãos trocam de ordem ou atan2 cruza ±π.
        reference.angle_delta += (angle - reference.previous_angle + np.pi / 2) % np.pi - np.pi / 2
        reference.previous_angle = angle

        target_center = reference.shape_center + reference.smooth_center - reference.pinch_center
        target_scale = reference.shape_scale * reference.smooth_distance / reference.pinch_distance
        step = np.clip(target_scale / self.transform_scale, TRANSFORM_SCALE_STEP_MIN, TRANSFORM_SCALE_STEP_MAX)
        angle_error = reference.shape_rotation + reference.angle_delta - self.transform_rotation
        max_step = np.deg2rad(TRANSFORM_ROTATE_STEP_MAX)
        rotation_step = np.clip(angle_error * TRANSFORM_ROTATE_SMOOTHING, -max_step, max_step)
        self.set_transform(target_center, self.transform_scale * step, self.transform_rotation + rotation_step)

    def update(self, frame, controller: GestureController, dt: float) -> str:
        system = self.heart.system
        mode = controller.mode
        controls = [controller.hands[key] for key in controller.confirmation.hands]
        grab_hand = controller.confirmation.hands[0] if mode == InteractionMode.GRAB and self.heart_enabled else None
        if grab_hand != self._grab_hand:
            system.end_grab()
        self._grab_hand = grab_hand
        if controller.event == GestureEvent.SELECT_HEART:
            self.heart_enabled = True
            self.heart.visible = True

        self.transform_active = self.heart_enabled and mode == InteractionMode.TRANSFORM
        if not self.transform_active:
            self.transform_reference = None
        if not self.heart_enabled:
            self.interaction_mode = InteractionMode.IDLE
            self.active_effect = "IDLE"
            return self.active_effect

        self.interaction_mode = InteractionMode.SHAPE if mode == InteractionMode.IDLE else mode
        self.active_effect = "PARTICLE_HEART"
        if self.transform_active:
            self.active_effect = "TRANSFORM"
            self._update_transform(*(hand.pinch_point for hand in controls))
        elif controller.event == GestureEvent.BREAK:
            system.break_apart(controller.event_point)
        elif mode == InteractionMode.REPEL:
            for hand in controls:
                system.apply_local_repulsion(hand.palm_center, dt=dt, motion=hand.palm_motion)
        elif mode == InteractionMode.GRAB:
            system.begin_or_update_grab(controls[0].pinch_point)

        self.heart.update(dt)
        self.heart.draw(frame)
        return self.active_effect

    def draw(self, frame) -> None:
        if self.heart_enabled:
            self.heart.draw(frame)
