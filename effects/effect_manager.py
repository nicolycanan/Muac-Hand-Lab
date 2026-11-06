"""Aplica os gestos confirmados à forma atual e coordena os modos de interação."""

from __future__ import annotations

from dataclasses import dataclass
from random import choice
from string import ascii_uppercase

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
from particles.shapes import generate_letter_targets


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
    def __init__(self, start_with_heart: bool = START_WITH_HEART):
        self.heart = ParticleHeartEffect(PARTICLE_COUNT)
        self.current_shape = "HEART" if start_with_heart else None
        self.active_effect = self._shape_effect()
        self.interaction_mode = InteractionMode.SHAPE if self.heart_enabled else InteractionMode.FREE
        self.transform_center = np.array([CAMERA_WIDTH / 2, CAMERA_HEIGHT / 2], dtype=np.float32)
        self.transform_scale = 0.82
        self.transform_rotation = 0.0
        self.transform_reference: TransformReference | None = None
        self._grab_hand: str | None = None
        self.transform_active = False
        self.heart.set_transform(tuple(self.transform_center), self.transform_scale, 0.0)
        system = self.heart.system
        rng = np.random.default_rng()
        if self.heart_enabled:
            system.positions[:] = system.targets + rng.normal(0, 8, size=system.targets.shape)
        else:
            system.positions[:] = rng.uniform((0, 0), (CAMERA_WIDTH, CAMERA_HEIGHT), size=system.positions.shape)

    @property
    def heart_enabled(self) -> bool:
        return self.current_shape == "HEART"

    def _shape_effect(self) -> str:
        if self.current_shape is None:
            return "FREE"
        return "PARTICLE_HEART" if self.heart_enabled else "PARTICLE_LETTER"

    def _select_shape(self, shape: str | None) -> None:
        if shape == self.current_shape:
            return
        self.current_shape = shape
        if shape is not None:
            self.set_transform(tuple(self.transform_center), self.transform_scale, self.transform_rotation)
        self.active_effect = self._shape_effect()

    def _select_random_letter(self) -> None:
        options = [letter for letter in ascii_uppercase if letter != self.current_shape]
        self._select_shape(choice(options))

    def toggle_heart(self) -> None:
        self._select_shape(None if self.heart_enabled else "HEART")

    def set_transform(self, center: tuple[float, float], scale: float, rotation: float) -> None:
        self.transform_center = np.asarray(center, dtype=np.float32)
        self.transform_scale = float(max(MIN_SCALE, min(scale, MAX_SCALE)))
        self.transform_rotation = float(rotation)
        geometry = None
        if self.current_shape is not None and not self.heart_enabled:
            geometry = generate_letter_targets(self.current_shape, self.heart.count)
        self.heart.set_transform(tuple(self.transform_center), self.transform_scale, self.transform_rotation, geometry)

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
        grab_hand = controller.confirmation.hands[0] if mode == InteractionMode.GRAB else None
        if grab_hand != self._grab_hand:
            system.end_grab()
        self._grab_hand = grab_hand
        if controller.event == GestureEvent.SELECT_HEART:
            self._select_shape("HEART")
            self.heart.visible = True
        elif controller.event == GestureEvent.SELECT_LETTER or (
            controller.event == GestureEvent.CHANGE_LETTER
            and self.current_shape is not None and not self.heart_enabled
        ):
            self._select_random_letter()

        has_shape = self.current_shape is not None
        self.transform_active = has_shape and mode == InteractionMode.TRANSFORM
        if not self.transform_active:
            self.transform_reference = None
        base_mode = InteractionMode.SHAPE if has_shape else InteractionMode.FREE
        self.interaction_mode = mode if mode in (InteractionMode.REPEL, InteractionMode.GRAB) else base_mode
        self.active_effect = self._shape_effect()
        if self.transform_active:
            self.active_effect = "TRANSFORM"
            self.interaction_mode = InteractionMode.TRANSFORM
            self._update_transform(*(hand.pinch_point for hand in controls))
        elif controller.event == GestureEvent.BREAK:
            system.break_apart(controller.event_point)
        elif mode == InteractionMode.REPEL:
            for hand in controls:
                system.apply_local_repulsion(hand.palm_center, dt=dt, motion=hand.palm_motion)
        elif mode == InteractionMode.GRAB:
            system.begin_or_update_grab(controls[0].pinch_point)

        if has_shape:
            self.heart.update(dt)
        else:
            height, width = frame.shape[:2]
            system.update(dt, force=0.0, damping=0.97, bounds=(width, height))
        self.draw(frame)
        return self.active_effect

    def draw(self, frame) -> None:
        self.heart.draw(frame)
