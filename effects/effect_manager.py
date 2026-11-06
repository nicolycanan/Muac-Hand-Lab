"""Aplica os gestos confirmados à forma atual e coordena os modos de interação."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from random import choice
from string import ascii_uppercase

import numpy as np

from config import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    HEART_OVERLOAD_ENABLED,
    HEART_OVERLOAD_HOLD_SECONDS,
    HEART_OVERLOAD_SCALE,
    HEART_OVERLOAD_STRENGTH,
    MAX_SCALE,
    MIN_SCALE,
    PARTICLE_COUNT,
    RECTANGLE_STYLE_SWITCH_HOLD,
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
from effects.rectangle_frame import RectangleFrame, RectangleState
from effects.rectangle_manager import RectangleManager
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


class TransformTarget(StrEnum):
    NONE = "NONE"
    RECTANGLE = "RECTANGLE"
    PARTICLES = "PARTICLES"


class TransformTargetMode(StrEnum):
    AUTO = "AUTO"
    PARTICLES = "PARTICLES"
    RECTANGLE = "RECTANGLE"


@dataclass
class FreeParticleTransformReference:
    pinch_center: np.ndarray
    pinch_distance: float
    particle_center: np.ndarray
    positions: np.ndarray
    smooth_pinch_center: np.ndarray
    smooth_pinch_distance: float


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
        self.rectangle_transform_active = False
        self.transform_target = TransformTarget.NONE
        self.transform_target_mode = TransformTargetMode.AUTO
        self.selected_rectangle_id: int | None = None
        self._transform_session_active = False
        self._transform_release_required = False
        self._heart_overload_elapsed = 0.0
        self._rectangle_style_elapsed = 0.0
        self._rectangle_style_hand: str | None = None
        self._rectangle_style_target_id: int | None = None
        self._rectangle_style_latched = False
        self.free_particle_transform_reference: FreeParticleTransformReference | None = None
        self.rectangle_manager = RectangleManager()
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

    @property
    def selected_rectangle(self) -> RectangleFrame | None:
        return self.rectangle_manager.get(self.selected_rectangle_id)

    @property
    def rectangle_frame(self) -> RectangleFrame | None:
        return (
            self.selected_rectangle
            or self.rectangle_manager.creation
            or self.rectangle_manager.latest
        )

    @property
    def rectangle_delete_pending(self) -> bool:
        rectangle = self.selected_rectangle
        return rectangle is not None and rectangle.delete_pending

    @property
    def rectangle_style_feedback(self) -> tuple[int, str, float] | None:
        rectangle = self.rectangle_manager.get(self._rectangle_style_target_id)
        if rectangle is None:
            rectangle = self.selected_rectangle
        if rectangle is None:
            return None
        progress = (
            min(self._rectangle_style_elapsed / RECTANGLE_STYLE_SWITCH_HOLD, 1.0)
            if self._rectangle_style_hand is not None
            and not self._rectangle_style_latched
            else 0.0
        )
        return rectangle.instance_id, rectangle.fill_style.value, progress

    def _update_rectangle_style_switch(
        self,
        mode: InteractionMode,
        controls,
        hand_keys: tuple[str, ...],
        dt: float,
    ) -> bool:
        if mode != InteractionMode.GRAB or len(controls) != 1 or len(hand_keys) != 1:
            self._rectangle_style_elapsed = 0.0
            self._rectangle_style_hand = None
            self._rectangle_style_target_id = None
            self._rectangle_style_latched = False
            return False

        hand_key = hand_keys[0]
        if self._rectangle_style_latched:
            return True

        point = controls[0].pinch_point
        rectangle = self.rectangle_manager.select_containing(point)
        if rectangle is None:
            self._rectangle_style_elapsed = 0.0
            self._rectangle_style_hand = None
            self._rectangle_style_target_id = None
            return False

        if self._rectangle_style_hand != hand_key:
            self._rectangle_style_elapsed = 0.0
            self._rectangle_style_hand = hand_key
            self._rectangle_style_target_id = rectangle.instance_id
        elif self._rectangle_style_target_id != rectangle.instance_id:
            self._rectangle_style_elapsed = 0.0
            self._rectangle_style_target_id = rectangle.instance_id

        self._rectangle_style_elapsed += max(float(dt), 0.0)
        if self._rectangle_style_elapsed + 1e-9 >= RECTANGLE_STYLE_SWITCH_HOLD:
            changed = self.rectangle_manager.cycle_fill_style(rectangle.instance_id)
            self._rectangle_style_latched = changed is not None
            self._rectangle_style_elapsed = RECTANGLE_STYLE_SWITCH_HOLD
        return True

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

    def cycle_transform_target_mode(self) -> TransformTargetMode:
        modes = tuple(TransformTargetMode)
        index = modes.index(self.transform_target_mode)
        self.transform_target_mode = modes[(index + 1) % len(modes)]
        return self.transform_target_mode

    @property
    def free_particle_transform_scale(self) -> float:
        reference = self.free_particle_transform_reference
        if reference is None:
            return 1.0
        return reference.smooth_pinch_distance / reference.pinch_distance

    def _select_transform_target(self, controls, frame_shape: tuple[int, ...]) -> TransformTarget:
        for rectangle in self.rectangle_manager.instances:
            if rectangle.state == RectangleState.LOCKED:
                rectangle.constrain_geometry(frame_shape)

        if self.transform_target_mode == TransformTargetMode.RECTANGLE:
            rectangle = self.rectangle_manager.select_nearest(
                (controls[0].pinch_point + controls[1].pinch_point) * 0.5
            )
            self.selected_rectangle_id = rectangle.instance_id if rectangle else None
            if rectangle is not None:
                rectangle.constrain_geometry(frame_shape)
            return TransformTarget.RECTANGLE if rectangle else TransformTarget.NONE
        if self.transform_target_mode == TransformTargetMode.PARTICLES:
            self.selected_rectangle_id = None
            return TransformTarget.PARTICLES

        midpoint = (controls[0].pinch_point + controls[1].pinch_point) * 0.5
        rectangle = self.rectangle_manager.select_containing(midpoint)
        if rectangle is not None:
            rectangle.constrain_geometry(frame_shape)
            self.selected_rectangle_id = rectangle.instance_id
            return TransformTarget.RECTANGLE
        self.selected_rectangle_id = None
        return TransformTarget.PARTICLES

    def _update_free_particle_transform(self, first: np.ndarray, second: np.ndarray) -> None:
        center = (first + second) * 0.5
        distance = float(np.linalg.norm(second - first))
        if distance < 8.0:
            self.free_particle_transform_reference = None
            return

        reference = self.free_particle_transform_reference
        if reference is None:
            positions = self.heart.system.positions.copy()
            particle_center = positions.mean(axis=0)
            self.free_particle_transform_reference = FreeParticleTransformReference(
                pinch_center=center.copy(),
                pinch_distance=distance,
                particle_center=particle_center,
                positions=positions,
                smooth_pinch_center=center.copy(),
                smooth_pinch_distance=distance,
            )
            return

        reference.smooth_pinch_center += TRANSFORM_MOVE_SMOOTHING * (
            center - reference.smooth_pinch_center
        )
        reference.smooth_pinch_distance += TRANSFORM_SCALE_SMOOTHING * (
            distance - reference.smooth_pinch_distance
        )

    def _apply_free_particle_transform(self) -> None:
        reference = self.free_particle_transform_reference
        if reference is None:
            return

        scale = np.clip(
            reference.smooth_pinch_distance / reference.pinch_distance,
            MIN_SCALE,
            MAX_SCALE,
        )
        center = (
            reference.particle_center
            + reference.smooth_pinch_center
            - reference.pinch_center
        )
        self.heart.system.positions[:] = center + (
            reference.positions - reference.particle_center
        ) * scale

    def _update_heart_overload(self, dt: float) -> bool:
        eligible = (
            HEART_OVERLOAD_ENABLED
            and self.heart_enabled
            and self.transform_target == TransformTarget.PARTICLES
            and self.transform_active
            and self.transform_scale > HEART_OVERLOAD_SCALE
        )
        if not eligible:
            self._heart_overload_elapsed = 0.0
            return False

        self._heart_overload_elapsed += max(float(dt), 0.0)
        if self._heart_overload_elapsed < HEART_OVERLOAD_HOLD_SECONDS:
            return False

        self.heart.system.break_apart(
            self.transform_center.copy(),
            HEART_OVERLOAD_STRENGTH,
        )
        self.current_shape = None
        self.active_effect = self._shape_effect()
        self.transform_reference = None
        self.free_particle_transform_reference = None
        self._heart_overload_elapsed = 0.0
        self._transform_release_required = True
        self.transform_target = TransformTarget.NONE
        return True

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
        style_switch_reserved = self._update_rectangle_style_switch(
            mode,
            controls,
            controller.confirmation.hands,
            dt,
        )
        grab_hand = (
            controller.confirmation.hands[0]
            if mode == InteractionMode.GRAB and not style_switch_reserved
            else None
        )
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
        transform_gesture_active = (
            mode == InteractionMode.TRANSFORM
            and len(controls) == 2
            and not self._transform_release_required
        )
        if transform_gesture_active:
            if not self._transform_session_active:
                self.transform_target = self._select_transform_target(controls, frame.shape)
            self._transform_session_active = True
        else:
            self._transform_session_active = False
            self.transform_target = TransformTarget.NONE
            selected = self.selected_rectangle
            if selected is not None:
                selected.end_transform()
            self.selected_rectangle_id = None
            if mode != InteractionMode.TRANSFORM:
                self._transform_release_required = False
                self._heart_overload_elapsed = 0.0

        self.rectangle_transform_active = (
            transform_gesture_active
            and self.transform_target == TransformTarget.RECTANGLE
            and self.selected_rectangle is not None
        )
        particle_transform_active = (
            transform_gesture_active and self.transform_target == TransformTarget.PARTICLES
        )
        self.transform_active = self.rectangle_transform_active or particle_transform_active
        if not particle_transform_active:
            self.transform_reference = None
            self.free_particle_transform_reference = None
        if not self.rectangle_transform_active:
            selected = self.selected_rectangle
            if selected is not None:
                selected.end_transform()
        base_mode = InteractionMode.SHAPE if has_shape else InteractionMode.FREE
        if self.transform_active:
            self.interaction_mode = InteractionMode.TRANSFORM
        elif style_switch_reserved:
            self.interaction_mode = base_mode
        else:
            self.interaction_mode = mode if mode in (InteractionMode.REPEL, InteractionMode.GRAB) else base_mode
        self.active_effect = self._shape_effect()
        overload_triggered = False
        if self.rectangle_transform_active:
            controls = [controller.hands[key] for key in controller.confirmation.hands]
            rectangle = self.selected_rectangle
            if len(controls) == 2 and rectangle is not None:
                rectangle.update_transform(
                    controls[0].pinch_point,
                    controls[1].pinch_point,
                    frame.shape,
                    dt,
                )
        elif particle_transform_active:
            self.active_effect = "TRANSFORM"
            self.interaction_mode = InteractionMode.TRANSFORM
            if has_shape:
                self._update_transform(*(hand.pinch_point for hand in controls))
                overload_triggered = self._update_heart_overload(dt)
                if overload_triggered:
                    particle_transform_active = False
                    self.transform_active = False
                    self.interaction_mode = InteractionMode.FREE
            else:
                self._update_free_particle_transform(*(hand.pinch_point for hand in controls))
        elif controller.event == GestureEvent.BREAK:
            system.break_apart(controller.event_point)
        elif mode == InteractionMode.REPEL:
            for hand in controls:
                system.apply_local_repulsion(hand.palm_center, dt=dt, motion=hand.palm_motion)
        elif mode == InteractionMode.GRAB and not style_switch_reserved:
            system.begin_or_update_grab(controls[0].pinch_point)

        if has_shape and not overload_triggered:
            self.heart.update(dt)
        else:
            height, width = frame.shape[:2]
            system.update(dt, force=0.0, damping=0.97, bounds=(width, height))
            if particle_transform_active:
                self._apply_free_particle_transform()
        glitch_sources = self.rectangle_manager.capture_glitch_sources(frame)
        self.draw(frame)
        self.rectangle_manager.update(
            frame,
            controller.hands,
            dt,
            glitch_sources,
        )
        if self.selected_rectangle_id is not None and self.selected_rectangle is None:
            self.selected_rectangle_id = None
            self.rectangle_transform_active = False
            self.transform_active = False
            self.interaction_mode = base_mode
        if self.rectangle_manager.has_visible:
            self.active_effect = "RECTANGLE_FRAME"
        return self.active_effect

    def draw(self, frame) -> None:
        self.heart.draw(frame)
