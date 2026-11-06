"""Owns independent rectangle figures and routes creation and selection."""

from __future__ import annotations

import numpy as np

from config import RECTANGLE_MAX_INSTANCES, RECTANGLE_TRACKING_GRACE_FRAMES
from core.gesture_controller import HandGesture
from effects.rectangle_frame import RectangleFrame, RectangleState
from effects.rectangle_style import RectangleFillStyle


class RectangleManager:
    def __init__(self, max_instances: int = RECTANGLE_MAX_INSTANCES) -> None:
        if max_instances < 1:
            raise ValueError("max_instances must be at least one")
        self.max_instances = max_instances
        self.instances: list[RectangleFrame] = []
        self.creation_id: int | None = None
        self._next_id = 1
        self._gesture_latched = False
        self._no_hand_frames = 0
        self._non_l_frames = 0
        self._limit_notice_elapsed = 0.0
        self._last_rectangle: RectangleFrame | None = None

    @property
    def count(self) -> int:
        return len(self.instances)

    @property
    def limit_reached(self) -> bool:
        return self._limit_notice_elapsed > 0.0

    @property
    def creation(self) -> RectangleFrame | None:
        return self.get(self.creation_id)

    @property
    def has_visible(self) -> bool:
        return any(rectangle.visible for rectangle in self.instances)

    @property
    def latest(self) -> RectangleFrame | None:
        if self.instances:
            return self.instances[-1]
        return self._last_rectangle

    def get(self, instance_id: int | None) -> RectangleFrame | None:
        if instance_id is None:
            return None
        return next(
            (rectangle for rectangle in self.instances if rectangle.instance_id == instance_id),
            None,
        )

    def select_containing(self, point: np.ndarray) -> RectangleFrame | None:
        return next(
            (rectangle for rectangle in reversed(self.instances) if rectangle.contains(point)),
            None,
        )

    def cycle_fill_style(self, instance_id: int) -> RectangleFrame | None:
        rectangle = self.get(instance_id)
        if rectangle is None or rectangle.state != RectangleState.LOCKED:
            return None
        rectangle.cycle_fill_style()
        return rectangle

    def capture_glitch_sources(self, frame: np.ndarray) -> dict[int, np.ndarray]:
        sources = {}
        for rectangle in self.instances:
            if rectangle.visible and rectangle.fill_style == RectangleFillStyle.GLITCH:
                source_roi = rectangle.capture_source_roi(frame)
                if source_roi is not None:
                    sources[rectangle.instance_id] = source_roi
        return sources

    def select_nearest(self, point: np.ndarray) -> RectangleFrame | None:
        candidates = [
            (index, rectangle)
            for index, rectangle in enumerate(self.instances)
            if rectangle.state == RectangleState.LOCKED and rectangle.points is not None
        ]
        if not candidates:
            return None

        def distance_to_bounds(rectangle: RectangleFrame) -> float:
            minimum = np.minimum(rectangle.points[0], rectangle.points[1])
            maximum = np.maximum(rectangle.points[0], rectangle.points[1])
            offset = np.maximum(np.maximum(minimum - point, point - maximum), 0.0)
            return float(np.dot(offset, offset))

        _, selected = min(
            candidates,
            key=lambda item: (distance_to_bounds(item[1]), -item[0]),
        )
        return selected

    def update(
        self,
        frame: np.ndarray,
        hands: dict[str, HandGesture],
        dt: float,
        glitch_sources: dict[int, np.ndarray] | None = None,
    ) -> None:
        dt = max(float(dt), 0.0)
        self._limit_notice_elapsed = max(0.0, self._limit_notice_elapsed - dt)
        recognized = RectangleFrame._valid_hands(hands)
        gesture_active = len(recognized) == 2

        if gesture_active:
            self._no_hand_frames = 0
            self._non_l_frames = 0
            if not self._gesture_latched:
                self._gesture_latched = True
                if self.count < self.max_instances:
                    rectangle = RectangleFrame(instance_id=self._next_id)
                    self._next_id += 1
                    self.instances.append(rectangle)
                    self.creation_id = rectangle.instance_id
                else:
                    self._limit_notice_elapsed = 2.0
        elif hands:
            self._no_hand_frames = 0
            self._non_l_frames += 1
            if self._non_l_frames > RECTANGLE_TRACKING_GRACE_FRAMES:
                self._gesture_latched = False
        else:
            self._no_hand_frames += 1
            if self._no_hand_frames > RECTANGLE_TRACKING_GRACE_FRAMES:
                self._gesture_latched = False

        for rectangle in tuple(self.instances):
            rectangle_hands = hands if rectangle.instance_id == self.creation_id else {}
            rectangle.update(
                frame,
                rectangle_hands,
                dt,
                (glitch_sources or {}).get(rectangle.instance_id),
            )
            if rectangle.state == RectangleState.IDLE:
                self.instances.remove(rectangle)
                self._last_rectangle = rectangle
                if self.creation_id == rectangle.instance_id:
                    self.creation_id = None
            elif rectangle.state == RectangleState.LOCKED and self.creation_id == rectangle.instance_id:
                self.creation_id = None
            elif rectangle.state == RectangleState.CLOSING and not rectangle.visible:
                rectangle.reset()
                self.instances.remove(rectangle)
                self._last_rectangle = rectangle
