"""Two-hand L gesture recognition state and cyan rectangle rendering."""

from __future__ import annotations

from enum import StrEnum

import cv2
import numpy as np

from config import (
    RECTANGLE_ARM_DISTANCE_PALMS,
    RECTANGLE_COLOR,
    RECTANGLE_FADE_SECONDS,
    RECTANGLE_POINT_SMOOTHING,
    RECTANGLE_START_SEPARATION_PALMS,
    RECTANGLE_THICKNESS,
    RECTANGLE_TRACKING_GRACE_FRAMES,
)
from core.gesture_controller import HandGesture


class RectangleState(StrEnum):
    IDLE = "IDLE"
    ARMED = "ARMED"
    DRAWING = "DRAWING"
    CLOSING = "CLOSING"


class RectangleFrame:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.state = RectangleState.IDLE
        self.points: tuple[np.ndarray, np.ndarray] | None = None
        self._arm_distance = 0.0
        self._tracking_lost_frames = 0
        self._closing_elapsed = 0.0
        self.alpha = 0.0

    @property
    def visible(self) -> bool:
        return self.state in (RectangleState.DRAWING, RectangleState.CLOSING) and self.alpha > 0.0

    @staticmethod
    def _valid_hands(hands: dict[str, HandGesture]) -> list[HandGesture]:
        return [
            hand for _, hand in sorted(hands.items())
            if hand.rectangle_point is not None
        ]

    @staticmethod
    def _distance_ratio(first: HandGesture, second: HandGesture) -> float:
        distance = float(np.linalg.norm(first.rectangle_point - second.rectangle_point))
        palm = (first.palm_size + second.palm_size) * 0.5
        return distance / max(palm, 1.0)

    def _smooth_points(self, first: HandGesture, second: HandGesture) -> None:
        target = (first.rectangle_point.copy(), second.rectangle_point.copy())
        if self.points is None:
            self.points = target
            return
        smoothing = RECTANGLE_POINT_SMOOTHING
        self.points = tuple(
            previous + smoothing * (current - previous)
            for previous, current in zip(self.points, target)
        )

    def _begin_closing(self) -> None:
        if self.state == RectangleState.ARMED:
            self.reset()
        elif self.state == RectangleState.DRAWING:
            self.state = RectangleState.CLOSING
            self._closing_elapsed = 0.0

    def _draw(self, frame: np.ndarray, dt: float) -> None:
        if self.state == RectangleState.CLOSING:
            self._closing_elapsed += dt
            self.alpha = max(0.0, 1.0 - self._closing_elapsed / RECTANGLE_FADE_SECONDS)
        elif self.state == RectangleState.DRAWING:
            self.alpha = 1.0

        if self.state == RectangleState.CLOSING and self.alpha <= 0.0:
            self.reset()
            return
        if not self.visible or self.points is None:
            return

        height, width = frame.shape[:2]
        corners = [
            (
                int(np.clip(np.rint(point[0]), 0, width - 1)),
                int(np.clip(np.rint(point[1]), 0, height - 1)),
            )
            for point in self.points
        ]
        overlay = frame.copy()
        cv2.rectangle(overlay, corners[0], corners[1], RECTANGLE_COLOR, RECTANGLE_THICKNESS)
        cv2.addWeighted(overlay, self.alpha, frame, 1.0 - self.alpha, 0, dst=frame)

    def update(self, frame: np.ndarray, hands: dict[str, HandGesture], dt: float) -> None:
        dt = max(float(dt), 0.0)
        if self.state == RectangleState.CLOSING:
            self._draw(frame, dt)
            return

        recognized = self._valid_hands(hands)
        if len(recognized) == 2:
            self._tracking_lost_frames = 0
            first, second = recognized
            distance = self._distance_ratio(first, second)
            if self.state == RectangleState.IDLE:
                if distance <= RECTANGLE_ARM_DISTANCE_PALMS:
                    self.state = RectangleState.ARMED
                    self._arm_distance = distance
                    self._smooth_points(first, second)
            elif self.state == RectangleState.ARMED:
                self._smooth_points(first, second)
                if distance >= self._arm_distance + RECTANGLE_START_SEPARATION_PALMS:
                    self.state = RectangleState.DRAWING
                    self.alpha = 1.0
            elif self.state == RectangleState.DRAWING:
                self._smooth_points(first, second)
        elif not hands:
            self._tracking_lost_frames += 1
            if self._tracking_lost_frames > RECTANGLE_TRACKING_GRACE_FRAMES:
                self._begin_closing()
        else:
            self._begin_closing()

        self._draw(frame, dt)
