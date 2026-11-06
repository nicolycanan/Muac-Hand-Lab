"""Two-hand L gesture state, persistent geometry, and animated fill rendering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from config import (
    PSYCHEDELIC_FILL_ENABLED,
    RECTANGLE_ARM_DISTANCE_PALMS,
    RECTANGLE_DELETE_ENABLED,
    RECTANGLE_DELETE_HEIGHT_RATIO,
    RECTANGLE_DELETE_HOLD_SECONDS,
    RECTANGLE_DELETE_HYSTERESIS,
    RECTANGLE_DELETE_WIDTH_RATIO,
    RECTANGLE_DEFAULT_FILL_STYLE,
    RECTANGLE_FADE_SECONDS,
    RECTANGLE_SCREEN_MARGIN,
    RECTANGLE_POINT_SMOOTHING,
    RECTANGLE_START_SEPARATION_PALMS,
    RECTANGLE_TRACKING_GRACE_FRAMES,
    TRANSFORM_MOVE_SMOOTHING,
    TRANSFORM_SCALE_SMOOTHING,
)
from core.gesture_controller import HandGesture
from effects.psychedelic_fill import PsychedelicFill
from effects.rectangle_style import RectangleFillStyle


class RectangleState(StrEnum):
    IDLE = "IDLE"
    ARMED = "ARMED"
    DRAWING = "DRAWING"
    LOCKED = "LOCKED"
    CLOSING = "CLOSING"


@dataclass
class RectangleTransformReference:
    pinch_center: np.ndarray
    pinch_distance: float
    rectangle_center: np.ndarray
    width: float
    height: float
    current_scale: float
    smooth_pinch_center: np.ndarray
    smooth_pinch_distance: float
    center_offset: np.ndarray
    maximum_scale: float


class RectangleFrame:
    def __init__(self, instance_id: int = 0) -> None:
        self.instance_id = instance_id
        self.fill = PsychedelicFill()
        self.fill_style = RectangleFillStyle(RECTANGLE_DEFAULT_FILL_STYLE)
        self.reset()

    def reset(self) -> None:
        self.state = RectangleState.IDLE
        self.points: tuple[np.ndarray, np.ndarray] | None = None
        self._arm_distance = 0.0
        self._tracking_lost_frames = 0
        self._closing_elapsed = 0.0
        self._delete_elapsed = 0.0
        self.transform_reference: RectangleTransformReference | None = None
        self.alpha = 0.0

    @property
    def visible(self) -> bool:
        return self.state in (
            RectangleState.DRAWING,
            RectangleState.LOCKED,
            RectangleState.CLOSING,
        ) and self.alpha > 0.0

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

    def _finish_creation(self) -> None:
        if self.state == RectangleState.ARMED:
            self.reset()
        elif self.state == RectangleState.DRAWING:
            self.state = RectangleState.LOCKED

    @property
    def transform_scale(self) -> float:
        if self.transform_reference is None:
            return 1.0
        return self.transform_reference.current_scale

    def end_transform(self) -> None:
        self.transform_reference = None
        self._delete_elapsed = 0.0

    @property
    def delete_pending(self) -> bool:
        return self._delete_elapsed > 0.0

    def cycle_fill_style(self) -> RectangleFillStyle:
        self.fill_style = self.fill_style.next()
        return self.fill_style

    def contains(self, point: np.ndarray) -> bool:
        if self.state != RectangleState.LOCKED or self.points is None:
            return False
        minimum = np.minimum(self.points[0], self.points[1])
        maximum = np.maximum(self.points[0], self.points[1])
        return bool(np.all(point >= minimum) and np.all(point <= maximum))

    def constrain_geometry(self, frame_shape: tuple[int, ...]) -> None:
        if self.points is None or len(frame_shape) < 2:
            return
        height, width = frame_shape[:2]
        if width <= 0 or height <= 0:
            return

        margin_x = min(float(RECTANGLE_SCREEN_MARGIN), max((width - 1) * 0.5, 0.0))
        margin_y = min(float(RECTANGLE_SCREEN_MARGIN), max((height - 1) * 0.5, 0.0))
        available = np.asarray((
            max(width - 1 - 2 * margin_x, 0.0),
            max(height - 1 - 2 * margin_y, 0.0),
        ))
        minimum = np.minimum(self.points[0], self.points[1]).astype(np.float64)
        maximum = np.maximum(self.points[0], self.points[1]).astype(np.float64)
        size = np.maximum(maximum - minimum, 0.0)
        if not np.isfinite(size).all() or not np.isfinite(minimum).all():
            return

        scale = min(1.0, available[0] / max(size[0], 1.0), available[1] / max(size[1], 1.0))
        size *= scale
        center = (minimum + maximum) * 0.5
        half_size = size * 0.5
        center = np.clip(
            center,
            np.asarray((margin_x, margin_y)) + half_size,
            np.asarray((width - 1 - margin_x, height - 1 - margin_y)) - half_size,
        )
        self.points = (
            (center - half_size).astype(np.float32),
            (center + half_size).astype(np.float32),
        )

    def capture_source_roi(self, frame: np.ndarray) -> np.ndarray | None:
        if self.points is None:
            return None
        height, width = frame.shape[:2]
        corners = [tuple(np.rint(point).astype(int)) for point in self.points]
        x_min = min(point[0] for point in corners)
        x_max = max(point[0] for point in corners)
        y_min = min(point[1] for point in corners)
        y_max = max(point[1] for point in corners)
        left = max(0, x_min + 1)
        top = max(0, y_min + 1)
        right = min(width, x_max)
        bottom = min(height, y_max)
        if right <= left or bottom <= top:
            return None
        return frame[top:bottom, left:right].copy()

    def _update_delete_timer(self, width: float, height: float, frame_shape: tuple[int, ...], dt: float) -> None:
        if not RECTANGLE_DELETE_ENABLED or self.points is None:
            self._delete_elapsed = 0.0
            return

        frame_height, frame_width = frame_shape[:2]
        width_limit = frame_width * RECTANGLE_DELETE_WIDTH_RATIO
        height_limit = frame_height * RECTANGLE_DELETE_HEIGHT_RATIO
        if self._delete_elapsed <= 0.0:
            if width <= width_limit and height <= height_limit:
                self._delete_elapsed = max(dt, 0.0)
            return

        if (
            width > width_limit * RECTANGLE_DELETE_HYSTERESIS
            or height > height_limit * RECTANGLE_DELETE_HYSTERESIS
        ):
            self._delete_elapsed = 0.0
            return

        self._delete_elapsed += max(dt, 0.0)
        if self._delete_elapsed >= RECTANGLE_DELETE_HOLD_SECONDS:
            self._delete_elapsed = 0.0
            self.transform_reference = None
            self.state = RectangleState.CLOSING
            self._closing_elapsed = 0.0

    def update_transform(
        self,
        first: np.ndarray,
        second: np.ndarray,
        frame_shape: tuple[int, ...],
        dt: float,
    ) -> None:
        if self.state != RectangleState.LOCKED or self.points is None:
            return

        self.constrain_geometry(frame_shape)
        pinch_center = (first + second) * 0.5
        pinch_distance = float(np.linalg.norm(second - first))
        if pinch_distance < 8.0:
            return

        reference = self.transform_reference
        if reference is None:
            minimum = np.minimum(self.points[0], self.points[1])
            maximum = np.maximum(self.points[0], self.points[1])
            rectangle_center = (minimum + maximum) * 0.5
            width, height = maximum - minimum
            if width <= 0.0 or height <= 0.0:
                return
            frame_height, frame_width = frame_shape[:2]
            margin_x = min(float(RECTANGLE_SCREEN_MARGIN), max((frame_width - 1) * 0.5, 0.0))
            margin_y = min(float(RECTANGLE_SCREEN_MARGIN), max((frame_height - 1) * 0.5, 0.0))
            self.transform_reference = RectangleTransformReference(
                pinch_center=pinch_center.copy(),
                pinch_distance=pinch_distance,
                rectangle_center=rectangle_center,
                width=float(width),
                height=float(height),
                current_scale=1.0,
                smooth_pinch_center=pinch_center.copy(),
                smooth_pinch_distance=pinch_distance,
                center_offset=np.zeros(2, dtype=np.float32),
                maximum_scale=min(
                    max(frame_width - 1 - 2 * margin_x, 0.0) / float(width),
                    max(frame_height - 1 - 2 * margin_y, 0.0) / float(height),
                ),
            )
            return

        reference.smooth_pinch_center += TRANSFORM_MOVE_SMOOTHING * (
            pinch_center - reference.smooth_pinch_center
        )
        reference.smooth_pinch_distance += TRANSFORM_SCALE_SMOOTHING * (
            pinch_distance - reference.smooth_pinch_distance
        )
        reference.current_scale = min(
            reference.smooth_pinch_distance / reference.pinch_distance,
            reference.maximum_scale,
        )

        center = (
            reference.rectangle_center
            + reference.smooth_pinch_center
            - reference.pinch_center
            + reference.center_offset
        )
        half_size = np.asarray(
            (reference.width, reference.height), dtype=np.float32,
        ) * (0.5 * reference.current_scale)
        proposed_center = center.copy()
        height, width = frame_shape[:2]
        margin_x = min(float(RECTANGLE_SCREEN_MARGIN), max((width - 1) * 0.5, 0.0))
        margin_y = min(float(RECTANGLE_SCREEN_MARGIN), max((height - 1) * 0.5, 0.0))
        center = np.clip(
            center,
            np.asarray((margin_x, margin_y)) + half_size,
            np.asarray((width - 1 - margin_x, height - 1 - margin_y)) - half_size,
        )
        reference.center_offset += center - proposed_center
        self.points = (center - half_size, center + half_size)
        self._update_delete_timer(
            float(half_size[0] * 2),
            float(half_size[1] * 2),
            frame_shape,
            dt,
        )

    def _draw(
        self,
        frame: np.ndarray,
        dt: float,
        source_roi: np.ndarray | None = None,
    ) -> None:
        if self.state == RectangleState.CLOSING:
            self._closing_elapsed += max(dt, 0.0)
            self.alpha = max(
                0.0,
                1.0 - self._closing_elapsed / RECTANGLE_FADE_SECONDS
            )

        elif self.state in (RectangleState.DRAWING, RectangleState.LOCKED):
            self.alpha = 1.0

        if self.state == RectangleState.CLOSING and self.alpha <= 0.0:
            self.reset()
            return

        if not self.visible or self.points is None:
            return

        self.constrain_geometry(frame.shape)

        if not PSYCHEDELIC_FILL_ENABLED:
            return

        height, width = frame.shape[:2]

        if width <= 0 or height <= 0:
            return

        raw_corners = [
            tuple(np.rint(point).astype(int))
            for point in self.points
        ]

        x_min = min(point[0] for point in raw_corners)
        x_max = max(point[0] for point in raw_corners)
        y_min = min(point[1] for point in raw_corners)
        y_max = max(point[1] for point in raw_corners)

        if (
            x_max < 0 or y_max < 0
            or x_min >= width or y_min >= height
        ):
            return

        first = (
            int(np.clip(x_min, 0, width - 1)),
            int(np.clip(y_min, 0, height - 1))
        )

        second = (
            int(np.clip(x_max, 0, width - 1)),
            int(np.clip(y_max, 0, height - 1))
        )

        if second[0] <= first[0] or second[1] <= first[1]:
            return

        self.fill.render(
            frame,
            first,
            second,
            dt,
            self.alpha,
            self.fill_style,
            source_roi=source_roi,
        )

    def update(
        self,
        frame: np.ndarray,
        hands: dict[str, HandGesture],
        dt: float,
        source_roi: np.ndarray | None = None,
    ) -> None:
        dt = max(float(dt), 0.0)
        if self.state == RectangleState.CLOSING:
            self._draw(frame, dt, source_roi)
            return
        if self.state == RectangleState.LOCKED:
            self._draw(frame, dt, source_roi)
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
                self._finish_creation()
        else:
            self._finish_creation()

        self._draw(frame, dt, source_roi)
