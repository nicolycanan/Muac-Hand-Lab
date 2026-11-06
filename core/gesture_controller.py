"""Classifica a geometria das mãos e confirma gestos com base na sequência de quadros."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from config import (
    CONTINUOUS_HOLD_TIME,
    GESTURE_HOLD_TIME,
    LETTER_CHANGE_HOLD_TIME,
    PINCH_THRESHOLD,
    SNAP_CLOSE_RATIO,
    SNAP_COOLDOWN,
    SNAP_MAX_TIME,
    SNAP_MIN_SPEED,
    SNAP_RELEASE_RATIO,
)


class RawGesture(StrEnum):
    IDLE = "IDLE"
    INDEX_RAISED = "INDEX_RAISED"
    TWO_FINGERS = "TWO_FINGERS"
    THREE_FINGERS = "THREE_FINGERS"
    OPEN_HAND = "OPEN_HAND"
    PINCH = "PINCH"
    FIST = "FIST"
    SNAP = "SNAP"
    TRANSFORM = "TRANSFORM"


@dataclass
class CandidateGesture:
    gesture: RawGesture = RawGesture.IDLE
    hands: tuple[str, ...] = ()
    elapsed: float = 0.0


@dataclass(frozen=True)
class ConfirmedGesture:
    gesture: RawGesture = RawGesture.IDLE
    hands: tuple[str, ...] = ()


class InteractionMode(StrEnum):
    IDLE = "IDLE"
    SHAPE = "SHAPE"
    FREE = "FREE"
    REPEL = "REPEL"
    GRAB = "GRAB"
    TRANSFORM = "TRANSFORM"


class GestureEvent(StrEnum):
    SELECT_HEART = "SELECT_HEART"
    SELECT_LETTER = "SELECT_LETTER"
    CHANGE_LETTER = "CHANGE_LETTER"
    BREAK = "BREAK"


@dataclass(frozen=True)
class HandGesture:
    raw: RawGesture
    fingers: dict[str, bool]
    pinch_ratio: float
    pinch_point: np.ndarray
    palm_center: np.ndarray
    palm_motion: np.ndarray


@dataclass
class SnapState:
    gap: float
    middle_relative: np.ndarray
    armed_at: float | None = None


class GestureController:
    FINGER_LANDMARKS = {
        "INDEX": (5, 6, 7, 8),
        "MIDDLE": (9, 10, 11, 12),
        "RING": (13, 14, 15, 16),
        "PINKY": (17, 18, 19, 20),
    }
    SHAPE_EVENTS = {
        RawGesture.INDEX_RAISED: GestureEvent.SELECT_HEART,
        RawGesture.TWO_FINGERS: GestureEvent.SELECT_LETTER,
        RawGesture.THREE_FINGERS: GestureEvent.CHANGE_LETTER,
    }
    MODES = {
        RawGesture.OPEN_HAND: InteractionMode.REPEL,
        RawGesture.PINCH: InteractionMode.GRAB,
        RawGesture.TRANSFORM: InteractionMode.TRANSFORM,
    }

    def __init__(
        self,
        hold_time: float = GESTURE_HOLD_TIME,
        pinch_threshold: float = PINCH_THRESHOLD,
        pinch_release_threshold: float = 0.55,
        snap_cooldown: float = SNAP_COOLDOWN,
    ):
        self.hold_time = hold_time
        self.pinch_threshold = pinch_threshold
        self.pinch_release_threshold = max(pinch_release_threshold, pinch_threshold)
        self.snap_cooldown = snap_cooldown
        self.pending = CandidateGesture()
        self.confirmation = ConfirmedGesture()
        self.raw = RawGesture.IDLE
        self.mode = InteractionMode.IDLE
        self.event: GestureEvent | None = None
        self.hands: dict[str, HandGesture] = {}
        self.event_point: np.ndarray | None = None
        self._snap_tracking: dict[str, SnapState] = {}
        self._time = 0.0
        self._snap_ready_at = 0.0
        self._awaiting_transform_release = False
        self._shape_latches: dict[str, RawGesture] = {}

    @property
    def candidate(self) -> RawGesture:
        return self.pending.gesture

    @property
    def confirmed(self) -> RawGesture:
        return self.confirmation.gesture

    @property
    def triggered(self) -> bool:
        return self.event is not None

    @property
    def action(self) -> str:
        return self.event or self.SHAPE_EVENTS.get(self.confirmed) or self.MODES.get(self.confirmed, "NONE")

    def confirmation_delay(self, raw: RawGesture) -> float:
        if raw == RawGesture.THREE_FINGERS:
            return LETTER_CHANGE_HOLD_TIME
        if raw in self.SHAPE_EVENTS:
            return self.hold_time
        if raw in self.MODES and self.hold_time > 0:
            return CONTINUOUS_HOLD_TIME
        return 0.0

    @property
    def finger_states(self) -> dict[str, dict[str, bool]]:
        return {key: hand.fingers for key, hand in self.hands.items()}

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        first = a - b
        second = c - b
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator <= 1e-6:
            return 0.0
        cosine = float(np.dot(first, second) / denominator)
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    def _finger_extension(self, points: np.ndarray, palm: float, indices: tuple[int, int, int, int]) -> bool:
        mcp, pip, dip, tip = points[list(indices)]
        wrist = points[0]
        return (
            self._angle(mcp, pip, dip) >= 145.0
            and self._angle(pip, dip, tip) >= 135.0
            and self._distance(tip, wrist) > 1.10 * self._distance(pip, wrist)
            and self._distance(tip, mcp) >= 0.48 * palm
        )

    def _thumb_extension(self, points: np.ndarray, palm: float) -> bool:
        thumb_from_index = points[4] - points[5]
        outward = points[5] - points[17]
        return (
            np.dot(thumb_from_index, outward) > 0.05 * np.dot(outward, outward)
            and self._angle(points[1], points[2], points[3]) >= 125.0
            and self._angle(points[2], points[3], points[4]) >= 130.0
            and self._distance(points[4], points[5]) >= 0.55 * palm
        )

    def _read_hand(self, points: np.ndarray, palm: float, previous: HandGesture | None) -> HandGesture:
        fingers = {"THUMB": self._thumb_extension(points, palm)}
        fingers.update({
            name: self._finger_extension(points, palm, indices)
            for name, indices in self.FINGER_LANDMARKS.items()
        })
        gap = self._distance(points[4], points[8])
        ratio = gap / palm
        was_pinching = previous is not None and previous.raw == RawGesture.PINCH
        threshold = self.pinch_release_threshold if was_pinching else self.pinch_threshold
        other_gaps = np.linalg.norm(points[[12, 16, 20]] - points[4], axis=1)
        palm_center = points[[0, 5, 9, 13, 17]].mean(axis=0)
        pinch_point = (points[4] + points[8]) * 0.5
        palm_motion = np.clip(palm_center - previous.palm_center, -35, 35) if previous else np.zeros(2)
        contact = gap < other_gaps.min() and self._distance(pinch_point, palm_center) > 0.3 * palm
        raised = sum(fingers[name] for name in self.FINGER_LANDMARKS)
        if all(fingers.values()):
            raw = RawGesture.OPEN_HAND
        elif ratio < threshold and contact:
            raw = RawGesture.PINCH
        elif raised == 1 and fingers["INDEX"]:
            raw = RawGesture.INDEX_RAISED
        elif raised == 2 and not fingers["THUMB"]:
            raw = RawGesture.TWO_FINGERS
        elif raised == 3 and not fingers["THUMB"]:
            raw = RawGesture.THREE_FINGERS
        elif raised == 0:
            raw = RawGesture.FIST
        else:
            raw = RawGesture.IDLE
        return HandGesture(raw, fingers, ratio, pinch_point, palm_center, palm_motion)

    def _snap_detected(
        self, key: str, hand: HandGesture, points: np.ndarray, palm: float, dt: float
    ) -> bool:
        gap = self._distance(points[4], points[12]) / palm
        middle_relative = points[12] - hand.palm_center
        previous = self._snap_tracking.get(key)
        state = SnapState(gap, middle_relative)
        self._snap_tracking[key] = state
        if self._time < self._snap_ready_at or (hand.raw in self.SHAPE_EVENTS or hand.raw == RawGesture.OPEN_HAND):
            return False

        armed_at = previous.armed_at if previous else None
        if armed_at is not None and self._time - armed_at > SNAP_MAX_TIME:
            armed_at = None
        if gap <= SNAP_CLOSE_RATIO:
            if (previous is None or previous.gap > SNAP_CLOSE_RATIO) and hand.raw != RawGesture.PINCH:
                armed_at = self._time
        elif gap >= SNAP_RELEASE_RATIO:
            # Encerra a tentativa na soltura. Um movimento lento exige novo contato.
            return (
                armed_at is not None
                and (gap - previous.gap) / max(dt, 1e-6) > 2.0
                and self._distance(middle_relative, previous.middle_relative) >= SNAP_MIN_SPEED * 30.0 * dt
            )
        state.armed_at = armed_at
        return False

    def _hand_inputs(self, hands: list[dict]) -> Iterator[tuple[str, np.ndarray, float]]:
        seen = set()
        for index, hand in enumerate(hands):
            key = str(hand.get("label", f"HAND_{index}")).upper()
            if key in seen:
                key = f"{key}_{index}"
            points = np.asarray(hand["landmarks"], dtype=np.float32)
            if points.shape != (21, 2) or not np.isfinite(points).all():
                continue
            palm = float(hand.get("palm_size", self._distance(points[0], points[9])))
            if not np.isfinite(palm) or palm <= 1.0:
                continue
            seen.add(key)
            yield key, points, palm

    def detect_two_hand_transform(self, hands: list[dict]) -> bool:
        return sum(
            self._read_hand(points, palm, self.hands.get(key)).raw == RawGesture.PINCH
            for key, points, palm in self._hand_inputs(hands)
        ) == 2

    def update(self, hands: list[dict], dt: float) -> RawGesture:
        dt = max(float(dt), 0.0)
        self._time += dt
        previous = self.hands
        self.hands = {}
        self.event = None
        self.event_point = None
        snaps: dict[str, np.ndarray] = {}
        for key, points, palm in self._hand_inputs(hands):
            self.hands[key] = self._read_hand(points, palm, previous.get(key))
            if self._snap_detected(key, self.hands[key], points, palm, dt):
                snaps[key] = points[12].copy()

        self._snap_tracking = {key: state for key, state in self._snap_tracking.items() if key in self.hands}
        self._shape_latches = {
            key: gesture for key, gesture in self._shape_latches.items()
            if key in self.hands and self.hands[key].raw == gesture
        }
        pins = tuple(sorted(key for key, hand in self.hands.items() if hand.raw == RawGesture.PINCH))
        if len(pins) == 2:
            self._awaiting_transform_release = True
        elif self.hands and (not pins or len(self.hands) == 2):
            self._awaiting_transform_release = False
        if len(pins) == 2:
            raw, keys = RawGesture.TRANSFORM, pins
            for state in self._snap_tracking.values():
                state.armed_at = None
        elif snaps:
            raw, keys = RawGesture.SNAP, (min(snaps),)
        else:
            raw, keys = RawGesture.IDLE, ()
            for preferred in (
                RawGesture.PINCH, RawGesture.OPEN_HAND, RawGesture.THREE_FINGERS,
                RawGesture.TWO_FINGERS, RawGesture.INDEX_RAISED, RawGesture.FIST,
            ):
                if preferred == RawGesture.PINCH and self._awaiting_transform_release:
                    continue
                matching = tuple(sorted(key for key, hand in self.hands.items() if hand.raw == preferred))
                if matching:
                    raw, keys = preferred, matching if preferred == RawGesture.OPEN_HAND else matching[:1]
                    break

        self.raw = raw
        if (raw, keys) != (self.pending.gesture, self.pending.hands):
            self.pending = CandidateGesture(raw, keys)
        else:
            self.pending.elapsed += dt

        delay = self.confirmation_delay(raw)
        confirmed = ConfirmedGesture(raw, keys) if self.pending.elapsed + 1e-9 >= delay else ConfirmedGesture()
        if confirmed.gesture == RawGesture.SNAP:
            self.event = GestureEvent.BREAK
            self.event_point = snaps[keys[0]]
            self.hands[keys[0]] = replace(self.hands[keys[0]], raw=RawGesture.SNAP)
            self._snap_ready_at = self._time + self.snap_cooldown
            for state in self._snap_tracking.values():
                state.armed_at = None
        elif confirmed.gesture in self.SHAPE_EVENTS:
            hand_key = keys[0]
            if self._shape_latches.get(hand_key) != raw:
                self._shape_latches[hand_key] = raw
                self.event = self.SHAPE_EVENTS[raw]
        self.confirmation = confirmed
        self.mode = self.MODES.get(self.confirmed, InteractionMode.IDLE)
        return self.confirmed
