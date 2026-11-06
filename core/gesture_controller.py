"""Gesture recognition and confirmation logic for one or two hands."""

from __future__ import annotations

import math

import numpy as np


class GestureController:
    def __init__(self, hold_time: float = 0.5, pinch_threshold: float = 0.40):
        self.hold_time = hold_time
        self.pinch_threshold = pinch_threshold
        self.confirmed = "IDLE"
        self.active = "IDLE"
        self.timer = 0.0

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _mean(points: np.ndarray) -> np.ndarray:
        return points.mean(axis=0)

    def _finger_is_extended(self, hand: dict, tip_index: int, pip_index: int, mcp_index: int) -> bool:
        tips = hand["landmarks"]
        tip = tips[tip_index]
        pip = tips[pip_index]
        mcp = tips[mcp_index]
        palm_size = hand["palm_size"]
        if palm_size <= 0:
            return False
        extension = np.linalg.norm(tip - mcp) / max(palm_size, 1.0)
        bend = np.linalg.norm(pip - mcp) / max(palm_size, 1.0)
        return extension > 0.75 and bend < 0.9

    def _estimate_open_hand(self, hand: dict) -> float:
        if not hand["landmarks"].size:
            return 0.0
        tips = hand["landmarks"]
        palm = max(hand["palm_size"], 1.0)
        finger_scores = [
            self._finger_is_extended(hand, 8, 6, 5),
            self._finger_is_extended(hand, 12, 10, 9),
            self._finger_is_extended(hand, 16, 14, 13),
            self._finger_is_extended(hand, 20, 18, 17),
        ]
        open_score = sum(int(v) for v in finger_scores) / 4.0
        thumb_score = (self._distance(tips[4], tips[2]) / palm) > 0.65
        return float(open_score * 0.8 + (0.2 if thumb_score else 0.0))

    def _estimate_fist(self, hand: dict) -> float:
        tips = hand["landmarks"]
        palm = max(hand["palm_size"], 1.0)
        finger_bend = [
            self._distance(tips[8], tips[5]) / palm,
            self._distance(tips[12], tips[9]) / palm,
            self._distance(tips[16], tips[13]) / palm,
            self._distance(tips[20], tips[17]) / palm,
        ]
        return float(1.0 if sum(v < 0.7 for v in finger_bend) >= 3 else 0.0)

    def _estimate_index_raise(self, hand: dict) -> float:
        tips = hand["landmarks"]
        palm = max(hand["palm_size"], 1.0)
        index_ext = self._finger_is_extended(hand, 8, 6, 5)
        middle_fold = self._distance(tips[12], tips[9]) / palm < 0.55
        ring_fold = self._distance(tips[16], tips[13]) / palm < 0.55
        pinky_fold = self._distance(tips[20], tips[17]) / palm < 0.55
        if index_ext and middle_fold and ring_fold and pinky_fold:
            return 1.0
        return 0.0

    def _estimate_pinch(self, hand: dict) -> float:
        tips = hand["landmarks"]
        palm = max(hand["palm_size"], 1.0)
        pinch_dist = self._distance(tips[4], tips[8]) / palm
        return float(1.0 if pinch_dist < self.pinch_threshold else 0.0)

    def _has_two_hand_pinch(self, hands: list[dict]) -> float:
        if len(hands) < 2:
            return 0.0
        left = hands[0]
        right = hands[1]
        left_pin = self._estimate_pinch(left)
        right_pin = self._estimate_pinch(right)
        if left_pin > 0.0 and right_pin > 0.0:
            left_center = np.asarray(left["center"], dtype=np.float32)
            right_center = np.asarray(right["center"], dtype=np.float32)
            if np.linalg.norm(left_center - right_center) > 0.0:
                return 1.0
        return 0.0

    def update(self, hands: list[dict], dt: float) -> str:
        if not hands:
            self.active = "IDLE"
            self.confirmed = "IDLE"
            self.timer = 0.0
            return "IDLE"

        two_hand = self._has_two_hand_pinch(hands)
        if two_hand > 0.0:
            self.active = "TRANSFORM"
            self.timer += dt
            if self.timer >= self.hold_time:
                self.confirmed = "TRANSFORM"
            return self.confirmed

        scores = {
            "OPEN_HAND": max(self._estimate_open_hand(hand) for hand in hands),
            "FIST": max(self._estimate_fist(hand) for hand in hands),
            "INDEX_RAISED": max(self._estimate_index_raise(hand) for hand in hands),
            "PINCH": max(self._estimate_pinch(hand) for hand in hands),
        }

        if scores["PINCH"] > 0.5:
            best_gesture = "PINCH"
        else:
            best_gesture = max(scores, key=scores.get)
        best_score = scores[best_gesture]
        if best_score < 0.5:
            best_gesture = "IDLE"

        if best_gesture == "IDLE":
            self.timer = 0.0
            self.active = "IDLE"
            self.confirmed = "IDLE"
            return "IDLE"

        if best_gesture != self.active:
            self.active = best_gesture
            self.timer = 0.0
        else:
            self.timer += dt

        if self.timer >= self.hold_time:
            self.confirmed = best_gesture
        return self.confirmed

    def detect_two_hand_transform(self, hands: list[dict]) -> bool:
        return self._has_two_hand_pinch(hands) > 0.0
