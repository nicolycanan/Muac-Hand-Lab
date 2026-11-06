"""MediaPipe-backed hand detection utilities."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    def __init__(self, max_hands: int = 2, confidence: float = 0.6):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=0,
            min_detection_confidence=confidence,
            min_tracking_confidence=0.5,
        )
        self.is_active = True

    def detect(self, frame: np.ndarray) -> list[dict]:
        if frame is None or frame.size == 0:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if results is None or results.multi_hand_landmarks is None:
            return []

        detected: list[dict] = []
        for index, landmarks in enumerate(results.multi_hand_landmarks):
            points = np.asarray(
                [
                    (lm.x * frame.shape[1], lm.y * frame.shape[0])
                    for lm in landmarks.landmark
                ],
                dtype=np.float32,
            )
            palm_size = np.linalg.norm(points[9] - points[0])
            if palm_size <= 1:
                palm_size = 1.0
            label = "LEFT"
            if results.multi_handedness and index < len(results.multi_handedness):
                label = results.multi_handedness[index].classification[0].label
            detected.append(
                {
                    "landmarks": points,
                    "palm_size": float(palm_size),
                    "center": points.mean(axis=0),
                    "label": label,
                }
            )
        return detected

    def close(self):
        self.hands.close()
