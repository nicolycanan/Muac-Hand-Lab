"""Rastreamento de até duas mãos com o MediaPipe e desenho do esqueleto."""

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

    @staticmethod
    def draw_landmarks(frame: np.ndarray, hands: list[dict]) -> None:
        """Desenha os 21 pontos e as conexões de cada mão no quadro de exibição."""
        for hand in hands:
            landmarks = hand["landmarks"]
            color = (0, 255, 255) if hand.get("label") == "LEFT" else (255, 0, 255)
            for start, end in mp.solutions.hands.HAND_CONNECTIONS:
                pt_start = tuple(landmarks[start].astype(int))
                pt_end = tuple(landmarks[end].astype(int))
                cv2.line(frame, pt_start, pt_end, color, 2)

            for point in landmarks:
                center = tuple(point.astype(int))
                cv2.circle(frame, center, 4, color, -1)

            wrist = tuple(landmarks[0].astype(int))
            cv2.putText(
                frame,
                hand.get("label", "LEFT"),
                (wrist[0], wrist[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    def close(self):
        self.hands.close()
