"""MUAC Magic Hand Lab webcam application."""

from __future__ import annotations

import time

import cv2
import mediapipe as mp
import numpy as np

from config import (
    CAMERA_BACKEND,
    CAMERA_DARKNESS,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    GESTURE_HOLD_TIME,
    PARTICLE_COUNT,
    PINCH_THRESHOLD,
    START_WITH_HEART,
    WINDOW_RESIZABLE,
)
from core.app_state import AppState
from core.camera import CameraController
from core.gesture_controller import GestureController
from core.hand_tracker import HandTracker
from effects.effect_manager import EffectManager
from rendering.hud import draw_hud


class MagicHandApp:
    def __init__(self):
        self.state = AppState()
        self.state.particle_count = PARTICLE_COUNT
        self.state.heart_enabled = START_WITH_HEART
        self.state.active_effect = "PARTICLE_HEART" if START_WITH_HEART else "IDLE"
        self.state.display_resolution = f"{DISPLAY_WIDTH}x{DISPLAY_HEIGHT}"
        self.state.camera_resolution = f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"
        self.display_width = DISPLAY_WIDTH
        self.display_height = DISPLAY_HEIGHT

        self.camera = CameraController(
            camera_index=CAMERA_INDEX,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
            backend_name=CAMERA_BACKEND,
        )
        self.tracker = HandTracker(max_hands=2, confidence=0.5)
        self.gesture_controller = GestureController(hold_time=GESTURE_HOLD_TIME, pinch_threshold=PINCH_THRESHOLD)
        self.effect_manager = EffectManager()
        self.effect_manager.heart_enabled = START_WITH_HEART
        self.effect_manager.active_effect = "PARTICLE_HEART" if START_WITH_HEART else "IDLE"
        self.running = True
        self.previous_time = time.perf_counter()

    def _letterbox(self, frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return canvas

    def _handle_keys(self, key: int) -> None:
        if key == ord('q') or key == 27:
            self.running = False
        elif key in (ord('h'), ord('H')):
            self.effect_manager.toggle_heart()
            self.state.heart_enabled = self.effect_manager.heart_enabled
            self.state.active_effect = self.effect_manager.active_effect
        elif key in (ord('r'), ord('R')):
            self.effect_manager.heart.system.scatter()
        elif key in (ord('e'), ord('E')):
            self.effect_manager.heart.system.explode()
        elif key in (ord('g'), ord('G')):
            self.state.gesture_enabled = not self.state.gesture_enabled
        elif key in (ord('d'), ord('D')):
            self.state.show_debug = not self.state.show_debug
        elif key in (ord('m'), ord('M')):
            self.state.show_landmarks = not self.state.show_landmarks

    def _draw_landmarks(self, frame, hands):
        if not self.state.show_landmarks and not self.state.show_debug:
            return

        for idx, hand in enumerate(hands):
            landmarks = hand["landmarks"]
            label = hand.get("label", "LEFT" if idx == 0 else "RIGHT")
            color = (0, 255, 255) if label == "LEFT" else (255, 0, 255)
            for connection in mp.solutions.hands.HAND_CONNECTIONS:
                a_index, b_index = connection
                pt_a = tuple(np.round(landmarks[a_index]).astype(int))
                pt_b = tuple(np.round(landmarks[b_index]).astype(int))
                cv2.line(frame, pt_a, pt_b, (255, 255, 255), 1, cv2.LINE_AA)
            for point in landmarks:
                center = tuple(np.round(point).astype(int))
                cv2.circle(frame, center, 2, (0, 0, 255), -1)
            cv2.putText(frame, label, tuple(np.round(landmarks[0]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def run(self):
        if not self.camera.open():
            print(f"Unable to open camera {self.camera.info}; please check CAMERA_INDEX and backend settings.")
            return

        print(f"Camera ready: {self.camera.info}")
        cv2.namedWindow("MUAC / MAGIC HAND LAB", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MUAC / MAGIC HAND LAB", self.display_width, self.display_height)

        while self.running:
            ok, frame = self.camera.read()
            if not ok or frame is None:
                print("Camera read failed; stopping application.")
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.addWeighted(frame, 1.0 - CAMERA_DARKNESS, np.zeros_like(frame), CAMERA_DARKNESS, 0.0)
            self.state.camera_resolution = f"{frame.shape[1]}x{frame.shape[0]}"
            self.state.display_resolution = f"{self.display_width}x{self.display_height}"
            self.state.tracker_active = self.tracker.is_active

            current_time = time.perf_counter()
            dt = max(0.016, min(0.05, current_time - self.previous_time))
            self.previous_time = current_time

            hands = self.tracker.detect(frame)
            self.state.hands_detected = len(hands)
            gesture = "IDLE"
            if self.state.gesture_enabled:
                gesture = self.gesture_controller.update(hands, dt)
            else:
                self.gesture_controller.candidate = "IDLE"
                self.gesture_controller.confirmed = "IDLE"
                self.gesture_controller.active = "IDLE"
                self.gesture_controller.timer = 0.0
                self.gesture_controller.triggered = False
                gesture = "IDLE"
            self.state.gesture = gesture
            self.state.confirmed_gesture = gesture if gesture != "IDLE" else "IDLE"

            self.state.active_effect = self.effect_manager.update(frame, hands, gesture, dt)
            if self.effect_manager.heart_enabled and self.state.active_effect == "IDLE":
                self.effect_manager.heart.draw(frame)
            self._draw_landmarks(frame, hands)

            fps = 1.0 / dt if dt > 0 else 0.0
            self.state.fps = fps
            display_frame = self._letterbox(frame, self.display_width, self.display_height)
            draw_hud(display_frame, self.state, fps, self.state.hands_detected, gesture)

            cv2.imshow("MUAC / MAGIC HAND LAB", display_frame)
            key = cv2.waitKey(1) & 0xFF
            self._handle_keys(key)

        self.close()

    def close(self):
        self.camera.close()
        self.tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    MagicHandApp().run()
