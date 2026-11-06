"""Loop principal do MUAC, da captura de vídeo à exibição do coração."""

from __future__ import annotations

import time

import cv2
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
    SHOW_LANDMARKS,
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
        self.state.show_landmarks = SHOW_LANDMARKS
        self.state.active_effect = "PARTICLE_HEART" if START_WITH_HEART else "IDLE"
        self.state.display_resolution = f"{DISPLAY_WIDTH}x{DISPLAY_HEIGHT}"
        self.state.camera_resolution = f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"
        self.display_width = DISPLAY_WIDTH
        self.display_height = DISPLAY_HEIGHT
        self._last_status_log = 0.0
        self._startup_logged = False

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

    def _emit_runtime_log(self, reason: str) -> None:
        if not self._startup_logged:
            print(f"[MagicHandApp] file={__file__}")
            print(f"[MagicHandApp] tracker={__import__('core.hand_tracker', fromlist=['__file__']).__file__}")
            print(f"[MagicHandApp] shapes={__import__('particles.shapes', fromlist=['__file__']).__file__}")
            print(f"[MagicHandApp] camera index={CAMERA_INDEX} backend={CAMERA_BACKEND} resolution={CAMERA_WIDTH}x{CAMERA_HEIGHT}")
            self._startup_logged = True
        print(f"[MagicHandApp] {reason}")

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

    def run(self):
        if not self.camera.open():
            print(f"Unable to open camera {self.camera.info}; please check CAMERA_INDEX and backend settings.")
            return

        self._emit_runtime_log(f"Camera ready: {self.camera.info}")
        cv2.namedWindow("MUAC / MAGIC HAND LAB", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MUAC / MAGIC HAND LAB", self.display_width, self.display_height)

        while self.running:
            ok, frame = self.camera.read()
            if not ok or frame is None:
                print("Camera read failed; stopping application.")
                break

            camera_frame = cv2.flip(frame, 1)
            hands = self.tracker.detect(camera_frame)

            render_frame = camera_frame.copy()
            if CAMERA_DARKNESS:
                render_frame = cv2.convertScaleAbs(
                    render_frame, alpha=1.0 - CAMERA_DARKNESS, beta=0
                )
            self.state.camera_resolution = f"{camera_frame.shape[1]}x{camera_frame.shape[0]}"
            self.state.display_resolution = f"{self.display_width}x{self.display_height}"
            self.state.tracker_active = self.tracker.is_active

            current_time = time.perf_counter()
            dt = current_time - self.previous_time
            self.previous_time = current_time

            self.state.hands_detected = len(hands)
            if current_time - self._last_status_log >= 1.0:
                landmark_counts = [len(hand["landmarks"]) for hand in hands]
                self._emit_runtime_log(
                    f"hands={self.state.hands_detected} landmarks_per_hand={landmark_counts} "
                    f"skeleton={self.state.show_landmarks} "
                    f"resolution={self.state.camera_resolution}"
                )
                self._last_status_log = current_time
            gesture = self.gesture_controller.update(hands if self.state.gesture_enabled else [], dt)
            self.state.gesture = gesture
            self.state.confirmed_gesture = gesture if gesture != "IDLE" else "IDLE"

            self.state.active_effect = self.effect_manager.update(render_frame, self.gesture_controller, dt)
            if self.state.show_landmarks:
                self.tracker.draw_landmarks(render_frame, hands)

            self.state.heart_enabled = self.effect_manager.heart_enabled
            self.state.transform_active = self.effect_manager.transform_active
            if dt > 0.0:
                fps = 1.0 / dt
                self.state.fps += 0.15 * (fps - self.state.fps)
            display_frame = self._letterbox(render_frame, self.display_width, self.display_height)
            draw_hud(
                display_frame, self.state, self.gesture_controller,
                self.effect_manager.interaction_mode, self.effect_manager.current_shape,
            )

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
