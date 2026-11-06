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
)
from core.app_state import AppState
from core.camera import CameraController
from core.gesture_controller import GestureController
from core.hand_tracker import HandTracker
from effects.effect_manager import EffectManager
from rendering.camera_picker import CameraPicker
from rendering.hud import draw_hand_controls, draw_hud


WINDOW_TITLE = "MUAC / MAGIC HAND LAB"


class MagicHandApp:
    def __init__(self):
        self.state = AppState()
        self.state.particle_count = PARTICLE_COUNT
        self.state.show_landmarks = SHOW_LANDMARKS
        self.state.display_resolution = f"{DISPLAY_WIDTH}x{DISPLAY_HEIGHT}"
        self.state.camera_resolution = f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"
        self.display_width = DISPLAY_WIDTH
        self.display_height = DISPLAY_HEIGHT
        self._last_status_log = 0.0
        self._startup_logged = False

        self.camera: CameraController | None = None
        self.camera_index: int | None = None
        self.camera_status = ""
        self._camera_read_failed = False
        self._camera_frame: np.ndarray | None = None
        self._empty_frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        self.camera_picker = CameraPicker(self.display_height)
        self.tracker = HandTracker(max_hands=2, confidence=0.5)
        self.gesture_controller = GestureController(hold_time=GESTURE_HOLD_TIME, pinch_threshold=PINCH_THRESHOLD)
        self.effect_manager = EffectManager()
        self.running = True
        self.previous_time = time.perf_counter()

    def _select_camera(self, index: int) -> None:
        if self.camera is not None and index == self.camera_index:
            if not self._camera_read_failed:
                self.camera_status = ""
                return
            self.camera.close()
            self.camera = None
            self.camera_index = None

        candidate = CameraController(
            camera_index=index, width=CAMERA_WIDTH, height=CAMERA_HEIGHT,
            fps=CAMERA_FPS, backend_name=CAMERA_BACKEND,
        )
        try:
            ok, frame = candidate.read() if candidate.open() else (False, None)
        except cv2.error:
            ok, frame = False, None
        if not ok or frame is None or not frame.size:
            candidate.close()
            self.camera_status = f"Camera {index} indisponivel. Escolha outro indice."
            return

        if self.camera is not None:
            self.camera.close()
        self.camera = candidate
        self.camera_index = index
        self.camera_status = ""
        self._camera_read_failed = False
        self._camera_frame = frame
        self.gesture_controller = GestureController(hold_time=GESTURE_HOLD_TIME, pinch_threshold=PINCH_THRESHOLD)
        self._emit_runtime_log(f"Camera pronta: {candidate.info}")

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
        elif key in (ord('c'), ord('C')):
            self.camera_picker.expanded = not self.camera_picker.expanded
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

    def _read_camera_frame(self) -> np.ndarray | None:
        if self._camera_frame is not None:
            frame, self._camera_frame = self._camera_frame, None
            return frame
        if self.camera is None:
            return None
        try:
            ok, frame = self.camera.read()
        except cv2.error:
            ok, frame = False, None
        if ok and frame is not None and frame.size:
            if self._camera_read_failed:
                self.camera_status = ""
                self._camera_read_failed = False
            return frame
        self._camera_read_failed = True
        self.camera_status = f"Sem video na camera {self.camera_index}. Use o seletor."
        return None

    def run(self):
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(WINDOW_TITLE, self.display_width, self.display_height)
            cv2.setMouseCallback(WINDOW_TITLE, self.camera_picker.on_mouse)
            self._select_camera(CAMERA_INDEX)
            self.previous_time = time.perf_counter()
            self._run_loop()
        finally:
            self.close()

    def _run_loop(self):
        while self.running:
            index = self.camera_picker.take_selection()
            if index is not None:
                self._select_camera(index)
                self.previous_time = time.perf_counter()
            frame = self._read_camera_frame()
            hands = []
            if frame is None:
                render_frame = self._empty_frame.copy()
                self.state.camera_resolution = "--"
            else:
                height, width = frame.shape[:2]
                self.state.camera_resolution = f"{width}x{height}"
                camera_frame = cv2.flip(frame, 1)
                if (width, height) != (CAMERA_WIDTH, CAMERA_HEIGHT):
                    camera_frame = self._letterbox(camera_frame, CAMERA_WIDTH, CAMERA_HEIGHT)
                hands = self.tracker.detect(camera_frame)
                render_frame = cv2.convertScaleAbs(camera_frame, alpha=1.0 - CAMERA_DARKNESS)

            self.state.tracker_active = frame is not None and self.tracker.is_active

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
            if self.state.show_landmarks and hands:
                self.tracker.draw_landmarks(render_frame, hands)

            draw_hand_controls(render_frame, self.gesture_controller, self.effect_manager)
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

            self.camera_picker.draw(display_frame, self.camera_index, self.camera_status)
            cv2.imshow(WINDOW_TITLE, display_frame)
            key = cv2.waitKey(16 if frame is None else 1) & 0xFF
            self._handle_keys(key)
            if self.running:
                try:
                    self.running = cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) >= 1
                except cv2.error:
                    self.running = False

    def close(self):
        if self.camera is not None:
            self.camera.close()
        self.tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    MagicHandApp().run()
