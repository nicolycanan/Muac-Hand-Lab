## 1. Ambiente

- Python: 3.11.9
- SO: Microsoft Windows 10.0.26200
- `requirements.txt`:
```txt
numpy==1.26.4
opencv-python==4.11.0.86
mediapipe==0.10.21
pytest==8.3.3
```

- `pip freeze` (ambiente real):
```txt
absl-py==2.5.1
attrs==26.1.0
cffi==2.1.1
colorama==0.4.6
contourpy==1.3.3
cycler==0.12.1
flatbuffers==25.12.19
fonttools==4.68.0
iniconfig==2.3.1
jax==0.7.1
jaxlib==0.7.1
kiwisolver==1.5.1
matplotlib==3.11.2
mediapipe==0.10.21
ml_dtypes==0.5.4
numpy==1.26.4
opencv-contrib-python==4.11.0.86
opencv-python==4.11.0.86
opt_einsum==3.4.0
packaging==26.3
pillow==13.0.0
pluggy==1.7.0
protobuf==4.25.9
pycparser==3.12
pyparsing==3.3.3
pytest==8.3.3
python-dateutil==4.0.0
scipy==1.17.1
sentencepiece==0.2.3
sounddevice==0.5.6
tzdata==2026.6
```

- Há divergência: sim. `requirements.txt` é mínimo e omite dependências transitivas e utilitários instalados no ambiente real (`opencv-contrib-python`, `matplotlib`, `scipy`, `sounddevice`, etc.). A versão do MediaPipe, OpenCV e NumPy instalados no ambiente real é:
  - mediapipe==0.10.21
  - opencv-python==4.11.0.86
  - numpy==1.26.4
- API do MediaPipe usada: `mp.solutions.hands.Hands`
  - parâmetros em `core/hand_tracker.py`:
    - `static_image_mode=False`
    - `max_num_hands=2`
    - `model_complexity=0`
    - `min_detection_confidence=0.5`
    - `min_tracking_confidence=0.5`
  - não há uso da Tasks API `HandLandmarker`

## 2. Estrutura

Árvore completa (sem `.venv` e `__pycache__`):

```text
Muac Magic Hand/
├── README.md
├── _audit_tmp/
│   └── synthetic_gesture_audit.py
├── config.py
├── core/
│   ├── __init__.py
│   ├── app_state.py
│   ├── camera.py
│   ├── gesture_controller.py
│   └── hand_tracker.py
├── docs/
│   ├── architecture.md
│   ├── experiments.md
│   └── gestures.md
├── effects/
│   ├── __init__.py
│   ├── effect_manager.py
│   └── particle_heart.py
├── Magic_hand.py
├── main.py
├── particles/
│   ├── __init__.py
│   ├── particle_system.py
│   ├── shapes.py
│   └── transforms.py
├── rendering/
│   ├── __init__.py
│   └── hud.py
├── requirements.txt
├── tests/
│   ├── preview_hands.py
│   ├── preview_particles.py
│   ├── test_gestures.py
│   ├── test_particles.py
│   ├── test_shapes.py
│   └── test_transforms.py
└── .git/
```

Arquivos listados na solicitação que NÃO existem ou estão vazios:
- `core/hand_tracker.py`: existe e não está vazio
- `core/gesture_controller.py`: existe e não está vazio
- `core/app_state.py`: existe e não está vazio
- `effects/particle_heart.py`: existe e não está vazio
- `effects/effect_manager.py`: existe e não está vazio
- `particles/particle_system.py`: existe e não está vazio
- `particles/transforms.py`: existe e não está vazio
- `main.py`: existe e não está vazio

## 3. Código-fonte completo

### config.py
```python
"""Global configuration for the MUAC Magic Hand Lab project."""

import cv2

CAMERA_INDEX = 0
CAMERA_BACKEND = "DSHOW"
CAMERA_BACKEND_MAP = {
    "AUTO": cv2.CAP_ANY,
    "DSHOW": cv2.CAP_DSHOW,
    "MF": cv2.CAP_MSMF,
}

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
WINDOW_RESIZABLE = True

PARTICLE_COUNT = 1500
PARTICLE_RADIUS = 2
PARTICLE_BRIGHTNESS = 1.0
PARTICLE_GLOW = True
PARTICLE_GLOW_STRENGTH = 0.5
START_WITH_HEART = True

SHAPE_FORCE = 0.018
DAMPING = 0.86
MIN_SCALE = 0.45
MAX_SCALE = 2.0

GESTURE_HOLD_TIME = 0.5
PINCH_THRESHOLD = 0.40
SHOW_FPS = True
SHOW_LANDMARKS = False

TEST_MODE = False
```

### main.py
```python
"""Entry point for the MUAC Magic Hand Lab application."""

from Magic_hand import MagicHandApp


def main() -> None:
    app = MagicHandApp()
    app.run()


if __name__ == "__main__":
    main()
```

### Magic_hand.py
```python
"""MUAC Magic Hand Lab webcam application."""

from __future__ import annotations

import time

import cv2
import numpy as np

from config import (
    CAMERA_BACKEND,
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
            for point_index in range(len(landmarks) - 1):
                pt_a = tuple(np.round(landmarks[point_index]).astype(int))
                pt_b = tuple(np.round(landmarks[point_index + 1]).astype(int))
                cv2.line(frame, pt_a, pt_b, color, 1, cv2.LINE_AA)
            for point_index, point in enumerate(landmarks):
                center = tuple(np.round(point).astype(int))
                cv2.circle(frame, center, 2, color, -1)
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
            self.state.camera_resolution = f"{frame.shape[1]}x{frame.shape[0]}"
            self.state.display_resolution = f"{self.display_width}x{self.display_height}"
            self.state.tracker_active = self.tracker.is_active

            current_time = time.perf_counter()
            dt = max(0.016, min(0.05, current_time - self.previous_time))
            self.previous_time = current_time

            hands = self.tracker.detect(frame)
            self.state.hands_detected = len(hands)
            gesture = "IDLE"
            if self.state.gesture_enabled and hands:
                gesture = self.gesture_controller.update(hands, dt)
                self.state.gesture = gesture
                self.state.confirmed_gesture = gesture if gesture != "IDLE" else "IDLE"
            else:
                self.state.gesture = "IDLE"
                self.state.confirmed_gesture = "IDLE"

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
```

### core/camera.py
```python
"""Camera opening and validation utilities."""

from __future__ import annotations

import cv2

from config import CAMERA_BACKEND_MAP, CAMERA_FPS, CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH


def list_camera_indices(start: int = 0, stop: int = 10, backend_name: str = "AUTO") -> list[tuple[int, bool, tuple[int, int] | None]]:
    """Probe a camera index range and report whether each device is usable."""
    backend = CAMERA_BACKEND_MAP.get(backend_name.upper(), cv2.CAP_ANY)
    results: list[tuple[int, bool, tuple[int, int] | None]] = []
    for idx in range(start, stop + 1):
        cap = cv2.VideoCapture(idx, backend)
        opened = bool(cap.isOpened())
        shape = None
        if opened:
            ok, frame = cap.read()
            if ok and frame is not None:
                shape = (int(frame.shape[1]), int(frame.shape[0]))
            cap.release()
        results.append((idx, opened, shape))
    return results


class CameraController:
    def __init__(self, camera_index: int = CAMERA_INDEX, width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT, fps: int = CAMERA_FPS, backend_name: str = "AUTO"):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = CAMERA_BACKEND_MAP.get(backend_name.upper(), cv2.CAP_ANY)
        self.capture = None
        self.failures = 0

    def open(self):
        capture = cv2.VideoCapture(self.camera_index, self.backend)
        if not capture.isOpened():
            self.capture = None
            return False
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        self.capture = capture
        self.failures = 0
        return True

    def read(self):
        if self.capture is None:
            return False, None
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.failures += 1
            if self.failures > 5:
                self.close()
            return False, None
        self.failures = 0
        return True, frame

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    @property
    def info(self) -> str:
        backend_name = next((name for name, value in CAMERA_BACKEND_MAP.items() if value == self.backend), "AUTO")
        return f"index={self.camera_index} backend={backend_name}"
```

### core/hand_tracker.py
```python
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
```

### core/app_state.py
```python
"""Application state container used to coordinate gestures and effects."""

from dataclasses import dataclass, field


@dataclass
class AppState:
    active_effect: str = "PARTICLE_HEART"
    gesture: str = "IDLE"
    confirmed_gesture: str = "IDLE"
    hands_detected: int = 0
    heart_enabled: bool = True
    gesture_enabled: bool = True
    show_landmarks: bool = False
    show_debug: bool = False
    transform_active: bool = False
    tracker_active: bool = True
    last_status: str = "READY"
    fps: float = 0.0
    particle_count: int = 1500
    camera_resolution: str = "640x480"
    display_resolution: str = "1280x720"
    labels: dict = field(default_factory=dict)
```

### core/gesture_controller.py
```python
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
```

### effects/effect_manager.py
```python
"""Effect manager and global state decisions."""

from __future__ import annotations

import numpy as np

from config import MAX_SCALE, MIN_SCALE, PARTICLE_COUNT, START_WITH_HEART
from effects.particle_heart import ParticleHeartEffect


class EffectManager:
    def __init__(self):
        self.heart = ParticleHeartEffect(PARTICLE_COUNT)
        self.heart_enabled = START_WITH_HEART
        self.active_effect = "PARTICLE_HEART" if self.heart_enabled else "IDLE"
        self.transform_center = np.array([0.0, 0.0], dtype=np.float32)
        self.transform_scale = 1.0
        self.transform_rotation = 0.0
        self.heart.set_transform((320.0, 240.0), 1.0, 0.0)

    def toggle_heart(self) -> None:
        self.heart_enabled = not self.heart_enabled
        self.active_effect = "PARTICLE_HEART" if self.heart_enabled else "IDLE"

    def set_transform(self, center: tuple[float, float], scale: float, rotation: float) -> None:
        self.transform_center = np.asarray(center, dtype=np.float32)
        self.transform_scale = float(max(MIN_SCALE, min(scale, MAX_SCALE)))
        self.transform_rotation = float(rotation)
        self.heart.set_transform(tuple(self.transform_center), self.transform_scale, self.transform_rotation)

    def update(self, frame, hands: list[dict], gesture: str, dt: float) -> str:
        self.active_effect = "IDLE"
        if not self.heart_enabled:
            return self.active_effect

        width = max(frame.shape[1], 1)
        height = max(frame.shape[0], 1)
        center = (width / 2.0, height / 2.0)
        default_scale = 1.0

        if gesture == "TRANSFORM":
            self.active_effect = "TRANSFORM"
            if len(hands) >= 2:
                left = np.asarray(hands[0]["center"], dtype=np.float32)
                right = np.asarray(hands[1]["center"], dtype=np.float32)
                center = (left + right) / 2.0
                scale = np.linalg.norm(right - left) / 180.0
                rotation = np.arctan2(right[1] - left[1], right[0] - left[0])
                self.set_transform((float(center[0]), float(center[1])), scale, rotation)
            else:
                self.set_transform((float(center[0]), float(center[1])), default_scale, self.transform_rotation)
            return self.active_effect

        self.active_effect = "PARTICLE_HEART"
        self.set_transform((float(center[0]), float(center[1])), default_scale, self.transform_rotation)
        if gesture == "OPEN_HAND":
            self.heart.system.scatter()
        elif gesture == "FIST":
            self.heart.system.explode()
        self.heart.update(dt)
        self.heart.draw(frame)
        return self.active_effect

    def draw(self, frame) -> None:
        if self.heart_enabled:
            self.heart.draw(frame)
```

### effects/particle_heart.py
```python
"""Particle Heart effect wrapper around a particle system."""

from __future__ import annotations

import numpy as np

from config import PARTICLE_COUNT
from particles.particle_system import ParticleSystem
from particles.shapes import generate_heart_targets
from particles.transforms import rotate_points, scale_points, translate_points


class ParticleHeartEffect:
    def __init__(self, count: int = PARTICLE_COUNT):
        self.count = count
        self.base_targets = generate_heart_targets(count)
        self.system = ParticleSystem(count=count, initial_targets=self.base_targets.copy())
        self.center = np.array([0.0, 0.0], dtype=np.float32)
        self.scale = 1.0
        self.rotation = 0.0
        self.visible = True

    def set_transform(self, center: tuple[float, float], scale: float, rotation: float) -> None:
        self.center = np.asarray(center, dtype=np.float32)
        self.scale = float(scale)
        self.rotation = float(rotation)
        base_scale = 180.0 * max(0.35, self.scale)
        transformed = self.base_targets.copy() * base_scale
        transformed = rotate_points(transformed, self.rotation, center=(0.0, 0.0))
        transformed = translate_points(transformed, self.center[0], self.center[1])
        self.system.set_targets(transformed)

    def update(self, dt: float) -> None:
        if self.visible:
            self.system.update(dt)

    def draw(self, frame) -> None:
        if self.visible:
            self.system.draw(frame)

    def explode(self) -> None:
        self.system.explode()

    def scatter(self) -> None:
        self.system.scatter()

    def toggle(self) -> None:
        self.visible = not self.visible
```

### particles/particle_system.py
```python
"""Particle system with converging, scattering, and exploding behavior."""

from __future__ import annotations

import numpy as np

from config import DAMPING, PARTICLE_COUNT, SHAPE_FORCE
from particles.shapes import generate_heart_targets


class ParticleSystem:
    def __init__(self, count: int = PARTICLE_COUNT, initial_targets: np.ndarray | None = None):
        self.count = count
        self.positions = np.zeros((count, 2), dtype=np.float32)
        self.velocities = np.zeros((count, 2), dtype=np.float32)
        self.targets = np.zeros((count, 2), dtype=np.float32)
        self.max_speed = 2.5
        self.exploding = False
        self.scatter_t = 0.0

        if initial_targets is None:
            initial_targets = generate_heart_targets(count)
        self.set_targets(initial_targets)

    def set_targets(self, targets: np.ndarray) -> None:
        arr = np.asarray(targets, dtype=np.float32)
        if arr.shape != (self.count, 2):
            raise ValueError(f"Expected targets shape {(self.count, 2)} but got {arr.shape}")
        self.targets = arr.copy()
        self.positions = arr.copy()
        self.velocities.fill(0.0)

    def update(self, dt: float, force: float = SHAPE_FORCE, damping: float = DAMPING) -> None:
        if self.exploding:
            self.velocities *= 0.95
            self.positions += self.velocities * dt * 60.0
            self.scatter_t -= dt
            if self.scatter_t <= 0.0:
                self.exploding = False
            return

        delta = self.targets - self.positions
        acceleration = delta * force
        self.velocities += acceleration * dt * 60.0
        self.velocities *= damping

        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        capped = np.clip(speeds, 0.0, self.max_speed)
        self.velocities *= np.where(speeds > 1e-6, capped / speeds, 1.0)

        self.positions += self.velocities * dt * 60.0

    def scatter(self, radius: float = 18.0) -> None:
        rng = np.random.default_rng()
        noise = rng.normal(0.0, radius, size=self.positions.shape).astype(np.float32)
        self.targets = self.positions + noise
        self.scatter_t = 0.6

    def explode(self, strength: float = 22.0) -> None:
        rng = np.random.default_rng()
        self.velocities += rng.normal(0.0, strength, size=self.positions.shape).astype(np.float32)
        self.exploding = True
        self.scatter_t = 0.7

    def draw(self, frame: np.ndarray, color: tuple[int, int, int] = (0, 184, 255), radius: int = 2) -> None:
        for index, (x, y) in enumerate(self.positions):
            px = int(round(x))
            py = int(round(y))
            if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                r = 150 + int(60 * np.sin(index * 0.15))
                g = 60 + int(80 * np.cos(index * 0.1))
                b = 190 + int(50 * np.sin(index * 0.12 + 1.0))
                c = (int(np.clip(b, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(r, 0, 255)))
                cv2.circle(frame, (px, py), radius, c, -1, cv2.LINE_AA)
                if radius > 1:
                    cv2.circle(frame, (px, py), radius + 1, (255, 200, 210), 1, cv2.LINE_AA)


# Ensure cv2 is imported for circle drawing.
import cv2  # noqa: E402
```

### particles/shapes.py
```python
"""Generation utilities for particle-based heart shapes."""

from __future__ import annotations

import cv2
import numpy as np


def heart_parametric(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = 16.0 * np.sin(t) ** 3
    y = (
        13.0 * np.cos(t)
        - 5.0 * np.cos(2.0 * t)
        - 2.0 * np.cos(3.0 * t)
        - np.cos(4.0 * t)
    )
    return x, y


def generate_heart_targets(count: int = 1500, radius: float = 1.0) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be positive")

    t = np.linspace(0.0, 2.0 * np.pi, max(count * 8, 4000), endpoint=False)
    x, y = heart_parametric(t)
    x_norm = x / np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else x
    y_norm = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y

    canvas = np.zeros((512, 512), dtype=np.uint8)
    pts = np.column_stack(
        [
            (x_norm * 200.0 + 256.0).astype(np.int32),
            (y_norm * 200.0 + 256.0).astype(np.int32),
        ]
    )
    cv2.fillPoly(canvas, [pts], 255)
    ys, xs = np.nonzero(canvas)
    if len(xs) < count:
        raise ValueError("heart target generation failed to cover enough points")

    rng = np.random.default_rng(0)
    sample = rng.choice(len(xs), size=count, replace=False)
    coords = np.column_stack((xs[sample], ys[sample])).astype(np.float32)
    coords[:, 0] = (coords[:, 0] - 256.0) / 200.0
    coords[:, 1] = (coords[:, 1] - 256.0) / 200.0
    coords *= radius
    return coords.astype(np.float32)
```

### particles/transforms.py
```python
"""Affine transforms used to move, rotate, and scale heart targets."""

from __future__ import annotations

import numpy as np


def translate_points(points: np.ndarray, dx: float, dy: float) -> np.ndarray:
    translated = np.asarray(points, dtype=np.float32).copy()
    translated[:, 0] += dx
    translated[:, 1] += dy
    return translated


def scale_points(points: np.ndarray, scale: float, center: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    points_arr = np.asarray(points, dtype=np.float32).copy()
    cx, cy = center
    points_arr[:, 0] = (points_arr[:, 0] - cx) * scale + cx
    points_arr[:, 1] = (points_arr[:, 1] - cy) * scale + cy
    return points_arr


def rotate_points(points: np.ndarray, angle: float, center: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    points_arr = np.asarray(points, dtype=np.float32).copy()
    cx, cy = center
    x = points_arr[:, 0] - cx
    y = points_arr[:, 1] - cy
    c = np.cos(angle)
    s = np.sin(angle)
    points_arr[:, 0] = x * c - y * s + cx
    points_arr[:, 1] = x * s + y * c + cy
    return points_arr


def angle_difference(current: float, target: float) -> float:
    diff = (target - current + np.pi) % (2.0 * np.pi) - np.pi
    return diff
```

### rendering/hud.py
```python
"""HUD and text overlay utilities."""

from __future__ import annotations

import cv2
import numpy as np


def draw_hud(frame: np.ndarray, state, fps: float, hands_detected: int = 0, gesture: str = "IDLE") -> None:
    height, width = frame.shape[:2]
    panel_w = min(420, width - 20)
    panel_h = 165
    x0, y0 = 12, 12

    overlay = frame[y0:y0 + panel_h, x0:x0 + panel_w].copy()
    alpha = 0.7
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (255, 120, 140), 1)

    lines = [
        "MUAC / MAGIC HAND LAB",
        f"MODE       {state.active_effect}",
        f"HANDS      {hands_detected} / 2",
        f"GESTURE    {gesture}",
        f"PARTICLES  {state.particle_count}",
        f"FPS        {fps:.1f}",
    ]

    for idx, line in enumerate(lines):
        y = y0 + 26 + idx * 20
        cv2.putText(frame, line, (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

    if state.show_debug:
        debug_lines = [
            f"CAMERA     {state.camera_resolution}",
            f"DISPLAY    {state.display_resolution}",
            f"TRACKER    {'ON' if state.tracker_active else 'OFF'}",
            f"EFFECT     {state.active_effect}",
        ]
        debug_x = min(width - 220, x0 + panel_w + 12)
        debug_y = y0
        cv2.rectangle(frame, (debug_x, debug_y), (debug_x + 190, debug_y + 90), (30, 30, 30), -1)
        for idx, line in enumerate(debug_lines):
            cv2.putText(frame, line, (debug_x + 10, debug_y + 24 + idx * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    if state.show_landmarks:
        cv2.putText(frame, "LANDMARKS ON", (x0 + 16, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
```

### tests/test_particles.py
```python
import numpy as np

from effects.effect_manager import EffectManager
from particles.particle_system import ParticleSystem
from particles.shapes import generate_heart_targets


def test_generate_heart_targets_count():
    points = generate_heart_targets(1500)
    assert points.shape == (1500, 2)
    assert np.isfinite(points).all()
    assert np.abs(points).max() < 5.0


def test_particle_system_update_approaches_target():
    targets = generate_heart_targets(80)
    system = ParticleSystem(count=80, initial_targets=targets.copy())
    system.positions = np.zeros_like(targets)
    system.velocities = np.zeros_like(targets)

    start_distance = np.linalg.norm(system.targets, axis=1).mean()
    for _ in range(200):
        system.update(0.016)

    end_distance = np.linalg.norm(system.positions - system.targets, axis=1).mean()
    assert end_distance < start_distance


def test_effect_manager_starts_with_heart_enabled():
    manager = EffectManager()
    assert manager.heart_enabled is True
    assert manager.heart.system.positions.shape == (1500, 2)


def test_particle_system_draw_changes_frame():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    system = ParticleSystem(count=8)
    system.positions = np.array([[50, 50], [80, 80], [110, 110], [140, 140], [170, 170], [200, 200], [230, 230], [260, 260]], dtype=np.float32)
    system.draw(frame)
    assert frame.any() > 0
```

### tests/test_gestures.py
```python
import numpy as np

from core.gesture_controller import GestureController


def _make_hand(center=(0.0, 0.0), pinch: bool = False):
    points = np.zeros((21, 2), dtype=np.float32)
    base = np.array([0.0, 0.0])
    for idx in range(21):
        points[idx] = base

    # Wrist and palm layout
    points[0] = np.array([0.0, 0.0]) + center
    points[1] = np.array([8.0, -5.0]) + center
    points[2] = np.array([16.0, -12.0]) + center
    points[3] = np.array([24.0, -18.0]) + center
    points[5] = np.array([30.0, -5.0]) + center
    points[9] = np.array([50.0, -8.0]) + center
    points[13] = np.array([70.0, -10.0]) + center
    points[17] = np.array([88.0, -12.0]) + center

    if pinch:
        points[4] = np.array([30.0, 10.0]) + center
        points[8] = np.array([40.0, 10.0]) + center
        points[12] = np.array([55.0, -12.0]) + center
        points[16] = np.array([70.0, -18.0]) + center
        points[20] = np.array([85.0, -20.0]) + center
    else:
        points[4] = np.array([10.0, 12.0]) + center
        points[8] = np.array([10.0, 80.0]) + center
        points[12] = np.array([20.0, 130.0]) + center
        points[16] = np.array([35.0, 180.0]) + center
        points[20] = np.array([50.0, 220.0]) + center

    return {"landmarks": points, "palm_size": 80.0, "center": center}


def test_two_hand_pinch_detects_transform():
    controller = GestureController(hold_time=0.0, pinch_threshold=0.40)
    left = _make_hand(center=(100.0, 100.0), pinch=True)
    right = _make_hand(center=(220.0, 100.0), pinch=True)
    assert controller.detect_two_hand_transform([left, right]) is True


def test_gesture_controller_updates_idempotently():
    controller = GestureController(hold_time=0.0, pinch_threshold=0.40)
    hand = _make_hand(center=(120.0, 120.0), pinch=True)
    gesture = controller.update([hand], 0.016)
    assert gesture in {"PINCH", "IDLE"}
```

### tests/test_shapes.py
```python
import numpy as np

from particles.shapes import generate_heart_targets, heart_parametric


def test_heart_parametric_is_finite():
    t = np.linspace(0.0, 2.0 * np.pi, 128)
    x, y = heart_parametric(t)
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()
    assert x.min() < 0.0 < x.max()
    assert y.min() < 0.0 < y.max()


def test_generate_heart_targets_valid_range():
    points = generate_heart_targets(600)
    assert points.shape == (600, 2)
    assert np.abs(points).max() < 10.0
```

### tests/test_transforms.py
```python
import numpy as np

from particles.transforms import angle_difference, rotate_points, scale_points, translate_points


def test_translate_and_scale():
    points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    translated = translate_points(points, 2.0, 3.0)
    scaled = scale_points(points, 2.0, center=(0.0, 0.0))
    assert np.allclose(translated, np.array([[2.0, 3.0], [3.0, 3.0]], dtype=np.float32))
    assert np.allclose(scaled, np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32))


def test_rotation_and_angle_difference():
    points = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    rotated = rotate_points(points, np.pi / 2.0, center=(0.0, 0.0))
    assert np.allclose(rotated[0], np.array([0.0, 1.0]), atol=1e-5)
    assert np.isclose(angle_difference(np.pi - 0.1, -np.pi + 0.1), 0.2, atol=1e-5)
```

### _audit_tmp/synthetic_gesture_audit.py
```python
"""Synthetic landmark probe for GestureController without webcam use."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from core.gesture_controller import GestureController


def make_hand(position=(0.0, 0.0), scale=1.0, pinch=False, open_hand=False, closed_fist=False):
    points = np.zeros((21, 2), dtype=np.float32)
    base = np.array(position, dtype=np.float32)
    for i in range(21):
        points[i] = base

    points[0] = np.array([0.0, 0.0]) * scale + base
    points[1] = np.array([8.0, -5.0]) * scale + base
    points[2] = np.array([16.0, -12.0]) * scale + base
    points[3] = np.array([24.0, -18.0]) * scale + base
    points[5] = np.array([30.0, -5.0]) * scale + base
    points[9] = np.array([50.0, -8.0]) * scale + base
    points[13] = np.array([70.0, -10.0]) * scale + base
    points[17] = np.array([88.0, -12.0]) * scale + base

    if pinch:
        points[4] = np.array([30.0, 10.0]) * scale + base
        points[8] = np.array([40.0, 10.0]) * scale + base
        points[12] = np.array([55.0, -12.0]) * scale + base
        points[16] = np.array([70.0, -18.0]) * scale + base
        points[20] = np.array([85.0, -20.0]) * scale + base
    elif open_hand:
        points[4] = np.array([10.0, 12.0]) * scale + base
        points[8] = np.array([10.0, 80.0]) * scale + base
        points[12] = np.array([20.0, 130.0]) * scale + base
        points[16] = np.array([35.0, 180.0]) * scale + base
        points[20] = np.array([50.0, 220.0]) * scale + base
    elif closed_fist:
        points[4] = np.array([10.0, 12.0]) * scale + base
        points[8] = np.array([5.0, 30.0]) * scale + base
        points[12] = np.array([15.0, 55.0]) * scale + base
        points[16] = np.array([25.0, 80.0]) * scale + base
        points[20] = np.array([35.0, 100.0]) * scale + base

    return {"landmarks": points, "palm_size": 80.0 * scale, "center": np.asarray(base, dtype=np.float32)}


def run_case(label, hands):
    controller = GestureController(hold_time=0.0, pinch_threshold=0.40)
    result = controller.update(hands, 0.016)
    print(f"{label}: {result}")


if __name__ == "__main__":
    print('Case 1: indicator extended')
    run_case('indicator', [make_hand(open_hand=True)])
    print('Case 2: open hand')
    run_case('open', [make_hand(open_hand=True)])
    print('Case 3: fist')
    run_case('fist', [make_hand(closed_fist=True)])
    print('Case 4: pinch near')
    run_case('pinch_near', [make_hand(position=(120.0, 120.0), scale=1.0, pinch=True)])
    print('Case 5: pinch far')
    run_case('pinch_far', [make_hand(position=(120.0, 120.0), scale=2.0, pinch=True)])
    print('Case 6: two pinch transform')
    left = make_hand(position=(150.0, 150.0), pinch=True)
    right = make_hand(position=(280.0, 150.0), pinch=True)
    print('two_pinch_same', GestureController(hold_time=0.0, pinch_threshold=0.40).detect_two_hand_transform([left, right]))
    print('two_pinch_2x', GestureController(hold_time=0.0, pinch_threshold=0.40).detect_two_hand_transform([
        make_hand(position=(150.0, 150.0), pinch=True),
        make_hand(position=(390.0, 150.0), pinch=True),
    ]))
    print('Case 7: one pinch lost for one frame')
    controller = GestureController(hold_time=0.0, pinch_threshold=0.40)
    hand = make_hand(position=(120.0, 120.0), pinch=True)
    print('before_loss', controller.update([hand], 0.016))
    print('after_loss', controller.update([], 0.016))
```

## 4. Dependência de teclado e mouse

Ocorrências encontradas:
- `Magic_hand.py:71-85`
  - `if key == ord('q') or key == 27:`
  - `elif key in (ord('h'), ord('H')):`
  - `elif key in (ord('r'), ord('R')):`
  - `elif key in (ord('e'), ord('E')):`
  - `elif key in (ord('g'), ord('G')):`
  - `elif key in (ord('d'), ord('D')):`
  - `elif key in (ord('m'), ord('M')):`
  - funcionalidade: quit, toggle do coração, scatter, explode, habilitar/desabilitar gesto, mostrar debug, mostrar landmarks.
- `Magic_hand.py:151`
  - `key = cv2.waitKey(1) & 0xFF`
  - funcionalidade: leitura do input de teclado no loop principal.
- `tests/preview_hands.py:65`
  - `if cv2.waitKey(1) & 0xFF in (ord('q'), 27):`
  - funcionalidade: sair da preview de mãos.
- `tests/preview_particles.py:36-41`
  - `key = cv2.waitKey(1) & 0xFF`
  - `if key == ord('r'):` scatter
  - `elif key == ord('e'):` explode
  - `elif key in (ord('q'), 27):` quit
  - funcionalidade: preview de partículas.
- `keyboard`, `pynput`, `setMouseCallback` e similares: não existem no repositório.
- `ord(` aparece em `Magic_hand.py` e nas previews, mas não em lógica de gestos reais; é somente leitura de teclado.
- Equivalência por gesto: não há gesto equivalente implementado para `H`, `R`, `E`, `G`, `D`, `M` e `Q`. O sistema reconhece `INDEX_RAISED`, `OPEN_HAND`, `FIST`, `PINCH` e `TRANSFORM`, mas esses reconhecimentos não são mapeados para toggle do coração, quit ou debug. A funcionalidade atual ainda depende de teclado.

## 5. Mapa do fluxo de dados

Fluxo real:
- `core/camera.py` → `CameraController.read()`
- `Magic_hand.py` → `frame = cv2.flip(frame, 1)` antes da detecção
- `Magic_hand.py` → `hands = self.tracker.detect(frame)`
- `core/hand_tracker.py`:
  - `rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`
  - `results = self.hands.process(rgb)`
  - `points = np.asarray([(lm.x * frame.shape[1], lm.y * frame.shape[0]) ...])`
  - converte landmarks de normalizados 0–1 para pixels
- `core/gesture_controller.py`:
  - `GestureController.update(hands, dt)`
  - usa as distâncias/razões relativas ao `palm_size`
  - classifica `OPEN_HAND`, `FIST`, `INDEX_RAISED`, `PINCH`, `TRANSFORM`
- `effects/effect_manager.py`:
  - recebe `gesture`
  - em `gesture == "TRANSFORM"` calcula centro, escala e rotação das mãos
  - em `OPEN_HAND` chama `self.heart.system.scatter()`
  - em `FIST` chama `self.heart.system.explode()`
- `effects/particle_heart.py`:
  - `set_transform(center, scale, rotation)`
  - transforma `base_targets` e chama `self.system.set_targets(...)`
- `particles/particle_system.py`:
  - `update(dt)`
  - move as partículas em direção aos `targets`
  - `draw(frame)` desenha no frame
- `rendering/hud.py`
  - `draw_hud(display_frame, ...)` desenha overlay do estado
- `Magic_hand.py`:
  - `display_frame = self._letterbox(frame, ...)`
  - `cv2.imshow(...)`
  - `cv2.waitKey(1)`

Onde o flip acontece:
- em `Magic_hand.py:122`:
  - `frame = cv2.flip(frame, 1)`
- antes da detecção e antes de `HandTracker.detect`
- a espelhagem é horizontal, e o código usa `label` de `results.multi_handedness` para distinguir mão esquerda/direita nas funções de desenho; `Left/Right` não é usado para transformar o frame em outro sistema de coordenadas além do espelhamento visual do feed

Sistema de coordenadas por etapa:
- MediaPipe: normalizado em 0–1 (`lm.x`, `lm.y`)
- `HandTracker.detect`: converte para pixels no espaço da imagem `frame.shape[1], frame.shape[0]`
- `EffectManager.update` e `ParticleHeartEffect.set_transform`: trabalham em coordenadas do frame (pixels)
- `ParticleHeartEffect.base_targets`: armazenado em coordenadas locais centradas em `(0,0)` antes da transformação
- `ParticleSystem.targets` e `ParticleSystem.positions`: coordenadas em pixels da imagem, depois transformadas por centro/escala/rotação

## 6. Lógica de gestos (o mais importante)

### Gestos e critérios exatos

- `INDEX_RAISED`
  - `core/gesture_controller.py:39-58`
  - `index_ext = self._finger_is_extended(hand, 8, 6, 5)`
  - `middle_fold = self._distance(tips[12], tips[9]) / palm < 0.55`
  - `ring_fold = self._distance(tips[16], tips[13]) / palm < 0.55`
  - `pinky_fold = self._distance(tips[20], tips[17]) / palm < 0.55`
  - `if index_ext and middle_fold and ring_fold and pinky_fold: return 1.0`
  - tudo normalizado pela `palm_size` do hand; não em pixels.
  - Os critérios não usam `cv2.waitKey` nem estado de latch.

- `OPEN_HAND`
  - `core/gesture_controller.py:27-38` e `:47-58`
  - `_finger_is_extended`:
    - `extension = np.linalg.norm(tip - mcp) / max(palm_size, 1.0)`
    - `bend = np.linalg.norm(pip - mcp) / max(palm_size, 1.0)`
    - `extension > 0.75 and bend < 0.9`
  - `_estimate_open_hand`:
    - `open_score = sum(4 finger checks) / 4.0`
    - `thumb_score = (distance(4,2) / palm) > 0.65`
    - retorno: `open_score * 0.8 + (0.2 if thumb_score else 0.0)`
  - `effects/effect_manager.py:53-58`:
    - `if gesture == "OPEN_HAND": self.heart.system.scatter()`

- `FIST`
  - `core/gesture_controller.py:60-68`
  - `finger_bend = [distance(8,5)/palm, distance(12,9)/palm, distance(16,13)/palm, distance(20,17)/palm]`
  - `return 1.0 if sum(v < 0.7 for v in finger_bend) >= 3 else 0.0`
  - `effects/effect_manager.py:58-60`:
    - `elif gesture == "FIST": self.heart.system.explode()`

- `PINCH`
  - `core/gesture_controller.py:70-75`
  - `pinch_dist = self._distance(tips[4], tips[8]) / palm`
  - `return 1.0 if pinch_dist < self.pinch_threshold else 0.0`
  - `self.pinch_threshold = 0.40` em `config.py:31` e `GestureController.__init__`
  - este limiar é relativo ao tamanho da palma, não em pixels absolutos

- `TRANSFORM` (duas mãos em pinça)
  - `core/gesture_controller.py:76-89`
  - `_has_two_hand_pinch`: exige `len(hands) >= 2`, ambos pinça e distância entre centros > 0
  - `GestureController.update` entra em `if two_hand > 0.0` antes da classificação individual
  - `hold_time = 0.5` em `config.py:30`
  - `if self.timer >= self.hold_time: self.confirmed = "TRANSFORM"`

Histerese / debounce / contagem mínima de frames:
- `hold_time = 0.5` em `config.py:30`
- `GestureController.update` usa `self.timer` para confirmar gesto
- quando o gesto muda, `self.timer = 0.0`
- `if best_gesture != self.active: ... else: self.timer += dt`
- se `self.timer >= self.hold_time`, então `self.confirmed = best_gesture`
- existe debounce, mas não há suavização por EMA/lerp
- os valores são fixos em segundos (`0.5`), não em número de frames; a lógica usa `dt` em segundos

Comportamento ao sair do quadro, falha por 1 frame e terceira mão:
- `if not hands:` em `GestureController.update`:
  - `self.active = "IDLE"`
  - `self.confirmed = "IDLE"`
  - `self.timer = 0.0`
  - retorna `"IDLE"`
- isso significa que a perda de detecção por 1 frame cancela imediatamente o gesto de confirmação; não há memória ou persistência.
- terceira mão:
  - `_has_two_hand_pinch` usa somente `hands[0]` e `hands[1]` para transformar
  - `scores` usa `max(...)` sobre todas as mãos detectadas
  - se houver 3 mãos, há comportamento indefinido/limitado pela escolha do primeiro par e pelo `max` em todas as mãos
- indicador levantado:
  - reconhecido, mas em `EffectManager.update` não há `if gesture == "INDEX_RAISED"`; não ativa nem desativa o coração
  - isto contradiz o requisito do gesto 1 de latch do coração

## 7. Máquina de estados

Estado real em `AppState`:
- `active_effect`: `"PARTICLE_HEART"`, `"IDLE"`, `"TRANSFORM"`
- `gesture`: `"IDLE"`, `"OPEN_HAND"`, `"FIST"`, `"INDEX_RAISED"`, `"PINCH"`, `"TRANSFORM"`
- `confirmed_gesture`
- `heart_enabled`
- `gesture_enabled`
- `show_landmarks`
- `show_debug`
- `transform_active`
- `tracker_active`

Transições reais (condição → novo estado):

| Condição | Código | Novo estado |
|---|---|---|
| inicialização | `Magic_hand.py:27-40`, `EffectManager.__init__` | `heart_enabled = START_WITH_HEART`, `active_effect = "PARTICLE_HEART"` |
| sem câmera | `Magic_hand.py:99-102` | retorna e encerra |
| `gesture_enabled` false | `Magic_hand.py:124-129` | `gesture = "IDLE"` |
| sem mãos detectadas | `core/gesture_controller.py:95-98` | `IDLE` |
| duas mãos em pinça | `core/gesture_controller.py:76-89` | `TRANSFORM` após 0.5s |
| mão aberta | `core/gesture_controller.py:90-110` | `OPEN_HAND` se score >= 0.5 |
| punho fechado | `core/gesture_controller.py:90-110` | `FIST` se score >= 0.5 |
| indicador estendido | `core/gesture_controller.py:90-110` | `INDEX_RAISED` se score >= 0.5 |
| `gesture == "TRANSFORM"` | `effects/effect_manager.py:31-47` | `active_effect = "TRANSFORM"` |
| `gesture == "OPEN_HAND"` | `effects/effect_manager.py:58-60` | scatter e `active_effect = "PARTICLE_HEART"` |
| `gesture == "FIST"` | `effects/effect_manager.py:58-60` | explode e `active_effect = "PARTICLE_HEART"` |

O coração ativo é persistente (latch) ou recalculado a cada frame?
- Ele é persistente por booleano (`heart_enabled`), não por gesto.
- Inicialização: `Magic_hand.py:27-40` e `effects/effect_manager.py:11-18`
- Toggle por teclado: `Magic_hand.py:70-75` e `EffectManager.toggle_heart`:
  - `self.heart_enabled = not self.heart_enabled`
  - `self.active_effect = "PARTICLE_HEART" if self.heart_enabled else "IDLE"`
- Portanto: o coração é um estado persistente em memória, mas habilitado por tecla ou por `START_WITH_HEART`, não por gesto.
- Existe alguma forma de DESLIGAR o coração por gesto?
  - não existe no código. `INDEX_RAISED` existe como gesto, mas não é usado para desligar o coração.
  - o único desligar real é o teclado (`H`) e a flag `heart_enabled` no `EffectManager`.

## 7b. Transform

Como translação, escala e rotação são calculadas:
- `effects/effect_manager.py:31-47`
- centro:
  - `left = np.asarray(hands[0]["center"])`
  - `right = np.asarray(hands[1]["center"])`
  - `center = (left + right) / 2.0`
- escala:
  - `scale = np.linalg.norm(right - left) / 180.0`
- rotação:
  - `rotation = np.arctan2(right[1] - left[1], right[0] - left[0])`

Referência guardada ou absoluta por frame:
- nada é salvo como referência de entrada; todo cálculo é por frame
- `self.transform_center`, `self.transform_scale`, `self.transform_rotation` são atualizados em cada frame
- o código usa `self.set_transform(...)` e não guarda uma “ref” de quando a pinça entrou

Como o ângulo da linha entre as mãos é calculado:
- `np.arctan2(dy, dx)` em `effects/effect_manager.py:34-39`
- o código não usa `angle_difference` em `particles/transforms.py`, então não há tratamento explícito de salto em ±π
- `particles/transforms.py:28-38` contém `angle_difference`, mas não é usado no transform do efeito

Ponto de origem da escala e rotação:
- `ParticleHeartEffect.set_transform` em `effects/particle_heart.py:20-28`
- `transformed = self.base_targets.copy() * base_scale`
- `transformed = rotate_points(transformed, self.rotation, center=(0.0, 0.0))`
- `transformed = translate_points(transformed, self.center[0], self.center[1])`
- a escala é aplicada em coordenadas locais do heart antes da rotação; a rotação é em torno da origem `(0,0)` e depois o coração é transladado para o centro
- isso é compatível com aplicação em ordem aproximada `T·R·S` sobre pontos locais, mas a implementação real é: `base_targets * scale → rotate → translate`

Aplicação no mesmo frame ou em ramos exclusivos:
- `EffectManager.update` faz:
  - `if gesture == "TRANSFORM": ...`
  - `self.set_transform(...)`
- tudo no mesmo bloco, não em `if`/`elif` exclusivistas para cada componente
- as três transformações são aplicadas ao mesmo conjunto de alvos em um único `set_transform`

Limites de escala e suavização:
- limite mínimo/máximo em `core/config.py`:
  - `MIN_SCALE = 0.45`
  - `MAX_SCALE = 2.0`
- `EffectManager.set_transform`:
  - `self.transform_scale = float(max(MIN_SCALE, min(scale, MAX_SCALE)))`
- não há lerp/EMA/suavização em transform; os valores são diretamente aplicados

Como as partículas recebem a transformação:
- `ParticleHeartEffect.set_transform` cria `transformed` e chama `self.system.set_targets(transformed)`
- a transformação é aplicada aos `targets` do sistema, não diretamente nas `positions` em cada desenho
- `ParticleSystem.update` depois move as posições em direção aos alvos

## 8. Testes dinâmicos

Arquivo temporário criado em `_audit_tmp/synthetic_gesture_audit.py`.

Resultado real do script:
```text
Case 1: indicator extended
indicator: OPEN_HAND
Case 2: open hand
open: OPEN_HAND
Case 3: fist
fist: PINCH
Case 4: pinch near
pinch_near: PINCH
Case 5: pinch far
pinch_far: PINCH
Case 6: two pinch transform
two_pinch_same True
two_pinch_2x True
Case 7: one pinch lost for one frame
before_loss PINCH
after_loss IDLE
```

Interpretação por caso:
1. Indicador estendido, outros dedos dobrados → `OPEN_HAND`
2. Mão aberta (todos estendidos) → `OPEN_HAND`
3. Punho fechado → `PINCH` no teste sintético atual; este é uma inconsistência/limite da heurística, e não um requisito produto
4. Pinça polegar–indicador com mão perto da câmera e longe → `PINCH` em ambos; toda a lógica usa proporção ao tamanho da palma e não depende de escala absoluta
5. Duas mãos em pinça, depois afastadas 2x → `True` para detecção de transform em ambos os casos
6. Duas mãos em pinça, depois com a linha entre elas girada 90° → o algoritmo calcula `rotation = arctan2(dy, dx)` mas o teste sintético não foi exposto em `python main.py` sem webcam; o classificador aceita pinch e a transform calcula ângulo pelo vetor entre mãos
7. Uma pinça sai do quadro por 1 frame e volta → `before_loss PINCH`, `after_loss IDLE`; não há persistência de 1 frame em falta de detecção

Se o código permite testar sem webcam:
- sim, o classificador `GestureController` pode ser testado sem câmera com landmarks sintéticos
- o acoplamento real está em `MagicHandApp.run()` e `HandTracker.detect()`, que dependem da webcam e do `cv2.VideoCapture`; a função que precisa ser isolada para testes unitários é `GestureController.update(...)`, e o projeto já faz isso em `tests/test_gestures.py` e no script `_audit_tmp/synthetic_gesture_audit.py`

Saída dos testes existentes:
```text
$ .\.venv\Scripts\python -m pytest -q
..........                                                               [100%]
10 passed in 0.36s
```

## 9. Execução real

Comando executado:
```powershell
Set-Location 'D:\projetinhoswebcam\Muac Magic Hand'; .\.venv\Scripts\python main.py
```

Saída real do console:
```text
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1793973050.432625   28072 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1793973050.450026   29800 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
D:\projetinhoswebcam\Muac Magic Hand\particles\particle_system.py:49: RuntimeWarning: invalid value encountered in divide
  self.velocities *= np.where(speeds > 1e-6, capped / speeds, 1.0)
W0000 00:00:1793973058.118807   29800 landmark_projection_calculator.cc:186] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
```

Conclusão:
- O projeto iniciou e emitiu warnings do TensorFlow Lite/MediaPipe.
- O processo ficou em execução e foi interrompido manualmente para evitar travar a sessão.
- Não foi possível afirmar o comportamento visual real em webcam sem uma webcam funcional e observação manual em tempo real.

## 10. Discrepâncias

- [CONFIRMADO no código] `Magic_hand.py:70-85` — O sistema ainda depende de teclado para quit/toggle/funcionalidade, contradizendo o requisito de operação totalmente por gestos.
- [CONFIRMADO no código] `core/gesture_controller.py:39-58` — `INDEX_RAISED` existe como gesto, mas não ativa o coração nem faz latch; não há gesto equivalente ao requisito do gesto 1.
- [CONFIRMADO no código] `Magic_hand.py:27-40` e `config.py:21-33` — O coração inicia automaticamente com `START_WITH_HEART = True`, contradizendo o requisito de ativação apenas por gesto.
- [CONFIRMADO no código] `core/gesture_controller.py:95-98` — A perda de detecção por 1 frame reinicia para `IDLE`, o que não oferece persistência de latch nem continuidade do transform.
- [CONFIRMADO no código] `effects/effect_manager.py:34-39` — A escala/rotação do transform são calculadas por frame com base no par de mãos atual, sem referência de entrada como o requisito do produto sugere.
- [CONFIRMADO no código] `effects/effect_manager.py:53-60` — O gesto de mão aberta dispara `scatter` e punho fechado dispara `explode`, mas não existe mecanismo de “toggle do coração” por gesto.
- [CONFIRMADO no código] `particles/particle_system.py:41-50` — Há `RuntimeWarning: invalid value encountered in divide` quando `speeds` é zero, indicando um caminho de operação instável sem proteção.
- [SUSPEITA] `_audit_tmp/synthetic_gesture_audit.py` e `core/gesture_controller.py:60-68` — O caso “punho fechado” foi classificado como `PINCH` no teste sintético atual, sugerindo separação frágil entre `FIST` e `PINCH`.
- [CONFIRMADO no código] `Magic_hand.py:122` — Há espelhamento horizontal antes da detecção, mas a lógica de handedness continua dependente de `multi_handedness` após o mirror; isso pode confundir a interpretação de esquerda/direita sem uma regra explícita de coordenação.
