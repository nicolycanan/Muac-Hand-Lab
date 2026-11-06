"""Captura de vídeo e verificação dos dispositivos disponíveis no OpenCV."""

from __future__ import annotations

from threading import Event

import cv2
import numpy as np

from config import (
    CAMERA_BACKEND_MAP,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_PROBE_END,
    CAMERA_PROBE_START,
    CAMERA_WIDTH,
)


def list_camera_indices(
    start: int = CAMERA_PROBE_START,
    stop: int = CAMERA_PROBE_END,
    backend_name: str = "AUTO",
    active_index: int | None = None,
    active_available: bool = False,
    cancel_event: Event | None = None,
) -> list[tuple[int, bool, tuple[int, int] | None]]:
    """Test indexes and report availability only after a valid BGR frame is received."""
    backend = CAMERA_BACKEND_MAP.get(backend_name.upper(), cv2.CAP_ANY)
    results: list[tuple[int, bool, tuple[int, int] | None]] = []
    for idx in range(start, stop + 1):
        if cancel_event is not None and cancel_event.is_set():
            break
        if idx == active_index:
            results.append((idx, active_available, None))
            continue

        shape = None
        cap = None
        try:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ok, frame = cap.read()
                if (
                    ok
                    and isinstance(frame, np.ndarray)
                    and frame.size > 0
                    and frame.ndim == 3
                    and frame.shape[0] > 0
                    and frame.shape[1] > 0
                    and frame.shape[2] == 3
                ):
                    shape = (int(frame.shape[1]), int(frame.shape[0]))
        except cv2.error:
            shape = None
        finally:
            if cap is not None:
                cap.release()

        results.append((idx, shape is not None, shape))
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
            capture.release()
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
