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
