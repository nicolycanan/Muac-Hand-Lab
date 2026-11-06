"""Testa os índices de câmera do OpenCV para localizar um dispositivo como a Iriun."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CAMERA_BACKEND, CAMERA_BACKEND_MAP, CAMERA_PROBE_END, CAMERA_PROBE_START
from core.camera import list_camera_indices


def main() -> None:
    backend_name = CAMERA_BACKEND.upper()
    backend = CAMERA_BACKEND_MAP.get(backend_name, cv2.CAP_ANY)
    print(f"Scanning camera indices {CAMERA_PROBE_START}..{CAMERA_PROBE_END} using backend={backend_name}")
    for idx, ok, shape in list_camera_indices(CAMERA_PROBE_START, CAMERA_PROBE_END, backend_name):
        resolution = f"{shape[0]}x{shape[1]}" if shape else "FAIL"
        status = "OK" if ok else "FAIL"
        print(f"[{idx}] {status} {resolution}")
        if ok:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                print(f"  -> index {idx} reported open but could not read a frame")
                continue
            ok_frame, frame = cap.read()
            cap.release()
            if ok_frame and frame is not None:
                print(f"  -> captured frame {frame.shape[1]}x{frame.shape[0]}")
            else:
                print(f"  -> read failed for index {idx}")


if __name__ == "__main__":
    main()
