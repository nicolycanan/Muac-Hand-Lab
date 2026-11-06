"""Testa os índices de câmera do OpenCV para localizar um dispositivo como a Iriun."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CAMERA_BACKEND, CAMERA_PROBE_END, CAMERA_PROBE_START
from core.camera import list_camera_indices


def main() -> None:
    backend_name = CAMERA_BACKEND.upper()
    print(f"Scanning camera indices {CAMERA_PROBE_START}..{CAMERA_PROBE_END} using backend={backend_name}")
    for idx, ok, shape in list_camera_indices(CAMERA_PROBE_START, CAMERA_PROBE_END, backend_name):
        resolution = f"{shape[0]}x{shape[1]}" if shape else "FAIL"
        status = "AVAILABLE" if ok else "UNAVAILABLE"
        print(f"[{idx}] {status} {resolution}")


if __name__ == "__main__":
    main()
