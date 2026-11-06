"""Prévia do rastreamento e do esqueleto das mãos, sem partículas."""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CAMERA_BACKEND, CAMERA_FPS, CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH
from core.camera import CameraController
from core.hand_tracker import HandTracker


def main() -> None:
    camera = CameraController(
        camera_index=CAMERA_INDEX,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
        backend_name=CAMERA_BACKEND,
    )
    if not camera.open():
        raise RuntimeError("Unable to open webcam for hand preview.")

    tracker = HandTracker(max_hands=2, confidence=0.5)
    prev = cv2.getTickCount()

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                break
            frame = cv2.flip(frame, 1)
            hands = tracker.detect(frame)
            tracker.draw_landmarks(frame, hands)

            ticks = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (ticks - prev)
            prev = ticks
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("MUAC / HAND PREVIEW", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        camera.close()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
