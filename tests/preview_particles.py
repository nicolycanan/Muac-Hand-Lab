"""Preview-only particle heart without MediaPipe."""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CAMERA_BACKEND, CAMERA_FPS, CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH, DISPLAY_HEIGHT, DISPLAY_WIDTH
from effects.effect_manager import EffectManager


def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if CAMERA_BACKEND == "DSHOW" else cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam for particle preview.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    manager = EffectManager()
    manager.heart_enabled = True
    manager.active_effect = "PARTICLE_HEART"
    previous = cv2.getTickCount()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        dt = (cv2.getTickCount() - previous) / cv2.getTickFrequency()
        previous = cv2.getTickCount()
        manager.heart.system.update(dt or 1 / 30.0)
        manager.heart.draw(frame)

        cv2.putText(frame, "MUAC / PARTICLE HEART PREVIEW", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 200), 2)
        cv2.putText(frame, "R: scatter  E: explode  Q: quit", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("MUAC / PARTICLE HEART PREVIEW", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            manager.heart.system.scatter()
        elif key == ord('e'):
            manager.heart.system.explode()
        elif key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
