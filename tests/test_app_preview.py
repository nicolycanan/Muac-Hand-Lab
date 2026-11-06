import runpy
from pathlib import Path

import cv2
import numpy as np

import Magic_hand
from core import camera, hand_tracker
from core.hand_tracker import HandTracker
from test_gestures import _make_hand


class FrameCamera:
    info = "synthetic"

    def __init__(self, **kwargs):
        self.closed = False

    def open(self):
        return True

    def read(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, 0] = (5, 25, 245)
        return True, frame

    def close(self):
        self.closed = True


class FrameTracker:
    is_active = True

    def __init__(self, **kwargs):
        self.closed = False
        self.detections = 0
        self.draws = 0

    def detect(self, frame):
        # Os dois pontos de entrada devem fornecer o mesmo quadro BGR espelhado na horizontal.
        assert tuple(frame[0, -1]) == (5, 25, 245)
        assert not frame[0, 0].any()
        self.detections += 1
        return [_make_hand(center=(240, 210))]

    def draw_landmarks(self, frame, hands):
        assert hands[0]["landmarks"].shape == (21, 2)
        self.draws += 1
        HandTracker.draw_landmarks(frame, hands)

    def close(self):
        self.closed = True


def test_main_loop_preserves_preview_skeleton_toggle_and_hud(monkeypatch):
    source = FrameCamera()
    tracker = FrameTracker()
    monkeypatch.setattr(Magic_hand, "CameraController", lambda **kwargs: source)
    monkeypatch.setattr(Magic_hand, "HandTracker", lambda **kwargs: tracker)
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    shown = []
    keys = iter([ord("m"), ord("q")])
    monkeypatch.setattr(cv2, "imshow", lambda title, image: shown.append(image.shape))
    monkeypatch.setattr(cv2, "waitKey", lambda delay: next(keys))

    lines = []
    put_text = cv2.putText

    def record_text(frame, label, *args):
        lines.append(label)
        return put_text(frame, label, *args)

    monkeypatch.setattr(cv2, "putText", record_text)
    app = Magic_hand.MagicHandApp()
    app.state.show_debug = True
    app.run()

    assert len(shown) == tracker.detections == 2
    assert tracker.draws == 1
    assert source.closed and tracker.closed
    assert "RAW: INDEX_RAISED" in lines
    assert "CANDIDATE: INDEX_RAISED" in lines
    assert "INTERACTION: SHAPE" in lines
    assert "PARTICLES: 2000" in lines
    assert any(line.startswith("LEFT: INDEX_RAISED") for line in lines)
    assert any(line.startswith("FPS: ") for line in lines)


def test_standalone_hand_preview_mirrors_and_draws(monkeypatch):
    source = FrameCamera()
    tracker = FrameTracker()
    monkeypatch.setattr(camera, "CameraController", lambda **kwargs: source)
    monkeypatch.setattr(hand_tracker, "HandTracker", lambda **kwargs: tracker)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: ord("q"))
    shown = []
    monkeypatch.setattr(cv2, "imshow", lambda title, image: shown.append(image.shape))

    runpy.run_path(str(Path(__file__).with_name("preview_hands.py")), run_name="__main__")

    assert shown == [(480, 640, 3)]
    assert tracker.detections == tracker.draws == 1
    assert source.closed and tracker.closed
