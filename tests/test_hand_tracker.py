from types import SimpleNamespace

import numpy as np

from core.hand_tracker import HandTracker


def _mediapipe_result(label="LEFT"):
    points = [
        SimpleNamespace(x=index / 20.0, y=index / 40.0)
        for index in range(21)
    ]
    hand_landmarks = SimpleNamespace(landmark=points)
    handedness = SimpleNamespace(
        classification=[SimpleNamespace(label=label)]
    )
    return SimpleNamespace(
        multi_hand_landmarks=[hand_landmarks],
        multi_handedness=[handedness],
    )


def test_detect_returns_21_landmarks_and_does_not_modify_camera_frame():
    tracker = HandTracker(max_hands=2, confidence=0.5)
    camera_frame = np.zeros((40, 60, 3), dtype=np.uint8)
    camera_frame[:] = (10, 70, 230)
    original = camera_frame.copy()
    received_rgb = []

    def process(rgb):
        received_rgb.append(rgb.copy())
        return _mediapipe_result("LEFT")

    tracker.hands.process = process
    try:
        hands = tracker.detect(camera_frame)
    finally:
        tracker.close()

    assert np.array_equal(camera_frame, original)
    assert np.array_equal(received_rgb[0][0, 0], np.array([230, 70, 10], dtype=np.uint8))
    assert len(hands) == 1
    assert hands[0]["landmarks"].shape == (21, 2)
    assert hands[0]["label"] == "LEFT"
    assert np.allclose(hands[0]["landmarks"][1], [3.0, 1.0])


def test_draw_landmarks_changes_frame_pixels():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    points = np.array(
        [[20 + (index % 5) * 15, 20 + (index // 5) * 15] for index in range(21)],
        dtype=np.float32,
    )

    HandTracker.draw_landmarks(
        frame,
        [{"landmarks": points, "label": "LEFT"}],
    )

    assert np.any(frame != 0)
    assert np.any(np.all(frame == (0, 255, 255), axis=2))
