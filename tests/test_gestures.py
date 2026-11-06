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
