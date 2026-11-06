"""Diagnóstico manual do controlador de gestos com pontos sintéticos, sem webcam."""

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
