import numpy as np

from config import RECTANGLE_TRACKING_GRACE_FRAMES
from core.gesture_controller import GestureController, RawGesture
from effects.effect_manager import EffectManager
from effects.rectangle_frame import RectangleState


def _make_l_hand(center, label):
    center = np.asarray(center, dtype=np.float32)
    points = np.zeros((21, 2), dtype=np.float32)
    points[0] = center
    for index, offset in {
        1: (-40, -20),
        2: (-55, -25),
        3: (-70, -32),
        4: (-85, -40),
        5: (-35, -25),
        6: (-35, -55),
        7: (-35, -85),
        8: (-35, -115),
    }.items():
        points[index] = center + offset

    for mcp, pip, dip, tip, base in (
        (9, 10, 11, 12, (0, -30)),
        (13, 14, 15, 16, (30, -25)),
        (17, 18, 19, 20, (55, -15)),
    ):
        base_point = center + np.asarray(base, dtype=np.float32)
        points[mcp] = base_point
        points[pip] = base_point + (0, -15)
        points[dip] = base_point + (20, -5)
        points[tip] = base_point + (10, 10)

    return {
        "landmarks": points,
        "palm_size": 40.0,
        "center": center,
        "label": label,
    }


def _make_fist(center, label):
    hand = _make_l_hand(center, label)
    points = hand["landmarks"]
    points[4] = np.asarray(center, dtype=np.float32) + (-25, 5)
    points[8] = np.asarray(center, dtype=np.float32) + (-25, 5)
    points[5] = np.asarray(center, dtype=np.float32) + (-35, -25)
    points[6] = np.asarray(center, dtype=np.float32) + (-35, -40)
    points[7] = np.asarray(center, dtype=np.float32) + (-15, -30)
    points[8] = np.asarray(center, dtype=np.float32) + (-25, -15)
    return hand


def _update(controller, manager, hands, dt=1 / 30):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    gesture = controller.update(hands, dt)
    manager.update(frame, controller, dt)
    return gesture, frame


def _nearby_hands():
    return [
        _make_l_hand((100, 180), "LEFT"),
        _make_l_hand((145, 180), "RIGHT"),
    ]


def test_two_hands_forming_l_are_recognized_and_arm_rectangle():
    controller = GestureController(hold_time=0.0)
    manager = EffectManager(start_with_heart=False)

    gesture, _ = _update(controller, manager, _nearby_hands())

    assert gesture == RawGesture.RECTANGLE
    assert all(hand.rectangle_point is not None for hand in controller.hands.values())
    assert manager.rectangle_frame.state == RectangleState.ARMED


def test_approaching_then_separating_draws_rectangle():
    controller = GestureController(hold_time=0.0)
    manager = EffectManager(start_with_heart=False)
    _, _ = _update(controller, manager, _nearby_hands())

    gesture, frame = _update(
        controller,
        manager,
        [_make_l_hand((100, 180), "LEFT"), _make_l_hand((205, 260), "RIGHT")],
    )

    assert gesture == RawGesture.RECTANGLE
    assert manager.rectangle_frame.state == RectangleState.DRAWING
    assert manager.rectangle_frame.visible
    assert np.any(frame[:, :, 0] == 255)
    assert manager.active_effect == "RECTANGLE_FRAME"
    outline = np.zeros_like(frame)
    manager.rectangle_frame.update(outline, controller.hands, 0.0)
    center = tuple(np.rint(np.mean(manager.rectangle_frame.points, axis=0)).astype(int))
    assert np.array_equal(outline[center[1], center[0]], (0, 0, 0))


def test_hands_already_separated_do_not_arm():
    controller = GestureController(hold_time=0.0)
    manager = EffectManager(start_with_heart=False)
    far_hands = [
        _make_l_hand((100, 180), "LEFT"),
        _make_l_hand((300, 180), "RIGHT"),
    ]

    for _ in range(5):
        _update(controller, manager, far_hands)

    assert manager.rectangle_frame.state == RectangleState.IDLE
    assert not manager.rectangle_frame.visible


def test_rectangle_size_tracks_hand_separation():
    controller = GestureController(hold_time=0.0)
    manager = EffectManager(start_with_heart=False)
    _update(controller, manager, _nearby_hands())
    _update(
        controller,
        manager,
        [_make_l_hand((100, 180), "LEFT"), _make_l_hand((205, 180), "RIGHT")],
    )
    initial_width = abs(manager.rectangle_frame.points[1][0] - manager.rectangle_frame.points[0][0])

    _update(
        controller,
        manager,
        [_make_l_hand((100, 180), "LEFT"), _make_l_hand((265, 180), "RIGHT")],
    )
    new_width = abs(manager.rectangle_frame.points[1][0] - manager.rectangle_frame.points[0][0])

    assert new_width > initial_width


def test_changing_gesture_fades_rectangle():
    controller = GestureController(hold_time=0.0)
    manager = EffectManager(start_with_heart=False)
    _update(controller, manager, _nearby_hands())
    _update(
        controller,
        manager,
        [_make_l_hand((100, 180), "LEFT"), _make_l_hand((205, 180), "RIGHT")],
    )

    _update(
        controller,
        manager,
        [_make_fist((100, 180), "LEFT"), _make_fist((205, 180), "RIGHT")],
    )

    assert manager.rectangle_frame.state == RectangleState.CLOSING
    assert 0.0 < manager.rectangle_frame.alpha < 1.0


def test_short_tracking_loss_does_not_close_rectangle_immediately():
    controller = GestureController(hold_time=0.0)
    manager = EffectManager(start_with_heart=False)
    _update(controller, manager, _nearby_hands())
    _update(
        controller,
        manager,
        [_make_l_hand((100, 180), "LEFT"), _make_l_hand((205, 180), "RIGHT")],
    )

    for _ in range(RECTANGLE_TRACKING_GRACE_FRAMES):
        _update(controller, manager, [])

    assert manager.rectangle_frame.state == RectangleState.DRAWING
    assert manager.rectangle_frame.visible
    _update(controller, manager, [])
    assert manager.rectangle_frame.state == RectangleState.CLOSING


def test_one_hand_cannot_arm_rectangle_and_closes_drawing():
    controller = GestureController(hold_time=0.0)
    manager = EffectManager(start_with_heart=False)
    _update(controller, manager, [_make_l_hand((100, 180), "LEFT")])
    assert manager.rectangle_frame.state == RectangleState.IDLE

    _update(controller, manager, _nearby_hands())
    _update(
        controller,
        manager,
        [_make_l_hand((100, 180), "LEFT"), _make_l_hand((205, 180), "RIGHT")],
    )
    gesture, _ = _update(controller, manager, [_make_l_hand((100, 180), "LEFT")])

    assert gesture == RawGesture.RECTANGLE
    assert controller.event is None
    assert manager.rectangle_frame.state == RectangleState.CLOSING


def test_rectangle_l_does_not_select_heart_or_transform_particles():
    controller = GestureController(hold_time=0.0)
    manager = EffectManager(start_with_heart=False)

    _update(controller, manager, _nearby_hands())

    assert controller.event is None
    assert controller.mode.value != "TRANSFORM"
    assert not controller.detect_two_hand_transform(_nearby_hands())
    assert manager.current_shape is None
    assert not manager.transform_active
