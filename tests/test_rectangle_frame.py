import numpy as np
import pytest

from config import RECTANGLE_SCREEN_MARGIN, RECTANGLE_TRACKING_GRACE_FRAMES
from core.gesture_controller import GestureController, HandGesture, RawGesture
from effects.effect_manager import EffectManager, TransformTarget, TransformTargetMode
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


def _pinch_hands(first=(180, 200), second=(300, 200)):
    from test_gestures import _make_hand

    return [
        _make_hand(center=first, extended=(), pinch=True, label="LEFT"),
        _make_hand(center=second, extended=(), pinch=True, label="RIGHT"),
    ]


def _pinches_for_rectangle(manager, distance=120.0):
    center = np.mean(manager.rectangle_frame.points, axis=0)
    half_distance = distance * 0.5
    offset = np.asarray((0.0, 65.0), dtype=np.float32)
    first = center + (-half_distance, 0.0) + offset
    second = center + (half_distance, 0.0) + offset
    return _pinch_hands(tuple(first), tuple(second))


def _tracked_rectangle_hand(point):
    point = np.asarray(point, dtype=np.float32)
    return HandGesture(
        RawGesture.RECTANGLE,
        {},
        1.0,
        point,
        point,
        np.zeros(2, dtype=np.float32),
        40.0,
        point,
    )


def _locked_rectangle(start_with_heart=False):
    controller = GestureController(hold_time=0.0)
    manager = EffectManager(start_with_heart=start_with_heart)
    _update(controller, manager, _nearby_hands())
    _update(
        controller,
        manager,
        [_make_l_hand((100, 180), "LEFT"), _make_l_hand((205, 260), "RIGHT")],
    )
    _update(
        controller,
        manager,
        [_make_fist((100, 180), "LEFT"), _make_fist((205, 260), "RIGHT")],
    )
    assert manager.rectangle_frame.state == RectangleState.LOCKED
    return controller, manager


def _activate_rectangle_transform(controller, manager, hands=None):
    hands = hands or _pinches_for_rectangle(manager)
    _update(controller, manager, hands)
    gesture, _ = _update(controller, manager, hands)
    assert gesture == RawGesture.TRANSFORM
    assert manager.transform_active


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
    assert np.any(frame[110:175, 50:135] != 0)
    assert manager.active_effect == "RECTANGLE_FRAME"
    outline = np.zeros_like(frame)
    manager.rectangle_frame.update(outline, controller.hands, 0.0)
    center = tuple(np.rint(np.mean(manager.rectangle_frame.points, axis=0)).astype(int))
    assert np.any(outline[center[1], center[0]] != 0)


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


def test_changing_gesture_locks_rectangle_without_fading():
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

    assert manager.rectangle_frame.state == RectangleState.LOCKED
    assert manager.rectangle_frame.alpha == 1.0
    assert manager.rectangle_frame.visible


def test_tracking_loss_keeps_creation_alive_during_grace_then_locks():
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
    assert manager.rectangle_frame.state == RectangleState.LOCKED
    assert manager.rectangle_frame.visible


def test_one_hand_cannot_arm_rectangle_and_locks_existing_drawing():
    controller = GestureController(hold_time=0.0)
    manager = EffectManager(start_with_heart=False)
    _update(controller, manager, [_make_l_hand((100, 180), "LEFT")])
    assert manager.rectangle_manager.count == 0

    _update(controller, manager, _nearby_hands())
    _update(
        controller,
        manager,
        [_make_l_hand((100, 180), "LEFT"), _make_l_hand((205, 180), "RIGHT")],
    )
    gesture, _ = _update(controller, manager, [_make_l_hand((100, 180), "LEFT")])

    assert gesture == RawGesture.RECTANGLE
    assert controller.event is None
    assert manager.rectangle_frame.state == RectangleState.LOCKED


def test_locked_rectangle_persists_without_hands_and_keeps_geometry():
    controller = GestureController(hold_time=0.0)
    manager = EffectManager(start_with_heart=False)
    _update(controller, manager, _nearby_hands())
    _update(
        controller,
        manager,
        [_make_l_hand((100, 180), "LEFT"), _make_l_hand((205, 260), "RIGHT")],
    )
    for _ in range(RECTANGLE_TRACKING_GRACE_FRAMES + 1):
        _update(controller, manager, [])

    rectangle = manager.rectangle_frame
    assert rectangle.state == RectangleState.LOCKED
    original_points = tuple(point.copy() for point in rectangle.points)

    for _ in range(10):
        _update(controller, manager, [])

    assert rectangle.state == RectangleState.LOCKED
    assert rectangle.visible
    for actual, expected in zip(rectangle.points, original_points):
        np.testing.assert_array_equal(actual, expected)


def test_locked_rectangle_stays_animated_and_ignores_new_l_gestures():
    controller = GestureController(hold_time=0.0)
    manager = EffectManager(start_with_heart=False)
    _update(controller, manager, _nearby_hands())
    _update(
        controller,
        manager,
        [_make_l_hand((100, 180), "LEFT"), _make_l_hand((205, 260), "RIGHT")],
    )
    _update(
        controller,
        manager,
        [_make_fist((100, 180), "LEFT"), _make_fist((205, 260), "RIGHT")],
    )
    rectangle = manager.rectangle_frame
    locked_points = tuple(point.copy() for point in rectangle.points)
    start_time = rectangle.fill.time

    first_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rectangle.update(first_frame, controller.hands, 0.2)
    second_frame = np.zeros_like(first_frame)
    rectangle.update(
        second_frame,
        {
            "LEFT": controller.hands["LEFT"],
            "RIGHT": controller.hands["RIGHT"],
        },
        0.2,
    )
    for actual, expected in zip(rectangle.points, locked_points):
        np.testing.assert_array_equal(actual, expected)
    assert rectangle.fill.time > start_time
    assert not np.array_equal(first_frame, second_frame)

    _update(
        controller,
        manager,
        [_make_l_hand((250, 150), "LEFT"), _make_l_hand((355, 230), "RIGHT")],
    )
    assert rectangle.state == RectangleState.LOCKED
    for actual, expected in zip(rectangle.points, locked_points):
        np.testing.assert_array_equal(actual, expected)


def test_locked_rectangle_does_not_block_heart_letter_or_particle_updates():
    controller = GestureController(hold_time=0.0)
    manager = EffectManager(start_with_heart=False)
    _update(controller, manager, _nearby_hands())
    _update(
        controller,
        manager,
        [_make_l_hand((100, 180), "LEFT"), _make_l_hand((205, 260), "RIGHT")],
    )
    _update(
        controller,
        manager,
        [_make_fist((100, 180), "LEFT"), _make_fist((205, 260), "RIGHT")],
    )
    rectangle = manager.rectangle_frame
    original_points = tuple(point.copy() for point in rectangle.points)

    index_hand = _make_l_hand((100, 180), "LEFT")
    points = index_hand["landmarks"]
    points[1:5] = np.asarray([
        (60, 160), (70, 165), (80, 175), (75, 185),
    ], dtype=np.float32)
    heart_updates = []
    manager.heart.update = lambda dt: heart_updates.append(dt)

    _, _ = _update(controller, manager, [index_hand])
    assert controller.event is not None
    assert manager.current_shape == "HEART"
    assert rectangle.state == RectangleState.LOCKED
    for actual, expected in zip(rectangle.points, original_points):
        np.testing.assert_array_equal(actual, expected)

    two_finger_hand = _make_l_hand((100, 180), "LEFT")
    points = two_finger_hand["landmarks"]
    points[1:5] = np.asarray([
        (60, 160), (70, 165), (80, 175), (75, 185),
    ], dtype=np.float32)
    base = np.asarray((100, 150), dtype=np.float32)
    points[9] = base
    points[10] = base + (0, -30)
    points[11] = base + (0, -60)
    points[12] = base + (0, -90)
    _update(controller, manager, [two_finger_hand])

    assert manager.current_shape not in (None, "HEART")
    assert rectangle.state == RectangleState.LOCKED
    assert heart_updates
    for actual, expected in zip(rectangle.points, original_points):
        np.testing.assert_array_equal(actual, expected)


def test_two_pinches_move_locked_rectangle_and_release_keeps_position():
    controller, manager = _locked_rectangle()
    _activate_rectangle_transform(controller, manager)
    rectangle = manager.rectangle_frame
    initial_center = (rectangle.points[0] + rectangle.points[1]) * 0.5

    _update(controller, manager, _pinch_hands((220, 230), (340, 230)))
    moved_center = (rectangle.points[0] + rectangle.points[1]) * 0.5
    assert moved_center[0] > initial_center[0]
    assert moved_center[1] > initial_center[1]

    moved_points = tuple(point.copy() for point in rectangle.points)
    _update(controller, manager, [])
    assert not manager.rectangle_transform_active
    for actual, expected in zip(rectangle.points, moved_points):
        np.testing.assert_array_equal(actual, expected)


def test_two_pinches_scale_locked_rectangle_up_and_down():
    controller, manager = _locked_rectangle()
    _activate_rectangle_transform(controller, manager)
    rectangle = manager.rectangle_frame
    initial_size = rectangle.points[1] - rectangle.points[0]

    _update(controller, manager, _pinches_for_rectangle(manager, 200.0))
    larger_size = rectangle.points[1] - rectangle.points[0]
    assert larger_size[0] > initial_size[0]
    assert larger_size[1] > initial_size[1]

    _update(controller, manager, _pinches_for_rectangle(manager))
    smaller_size = rectangle.points[1] - rectangle.points[0]
    assert smaller_size[0] < larger_size[0]
    assert smaller_size[1] < larger_size[1]


def test_one_pinch_does_not_move_locked_rectangle():
    from test_gestures import _make_hand

    controller, manager = _locked_rectangle()
    rectangle = manager.rectangle_frame
    original_points = tuple(point.copy() for point in rectangle.points)
    hand = _make_hand(center=(220, 220), extended=(), pinch=True, label="LEFT")

    _update(controller, manager, [hand])
    assert not manager.rectangle_transform_active
    for actual, expected in zip(rectangle.points, original_points):
        np.testing.assert_array_equal(actual, expected)


def test_locked_rectangle_transform_does_not_transform_particles():
    controller, manager = _locked_rectangle(start_with_heart=True)
    _activate_rectangle_transform(controller, manager)
    rectangle = manager.rectangle_frame
    particle_center = manager.transform_center.copy()
    particle_scale = manager.transform_scale
    original_points = tuple(point.copy() for point in rectangle.points)

    _update(controller, manager, _pinch_hands((220, 230), (420, 230)))

    assert manager.rectangle_transform_active
    assert manager.transform_active
    for actual, expected in zip(rectangle.points, original_points):
        assert not np.array_equal(actual, expected)
    np.testing.assert_array_equal(manager.transform_center, particle_center)
    assert manager.transform_scale == particle_scale


def test_transform_line_is_drawn_for_locked_rectangle():
    from rendering.hud import draw_hand_controls

    controller, manager = _locked_rectangle()
    _activate_rectangle_transform(controller, manager)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_hand_controls(frame, controller, manager)

    assert manager.rectangle_transform_active
    assert np.any(frame[115, 10:140])


def test_auto_selection_uses_rectangle_geometry_and_locks_target_for_session():
    controller, manager = _locked_rectangle(start_with_heart=True)
    _activate_rectangle_transform(controller, manager)

    assert manager.transform_target == TransformTarget.RECTANGLE
    rectangle_points = tuple(point.copy() for point in manager.rectangle_frame.points)
    particle_center = manager.transform_center.copy()
    _update(controller, manager, _pinch_hands((220, 320), (340, 320)))
    assert manager.transform_target == TransformTarget.RECTANGLE
    assert not np.array_equal(manager.rectangle_frame.points[0], rectangle_points[0])
    np.testing.assert_array_equal(manager.transform_center, particle_center)


def test_auto_selection_chooses_particles_outside_rectangle_and_does_not_switch():
    controller, manager = _locked_rectangle(start_with_heart=True)
    outside = _pinch_hands((450, 100), (550, 100))
    _activate_rectangle_transform(controller, manager, outside)

    assert manager.transform_target == TransformTarget.PARTICLES
    rectangle_points = tuple(point.copy() for point in manager.rectangle_frame.points)
    particle_center = manager.transform_center.copy()
    _update(controller, manager, _pinch_hands((460, 110), (560, 110)))
    assert manager.transform_target == TransformTarget.PARTICLES
    assert not np.array_equal(manager.transform_center, particle_center)

    _update(controller, manager, _pinches_for_rectangle(manager))
    assert manager.transform_target == TransformTarget.PARTICLES
    for actual, expected in zip(manager.rectangle_frame.points, rectangle_points):
        np.testing.assert_array_equal(actual, expected)


def test_particle_target_transforms_free_cloud_when_rectangle_exists():
    controller, manager = _locked_rectangle()
    outside = _pinch_hands((450, 100), (550, 100))
    _activate_rectangle_transform(controller, manager, outside)
    rectangle_points = tuple(point.copy() for point in manager.rectangle_frame.points)
    particle_positions = manager.heart.system.positions.copy()

    _update(controller, manager, _pinch_hands((500, 140), (600, 140)))

    assert manager.transform_target == TransformTarget.PARTICLES
    assert manager.transform_active
    assert not manager.rectangle_transform_active
    assert not np.array_equal(manager.heart.system.positions, particle_positions)
    for actual, expected in zip(manager.rectangle_frame.points, rectangle_points):
        np.testing.assert_array_equal(actual, expected)


def test_transform_target_manual_override_cycles_and_forces_selection():
    controller, manager = _locked_rectangle(start_with_heart=True)
    assert manager.transform_target_mode == TransformTargetMode.AUTO
    assert manager.cycle_transform_target_mode() == TransformTargetMode.PARTICLES
    _activate_rectangle_transform(controller, manager)
    assert manager.transform_target == TransformTarget.PARTICLES

    _update(controller, manager, [])
    assert manager.cycle_transform_target_mode() == TransformTargetMode.RECTANGLE
    _activate_rectangle_transform(
        controller,
        manager,
        _pinch_hands((450, 100), (550, 100)),
    )
    assert manager.transform_target == TransformTarget.RECTANGLE

    _update(controller, manager, [])
    assert manager.cycle_transform_target_mode() == TransformTargetMode.AUTO


@pytest.mark.parametrize(
    "points",
    [
        ((-80, 120), (40, 260)),
        ((600, 120), (760, 260)),
        ((120, -90), (260, 40)),
        ((120, 420), (260, 600)),
        ((-500, -400), (-300, -200)),
    ],
)
def test_geometry_is_recovered_inside_camera_margin(points):
    _, manager = _locked_rectangle()
    rectangle = manager.rectangle_frame
    rectangle.points = tuple(np.asarray(point, dtype=np.float32) for point in points)
    rectangle.constrain_geometry((480, 640, 3))

    first, second = rectangle.points
    assert first[0] >= RECTANGLE_SCREEN_MARGIN - 1e-5
    assert first[1] >= RECTANGLE_SCREEN_MARGIN - 1e-5
    assert second[0] <= 639 - RECTANGLE_SCREEN_MARGIN + 1e-5
    assert second[1] <= 479 - RECTANGLE_SCREEN_MARGIN + 1e-5


def test_transform_limits_size_and_preserves_aspect_ratio():
    _, manager = _locked_rectangle()
    rectangle = manager.rectangle_frame
    rectangle.points = (
        np.asarray((220, 190), dtype=np.float32),
        np.asarray((420, 270), dtype=np.float32),
    )
    rectangle.update_transform(
        np.asarray((220, 230), dtype=np.float32),
        np.asarray((420, 230), dtype=np.float32),
        (480, 640, 3),
        1 / 60,
    )
    rectangle.update_transform(
        np.asarray((-300, 230), dtype=np.float32),
        np.asarray((800, 230), dtype=np.float32),
        (480, 640, 3),
        1 / 60,
    )

    size = rectangle.points[1] - rectangle.points[0]
    assert size[0] <= 640 - 2 * RECTANGLE_SCREEN_MARGIN
    assert size[1] <= 480 - 2 * RECTANGLE_SCREEN_MARGIN
    assert np.isclose(size[0] / size[1], 200 / 80, rtol=1e-5)
    assert rectangle.points[0][0] >= RECTANGLE_SCREEN_MARGIN
    assert rectangle.points[1][1] <= 480 - 1 - RECTANGLE_SCREEN_MARGIN


@pytest.mark.parametrize(
    "direction",
    [(1, 0), (-1, 0), (0, 1), (0, -1)],
)
def test_transform_movement_stays_inside_each_camera_edge(direction):
    _, manager = _locked_rectangle()
    rectangle = manager.rectangle_frame
    rectangle.points = (
        np.asarray((220, 190), dtype=np.float32),
        np.asarray((420, 290), dtype=np.float32),
    )
    center = np.asarray((320, 240), dtype=np.float32)
    half_line = np.asarray((60, 0), dtype=np.float32)
    baseline = center + np.asarray((0, 65), dtype=np.float32)
    rectangle.update_transform(
        baseline - half_line,
        baseline + half_line,
        (480, 640, 3),
        1 / 60,
    )

    target_center = center + np.asarray(direction, dtype=np.float32) * 1000
    target_line = target_center + np.asarray((0, 65), dtype=np.float32)
    for _ in range(40):
        rectangle.update_transform(
            target_line - half_line,
            target_line + half_line,
            (480, 640, 3),
            1 / 60,
        )

    first, second = rectangle.points
    assert first[0] >= RECTANGLE_SCREEN_MARGIN - 1e-4
    assert first[1] >= RECTANGLE_SCREEN_MARGIN - 1e-4
    assert second[0] <= 639 - RECTANGLE_SCREEN_MARGIN + 1e-4
    assert second[1] <= 479 - RECTANGLE_SCREEN_MARGIN + 1e-4


def _delete_test_rectangle():
    from effects.rectangle_frame import RectangleFrame

    rectangle = RectangleFrame()
    rectangle.state = RectangleState.LOCKED
    rectangle.alpha = 1.0
    rectangle.points = (
        np.asarray((220, 190), dtype=np.float32),
        np.asarray((420, 290), dtype=np.float32),
    )
    return rectangle


@pytest.mark.parametrize("dt", [1 / 30, 1 / 60])
def test_shrink_delete_requires_both_limits_and_hold_time_at_30_and_60_fps(dt):
    from config import RECTANGLE_DELETE_HOLD_SECONDS

    rectangle = _delete_test_rectangle()
    first = np.asarray((220, 240), dtype=np.float32)
    second = np.asarray((420, 240), dtype=np.float32)
    rectangle.update_transform(first, second, (480, 640, 3), dt)
    small_first = np.asarray((315, 240), dtype=np.float32)
    small_second = np.asarray((325, 240), dtype=np.float32)
    for _ in range(60):
        rectangle.update_transform(small_first, small_second, (480, 640, 3), dt)
        if rectangle.delete_pending:
            break
    assert rectangle.delete_pending
    assert rectangle.state == RectangleState.LOCKED

    elapsed = rectangle._delete_elapsed
    while rectangle.state == RectangleState.LOCKED:
        elapsed += dt
        rectangle.update_transform(small_first, small_second, (480, 640, 3), dt)
        if rectangle.state == RectangleState.CLOSING:
            break

    assert rectangle.state == RectangleState.CLOSING
    assert elapsed >= RECTANGLE_DELETE_HOLD_SECONDS


def test_motion_alone_does_not_trigger_shrink_delete():
    rectangle = _delete_test_rectangle()
    rectangle.update_transform(
        np.asarray((220, 240), dtype=np.float32),
        np.asarray((420, 240), dtype=np.float32),
        (480, 640, 3),
        0.5,
    )
    rectangle.update_transform(
        np.asarray((400, 240), dtype=np.float32),
        np.asarray((600, 240), dtype=np.float32),
        (480, 640, 3),
        1.0,
    )

    assert rectangle.state == RectangleState.LOCKED
    assert not rectangle.delete_pending


def test_shrink_delete_is_cancelled_after_regrowth_past_hysteresis():
    rectangle = _delete_test_rectangle()
    rectangle.update_transform(
        np.asarray((220, 240), dtype=np.float32),
        np.asarray((420, 240), dtype=np.float32),
        (480, 640, 3),
        1 / 60,
    )
    small_first = np.asarray((315, 240), dtype=np.float32)
    small_second = np.asarray((325, 240), dtype=np.float32)
    for _ in range(60):
        rectangle.update_transform(small_first, small_second, (480, 640, 3), 1 / 60)
        if rectangle.delete_pending:
            break
    assert rectangle.delete_pending

    for _ in range(5):
        rectangle.update_transform(
            np.asarray((220, 240), dtype=np.float32),
            np.asarray((420, 240), dtype=np.float32),
            (480, 640, 3),
            1 / 60,
        )

    assert rectangle.state == RectangleState.LOCKED
    assert not rectangle.delete_pending


def test_shrink_delete_does_not_trigger_when_only_one_dimension_is_small():
    rectangle = _delete_test_rectangle()
    rectangle.points = (
        np.asarray((220, 220), dtype=np.float32),
        np.asarray((420, 260), dtype=np.float32),
    )
    rectangle.update_transform(
        np.asarray((220, 240), dtype=np.float32),
        np.asarray((420, 240), dtype=np.float32),
        (480, 640, 3),
        1 / 60,
    )
    smaller_first = np.asarray((260, 240), dtype=np.float32)
    smaller_second = np.asarray((340, 240), dtype=np.float32)
    for _ in range(60):
        rectangle.update_transform(smaller_first, smaller_second, (480, 640, 3), 1 / 60)

    size = rectangle.points[1] - rectangle.points[0]
    assert size[1] < 480 * 0.10
    assert size[0] > 640 * 0.10
    assert not rectangle.delete_pending
    assert rectangle.state == RectangleState.LOCKED


def test_tracking_loss_cancels_delete_timer_and_closing_returns_to_idle_for_recreation():
    from effects.rectangle_frame import RectangleFrame

    rectangle = _delete_test_rectangle()
    rectangle.update_transform(
        np.asarray((220, 240), dtype=np.float32),
        np.asarray((420, 240), dtype=np.float32),
        (480, 640, 3),
        1 / 60,
    )
    small_first = np.asarray((315, 240), dtype=np.float32)
    small_second = np.asarray((325, 240), dtype=np.float32)
    for _ in range(60):
        rectangle.update_transform(small_first, small_second, (480, 640, 3), 1 / 60)
        if rectangle.delete_pending:
            break
    rectangle.end_transform()
    assert not rectangle.delete_pending

    rectangle.state = RectangleState.CLOSING
    rectangle._closing_elapsed = 0.0
    rectangle.update(np.zeros((480, 640, 3), dtype=np.uint8), {}, 0.30)
    assert rectangle.state == RectangleState.IDLE
    assert rectangle.points is None

    rectangle.update(np.zeros((480, 640, 3), dtype=np.uint8), {
        "LEFT": _tracked_rectangle_hand((120, 200)),
        "RIGHT": _tracked_rectangle_hand((140, 200)),
    }, 1 / 30)
    assert rectangle.state == RectangleState.ARMED


def test_deleted_rectangle_returns_to_idle_and_can_be_created_again():
    controller, manager = _locked_rectangle()
    _activate_rectangle_transform(controller, manager)

    for _ in range(120):
        _update(controller, manager, _pinches_for_rectangle(manager, 10.0))
        if manager.rectangle_frame.state == RectangleState.CLOSING:
            break
    assert manager.rectangle_frame.state == RectangleState.CLOSING
    assert manager.transform_target == TransformTarget.RECTANGLE

    for _ in range(60):
        _update(controller, manager, _pinches_for_rectangle(manager, 10.0))
        assert manager.transform_target == TransformTarget.RECTANGLE
        if manager.rectangle_frame.state == RectangleState.IDLE:
            break
    assert manager.rectangle_frame.state == RectangleState.IDLE

    _update(controller, manager, [])
    _update(controller, manager, _nearby_hands())
    assert manager.rectangle_frame.state == RectangleState.ARMED
    _update(
        controller,
        manager,
        [_make_l_hand((100, 180), "LEFT"), _make_l_hand((205, 260), "RIGHT")],
    )
    assert manager.rectangle_frame.state == RectangleState.DRAWING


def test_t_key_cycles_manual_transform_selection():
    from Magic_hand import MagicHandApp

    manager = EffectManager(start_with_heart=True)
    app = MagicHandApp.__new__(MagicHandApp)
    app.effect_manager = manager

    for expected in (
        TransformTargetMode.PARTICLES,
        TransformTargetMode.RECTANGLE,
        TransformTargetMode.AUTO,
    ):
        app._handle_keys(ord("t"))
        assert manager.transform_target_mode == expected


def test_hud_displays_transform_target_and_delete_cancellation_hint(monkeypatch):
    from core.app_state import AppState
    from core.gesture_controller import InteractionMode
    from effects.effect_manager import TransformTarget
    from rendering import hud

    text = []
    monkeypatch.setattr(
        hud.cv2,
        "putText",
        lambda frame, value, *args, **kwargs: text.append(value),
    )
    hud.draw_hud(
        np.zeros((720, 1280, 3), dtype=np.uint8),
        AppState(),
        GestureController(),
        InteractionMode.TRANSFORM,
        "HEART",
        TransformTargetMode.AUTO,
        TransformTarget.RECTANGLE,
        True,
    )

    assert "TRANSFORM / RECTANGLE" in text
    assert "RECTANGLE / RELEASE TO CANCEL" in text


def test_rectangle_l_does_not_select_heart_or_transform_particles():
    controller = GestureController(hold_time=0.0)
    manager = EffectManager(start_with_heart=False)

    _update(controller, manager, _nearby_hands())

    assert controller.event is None
    assert controller.mode.value != "TRANSFORM"
    assert not controller.detect_two_hand_transform(_nearby_hands())
    assert manager.current_shape is None
    assert not manager.transform_active
