import numpy as np

from core.gesture_controller import GestureController
from effects.effect_manager import EffectManager


def _make_hand(
    center=(0.0, 0.0),
    extended=("INDEX",),
    pinch=False,
    label="LEFT",
):
    center = np.asarray(center, dtype=np.float32)
    points = np.zeros((21, 2), dtype=np.float32)
    points[0] = center

    thumb = {
        1: (-40, -20),
        2: (-55, -25),
        3: (-70, -32),
        4: (-85, -40),
    }
    if "THUMB" not in extended:
        thumb = {
            1: (-40, -20),
            2: (-30, -15),
            3: (-20, -5),
            4: (-25, 5),
        }
    for index, point in thumb.items():
        points[index] = center + point

    fingers = {
        "INDEX": (5, 6, 7, 8, (-35, -25)),
        "MIDDLE": (9, 10, 11, 12, (0, -30)),
        "RING": (13, 14, 15, 16, (30, -25)),
        "PINKY": (17, 18, 19, 20, (55, -15)),
    }
    for name, (mcp, pip, dip, tip, base) in fingers.items():
        base_point = center + np.asarray(base, dtype=np.float32)
        if name in extended:
            points[mcp] = base_point
            points[pip] = base_point + (0, -30)
            points[dip] = base_point + (0, -60)
            points[tip] = base_point + (0, -90)
        else:
            points[mcp] = base_point
            points[pip] = base_point + (0, -15)
            points[dip] = base_point + (20, -5)
            points[tip] = base_point + (10, 10)

    if pinch:
        points[4] = center + (-5, -65)
        points[8] = center + (5, -65)

    return {
        "landmarks": points,
        "palm_size": 40.0,
        "center": center,
        "label": label,
    }


def test_index_raised_is_not_open_hand_or_break():
    controller = GestureController(hold_time=0.0)
    hand = _make_hand(extended=("INDEX",))
    manager = EffectManager()
    scatter_calls = []
    explode_calls = []
    manager.heart.system.scatter = lambda *args, **kwargs: scatter_calls.append((args, kwargs))
    manager.heart.system.explode = lambda *args, **kwargs: explode_calls.append((args, kwargs))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    assert controller.update([hand], 0.016) == "INDEX_RAISED"
    assert controller.raw == "INDEX_RAISED"
    assert controller.action == "SELECT_HEART"
    assert controller.finger_states["LEFT"]["INDEX"] is True
    assert controller.finger_states["LEFT"]["MIDDLE"] is False
    assert controller.confirmed != "OPEN_HAND"
    assert controller.action != "BREAK"
    assert controller.triggered is True
    manager.update(frame, controller, 0.016)
    assert scatter_calls == []
    assert explode_calls == []
    controller.update([hand], 0.016)
    assert controller.triggered is False

    thumb_extended = _make_hand(extended=("THUMB", "INDEX"))
    assert GestureController(hold_time=0.0).update([thumb_extended], 0.016) == "INDEX_RAISED"


def test_five_extended_fingers_are_open_hand():
    controller = GestureController(hold_time=0.0)
    hand = _make_hand(extended=("THUMB", "INDEX", "MIDDLE", "RING", "PINKY"))

    assert controller.update([hand], 0.016) == "OPEN_HAND"
    assert all(controller.finger_states["LEFT"].values())
    assert controller.action == "REPEL"


def test_fist_has_no_effect_action():
    controller = GestureController(hold_time=0.0)
    hand = _make_hand(extended=())
    manager = EffectManager()
    explode_calls = []
    manager.heart.system.explode = lambda **kwargs: explode_calls.append(kwargs)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    assert controller.update([hand], 0.016) == "FIST"
    manager.update(frame, controller, 0.016)
    assert controller.action == "NONE"
    assert explode_calls == []


def test_single_pinch_is_grab_and_two_pinches_transform():
    controller = GestureController(hold_time=0.0)
    left = _make_hand(center=(100, 100), extended=(), pinch=True, label="LEFT")
    right = _make_hand(center=(300, 100), extended=(), pinch=True, label="RIGHT")

    assert controller.update([left], 0.016) == "PINCH"
    assert controller.action == "GRAB"

    controller = GestureController(hold_time=0.0)
    assert controller.update([left, right], 0.016) == "TRANSFORM"
    assert controller.action == "TRANSFORM"


def test_snap_emits_break_once_after_contact_and_fast_separation():
    controller = GestureController(hold_time=0.0)
    hand = _make_hand(extended=())
    manager = EffectManager()
    system = manager.heart.system
    explode_calls = []
    original_break = system.break_apart

    def count_break(point):
        explode_calls.append(point)
        original_break(point)

    system.break_apart = count_break
    original_targets = system.targets.copy()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    points = hand["landmarks"]
    points[4] = np.array([0.0, 0.0], dtype=np.float32)
    points[12] = np.array([2.0, 0.0], dtype=np.float32)
    gesture = controller.update([hand], 0.02)
    manager.update(frame, controller, 0.02)

    points[12] = np.array([3.0, 0.0], dtype=np.float32)
    gesture = controller.update([hand], 0.02)
    manager.update(frame, controller, 0.02)
    points[12] = np.array([40.0, 0.0], dtype=np.float32)
    gesture = controller.update([hand], 0.02)
    assert gesture == "SNAP"
    assert controller.action == "BREAK"
    assert controller.triggered is True
    manager.update(frame, controller, 0.02)

    gesture = controller.update([hand], 0.02)
    manager.update(frame, controller, 0.02)
    assert gesture != "SNAP"
    assert controller.action != "BREAK"
    assert len(explode_calls) == 1
    assert np.array_equal(system.targets, original_targets)

    controller.update([], 0.016)
    for _ in range(300):
        manager.update(frame, controller, 0.016)
    assert not system.exploding
    assert np.linalg.norm(system.positions - system.targets, axis=1).mean() < 5.0


def test_open_hand_repels_locally_without_global_scatter():
    controller = GestureController(hold_time=0.0)
    hand = _make_hand(
        center=(100, 100),
        extended=("THUMB", "INDEX", "MIDDLE", "RING", "PINKY"),
    )
    manager = EffectManager()
    system = manager.heart.system
    system.positions[:] = np.array([100.0, 100.0], dtype=np.float32)
    system.positions[-1] = (400, 400)
    system.targets[:] = system.positions
    original_targets = system.targets.copy()
    scatter_calls = []
    system.scatter = lambda *args, **kwargs: scatter_calls.append((args, kwargs))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    controller.update([hand], 0.016)
    manager.update(frame, controller, 0.016)

    assert scatter_calls == []
    assert np.array_equal(system.targets, original_targets)
    assert np.any(system.velocities[:-1] != 0.0)
    assert not system.velocities[-1].any()


def test_pinch_grabs_nearby_particles_then_releases_them_to_targets():
    controller = GestureController(hold_time=0.0)
    hand = _make_hand(center=(100, 100), extended=(), pinch=True)
    manager = EffectManager()
    system = manager.heart.system
    system.positions[:] = np.array([300.0, 300.0], dtype=np.float32)
    system.targets[:] = system.positions
    system.positions[:2] = np.array([[98.0, 100.0], [102.0, 100.0]], dtype=np.float32)
    system.targets[:2] = system.positions[:2]
    original_targets = system.targets.copy()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    controller.update([hand], 0.016)
    manager.update(frame, controller, 0.016)

    hand["landmarks"][4] += np.array([20.0, 0.0], dtype=np.float32)
    hand["landmarks"][8] += np.array([20.0, 0.0], dtype=np.float32)
    controller.update([hand], 0.016)
    manager.update(frame, controller, 0.016)
    assert system.positions[:2, 0].mean() > 110.0
    np.testing.assert_allclose(system.positions[1] - system.positions[0], (4, 0), atol=1e-5)
    np.testing.assert_array_equal(system.positions[2:], original_targets[2:])
    assert np.array_equal(system.targets, original_targets)

    controller.update([], 0.016)
    manager.update(frame, controller, 0.016)
    for _ in range(250):
        system.update(0.016)
    assert np.linalg.norm(system.positions[:2] - system.targets[:2], axis=1).mean() < 5.0


def test_pinch_hysteresis_keeps_gesture_until_release_threshold():
    controller = GestureController(hold_time=0.0, pinch_threshold=0.3, pinch_release_threshold=0.5)
    hand = _make_hand(extended=(), pinch=True)
    assert controller.update([hand], 0.016) == "PINCH"

    points = hand["landmarks"]
    midpoint = (points[4] + points[8]) * 0.5
    points[4] = midpoint + (-8.0, 0.0)
    points[8] = midpoint + (8.0, 0.0)
    assert controller.update([hand], 0.016) == "PINCH"

    points[4] = midpoint + (-12.0, 0.0)
    points[8] = midpoint + (12.0, 0.0)
    assert controller.update([hand], 0.016) != "PINCH"


def test_transform_moves_scales_rotates_and_preserves_on_release():
    controller = GestureController(hold_time=0.0)
    manager = EffectManager()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    left = _make_hand(center=(100, 100), extended=(), pinch=True, label="LEFT")
    right = _make_hand(center=(300, 100), extended=(), pinch=True, label="RIGHT")

    def step(hands):
        controller.update(hands, 0.033)
        manager.update(frame, controller, 0.033)

    step([left, right])
    initial_center = manager.transform_center.copy()
    initial_scale = manager.transform_scale
    left["landmarks"] += (5, 25)
    right["landmarks"] += (75, 65)
    step([right, left])
    assert np.all(manager.transform_center > initial_center)
    assert initial_scale < manager.transform_scale <= initial_scale * 1.03
    assert 0 < manager.transform_rotation <= np.deg2rad(4)

    expected_center = manager.transform_center.copy()
    expected_scale = manager.transform_scale
    expected_rotation = manager.transform_rotation
    step([left])
    assert not manager.transform_active
    assert controller.action == "NONE"
    right["landmarks"] += (80, 0)
    step([right, left])
    assert manager.transform_active
    step([])
    np.testing.assert_array_equal(manager.transform_center, expected_center)
    assert manager.transform_scale == expected_scale
    assert manager.transform_rotation == expected_rotation


def test_transform_angle_wrap_and_degenerate_baseline():
    manager = EffectManager()
    origin = np.array((260, 180), dtype=np.float32)

    def point(degrees):
        angle = np.deg2rad(degrees)
        return origin + 140 * np.array([np.cos(angle), np.sin(angle)])

    manager._update_transform(origin, origin)
    assert manager.transform_reference is None
    manager._update_transform(origin, point(179))
    manager._update_transform(origin, point(-179))
    assert 0 < manager.transform_rotation < np.deg2rad(2)
    previous = manager.transform_rotation
    manager._update_transform(point(-177), origin)
    assert previous < manager.transform_rotation < np.deg2rad(4)


def test_snap_rejects_slow_release_old_contact_and_cooldown_after_tracking_loss():
    hand = _make_hand(extended=())
    points = hand["landmarks"]
    points[4] = (0, 0)
    controller = GestureController()

    def frame(x, dt=1 / 30):
        points[12] = (x, 0)
        controller.update([hand], dt)
        return controller.triggered

    assert not frame(2)
    for x in range(4, 42, 2):
        assert not frame(x)
    assert not frame(2)
    for _ in range(16):
        assert not frame(2)
    assert not frame(40)

    assert not frame(2)
    assert frame(40)
    controller.update([], 0.1)
    assert not frame(2)
    assert not frame(40)
    controller.update([], 0.8)
    assert not frame(2)
    assert frame(40)


def test_selection_holds_and_persists_after_tracking_loss():
    controller = GestureController()
    manager = EffectManager()
    manager.toggle_heart()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hand = _make_hand()

    for _ in range(21):
        controller.update([hand], 1 / 30)
        manager.update(frame, controller, 1 / 30)
        assert manager.current_shape is None
    controller.update([hand], 1 / 30 + 0.001)
    manager.update(frame, controller, 1 / 30)
    assert controller.triggered
    assert manager.current_shape == "HEART"
    controller.update([hand], 1 / 30)
    assert not controller.triggered
    controller.update([], 1 / 30)
    manager.update(frame, controller, 1 / 30)
    assert manager.current_shape == "HEART"


def test_gesture_geometry_is_invariant_under_in_plane_rotation_and_scale():
    from particles.transforms import rotate_points

    for fingers, expected in (
        (("INDEX",), "INDEX_RAISED"),
        (("THUMB", "INDEX", "MIDDLE", "RING", "PINKY"), "OPEN_HAND"),
        (("INDEX", "MIDDLE", "RING", "PINKY"), "IDLE"),
        ((), "FIST"),
    ):
        hand = _make_hand(extended=fingers)
        hand["landmarks"] = rotate_points(hand["landmarks"] * 0.55, 1.7) + (90, 120)
        hand["palm_size"] *= 0.55
        assert GestureController(hold_time=0).update([hand], 1 / 30) == expected


def test_continuous_modes_wait_briefly_and_respect_priority():
    controller = GestureController()
    palm = _make_hand(extended=("THUMB", "INDEX", "MIDDLE", "RING", "PINKY"), label="RIGHT")
    point = _make_hand()
    assert controller.update([point, palm], 1 / 30) == "IDLE"
    assert controller.candidate == "OPEN_HAND"
    assert controller.update([point, palm], 1 / 30) == "IDLE"
    assert controller.update([point, palm], 1 / 30) == "OPEN_HAND"
    pinch = _make_hand(extended=(), pinch=True)
    for _ in range(3):
        controller.update([palm, pinch], 1 / 30)
    assert controller.action == "GRAB"
    assert not controller.triggered


def test_two_pinches_take_priority_over_snap_release():
    controller = GestureController(hold_time=0)
    left = _make_hand(extended=(), label="LEFT")
    right = _make_hand(center=(220, 180), extended=(), pinch=True, label="RIGHT")
    left["landmarks"][[4, 12]] = ((0, -65), (2, -65))
    controller.update([left, right], 1 / 30)
    left["landmarks"][[8, 12]] = ((5, -65), (40, -65))
    assert controller.update([left, right], 1 / 30) == "TRANSFORM"
    assert not controller.triggered
