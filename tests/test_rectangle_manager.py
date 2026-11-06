from types import SimpleNamespace

import numpy as np

from config import RECTANGLE_MAX_INSTANCES, RECTANGLE_TRACKING_GRACE_FRAMES
from core.gesture_controller import (
    ConfirmedGesture,
    HandGesture,
    InteractionMode,
    RawGesture,
)
from effects.effect_manager import EffectManager, TransformTarget, TransformTargetMode
from effects.rectangle_frame import RectangleFrame, RectangleState
from effects.rectangle_manager import RectangleManager
from effects.rectangle_style import RectangleFillStyle


def _hand(point, rectangle=True, label="LEFT"):
    point = np.asarray(point, dtype=np.float32)
    return HandGesture(
        RawGesture.RECTANGLE if rectangle else RawGesture.IDLE,
        {},
        1.0,
        point,
        point,
        np.zeros(2, dtype=np.float32),
        40.0,
        point if rectangle else None,
    )


def _l_hands(first=(100, 100), second=(140, 100)):
    return {
        "LEFT": _hand(first, label="LEFT"),
        "RIGHT": _hand(second, label="RIGHT"),
    }


def _non_l_hands():
    return {
        "LEFT": _hand((100, 100), rectangle=False),
        "RIGHT": _hand((140, 100), rectangle=False),
    }


def _update(rectangles, hands, dt=1 / 60):
    rectangles.update(np.zeros((480, 640, 3), dtype=np.uint8), hands, dt)


def _create_locked(rectangles):
    _update(rectangles, _l_hands())
    _update(rectangles, _l_hands((100, 100), (180, 140)))
    assert rectangles.creation is not None
    assert rectangles.creation.state == RectangleState.DRAWING
    for _ in range(RECTANGLE_TRACKING_GRACE_FRAMES + 1):
        _update(rectangles, _non_l_hands())
    assert rectangles.creation is None
    assert rectangles.instances[-1].state == RectangleState.LOCKED
    return rectangles.instances[-1]


def _locked_rectangle(manager, instance_id, points):
    rectangle = RectangleFrame(instance_id=instance_id)
    rectangle.state = RectangleState.LOCKED
    rectangle.alpha = 1.0
    rectangle.points = tuple(np.asarray(point, dtype=np.float32) for point in points)
    manager.rectangle_manager.instances.append(rectangle)
    return rectangle


def _pinch_hand(point, label):
    point = np.asarray(point, dtype=np.float32)
    return HandGesture(
        RawGesture.PINCH,
        {},
        0.1,
        point,
        point,
        np.zeros(2, dtype=np.float32),
        40.0,
    )


def _transform_controller(first, second, mode=InteractionMode.TRANSFORM):
    hands = {
        "LEFT": _pinch_hand(first, "LEFT"),
        "RIGHT": _pinch_hand(second, "RIGHT"),
    }
    return SimpleNamespace(
        mode=mode,
        hands=hands,
        confirmation=ConfirmedGesture(RawGesture.TRANSFORM, ("LEFT", "RIGHT")),
        event=None,
    )


def _grab_controller(point, hand_key="LEFT"):
    hands = {hand_key: _pinch_hand(point, hand_key)}
    return SimpleNamespace(
        mode=InteractionMode.GRAB,
        hands=hands,
        confirmation=ConfirmedGesture(RawGesture.PINCH, (hand_key,)),
        event=None,
    )


def test_two_consecutive_creation_gestures_add_independent_rectangles_once_each():
    rectangles = RectangleManager()
    _update(rectangles, _l_hands())
    _update(rectangles, _l_hands((100, 100), (180, 140)))
    first = rectangles.creation
    assert first is not None
    assert first.state == RectangleState.DRAWING
    for _ in range(3):
        _update(rectangles, _l_hands((100, 100), (180, 140)))
    assert rectangles.count == 1
    assert first.state == RectangleState.DRAWING
    first_points = tuple(point.copy() for point in first.points)
    first_fill_time = first.fill.time

    _update(rectangles, _non_l_hands())
    for _ in range(RECTANGLE_TRACKING_GRACE_FRAMES):
        _update(rectangles, _non_l_hands())
    assert first.state == RectangleState.LOCKED
    second = _create_locked(rectangles)

    assert rectangles.count == 2
    assert second.instance_id != first.instance_id
    assert second.fill is not first.fill
    for actual, expected in zip(first.points, first_points):
        np.testing.assert_array_equal(actual, expected)
    assert first.fill.time > first_fill_time


def test_creation_limit_allows_six_and_blocks_seventh_until_new_gesture():
    rectangles = RectangleManager()
    for _ in range(RECTANGLE_MAX_INSTANCES):
        _create_locked(rectangles)
    assert rectangles.count == 6

    _update(rectangles, _l_hands())
    assert rectangles.count == 6
    assert rectangles.creation is None
    assert rectangles.limit_reached
    _update(rectangles, _l_hands((100, 100), (180, 140)))
    assert rectangles.count == 6


def test_one_gesture_cannot_create_duplicate_during_recognition_oscillation():
    rectangles = RectangleManager()
    _update(rectangles, _l_hands())
    _update(rectangles, _l_hands((100, 100), (180, 140)))
    _update(rectangles, _non_l_hands())
    for _ in range(3):
        _update(rectangles, _l_hands((100, 100), (180, 140)))
    assert rectangles.count == 1
    assert rectangles.creation is None
    assert rectangles.instances[0].state == RectangleState.LOCKED


def test_locked_rectangles_keep_independent_geometry_and_animated_fills():
    rectangles = RectangleManager()
    first = _create_locked(rectangles)
    second = _create_locked(rectangles)
    first_points = tuple(point.copy() for point in first.points)
    second_points = tuple(point.copy() for point in second.points)
    first_time, second_time = first.fill.time, second.fill.time

    for _ in range(4):
        _update(rectangles, {})

    for actual, expected in zip(first.points, first_points):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(second.points, second_points):
        np.testing.assert_array_equal(actual, expected)
    assert first.fill.time > first_time
    assert second.fill.time > second_time
    assert first.fill.time != second.fill.time


def test_auto_selection_uses_point_inside_and_topmost_for_overlap():
    manager = EffectManager(start_with_heart=True)
    lower = _locked_rectangle(
        manager, 1, ((100, 100), (300, 300)),
    )
    upper = _locked_rectangle(
        manager, 2, ((200, 150), (400, 350)),
    )

    target = manager._select_transform_target(
        [_pinch_hand((250, 220), "LEFT"), _pinch_hand((270, 220), "RIGHT")],
        (480, 640, 3),
    )

    assert target == TransformTarget.RECTANGLE
    assert manager.selected_rectangle_id == upper.instance_id
    assert manager.selected_rectangle is upper
    assert lower is not upper


def test_auto_selection_uses_particles_outside_all_rectangles():
    manager = EffectManager(start_with_heart=True)
    _locked_rectangle(manager, 1, ((100, 100), (300, 300)))

    target = manager._select_transform_target(
        [_pinch_hand((500, 400), "LEFT"), _pinch_hand((520, 400), "RIGHT")],
        (480, 640, 3),
    )

    assert target == TransformTarget.PARTICLES
    assert manager.selected_rectangle_id is None


def test_transform_keeps_selected_id_until_release_even_after_midpoint_leaves_it():
    manager = EffectManager(start_with_heart=False)
    first = _locked_rectangle(manager, 1, ((100, 100), (240, 240)))
    second = _locked_rectangle(manager, 2, ((350, 250), (500, 400)))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    manager.update(
        frame.copy(),
        _transform_controller((140, 170), (200, 170)),
        1 / 60,
    )
    assert manager.selected_rectangle_id == first.instance_id

    manager.update(
        frame.copy(),
        _transform_controller((550, 420), (610, 420)),
        1 / 60,
    )
    assert manager.transform_target == TransformTarget.RECTANGLE
    assert manager.selected_rectangle_id == first.instance_id

    manager.update(
        frame.copy(),
        SimpleNamespace(
            mode=InteractionMode.IDLE,
            hands={},
            confirmation=ConfirmedGesture(),
            event=None,
        ),
        1 / 60,
    )
    assert manager.transform_target == TransformTarget.NONE
    assert manager.selected_rectangle_id is None


def test_manual_rectangle_selects_containing_or_nearest_deterministically():
    manager = EffectManager(start_with_heart=True)
    first = _locked_rectangle(manager, 1, ((100, 100), (200, 200)))
    second = _locked_rectangle(manager, 2, ((300, 100), (400, 200)))
    manager.transform_target_mode = TransformTargetMode.RECTANGLE

    target = manager._select_transform_target(
        [_pinch_hand((330, 150), "LEFT"), _pinch_hand((350, 150), "RIGHT")],
        (480, 640, 3),
    )
    assert target == TransformTarget.RECTANGLE
    assert manager.selected_rectangle_id == second.instance_id

    target = manager._select_transform_target(
        [_pinch_hand((240, 150), "LEFT"), _pinch_hand((260, 150), "RIGHT")],
        (480, 640, 3),
    )
    assert target == TransformTarget.RECTANGLE
    assert manager.selected_rectangle_id == second.instance_id


def test_particles_manual_mode_keeps_rectangle_unselected():
    manager = EffectManager(start_with_heart=True)
    _locked_rectangle(manager, 1, ((100, 100), (300, 300)))
    manager.transform_target_mode = TransformTargetMode.PARTICLES

    target = manager._select_transform_target(
        [_pinch_hand((150, 150), "LEFT"), _pinch_hand((170, 150), "RIGHT")],
        (480, 640, 3),
    )

    assert target == TransformTarget.PARTICLES
    assert manager.selected_rectangle_id is None


def test_transform_moves_and_scales_only_selected_rectangle():
    manager = EffectManager(start_with_heart=False)
    selected = _locked_rectangle(manager, 1, ((100, 100), (220, 200)))
    other = _locked_rectangle(manager, 2, ((350, 250), (470, 350)))
    other_points = tuple(point.copy() for point in other.points)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    baseline = _transform_controller((110, 150), (210, 150))
    manager.update(frame.copy(), baseline, 1 / 60)
    assert manager.transform_target == TransformTarget.RECTANGLE
    assert manager.selected_rectangle_id == selected.instance_id

    manager.update(
        frame.copy(),
        _transform_controller((130, 160), (250, 160)),
        1 / 60,
    )
    assert not np.array_equal(selected.points[0], np.asarray((100, 100), dtype=np.float32))
    for actual, expected in zip(other.points, other_points):
        np.testing.assert_array_equal(actual, expected)

    width_before = selected.points[1][0] - selected.points[0][0]
    manager.update(
        frame.copy(),
        _transform_controller((100, 180), (300, 180)),
        1 / 60,
    )
    assert selected.points[1][0] - selected.points[0][0] > width_before
    for actual, expected in zip(other.points, other_points):
        np.testing.assert_array_equal(actual, expected)


def test_closing_removes_only_selected_instance_and_allows_recreation():
    rectangles = RectangleManager()
    first = _create_locked(rectangles)
    second = _create_locked(rectangles)
    second.state = RectangleState.CLOSING
    second._closing_elapsed = 0.0

    _update(rectangles, {}, dt=0.30)

    assert rectangles.count == 1
    assert rectangles.instances[0] is first
    _create_locked(rectangles)
    assert rectangles.count == 2
    assert rectangles.instances[-1].instance_id > second.instance_id


def test_fading_selected_rectangle_releases_transform_target_without_dangling_draw():
    from rendering.hud import draw_hand_controls

    manager = EffectManager(start_with_heart=False)
    selected = _locked_rectangle(manager, 1, ((100, 100), (240, 240)))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    controller = _transform_controller((140, 170), (200, 170))
    manager.update(frame, controller, 1 / 60)
    selected.state = RectangleState.CLOSING
    selected._closing_elapsed = 0.24

    manager.update(frame, controller, 1 / 60)

    assert manager.selected_rectangle_id is None
    assert not manager.rectangle_transform_active
    assert not manager.transform_active
    draw_hand_controls(frame, controller, manager)


def test_instances_render_in_creation_order_with_independent_fill_calls():
    rectangles = RectangleManager()
    first = _create_locked(rectangles)
    second = _create_locked(rectangles)
    calls = []
    first.fill.render = lambda *args, **kwargs: calls.append(first.instance_id)
    second.fill.render = lambda *args, **kwargs: calls.append(second.instance_id)

    _update(rectangles, {})

    assert calls == [first.instance_id, second.instance_id]


def test_single_pinch_inside_cycles_style_once_after_hold_and_reserves_grab(monkeypatch):
    import effects.effect_manager as manager_module

    monkeypatch.setattr(manager_module, "RECTANGLE_STYLE_SWITCH_HOLD", 0.45)
    manager = EffectManager(start_with_heart=False)
    rectangle = _locked_rectangle(manager, 1, ((100, 100), (300, 300)))
    begin_grab = []
    manager.heart.system.begin_or_update_grab = lambda point: begin_grab.append(point.copy())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    controller = _grab_controller((150, 150))

    for _ in range(26):
        manager.update(frame.copy(), controller, 1 / 60)
    assert rectangle.fill_style == RectangleFillStyle.PSYCHEDELIC
    assert manager.rectangle_style_feedback[:2] == (1, "PSYCHEDELIC")
    assert np.isclose(manager.rectangle_style_feedback[2], 26 / 27)
    assert begin_grab == []
    assert manager.interaction_mode != InteractionMode.GRAB

    manager.update(frame.copy(), controller, 1 / 60)
    assert rectangle.fill_style == RectangleFillStyle.MONOCHROME
    assert begin_grab == []
    assert manager.rectangle_style_feedback == (1, "MONOCHROME", 0.0)

    for _ in range(10):
        manager.update(frame.copy(), controller, 1 / 60)
    assert rectangle.fill_style == RectangleFillStyle.MONOCHROME
    assert begin_grab == []


def test_single_pinch_outside_rectangle_remains_particle_grab():
    manager = EffectManager(start_with_heart=False)
    rectangle = _locked_rectangle(manager, 1, ((100, 100), (300, 300)))
    grabs = []
    manager.heart.system.begin_or_update_grab = lambda point: grabs.append(point.copy())

    manager.update(np.zeros((480, 640, 3), dtype=np.uint8), _grab_controller((500, 400)), 0.1)

    assert len(grabs) == 1
    assert rectangle.fill_style == RectangleFillStyle.PSYCHEDELIC
    assert manager.rectangle_style_feedback is None


def test_style_switch_cancels_when_tracking_is_lost_or_second_pinch_starts():
    manager = EffectManager(start_with_heart=False)
    rectangle = _locked_rectangle(manager, 1, ((100, 100), (300, 300)))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    manager.update(frame.copy(), _grab_controller((150, 150)), 0.2)
    assert manager.rectangle_style_feedback[2] > 0.0

    manager.update(frame.copy(), _transform_controller((150, 150), (250, 150)), 0.1)
    assert rectangle.fill_style == RectangleFillStyle.PSYCHEDELIC
    assert manager.rectangle_style_feedback[2] == 0.0
    assert manager.transform_active

    manager.update(
        frame.copy(),
        SimpleNamespace(
            mode=InteractionMode.IDLE,
            hands={},
            confirmation=ConfirmedGesture(),
            event=None,
        ),
        1 / 60,
    )
    manager.update(frame.copy(), _grab_controller((150, 150)), 0.2)
    manager.update(
        frame.copy(),
        SimpleNamespace(
            mode=InteractionMode.IDLE,
            hands={},
            confirmation=ConfirmedGesture(),
            event=None,
        ),
        1 / 60,
    )
    assert rectangle.fill_style == RectangleFillStyle.PSYCHEDELIC
    assert manager.rectangle_style_feedback is None


def test_style_is_per_rectangle_and_overlap_selects_top_layer():
    manager = EffectManager(start_with_heart=False)
    bottom = _locked_rectangle(manager, 1, ((100, 100), (300, 300)))
    top = _locked_rectangle(manager, 2, ((200, 200), (400, 400)))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    controller = _grab_controller((250, 250))

    for _ in range(28):
        manager.update(frame.copy(), controller, 1 / 60)

    assert bottom.fill_style == RectangleFillStyle.PSYCHEDELIC
    assert top.fill_style == RectangleFillStyle.MONOCHROME

    manager.update(frame.copy(), _transform_controller((220, 250), (280, 250)), 1 / 60)
    assert top.fill_style == RectangleFillStyle.MONOCHROME
    for _ in range(27):
        manager.update(frame.copy(), _grab_controller((250, 250)), 1 / 60)
    assert top.fill_style == RectangleFillStyle.GLITCH
    assert bottom.fill_style == RectangleFillStyle.PSYCHEDELIC


def test_fill_style_cycle_returns_to_psychedelic_after_glitch():
    rectangle = RectangleFrame(instance_id=1)
    assert rectangle.fill_style == RectangleFillStyle.PSYCHEDELIC
    assert rectangle.cycle_fill_style() == RectangleFillStyle.MONOCHROME
    assert rectangle.cycle_fill_style() == RectangleFillStyle.GLITCH
    assert rectangle.cycle_fill_style() == RectangleFillStyle.PSYCHEDELIC


def test_single_pinch_inside_does_not_change_rectangle_geometry():
    manager = EffectManager(start_with_heart=False)
    rectangle = _locked_rectangle(manager, 1, ((100, 100), (300, 300)))
    original = tuple(point.copy() for point in rectangle.points)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for _ in range(35):
        manager.update(frame.copy(), _grab_controller((150, 150)), 1 / 60)

    for actual, expected in zip(rectangle.points, original):
        np.testing.assert_array_equal(actual, expected)


def test_hud_shows_rectangle_count_selected_id_and_target():
    from core.app_state import AppState
    from rendering import hud

    rendered = []
    original = hud._draw_panel
    hud._draw_panel = lambda frame, x, width, lines: rendered.extend(lines)
    try:
        hud.draw_hud(
            np.zeros((720, 1280, 3), dtype=np.uint8),
            AppState(),
            SimpleNamespace(
                candidate=RawGesture.IDLE,
                confirmed=RawGesture.IDLE,
                pending=SimpleNamespace(elapsed=0.0),
                hands={},
            ),
            InteractionMode.TRANSFORM,
            "HEART",
            "AUTO",
            "RECTANGLE",
            False,
            3,
            RECTANGLE_MAX_INSTANCES,
            2,
            False,
        )
    finally:
        hud._draw_panel = original

    assert "TARGET: AUTO" in rendered
    assert "RECTANGLES: 3/6" in rendered
    assert "SELECTED: RECTANGLE 02" in rendered
