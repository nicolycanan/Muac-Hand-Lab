import numpy as np

from core.gesture_controller import HandGesture, RawGesture
from effects.psychedelic_fill import PsychedelicFill
from effects.rectangle_frame import RectangleFrame, RectangleState


def _hand(point):
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


def _hands(first=(20, 20), second=(40, 20)):
    return {"LEFT": _hand(first), "RIGHT": _hand(second)}


def _drawing_frame(points=((10, 10), (50, 45))):
    rectangle = RectangleFrame()
    rectangle.state = RectangleState.DRAWING
    rectangle.alpha = 1.0
    rectangle.points = tuple(np.asarray(point, dtype=np.float32) for point in points)
    return rectangle


def test_fill_is_not_processed_in_idle_or_armed():
    rectangle = RectangleFrame()
    calls = []
    rectangle.fill.render = lambda *args: calls.append(args)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    rectangle.update(frame, {}, 1 / 30)
    rectangle.update(frame, _hands((20, 20), (50, 20)), 1 / 30)

    assert rectangle.state == RectangleState.ARMED
    assert calls == []


def test_fill_configuration_can_disable_texture(monkeypatch):
    import effects.rectangle_frame as rectangle_module

    monkeypatch.setattr(rectangle_module, "PSYCHEDELIC_FILL_ENABLED", False)
    rectangle = _drawing_frame()
    calls = []
    rectangle.fill.render = lambda *args: calls.append(args)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    rectangle.update(frame, _hands((10, 10), (50, 45)), 1 / 30)

    assert calls == []
    assert np.any(frame != 0)


def test_fill_is_drawn_inside_the_rectangle_only():
    rectangle = _drawing_frame(((10, 12), (48, 42)))
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    rectangle.update(frame, _hands((10, 12), (48, 42)), 1 / 30)

    assert np.any(frame[20:35, 20:40] != 0)
    assert np.array_equal(frame[0, 0], (0, 0, 0))
    assert np.array_equal(frame[60, 60], (0, 0, 0))


def test_texture_changes_over_elapsed_time():
    fill = PsychedelicFill()
    first = np.zeros((100, 120, 3), dtype=np.uint8)
    second = np.zeros_like(first)
    fill.render(first, (10, 10), (110, 90), 0.0)
    fill.render(second, (10, 10), (110, 90), 0.4)

    assert not np.array_equal(first[20:80, 20:100], second[20:80, 20:100])


def test_opacity_blends_with_camera_and_zero_fade_is_invisible(monkeypatch):
    import effects.psychedelic_fill as fill_module

    monkeypatch.setattr(fill_module, "PSYCHEDELIC_FILL_OPACITY", 0.5)
    fill = PsychedelicFill()
    frame = np.full((60, 60, 3), 40, dtype=np.uint8)
    fill.render(frame, (10, 10), (50, 50), 0.0)
    assert not np.array_equal(frame[25, 25], (40, 40, 40))

    unchanged = np.full((60, 60, 3), 40, dtype=np.uint8)
    fill.render(unchanged, (10, 10), (50, 50), 0.2, fade_alpha=0.0)
    assert np.all(unchanged == 40)


def test_fill_tracks_smoothed_rectangle_resize():
    rectangle = _drawing_frame(((20, 20), (50, 50)))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    rectangle.update(frame, _hands((20, 20), (50, 50)), 0.0)
    initial_width = rectangle.points[1][0] - rectangle.points[0][0]

    rectangle.update(frame, _hands((10, 10), (80, 80)), 1 / 30)
    updated_width = rectangle.points[1][0] - rectangle.points[0][0]

    assert updated_width > initial_width


def test_fade_applies_to_fill_and_outline_until_idle():
    rectangle = _drawing_frame()
    rectangle.state = RectangleState.CLOSING
    alphas = []
    rectangle.fill.render = lambda frame, first, second, dt, alpha: alphas.append(alpha)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    rectangle.update(frame, {}, 0.05)
    first_alpha = rectangle.alpha
    for _ in range(4):
        rectangle.update(frame, {}, 0.05)

    assert first_alpha < 1.0
    assert alphas[0] == first_alpha
    assert rectangle.state == RectangleState.IDLE
    count = len(alphas)
    rectangle.update(frame, {}, 1 / 30)
    assert len(alphas) == count


def test_fading_outline_uses_the_same_alpha_as_the_fill(monkeypatch):
    import effects.rectangle_frame as rectangle_module

    monkeypatch.setattr(rectangle_module, "PSYCHEDELIC_FILL_ENABLED", False)
    rectangle = _drawing_frame(((10, 10), (50, 45)))
    rectangle.state = RectangleState.CLOSING
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    rectangle.update(frame, {}, 0.125)

    assert rectangle.alpha == 0.5
    assert np.array_equal(frame[10, 30], (128, 128, 0))


def test_inverted_clipped_and_degenerate_rectangles_are_safe():
    rectangle = _drawing_frame(((70, 65), (-10, 8)))
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    rectangle.update(frame, _hands((70, 65), (-10, 8)), 1 / 30)
    assert np.any(frame[15:60, 5:65] != 0)
    assert np.array_equal(frame[0, 99], (0, 0, 0))

    rectangle.points = (
        np.asarray((30, 30), dtype=np.float32),
        np.asarray((30, 30), dtype=np.float32),
    )
    rectangle.update(frame, _hands((30, 30), (30, 30)), 1 / 30)
