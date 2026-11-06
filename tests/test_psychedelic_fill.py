import numpy as np

from core.gesture_controller import HandGesture, RawGesture
from effects.psychedelic_fill import PsychedelicFill
from effects.rectangle_frame import RectangleFrame, RectangleState
from effects.rectangle_style import RectangleFillStyle


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
    assert not np.any(frame)


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


def test_fade_applies_to_fill_until_idle():
    rectangle = _drawing_frame()
    rectangle.state = RectangleState.CLOSING
    alphas = []
    rectangle.fill.render = lambda frame, first, second, dt, alpha, style, source_roi=None: alphas.append(alpha)
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


def test_fade_passes_alpha_to_fill_without_drawing_an_outline():
    rectangle = _drawing_frame(((10, 10), (50, 45)))
    rectangle.state = RectangleState.CLOSING
    alphas = []
    rectangle.fill.render = lambda frame, first, second, dt, alpha, style, source_roi=None: alphas.append(alpha)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    rectangle.update(frame, {}, 0.125)

    assert rectangle.alpha == 0.5
    assert alphas == [0.5]
    assert not np.any(frame)


def test_monochrome_style_is_animated_and_changes_only_rectangle_roi():
    fill = PsychedelicFill()
    source = np.full((100, 120, 3), (30, 80, 140), dtype=np.uint8)
    first = source.copy()
    second = source.copy()

    fill.render(first, (10, 10), (110, 90), 0.0, style=RectangleFillStyle.MONOCHROME)
    fill.render(second, (10, 10), (110, 90), 0.4, style=RectangleFillStyle.MONOCHROME)

    assert not np.array_equal(first[20:80, 20:100], second[20:80, 20:100])
    assert np.array_equal(first[0:10], source[0:10])
    assert np.array_equal(first[:, 0:10], source[:, 0:10])
    assert np.array_equal(first[:, 110:], source[:, 110:])
    assert np.array_equal(first[90:], source[90:])


def test_glitch_style_animates_inside_roi_without_changing_outside_pixels():
    fill = PsychedelicFill()
    y, x = np.indices((100, 120))
    source = np.stack(
        ((x * 2) % 256, (y * 3) % 256, (x + y) % 256),
        axis=2,
    ).astype(np.uint8)
    first = source.copy()
    second = source.copy()

    fill.render(first, (10, 10), (110, 90), 0.0, style=RectangleFillStyle.GLITCH)
    fill.render(second, (10, 10), (110, 90), 0.4, style=RectangleFillStyle.GLITCH)

    assert not np.array_equal(first[20:80, 20:100], source[20:80, 20:100])
    assert not np.array_equal(first[20:80, 20:100], second[20:80, 20:100])
    assert np.array_equal(first[0:10], source[0:10])
    assert np.array_equal(first[:, 0:10], source[:, 0:10])
    assert np.array_equal(first[:, 110:], source[:, 110:])
    assert np.array_equal(first[90:], source[90:])


def test_glitch_uses_strongly_blended_camera_roi_as_its_base():
    from config import GLITCH_OPACITY

    y, x = np.indices((120, 160))
    source = np.stack(
        ((x * 3 + 20) % 256, (y * 2 + 50) % 256, (x + y * 2 + 90) % 256),
        axis=2,
    ).astype(np.uint8)
    frame = source.copy()
    fill = PsychedelicFill()
    source_roi = source[11:110, 11:150].copy()

    fill.render(
        frame,
        (10, 10),
        (150, 110),
        0.3,
        style=RectangleFillStyle.GLITCH,
        source_roi=source_roi,
    )

    roi = frame[11:110, 11:150]
    original_roi = source[11:110, 11:150]
    assert not np.array_equal(roi, original_roi)
    assert np.mean(np.abs(roi.astype(np.int16) - original_roi.astype(np.int16))) > 3.0
    assert np.mean(np.abs(roi.astype(np.int16) - original_roi.astype(np.int16))) < 70.0
    assert GLITCH_OPACITY >= 0.8
    assert np.array_equal(frame[:10], source[:10])
    assert np.array_equal(frame[:, :10], source[:, :10])
    assert np.array_equal(frame[:, 150:], source[:, 150:])
    assert np.array_equal(frame[110:], source[110:])


def test_glitch_output_depends_on_camera_source_not_synthetic_texture():
    first_source = np.full((100, 120, 3), (20, 50, 90), dtype=np.uint8)
    second_source = np.full((100, 120, 3), (180, 130, 70), dtype=np.uint8)
    first = first_source.copy()
    second = second_source.copy()
    fill_a = PsychedelicFill()
    fill_b = PsychedelicFill()

    fill_a.render(first, (10, 10), (110, 90), 0.2, style=RectangleFillStyle.GLITCH)
    fill_b.render(second, (10, 10), (110, 90), 0.2, style=RectangleFillStyle.GLITCH)

    first_roi = first[11:90, 11:110]
    second_roi = second[11:90, 11:110]
    assert np.mean(first_roi[:, :, 0]) < np.mean(second_roi[:, :, 0])
    assert np.mean(first_roi[:, :, 1]) < np.mean(second_roi[:, :, 1])
    assert np.mean(first_roi[:, :, 2]) > np.mean(second_roi[:, :, 2])


def test_glitch_remains_applied_after_rectangle_geometry_changes():
    rectangle = _drawing_frame(((20, 20), (70, 60)))
    rectangle.state = RectangleState.LOCKED
    rectangle.fill_style = RectangleFillStyle.GLITCH
    frame = np.full((100, 120, 3), (35, 95, 155), dtype=np.uint8)
    first_bounds = tuple(point.copy() for point in rectangle.points)
    first_source = rectangle.capture_source_roi(frame)

    rectangle._draw(frame, 0.1, first_source)
    for actual, expected in zip(rectangle.points, first_bounds):
        np.testing.assert_array_equal(actual, expected)

    rectangle.points = (
        np.asarray((30, 25), dtype=np.float32),
        np.asarray((100, 80), dtype=np.float32),
    )
    moved_bounds = tuple(point.copy() for point in rectangle.points)
    moved_source = rectangle.capture_source_roi(frame)
    rectangle._draw(frame, 0.1, moved_source)

    assert moved_source is not None
    assert moved_source.shape[:2] == (54, 69)
    for actual, expected in zip(rectangle.points, moved_bounds):
        np.testing.assert_array_equal(actual, expected)


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
