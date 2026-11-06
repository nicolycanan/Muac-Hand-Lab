import cv2
import numpy as np

from core.camera import list_camera_indices
from rendering import camera_picker as picker_module
from rendering.camera_picker import CameraDevice, CameraPicker


class MockCapture:
    def __init__(self, opened, frame):
        self.opened = opened
        self.frame = frame
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return self.frame is not None, self.frame

    def release(self):
        self.released = True


def _finish_scan(picker):
    picker._scan_thread.join(timeout=2.0)
    assert not picker._scan_thread.is_alive()
    assert picker.poll_scan_results()


def test_scanner_requires_open_device_to_deliver_a_valid_frame(monkeypatch):
    captures = {
        0: MockCapture(True, np.zeros((24, 32, 3), dtype=np.uint8)),
        1: MockCapture(False, None),
        2: MockCapture(True, None),
        3: MockCapture(True, np.empty((0, 0, 3), dtype=np.uint8)),
        4: MockCapture(True, np.zeros((24, 32), dtype=np.uint8)),
    }
    monkeypatch.setattr(cv2, "VideoCapture", lambda index, backend: captures[index])

    results = list_camera_indices(0, 4, "AUTO")

    assert results == [
        (0, True, (32, 24)),
        (1, False, None),
        (2, False, None),
        (3, False, None),
        (4, False, None),
    ]
    assert all(capture.released for capture in captures.values())


def test_scanner_does_not_open_the_active_camera_again(monkeypatch):
    opened = []

    def open_camera(index, backend):
        opened.append(index)
        return MockCapture(True, np.zeros((10, 12, 3), dtype=np.uint8))

    monkeypatch.setattr(cv2, "VideoCapture", open_camera)

    results = list_camera_indices(
        0, 2, "AUTO", active_index=1, active_available=True,
    )

    assert opened == [0, 2]
    assert results[1] == (1, True, None)


def test_picker_displays_only_discovered_devices_and_highlights_active(monkeypatch):
    monkeypatch.setattr(
        picker_module,
        "list_camera_indices",
        lambda *args, **kwargs: [
            (0, True, (640, 480)),
            (1, False, None),
            (2, True, None),
        ],
    )
    picker = CameraPicker(360, width=640)
    assert picker.rescan()
    _finish_scan(picker)
    picker.expanded = True
    labels = []
    put_text = cv2.putText
    monkeypatch.setattr(
        cv2,
        "putText",
        lambda frame, text, *args: (labels.append(text), put_text(frame, text, *args))[1],
    )
    picker.draw(np.zeros((360, 640, 3), dtype=np.uint8), active=2, status="")

    assert {device.index for device in picker.available_devices} == {0, 2}
    assert set(picker.options) == {0, 2}
    assert any("[2] Camera 2  ACTIVE" in text for text in labels)


def test_rescan_replaces_the_device_list(monkeypatch):
    scans = iter([
        [(1, True, None), (2, True, None)],
        [(2, True, None)],
    ])
    monkeypatch.setattr(picker_module, "list_camera_indices", lambda *args, **kwargs: next(scans))
    picker = CameraPicker(360)

    picker.rescan()
    _finish_scan(picker)
    assert [device.index for device in picker.available_devices] == [1, 2]
    picker.rescan()
    _finish_scan(picker)

    assert [device.index for device in picker.available_devices] == [2]


def test_empty_scan_shows_no_cameras_message(monkeypatch):
    monkeypatch.setattr(
        picker_module,
        "list_camera_indices",
        lambda *args, **kwargs: [(0, False, None), (1, False, None)],
    )
    picker = CameraPicker(240, width=480)
    picker.rescan()
    _finish_scan(picker)
    picker.expanded = True
    labels = []
    put_text = cv2.putText
    monkeypatch.setattr(
        cv2,
        "putText",
        lambda frame, text, *args: (labels.append(text), put_text(frame, text, *args))[1],
    )
    picker.draw(np.zeros((240, 480, 3), dtype=np.uint8), active=None, status="")

    assert picker.available_devices == []
    assert picker.options == {}
    assert "Nenhuma câmera disponível" in labels


def test_picker_controls_stay_within_small_window_bounds():
    for width, height in ((640, 360), (480, 240)):
        picker = CameraPicker(height, width=width)
        picker.available_devices = [CameraDevice(index) for index in range(12)]
        picker.expanded = True
        picker.draw(np.zeros((height, width, 3), dtype=np.uint8), active=7, status="")

        rects = [
            picker._panel,
            picker.rescan_button,
            *picker.options.values(),
            picker.previous_button,
            picker.next_button,
        ]
        assert all(
            0 <= left <= right < width and 0 <= top <= bottom < height
            for left, top, right, bottom in rects
        )


def test_picker_collapses_safely_when_window_cannot_fit_panel():
    picker = CameraPicker(80, width=100)
    picker.expanded = True
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    picker.draw(frame, active=None, status="")

    left, top, right, bottom = picker.button
    assert not picker.expanded
    assert 0 <= left <= right < 100
    assert 0 <= top <= bottom < 80


def test_rescan_action_is_requested_once_from_the_panel():
    picker = CameraPicker(360)
    picker.expanded = True
    picker._layout(640, 360)
    x, y, _, _ = picker.rescan_button

    picker.on_mouse(cv2.EVENT_LBUTTONUP, x + 1, y + 1, 0, None)

    assert picker.take_rescan_request()
    assert not picker.take_rescan_request()
