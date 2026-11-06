"""Compact camera-source panel backed by asynchronous device discovery."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread

import cv2
import numpy as np

from config import CAMERA_BACKEND, CAMERA_PROBE_END, CAMERA_PROBE_START, DISPLAY_WIDTH
from core.camera import list_camera_indices


@dataclass(frozen=True)
class CameraDevice:
    index: int
    resolution: tuple[int, int] | None = None

    @property
    def name(self) -> str:
        return f"Camera {self.index}"


class CameraPicker:
    PANEL_WIDTH = 360
    PANEL_MAX_HEIGHT = 440
    ROW_HEIGHT = 34
    HEADER_HEIGHT = 34
    RESCAN_HEIGHT = 32
    INFO_HEIGHT = 22
    STATUS_HEIGHT = 18
    FOOTER_HEIGHT = 28
    MARGIN = 8

    def __init__(
        self,
        height: int,
        width: int = DISPLAY_WIDTH,
        probe_start: int = CAMERA_PROBE_START,
        probe_end: int = CAMERA_PROBE_END,
        backend_name: str = CAMERA_BACKEND,
    ):
        self.height = height
        self.width = width
        self.probe_start = probe_start
        self.probe_end = probe_end
        self.backend_name = backend_name
        self.button = (12, max(0, height - 52), min(width - 12, 340), max(1, height - 12))
        self.options: dict[int, tuple[int, int, int, int]] = {}
        self.header_button = (0, 0, 0, 0)
        self.rescan_button = (0, 0, 0, 0)
        self.previous_button = (0, 0, 0, 0)
        self.next_button = (0, 0, 0, 0)
        self.expanded = False
        self.page = 0
        self.available_devices: list[CameraDevice] = []
        self.scanning = False
        self.scan_error = ""
        self.requested_index: int | None = None
        self.rescan_requested = False
        self._results: Queue[tuple[list[tuple[int, bool, tuple[int, int] | None]], str]] = Queue()
        self._cancel_scan = Event()
        self._scan_thread: Thread | None = None
        self._scale = 1.0

    @staticmethod
    def _contains(rect, x, y):
        left, top, right, bottom = rect
        return left <= x <= right and top <= y <= bottom



    def _layout(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        if width < 120 or height < 96:
            self.expanded = False
            self._scale = 0.5
            left = min(12, max(0, width // 8))
            top = max(0, height - 40)
            self.button = (
                left,
                top,
                max(left, min(width - 1, width - left)),
                max(top, height - 4),
            )
            self.options = {}
            return
        self._scale = min(1.0, max(0.5, (height - 24) / self.PANEL_MAX_HEIGHT))
        collapsed_left = min(12, max(0, width // 8))
        self.button = (
            collapsed_left,
            max(0, height - 52),
            max(collapsed_left, min(width - collapsed_left, 340)),
            height - 12,
        )
        if not self.expanded:
            self.options = {}
            return

        margin = max(4, int(round(self.MARGIN * self._scale)))
        header_height = max(18, int(round(self.HEADER_HEIGHT * self._scale)))
        rescan_height = max(18, int(round(self.RESCAN_HEIGHT * self._scale)))
        info_height = max(14, int(round(self.INFO_HEIGHT * self._scale)))
        status_height = 0 if self._scale < 0.7 else max(14, int(round(self.STATUS_HEIGHT * self._scale)))
        row_height = max(20, int(round(self.ROW_HEIGHT * self._scale)))
        footer_height = max(18, int(round(self.FOOTER_HEIGHT * self._scale)))
        panel_width = min(self.PANEL_WIDTH, width - 2 * margin)
        panel_height = min(
            max(1, int(round(self.PANEL_MAX_HEIGHT * self._scale))),
            height - 2 * margin,
        )
        base_height = 2 * margin + header_height + rescan_height + info_height + status_height
        row_capacity = max(1, (panel_height - base_height - footer_height) // row_height)
        visible_rows = min(max(1, len(self.available_devices)), row_capacity)
        page_count = max(1, (len(self.available_devices) + visible_rows - 1) // visible_rows)
        self.page = min(self.page, page_count - 1)
        actual_footer_height = footer_height if page_count > 1 else 0
        panel_height = min(
            height - 2 * margin,
            base_height + visible_rows * row_height + actual_footer_height,
        )
        left = max(margin, min(12, width - panel_width - margin))
        top = max(margin, height - panel_height - margin)
        right = left + panel_width
        header_top = top + margin
        rescan_width = max(48, int(round(100 * self._scale)))
        rescan_left = max(left + margin, right - margin - rescan_width)
        self.header_button = (left + margin, header_top, rescan_left, header_top + header_height - 2)
        self.rescan_button = (rescan_left, header_top, right - margin, header_top + header_height - 2)
        cursor = header_top + header_height
        self.info_rect = (left + margin, cursor, right - margin, cursor + info_height)
        cursor += info_height
        self.status_rect = (left + margin, cursor, right - margin, cursor + status_height)
        cursor += status_height

        start = self.page * visible_rows
        page_devices = self.available_devices[start:start + visible_rows]
        self.options = {}
        for device in page_devices:
            self.options[device.index] = (
                left + margin,
                cursor,
                right - margin,
                cursor + row_height - 4,
            )
            cursor += row_height

        self.previous_button = (
            left + margin,
            top + panel_height - margin - footer_height,
            min(right - margin, left + margin + int(round(84 * self._scale))),
            top + panel_height - margin,
        )
        self.next_button = (
            max(left + margin, right - margin - int(round(84 * self._scale))),
            top + panel_height - margin - footer_height,
            right - margin,
            top + panel_height - margin,
        )
        self._page_count = page_count
        self._visible_rows = visible_rows
        self._footer_height = actual_footer_height
        self._margin = margin
        self._header_height = header_height
        self._row_height = row_height
        self._font_scale = max(0.24, 0.42 * self._scale)
        self._panel = (left, top, right, top + panel_height)

    def on_mouse(self, event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONUP:
            return
        if self._contains(self.button, x, y) and not self.expanded:
            self.expanded = True
            self.page = 0
            self._layout(self.width, self.height)
            return
        if not self.expanded:
            return
        if self._contains(self.header_button, x, y):
            self.expanded = False
            return
        if self._contains(self.rescan_button, x, y):
            if not self.scanning:
                self.rescan_requested = True
            return
        if self._page_count > 1 and self._contains(self.previous_button, x, y):
            self.page = max(0, self.page - 1)
            return
        if self._page_count > 1 and self._contains(self.next_button, x, y):
            self.page = min(self._page_count - 1, self.page + 1)
            return
        if not self.scanning:
            for index, rect in self.options.items():
                if self._contains(rect, x, y):
                    self.requested_index = index
                    self.expanded = False
                    break

    def take_selection(self) -> int | None:
        index = self.requested_index
        self.requested_index = None
        return index

    def mark_unavailable(self, index: int) -> None:
        self.available_devices = [
            device for device in self.available_devices if device.index != index
        ]
        self.page = min(
            self.page,
            max(0, (len(self.available_devices) - 1) // max(1, getattr(self, "_visible_rows", 1))),
        )

    def take_rescan_request(self) -> bool:
        requested = self.rescan_requested
        self.rescan_requested = False
        return requested

    def rescan(self, active_index: int | None = None, active_available: bool = False) -> bool:
        if self.scanning:
            return False
        if self._scan_thread is not None and self._scan_thread.is_alive():
            return False
        self._cancel_scan.clear()
        self.scanning = True
        self.scan_error = ""

        def scan() -> None:
            try:
                results = list_camera_indices(
                    self.probe_start,
                    self.probe_end,
                    self.backend_name,
                    active_index=active_index,
                    active_available=active_available,
                    cancel_event=self._cancel_scan,
                )
                self._results.put((results, ""))
            except Exception as error:
                self._results.put(([], str(error)))

        self._scan_thread = Thread(target=scan, name="camera-discovery", daemon=True)
        self._scan_thread.start()
        return True

    def poll_scan_results(self) -> bool:
        try:
            results, error = self._results.get_nowait()
        except Empty:
            return False
        self.scanning = False
        self.scan_error = error
        if not error:
            self.available_devices = [
                CameraDevice(index, resolution)
                for index, available, resolution in results
                if available
            ]
            self.page = min(
                self.page,
                max(0, (len(self.available_devices) - 1) // max(1, getattr(self, "_visible_rows", 1))),
            )
        return True

    def close(self) -> None:
        self._cancel_scan.set()
        if self._scan_thread is not None and self._scan_thread.is_alive():
            self._scan_thread.join(timeout=0.1)

    @staticmethod
    def _draw_button(frame, rect, label, selected=False, enabled=True):
        left, top, right, bottom = rect
        if right <= left or bottom <= top:
            return
        border = (255, 255, 0) if selected else (95, 120, 130)
        text = (220, 245, 245) if enabled else (120, 130, 130)
        cv2.rectangle(frame, (left, top), (right, bottom), (22, 24, 28), -1)
        cv2.rectangle(frame, (left, top), (right, bottom), border, 1)
        cv2.putText(
            frame,
            label,
            (left + 5, top + max(1, min(bottom - top - 2, int((bottom - top) * 0.68)))),
            cv2.FONT_HERSHEY_SIMPLEX,
            min(0.42, max(0.22, (bottom - top) * 0.014)),
            text,
            1,
            cv2.LINE_AA,
        )

    def draw(self, frame: np.ndarray, active: int | None, status: str) -> None:
        height, width = frame.shape[:2]
        self._layout(width, height)
        if not self.expanded:
            if width >= 320:
                label = "MUAC / VIDEO SOURCES  [C]"
            elif width >= 120:
                label = "VIDEO [C]"
            else:
                label = ""
            self._draw_button(
                frame,
                self.button,
                label,
                selected=False,
            )
            return

        left, top, right, bottom = self._panel
        cv2.rectangle(frame, (left, top), (right, bottom), (18, 20, 24), -1)
        cv2.rectangle(frame, (left, top), (right, bottom), (70, 90, 100), 1)
        cv2.putText(
            frame,
            "MUAC / VIDEO SOURCES" if self._scale >= 0.7 else "VIDEO SOURCES",
            (left + self._margin, top + max(12, int(23 * self._scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.26, 0.48 * self._scale),
            (225, 235, 238),
            1,
            cv2.LINE_AA,
        )
        self._draw_button(
            frame,
            self.rescan_button,
            ("SCAN..." if self.scanning else "SCAN") if self._scale < 0.7 else (
                "SCANNING..." if self.scanning else "RESCAN"
            ),
            enabled=not self.scanning,
        )
        if self.scanning:
            info = "Procurando fontes de vídeo..."
        elif not self.available_devices:
            info = "Nenhuma câmera disponível"
        else:
            info = f"{len(self.available_devices)} fonte(s) disponível(is)"
        cv2.putText(
            frame,
            info[: max(4, (right - left - 16) // 6)],
            (left + self._margin, self.info_rect[3] - max(3, int(5 * self._scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.22, 0.37 * self._scale),
            (165, 190, 195),
            1,
            cv2.LINE_AA,
        )
        status_text = (
            f"Falha ao procurar câmeras: {self.scan_error}"
            if self.scan_error
            else status
        )
        if status_text and self.status_rect[3] > self.status_rect[1]:
            cv2.putText(
                frame,
                status_text[: max(4, (right - left - 16) // 6)],
                (left + self._margin, self.status_rect[3] - max(3, int(4 * self._scale))),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.2, 0.32 * self._scale),
                (180, 170, 150),
                1,
                cv2.LINE_AA,
            )

        for index, rect in self.options.items():
            device = next(device for device in self.available_devices if device.index == index)
            suffix = "  ACTIVE" if index == active else ""
            self._draw_button(
                frame,
                rect,
                f"[{index}] {device.name}{suffix}",
                selected=index == active,
                enabled=not self.scanning,
            )

        if self._page_count > 1:
            self._draw_button(
                frame, self.previous_button, "PREV",
                enabled=self.page > 0,
            )
            self._draw_button(
                frame, self.next_button, "NEXT",
                enabled=self.page < self._page_count - 1,
            )
            cv2.putText(
                frame, f"{self.page + 1}/{self._page_count}",
                (left + (right - left) // 2 - 14, self.previous_button[3] - max(3, int(8 * self._scale))),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.22, 0.38 * self._scale),
                (165, 190, 195),
                1,
                cv2.LINE_AA,
            )
