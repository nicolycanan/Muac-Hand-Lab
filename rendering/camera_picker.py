"""Seletor de índices de câmera na janela do OpenCV."""

from __future__ import annotations

import cv2
import numpy as np

from config import CAMERA_INDEX, CAMERA_PROBE_END, CAMERA_PROBE_START


class CameraPicker:
    def __init__(self, height: int):
        self.button = (12, height - 72, 190, height - 34)
        self.options = {}
        indices = range(CAMERA_PROBE_START, CAMERA_PROBE_END + 1)
        for position, index in enumerate(indices):
            x = 12 + (position % 3) * 120
            y = self.button[1] - 38 * (position // 3 + 1) - 6
            self.options[index] = (x, y, x + 114, y + 32)
        self.expanded = False
        self.requested_index: int | None = None

    @staticmethod
    def _contains(rect, x, y):
        left, top, right, bottom = rect
        return left <= x <= right and top <= y <= bottom

    def on_mouse(self, event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONUP:
            return
        if self._contains(self.button, x, y):
            self.expanded = not self.expanded
        elif self.expanded:
            self.expanded = False
            for index, rect in self.options.items():
                if self._contains(rect, x, y):
                    self.requested_index = index
                    break

    def take_selection(self) -> int | None:
        index = self.requested_index
        self.requested_index = None
        return index

    @staticmethod
    def _draw_button(frame, rect, label, selected=False):
        left, top, right, bottom = rect
        border = (190, 210, 110) if selected else (255, 120, 140)
        cv2.rectangle(frame, (left, top), (right, bottom), (24, 24, 24), -1)
        cv2.rectangle(frame, (left, top), (right, bottom), border, 1)
        cv2.putText(frame, label, (left + 10, top + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (245, 245, 245), 1, cv2.LINE_AA)

    def draw(self, frame: np.ndarray, active: int | None, status: str) -> None:
        label = "CAMERA: --" if active is None else f"CAMERA: {active}"
        self._draw_button(frame, self.button, label + "  [C]", self.expanded)
        if status:
            x = self.button[2] + 12
            cv2.putText(frame, status, (x, self.button[1] + 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (180, 210, 255), 1, cv2.LINE_AA)
        if self.expanded:
            for index, rect in self.options.items():
                label = f"{index} - Padrao" if index == CAMERA_INDEX else f"Camera {index}"
                self._draw_button(frame, rect, label, index == active)
