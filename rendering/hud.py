"""HUD and text overlay utilities."""

from __future__ import annotations

import cv2
import numpy as np


def draw_hud(frame: np.ndarray, state, fps: float, hands_detected: int = 0, gesture: str = "IDLE") -> None:
    height, width = frame.shape[:2]
    panel_w = min(420, width - 20)
    panel_h = 165
    x0, y0 = 12, 12

    overlay = frame[y0:y0 + panel_h, x0:x0 + panel_w].copy()
    alpha = 0.7
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (255, 120, 140), 1)

    lines = [
        "MUAC / MAGIC HAND LAB",
        f"MODE       {state.active_effect}",
        f"HANDS      {hands_detected} / 2",
        f"GESTURE    {gesture}",
        f"PARTICLES  {state.particle_count}",
        f"FPS        {fps:.1f}",
    ]

    for idx, line in enumerate(lines):
        y = y0 + 26 + idx * 20
        cv2.putText(frame, line, (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

    if state.show_debug:
        debug_lines = [
            f"CAMERA     {state.camera_resolution}",
            f"DISPLAY    {state.display_resolution}",
            f"TRACKER    {'ON' if state.tracker_active else 'OFF'}",
            f"EFFECT     {state.active_effect}",
        ]
        debug_x = min(width - 220, x0 + panel_w + 12)
        debug_y = y0
        cv2.rectangle(frame, (debug_x, debug_y), (debug_x + 190, debug_y + 90), (30, 30, 30), -1)
        for idx, line in enumerate(debug_lines):
            cv2.putText(frame, line, (debug_x + 10, debug_y + 24 + idx * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    if state.show_landmarks:
        cv2.putText(frame, "LANDMARKS ON", (x0 + 16, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
