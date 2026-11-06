"""Estado compartilhado entre o loop principal, os efeitos e o painel de informações."""

from dataclasses import dataclass, field

from config import PARTICLE_COUNT


@dataclass
class AppState:
    active_effect: str = "PARTICLE_HEART"
    gesture: str = "IDLE"
    confirmed_gesture: str = "IDLE"
    hands_detected: int = 0
    heart_enabled: bool = True
    gesture_enabled: bool = True
    show_landmarks: bool = False
    show_debug: bool = False
    transform_active: bool = False
    tracker_active: bool = True
    last_status: str = "READY"
    fps: float = 0.0
    particle_count: int = PARTICLE_COUNT
    camera_resolution: str = "640x480"
    display_resolution: str = "1280x720"
    labels: dict = field(default_factory=dict)
