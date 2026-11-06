"""Configuração da câmera, das partículas e dos limiares de gesto do MUAC."""

import cv2

CAMERA_INDEX = 0
CAMERA_BACKEND = "DSHOW"
CAMERA_PROBE_START = 0
CAMERA_PROBE_END = 10
CAMERA_BACKEND_MAP = {
    "AUTO": cv2.CAP_ANY,
    "DSHOW": cv2.CAP_DSHOW,
    "MF": cv2.CAP_MSMF,
}

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_DARKNESS = 0.45
SHOW_LANDMARKS = True

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
WINDOW_RESIZABLE = True

PARTICLE_COUNT = 2000
PARTICLE_RADIUS = 1
PARTICLE_BASE_COLOR = (40, 40, 255)  # Vermelho vivo na ordem BGR usada pelo OpenCV.
HEART_BASE_SCALE = 120.0
START_WITH_HEART = False

REPEL_RADIUS = 90.0
REPEL_FORCE = 3.0
SWIPE_FORCE = 0.08
GRAB_RADIUS = 180.0
GRAB_FORCE = 0.95
GRAB_FOLLOW_FORCE = 0.15
GRAB_CLOUD_RADIUS = 22.0  # Limite em pixels para aproximar a nuvem da pinça em um quadro.

SHAPE_FORCE = 0.018
DAMPING = 0.86
MIN_SCALE = 0.45
MAX_SCALE = 2.0
TRANSFORM_MOVE_SMOOTHING = 0.35
TRANSFORM_SCALE_SMOOTHING = 0.25
TRANSFORM_SCALE_STEP_MIN = 0.97
TRANSFORM_SCALE_STEP_MAX = 1.03
TRANSFORM_ROTATE_SMOOTHING = 0.25
TRANSFORM_ROTATE_STEP_MAX = 4.0  # Limite de rotação por quadro, em graus.

GESTURE_HOLD_TIME = 0.7
LETTER_CHANGE_HOLD_TIME = 2.0
CONTINUOUS_HOLD_TIME = 0.05
PINCH_THRESHOLD = 0.42
SNAP_CLOSE_RATIO = 0.48
SNAP_RELEASE_RATIO = 0.85
SNAP_MAX_TIME = 0.50
SNAP_MIN_SPEED = 10.0  # Deslocamento mínimo do dedo médio, em pixels por quadro a 30 FPS.
SNAP_COOLDOWN = 0.8

# Retângulo de contorno. As distâncias usam a média do tamanho das palmas,
# tornando os limiares proporcionais em diferentes resoluções e tamanhos de mão.
RECTANGLE_ARM_DISTANCE_PALMS = 2.0  # Arma se os pontos de referência estiverem a até 2 palmas.
RECTANGLE_START_SEPARATION_PALMS = 0.35  # Abertura adicional necessária após armar.
RECTANGLE_POINT_SMOOTHING = 0.35  # Fator EMA por quadro; valores maiores seguem mais rápido.
RECTANGLE_TRACKING_GRACE_FRAMES = 4  # Quadros sem mãos antes de iniciar o fechamento.
RECTANGLE_FADE_SECONDS = 0.25  # Duração do fade ao perder o gesto ou tracking.
RECTANGLE_COLOR = (255, 255, 0)  # Cyan em BGR.
RECTANGLE_THICKNESS = 2  # Espessura da borda OpenCV, em pixels.
RECTANGLE_L_MIN_ANGLE = 35.0  # Ângulo mínimo entre os eixos do polegar e indicador.
RECTANGLE_L_MAX_ANGLE = 145.0  # Ângulo máximo entre os eixos do polegar e indicador.

SHOW_FPS = True

TEST_MODE = False
