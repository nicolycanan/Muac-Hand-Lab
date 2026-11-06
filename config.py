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
CAMERA_FPS = 60
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
GRAB_RADIUS = 150.0
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

HEART_OVERLOAD_ENABLED = True  # Explode o coração ao ampliar além do limite em TRANSFORM/PARTICLES.
HEART_OVERLOAD_SCALE = 1.85  # Escala efetiva para iniciar a condição de overload (MAX_SCALE é 2.0).
HEART_OVERLOAD_HOLD_SECONDS = 0.15  # Tempo acima do limite para confirmar o overload.
HEART_OVERLOAD_STRENGTH = 22.0  # Intensidade do impulso radial aplicado às partículas.

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
RECTANGLE_TRACKING_GRACE_FRAMES = 4  # Quadros sem mãos antes de bloquear a geometria criada.
RECTANGLE_FADE_SECONDS = 0.25  # Duração do fade no estado CLOSING, usado pela exclusão explícita.
RECTANGLE_MAX_INSTANCES = 6  # Máximo de retângulos simultâneos, incluindo os que estão fechando.
RECTANGLE_STYLE_SWITCH_HOLD = 0.45  # Tempo de pinça dentro do retângulo para trocar o preenchimento.
RECTANGLE_DEFAULT_FILL_STYLE = "PSYCHEDELIC"  # Estilo inicial independente para cada retângulo.
RECTANGLE_SCREEN_MARGIN = 8  # Margem mínima mantida entre a figura e as bordas do frame.
RECTANGLE_DELETE_ENABLED = True  # Permite excluir ao reduzir durante transformação ativa.
RECTANGLE_DELETE_WIDTH_RATIO = 0.10  # Largura máxima para iniciar a contagem de exclusão.
RECTANGLE_DELETE_HEIGHT_RATIO = 0.10  # Altura máxima para iniciar a contagem de exclusão.
RECTANGLE_DELETE_HOLD_SECONDS = 0.30  # Tempo contínuo na zona pequena antes do fade-out.
RECTANGLE_DELETE_HYSTERESIS = 1.25  # Multiplicador para cancelar se a figura voltar a crescer.
RECTANGLE_COLOR = (255, 255, 0)  # Cyan em BGR; mantido para compatibilidade, sem borda nesta versão.
RECTANGLE_THICKNESS = 2  # Espessura legada; não há contorno renderizado nesta versão.
RECTANGLE_L_MIN_ANGLE = 35.0  # Ângulo mínimo entre os eixos do polegar e indicador.
RECTANGLE_L_MAX_ANGLE = 145.0  # Ângulo máximo entre os eixos do polegar e indicador.

# Preenchimento psicodélico. A textura processa somente o interior do retângulo.
PSYCHEDELIC_FILL_ENABLED = True  # Desativa completamente o cálculo e o desenho da textura.
PSYCHEDELIC_FILL_OPACITY = 0.30  # Mistura com a imagem da câmera; intervalo válido de 0 a 1.
PSYCHEDELIC_FILL_SPEED = 2.4  # Multiplicador do tempo da animação.
PSYCHEDELIC_FILL_SCALE = 1.5  # Multiplicador das frequências das ondas.
PSYCHEDELIC_FILL_SATURATION = 1.0  # Saturação HSV; valores acima de 1 são limitados.
PSYCHEDELIC_FILL_HUE_SHIFT = 0.0  # Deslocamento da paleta HSV, em graus (0 a 360).
PSYCHEDELIC_FILL_MAX_DIMENSION = 320  # Maior dimensão do cálculo interno da textura, em pixels.
MONOCHROME_CONTRAST = 1.1  # Contraste do plasma monocromático ao redor do cinza médio.
GLITCH_INTENSITY = 0.35  # Intensidade das distorções internas, entre 0 e 1.
GLITCH_BLOCK_SIZE = 12  # Tamanho base, em pixels de textura, das faixas e blocos.
GLITCH_BAND_COUNT = 12  # Quantidade de faixas horizontais deslocadas por quadro.
GLITCH_RGB_SHIFT = 2  # Deslocamento máximo dos canais azul/vermelho, em pixels.
GLITCH_SCANLINE_STRENGTH = 0.08  # Escurecimento leve das linhas de varredura.
GLITCH_OPACITY = 0.88  # Mistura forte da imagem distorcida com a webcam na ROI.
GLITCH_SPEED = 1.0  # Multiplicador temporal das distorções digitais.

SHOW_FPS = True

TEST_MODE = False
