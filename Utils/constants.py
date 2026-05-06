"""
Constants del proyecto - Valores de configuración centralizados.

Este módulo contiene todas las constantes usadas en el proyecto para evitar
magic numbers dispersos en el código.

USO:
    from Utils.constants import DEFAULT_BATCH_SIZE, IMAGE_SIZE, COLORS

"""

# ============================================================
# CONFIGURACIÓN DE MODELO (DEFAULT)
# ============================================================

IMAGE_SIZE: int = 224
DEFAULT_BATCH_SIZE: int = 64

# Capas ocultas de la MLP
HIDDEN1_DEFAULT: int = 1024
HIDDEN2_DEFAULT: int = 512

# Learning rates
DEFAULT_LR: float = 0.01
DEFAULT_LR_CNN: float = 0.001
DEFAULT_STALENESS_LAMBDA: float = 0.1

# Valores predeterminados runtime
STEPS_PER_REPORT_DEFAULT: int = 10
METRICS_WINDOW_DEFAULT: int = 50
VAL_BATCHES_DEFAULT: int = 50
WORKER_ACCUM_STEPS_DEFAULT: int = 1
MAX_STEPS_UNLIMITED: int = 0

# Clasificación
NUM_CLASSES: int = 1000

# Feature dimensions
FEATURE_DIM: int = 512

# ============================================================
# CONEXIÓN Y RED
# ============================================================

DEFAULT_HOST: str = "0.0.0.0"  # noqa: S104 (PS server binds to all interfaces intentionally)
DEFAULT_PORT: int = 9999

# Streaming de datos
STREAM_RECONNECT_MAX_ATTEMPTS: int = 5  # Máximo número de reconexiones para ImageNet streaming
STREAM_RECONNECT_DELAY_SECONDS: int = 5  # Segundos de espera entre intentos de reconexión

# ============================================================
# COLORES (GUI & Gráficas)
# ============================================================

COLORS = {
    # Estados
    "offline": "#607D8B",
    "listening": "#795548",
    "loading": "#FF9800",
    "training": "#1565C0",
    # Métricas
    "loss": "#F44336",
    "accuracy": "#2196F3",
    "workers": "#4CAF50",
    "validation": "#FF9800",
    # UI
    "primary": "#2196F3",
    "warning": "#FF9800",
    "error": "#F44336",
    "info": "#1565C0",
    "muted": "#607D8B",
}

# Paleta de la GUI (sidebar, backgrounds)
GUI_COLORS = {
    "sidebar_bg": "#F5F5F5",
    "frame_bg": "#FAFAFA",
    "text": "#212121",
    "label": "#1565C0",
}

# ============================================================
# GUI CONSTANTS
# ============================================================

MAX_LOG_LINES: int = 300
POLL_TIMEOUT_MS: int = 100
CLOCK_UPDATE_MS: int = 1000
GUI_INITIAL_XMAX: int = 4

# Tooltip
TOOLTIP_DELAY_MS: int = 500

# Worker/UI palette
WORKER_COLORS = [
    "#2196F3",
    "#4CAF50",
    "#FF9800",
    "#9C27B0",
    "#F44336",
    "#00BCD4",
    "#795548",
    "#E91E63",
]

# ============================================================
# EXPORTACIÓN
# ============================================================

EXPORT_DIR_DEFAULT: str = "./Exports"

# Validación
VALIDATION_BATCH_SIZE_DEFAULT: int = 256

# Predeterminado en la CLI del worker
WORKER_SERVER_HOST_DEFAULT: str = "127.0.0.1"

# ============================================================
# STREAMING
# ============================================================

PREFETCH_DEFAULT: int = 4
SHUFFLE_BUFFER_DEFAULT: int = 1000

# HuggingFace
HF_DATASET_DEFAULT: str = "ILSVRC/imagenet-1k"

# ============================================================
# ENTRENAMIENTO
# ============================================================

GRAD_CLIP_MAX_NORM: float = 10.0  # Umbral de recorte del gradiente (modo E2E)
LABEL_SMOOTHING: float = 0.1  # Suavizado de etiquetas en CrossEntropyLoss
WEIGHT_DECAY: float = 1e-4  # L2 regularización en SGD (modo E2E)

# Semilla por defecto (None = aleatorio)
DEFAULT_SEED: int | None = None
