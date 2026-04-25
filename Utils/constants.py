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

# MLP hidden layers
HIDDEN1_DEFAULT: int = 1024
HIDDEN2_DEFAULT: int = 512

# Learning rates
DEFAULT_LR: float = 0.01
DEFAULT_LR_CNN: float = 0.001
DEFAULT_STALENESS_LAMBDA: float = 0.1

# Report & window
STEPS_PER_REPORT: int = 500
METRICS_WINDOW: int = 200
VAL_BATCHES_DEFAULT: int = 50

# Clasificación
NUM_CLASSES: int = 1000

# Feature dimensions por arquitectura
RESNET18_FEATURE_DIM: int = 512
SIMPLECNN_FEATURE_DIM: int = 512

# ============================================================
# CONEXIÓN Y MÉTRICAS
# ============================================================

DEFAULT_HOST: str = "0.0.0.0"
DEFAULT_PORT: int = 9999

STEPS_PER_REPORT: int = 10
METRICS_WINDOW: int = 50

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

# Tooltip
TOOLTIP_DELAY_MS: int = 500

# ============================================================
# EXPORTACIÓN
# ============================================================

EXPORT_DIR_DEFAULT: str = "./Exports"

# ============================================================
# STREAMING
# ============================================================

PREFETCH_DEFAULT: int = 4
SHUFFLE_BUFFER_DEFAULT: int = 1000

# HuggingFace
HF_DATASET_DEFAULT: str = "ILSVRC/imagenet-1k"
