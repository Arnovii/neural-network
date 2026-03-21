"""
Model — Lógica central de la red neuronal CNN + MLP distribuida.

──────────────────────────────────────────────────────────────────
ARQUITECTURA
──────────────────────────────────────────────────────────────────
    Imagen (3×H×W)
        │
        ▼  CNNExtractor — PyTorch, pesos preentrenados o propios
        │
        ▼  Feature vector (FEATURE_DIM,)
        │
        ▼  MLP NumPy — pesos distribuidos por Algoritmo de Diego
        │
        ▼  Logits (10,) → Softmax → Predicción

──────────────────────────────────────────────────────────────────
Módulos
──────────────────────────────────────────────────────────────────
cnn_extractor    CNNExtractor: extracción de features con caché.
                 Arquitecturas: "simple" (propia) o "resnet18" (torchvision).

mlp              Clasificador MLP NumPy con 2 capas ocultas.
                 Funciones: init_params, forward_and_gradients,
                           evaluate, apply_gradients.

──────────────────────────────────────────────────────────────────
Uso rápido
──────────────────────────────────────────────────────────────────
    from Model import (
        CNNExtractor, FEATURE_DIM,
        init_params, forward_and_gradients, evaluate, apply_gradients
    )
"""

from Model.cnn_extractor import CNNExtractor, FEATURE_DIM
from Model.mlp import (
    init_params,
    forward_and_gradients,
    evaluate,
    apply_gradients,
)

__all__ = [
    "CNNExtractor",
    "FEATURE_DIM",
    "init_params",
    "forward_and_gradients",
    "evaluate",
    "apply_gradients",
]
