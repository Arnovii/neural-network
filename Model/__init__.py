"""
Model — Lógica central de la red neuronal.

Módulos
-------
cnn_extractor    CNN (PyTorch) para extracción de features de imágenes.
mlp              MLP (NumPy) para clasificación sobre features.
nn               Red neuronal simple (NumPy, sin uso de CNN).

Uso rápido
----------
    from Model import CNNExtractor, init_params, forward_and_gradients, evaluate
"""

from Model.cnn_extractor import CNNExtractor
from Model.mlp import init_params, forward_and_gradients, evaluate, apply_gradients

__all__ = [
    "CNNExtractor",
    "init_params",
    "forward_and_gradients",
    "evaluate",
    "apply_gradients",
]
