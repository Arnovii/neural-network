"""
Model — Lógica central de la red neuronal.

Módulos
-------
nn    Inicialización, forward pass, pérdida y actualización de pesos.

Uso rápido
----------
    from Model.nn import init_params, forward_pass, cross_entropy_loss, apply_gradients
"""

from Model.nn import init_params, forward_pass, cross_entropy_loss, apply_gradients

__all__ = [
    "init_params",
    "forward_pass",
    "cross_entropy_loss",
    "apply_gradients",
]
