"""
Model/mlp_pytorch.py

Clasificador MLP en PyTorch para entrenamien​to end-to-end distribuido.

Este módulo proporciona una implementación de MLP usando PyTorch
que permite entrenamiento end-to-end (CNN + MLP conjuntamente) con
verdaderas gradientes vía autograd.

Se usa SOLO en modo "end_to_end". El modo "precomputed" sigue
usando la implementación NumPy original (Model/mlp.py).

──────────────────────────────────────────────────────────────────
ARQUITECTURA
──────────────────────────────────────────────────────────────────
MLP de DOS capas ocultas con ReLU:

    Entrada (feature_dim)
        └─► Oculta 1 (hidden1, ReLU)
                └─► Oculta 2 (hidden2, ReLU)
                        └─► Salida (n_classes=10, Softmax)

No hay Batch Norm ni Dropout en este módulo — son decisiones
pedagógicas para mantener la arquitectura simple y comparable
con la versión NumPy.

──────────────────────────────────────────────────────────────────
USO TÍPICO
──────────────────────────────────────────────────────────────────
    import torch
    from Model.mlp_pytorch import MLPPyTorch

    mlp = MLPPyTorch(feature_dim=512, hidden1=256, hidden2=128, n_classes=10)
    mlp = mlp.to(device)  # CPU o GPU

    # Forward
    logits = mlp(features)  # (N, 10)

    # Loss
    loss = torch.nn.functional.cross_entropy(logits, labels)

    # Backward
    loss.backward()

    # Gradient descent
    with torch.no_grad():
        for param in mlp.parameters():
            param.data -= learning_rate * param.grad
            param.grad.zero_()
"""

import torch
import torch.nn as nn
from typing import Dict

import numpy as np


class MLPPyTorch(nn.Module):
    """
    MLP de dos capas ocultas con ReLU para clasificación de features CNN.

    :param feature_dim: Dimensión del vector de entrada (features de CNN).
    :param hidden1: Número de neuronas en capa oculta 1.
    :param hidden2: Número de neuronas en capa oculta 2.
    :param n_classes: Número de clases (10 para CIFAR-10).
    :param seed: Semilla para inicialización de pesos (opcional).
    """

    def __init__(
        self,
        feature_dim: int,
        hidden1: int,
        hidden2: int,
        n_classes: int = 10,
        seed: int | None = None,
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        # He initialization para ReLU
        self.fc1 = nn.Linear(feature_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_classes)

        # He initialization manualmente para mejor control
        def _init_he(layer):
            fan_in = layer.weight.shape[1]
            std = (2.0 / fan_in) ** 0.5
            with torch.no_grad():
                layer.weight.normal_(0, std)
                layer.bias.zero_()

        _init_he(self.fc1)
        _init_he(self.fc2)
        _init_he(self.fc3)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: features → MLP → logits (sin softmax, para cross_entropy).

        EXPLICACIÓN DEL FLUJO:
        ─────────────────────
        • No se aplica Softmax aquí porque PyTorch's cross_entropy combina
          log_softmax y NLL en una operación (es más numéricamente estable).
        • Salida: (N, n_classes) con valores raw (logits), típicamente negativos/positivos.
        • Cada fila es el vector lógits para un ejemplo.

        SHAPES A TRAVÉS DEL MLP:
        (N, feature_dim) → fc1 → (N, hidden1) → ReLU → fc2
                                 → (N, hidden2) → ReLU → fc3 → (N, n_classes)

        :param x: Tensor de entrada con features extraídas de CNN.
        :type x: torch.Tensor de shape (batch_size, feature_dim), dtype float32.

        :return: Logits sin normalizar (sin softmax).
        :rtype: torch.Tensor de shape (batch_size, n_classes) float32.
        """
        x = self.fc1(x)  # (N, feature_dim) → (N, hidden1)
        x = self.relu(x)  # Máximo con 0 para sparsidad
        x = self.fc2(x)  # (N, hidden1) → (N, hidden2)
        x = self.relu(x)
        x = self.fc3(x)  # (N, hidden2) → (N, n_classes)
        return x  # Sin aplicar softmax — sea io para nn.CrossEntropyLoss

    def state_dict_numpy(self) -> dict:
        """
        Convierte weights a NumPy para serialización en protocolo.

        :return: Dict con keys 'fc1.weight', 'fc1.bias', etc.
        """
        numpy_dict = {}
        for name, param in self.named_parameters():
            numpy_dict[name] = param.data.cpu().numpy().copy()
        return numpy_dict

    def load_state_dict_numpy(self, state_dict: Dict[str, "np.ndarray"]) -> None:
        """
        Carga weights desde NumPy (después de promediar en PS).

        :param state_dict: Dict con keys 'fc1.weight', 'fc1.bias', etc.
        """
        for name, param in self.named_parameters():
            if name in state_dict:
                numpy_array = state_dict[name]
                tensor = torch.from_numpy(numpy_array.astype("float32"))
                param.data.copy_(tensor.to(param.device))
