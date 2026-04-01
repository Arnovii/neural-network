"""
Model/mlp_pytorch.py

MLP PyTorch para clasificación sobre features CNN en ImageNet.

Este módulo es el único clasificador MLP del sistema. El MLP NumPy
del sistema anterior ha sido eliminado: aquí todo el pipeline
de entrenamiento usa PyTorch de principio a fin, permitiendo
backpropagation E2E sin conversiones de framework.

ARQUITECTURA:
    features (feature_dim)
        → fc1 (hidden1, ReLU)
        → fc2 (hidden2, ReLU)
        → fc3 (n_classes)   ← sin activación (CrossEntropyLoss la incluye)

INICIALIZACIÓN:
    He initialization para capas con ReLU.

FORMATO DE PARÁMETROS:
    El PS y los Workers intercambian parámetros como state_dict PyTorch:
        fc1.weight: (hidden1, feature_dim)
        fc1.bias:   (hidden1,)
        fc2.weight: (hidden2, hidden1)
        fc2.bias:   (hidden2,)
        fc3.weight: (n_classes, hidden2)
        fc3.bias:   (n_classes,)
    No hay transposición ni conversión de keys.
"""

import torch
import torch.nn as nn
from typing import Dict
import numpy as np


class MLPPyTorch(nn.Module):
    """
    MLP de dos capas ocultas con ReLU para clasificación ImageNet.

    :param feature_dim: Dimensión del vector de features de la CNN (512 para ResNet-18).
    :param hidden1:     Neuronas en la primera capa oculta.
    :param hidden2:     Neuronas en la segunda capa oculta.
    :param n_classes:   Clases de salida (1000 para ImageNet).
    """

    def __init__(
        self,
        feature_dim: int,
        hidden1: int,
        hidden2: int,
        n_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_classes)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in (self.fc1, self.fc2, self.fc3):
            fan_in = layer.weight.shape[1]
            std = (2.0 / fan_in) ** 0.5
            nn.init.normal_(layer.weight, 0.0, std)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor (N, feature_dim).
        :return:  Logits (N, n_classes) — sin softmax.
        """
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

    def state_dict_numpy(self) -> Dict[str, np.ndarray]:
        """Devuelve el state_dict como Dict[str, np.ndarray] para transporte por TCP."""
        return {
            name: param.data.cpu().numpy().copy()
            for name, param in self.named_parameters()
        }

    def load_state_dict_numpy(self, state: Dict[str, np.ndarray]) -> None:
        """Carga parámetros desde Dict[str, np.ndarray] (formato del PS)."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in state:
                    param.data.copy_(
                        torch.from_numpy(state[name]).to(param.device)
                    )