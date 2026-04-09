"""
Model/mlp_pytorch.py

MLP PyTorch para clasificación sobre features CNN en ImageNet.

ARQUITECTURA:
    fc = Fully Connected
    features (feature_dim)
        → fc1 (hidden1, ReLU)
        → fc2 (hidden2, ReLU)
        → fc3 (n_classes)   ← sin activación (CrossEntropyLoss la incluye)

INICIALIZACIÓN:
    Kaiming uniform (He) para capas con ReLU.
    Produce logits con varianza razonable desde el primer paso,
    evitando softmax uniforme y accuracy≈0% en las primeras iteraciones.

FORMATO DE PARÁMETROS:
    PS y Workers intercambian parámetros como state_dict PyTorch:
        fc1.weight: (hidden1, feature_dim)
        fc1.bias:   (hidden1,)
        fc2.weight: (hidden2, hidden1)
        fc2.bias:   (hidden2,)
        fc3.weight: (n_classes, hidden2)
        fc3.bias:   (n_classes,)
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn


class MLPPyTorch(nn.Module):
    """
    Clasificador MLP de 2 capas ocultas para ImageNet (1000 clases).

    Arquitectura:
    - Input: vector de features CNN (feature_dim, ej 512 de ResNet-18)
    - Capa 1: feature_dim → hidden1 (ej 1024) + ReLU
    - Capa 2: hidden1 → hidden2 (ej 512) + ReLU
    - Output: hidden2 → 1000 (logits sin activación)

    Thread-safe: Múltiples Workers cargan state_dict sin conflictos.
    Serialización: state_dict_numpy() para transporte por TCP (numpy arrays).
    Inicialización: Kaiming uniform (He) para producir logits con varianza razonable.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden1: int,
        hidden2: int,
        n_classes: int = 1000,
    ) -> None:
        """
        Inicializa el MLP con arquitectura configurable.

        :param feature_dim: Dimensión del vector de entrada CNN (ej: 512 para ResNet-18)
        :type feature_dim: int

        :param hidden1: Unidades de la 1ª capa oculta (defecto config PS: 1024).
                       Distribuida por PS a todos los Workers via CONFIG.
        :type hidden1: int

        :param hidden2: Unidades de la 2ª capa oculta (defecto config PS: 512).
                       Distribuida por PS a todos los Workers via CONFIG.
        :type hidden2: int

        :param n_classes: Número de clases (defecto: 1000 para ImageNet)
        :type n_classes: int
        """
        super().__init__()
        self.fc1 = nn.Linear(
            feature_dim, hidden1
        )  # nn.Linear = Capa totalmente conectada
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_classes)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Kaiming uniform (He) initialization para capas con ReLU.

        Produce activaciones con varianza ~1 en cada capa, lo que
        garantiza que los logits iniciales sean distintos de cero y el
        loss sea ≈ log(n_classes) ≈ 6.9 desde el primer batch.
        """
        # Itera sobre las 3 capas lineales del modelo
        for layer in (self.fc1, self.fc2, self.fc3):
            # El modo fan_in hace que los pesos se escalen según cuántas entradas tiene la capa.
            nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ejecuta forward pass a través del MLP (3 capas totalmente conectadas).

        Aplica transformación lineal + ReLU en capas ocultas, sin activación en salida.

        :param x: Tensor de entrada con features CNN (batch_size, feature_dim)
        :type x: torch.Tensor

        :returns: Logits sin softmax (batch_size, n_classes)
        :rtype: torch.Tensor
        """
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

    def state_dict_numpy(self) -> Dict[str, np.ndarray]:
        """
        Exporta el state_dict de parámetros como Dict[str, np.ndarray].

        Convierte todos los parámetros entrenables (pesos y sesgos) a numpy arrays
        en CPU. Útil para serializar el modelo a través de TCP hacia el Parameter Server.

        :returns: Diccionario con claves de parámetros y valores como numpy arrays.
                  Ejemplo: {'fc1.weight': array(...), 'fc1.bias': array(...), ...}
        :rtype: Dict[str, np.ndarray]
        """
        return {
            name: param.data.cpu().numpy().copy()
            for name, param in self.named_parameters()
        }
