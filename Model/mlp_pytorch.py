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

from Utils.constants import NUM_CLASSES


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
        n_classes: int = NUM_CLASSES,
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
        self.bn0 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn1 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, n_classes)
        self.bn2 = nn.BatchNorm1d(n_classes)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Xavier uniform initialization para capas, compatible con BatchNorm1d.

        Xavier produce activaciones con varianza más predecible (~1) compatible
        con BatchNorm. En contraste con Kaiming, evita normas grandes que producen
        updates pequeños en Async-SGD distribuido.
        """
        # Itera sobre las 3 capas lineales del modelo
        for layer in (self.fc1, self.fc2, self.fc3):
            # Xavier uniform es más compatible con BatchNorm que Kaiming
            nn.init.xavier_uniform_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ejecuta forward pass a través del MLP con BatchNorm1d.

        Arquitectura: fc1 → bn0 → relu → fc2 → bn1 → relu → fc3 (sin BN en salida).
        BatchNorm estabiliza features para Async-SGD distribuido.

        :param x: Tensor de entrada con features CNN (batch_size, feature_dim)
        :type x: torch.Tensor

        :returns: Logits sin softmax (batch_size, n_classes)
        :rtype: torch.Tensor
        """
        x = self.relu(self.bn0(self.fc1(x)))
        x = self.relu(self.bn1(self.fc2(x)))
        x = self.fc3(x)
        return x

    def state_dict_numpy(self) -> Dict[str, np.ndarray]:
        """
        Exporta el state_dict completo (parámetros + buffers BN) como Dict[str, np.ndarray].

        Convierte todos los parámetros entrenables Y buffers de BatchNorm (running_mean, running_var, num_batches_tracked)
        a numpy arrays en CPU. Útil para serializar el modelo a través de TCP hacia el Parameter Server.

        :returns: Diccionario con claves de parámetros y buffers, valores como numpy arrays.
                  Ejemplo: {'fc1.weight': array(...), 'fc1.bias': array(...), 'bn0.running_mean': array(...), ...}
        :rtype: Dict[str, np.ndarray]
        """
        return {
            name: tensor.data.cpu().numpy().copy()
            for name, tensor in self.state_dict().items()
        }
