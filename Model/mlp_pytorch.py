"""
Model/mlp_pytorch.py

MLP PyTorch para clasificación sobre features CNN en ImageNet.

ARQUITECTURA:
    features (feature_dim)
        → fc1 (hidden1) → BN → ReLU → Dropout(0.4)
        → fc2 (hidden2) → BN → ReLU → Dropout(0.3)
        → fc3 (n_classes)   ← sin activación ni BN (CrossEntropyLoss la incluye)

CAMBIOS RESPECTO A VERSIÓN ANTERIOR:
    - Eliminado bn2 (BatchNorm en capa de salida): distorsionaba los logits
      que CrossEntropyLoss necesita en su distribución natural. Era además
      una capa fantasma: existía en state_dict pero forward() nunca la usaba.
    - Agregado Dropout(0.4) y Dropout(0.3): mejora generalización con 1000 clases
      donde el MLP tiende a memorizar sobre features preentrenadas de ResNet-18.
    - Inicialización cambiada de Xavier uniform a Kaiming normal: Xavier está
      diseñado para activaciones simétricas (tanh). Con ReLU, Kaiming es
      matemáticamente correcto y produce gradientes más estables en el arranque.
    - Arquitectura definida con nn.Sequential: simplifica forward(), reduce
      código duplicado y hace el grafo de cómputo más legible.

INICIALIZACIÓN:
    Kaiming normal (fan_out, relu) para capas lineales.
    Produce varianza de activaciones ~1 a través de capas profundas con ReLU.

FORMATO DE PARÁMETROS:
    PS y Workers intercambian parámetros como state_dict PyTorch.
    Los nombres cambian respecto a la versión anterior por el Sequential:
        classifier.0.weight / classifier.0.bias     ← fc1
        classifier.1.weight / classifier.1.bias     ← BN1 (weight=gamma, bias=beta)
        classifier.1.running_mean / running_var / num_batches_tracked
        classifier.4.weight / classifier.4.bias     ← fc2
        classifier.5.weight / classifier.5.bias     ← BN2
        classifier.5.running_mean / running_var / num_batches_tracked
        classifier.8.weight / classifier.8.bias     ← fc3

    NOTA: El PS detecta running_mean/running_var/num_batches_tracked por
    subcadena en el nombre, por lo que el cambio a Sequential es transparente
    para el mecanismo de FedAvg del Parameter Server.

DROPOUT Y MODOS train()/eval():
    Dropout se comporta diferente según el modo del modelo:
    - model.train(): desactiva neuronas aleatoriamente (regularización)
    - model.eval():  pasa todas las neuronas (inferencia determinista)
    El worker DEBE llamar model.train() antes del forward de entrenamiento
    y model.eval() antes de evaluar o extraer features. Si no lo hace,
    el Dropout activo durante evaluación degradará el accuracy medido.

LR RECOMENDADO CON ESTA ARQUITECTURA:
    Con ResNet-18 congelado y este MLP:
        LR MLP: 0.001 – 0.01  (no 0.1 — demasiado agresivo para features preentrenadas)
    Con staleness λ=0.1 y 2 workers (staleness promedio ~2):
        α ≈ 0.83  →  LR efectivo real ≈ LR × 0.83
    Con 6 workers el staleness sube; considerar reducir LR o aumentar λ.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from Utils.constants import NUM_CLASSES


class MLPPyTorch(nn.Module):
    """Clasificador MLP de 2 capas ocultas para ImageNet (1000 clases).

    Arquitectura:
    - Input: vector de features CNN (feature_dim, típicamente 512 de ResNet-18)
    - Capa 1: feature_dim → hidden1 + BatchNorm + ReLU + Dropout(0.4)
    - Capa 2: hidden1 → hidden2 + BatchNorm + ReLU + Dropout(0.3)
    - Output: hidden2 → n_classes (logits, sin BN ni activación)

    Diseñado para entrenamiento asíncrono distribuido (Async-SGD):
    - BatchNorm estabiliza features entre updates de distintos workers
    - Dropout reduce overfitting con 1000 clases sobre features preentrenadas
    - Sin BatchNorm en la salida: preserva distribución de logits para CrossEntropyLoss
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

        :param hidden1: Unidades de la 1ª capa oculta (recomendado: 1024).
        :type hidden1: int

        :param hidden2: Unidades de la 2ª capa oculta (recomendado: 512).
        :type hidden2: int

        :param n_classes: Número de clases (defecto: 1000 para ImageNet).
        :type n_classes: int

        :returns: None
        :rtype: None
        """
        super().__init__()

        self.classifier = nn.Sequential(
            # Capa 1: expansión + normalización + activación + regularización
            nn.Linear(feature_dim, hidden1),  # índice 0
            nn.BatchNorm1d(hidden1),  # índice 1
            nn.ReLU(inplace=True),  # índice 2
            nn.Dropout(p=0.4),  # índice 3
            # Capa 2: compresión + normalización + activación + regularización
            nn.Linear(hidden1, hidden2),  # índice 4
            nn.BatchNorm1d(hidden2),  # índice 5
            nn.ReLU(inplace=True),  # índice 6
            nn.Dropout(p=0.3),  # índice 7
            # Salida: logits sin activación ni BN
            nn.Linear(hidden2, n_classes),  # índice 8
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Inicializa pesos con Kaiming normal para capas con ReLU.

        Kaiming normal (fan_out) mantiene la varianza de los gradientes
        constante a través de las capas durante el backpropagation,
        asumiendo activaciones ReLU. Produce arranques más estables que
        Xavier uniform en redes con ReLU.

        BatchNorm se inicializa con gamma=1, beta=0 (identidad) por defecto
        de PyTorch — no requiere inicialización manual.

        :returns: None
        :rtype: None
        """
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ejecuta forward pass a través del clasificador.

        El comportamiento de Dropout depende del modo del modelo:
        - En train(): aplica dropout (regularización activa)
        - En eval(): desactiva dropout (inferencia determinista)

        Llamar model.train() antes de entrenar y model.eval() antes
        de evaluar para garantizar comportamiento correcto.

        :param x: Tensor de features CNN (batch_size, feature_dim)
        :type x: torch.Tensor

        :returns: Logits sin softmax (batch_size, n_classes)
        :rtype: torch.Tensor
        """
        return self.classifier(x)

    def state_dict_numpy(self) -> Dict[str, np.ndarray]:
        """
        Exporta el state_dict completo como Dict[str, np.ndarray].

        Incluye parámetros entrenables (weight, bias) y buffers de BatchNorm
        (running_mean, running_var, num_batches_tracked). Todos convertidos
        a numpy arrays en CPU para serialización TCP hacia el Parameter Server.

        El PS detecta los buffers de BN por subcadena en el nombre de clave,
        por lo que el cambio a Sequential es transparente para FedAvg.

        :returns: Diccionario {nombre_parametro: numpy_array}
                  Ejemplo de claves con Sequential:
                  'classifier.0.weight', 'classifier.1.running_mean', etc.
        :rtype: Dict[str, np.ndarray]
        """
        return {
            name: tensor.data.cpu().numpy().copy() for name, tensor in self.state_dict().items()
        }
