"""
Model/nn.py

Lógica central de la red neuronal compartida entre el Parameter Server
y los Workers: inicialización de parámetros, forward pass, cálculo de
pérdida y actualización de pesos.

Centralizar estas operaciones evita duplicar código entre
``Distributed/parameter_server.py`` y ``Distributed/worker_node.py``.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from Utils.math_utils import sigmoid, softmax, xavier_initialization, vector_zeros


# ================================================================
# INICIALIZACIÓN
# ================================================================


def init_params(
    input_size: int,
    hidden_size: int,
    output_size: int,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Inicializa los parámetros de la red con Xavier.

    :param input_size: Neuronas de entrada.
    :type input_size: int

    :param hidden_size: Neuronas en la capa oculta.
    :type hidden_size: int

    :param output_size: Neuronas de salida (clases).
    :type output_size: int

    :param seed: Semilla aleatoria para reproducibilidad.
    :type seed: int | None

    :return: Diccionario con W1, b1, W2, b2.
    :rtype: Dict[str, np.ndarray]
    """
    if seed is not None:
        np.random.seed(seed)

    return {
        "W1": xavier_initialization(input_size, hidden_size),
        "b1": vector_zeros(hidden_size),
        "W2": xavier_initialization(hidden_size, output_size),
        "b2": vector_zeros(output_size),
    }


# ================================================================
# FORWARD PASS
# ================================================================


def forward_pass(
    params: Dict[str, np.ndarray],
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward pass de la red de dos capas (sigmoid + softmax).

    Comparte la misma lógica entre el Parameter Server (evaluación)
    y los Workers (cálculo de gradientes).

    :param params: Diccionario con W1, b1, W2, b2.
    :type params: Dict[str, np.ndarray]

    :param X: Batch de imágenes, forma ``(N, input_size)``.
    :type X: np.ndarray

    :return: ``(A1, A2)`` — activaciones de capa oculta ``(hidden, N)``
             y probabilidades de salida ``(output, N)``.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    Z1 = W1 @ X.T + b1[:, np.newaxis]  # (hidden, N)
    A1 = sigmoid(Z1)  # (hidden, N)
    Z2 = W2 @ A1 + b2[:, np.newaxis]  # (output, N)
    A2 = softmax(Z2)  # (output, N)

    return A1, A2


# ================================================================
# PÉRDIDA
# ================================================================


def cross_entropy_loss(A2: np.ndarray, Y: np.ndarray) -> float:
    """
    Pérdida de entropía cruzada, promediada por muestra.

    :param A2: Probabilidades de salida, forma ``(output, N)``.
    :type A2: np.ndarray

    :param Y: Etiquetas enteras, forma ``(N,)``.
    :type Y: np.ndarray

    :return: Pérdida media sobre el batch.
    :rtype: float
    """
    n = A2.shape[1]
    log_probs = np.log(np.clip(A2, 1e-15, 1.0))
    return -float(np.sum(log_probs[Y, np.arange(n)])) / n
