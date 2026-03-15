"""
Model/mlp.py

Clasificador MLP en NumPy puro para el Algoritmo de Diego distribuido.

──────────────────────────────────────────────────────────────────
ROL EN LA ARQUITECTURA
──────────────────────────────────────────────────────────────────
Este módulo implementa el clasificador que opera sobre los vectores
de features producidos por el extractor CNN (Model/cnn_extractor.py).

    features (N, feature_dim) → MLP → logits (N, 10) → softmax → clases

Es el único componente cuyos pesos viajan por la red: el PS lo
inicializa, los Workers calculan sus gradientes sobre sus chunks
de features, y el PS los promedia y actualiza.

──────────────────────────────────────────────────────────────────
ARQUITECTURA
──────────────────────────────────────────────────────────────────
MLP de DOS capas ocultas con ReLU:

    Entrada (feature_dim)
        └─► Oculta 1 (hidden1, ReLU)
                └─► Oculta 2 (hidden2, ReLU)
                        └─► Salida (10, Softmax)

Dos capas ocultas en lugar de una porque los features de una CNN
tienen estructura compleja; una sola capa lineal no es suficiente
para aprender fronteras de decisión no lineales sobre ellos.

──────────────────────────────────────────────────────────────────
POR QUÉ NUMPY Y NO PYTORCH AQUÍ
──────────────────────────────────────────────────────────────────
  • Los gradientes se serializan con Pickle y viajan por TCP.
    Los tensores PyTorch serializados con Pickle incluyen el grafo
    de cómputo y pesan mucho más que arrays NumPy equivalentes.

  • La implementación manual del forward/backward es parte del
    objetivo pedagógico: se entiende exactamente qué se envía,
    qué se promedia, y por qué funciona el Algoritmo de Diego.

  • NumPy float32 es suficientemente eficiente para un MLP de pocas
    capas sobre features ya extraídos.

──────────────────────────────────────────────────────────────────
FUNCIONES PÚBLICAS
──────────────────────────────────────────────────────────────────
    init_params(feature_dim, hidden1, hidden2, n_classes, seed)
        → Dict con W1, b1, W2, b2, W3, b3

    forward_and_gradients(params, X, Y)
        → (gradients, mean_loss, accuracy_pct)
        Usado por los Workers en cada época.

    evaluate(params, X, Y)
        → (accuracy_pct, mean_loss)
        Usado por el PS para evaluar en datos de prueba.

    apply_gradients(params, gradients, learning_rate)
        → None (modifica params in-place)
        Usado por el PS tras promediar los gradientes.
"""

from typing import Dict, Tuple

import numpy as np


# ================================================================
# ACTIVACIONES
# ================================================================


def _relu(Z: np.ndarray) -> np.ndarray:
    """
    ReLU: f(x) = max(0, x).

    Ventaja sobre sigmoide en capas ocultas profundas: su gradiente
    es 1 para x > 0 (no se satura), lo que evita el problema del
    gradiente desvaneciente en capas sucesivas.

    :param Z: Pre-activaciones de cualquier forma.
    :return: Mismo shape, valores negativos → 0.
    """
    return np.maximum(0.0, Z)


def _relu_grad(Z: np.ndarray) -> np.ndarray:
    """
    Derivada de ReLU respecto a la pre-activación Z: 1 si Z>0, 0 si Z≤0.

    Se calcula sobre Z (no sobre la activación A) para evitar recalcular
    la pre-activación en el backward. El caller guarda Z del forward.

    :param Z: Pre-activaciones.
    :return: Array de 0s y 1s, mismo shape.
    """
    return (Z > 0).astype(Z.dtype)


def _softmax(Z: np.ndarray) -> np.ndarray:
    """
    Softmax numéricamente estable por columna: Z tiene forma (C, N).

    Restar el máximo por columna antes de exp() evita overflow.
    El resultado es matemáticamente idéntico al softmax sin la resta.

    :param Z: Logits, forma (n_classes, N).
    :return: Probabilidades, misma forma.
    """
    Z_shift = Z - Z.max(axis=0, keepdims=True)
    E = np.exp(Z_shift)
    return E / E.sum(axis=0, keepdims=True)


# ================================================================
# INICIALIZACIÓN
# ================================================================


def init_params(
    feature_dim: int,
    hidden1: int,
    hidden2: int,
    n_classes: int,
    seed: int | None = None,
) -> Dict[str, np.ndarray]:
    """
    Inicializa los pesos del MLP con He initialization.

    He initialization: W ~ N(0, √(2/fan_in))

    Es la elección óptima para capas con activación ReLU porque
    compensa que la mitad de las neuronas tienen activación 0
    en promedio, manteniendo la varianza de la señal estable entre
    capas. Xavier (usada con sigmoide) subestimaría la varianza para ReLU.

    Todos los pesos son float32: ocupa la mitad de memoria que
    float64 sin pérdida apreciable para este rango de arquitecturas,
    y es el tipo nativo de PyTorch — consistente con los features de la CNN.

    :param feature_dim: Dimensión del vector de entrada (= FEATURE_DIM de la CNN).
    :param hidden1: Neuronas en la primera capa oculta.
    :param hidden2: Neuronas en la segunda capa oculta.
    :param n_classes: Número de clases de salida (10 para CIFAR-10).
    :param seed: Semilla aleatoria para reproducibilidad.

    :return: Dict con claves W1, b1, W2, b2, W3, b3, todos float32.
    """
    rng = np.random.RandomState(seed)

    def _he(fan_in: int, fan_out: int) -> np.ndarray:
        std = np.sqrt(2.0 / fan_in)
        return rng.normal(0.0, std, (fan_out, fan_in)).astype(np.float32)

    def _zeros(n: int) -> np.ndarray:
        return np.zeros(n, dtype=np.float32)

    return {
        "W1": _he(feature_dim, hidden1),  # (hidden1, feature_dim)
        "b1": _zeros(hidden1),  # (hidden1,)
        "W2": _he(hidden1, hidden2),  # (hidden2, hidden1)
        "b2": _zeros(hidden2),  # (hidden2,)
        "W3": _he(hidden2, n_classes),  # (n_classes, hidden2)
        "b3": _zeros(n_classes),  # (n_classes,)
    }


# ================================================================
# FORWARD (compartido entre forward_and_gradients y evaluate)
# ================================================================


def _forward(
    params: Dict[str, np.ndarray],
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward pass vectorizado. Devuelve las pre-activaciones y activaciones
    de cada capa, necesarias para el backward.

    :param params: W1, b1, W2, b2, W3, b3.
    :param X: Features de entrada, forma (N, feature_dim).

    :return: (Z1, A1, Z2, A2, A3) donde:
             Z1 (hidden1, N) — pre-activaciones capa 1
             A1 (hidden1, N) — activaciones capa 1 (ReLU)
             Z2 (hidden2, N) — pre-activaciones capa 2
             A2 (hidden2, N) — activaciones capa 2 (ReLU)
             A3 (n_classes, N) — probabilidades softmax salida
    """
    X_T = X.T  # (feature_dim, N) — orientación columna para @ eficiente

    Z1 = params["W1"] @ X_T + params["b1"][:, np.newaxis]  # (hidden1, N)
    A1 = _relu(Z1)

    Z2 = params["W2"] @ A1 + params["b2"][:, np.newaxis]  # (hidden2, N)
    A2 = _relu(Z2)

    Z3 = params["W3"] @ A2 + params["b3"][:, np.newaxis]  # (n_classes, N)
    A3 = _softmax(Z3)  # (n_classes, N)

    return Z1, A1, Z2, A2, A3


# ================================================================
# FORWARD + BACKWARD — para los Workers
# ================================================================


def forward_and_gradients(
    params: Dict[str, np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], float, float]:
    """
    Forward pass + backward pass sobre un batch de features.

    No modifica ``params``. Los gradientes se devuelven como Dict
    para que el PS los serialice con Pickle y los promedíe.

    La retropropagación sigue la regla de la cadena capa por capa:

        δ3 = A3 − Y_onehot          ← gradiente de cross-entropy + softmax
        δ2 = (W3.T @ δ3) ⊙ relu'(Z2)
        δ1 = (W2.T @ δ2) ⊙ relu'(Z1)

    Los gradientes de los pesos son el promedio sobre el batch (1/N),
    lo que hace que los gradientes de distintos Workers sean directamente
    comparables aunque tengan batch sizes distintos.

    :param params: Pesos actuales del MLP.
    :param X: Features del batch, forma (N, feature_dim). float32.
    :param Y: Etiquetas del batch, forma (N,). int32.

    :return: (gradients, mean_loss, accuracy_pct)
             gradients: Dict con dW1, db1, dW2, db2, dW3, db3.
    """
    N = len(X)
    W2, W3 = params["W2"], params["W3"]

    # ── Forward ──────────────────────────────────────────────────
    Z1, A1, Z2, A2, A3 = _forward(params, X)

    # ── Métricas ──────────────────────────────────────────────────
    preds = np.argmax(A3, axis=0)
    correct = int(np.sum(preds == Y))

    # Cross-entropy: −Σ log(p_correcta) / N
    log_p = np.log(np.clip(A3, 1e-15, 1.0))
    total_loss = -float(np.sum(log_p[Y, np.arange(N)]))

    # ── Backward ─────────────────────────────────────────────────
    # Capa salida — softmax + cross-entropy se combinan en un gradiente limpio
    Y_onehot = np.zeros_like(A3)  # (n_classes, N)
    Y_onehot[Y, np.arange(N)] = 1.0

    delta3 = A3 - Y_onehot  # (n_classes, N)
    dW3 = (1.0 / N) * (delta3 @ A2.T)  # (n_classes, hidden2)
    db3 = (1.0 / N) * delta3.sum(axis=1)  # (n_classes,)

    # Capa 2 — ReLU
    delta2 = (W3.T @ delta3) * _relu_grad(Z2)  # (hidden2, N)
    dW2 = (1.0 / N) * (delta2 @ A1.T)  # (hidden2, hidden1)
    db2 = (1.0 / N) * delta2.sum(axis=1)  # (hidden2,)

    # Capa 1 — ReLU
    delta1 = (W2.T @ delta2) * _relu_grad(Z1)  # (hidden1, N)
    dW1 = (1.0 / N) * (delta1 @ X)  # (hidden1, feature_dim)
    db1 = (1.0 / N) * delta1.sum(axis=1)  # (hidden1,)

    gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
        "dW3": dW3,
        "db3": db3,
    }

    return gradients, total_loss / N, 100.0 * correct / N


# ================================================================
# SOLO EVALUACIÓN — para el PS
# ================================================================


def evaluate(
    params: Dict[str, np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
) -> Tuple[float, float]:
    """
    Evalúa el MLP sobre un conjunto completo sin calcular gradientes.

    El PS llama esta función después de actualizar los pesos para
    obtener las métricas de la época sobre el conjunto de prueba.

    :param params: Pesos del MLP.
    :param X: Features de prueba, forma (N, feature_dim).
    :param Y: Etiquetas de prueba, forma (N,).

    :return: (accuracy_pct, mean_loss)
    """
    N = len(X)
    _, _, _, _, A3 = _forward(params, X)

    preds = np.argmax(A3, axis=0)
    accuracy = 100.0 * float(np.sum(preds == Y)) / N

    log_p = np.log(np.clip(A3, 1e-15, 1.0))
    loss = -float(np.sum(log_p[Y, np.arange(N)])) / N

    return accuracy, loss


# ================================================================
# APLICAR GRADIENTES — para el PS
# ================================================================


def apply_gradients(
    params: Dict[str, np.ndarray],
    gradients: Dict[str, np.ndarray],
    learning_rate: float,
) -> None:
    """
    Actualiza los pesos del MLP in-place: θ ← θ − lr × ∇θ.

    El PS llama esta función después de promediar los gradientes
    de todos los Workers. Centralizar la actualización en Model/mlp.py
    mantiene el PS agnóstico al número de capas.

    :param params: Pesos del MLP a actualizar.
    :param gradients: Gradientes promediados (dW1, db1, …, dW3, db3).
    :param learning_rate: Tasa de aprendizaje.
    """
    params["W1"] -= learning_rate * gradients["dW1"]
    params["b1"] -= learning_rate * gradients["db1"]
    params["W2"] -= learning_rate * gradients["dW2"]
    params["b2"] -= learning_rate * gradients["db2"]
    params["W3"] -= learning_rate * gradients["dW3"]
    params["b3"] -= learning_rate * gradients["db3"]
