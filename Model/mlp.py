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
    :type Z: np.ndarray de cualquier shape, dtype float32 o float64.
    :return: Array con misma forma, valores negativos convertidos a 0.
    :rtype: np.ndarray de mismo shape y dtype que Z.
    """
    return np.maximum(0.0, Z)


def _relu_grad(Z: np.ndarray) -> np.ndarray:
    """
    Derivada de ReLU respecto a la pre-activación Z: 1 si Z>0, 0 si Z≤0.

    Se calcula sobre Z (no sobre la activación A) para evitar recalcular
    la pre-activación en el backward. El caller guarda Z del forward.

    :param Z: Pre-activaciones de salida de una capa lineal.
    :type Z: np.ndarray de cualquier shape, dtype float32 o float64.
    :return: Array de 0s y 1s indicando dónde ReLU transmite gradiente.
    :rtype: np.ndarray de mismo shape, dtype bool convertido a float.
    """
    return (Z > 0).astype(Z.dtype)


def _softmax(Z: np.ndarray) -> np.ndarray:
    """
    Softmax numéricamente estable por columna: Z tiene forma (C, N).

    Restar el máximo por columna antes de exp() evita overflow.
    El resultado es matemáticamente idéntico al softmax sin la resta.

    :param Z: Logits o pre-activaciones.
    :type Z: np.ndarray de shape (n_classes, N) float32 o float64.
    :return: Probabilidades normalizadas (suma a 1 por columna).
    :rtype: np.ndarray de mismo shape, dtype float32 o float64.
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
    :type feature_dim: int, típicamente 512.

    :param hidden1: Neuronas en la primera capa oculta.
    :type hidden1: int, típicamente 256 o 512.

    :param hidden2: Neuronas en la segunda capa oculta.
    :type hidden2: int, típicamente 128 o 256.

    :param n_classes: Número de clases de salida (10 para CIFAR-10).
    :type n_classes: int.

    :param seed: Semilla aleatoria para reproducibilidad.
    :type seed: int | None, default=None.

    :return: Diccionario con 6 claves: W1, b1, W2, b2, W3, b3 (todos float32).
    :rtype: Dict[str, np.ndarray].
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

    OPERACIONES VECTORIZADAS EN NUMPY:
    -----------------------------------
    • X tiene forma (N, feature_dim) — N ejemplos en filas.
    • X.T transpone a (feature_dim, N) — N ejemplos en columnas.
    • Multiplicación matricial W @ X.T con W shape (output, input):
      - Resultado: (output, N) — cada columna es la salida de un ejemplo.
    • Broadcasting con b[:, np.newaxis] expande el vector (output,) a
      (output, 1) para que NumPy lo replique sobre las N columnas.

    Flujo de dimensiones:
        (N, feature_dim) → _forward → (n_classes, N)

    :param params: Diccionario con pesos: W1, b1, W2, b2, W3, b3.
    :type params: Dict[str, np.ndarray].

    :param X: Features de entrada.
    :type X: np.ndarray de shape (N, feature_dim) float32.

    :return: Tupla (Z1, A1, Z2, A2, A3) donde cada componente es:
                - Z1: pre-activaciones capa 1, shape (hidden1, N).
                - A1: activaciones capa 1 (ReLU), shape (hidden1, N).
                - Z2: pre-activaciones capa 2, shape (hidden2, N).
                - A2: activaciones capa 2 (ReLU), shape (hidden2, N).
                - A3: probabilidades softmax salida, shape (n_classes, N).
                        Están normalizadas (suma a 1 por columna).
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray].
    """
    # Transpone X a (feature_dim, N) para que la multiplicación matricial
    # sea eficiente: (W @ X.T) procesa los N ejemplos en paralelo.
    X_T = X.T  # (feature_dim, N)

    # Capa 1: proyección lineal seguida de ReLU
    # W1 @ X.T: (hidden1, feature_dim) @ (feature_dim, N) = (hidden1, N)
    # b1[:, np.newaxis]: expande (hidden1,) a (hidden1, 1) que se replica
    # sobre las N columnas para que cada ejemplo reciba el mismo sesgo.
    Z1 = params["W1"] @ X_T + params["b1"][:, np.newaxis]
    A1 = _relu(Z1)

    # Capa 2: igual que capa 1
    # W2 @ A1: (hidden2, hidden1) @ (hidden1, N) = (hidden2, N)
    Z2 = params["W2"] @ A1 + params["b2"][:, np.newaxis]
    A2 = _relu(Z2)

    # Capa 3: igual patrón pero softmax en lugar de ReLU
    # W3 @ A2: (n_classes, hidden2) @ (hidden2, N) = (n_classes, N)
    Z3 = params["W3"] @ A2 + params["b3"][:, np.newaxis]
    A3 = _softmax(Z3)  # (n_classes, N) con valores en [0,1], suma=1 por columna

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
    Forward pass + backward pass completo sobre un batch de features.

    DIAGRAMA DEL FLUJO:
    -------------------
    X (N, feature_dim)
      │
      ├─ forward(params, X)    →  Z1,A1,Z2,A2,A3
      │
      ├─ predicciones: argmax(A3) = [0..9]  (un índice por ejemplo)
      │
      ├─ loss: -Σ log(p_etiqueta_correcta)  (cross-entropy)
      │
      ├─ backward: regla de cadena inversa
      │   δ3 = A3 - one_hot(Y)
      │   δ2 = W3.T @ δ3 ⊙ relu'(Z2)
      │   δ1 = W2.T @ δ2 ⊙ relu'(Z1)
      │
      └─ gradientes de pesos normalizados por N

    No modifica ``params``. Los gradientes se devuelven como Dict para
    que el Parameter Server los serialice con Pickle y los promedíe.

    :param params: Pesos actuales del MLP.
    :type params: Dict[str, np.ndarray] con W1, b1, W2, b2, W3, b3.

    :param X: Features del batch (típicamente 50-500 ejemplos).
    :type X: np.ndarray de shape (N, feature_dim) float32.

    :param Y: Etiquetas del batch como enteros 0-9.
    :type Y: np.ndarray de shape (N,) int32 con valores 0-9.

    :return: Tupla (gradients, mean_loss, accuracy_pct) donde:
                - gradients: Dict con 6 claves dW1, db1, dW2, db2, dW3, db3.
                  Los gradientes se normalizan por N (batch size) para que
                  sean comparables independientemente del tamaño del batch.
                - mean_loss: Cross-entropy loss promediada (float).
                - accuracy_pct: Porcentaje de predicciones correctas 0-100 (float).
    :rtype: Tuple[Dict[str, np.ndarray], float, float].
    """
    N = len(X)
    W2, W3 = params["W2"], params["W3"]

    # ── Forward: todas las activaciones se guardan para backward ──
    Z1, A1, Z2, A2, A3 = _forward(params, X)

    # ── MÉTRICAS: accuracy y loss ─────────────────────────────────
    # argmax(A3, axis=0): para cada ejemplo (columna), obtén la clase
    # con máxima probabilidad. Resultado: (N,) con índices 0-9.
    preds = np.argmax(A3, axis=0)
    correct = int(np.sum(preds == Y))  # contar predicciones correctas

    # Cross-entropy loss: -log(p_correcta) promediado
    # log(A3) da log de probabilidades de todas las clases.
    # A3[Y, np.arange(N)] extrae la probabilidad de la clase correcta
    # por cada ejemplo usando advanced indexing: Y[i] es la fila (clase),
    # np.arange(N)[i]=i es la columna (ejemplo).
    log_p = np.log(np.clip(A3, 1e-15, 1.0))  # clip evita log(0)
    total_loss = -float(np.sum(log_p[Y, np.arange(N)]))
    mean_loss = total_loss / N
    accuracy = 100.0 * correct / N

    # ─── BACKWARD: calcula gradientes usando regla de cadena ───
    # EXPLICACIÓN: La retropropagación (backpropagation) calcula cómo afecta
    # cada parámetro al loss final. Usamos la regla de la cadena (chain rule)
    # capa por capa, desde la salida hacia la entrada.
    #
    # Capa salida — softmax + cross-entropy combinadas dan gradiente limpio
    # Con probabilidades softmax P = A3 y etiquetas one-hot E (donde E[i,j]=1
    # si j es la clase correcta), el gradiente es simplemente: dL/dZ3 = P - E
    # Optimización: en lugar de crear matriz E completa (n_classes, N),
    # copiamos A3 y restamos 1 solo en posiciones de etiquetas correctas.
    # Índexación NumPy: Y[i] es la clase de ejemplo i; np.arange(N) es [0..N-1]
    # delta3[Y, np.arange(N)] selecciona la diagonal Y[i] en ejemplo i.
    delta3 = A3.copy()  # Copia para no modificar A3 original (n_classes, N)
    delta3[Y, np.arange(N)] -= 1.0  # Resta 1 en clases correctas
    # Gradiente respecto a W3: (dL/dZ3) @ A2.T con promediado 1/N
    # Dimensiones: (n_classes,N) @ (N,hidden2) = (n_classes,hidden2) ✓
    dW3 = (1.0 / N) * (delta3 @ A2.T)
    # Gradiente respecto a b3: suma de gradientes por ejemplo, promediado
    db3 = (1.0 / N) * delta3.sum(axis=1)

    # Capa 2 — ReLU: gradiente fluye a través de la derivada ReLU
    # Multiplicación element-wise (⊙) del gradiente con la máscara ReLU
    # ReLU transmite gradientes donde Z2>0, anula donde Z2≤0
    delta2 = (W3.T @ delta3) * _relu_grad(Z2)  # (hidden2, hidden1) x (hidden2, N)
    dW2 = (1.0 / N) * (delta2 @ A1.T)  # (hidden2, hidden1)
    db2 = (1.0 / N) * delta2.sum(axis=1)  # (hidden2,)

    # Capa 1 — ReLU: mismo patrón que capa 2
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
# BACKWARD A TRAVÉS DE MLP — para modo End-to-End
# ================================================================


def mlp_backward_to_input(
    params: Dict[str, np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], float, float]:
    """
    Realiza forward + backward completo sobre el MLP y retorna:
    1. Gradiente respecto a la entrada (X) — necesario para backprop CNN
    2. Gradientes respecto a los pesos del MLP (como forward_and_gradients)
    3. Loss y accuracy

    Usado en modo End-to-End donde se necesita backpropagar a través de la CNN
    después de calcular el gradiente de loss respecto a los features.

    :param params: Pesos del MLP.
    :type params: Dict[str, np.ndarray]

    :param X: Features de entrada.
    :type X: np.ndarray de shape (N, feature_dim) float32.

    :param Y: Etiquetas.
    :type Y: np.ndarray de shape (N,) int32.

    :return: Tupla (dX, gradients, mean_loss, accuracy_pct) donde:
                - dX: gradiente respecto a X, shape (N, feature_dim).
                  Se usa para backpropagar a través de la CNN.
                - gradients: Dict con dW1, db1, ..., db3 (mismo que forward_and_gradients).
                - mean_loss: Cross-entropy loss promediada.
                - accuracy_pct: Porcentaje de aciertos 0-100.
    :rtype: Tuple[np.ndarray, Dict[str, np.ndarray], float, float].
    """
    N = len(X)
    W1, W2, W3 = params["W1"], params["W2"], params["W3"]

    # Forward
    Z1, A1, Z2, A2, A3 = _forward(params, X)

    # Métricas
    preds = np.argmax(A3, axis=0)
    correct = int(np.sum(preds == Y))

    log_p = np.log(np.clip(A3, 1e-15, 1.0))
    total_loss = -float(np.sum(log_p[Y, np.arange(N)]))

    # Backward — igual que en forward_and_gradients
    delta3 = A3.copy()
    delta3[Y, np.arange(N)] -= 1.0
    dW3 = (1.0 / N) * (delta3 @ A2.T)
    db3 = (1.0 / N) * delta3.sum(axis=1)

    delta2 = (W3.T @ delta3) * _relu_grad(Z2)
    dW2 = (1.0 / N) * (delta2 @ A1.T)
    db2 = (1.0 / N) * delta2.sum(axis=1)

    delta1 = (W2.T @ delta2) * _relu_grad(Z1)
    dW1 = (1.0 / N) * (delta1 @ X)
    db1 = (1.0 / N) * delta1.sum(axis=1)

    # MLP gradients respecto a su entrada X
    # Necesario para backpropagar a la CNN en modo E2E
    dX = W1.T @ delta1  # (feature_dim, hidden1) @ (hidden1, N) = (feature_dim, N)
    dX = dX.T  # Transponer a (N, feature_dim)

    gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
        "dW3": dW3,
        "db3": db3,
    }

    return dX, gradients, total_loss / N, 100.0 * correct / N


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
    Realiza un forward pass limpio sin retención de gradientes.

    :param params: Pesos del MLP.
    :type params: Dict[str, np.ndarray] con W1, b1, W2, b2, W3, b3.

    :param X: Features de prueba.
    :type X: np.ndarray de shape (N, feature_dim) float32.

    :param Y: Etiquetas de prueba.
    :type Y: np.ndarray de shape (N,) int32 con valores 0-9.

    :return: Tupla (accuracy_pct, mean_loss) donde:
                - accuracy_pct: Porcentaje de aciertos 0-100 (float).
                - mean_loss: Cross-entropy loss promediada (float).
    :rtype: Tuple[float, float].
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
    momentum: float = 0.0,
    velocities: Dict[str, np.ndarray] | None = None,
) -> None:
    """
    Actualiza los pesos del MLP in-place con SGD (+ momentum opcional).

    SGD puro (momentum=0):
        θ ← θ − lr × ∇θ

    SGD con momentum (momentum > 0):
        v ← μ × v + ∇θ
        θ ← θ − lr × v

    El momentum acumula gradientes de épocas anteriores, dando al
    optimizador "inercia" en la dirección correcta y amortiguando
    las oscilaciones. Valores típicos: 0.9 – 0.99.

    Cuando se usa momentum, el PS mantiene el dict ``velocities``
    entre épocas y lo pasa en cada llamada. Si ``velocities`` es None
    se inicializa a ceros automáticamente la primera vez.

    :param params: Pesos del MLP a actualizar in-place.
    :type params: Dict[str, np.ndarray] con W1, b1, W2, b2, W3, b3.

    :param gradients: Gradientes promediados por el PS.
    :type gradients: Dict[str, np.ndarray] con dW1, db1, dW2, db2, dW3, db3.

    :param learning_rate: Tasa de aprendizaje multiplicada por los gradientes.
    :type learning_rate: float, típicamente 1e-3 a 1e-2.

    :param momentum: Coeficiente de momentum (0.0 = SGD puro, 0.9-0.99 = típico).
    :type momentum: float, default=0.0.

    :param velocities: Diccionario mutable con velocidades acumuladas.
                       Debe persistir entre llamadas. Ignorado si momentum=0.
                       Inicialización: {'W1': arr, ..., 'b3': arr}.
    :type velocities: Dict[str, np.ndarray] | None, default=None.
    :return: None (modifica params y velocities in-place).
    :rtype: NoneType.
    """
    keys = ["W1", "b1", "W2", "b2", "W3", "b3"]
    grad_keys = ["dW1", "db1", "dW2", "db2", "dW3", "db3"]

    if momentum == 0.0:
        # SGD puro — camino rápido sin estado adicional
        # Actualización: θ ← θ − lr × ∇θ
        # Cada parámetro se decrementa por su gradiente multiplicado por lr.
        for k, dk in zip(keys, grad_keys):
            params[k] -= learning_rate * gradients[dk]
        return

    if velocities is None:
        raise ValueError("velocities no puede ser None cuando momentum > 0.")

    # SGD con momentum — inicializa claves ausentes fuera del loop de actualización
    # Esto evita crear ceros bajo la llave equivocada si hay typos en los nombres.
    for k in keys:
        if k not in velocities:
            velocities[k] = np.zeros_like(params[k])

    # Actualización con momentum in-place: evita alocar arrays temporales
    # Fórmula (vectorizada en NumPy):
    #   v ← μ × v + ∇θ         (velocidad acumulada)
    #   θ ← θ − lr × v         (actualización)
    # Esto hace que la velocidad "acumule" en la dirección correcta (momentum).
    for k, dk in zip(keys, grad_keys):
        # In-place multiplication: v *= momentum (más rápido que v = v * momentum)
        velocities[k] *= momentum
        # Suma el gradiente actual a la velocidad acumulada
        velocities[k] += gradients[dk]
        # Decrementa el parámetro por la velocidad (no el gradiente directo)
        params[k] -= learning_rate * velocities[k]
