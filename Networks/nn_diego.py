"""
Networks/nn_diego.py
Red neuronal para clasificación de MNIST con entrenamiento federado.
Arquitectura: 784 (entrada) → oculta (sigmoide) → 10 (salida, softmax).
Implementación nativa con NumPy (operaciones matriciales vectorizadas).
"""

import numpy as np
from typing import Any, Callable, Dict, List, Tuple

# Importa utilidades matemáticas
from Utils.math_utils import (
    average_network_parameters,
    sigmoid,
    sigmoid_derivative_from_activation,
    softmax,
    xavier_initialization,
    vector_zeros,
)

# =============
# RED NEURONAL
# =============


class DiegoNeuronalNetwork:
    """
    Red neuronal con soporte para el algoritmo de entrenamiento de Diego.

    Todos los parámetros internos son np.ndarray.

    Attributes:
        input_size:  Número de neuronas de entrada (784 para MNIST).
        hidden_size: Número de neuronas en la capa oculta.
        output_size: Número de neuronas de salida (10 para MNIST).
        W1:          Pesos entrada → capa oculta  (hidden_size × input_size).
        b1:          Sesgos de la capa oculta     (hidden_size,).
        W2:          Pesos capa oculta → salida   (output_size × hidden_size).
        b2:          Sesgos de la capa de salida  (output_size,).
        training_history: Historial del último entrenamiento federado.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 30,
        output_size: int = 10,
        random_seed: int | None = None,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Establece la semilla si se proporciona
        if random_seed is not None:
            np.random.seed(random_seed)

        # Inicialización de parámetros
        self.W1 = xavier_initialization(input_size, hidden_size)
        self.b1 = vector_zeros(hidden_size)

        self.W2 = xavier_initialization(hidden_size, output_size)
        self.b2 = vector_zeros(output_size)

        # Historial de entrenamiento
        self.training_history: Dict[str, Any] = {}

    # ========================
    # GESTIÓN DE PARÁMETROS
    # ========================

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Devuelve una copia de los parámetros actuales.

        :return: Diccionario con W1, b1, W2, b2
        :rtype: Dict[str, np.ndarray]
        """
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """
        Establece los parámetros a partir de un diccionario.

        :param params: Diccionario con claves W1, b1, W2, b2
        :type params: Dict[str, np.ndarray]

        :return: None
        :rtype: None
        """
        self.W1 = params["W1"].copy()
        self.b1 = params["b1"].copy()
        self.W2 = params["W2"].copy()
        self.b2 = params["b2"].copy()

    # ========================
    # FORWARD PROPAGATION
    # ========================

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward para una sola entrada (vector 1-D).

        Recibe una sola imagen y retorna su predicción.
        Delega el cálculo a ``_forward_batch`` enviando el vector
        como batch de tamaño 1 y extrae los vectores resultantes.

        :param x: Vector de entrada (input_size,)
        :type x: np.ndarray

        :return: (probabilidades softmax, cache intermedio)
        :rtype: Tuple[np.ndarray, Dict[str, Any]]
        """
        # x.reshape(1, -1) convierte el vector (784,) a una matriz (1, 784)
        # Esto se hace ya que la red trabaja con batches (muchas imágenes a la vez)
        a1, output = self._forward_batch(x.reshape(1, -1))

        # a1 y output tienen forma (hidden, 1) y (output, 1); así que se aplanan a 1-D
        a1 = a1[:, 0]
        output = output[:, 0]

        # Guarda: imagen original, activaciones ocultas, salida final
        cache = {"x": x.copy(), "a1": a1, "output": output}
        return output, cache

    def _forward_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward vectorizado sobre un batch de N ejemplos.

        Centraliza la lógica de propagación hacia adelante que comparten
        ``train_on_batch`` y ``evaluate``. Ambos métodos la invocan y
        consumen directamente sus salidas, sin recalcular nada.

        :param X: Imágenes de forma ``(N, input_size)``
        :type X: np.ndarray

        :return: Tupla ``(A1, A2)`` donde:\n
                 - ``A1``: activaciones ocultas, forma ``(hidden_size, N)``
                 - ``A2``: probabilidades softmax, forma ``(output_size, N)``
        :rtype: Tuple[np.ndarray, np.ndarray]
        """

        # Si A tiene forma (m, n) y B tiene forma (p, q), la multiplicación es posible solo si n = p.
        # self.W1 es (hidden_size, 784) y X tiene forma (N, 784). Aquí: 784 (columnas de W1) ≠ N (filas de X)
        # Para poder multiplicarlos, X debe (784, N). Así que se le aplica la traspuesta
        X_T = X.T  # (input, N)

        # Multiplica los pesos por cada entrada: self.W1 @ X_T -> (hidden_size, N),
        # self.b1 tiene forma (hidden_size,), una dimensión por debajo de self.W1 @ X_T
        # np.newaxis agrega una dimensión extra, convirtiendo (hidden_size,) en (hidden_size, 1)
        # Capa oculta: z1 = W1·x + b1
        Z1 = self.W1 @ X_T + self.b1[:, np.newaxis]  # (hidden, N)

        # Activación sigmoide: a1 = σ(z1)
        A1 = sigmoid(Z1)  # (hidden, N)

        # Capa de salida: z2 = W2·a1 + b2
        Z2 = self.W2 @ A1 + self.b2[:, np.newaxis]  # (output, N)

        # Aplica softmax adaptado a una matriz 2-D
        A2 = softmax(Z2)  # (output, N)
        return A1, A2

    # ========================
    # ENTRENAMIENTO
    # ========================

    def train_on_batch(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        learning_rate: float,
    ) -> Tuple[float, float]:
        """
        Entrena sobre un batch completo usando gradiente descendente vectorizado.

        A diferencia de ``forward()``, este método entrena con MUCHAS imágenes a la vez.

        En lugar de iterar ejemplo por ejemplo, procesa todo el batch con
        operaciones matriciales de NumPy (mucho más rápido).

        :param X: Imágenes (N, 784)
        :type X: np.ndarray

        :param Y: Etiquetas (N,)
        :type Y: np.ndarray

        :param learning_rate: Tasa de aprendizaje
        :type learning_rate: float

        :param verbose: Si True muestra información adicional
        :type verbose: bool

        :return: (loss promedio, accuracy %)
        :rtype: Tuple[float, float]
        """
        # Número de muestras
        n = len(X)

        # Forward vectorizado: delega en _forward_batch para no duplicar código
        # A1: activaciones ocultas (hidden_size, N), necesarias para el backward
        # A2: probabilidades softmax (output_size, N)
        A1, A2 = self._forward_batch(X)

        # X_T se necesita en el backward (dW1 = delta1 @ X_T.T)
        X_T = X.T

        # Escoge la clase más probable
        predictions = np.argmax(A2, axis=0)  # (N,)

        # Cuenta cuántas predicciones fueron correctas
        correct = np.sum(predictions == Y)

        # Calcula el error usando "cross-entropy"
        log_probs = np.log(np.clip(A2, 1e-15, 1.0))

        # log_probs[Y, np.arange(n)] significa:
        # Para cada muestra, toma el logaritmo de la probabilidad de su clase correcta
        total_loss = -np.sum(log_probs[Y, np.arange(n)])

        # Backward vectorizado
        # One-hot de todas las etiquetas: (output, N)
        Y_onehot = np.zeros((self.output_size, n))
        Y_onehot[Y, np.arange(n)] = 1.0

        # δ2 = A2 - Y_onehot, forma (output, N)
        delta2 = A2 - Y_onehot

        # Gradientes capa 2
        # Cuando entrenas con varias imágenes a la vez, el gradiente que calculas es la suma de los errores de todas las imágenes
        # Pero en aprendizaje automático normalmente usamos el promedio del error, no la suma
        # Si no divides por n, el tamaño del paso dependería del tamaño del batch
        # Dividir por n hace que el aprendizaje sea estable e independiente del tamaño del batch (es simplemente calcular el promedio)
        dW2 = (1.0 / n) * (delta2 @ A1.T)  # (output, hidden)

        # np.sum(delta2, axis=1) hace que para cada fila (cada clase), suma los errores de todas las imágenes
        # Básicamente, suma de los errores de cada neurona en todas las muestras
        db2 = (1.0 / n) * np.sum(delta2, axis=1)  # (output,)

        # δ1 = (W2^T · δ2) ⊙ σ'(A1), forma (hidden, N)
        delta1 = (self.W2.T @ delta2) * sigmoid_derivative_from_activation(A1)

        # Gradientes capa 1
        dW1 = (1.0 / n) * (delta1 @ X_T.T)  # (hidden, input)
        db1 = (1.0 / n) * np.sum(delta1, axis=1)  # (hidden,)

        # Actualiza por parámetros
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        return total_loss / n, 100.0 * correct / n

    def train_federated(
        self,
        partitions: List[Tuple[np.ndarray, np.ndarray]],
        epochs: int,
        learning_rate: float,
        verbose: bool = True,
        on_epoch_end: Callable | None = None,
    ) -> Dict[str, Any]:
        """
        Entrena usando el algoritmo de Diego (entrenamiento federado).

        Por cada época:
        1. Guarda parámetros globales.
        2. Entrena cada partición independientemente.
        3. Promedia parámetros de todas las particiones.

        :param partitions: Lista de particiones (X_part, Y_part)
        :type partitions: List[Tuple[np.ndarray, np.ndarray]]

        :param epochs: Número de épocas
        :type epochs: int

        :param learning_rate: Tasa de aprendizaje
        :type learning_rate: float

        :param verbose: Si True muestra progreso
        :type verbose: bool

        :param on_epoch_end: Callback opcional al final de cada época
        :type on_epoch_end: Callable | None

        :return: Historial con accuracies, losses y métricas por partición
        :rtype: Dict[str, Any]
        """
        num_partitions = len(partitions)
        accuracies: List[float] = []
        losses: List[float] = []
        partition_accuracies: List[List[float]] = []

        if verbose:
            print("=" * 70)
            print("ENTRENAMIENTO CON ALGORITMO DE DIEGO")
            print("=" * 70)
            print(f"Particiones   : {num_partitions}")
            print(f"Épocas        : {epochs}")
            print(f"Learning rate : {learning_rate}")
            print(
                f"Arquitectura  : {self.input_size} → {self.hidden_size} → {self.output_size}"
            )
            print("=" * 70)

        for epoch in range(epochs):
            if verbose:
                print(f"\n--- Época {epoch + 1}/{epochs} ---")

            global_params = self.get_parameters()
            partition_params = []
            partition_metrics = []

            for p_idx, (X_part, Y_part) in enumerate(partitions):
                # Cada partición empieza desde los mismos pesos globales
                self.set_parameters(global_params)

                # Entrena localmente
                loss, accuracy = self.train_on_batch(X_part, Y_part, learning_rate)

                partition_params.append(self.get_parameters())
                partition_metrics.append(accuracy)

                if verbose:
                    print(
                        f"  Partición {p_idx + 1}: loss={loss:.4f}  acc={accuracy:.2f}%"
                    )

            # Actualiza los parámetros según el promedio de todos los modelos
            self.set_parameters(average_network_parameters(partition_params))

            # Apila verticalmente los X (imágenes de (N, 784)) de todas las particiones
            all_X = np.vstack([X for X, _ in partitions])

            # Apila verticalmente las Y (etiquetas de (N,)) de todas las particiones
            all_Y = np.concatenate([Y for _, Y in partitions])

            # Evalúa el modelo global usando TODOS los datos juntos
            # Esto sirve para monitorear el entrenamiento, revisando si el modelo está mejorando en cada época
            # Es básicamente revisar el modelo en su conjunto, en vez de partición por partición
            # NO es lo mismo que usar los datos de test, es solo una validación para debugging
            global_accuracy, global_loss = self.evaluate(all_X, all_Y)

            accuracies.append(global_accuracy)
            losses.append(global_loss)
            partition_accuracies.append(partition_metrics)

            if verbose:
                print(f"  Global → loss={global_loss:.4f}  acc={global_accuracy:.2f}%")

            if on_epoch_end is not None:
                on_epoch_end(epoch + 1, epochs, global_accuracy, global_loss)

        history = {
            "accuracies": accuracies,
            "losses": losses,
            "partition_accuracies": partition_accuracies,
        }
        self.training_history = history
        return history

    # ========================
    # INFERENCIA Y EVALUACIÓN
    # ========================

    def predict(self, x: np.ndarray) -> int:
        """
        Predice la clase de una sola imagen.

        Hace forward propagation y devuelve la clase más probable.

        :param x: Imagen (input_size,)
        :type x: np.ndarray

        :return: Clase predicha
        :rtype: int
        """
        output, _ = self.forward(x)
        return int(np.argmax(output))

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """
        Evalúa la red sobre un conjunto de datos (vectorizado).

        Delega el forward a ``_forward_batch`` para no duplicar código
        con ``train_on_batch``.

        :param X: Imágenes (N, 784)
        :type X: np.ndarray

        :param Y: Etiquetas (N,)
        :type Y: np.ndarray

        :return: (accuracy %, loss promedio)
        :rtype: Tuple[float, float]
        """
        n = len(X)
        _, A2 = self._forward_batch(X)  # solo necesita A2; A1 se descarta

        predictions = np.argmax(A2, axis=0)
        correct = np.sum(predictions == Y)

        log_probs = np.log(np.clip(A2, 1e-15, 1.0))
        total_loss = -np.sum(log_probs[Y, np.arange(n)])

        return 100.0 * correct / n, total_loss / n
