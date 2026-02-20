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
    argmax,
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

        :param x: Vector de entrada (input_size,)
        :type x: np.ndarray

        :return: (probabilidades softmax, cache intermedio)
        :rtype: Tuple[np.ndarray, Dict[str, Any]]
        """
        z1 = self.W1 @ x + self.b1
        a1 = sigmoid(z1)
        z2 = self.W2 @ a1 + self.b2
        output = softmax(z2)
        cache = {"x": x.copy(), "a1": a1, "output": output}
        return output, cache

    # ========================
    # ENTRENAMIENTO
    # ========================

    def train_on_batch(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        learning_rate: float,
        verbose: bool = False,
    ) -> Tuple[float, float]:
        """
        Entrena sobre un batch completo usando gradiente descendente vectorizado.

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
        n = len(X)

        # --- Forward vectorizado (todo el batch a la vez) ---
        # X tiene forma (N, 784), necesitamos (784, N) para la multiplicación
        X_T = X.T  # (784, N)

        Z1 = self.W1 @ X_T + self.b1[:, np.newaxis]  # (hidden, N)
        A1 = sigmoid(Z1)  # (hidden, N)

        Z2 = self.W2 @ A1 + self.b2[:, np.newaxis]  # (output, N)

        # Softmax por columna (cada columna es un ejemplo)
        Z2_stable = Z2 - np.max(Z2, axis=0, keepdims=True)
        exp_Z2 = np.exp(Z2_stable)
        A2 = exp_Z2 / np.sum(exp_Z2, axis=0, keepdims=True)  # (output, N)

        # --- Métricas ---
        predictions = np.argmax(A2, axis=0)  # (N,)
        correct = np.sum(predictions == Y)
        # Cross-entropy loss
        log_probs = np.log(np.clip(A2, 1e-15, 1.0))
        total_loss = -np.sum(log_probs[Y, np.arange(n)])

        # --- Backward vectorizado ---
        # One-hot de todas las etiquetas: (output, N)
        Y_onehot = np.zeros((self.output_size, n))
        Y_onehot[Y, np.arange(n)] = 1.0

        # δ2 = A2 - Y_onehot, forma (output, N)
        delta2 = A2 - Y_onehot

        # Gradientes capa 2
        dW2 = (1.0 / n) * (delta2 @ A1.T)  # (output, hidden)
        db2 = (1.0 / n) * np.sum(delta2, axis=1)  # (output,)

        # δ1 = (W2^T · δ2) ⊙ σ'(A1), forma (hidden, N)
        delta1 = (self.W2.T @ delta2) * sigmoid_derivative_from_activation(A1)

        # Gradientes capa 1
        dW1 = (1.0 / n) * (delta1 @ X_T.T)  # (hidden, input)
        db1 = (1.0 / n) * np.sum(delta1, axis=1)  # (hidden,)

        # --- Actualización ---
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
                self.set_parameters(global_params)
                loss, accuracy = self.train_on_batch(X_part, Y_part, learning_rate)
                partition_params.append(self.get_parameters())
                partition_metrics.append(accuracy)

                if verbose:
                    print(
                        f"  Partición {p_idx + 1}: loss={loss:.4f}  acc={accuracy:.2f}%"
                    )

            # Promedia parámetros
            self.set_parameters(average_network_parameters(partition_params))

            # Evalúa globalmente
            all_X = np.vstack([X for X, _ in partitions])
            all_Y = np.concatenate([Y for _, Y in partitions])
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

        :param x: Imagen (input_size,)
        :type x: np.ndarray
        :return: Clase predicha
        :rtype: int
        """
        output, _ = self.forward(x)
        return argmax(output)

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """
        Evalúa la red sobre un conjunto de datos (vectorizado).

        :param X: Imágenes (N, 784)
        :type X: np.ndarray
        :param Y: Etiquetas (N,)
        :type Y: np.ndarray
        :return: (accuracy %, loss promedio)
        :rtype: Tuple[float, float]
        """
        n = len(X)
        X_T = X.T

        Z1 = self.W1 @ X_T + self.b1[:, np.newaxis]
        A1 = sigmoid(Z1)
        Z2 = self.W2 @ A1 + self.b2[:, np.newaxis]

        Z2_stable = Z2 - np.max(Z2, axis=0, keepdims=True)
        exp_Z2 = np.exp(Z2_stable)
        A2 = exp_Z2 / np.sum(exp_Z2, axis=0, keepdims=True)

        predictions = np.argmax(A2, axis=0)
        correct = np.sum(predictions == Y)

        log_probs = np.log(np.clip(A2, 1e-15, 1.0))
        total_loss = -np.sum(log_probs[Y, np.arange(n)])

        return 100.0 * correct / n, total_loss / n
