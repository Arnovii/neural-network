"""
Red neuronal para clasificación de MNIST con entrenamiento federado.

Arquitectura: 784 (entrada) → oculta (sigmoide) → 10 (salida, softmax).

El método principal es train_federated, que implementa el algoritmo de Diego:
cada época entrena copias independientes de la red sobre distintas particiones
y luego promedia los parámetros resultantes.
"""

import math
import os
import random
import sys
from typing import Any, Callable, Dict, List, Tuple

# Agregaa directorio padre al path para importar Utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importa utilidades matemáticas
from Utils.math_utils import (
    accumulate_outer_inplace,
    accumulate_vector_inplace,
    argmax,
    average_network_parameters,
    compute_one_hot,
    matrix_transpose,
    matrix_vector_multiply,
    sigmoid,
    sigmoid_derivative_from_activation,
    softmax,
    vector_add,
    vector_subtract,
    vector_zeros,
    xavier_initialization,
)

# =============
# RED NEURONAL
# =============


class DiegoNeuronalNetwork:
    """
    Red neuronal con soporte para el algoritmo de entrenamiento de Diego.

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
        """
        Inicializa la red con pesos Xavier y sesgos en cero.

        :param input_size: Tamaño de la capa de entrada
        :type input_size: int

        :param hidden_size: Tamaño de la capa oculta
        :type hidden_size: int

        :param output_size: Tamaño de la capa de salida
        :type output_size: int

        :param random_seed: Semilla para reproducibilidad
        :type random_seed: int | None
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Establece la semilla si se proporciona
        if random_seed is not None:
            random.seed(random_seed)

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

    def get_parameters(self) -> Dict[str, Any]:
        """
        Devuelve una copia profunda de los parámetros actuales.

        :return: Diccionario con W1, b1, W2, b2
        :rtype: Dict[str, Any]
        """
        return {
            "W1": [row[:] for row in self.W1],
            "b1": self.b1[:],
            "W2": [row[:] for row in self.W2],
            "b2": self.b2[:],
        }

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Establece los parámetros de la red a partir de un diccionario.

        :param params: Diccionario con W1, b1, W2, b2
        :type params: Dict[str, Any]
        """
        self.W1 = [row[:] for row in params["W1"]]
        self.b1 = params["b1"][:]
        self.W2 = [row[:] for row in params["W2"]]
        self.b2 = params["b2"][:]

    # ========================
    # FORWARD PROPAGATION
    # ========================

    def forward(self, x: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Propagación hacia adelante para una sola entrada.

        Se usa en predict y evaluate. El bucle de entrenamiento (train_on_batch)
        tiene el forward inlined para evitar el overhead del diccionario cache.

        :param x: Vector de entrada de tamaño input_size
        :type x: List[float]

        :return: Tupla (probabilidades softmax, cache con valores intermedios)
        :rtype: Tuple[List[float], Dict[str, Any]]
        """
        # Capa oculta: z1 = W1·x + b1
        z1 = vector_add(matrix_vector_multiply(self.W1, x), self.b1)

        # Activación sigmoide: a1 = σ(z1)
        a1 = [sigmoid(z) for z in z1]

        # Capa de salida: z2 = W2·a1 + b2
        z2 = vector_add(matrix_vector_multiply(self.W2, a1), self.b2)

        # Softmax para probabilidades
        output = softmax(z2)

        # Guarda valores para backpropagation
        cache = {"x": x[:], "a1": a1, "output": output}

        return output, cache

    # ========================
    # ENTRENAMIENTO
    # ========================

    def train_on_batch(
        self,
        X: List[List[float]],
        Y: List[int],
        learning_rate: float,
        verbose: bool = False,
    ) -> Tuple[float, float]:
        """
        Entrena la red sobre un batch completo usando gradiente descendente.

        Realiza el forward y backward inline para cada ejemplo, acumulando
        gradientes in-place (sin crear matrices intermedias). Al final aplica
        una única actualización con el gradiente promedio del batch.

        :param X: Lista de imágenes de entrada
        :type X: List[List[float]]

        :param Y: Lista de etiquetas correspondientes
        :type Y: List[int]

        :param learning_rate: Tasa de aprendizaje
        :type learning_rate: float

        :param verbose: Si True, reporta el progreso cada 1000 ejemplos
        :type verbose: bool

        :return: Tupla (loss promedio, accuracy en porcentaje)
        :rtype: Tuple[float, float]
        """
        n = len(X)

        # Acumuladores inicializados a cero — se modifican in-place en cada ejemplo
        dW1_acc = [[0.0] * self.input_size for _ in range(self.hidden_size)]
        db1_acc = [0.0] * self.hidden_size
        dW2_acc = [[0.0] * self.hidden_size for _ in range(self.output_size)]
        db2_acc = [0.0] * self.output_size

        correct = 0
        total_loss = 0.0

        for i, (xi, yi) in enumerate(zip(X, Y)):
            # Forward Propagation
            z1 = vector_add(matrix_vector_multiply(self.W1, xi), self.b1)
            a1 = [sigmoid(z) for z in z1]
            z2 = vector_add(matrix_vector_multiply(self.W2, a1), self.b2)
            output = softmax(z2)

            # Métricas
            if argmax(output) == yi:
                correct += 1
            total_loss += -math.log(max(output[yi], 1e-15))

            # Backward Propagation
            # Aplica One-hot encoding para la etiqueta
            y_onehot = compute_one_hot(yi, self.output_size)

            # δ2 = output − y_onehot  (derivada combinada cross-entropy + softmax)
            delta2 = vector_subtract(output, y_onehot)

            # Gradiente de la capa oculta
            # δ1 = (W2ᵀ · δ2) ⊙ σ'(a1)
            W2_T = matrix_transpose(self.W2)
            delta1 = [
                d * sigmoid_derivative_from_activation(a)
                for d, a in zip(matrix_vector_multiply(W2_T, delta2), a1)
            ]

            # Acumulación in-place:
            # accumulate_outer_inplace fusiona outer_product + matrix_add en un
            # solo pase, eliminando dos allocations de lista por ejemplo.
            accumulate_outer_inplace(dW2_acc, delta2, a1)
            accumulate_outer_inplace(dW1_acc, delta1, xi)
            accumulate_vector_inplace(db2_acc, delta2)
            accumulate_vector_inplace(db1_acc, delta1)

            if verbose and (i + 1) % 1000 == 0:
                print(f"    Procesados {i + 1}/{n} ejemplos...")

        # Actualización con factor único lr/n
        self._update_parameters(dW1_acc, db1_acc, dW2_acc, db2_acc, learning_rate, n)

        return total_loss / n, 100.0 * correct / n

    def _update_parameters(
        self,
        dW1_acc: List[List[float]],
        db1_acc: List[float],
        dW2_acc: List[List[float]],
        db2_acc: List[float],
        learning_rate: float,
        n: int,
    ) -> None:
        """
        Actualiza los parámetros in-place con el gradiente promediado.

        Aplica ``lr / n`` como factor único para evitar crear una matriz
        intermedia escalada antes de restar.

        Sería algo tipo: θ <- θ - (lr / n) * ∇θ

        :param dW1_acc: Gradientes acumulados de W1 (sin escalar)
        :type dW1_acc: List[List[float]]

        :param db1_acc: Gradientes acumulados de b1 (sin escalar)
        :type db1_acc: List[float]

        :param dW2_acc: Gradientes acumulados de W2 (sin escalar)
        :type dW2_acc: List[List[float]]

        :param db2_acc: Gradientes acumulados de b2 (sin escalar)
        :type db2_acc: List[float]

        :param learning_rate: Tasa de aprendizaje
        :type learning_rate: float

        :param n: Número de ejemplos del batch (para promediar los gradientes)
        :type n: int
        """
        factor = learning_rate / n

        for wi, dwi in zip(self.W1, dW1_acc):
            for j in range(len(wi)):
                wi[j] -= factor * dwi[j]
        for j in range(self.hidden_size):
            self.b1[j] -= factor * db1_acc[j]

        for wi, dwi in zip(self.W2, dW2_acc):
            for j in range(len(wi)):
                wi[j] -= factor * dwi[j]
        for j in range(self.output_size):
            self.b2[j] -= factor * db2_acc[j]

    def train_federated(
        self,
        partitions: List[Tuple[List[List[float]], List[int]]],
        epochs: int,
        learning_rate: float,
        verbose: bool = True,
        on_epoch_end: Callable | None = None,
    ) -> Dict[str, Any]:
        """
        Entrena la red usando el algoritmo de Diego.

        Por cada época:

        1. Guarda los parámetros globales actuales.
        2. Para cada partición: restaura los parámetros globales y entrena
           de forma independiente con train_on_batch.
        3. Promedia los parámetros de todas las particiones.
        4. Establece el promedio como nuevos parámetros globales.

        :param partitions: Lista de tuplas (X_partition, Y_partition)
        :type partitions: List[Tuple[List[List[float]], List[int]]]

        :param epochs: Número de épocas
        :type epochs: int

        :param learning_rate: Tasa de aprendizaje
        :type learning_rate: float

        :param verbose: Si True, muestra progreso detallado por época y partición
        :type verbose: bool

        :param on_epoch_end: Callback invocado al finalizar cada época con la firma
                             ``on_epoch_end(epoch, total_epochs, accuracy, loss)``.
        :type on_epoch_end: Callable | None
        :return: Historial con las claves:

                 - ``accuracies``: precisión global por época (List[float])
                 - ``losses``: pérdida global por época (List[float])
                 - ``partition_accuracies``: precisión por partición y época
                   (List[List[float]]), forma [época][partición]
        :rtype: Dict[str, Any]
        """
        num_partitions = len(partitions)

        # Listas acumuladoras: formato esperado por experiment_runner y statistics_engine
        accuracies: List[float] = []
        losses: List[float] = []

        # partition_accuracies[epoca] = [acc_part0, acc_part1, ...]
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

            # Guarda parámetros globales al inicio de esta época
            global_params = self.get_parameters()

            partition_params = []
            partition_metrics = []

            # Entrena en cada partición de forma independiente
            for p_idx, (X_part, Y_part) in enumerate(partitions):
                # Cada partición parte de los mismos parámetros globales
                self.set_parameters(global_params)
                loss, accuracy = self.train_on_batch(X_part, Y_part, learning_rate)

                partition_params.append(self.get_parameters())
                partition_metrics.append(accuracy)

                if verbose:
                    print(
                        f"  Partición {p_idx + 1}: loss={loss:.4f}  acc={accuracy:.2f}%"
                    )

            # Promedia parámetros de todas las particiones
            self.set_parameters(average_network_parameters(partition_params))

            # Evalúa el modelo global sobre todos los datos combinados
            all_X: List[List[float]] = []
            all_Y: List[int] = []
            for X_part, Y_part in partitions:
                all_X.extend(X_part)
                all_Y.extend(Y_part)

            global_accuracy, global_loss = self.evaluate(all_X, all_Y)

            accuracies.append(global_accuracy)
            losses.append(global_loss)
            partition_accuracies.append(partition_metrics)

            if verbose:
                print(f"  Global → loss={global_loss:.4f}  acc={global_accuracy:.2f}%")
                print("-" * 50)

            if on_epoch_end is not None:
                on_epoch_end(epoch + 1, epochs, global_accuracy, global_loss)

        history: Dict[str, Any] = {
            "accuracies": accuracies,
            "losses": losses,
            "partition_accuracies": partition_accuracies,
        }
        self.training_history = history

        if verbose:
            print("\n" + "=" * 70)
            print("ENTRENAMIENTO COMPLETADO")
            print("=" * 70)

        return history

    # ========================
    # INFERENCIA Y EVALUACIÓN
    # ========================

    def predict(self, x: List[float]) -> int:
        """
        Predice la clase de una sola imagen.

        :param x: Imagen aplanada de tamaño input_size
        :type x: List[float]

        :return: Clase predicha (0-9)
        :rtype: int
        """
        output, _ = self.forward(x)
        return argmax(output)

    def evaluate(self, X: List[List[float]], Y: List[int]) -> Tuple[float, float]:
        """
        Evalúa la red sobre un conjunto de datos.

        :param X: Lista de imágenes
        :type X: List[List[float]]

        :param Y: Lista de etiquetas verdaderas
        :type Y: List[int]

        :return: Tupla (accuracy en porcentaje, loss promedio)
        :rtype: Tuple[float, float]
        """
        correct = 0
        total_loss = 0.0
        n = len(X)

        for xi, yi in zip(X, Y):
            output, _ = self.forward(xi)
            if argmax(output) == yi:
                correct += 1

            # Estamos evaluando loss = -log(p)
            # Si p = 0, entonces log(0) = - ∞
            # Llegado ese caso, reemplazamos 0 por 1e-15
            # Eso es Clipping Numéricos (estabilidad numérica)
            total_loss += -math.log(max(output[yi], 1e-15))

        return 100.0 * correct / n, total_loss / n
