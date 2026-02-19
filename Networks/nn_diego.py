"""
Red Neuronal para MNIST basado en el algoritmo de Diego.

Esta red implementa un algoritmo de entrenamiento donde:
1. Los datos se dividen en múltiples particiones estratificadas
2. Cada partición entrena la red de forma independiente
3. Los parámetros (W1, b1, W2, b2) se promedian entre particiones
4. La red se reinicializa con los parámetros promediados para la siguiente época

Arquitectura: 784 (entrada) → 30 (oculta, sigmoide) → 10 (salida, softmax)
"""
# ================
# IMPORTACIONES
# ================

import math
import os
import random
import sys
from typing import Any, Dict, List, Tuple

# Agregaa directorio padre al path para importar Utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importa utilidades
from Utils.math_utils import (
    accumulate_gradients,
    argmax,
    average_network_parameters,
    compute_one_hot,
    initialize_gradient_accumulators,
    matrix_transpose,
    matrix_vector_multiply,
    outer_product,
    scale_gradients,
    sigmoid,
    sigmoid_derivative_from_activation,
    softmax,
    vector_add,
    vector_subtract,
    vector_zeros,
    xavier_initialization,
)

from Utils.data_partitioner import partition_mnist_data_simple
from Utils.mnist_loader import load_mnist_test, load_mnist_train

# =============
# RED NEURONAL
# =============


class DiegoNeuronalNetwork:
    """
    Red Neuronal que soporta el entrenamiento tipo Diego.

    Attributes:
        input_size: Número de neuronas de entrada (784 para MNIST)
        hidden_size: Número de neuronas en capa oculta
        output_size: Número de neuronas de salida (10 para MNIST)
        W1: Pesos de capa entrada → oculta
        b1: Sesgos de capa oculta
        W2: Pesos de capa oculta → salida
        b2: Sesgos de capa de salidaad
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 30,
        output_size: int = 10,
        random_seed: int | None = None,
    ):
        """
        Inicializa la red con pesos aleatorios usando inicialización Xavier.

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

    def get_parameters(self) -> Dict[str, Any]:
        """
        Obtiene los parámetros actuales de la red.

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
        Establece los parámetros de la red.

        :param params: Diccionario con W1, b1, W2, b2
        :type params: Dict[str, Any]
        """
        self.W1 = [row[:] for row in params["W1"]]
        self.b1 = params["b1"][:]
        self.W2 = [row[:] for row in params["W2"]]
        self.b2 = params["b2"][:]

    def forward(self, x: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Realiza la propagación hacia adelante.

        :param x: Vector de entrada (imagen aplanada)
        :type x: List[float]

        :return:
            - output: Probabilidades de cada clase después de aplicar la función softmax.\n
            - cache: Valores intermedios para backpropagation
        :rtype: Tuple[List[float], Dict[str, Any]]
        """
        # Capa oculta: z1 = W1·x + b1
        z1 = matrix_vector_multiply(self.W1, x)
        z1 = vector_add(z1, self.b1)

        # Activación sigmoide: a1 = σ(z1)
        a1 = [sigmoid(z) for z in z1]

        # Capa de salida: z2 = W2·a1 + b2
        z2 = matrix_vector_multiply(self.W2, a1)
        z2 = vector_add(z2, self.b2)

        # Softmax para probabilidades
        output = softmax(z2)

        # Guarda valores para backpropagation
        cache = {"x": x[:], "z1": z1, "a1": a1, "z2": z2, "output": output}

        return output, cache

    def backward(
        self, y: int, cache: Dict[str, Any]
    ) -> Tuple[List[List[float]], List[float], List[List[float]], List[float]]:
        """
        Retropropagación para calcular gradientes.

        :param y: Etiqueta verdadera asociada a la entrada (0-9)
        :type y: int

        :param cache: Valores guardados en forward propagation
        :type cache: Dict[str, Any]

        :return:
                Gradientes calculados para cada parámetro del modelo:\n
                - dW1: Gradiente de los pesos de la primera capa.\n
                - db1: Gradiente de los sesgos de la primera capa.\n
                - dW2: Gradiente de los pesos de la segunda capa.\
                - db2: Gradiente de los sesgos de la segunda capa.
        :rtype: Tuple[List[List[float]], List[float], List[List[float]], List[float]]
        """
        # Recupera valores de la cache
        x = cache["x"]
        a1 = cache["a1"]
        output = cache["output"]

        # Aplica One-hot encoding para la etiqueta
        y_onehot = compute_one_hot(y, self.output_size)

        # Gradiente de la capa de salida
        # δ2 = output - y_onehot (derivada de cross-entropy + softmax)
        delta2 = vector_subtract(output, y_onehot)

        # Gradiente para W2 y b2
        dW2 = outer_product(delta2, a1)
        db2 = delta2[:]

        # Gradiente de la capa oculta
        # δ1 = (W2^T · δ2) ⊙ σ'(a1)
        W2_T = matrix_transpose(self.W2)
        delta1 = matrix_vector_multiply(W2_T, delta2)

        # Derivada de sigmoide: σ'(z) = σ(z) * (1 - σ(z)) = a1 * (1 - a1)
        sigmoid_prime = [sigmoid_derivative_from_activation(a) for a in a1]

        # Producto elemento a elemento
        delta1 = [delta1[i] * sigmoid_prime[i] for i in range(len(delta1))]

        # Gradientes para W1 y b1
        dW1 = outer_product(delta1, x)
        db1 = delta1[:]

        return dW1, db1, dW2, db2

    # Este método es que usaba la primera red neuronal que hicimos
    def train_on_batch(
        self,
        X: List[List[float]],
        Y: List[int],
        learning_rate: float,
        verbose: bool = False,
    ) -> Tuple[float, float]:
        """
        Entrena la red en un batch de datos usando gradiente descendente.

        :param X: Lista de entradas (por ejemplo, imágenes representadas como listas de floats).
        :type X: List[List[float]]

        :param Y: Lista de etiquetas correspondientes a cada entrada.
        :type Y: List[int]

        :param learning_rate: Tasa de aprendizaje utilizada para actualizar los parámetros.
        :type learning_rate: float

        :param verbose: Si es True, muestra el progreso durante el entrenamiento.
        :type verbose: bool

        :return:
            - loss (float): Pérdida promedio del batch.\n
            - accuracy (float): Precisión del batch en porcentaje.
        :rtype: Tuple[float, float]
        """
        # Calcula cantidad de imágenes de entrada
        n = len(X)

        # Inicializa acumuladores de gradientes
        accumulators = initialize_gradient_accumulators(
            {
                "W1": self.W1,
                "b1": self.b1,
                "W2": self.W2,
                "b2": self.b2,
            }
        )
        correct_predictions = 0
        total_loss = 0.0

        # Acumula gradientes de todos los ejemplos
        for i in range(n):
            # Forward pass
            output, cache = self.forward(X[i])

            # Calcula predicción y precisión
            prediction = argmax(output)
            if prediction == Y[i]:
                correct_predictions += 1

            # Calcular pérdida (cross-entropy)
            # Evita log(0) con clipping
            prob = max(output[Y[i]], 1e-15)
            total_loss += -math.log(prob)

            # Backward pass
            dW1, db1, dW2, db2 = self.backward(Y[i], cache)

            # Acumular gradientes
            accumulate_gradients(
                accumulators, {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
            )

            if verbose and (i + 1) % 1000 == 0:
                print(f"    Procesados {i + 1}/{n} ejemplos...")

        # Calcular gradientes promedio
        scale_gradients(accumulators, 1.0 / n)

        # Actualizar parámetros
        self._update_parameters(accumulators, learning_rate)

        # Calcular métricas
        loss = total_loss / n
        accuracy = 100.0 * correct_predictions / n

        return loss, accuracy

    def _update_parameters(
        self, accumulators: Dict[str, Any], learning_rate: float
    ) -> None:
        """
        Actualiza los parámetros usando los gradientes acumulados.

        :param accumulators: Diccionario con gradientes acumulados
        :type accumulators: Dict[str, Any]

        :param learning_rate: Tasa de aprendizaje
        :type learning_rate: float
        """
        # Actualizar W1 y b1
        for j in range(self.hidden_size):
            for k in range(self.input_size):
                self.W1[j][k] -= learning_rate * accumulators["dW1"][j][k]
            self.b1[j] -= learning_rate * accumulators["db1"][j]

        # Actualizar W2 y b2
        for j in range(self.output_size):
            for k in range(self.hidden_size):
                self.W2[j][k] -= learning_rate * accumulators["dW2"][j][k]
            self.b2[j] -= learning_rate * accumulators["db2"][j]

    def predict(self, x: List[float]) -> int:
        """
        Predice la clase de una imagen.

        :param x: Imagen aplanada
        :type x: List[float]

        :return: Clase predicha (0-9)
        :rtype: int
        """
        output, _ = self.forward(x)
        return argmax(output)

    def evaluate(self, X: List[List[float]], Y: List[int]) -> Tuple[float, float]:
        """
        Evalúa la red en un conjunto de datos.

        :param X: Lista de imágenes
        :type X: List[List[float]]

        :param Y: Lista de etiquetas verdaderas
        :type Y: List[int]

        :return:
            - accuracy (float): Precisión en porcentaje.\n
            - avg_loss (float): Pérdida promedio
        :rtype: Tuple[float, float]
        """

        correct = 0
        total_loss = 0.0
        n = len(X)

        for i in range(n):
            output, _ = self.forward(X[i])
            prediction = argmax(output)

            if prediction == Y[i]:
                correct += 1

            # Estamos evaluando loss = -log(p)
            # Si p = 0, entonces log(0) = - ∞
            # Llegado ese caso, reemplazamos 0 por 1e-15
            # Eso es Clipping Numéricos (estabilidad numérica)
            prob = max(output[Y[i]], 1e-15)
            total_loss += -math.log(prob)

        accuracy = 100.0 * correct / n
        avg_loss = total_loss / n

        return accuracy, avg_loss

    # Este es el algoritmo de Diego, pero "federated" sonaba más elegante
    def train_federated(
        self,
        partitions: List[Tuple[List[List[float]], List[int]]],
        epochs: int,
        learning_rate: float,
        verbose: bool = True,
        on_epoch_end: Any = None,
    ) -> Dict[str, Any]:
        """
        Entrena la red usando el algoritmo propuesto por Diego.

        Algoritmo:\n
        Para cada época:
            a. Guardar parámetros globales actuales
            b. Para cada partición:\n
                - Entrenar red en la partición
                - Guardar parámetros resultantes\n
            c. Promediar todos los parámetros obtenidos
            d. Establecer parámetros promediados como nuevos globales

        :param partitions: Lista de particiones (X_partition, Y_partition)
        :type partitions: List[Tuple[List[List[float]], List[int]]]
        :param epochs: Número de épocas de entrenamiento
        :type epochs: int
        :param learning_rate: Tasa de aprendizaje
        :type learning_rate: float
        :param verbose: Si True, muestra progreso detallado
        :type verbose: bool
        :param on_epoch_end: Callback opcional llamado al finalizar cada época.\n
                             Firma: on_epoch_end(epoch: int, total_epochs: int,\n
                             accuracy: float, loss: float) → None
        :type on_epoch_end: Callable | None
        :return: Historial de entrenamiento con las siguientes claves:\n
                 - 'accuracies': List[float] — precisión global por época.\n
                 - 'losses': List[float] — pérdida global por época.\n
                 - 'partition_accuracies': List[List[float]] — precisión de cada\n
                   partición por época. Forma: [epoca][particion].\n
                 - 'epochs_detail': List[Dict] — detalle completo por época.
        :rtype: Dict[str, Any]
        """
        num_partitions = len(partitions)

        # Listas acumuladoras — formato esperado por experiment_runner y statistics_engine
        accuracies: List[float] = []
        losses: List[float] = []
        # partition_accuracies[epoca] = [acc_part0, acc_part1, ...]
        partition_accuracies: List[List[float]] = []
        epochs_detail: List[Dict[str, Any]] = []

        if verbose:
            print("=" * 70)
            print("ENTRENAMIENTO FEDERADO")
            print("=" * 70)
            print(f"Número de particiones: {num_partitions}")
            print(f"Épocas: {epochs}")
            print(f"Tasa de aprendizaje: {learning_rate}")
            print(
                f"Arquitectura: {self.input_size} → {self.hidden_size} → {self.output_size}"
            )
            print("=" * 70)

        for epoch in range(epochs):
            if verbose:
                print(f"\n--- Época {epoch + 1}/{epochs} ---")

            # Guarda parámetros globales al inicio de esta época
            global_params = self.get_parameters()

            partition_parameters = []
            partition_metrics = []

            # Entrena en cada partición de forma independiente
            for partition_idx, (X_part, Y_part) in enumerate(partitions):
                if verbose:
                    print(f"\n  Partición {partition_idx + 1}/{num_partitions}:")
                    print(f"    Ejemplos: {len(X_part)}")

                # Cada partición parte de los mismos parámetros globales
                self.set_parameters(global_params)

                loss, accuracy = self.train_on_batch(
                    X_part, Y_part, learning_rate, verbose=False
                )

                partition_parameters.append(self.get_parameters())
                partition_metrics.append(
                    {"partition": partition_idx + 1, "loss": loss, "accuracy": accuracy}
                )

                if verbose:
                    print(f"    Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

            # Promedia parámetros de todas las particiones
            if verbose:
                print(f"\n  Promediando parámetros de {num_partitions} particiones...")

            averaged_params = average_network_parameters(partition_parameters)
            self.set_parameters(averaged_params)

            # Evalúa el modelo promediado sobre todos los datos combinados
            all_X = []
            all_Y = []
            for X_part, Y_part in partitions:
                all_X.extend(X_part)
                all_Y.extend(Y_part)

            global_accuracy, global_loss = self.evaluate(all_X, all_Y)

            # Acumula métricas en el formato esperado por Analytics
            accuracies.append(global_accuracy)
            losses.append(global_loss)
            partition_accuracies.append(
                [m["accuracy"] for m in partition_metrics]
            )
            epochs_detail.append(
                {
                    "epoch": epoch + 1,
                    "global_loss": global_loss,
                    "global_accuracy": global_accuracy,
                    "partition_metrics": partition_metrics,
                }
            )

            if verbose:
                print(f"\n  Resultado global Época {epoch + 1}:")
                print(f"    Loss: {global_loss:.4f}, Accuracy: {global_accuracy:.2f}%")
                print("-" * 50)

            # Notifica al caller que terminó esta época
            if on_epoch_end is not None:
                on_epoch_end(epoch + 1, epochs, global_accuracy, global_loss)

        history: Dict[str, Any] = {
            "accuracies": accuracies,
            "losses": losses,
            "partition_accuracies": partition_accuracies,
            "epochs_detail": epochs_detail,
        }

        self.training_history = history

        if verbose:
            print("\n" + "=" * 70)
            print("ENTRENAMIENTO FEDERADO COMPLETADO")
            print("=" * 70)

        return history


# ================
# DEMOSTRACIÓN
# ================


def demo_federated_learning():
    """
    Demostración completa del entrenamiento con Algoritmo de Diego con MNIST.
    """

    print("\n" + "=" * 70)
    print("DEMO: RED NEURONAL CON ALGORITMO DE DIEGO EN MNIST")
    print("=" * 70)

    # 1. Cargar datos
    print("\n1. Cargando datos MNIST...")
    X_train, Y_train = load_mnist_train(n_train=10000, verbose=False)
    X_test, Y_test = load_mnist_test(verbose=False)
    print(f"   Entrenamiento: {len(X_train)} ejemplos")
    print(f"   Prueba: {len(X_test)} ejemplos")

    # 2. Crear particiones estratificadas
    num_partitions = 2
    print(f"\n2. Creando {num_partitions} particiones estratificadas...")

    partitions = partition_mnist_data_simple(
        num_partitions=num_partitions,
        X_train=X_train,
        Y_train=Y_train,
        random_seed=42,
        verbose=False,
    )

    for i, (X_part, Y_part) in enumerate(partitions):
        print(f"   Partición {i + 1}: {len(X_part)} ejemplos")

    # 3. Crear red neuronal
    print("\n3. Creando red neuronal...")
    network = DiegoNeuronalNetwork(
        input_size=784, hidden_size=30, output_size=10, random_seed=42
    )
    print("   Arquitectura: 784 → 30 → 10")

    # 4. Entrenamiento con algoritmo de Diego
    print("\n4. Iniciando entrenamiento con Algoritmo de Diego...")
    history = network.train_federated(
        partitions=partitions, epochs=10, learning_rate=0.5, verbose=True
    )

    # Muestra evolución de precisión usando el nuevo formato
    print("\n   Evolución de precisión por época:")
    for epoch, (acc, loss) in enumerate(zip(history["accuracies"], history["losses"])):
        print(f"   Época {epoch + 1:2d}: {acc:.2f}%  loss={loss:.4f}")

    # 5. Evaluar en conjunto de prueba
    print("\n5. Evaluando en conjunto de prueba...")
    test_accuracy, test_loss = network.evaluate(X_test, Y_test)
    print(f"   Precisión en test: {test_accuracy:.2f}%")
    print(f"   Pérdida en test: {test_loss:.4f}")

    # 6. Mostrar algunas predicciones
    print("\n6. Ejemplos de predicciones:")
    for i in range(5):
        pred = network.predict(X_test[i])
        real = Y_test[i]
        status = "✓" if pred == real else "✗"
        print(f"   Imagen {i + 1}: Predicción={pred}, Real={real} {status}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETADA")
    print("=" * 70)

    return network, history


if __name__ == "__main__":
    # Ejecutar demo principal
    demo_federated_learning()
