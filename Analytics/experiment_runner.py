"""
Ejecutor de experimentos para Algoritmo de Diego.

Gestiona la ejecución de múltiples entrenamientos con diferentes
configuraciones y semillas aleatorias.
"""

import json
import random
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.mnist_loader import load_mnist_train, load_mnist_test
from Utils.data_partitioner import partition_mnist_data_simple
from Networks.nn_diego import DiegoNeuronalNetwork


def run_single_experiment(
    num_partitions: int = 2,
    num_epochs: int = 5,
    hidden_neurons: int = 30,
    learning_rate: float = 1.0,
    n_train: int = 5000,
    random_seed: int | None = None,
    verbose: bool = False,
    on_progress: Any = None,
) -> Dict[str, Any]:
    """
    Ejecuta un único experimento de Algoritmo de Diego.

    :param num_partitions: Número de particiones
    :type num_partitions: int

    :param num_epochs: Número de épocas
    :type num_epochs: int

    :param hidden_neurons: Neuronas en capa oculta
    :type hidden_neurons: int

    :param learning_rate: Tasa de aprendizaje
    :type learning_rate: float

    :param n_train: Ejemplos de entrenamiento
    :type n_train: int

    :param random_seed: Semilla para reproducibilidad
    :type random_seed: int | None

    :param verbose: Si True, muestra progreso
    :type verbose: bool

    :param on_progress: Callback opcional para reportar progreso.\n
                        Firma: on_progress(mensaje: str) → None
    :type on_progress: Callable | None

    :return: Historial del experimento con métricas
    :rtype: Dict[str, Any]
    """
    def _notify(msg: str) -> None:
        """Envía un mensaje de progreso si hay callback registrado."""
        if on_progress is not None:
            on_progress(msg)
        if verbose:
            print(msg)

    if random_seed is not None:
        random.seed(random_seed)

    # Carga datos
    _notify("[Cargando datos MNIST...]")
    X_train, Y_train = load_mnist_train(
        n_train=n_train, download_if_missing=True, verbose=False
    )

    # Crea particiones
    _notify(f"[Creando {num_partitions} particiones estratificadas...]")
    partitions = partition_mnist_data_simple(
        num_partitions=num_partitions,
        X_train=X_train,
        Y_train=Y_train,
        random_seed=random_seed,
        verbose=False,
    )

    # Crea la red
    _notify(f"[Inicializando red: 784 → {hidden_neurons} → 10]")
    network = DiegoNeuronalNetwork(
        input_size=784,
        hidden_size=hidden_neurons,
        output_size=10,
        random_seed=random_seed,
    )

    start_time = time.time()

    # Construye el callback de época que traduce el formato de train_federated
    # al mensaje de texto que entiende on_progress
    def _on_epoch_end(epoch: int, total: int, accuracy: float, loss: float) -> None:
        _notify(f"[Época {epoch}/{total} — Precisión: {accuracy:.2f}%  Loss: {loss:.4f}]")

    history = network.train_federated(
        partitions=partitions,
        epochs=num_epochs,
        learning_rate=learning_rate,
        verbose=verbose,
        on_epoch_end=_on_epoch_end,
    )

    elapsed_time = time.time() - start_time

    # Evalúa en conjunto de prueba
    _notify("[Evaluando en conjunto de prueba...]")
    X_test, Y_test = load_mnist_test(verbose=False)
    test_accuracy, test_loss = network.evaluate(X_test[:1000], Y_test[:1000])

    _notify(f"[Precisión en test: {test_accuracy:.2f}%]")

    return {
        "accuracies": history["accuracies"],
        "losses": history.get("losses", []),
        "partition_accuracies": history.get("partition_accuracies", []),
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "training_time": elapsed_time,
        "random_seed": random_seed,
        "final_accuracy": history["accuracies"][-1] if history["accuracies"] else 0,
    }


def run_multiple_experiments(
    num_partitions: int = 2,
    num_epochs: int = 5,
    num_experiments: int = 5,
    hidden_neurons: int = 30,
    learning_rate: float = 1.0,
    n_train: int = 5000,
    verbose: bool = True,
    on_progress: Any = None,
) -> Dict[str, Any]:
    """
    Ejecuta múltiples experimentos con diferentes semillas aleatorias.

    :param num_partitions: Número de particiones
    :type num_partitions: int

    :param num_epochs: Número de épocas
    :type num_epochs: int

    :param num_experiments: Número de experimentos a ejecutar
    :type num_experiments: int

    :param hidden_neurons: Neuronas en capa oculta
    :type hidden_neurons: int

    :param learning_rate: Tasa de aprendizaje
    :type learning_rate: float

    :param n_train: Ejemplos de entrenamiento
    :type n_train: int

    :param verbose: Si True, muestra progreso
    :type verbose: bool

    :param on_progress: Callback opcional para reportar progreso.\n
                        Firma: on_progress(mensaje: str) → None
    :type on_progress: Callable | None

    :return: Resultados agregados de todos los experimentos
    :rtype: Dict[str, Any]
    """
    def _notify(msg: str) -> None:
        """Envía un mensaje de progreso si hay callback registrado."""
        if on_progress is not None:
            on_progress(msg)
        if verbose:
            print(msg)

    _notify(f"{'=' * 70}")
    _notify(f"EJECUTANDO {num_experiments} EXPERIMENTOS")
    _notify(f"  Particiones: {num_partitions} | Épocas: {num_epochs} | "
            f"Neuronas: {hidden_neurons} | LR: {learning_rate} | N: {n_train}")
    _notify(f"{'=' * 70}")

    all_histories = []
    test_accuracies = []

    for exp_idx in range(num_experiments):
        _notify(f"EXPERIMENTO {exp_idx + 1}/{num_experiments}")

        # Semilla única para cada experimento
        seed = random.randint(0, 1000000)

        result = run_single_experiment(
            num_partitions=num_partitions,
            num_epochs=num_epochs,
            hidden_neurons=hidden_neurons,
            learning_rate=learning_rate,
            n_train=n_train,
            random_seed=seed,
            verbose=verbose,
            on_progress=on_progress,
        )

        all_histories.append(result)
        test_accuracies.append(result["test_accuracy"])
        _notify(f"  ✓ Completado — Precisión final: {result['final_accuracy']:.2f}%")

    # Calcula estadísticas finales
    final_accuracies = [h["final_accuracy"] for h in all_histories]
    mean_accuracy = sum(final_accuracies) / len(final_accuracies)

    import math

    variance = sum((x - mean_accuracy) ** 2 for x in final_accuracies) / len(
        final_accuracies
    )
    std_accuracy = math.sqrt(variance)

    mean_test = sum(test_accuracies) / len(test_accuracies)

    results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "parameters": {
            "num_partitions": num_partitions,
            "num_epochs": num_epochs,
            "num_experiments": num_experiments,
            "hidden_neurons": hidden_neurons,
            "learning_rate": learning_rate,
            "n_train": n_train,
        },
        "all_histories": all_histories,
        "final_mean_accuracy": mean_accuracy,
        "final_std_accuracy": std_accuracy,
        "test_mean_accuracy": mean_test,
        "test_accuracies": test_accuracies,
    }

    _notify(f"Precisión final promedio: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
    _notify(f"Precisión en test promedio: {mean_test:.2f}%")

    return results


def compare_configurations(
    configurations: List[Dict[str, Any]], verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Compara múltiples configuraciones de hiperparámetros.

    :param configurations: Lista de diccionarios con configuraciones
    :type configurations: List[Dict[str, Any]]

    :param verbose: Si True, muestra progreso
    :type verbose: bool

    :return: Lista de resultados para cada configuración
    :rtype: List[Dict[str, Any]]
    """
    all_results = []

    for idx, config in enumerate(configurations):
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"CONFIGURACIÓN {idx + 1}/{len(configurations)}")
            print(f"{'=' * 70}")
            print(f"Parámetros: {config}")

        result = run_multiple_experiments(**config, verbose=verbose)
        all_results.append(result)

    return all_results


def save_experiment_results(
    results: Dict[str, Any], filename: str | None = None
) -> str:
    """
    Guarda resultados de experimentos en archivo JSON.

    :param results: Diccionario con resultados
    :type results: Dict[str, Any]

    :param filename: Nombre del archivo (opcional)
    :type filename: str | None

    :return: Ruta del archivo guardado
    :rtype: str
    """
    if filename is None:
        filename = f"Results/experiment_{results['timestamp']}.json"

    # Asegura que existe el directorio
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    return filename


def load_experiment_results(filename: str) -> Dict[str, Any]:
    """
    Carga resultados de experimentos desde archivo JSON.

    :param filename: Ruta del archivo
    :type filename: str

    :return: Diccionario con resultados
    :rtype: Dict[str, Any]
    """
    with open(filename, "r") as f:
        return json.load(f)


# =================
# PRUEBAS
# =================


def _test_experiment_runner():
    """Pruebas del ejecutor de experimentos."""
    print("=" * 70)
    print("PRUEBAS DE EXPERIMENT_RUNNER")
    print("=" * 70)

    print("\n1. Ejecutando experimento único (rápido)...")
    result = run_single_experiment(
        num_partitions=2,
        num_epochs=2,
        hidden_neurons=10,
        n_train=1000,
        random_seed=42,
        verbose=True,
    )
    print(f"\n   Precisión final: {result['final_accuracy']:.2f}%")
    print(f"   Tiempo: {result['training_time']:.2f}s")

    print("\n2. Ejecutando múltiples experimentos...")
    results = run_multiple_experiments(
        num_partitions=2,
        num_epochs=2,
        num_experiments=2,
        hidden_neurons=10,
        n_train=1000,
        verbose=True,
    )
    print(f"\n   Media: {results['final_mean_accuracy']:.2f}%")
    print(f"   Desv: {results['final_std_accuracy']:.2f}%")

    print("\n" + "=" * 70)
    print("PRUEBAS COMPLETADAS")
    print("=" * 70)


if __name__ == "__main__":
    _test_experiment_runner()
