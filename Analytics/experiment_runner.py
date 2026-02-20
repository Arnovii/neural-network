"""
Analytics/experiment_runner.py
Orquestador de experimentos de Algoritmo de Diego.

Gestiona la ejecución de múltiples entrenamientos con distintas semillas
aleatorias y calcula estadísticas agregadas sobre los resultados.
"""

import math
import random
import time
from datetime import datetime
from typing import Any, Callable, Dict, List

import os
import sys

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
    on_progress: Callable | None = None,
) -> Dict[str, Any]:
    """
    Ejecuta un único experimento de Algoritmo de Diego.

    Carga los datos, crea las particiones, entrena la red con el algoritmo
    de Diego y evalúa sobre los primeros 1000 ejemplos del conjunto de prueba.

    :param num_partitions: Número de particiones
    :type num_partitions: int

    :param num_epochs: Número de épocas de entrenamiento
    :type num_epochs: int

    :param hidden_neurons: Neuronas en la capa oculta
    :type hidden_neurons: int

    :param learning_rate: Tasa de aprendizaje
    :type learning_rate: float

    :param n_train: Número de ejemplos de entrenamiento a usar
    :type n_train: int

    :param random_seed: Semilla para reproducibilidad
    :type random_seed: int | None

    :param verbose: Si True, muestra el progreso en consola
    :type verbose: bool

    :param on_progress: Callback para reportar progreso a la GUI.
                        Firma: ``on_progress(mensaje: str) → None``
    :type on_progress: Callable | None

    :return: Diccionario con accuracies, losses, partition_accuracies,
             test_accuracy, test_loss, training_time, random_seed y final_accuracy
    :rtype: Dict[str, Any]
    """

    def _notify(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)
        if verbose:
            print(msg)

    # Establece la semilla si se proprorciona
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
        _notify(
            f"[Época {epoch}/{total} — Precisión: {accuracy:.2f}%  Loss: {loss:.4f}]"
        )

    history = network.train_federated(
        partitions=partitions,
        epochs=num_epochs,
        learning_rate=learning_rate,
        verbose=verbose,
        on_epoch_end=_on_epoch_end,
    )

    elapsed = time.time() - start_time

    # Evalúa en conjunto de prueba
    _notify("[Evaluando en conjunto de prueba...]")
    X_test, Y_test = load_mnist_test(verbose=False)
    test_accuracy, test_loss = network.evaluate(X_test[:1000], Y_test[:1000])
    _notify(f"[Precisión en test: {test_accuracy:.2f}%]")

    return {
        "accuracies": history["accuracies"],
        "losses": history["losses"],
        "partition_accuracies": history["partition_accuracies"],
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "training_time": elapsed,
        "random_seed": random_seed,
        "final_accuracy": history["accuracies"][-1] if history["accuracies"] else 0.0,
    }


def run_multiple_experiments(
    num_partitions: int = 2,
    num_epochs: int = 5,
    num_experiments: int = 5,
    hidden_neurons: int = 30,
    learning_rate: float = 1.0,
    n_train: int = 5000,
    verbose: bool = True,
    on_progress: Callable | None = None,
) -> Dict[str, Any]:
    """
    Ejecuta múltiples experimentos con distintas semillas aleatorias.

    Cada experimento usa una semilla diferente para que los resultados
    reflejen la variabilidad real del algoritmo ante distintas
    inicializaciones y particionamientos.

    :param num_partitions: Número de particiones
    :type num_partitions: int

    :param num_epochs: Número de épocas por experimento
    :type num_epochs: int

    :param num_experiments: Número de experimentos a ejecutar
    :type num_experiments: int

    :param hidden_neurons: Neuronas en la capa oculta
    :type hidden_neurons: int

    :param learning_rate: Tasa de aprendizaje
    :type learning_rate: float

    :param n_train: Número de ejemplos de entrenamiento
    :type n_train: int

    :param verbose: Si True, muestra el progreso en consola
    :type verbose: bool

    :param on_progress: Callback para reportar progreso a la GUI.
                        Firma: ``on_progress(mensaje: str) → None``
    :type on_progress: Callable | None

    :return: Diccionario con timestamp, parameters, all_histories,
             final_mean_accuracy, final_std_accuracy, test_mean_accuracy
             y test_accuracies.
    :rtype: Dict[str, Any]
    """

    def _notify(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)
        if verbose:
            print(msg)

    _notify("=" * 70)
    _notify(f"EJECUTANDO {num_experiments} EXPERIMENTOS")
    _notify(
        f"  Particiones: {num_partitions} | Épocas: {num_epochs} | "
        f"Neuronas: {hidden_neurons} | LR: {learning_rate} | N: {n_train}"
    )
    _notify("=" * 70)

    all_histories = []
    test_accuracies = []

    for exp_idx in range(num_experiments):
        _notify(f"EXPERIMENTO {exp_idx + 1}/{num_experiments}")

        # Semilla única para cada experiment
        seed = random.randint(0, 1_000_000)

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
    variance = sum((x - mean_accuracy) ** 2 for x in final_accuracies) / len(
        final_accuracies
    )
    std_accuracy = math.sqrt(variance)
    mean_test = sum(test_accuracies) / len(test_accuracies)

    _notify(f"Precisión final promedio: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
    _notify(f"Precisión en test promedio: {mean_test:.2f}%")

    return {
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


def compare_configurations(
    configurations: List[Dict[str, Any]],
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Compara múltiples configuraciones de hiperparámetros.

    Ejecuta run_multiple_experiments para cada configuración y devuelve
    todos los resultados en una lista, en el mismo orden.

    :param configurations: Lista de diccionarios de parámetros compatibles
                           con run_multiple_experiments
    :type configurations: List[Dict[str, Any]]

    :param verbose: Si True, muestra el progreso en consola
    :type verbose: bool

    :return: Lista de resultados, uno por configuración
    :rtype: List[Dict[str, Any]]
    """
    results = []
    for idx, config in enumerate(configurations):
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"CONFIGURACIÓN {idx + 1}/{len(configurations)}: {config}")
            print("=" * 70)
        results.append(run_multiple_experiments(**config, verbose=verbose))
    return results
