"""
Ejecutor de experimentos para Federated Learning.

Gestiona la ejecución de múltiples entrenamientos con diferentes
configuraciones y semillas aleatorias.
"""

import json
import random
import time
from datetime import datetime
from typing import List, Dict, Any

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
) -> Dict[str, Any]:
    """
    Ejecuta un único experimento de Federated Learning.

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

    :return: Historial del experimento con métricas
    :rtype: Dict[str, Any]
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Carga datos
    if verbose:
        print("\n[Cargando datos...]")
    X_train, Y_train = load_mnist_train(
        n_train=n_train, download_if_missing=True, verbose=False
    )

    # Crea particiones
    partitions = partition_mnist_data_simple(
        num_partitions=num_partitions,
        X_train=X_train,
        Y_train=Y_train,
        random_seed=random_seed,
        verbose=False,
    )

    # Crea y entrena la red
    if verbose:
        print(f"[Creando red: 784 → {hidden_neurons} → 10]")

    network = DiegoNeuronalNetwork(
        input_size=784,
        hidden_size=hidden_neurons,
        output_size=10,
        random_seed=random_seed,
    )

    start_time = time.time()

    history = network.train_federated(
        partitions=partitions,
        epochs=num_epochs,
        learning_rate=learning_rate,
        verbose=verbose,
    )

    elapsed_time = time.time() - start_time

    # Evalua en conjunto de prueba
    if verbose:
        print("\n[Evaluando en conjunto de prueba...]")
    X_test, Y_test = load_mnist_test(verbose=False)
    test_accuracy, test_loss = network.evaluate(X_test[:1000], Y_test[:1000])

    if verbose:
        print(f"[Precisión en test: {test_accuracy:.2f}%]")

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

    :return: Resultados agregados de todos los experimentos
    :rtype: Dict[str, Any]
    """
    if verbose:
        print("=" * 70)
        print(f"EJECUTANDO {num_experiments} EXPERIMENTOS")
        print("=" * 70)
        print("Configuración:")
        print(f"  Particiones: {num_partitions}")
        print(f"  Épocas: {num_epochs}")
        print(f"  Neuronas ocultas: {hidden_neurons}")
        print(f"  Tasa de aprendizaje: {learning_rate}")
        print(f"  Ejemplos: {n_train}")
        print("=" * 70)

    all_histories = []
    test_accuracies = []

    for exp_idx in range(num_experiments):
        if verbose:
            print(f"\n{'-' * 70}")
            print(f"EXPERIMENTO {exp_idx + 1}/{num_experiments}")
            print(f"{'-' * 70}")

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
        )

        all_histories.append(result)
        test_accuracies.append(result["test_accuracy"])

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

    if verbose:
        print(f"\n{'=' * 70}")
        print("RESUMEN DE RESULTADOS")
        print(f"{'=' * 70}")
        print(f"Precisión final promedio: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
        print(f"Precisión en test promedio: {mean_test:.2f}%")
        print(f"{'=' * 70}")

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
        filename = f"results/experiment_{results['timestamp']}.json"

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
