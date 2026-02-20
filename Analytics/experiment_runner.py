"""
Analytics/experiment_runner.py
Orquestador de experimentos de Algoritmo de Diego.
"""

import numpy as np
import time
from datetime import datetime
from typing import Any, Callable, Dict, List

from Utils.mnist_loader import load_mnist_train, load_mnist_test
from Utils.data_partitioner import partition_mnist_data_simple
from Networks.nn_diego import DiegoNeuronalNetwork

# ================================================================
# EJECUCIÓN DE UN ÚNICO EXPERIMENTO
# ================================================================

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
    Ejecuta un único experimento del Algoritmo de Diego.

    Flujo del experimento:
        1. Fijar semilla (si se proporciona).
        2. Cargar subconjunto de MNIST.
        3. Crear particiones estratificadas.
        4. Inicializar red neuronal.
        5. Entrenar en modo federado.
        6. Evaluar en conjunto de prueba.
        7. Retornar métricas y metadatos.

    :param num_partitions: Número de particiones del dataset.
    :type num_partitions: int

    :param num_epochs: Número de épocas de entrenamiento.
    :type num_epochs: int

    :param hidden_neurons: Número de neuronas en capa oculta.
    :type hidden_neurons: int

    :param learning_rate: Tasa de aprendizaje.
    :type learning_rate: float

    :param n_train: Tamaño del subconjunto de entrenamiento.
    :type n_train: int

    :param random_seed: Semilla para reproducibilidad.
    :type random_seed: int | None

    :param verbose: Activa impresión en consola.
    :type verbose: bool

    :param on_progress: Callback opcional para UI.
    :type on_progress: Callable | None

    :return: Diccionario con historial, métricas y metadatos.
    :rtype: Dict[str, Any]
    """

    # Función interna de notificación (CLI / UI)
    def _notify(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)
        if verbose:
            print(msg)

    # Reproducibilidad
    if random_seed is not None:
        np.random.seed(random_seed)

    # Carga de datos
    _notify("[Cargando datos MNIST...]")
    X_train, Y_train = load_mnist_train(
        n_train=n_train, download_if_missing=True, verbose=False
    )

    # Particionado federado
    _notify(f"[Creando {num_partitions} particiones estratificadas...]")
    partitions = partition_mnist_data_simple(
        num_partitions=num_partitions,
        X_train=X_train,
        Y_train=Y_train,
        random_seed=random_seed,
    )

    # Inicializa la red
    _notify(f"[Inicializando red: 784 → {hidden_neurons} → 10]")
    network = DiegoNeuronalNetwork(
        input_size=784,
        hidden_size=hidden_neurons,
        output_size=10,
        random_seed=random_seed,
    )

    # Entrenamiento
    start_time = time.time()

    def _on_epoch_end(epoch: int, total: int, accuracy: float, loss: float) -> None:
        _notify(
            f"[Época {epoch}/{total}] — Precisión: {accuracy:.2f}%  Loss: {loss:.4f}]"
        )

    history = network.train_federated(
        partitions=partitions,
        epochs=num_epochs,
        learning_rate=learning_rate,
        verbose=verbose,
        on_epoch_end=_on_epoch_end,
    )

    elapsed = time.time() - start_time

    # Evaluación final
    _notify("[Evaluando en conjunto de prueba...]")
    X_test, Y_test = load_mnist_test(verbose=False)

    # Se evalúa sobre subconjunto fijo para consistencia temporal
    test_accuracy, test_loss = network.evaluate(X_test[:1000], Y_test[:1000])
    _notify(f"[Precisión en test: {test_accuracy:.2f}%]")

    # Retorno estructurado
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

# ================================================================
# EJECUCIÓN DE MÚLTIPLES EXPERIMENTOS
# ================================================================

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

    Objetivo:
        Medir estabilidad, variabilidad y robustez del modelo.

    Cada experimento usa una semilla distinta generada
    aleatoriamente.

    :return: Diccionario agregando resultados y métricas globales.
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

    # Bucle principal experimental
    for exp_idx in range(num_experiments):
        _notify(f"EXPERIMENTO {exp_idx + 1}/{num_experiments}")
        seed = np.random.randint(0, 1_000_000)

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

    # Estadísticas agregadas
    final_accs = np.array([h["final_accuracy"] for h in all_histories])
    test_accs = np.array(test_accuracies)

    _notify(
        f"Precisión final promedio: {np.mean(final_accs):.2f}% ± {np.std(final_accs):.2f}%"
    )
    _notify(f"Precisión en test promedio: {np.mean(test_accs):.2f}%")

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
        "final_mean_accuracy": float(np.mean(final_accs)),
        "final_std_accuracy": float(np.std(final_accs)),
        "test_mean_accuracy": float(np.mean(test_accs)),
        "test_accuracies": test_accs.tolist(),
    }

# ================================================================
# COMPARACIÓN DE CONFIGURACIONES
# ================================================================

def compare_configurations(
    configurations: List[Dict[str, Any]],
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Compara múltiples configuraciones de hiperparámetros.

    Cada configuración debe ser un diccionario compatible con
    `run_multiple_experiments`.

    Uso típico:
        - Comparar distintos learning rates
        - Comparar número de particiones
        - Comparar tamaño de capa oculta

    :param configurations: Lista de diccionarios de configuración.
    :type configurations: List[Dict[str, Any]]

    :param verbose: Activa impresión en consola.
    :type verbose: bool

    :return: Lista de resultados agregados.
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
