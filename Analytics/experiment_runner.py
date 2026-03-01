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
from Parallel.core_validator import validate_partition_count

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
    parallel: bool = False,
    on_progress: Callable | None = None,
) -> Dict[str, Any]:
    """
    Ejecuta un único experimento del Algoritmo de Diego.

    Flujo del experimento:
        1. Validar particiones vs núcleos físicos (solo si parallel=True).
        2. Fijar semilla (si se proporciona).
        3. Cargar subconjunto de MNIST.
        4. Crear particiones estratificadas.
        5. Inicializar red neuronal.
        6. Entrenar con el algoritmo de Diego (secuencial o paralelo).
        7. Retornar métricas, metadatos y tiempo de entrenamiento.

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

    :param parallel: Si True usa multiprocessing (1 proceso por partición).
                     El número de particiones no puede superar los núcleos
                     físicos de la CPU.
    :type parallel: bool

    :param on_progress: Callback opcional para UI.
    :type on_progress: Callable | None

    :return: Diccionario con historial, métricas, metadatos y training_time.
    :rtype: Dict[str, Any]
    """
    # Solo validamos la regla de oro cuando se usa multiprocessing.
    # En modo secuencial el número de particiones es libre.
    if parallel:
        validate_partition_count(num_partitions)

    # Función interna de notificación: imprime en consola o pasa a UI
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

    # Particionado de datos de entrenamiento
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

    # Carga los datos de test antes del entrenamiento para pasarlos
    # a train_diego, que evaluará sobre ellos al final de cada época.
    # Así history["accuracies"] refleja generalización real, no training accuracy.
    _notify("[Cargando datos de prueba...]")
    X_test, Y_test = load_mnist_test(verbose=False)

    # Callback que se llama al final de cada época, para mostrar métricas.
    def _on_epoch_end(epoch: int, total: int, accuracy: float, loss: float) -> None:
        _notify(
            f"[Época {epoch}/{total}] — Precisión: {accuracy:.2f}%  Loss: {loss:.4f}]"
        )

    mode_label = "paralelo" if parallel else "secuencial"

    # Entrena la red con las particiones
    _notify(f"[Iniciando entrenamiento {mode_label}...]")

    start_time = time.perf_counter()
    history = network.train_diego(
        partitions=partitions,
        epochs=num_epochs,
        learning_rate=learning_rate,
        X_test=X_test,
        Y_test=Y_test,
        verbose=verbose,
        on_epoch_end=_on_epoch_end,
        parallel=parallel,
    )
    elapsed = time.perf_counter() - start_time

    # La precisión final en test es el último valor de la serie por época,
    # ya registrada en history["accuracies"] por train_diego
    test_accuracy = history["accuracies"][-1] if history["accuracies"] else 0.0
    test_loss = history["losses"][-1] if history["losses"] else 0.0
    _notify(f"[Precisión en test: {test_accuracy:.2f}% | Tiempo: {elapsed:.2f}s]")

    # Retorno estructurado
    return {
        "accuracies": history["accuracies"],
        "losses": history["losses"],
        "partition_accuracies": history["partition_accuracies"],
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "training_time": elapsed,
        "parallel": parallel,
        "random_seed": random_seed,
        "final_accuracy": test_accuracy,
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
    parallel: bool = False,
    seeds: List[int] | None = None,
    on_progress: Callable | None = None,
) -> Dict[str, Any]:
    """
    Ejecuta múltiples experimentos con distintas semillas aleatorias.

    Objetivo:
        Medir estabilidad, variabilidad y robustez del modelo.
        Al registrar el tiempo por experimento permite comparar
        el rendimiento entre modo secuencial y paralelo.

    :param parallel: Si True cada experimento usa multiprocessing.
    :type parallel: bool

    :param seeds: Lista de semillas a usar, una por experimento.
                  Si se proporciona, su longitud debe coincidir con
                  ``num_experiments``. Si es None, las semillas se
                  generan aleatoriamente en cada llamada.
                  Pasar la misma lista a la ronda secuencial y a la
                  paralela garantiza que ambas parten de exactamente
                  la misma red inicial, haciendo el benchmark justo.
    :type seeds: List[int] | None

    :return: Diccionario con resultados, métricas y bloque benchmark.
    :rtype: Dict[str, Any]
    """

    def _notify(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)
        if verbose:
            print(msg)

    mode_label = "PARALELO" if parallel else "SECUENCIAL"

    # Encabezado informativo
    _notify("=" * 70)
    _notify(f"EJECUTANDO {num_experiments} EXPERIMENTOS  [{mode_label}]")
    _notify(
        f"  Particiones: {num_partitions} | Épocas: {num_epochs} | "
        f"Neuronas: {hidden_neurons} | LR: {learning_rate} | N: {n_train}"
    )
    _notify("=" * 70)

    all_histories = []
    test_accuracies = []
    training_times = []

    # Ejecuta cada experimento independiente
    for exp_idx in range(num_experiments):
        _notify(f"EXPERIMENTO {exp_idx + 1}/{num_experiments}")

        # Usa la semilla pre-fijada si se proporcionó, o genera una nueva.
        seed = (
            seeds[exp_idx]
            if seeds is not None
            else int(np.random.randint(0, 1_000_000))
        )

        result = run_single_experiment(
            num_partitions=num_partitions,
            num_epochs=num_epochs,
            hidden_neurons=hidden_neurons,
            learning_rate=learning_rate,
            n_train=n_train,
            random_seed=seed,
            verbose=verbose,
            parallel=parallel,
            on_progress=on_progress,
        )

        all_histories.append(result)
        test_accuracies.append(result["test_accuracy"])
        training_times.append(result["training_time"])
        _notify(
            f"  ✓ Completado — Precisión: {result['final_accuracy']:.2f}%"
            f"  |  Tiempo: {result['training_time']:.2f}s"
        )

    # Estadísticas agregadas
    final_accs = np.array([h["final_accuracy"] for h in all_histories])
    test_accs = np.array(test_accuracies)
    times = np.array(training_times)

    _notify(
        f"Precisión final promedio: {np.mean(final_accs):.2f}% ± {np.std(final_accs):.2f}%"
    )
    _notify(
        f"Tiempo promedio por experimento: {np.mean(times):.2f}s ± {np.std(times):.2f}s"
    )

    # Benchmark: métricas de rendimiento comparables entre modos
    benchmark = {
        "mode": mode_label,
        "parallel": parallel,
        "times": times.tolist(),
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "min_time": float(np.min(times)),
        "max_time": float(np.max(times)),
        "total_time": float(np.sum(times)),
        # Throughput: épocas procesadas por segundo
        "throughput_epochs_per_sec": float(num_epochs / np.mean(times)),
    }

    return {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "parameters": {
            "num_partitions": num_partitions,
            "num_epochs": num_epochs,
            "num_experiments": num_experiments,
            "hidden_neurons": hidden_neurons,
            "learning_rate": learning_rate,
            "n_train": n_train,
            "parallel": parallel,
        },
        "all_histories": all_histories,
        "final_mean_accuracy": float(np.mean(final_accs)),
        "final_std_accuracy": float(np.std(final_accs)),
        "test_mean_accuracy": float(np.mean(test_accs)),
        "test_accuracies": test_accs.tolist(),
        "benchmark": benchmark,
    }


# ================================================================
# COMPARACIÓN SECUENCIAL VS PARALELO
# ================================================================


def run_benchmark_comparison(
    num_partitions: int = 2,
    num_epochs: int = 5,
    num_experiments: int = 3,
    hidden_neurons: int = 30,
    learning_rate: float = 1.0,
    n_train: int = 5000,
    verbose: bool = True,
    on_progress: Callable | None = None,
) -> Dict[str, Any]:
    """
    Ejecuta el mismo experimento en ambos modos y compara tiempos.

    Propósito pedagógico: produce evidencia cuantitativa sobre el
    impacto real del paralelismo en el algoritmo de Diego, incluyendo
    speedup, overhead de procesos y eficiencia por núcleo.

    Se usan los mismos parámetros en ambas rondas para que la
    comparación sea justa.

    :param num_partitions: Particiones (y procesos en modo paralelo).
    :type num_partitions: int

    :param num_experiments: Experimentos a ejecutar por modo.
    :type num_experiments: int

    :return: Diccionario con resultados de ambos modos y análisis comparativo.
    :rtype: Dict[str, Any]
    """

    def _notify(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)
        if verbose:
            print(msg)

    base_params: Dict[str, Any] = {
        "num_partitions": num_partitions,
        "num_epochs": num_epochs,
        "num_experiments": num_experiments,
        "hidden_neurons": hidden_neurons,
        "learning_rate": learning_rate,
        "n_train": n_train,
        "verbose": verbose,
        "on_progress": on_progress,
    }

    _notify("\n" + "=" * 70)
    _notify("BENCHMARK: SECUENCIAL vs PARALELO")
    _notify("=" * 70)

    # Genera las semillas UNA SOLA VEZ y las comparte entre ambas fases.
    # Así el experimento i secuencial y el experimento i paralelo parten
    # de exactamente la misma inicialización de pesos y los mismos datos,
    # haciendo que la única variable sea el modo de entrenamiento.
    shared_seeds = [
        int(np.random.randint(0, 1_000_000)) for _ in range(num_experiments)
    ]
    _notify(f"  Semillas compartidas: {shared_seeds}")

    _notify("\n▶ Fase 1/2 — Modo SECUENCIAL")
    seq_results = run_multiple_experiments(
        **base_params, parallel=False, seeds=shared_seeds
    )

    _notify("\n▶ Fase 2/2 — Modo PARALELO")
    par_results = run_multiple_experiments(
        **base_params, parallel=True, seeds=shared_seeds
    )

    seq_bm = seq_results["benchmark"]
    par_bm = par_results["benchmark"]

    # Speedup = tiempo_secuencial / tiempo_paralelo
    speedup = (
        seq_bm["mean_time"] / par_bm["mean_time"] if par_bm["mean_time"] > 0 else 0.0
    )

    # Eficiencia: fracción del speedup teórico máximo que se logró.
    # El speedup teórico ideal sería igual al número de particiones
    # (cada una en un núcleo dedicado sin overhead).
    efficiency = (speedup / num_partitions * 100) if num_partitions > 0 else 0.0

    # Overhead = tiempo paralelo menos el tiempo secuencial dividido entre
    # los núcleos. Cuantifica el costo de fork, pickle y sincronización.
    overhead_sec = par_bm["mean_time"] - (seq_bm["mean_time"] / num_partitions)

    comparison = {
        "speedup": round(speedup, 3),
        "efficiency_pct": round(efficiency, 1),
        "overhead_sec": round(overhead_sec, 3),
        "seq_mean_time": seq_bm["mean_time"],
        "par_mean_time": par_bm["mean_time"],
        "seq_mean_accuracy": seq_results["final_mean_accuracy"],
        "par_mean_accuracy": par_results["final_mean_accuracy"],
        "accuracy_delta": round(
            par_results["final_mean_accuracy"] - seq_results["final_mean_accuracy"], 3
        ),
    }

    _notify("\n" + "=" * 70)
    _notify("RESULTADOS DEL BENCHMARK")
    _notify("=" * 70)
    _notify(f"  Tiempo medio SECUENCIAL : {seq_bm['mean_time']:.2f}s")
    _notify(f"  Tiempo medio PARALELO   : {par_bm['mean_time']:.2f}s")
    _notify(f"  Speedup obtenido        : {speedup:.2f}×")
    _notify(f"  Eficiencia por núcleo   : {efficiency:.1f}%")
    _notify(f"  Overhead de procesos    : {overhead_sec:.3f}s")
    _notify(f"  Δ Precisión (par - seq) : {comparison['accuracy_delta']:+.2f}%")
    _notify("=" * 70)

    return {
        "sequential": seq_results,
        "parallel": par_results,
        "comparison": comparison,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
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
    ``run_multiple_experiments``.

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
