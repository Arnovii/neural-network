"""
Proporciona funciones matemáticas puras para cálculo de estadísticas
sobre múltiples ejecuciones de entrenamiento.
"""

import math
from typing import List, Dict, Any, Tuple


def compute_epoch_statistics(histories: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Calcula estadísticas por época sobre múltiples experimentos.

    :param histories: Lista de historiales de entrenamiento, cada uno con 'accuracies'
    :type histories: List[Dict[str, Any]]

    :return: Diccionario con 'mean', 'std', 'min', 'max' por época
    :rtype: Dict[str, List[float]]
    """
    if not histories:
        return {"mean": [], "std": [], "min": [], "max": []}

    num_epochs = len(histories[0]["accuracies"])
    num_experiments = len(histories)

    means = []
    stds = []
    mins = []
    maxs = []

    for epoch in range(num_epochs):
        # Recolecta precisiones de todos los experimentos en esta época
        epoch_accuracies = [h["accuracies"][epoch] for h in histories]

        # Calcula estadísticas
        mean = sum(epoch_accuracies) / num_experiments
        variance = sum((x - mean) ** 2 for x in epoch_accuracies) / num_experiments
        std = math.sqrt(variance)
        min_val = min(epoch_accuracies)
        max_val = max(epoch_accuracies)

        means.append(mean)
        stds.append(std)
        mins.append(min_val)
        maxs.append(max_val)

    return {"mean": means, "std": stds, "min": mins, "max": maxs}


def compute_partition_statistics(histories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calcula estadísticas por partición sobre múltiples experimentos.

    :param histories: Lista de historiales con 'partition_accuracies'
    :type histories: List[Dict[str, Any]]

    :return: Estadísticas por partición
    :rtype: Dict[str, Any]
    """
    if not histories or not histories[0].get("partition_accuracies"):
        return {}

    num_partitions = len(histories[0]["partition_accuracies"][0])
    num_epochs = len(histories[0]["partition_accuracies"])

    # Estructura: stats[partition_idx][epoch] = {'mean', 'std', ...}
    partition_stats = []

    for p_idx in range(num_partitions):
        epoch_stats = []
        for epoch in range(num_epochs):
            # Recolecta precisiones de esta partición en esta época
            p_accuracies = [h["partition_accuracies"][epoch][p_idx] for h in histories]

            mean = sum(p_accuracies) / len(p_accuracies)
            variance = sum((x - mean) ** 2 for x in p_accuracies) / len(p_accuracies)

            epoch_stats.append(
                {
                    "mean": mean,
                    "std": math.sqrt(variance),
                    "min": min(p_accuracies),
                    "max": max(p_accuracies),
                }
            )
        partition_stats.append(epoch_stats)

    return {
        "by_partition": partition_stats,
        "num_partitions": num_partitions,
        "num_epochs": num_epochs,
    }


def aggregate_experiments(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Agrega resultados de múltiples configuraciones de experimentos.

    :param all_results: Lista de resultados de diferentes configuraciones
    :type all_results: List[Dict[str, Any]]

    :return: Diccionario con estadísticas agregadas
    :rtype: Dict[str, Any]
    """
    aggregated = {"configurations": [], "global_stats": {}}

    for result in all_results:
        config_summary = {
            "parameters": result.get("parameters", {}),
            "final_mean_accuracy": result.get("final_mean_accuracy", 0),
            "final_std_accuracy": result.get("final_std_accuracy", 0),
            "num_experiments": len(result.get("all_histories", [])),
        }
        aggregated["configurations"].append(config_summary)

    # Calcular estadísticas globales
    all_final_accuracies = [r["final_mean_accuracy"] for r in all_results]

    if all_final_accuracies:
        aggregated["global_stats"] = {
            "best_accuracy": max(all_final_accuracies),
            "worst_accuracy": min(all_final_accuracies),
            "mean_accuracy": sum(all_final_accuracies) / len(all_final_accuracies),
            "std_accuracy": compute_std(all_final_accuracies),
        }

    return aggregated


def compute_confidence_interval(
    values: List[float], confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calcula el intervalo de confianza para una lista de valores.

    :param values: Lista de valores
    :type values: List[float]

    :param confidence: Nivel de confianza (default 0.95 = 95%)
    :type confidence: float

    :return: Tupla (lower_bound, upper_bound)
    :rtype: Tuple[float, float]
    """
    if not values:
        return (0.0, 0.0)

    n = len(values)
    mean = sum(values) / n
    std = compute_std(values)

    # Z-scores para diferentes niveles de confianza
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)

    margin = z * (std / math.sqrt(n))

    return (mean - margin, mean + margin)


def compute_std(values: List[float]) -> float:
    """
    Calcula la desviación estándar de una lista de valores.

    :param values: Lista de valores numéricos
    :type values: List[float]

    :return: Desviación estándar
    :rtype: float

    """
    if len(values) < 2:
        return 0.0

    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def compute_percentile(values: List[float], percentile: float) -> float:
    """
    Calcula un percentil de una lista de valores.

    :param values: Lista de valores
    :type values: List[float]

    :param percentile: Percentil a calcular (0-100)
    :type percentile: float

    :return: Valor del percentil
    :rtype: float
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return sorted_values[int(k)]

    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def compute_convergence_epoch(
    accuracies: List[float], threshold: float = 0.01, window: int = 3
) -> int:
    """
    Determina en qué época el modelo converge.

    La convergencia se define como cuando la mejora promedio en una ventana\n
    de épocas es menor que el umbral.

    :param accuracies: Lista de precisiones por época
    :type accuracies: List[float]

    :param threshold: Umbral de mejora mínima
    :type threshold: float

    :param window: Tamaño de ventana para promediar
    :type window: int

    :return: Índice de la época de convergencia (0-based), o -1 si no converge
    :rtype: int
    """
    if len(accuracies) < window + 1:
        return -1

    for i in range(window, len(accuracies)):
        # Calcular mejora promedio en la ventana
        improvements = [
            accuracies[j] - accuracies[j - 1] for j in range(i - window + 1, i + 1)
        ]
        avg_improvement = sum(improvements) / len(improvements)

        if abs(avg_improvement) < threshold:
            return i

    return -1


def compute_stability_index(histories: List[Dict[str, Any]]) -> float:
    """
    Calcula un índice de estabilidad entre múltiples ejecuciones.

    Valores más bajos indican mayor estabilidad (menor variabilidad).

    :param histories: Lista de historiales de entrenamiento
    :type histories: List[Dict[str, Any]]

    :return: Índice de estabilidad (desviación estándar promedio)
    :rtype: float
    """
    if not histories:
        return 0.0

    stats = compute_epoch_statistics(histories)
    return sum(stats["std"]) / len(stats["std"])


# =================
# PRUEBAS
# =================


def _test_statistics_engine():
    """Pruebas básicas del motor de estadísticas."""
    print("=" * 60)
    print("PRUEBAS DE STATISTICS_ENGINE")
    print("=" * 60)

    # Datos de prueba simulados
    histories = [
        {"accuracies": [50.0, 60.0, 70.0, 75.0, 80.0]},
        {"accuracies": [52.0, 62.0, 72.0, 76.0, 82.0]},
        {"accuracies": [48.0, 58.0, 68.0, 74.0, 78.0]},
    ]

    print("\n1. Estadísticas por época:")
    stats = compute_epoch_statistics(histories)
    for epoch, (mean, std) in enumerate(zip(stats["mean"], stats["std"])):
        print(f"   Época {epoch + 1}: {mean:.2f}% ± {std:.2f}%")

    print("\n2. Intervalo de confianza (95%):")
    final_accs = [h["accuracies"][-1] for h in histories]
    lower, upper = compute_confidence_interval(final_accs)
    print(f"   [{lower:.2f}%, {upper:.2f}%]")

    print("\n3. Época de convergencia:")
    conv_epoch = compute_convergence_epoch(histories[0]["accuracies"])
    print(
        f"   Converge en época: {conv_epoch + 1 if conv_epoch >= 0 else 'No converge'}"
    )

    print("\n4. Índice de estabilidad:")
    stability = compute_stability_index(histories)
    print(f"   {stability:.4f}")

    print("\n" + "=" * 60)
    print("PRUEBAS COMPLETADAS")
    print("=" * 60)


if __name__ == "__main__":
    _test_statistics_engine()
