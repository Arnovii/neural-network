"""
Este módulo prepara datos en formatos listos para graficar,
separando la lógica matemática de la presentación visual.
"""

import os
import sys
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from statistics_engine import (
    compute_epoch_statistics,
    compute_partition_statistics,
    compute_convergence_epoch,
)


def prepare_accuracy_chart_data(
    histories: List[Dict[str, Any]], include_confidence_band: bool = True
) -> Dict[str, Any]:
    """
    Prepara datos para gráfica de evolución de precisión.

    :param histories: Lista de historiales de entrenamiento
    :type histories: List[Dict[str, Any]]

    :param include_confidence_band: Si True, incluye bandas de confianza
    :type include_confidence_band: bool

    :return: Diccionario con datos listos para graficar
    :rtype: Dict[str, Any]
    """
    stats = compute_epoch_statistics(histories)
    epochs = list(range(1, len(stats["mean"]) + 1))

    data = {
        "x": epochs,
        "y_mean": stats["mean"],
        "y_std": stats["std"],
        "y_min": stats["min"],
        "y_max": stats["max"],
        "title": "Evolución de Precisión",
        "xlabel": "Época",
        "ylabel": "Precisión (%)",
    }

    if include_confidence_band:
        # Calcula bandas de confianza (±1 std)
        data["y_upper"] = [m + s for m, s in zip(stats["mean"], stats["std"])]
        data["y_lower"] = [max(0, m - s) for m, s in zip(stats["mean"], stats["std"])]

    return data


def prepare_partition_comparison_data(
    histories: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Prepara datos para comparación entre particiones.

    :param histories: Lista de historiales con 'partition_accuracies'
    :type histories: List[Dict[str, Any]]

    :return: Diccionario con datos por partición
    :rtype: Dict[str, Any]
    """
    p_stats = compute_partition_statistics(histories)

    if not p_stats:
        return {}

    data = {
        "partitions": [],
        "title": "Comparación por Partición",
        "xlabel": "Época",
        "ylabel": "Precisión (%)",
    }

    for p_idx, epoch_stats in enumerate(p_stats["by_partition"]):
        partition_data = {
            "id": p_idx + 1,
            "x": list(range(1, len(epoch_stats) + 1)),
            "y": [e["mean"] for e in epoch_stats],
            "std": [e["std"] for e in epoch_stats],
        }
        data["partitions"].append(partition_data)

    return data


def prepare_convergence_data(histories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Prepara datos para análisis de convergencia.

    :param histories: Lista de historiales de entrenamiento
    :type histories: List[Dict[str, Any]]

    :return: Diccionario con métricas de convergencia
    :rtype: Dict[str, Any]
    """
    convergence_points = []
    all_improvements = []

    for h in histories:
        acc = h["accuracies"]
        conv_epoch = compute_convergence_epoch(acc)
        convergence_points.append(conv_epoch + 1 if conv_epoch >= 0 else len(acc))

        # Calcula mejoras por época
        improvements = [acc[i] - acc[i - 1] for i in range(1, len(acc))]
        all_improvements.append(improvements)

    # Promedia mejoras
    num_epochs = len(all_improvements[0])
    mean_improvements = []
    for epoch in range(num_epochs):
        epoch_imps = [imp[epoch] for imp in all_improvements]
        mean_improvements.append(sum(epoch_imps) / len(epoch_imps))

    return {
        "x": list(range(2, num_epochs + 2)),  # Épocas 2 en adelante
        "y": mean_improvements,
        "convergence_epochs": convergence_points,
        "mean_convergence": sum(convergence_points) / len(convergence_points),
        "title": "Mejora por Época",
        "xlabel": "Época",
        "ylabel": "Mejora (%)",
    }


def prepare_distribution_data(histories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Prepara datos para distribución de precisiones finales.

    :param histories: Lista de historiales de entrenamiento
    :type histories: List[Dict[str, Any]]

    :return: Diccionario con datos de distribución
    :rtype: Dict[str, Any]
    """
    final_accuracies = [h["accuracies"][-1] for h in histories]

    # Calcula histograma
    min_acc = min(final_accuracies)
    max_acc = max(final_accuracies)
    num_bins = min(10, len(final_accuracies))
    bin_width = (max_acc - min_acc) / num_bins if max_acc > min_acc else 1

    bins = [min_acc + i * bin_width for i in range(num_bins + 1)]
    counts = [0] * num_bins

    for acc in final_accuracies:
        bin_idx = min(int((acc - min_acc) / bin_width), num_bins - 1)
        counts[bin_idx] += 1

    return {
        "values": final_accuracies,
        "bins": bins,
        "counts": counts,
        "mean": sum(final_accuracies) / len(final_accuracies),
        "std": compute_std(final_accuracies),
        "min": min_acc,
        "max": max_acc,
        "title": "Distribución de Precisión Final",
        "xlabel": "Precisión (%)",
        "ylabel": "Frecuencia",
    }


def prepare_comparison_chart_data(
    all_results: List[Dict[str, Any]], config_labels: List[str] | None = None
) -> Dict[str, Any]:
    """
    Prepara datos para comparar múltiples configuraciones.

    :param all_results: Lista de resultados de diferentes configuraciones
    :type all_results: List[Dict[str, Any]]

    :param config_labels: Etiquetas para cada configuración
    :type config_labels: List[str] | None

    :return: Diccionario con datos comparativos
    :rtype: Dict[str, Any]
    """
    if config_labels is None:
        config_labels = [f"Config {i + 1}" for i in range(len(all_results))]

    data = {
        "configurations": [],
        "title": "Comparación de Configuraciones",
        "xlabel": "Época",
        "ylabel": "Precisión (%)",
    }

    for result, label in zip(all_results, config_labels):
        stats = compute_epoch_statistics(result["all_histories"])
        config_data = {
            "label": label,
            "x": list(range(1, len(stats["mean"]) + 1)),
            "y": stats["mean"],
            "std": stats["std"],
            "final_accuracy": result.get("final_mean_accuracy", 0),
            "final_std": result.get("final_std_accuracy", 0),
        }
        data["configurations"].append(config_data)

    return data


def compute_std(values: List[float]) -> float:
    """Calcula desviación estándar."""
    import math

    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def export_to_csv(histories: List[Dict[str, Any]], filename: str) -> None:
    """
    Exporta historiales a archivo CSV.

    :param histories: Lista de historiales
    :type histories: List[Dict[str, Any]]

    :param filename: Nombre del archivo de salida
    :type filename: str
    """
    with open(filename, "w") as f:
        # Header
        f.write("experiment,epoch,accuracy\n")

        # Datos
        for exp_idx, h in enumerate(histories):
            for epoch, acc in enumerate(h["accuracies"]):
                f.write(f"{exp_idx + 1},{epoch + 1},{acc:.4f}\n")


# =================
# PRUEBAS
# =================


def _test_chart_generator():
    """Pruebas del generador de gráficos."""
    print("=" * 60)
    print("PRUEBAS DE CHART_GENERATOR")
    print("=" * 60)

    # Datos de prueba
    histories = [
        {
            "accuracies": [50.0, 60.0, 70.0, 75.0, 80.0],
            "partition_accuracies": [[48, 52], [58, 62], [68, 72], [73, 77], [78, 82]],
        },
        {
            "accuracies": [52.0, 62.0, 72.0, 76.0, 82.0],
            "partition_accuracies": [[50, 54], [60, 64], [70, 74], [74, 78], [80, 84]],
        },
        {
            "accuracies": [48.0, 58.0, 68.0, 74.0, 78.0],
            "partition_accuracies": [[46, 50], [56, 60], [66, 70], [72, 76], [76, 80]],
        },
    ]

    print("\n1. Datos para gráfica de precisión:")
    acc_data = prepare_accuracy_chart_data(histories)
    print(f"   Épocas: {acc_data['x']}")
    print(f"   Medias: {[f'{m:.1f}' for m in acc_data['y_mean']]}")

    print("\n2. Datos para comparación de particiones:")
    part_data = prepare_partition_comparison_data(histories)
    print(f"   Número de particiones: {len(part_data['partitions'])}")

    print("\n3. Datos de convergencia:")
    conv_data = prepare_convergence_data(histories)
    print(f"   Convergencia promedio: época {conv_data['mean_convergence']:.1f}")

    print("\n4. Datos de distribución:")
    dist_data = prepare_distribution_data(histories)
    print(f"   Media final: {dist_data['mean']:.2f}%")
    print(f"   Desviación: {dist_data['std']:.2f}%")

    print("\n" + "=" * 60)
    print("PRUEBAS COMPLETADAS")
    print("=" * 60)


if __name__ == "__main__":
    _test_chart_generator()
