"""
Analytics/chart_generator.py
Preparación de datos para visualización.

Transforma los historiales de entrenamiento en estructuras listas para
graficar, separando el cálculo estadístico (statistics_engine) del
renderizado visual (main.py).

Ninguna función de este módulo llama a matplotlib directamente.
"""

from typing import Any, Dict, List

from Analytics.statistics_engine import (
    compute_convergence_epoch,
    compute_epoch_statistics,
    compute_partition_statistics,
    compute_std,
)


def prepare_accuracy_chart_data(
    histories: List[Dict[str, Any]],
    include_confidence_band: bool = True,
) -> Dict[str, Any]:
    """
    Prepara datos para la curva de evolución de precisión.

    :param histories: Lista de historiales de entrenamiento
    :type histories: List[Dict[str, Any]]

    :param include_confidence_band: Si True, incluye bandas ±1 desv. estándar
    :type include_confidence_band: bool

    :return: Diccionario con claves x, y_mean, y_std, y_min, y_max,
             y_upper, y_lower (si include_confidence_band), title, xlabel, ylabel
    :rtype: Dict[str, Any]
    """
    stats = compute_epoch_statistics(histories)
    epochs = list(range(1, len(stats["mean"]) + 1))

    data: Dict[str, Any] = {
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
        data["y_lower"] = [max(0.0, m - s) for m, s in zip(stats["mean"], stats["std"])]

    return data


def prepare_partition_comparison_data(
    histories: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Prepara datos para comparar la evolución de precisión por partición.

    :param histories: Lista de historiales con 'partition_accuracies'
    :type histories: List[Dict[str, Any]]

    :return: Diccionario con clave 'partitions' (lista de datos por partición),
             title, xlabel, ylabel. Vacío si no hay datos de partición.
    :rtype: Dict[str, Any]
    """
    p_stats = compute_partition_statistics(histories)
    if not p_stats:
        return {}

    data: Dict[str, Any] = {
        "partitions": [],
        "title": "Comparación por Partición",
        "xlabel": "Época",
        "ylabel": "Precisión (%)",
    }

    for p_idx, epoch_stats in enumerate(p_stats["by_partition"]):
        data["partitions"].append(
            {
                "id": p_idx + 1,
                "x": list(range(1, len(epoch_stats) + 1)),
                "y": [e["mean"] for e in epoch_stats],
                "std": [e["std"] for e in epoch_stats],
            }
        )

    return data


def prepare_convergence_data(histories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Prepara datos para el análisis de convergencia (mejora por época).

    :param histories: Lista de historiales de entrenamiento
    :type histories: List[Dict[str, Any]]

    :return: Diccionario con las mejoras promedio por época (x, y),
             los puntos de convergencia individuales, la convergencia
             media y las etiquetas del gráfico.
    :rtype: Dict[str, Any]
    """
    convergence_points = []
    all_improvements = []

    for h in histories:
        acc = h["accuracies"]
        conv = compute_convergence_epoch(acc)
        convergence_points.append(conv + 1 if conv >= 0 else len(acc))

        # Calcula mejoras por época
        all_improvements.append([acc[i] - acc[i - 1] for i in range(1, len(acc))])

    num_epochs = len(all_improvements[0])
    mean_improvements = [
        sum(imp[epoch] for imp in all_improvements) / len(all_improvements)
        for epoch in range(num_epochs)
    ]

    return {
        "x": list(range(2, num_epochs + 2)),
        "y": mean_improvements,
        "convergence_epochs": convergence_points,
        "mean_convergence": sum(convergence_points) / len(convergence_points),
        "title": "Mejora por Época",
        "xlabel": "Época",
        "ylabel": "Mejora (%)",
    }


def prepare_distribution_data(histories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Prepara datos para el histograma de precisiones finales.

    Calcula el histograma manualmente para no depender de matplotlib
    en la capa de datos.

    :param histories: Lista de historiales de entrenamiento
    :type histories: List[Dict[str, Any]]

    :return: Diccionario con los valores crudos, bins, conteos, media,
             desviación estándar, mínimo, máximo y etiquetas del gráfico.
    :rtype: Dict[str, Any]
    """
    final_accuracies = [h["accuracies"][-1] for h in histories]

    # Calcula histograma
    min_acc = min(final_accuracies)
    max_acc = max(final_accuracies)
    num_bins = min(10, len(final_accuracies))
    bin_width = (max_acc - min_acc) / num_bins if max_acc > min_acc else 1.0

    bins = [min_acc + i * bin_width for i in range(num_bins + 1)]
    counts = [0] * num_bins
    for acc in final_accuracies:
        idx = min(int((acc - min_acc) / bin_width), num_bins - 1)
        counts[idx] += 1

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
    all_results: List[Dict[str, Any]],
    config_labels: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Prepara datos para comparar múltiples configuraciones en un solo gráfico.

    :param all_results: Lista de resultados de run_multiple_experiments
    :type all_results: List[Dict[str, Any]]

    :param config_labels: Etiquetas para cada configuración. Si es None,
                          se usan "Config 1", "Config 2", etc.
    :type config_labels: List[str] | None

    :return: Diccionario con 'configurations' (lista de datos por config),
             title, xlabel, ylabel.
    :rtype: Dict[str, Any]
    """
    if config_labels is None:
        config_labels = [f"Config {i + 1}" for i in range(len(all_results))]

    data: Dict[str, Any] = {
        "configurations": [],
        "title": "Comparación de Configuraciones",
        "xlabel": "Época",
        "ylabel": "Precisión (%)",
    }

    for result, label in zip(all_results, config_labels):
        stats = compute_epoch_statistics(result["all_histories"])
        data["configurations"].append(
            {
                "label": label,
                "x": list(range(1, len(stats["mean"]) + 1)),
                "y": stats["mean"],
                "std": stats["std"],
                "final_accuracy": result.get("final_mean_accuracy", 0.0),
                "final_std": result.get("final_std_accuracy", 0.0),
            }
        )

    return data


def export_to_csv(histories: List[Dict[str, Any]], filename: str) -> None:
    """
    Exporta los historiales de precisión a un archivo CSV.

    Formato de salida: experiment,epoch,accuracy

    :param histories: Lista de historiales de entrenamiento
    :type histories: List[Dict[str, Any]]

    :param filename: Ruta del archivo de salida
    :type filename: str
    """
    with open(filename, "w") as f:
        f.write("experiment,epoch,accuracy\n")
        for exp_idx, h in enumerate(histories):
            for epoch, acc in enumerate(h["accuracies"]):
                f.write(f"{exp_idx + 1},{epoch + 1},{acc:.4f}\n")
