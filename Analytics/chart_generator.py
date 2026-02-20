"""
Analytics/chart_generator.py
Preparación de datos para visualización — implementación NumPy.
"""

import numpy as np
from typing import Any, Dict, List

from Analytics.statistics_engine import (
    compute_convergence_epoch,
    compute_epoch_statistics,
    compute_partition_statistics,
)


def prepare_accuracy_chart_data(
    histories: List[Dict[str, Any]],
    include_confidence_band: bool = True,
) -> Dict[str, Any]:
    """Prepara datos para la curva de evolución de precisión."""
    stats = compute_epoch_statistics(histories)
    mean = np.array(stats["mean"])
    std = np.array(stats["std"])
    epochs = list(range(1, len(mean) + 1))

    data: Dict[str, Any] = {
        "x": epochs,
        "y_mean": mean.tolist(),
        "y_std": std.tolist(),
        "y_min": stats["min"],
        "y_max": stats["max"],
        "title": "Evolución de Precisión",
        "xlabel": "Época",
        "ylabel": "Precisión (%)",
    }

    if include_confidence_band:
        data["y_upper"] = (mean + std).tolist()
        data["y_lower"] = np.maximum(0.0, mean - std).tolist()

    return data


def prepare_partition_comparison_data(
    histories: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Prepara datos para comparar precisión por partición."""
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
    """Prepara datos para análisis de convergencia."""
    convergence_points = []
    all_improvements = []

    for h in histories:
        acc = np.array(h["accuracies"])
        conv = compute_convergence_epoch(h["accuracies"])
        convergence_points.append(conv + 1 if conv >= 0 else len(acc))
        all_improvements.append(np.diff(acc))

    imp_matrix = np.array(all_improvements)
    mean_improvements = np.mean(imp_matrix, axis=0).tolist()

    return {
        "x": list(range(2, len(mean_improvements) + 2)),
        "y": mean_improvements,
        "convergence_epochs": convergence_points,
        "mean_convergence": float(np.mean(convergence_points)),
        "title": "Mejora por Época",
        "xlabel": "Época",
        "ylabel": "Mejora (%)",
    }


def prepare_distribution_data(histories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Prepara datos para el histograma de precisiones finales."""
    final_accs = np.array([h["accuracies"][-1] for h in histories])
    num_bins = min(10, len(final_accs))
    counts, bin_edges = np.histogram(final_accs, bins=num_bins)

    return {
        "values": final_accs.tolist(),
        "bins": bin_edges.tolist(),
        "counts": counts.tolist(),
        "mean": float(np.mean(final_accs)),
        "std": float(np.std(final_accs, ddof=0)),
        "min": float(np.min(final_accs)),
        "max": float(np.max(final_accs)),
        "title": "Distribución de Precisión Final",
        "xlabel": "Precisión (%)",
        "ylabel": "Frecuencia",
    }


def prepare_comparison_chart_data(
    all_results: List[Dict[str, Any]],
    config_labels: List[str] | None = None,
) -> Dict[str, Any]:
    """Prepara datos para comparar múltiples configuraciones."""
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
    """Exporta historiales de precisión a CSV."""
    with open(filename, "w") as f:
        f.write("experiment,epoch,accuracy\n")
        for exp_idx, h in enumerate(histories):
            for epoch, acc in enumerate(h["accuracies"]):
                f.write(f"{exp_idx + 1},{epoch + 1},{acc:.4f}\n")
