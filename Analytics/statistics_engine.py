"""
Analytics/statistics_engine.py
Funciones estadísticas para análisis de experimentos — implementación NumPy.
"""

import numpy as np
from typing import Any, Dict, List


def compute_std(values: List[float]) -> float:
    """Desviación estándar poblacional."""
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=0))


def compute_epoch_statistics(histories: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Estadísticas por época sobre múltiples experimentos.

    :return: Dict con listas 'mean', 'std', 'min', 'max'
    """
    if not histories:
        return {"mean": [], "std": [], "min": [], "max": []}

    # Matriz (num_experiments, num_epochs)
    acc_matrix = np.array([h["accuracies"] for h in histories])

    return {
        "mean": np.mean(acc_matrix, axis=0).tolist(),
        "std": np.std(acc_matrix, axis=0, ddof=0).tolist(),
        "min": np.min(acc_matrix, axis=0).tolist(),
        "max": np.max(acc_matrix, axis=0).tolist(),
    }


def compute_partition_statistics(histories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Estadísticas por partición sobre múltiples experimentos."""
    if not histories or not histories[0].get("partition_accuracies"):
        return {}

    # Forma: (num_experiments, num_epochs, num_partitions)
    pa = np.array([h["partition_accuracies"] for h in histories])
    num_partitions = pa.shape[2]
    num_epochs = pa.shape[1]

    by_partition = []
    for p_idx in range(num_partitions):
        epoch_stats = []
        for epoch in range(num_epochs):
            accs = pa[:, epoch, p_idx]
            epoch_stats.append(
                {
                    "mean": float(np.mean(accs)),
                    "std": float(np.std(accs, ddof=0)),
                    "min": float(np.min(accs)),
                    "max": float(np.max(accs)),
                }
            )
        by_partition.append(epoch_stats)

    return {
        "by_partition": by_partition,
        "num_partitions": num_partitions,
        "num_epochs": num_epochs,
    }


def compute_convergence_epoch(
    accuracies: List[float], threshold: float = 0.01, window: int = 3
) -> int:
    """
    Determina en qué época converge el modelo.

    :return: Índice (0-based) de convergencia, o -1 si no converge
    """
    accs = np.array(accuracies)
    if len(accs) < window + 1:
        return -1

    diffs = np.diff(accs)  # mejoras época a época

    for i in range(window - 1, len(diffs)):
        window_mean = np.mean(np.abs(diffs[i - window + 1 : i + 1]))
        if window_mean < threshold:
            return i + 1  # +1 porque diff reduce la longitud en 1

    return -1
