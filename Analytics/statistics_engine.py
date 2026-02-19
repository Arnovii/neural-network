"""
Funciones matemáticas puras para el análisis estadístico de experimentos.

Opera sobre los historiales producidos por experiment_runner y
proporciona los datos estadísticos que consume chart_generator.
"""

import math
from typing import Any, Dict, List


def compute_std(values: List[float]) -> float:
    """
    Calcula la desviación estándar poblacional de una lista de valores.

    :param values: Lista de valores numéricos
    :type values: List[float]

    :return: Desviación estándar (0.0 si hay menos de dos valores)
    :rtype: float
    """
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def compute_epoch_statistics(histories: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Calcula estadísticas por época sobre múltiples experimentos.

    Para cada época, recoge las precisiones de todos los experimentos
    y calcula su media, desviación estándar, mínimo y máximo.

    :param histories: Lista de historiales; cada uno debe tener la clave 'accuracies'
    :type histories: List[Dict[str, Any]]

    :return: Diccionario con listas 'mean', 'std', 'min', 'max' (una entrada por época)
    :rtype: Dict[str, List[float]]
    """
    if not histories:
        return {"mean": [], "std": [], "min": [], "max": []}

    num_epochs = len(histories[0]["accuracies"])
    num_experiments = len(histories)

    means, stds, mins, maxs = [], [], [], []

    for epoch in range(num_epochs):
        # Recolecta precisiones de todos los experimentos en esta época
        epoch_accs = [h["accuracies"][epoch] for h in histories]

        # Calcula estadísticas
        mean = sum(epoch_accs) / num_experiments
        means.append(mean)
        stds.append(compute_std(epoch_accs))
        mins.append(min(epoch_accs))
        maxs.append(max(epoch_accs))

    return {"mean": means, "std": stds, "min": mins, "max": maxs}


def compute_partition_statistics(histories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calcula estadísticas por partición sobre múltiples experimentos.

    :param histories: Lista de historiales con la clave 'partition_accuracies'.
                     Forma: histories[exp][epoch][partition]
    :type histories: List[Dict[str, Any]]

    :return: Diccionario con 'by_partition' (lista de listas de stats por época),
             'num_partitions' y 'num_epochs'. Vacío si los datos no están presentes.
    :rtype: Dict[str, Any]
    """
    if not histories or not histories[0].get("partition_accuracies"):
        return {}

    num_partitions = len(histories[0]["partition_accuracies"][0])
    num_epochs = len(histories[0]["partition_accuracies"])

    by_partition = []
    for p_idx in range(num_partitions):
        epoch_stats = []
        for epoch in range(num_epochs):
            # Recolecta precisiones de esta partición en esta época
            accs = [h["partition_accuracies"][epoch][p_idx] for h in histories]
            mean = sum(accs) / len(accs)
            epoch_stats.append(
                {
                    "mean": mean,
                    "std": compute_std(accs),
                    "min": min(accs),
                    "max": max(accs),
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

    Se considera convergencia cuando la mejora promedio en una ventana
    de ``window`` épocas cae por debajo de ``threshold``.

    :param accuracies: Lista de precisiones por época
    :type accuracies: List[float]

    :param threshold: Mejora mínima para considerar que el modelo sigue aprendiendo
    :type threshold: float

    :param window: Número de épocas a considerar en la ventana
    :type window: int

    :return: Índice (0-based) de la época de convergencia, o -1 si no converge
    :rtype: int
    """
    if len(accuracies) < window + 1:
        return -1

    for i in range(window, len(accuracies)):
        # Calcular mejora promedio en la ventana
        improvements = [
            accuracies[j] - accuracies[j - 1] for j in range(i - window + 1, i + 1)
        ]
        if abs(sum(improvements) / len(improvements)) < threshold:
            return i

    return -1
