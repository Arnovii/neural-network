"""
Analytics/statistics_engine.py

Módulo de funciones estadísticas para el análisis de experimentos
de entrenamiento de redes neuronales.

Todas las operaciones están implementadas utilizando NumPy y se
basan en estadística descriptiva poblacional (ddof=0).
"""

import numpy as np
from typing import Any, Dict, List


def compute_std(values: List[float]) -> float:
    """
    Calcula la desviación estándar poblacional de una lista de valores.

    Se utiliza ddof=0 (poblacional), ya que los experimentos se consideran
    el conjunto completo bajo análisis y no una muestra.

    :param values: Lista de valores numéricos.
    :type values: List[float]

    :return: Desviación estándar poblacional.
    :rtype: float
    """
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=0))


def compute_epoch_statistics(histories: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Calcula estadísticas descriptivas por época sobre múltiples experimentos.

    Cada experimento contiene una lista de precisiones por época bajo la
    clave ``"accuracies"``. Se construye una matriz de forma:

        (num_experiments, num_epochs)

    A partir de esta matriz se calculan métricas por columna (por época).

    :param histories: Lista de historiales de experimentos.
    :type histories: List[Dict[str, Any]]

    :return: Diccionario con listas por época:
             - mean: promedio
             - std: desviación estándar poblacional
             - min: mínimo
             - max: máximo
    :rtype: Dict[str, List[float]]
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

# ================================================================
# ESTADÍSTICAS POR PARTICIÓN
# ================================================================

def compute_partition_statistics(histories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calcula estadísticas por partición y por época.

    Se espera que cada historial contenga la clave
    ``"partition_accuracies"`` con forma:

        (num_epochs, num_partitions)

    Se construye un arreglo tridimensional:

        (num_experiments, num_epochs, num_partitions)

    Luego se calculan métricas por partición y por época.

    :param histories: Lista de historiales de experimentos.
    :type histories: List[Dict[str, Any]]

    :return: Diccionario con:
             - by_partition: lista indexada por partición,
               cada una contiene estadísticas por época
             - num_partitions: número total de particiones
             - num_epochs: número total de épocas
    :rtype: Dict[str, Any]
    """
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

# ================================================================
# RSD — RELATIVE STANDARD DEVIATION
# ================================================================

def compute_experiment_rsd(histories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calcula la precisión final de cada experimento y la RSD global.

    La RSD (Desviación Estándar Relativa) mide la dispersión de los
    resultados como porcentaje del promedio:
        RSD = (desviación_estándar / promedio) × 100%

    :param histories: Lista de historiales con la clave 'accuracies'
    :type histories: List[Dict[str, Any]]

    :return: Diccionario con:
             - per_experiment: lista de precisiones finales (una por experimento)
             - mean: promedio global
             - std: desviación estándar poblacional
             - rsd: desviación estándar relativa (%)
             - min, max, range: valores extremos
             - interpretation: texto interpretando la estabilidad
    :rtype: Dict[str, Any]
    """
    if not histories:
        return {
            "per_experiment": [],
            "mean": 0.0,
            "std": 0.0,
            "rsd": 0.0,
            "min": 0.0,
            "max": 0.0,
            "range": 0.0,
            "interpretation": "",
        }

    final_accs = np.array([h["accuracies"][-1] for h in histories])
    mean = float(np.mean(final_accs))
    std = float(np.std(final_accs, ddof=0))
    rsd = (std / mean) * 100.0 if mean > 0 else 0.0

    if rsd < 2:
        interpretation = "MUY ESTABLE — variabilidad mínima"
    elif rsd < 5:
        interpretation = "ESTABLE — variabilidad aceptable"
    elif rsd < 10:
        interpretation = "MODERADO — variabilidad notable"
    else:
        interpretation = "INESTABLE — alta variabilidad"

    return {
        "per_experiment": final_accs.tolist(),
        "mean": mean,
        "std": std,
        "rsd": rsd,
        "min": float(np.min(final_accs)),
        "max": float(np.max(final_accs)),
        "range": float(np.max(final_accs) - np.min(final_accs)),
        "interpretation": interpretation,
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

    # Mejoras época a época
    diffs = np.diff(accs)

    for i in range(window - 1, len(diffs)):
        window_mean = np.mean(np.abs(diffs[i - window + 1 : i + 1]))
        if window_mean < threshold:
            return i + 1  # +1 porque diff reduce la longitud en 1

    return -1
