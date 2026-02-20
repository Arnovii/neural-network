"""
Analytics/chart_generator.py

Módulo encargado de la preparación estructurada de datos para
visualización gráfica.

Este módulo NO realiza renderizado. Su única responsabilidad es
transformar los resultados crudos de experimentos en estructuras
listas para ser consumidas por matplotlib u otros motores de
visualización.

Implementación basada en NumPy para eficiencia numérica.
"""

import numpy as np
from typing import Any, Dict, List

from Analytics.statistics_engine import (
    compute_convergence_epoch,
    compute_epoch_statistics,
    compute_experiment_rsd,
    compute_partition_statistics,
)

# ================================================================
# CURVA DE APRENDIZAJE (PRECISIÓN MEDIA POR ÉPOCA)
# ================================================================

def prepare_accuracy_chart_data(
    histories: List[Dict[str, Any]],
    include_confidence_band: bool = True,
) -> Dict[str, Any]:
    """
    Prepara los datos para la curva de evolución de precisión.

    A partir de múltiples experimentos se calcula la media y la
    desviación estándar por época.

    Cuando `include_confidence_band=True`, se generan bandas.

    Estas bandas representan dispersión ±1σ (no intervalo de confianza formal).

    :param histories: Lista de historiales con la clave "accuracies".
    :type histories: List[Dict[str, Any]]

    :param include_confidence_band: Indica si se incluyen bandas ±1σ.
    :type include_confidence_band: bool

    :return: Diccionario estructurado para graficar.
    :rtype: Dict[str, Any]
    """
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

# ================================================================
# COMPARACIÓN POR PARTICIÓN
# ================================================================

def prepare_partition_comparison_data(
    histories: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Prepara datos para comparar la evolución de precisión entre particiones.

    Cada partición genera una serie temporal independiente basada
    en la media por época.

    :param histories: Lista de historiales con clave
                      "partition_accuracies".
    :type histories: List[Dict[str, Any]]

    :return: Diccionario con estructura lista para múltiples curvas.
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

# ================================================================
# ANÁLISIS DE CONVERGENCIA
# ================================================================

def prepare_convergence_data(histories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Prepara datos para análisis de convergencia del entrenamiento.

    Se calcula:

    1) La mejora media por época:
       Δ accuracy = accuracy_t - accuracy_(t-1)

    2) La época estimada de convergencia para cada experimento,
       según el criterio definido en compute_convergence_epoch.

    :param histories: Lista de historiales de entrenamiento.
    :type histories: List[Dict[str, Any]]

    :return: Diccionario con mejoras medias y estadísticas
             de convergencia.
    :rtype: Dict[str, Any]
    """
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

# ================================================================
# DISTRIBUCIÓN DE PRECISIÓN FINAL
# ================================================================

def prepare_distribution_data(histories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Prepara datos para un histograma de precisiones finales.

    Se calcula la precisión final de cada experimento y se
    construye una distribución discreta mediante numpy.histogram.

    :param histories: Lista de historiales.
    :type histories: List[Dict[str, Any]]

    :return: Diccionario con bins, frecuencias y estadísticas.
    :rtype: Dict[str, Any]
    """
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

# ================================================================
# PRECISIÓN POR EXPERIMENTO + RSD
# ================================================================

def prepare_experiment_rsd_data(histories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Prepara datos para la gráfica de precisión por experimento con RSD.

    Genera una barra por experimento (eje X) con su precisión final (eje Y),
    línea del promedio, bandas ±1σ y anotación de la RSD.

    :param histories: Lista de historiales de entrenamiento
    :type histories: List[Dict[str, Any]]

    :return: Diccionario con:
             - x: lista de números de experimento [1, 2, ..., N]
             - y: lista de precisiones finales por experimento
             - mean, std, rsd: estadísticas globales
             - upper, lower: bandas promedio ± 1σ
             - min, max, range: extremos
             - interpretation: texto de interpretación RSD
             - title, xlabel, ylabel: etiquetas del gráfico
    :rtype: Dict[str, Any]
    """
    stats = compute_experiment_rsd(histories)

    n = len(stats["per_experiment"])
    return {
        "x": list(range(1, n + 1)),
        "y": stats["per_experiment"],
        "mean": stats["mean"],
        "std": stats["std"],
        "rsd": stats["rsd"],
        "upper": stats["mean"] + stats["std"],
        "lower": max(0.0, stats["mean"] - stats["std"]),
        "min": stats["min"],
        "max": stats["max"],
        "range": stats["range"],
        "interpretation": stats["interpretation"],
        "title": f"Precisión por Experimento (RSD: {stats['rsd']:.2f}%)",
        "xlabel": "Experimento",
        "ylabel": "Precisión (%)",
    }

# ================================================================
# COMPARACIÓN ENTRE CONFIGURACIONES
# ================================================================

def prepare_comparison_chart_data(
    all_results: List[Dict[str, Any]],
    config_labels: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Prepara datos para comparar múltiples configuraciones
    de entrenamiento.

    Cada configuración genera una curva basada en la media
    de precisión por época.

    :param all_results: Lista de resultados completos de experimentos.
    :type all_results: List[Dict[str, Any]]

    :param config_labels: Etiquetas opcionales para cada configuración.
    :type config_labels: List[str] | None

    :return: Diccionario estructurado para múltiples curvas.
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

# ================================================================
# EXPORTACIÓN
# ================================================================

def export_to_csv(histories: List[Dict[str, Any]], filename: str) -> None:
    """
    Exporta historiales de precisión a un archivo CSV.

    Formato generado:
        experiment,epoch,accuracy

    :param histories: Lista de historiales.
    :type histories: List[Dict[str, Any]]

    :param filename: Ruta del archivo destino.
    :type filename: str
    """
    with open(filename, "w") as f:
        f.write("experiment,epoch,accuracy\n")
        for exp_idx, h in enumerate(histories):
            for epoch, acc in enumerate(h["accuracies"]):
                f.write(f"{exp_idx + 1},{epoch + 1},{acc:.4f}\n")
