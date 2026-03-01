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
# CURVA DE APRENDIZAJE (PANEL 1)
# ================================================================


def prepare_accuracy_chart_data(
    histories: List[Dict[str, Any]],
    include_confidence_band: bool = True,
) -> Dict[str, Any]:
    """
    Prepara los datos para la curva de evolución de precisión promedio.

    A partir de múltiples experimentos se calcula la media y la
    desviación estándar por época.

    Cuando `include_confidence_band=True`, se generan bandas. Estas
    bandas representan la dispersión ±1σ. No es un intervalo de confianza
    formal, es simplemente una medida de dispersión visual.

    :param histories: Lista de historiales con la clave "accuracies".
    :type histories: List[Dict[str, Any]]

    :param include_confidence_band: Indica si se incluyen bandas ±1σ.
    :type include_confidence_band: bool

    :return: Diccionario estructurado para graficar.
    :rtype: Dict[str, Any]
    """
    # Retorna estadísticas por época
    stats = compute_epoch_statistics(histories)

    mean = np.array(stats["mean"])
    std = np.array(stats["std"])

    # Crea lista [1, 2, 3, ..., número_de_épocas]
    epochs = list(range(1, len(mean) + 1))

    data: Dict[str, Any] = {
        "x": epochs,
        "y_mean": mean.tolist(),  # Se convierte a lista para exportar
        "y_std": std.tolist(),
        "y_min": stats["min"],
        "y_max": stats["max"],
        "title": "Evolución de Precisión",
        "xlabel": "Época",
        "ylabel": "Precisión (%)",
    }

    if include_confidence_band:
        # Banda superior = Media + Desviación Estándar
        data["y_upper"] = (mean + std).tolist()

        # Banda inferior = Media - Desviación Estándar
        # np.maxium aquí evita valores negativos
        data["y_lower"] = np.maximum(0.0, mean - std).tolist()

    return data


# ================================================================
# COMPARACIÓN POR PARTICIÓN (PANEL 2)
# ================================================================


def prepare_partition_comparison_data(
    histories: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Prepara datos para comparar la evolución de precisión entre particiones.

    Cada partición genera una serie temporal independiente basada
    en la media por época.

    :param histories: Lista de experimentos con "partition_accuracies"
    :type histories: List[Dict[str, Any]]

    :return: Diccionario con estructura lista para múltiples curvas.
    :rtype: Dict[str, Any]
    """
    # Calcula estadísticas por partición
    p_stats = compute_partition_statistics(histories)

    if not p_stats:
        return {}

    # Contenedor para los datos listos para graficar
    data: Dict[str, Any] = {
        "partitions": [],
        "title": "Comparación por Partición",
        "xlabel": "Época",
        "ylabel": "Precisión (%)",
    }

    # Itera sobre cada partición
    for p_idx, epoch_stats in enumerate(p_stats["by_partition"]):
        data["partitions"].append(
            {
                "id": p_idx + 1,  # Identificador legible para gráfica
                "x": list(
                    range(1, len(epoch_stats) + 1)
                ),  # Número de épocas para eje X
                "y": [e["mean"] for e in epoch_stats],  # Lista de promedios por época
                "std": [e["std"] for e in epoch_stats],  # Lista de std's por época
            }
        )

    return data


# ================================================================
# ANÁLISIS DE CONVERGENCIA (PANEL 4)
# ================================================================


def prepare_convergence_data(histories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Prepara datos para análisis de convergencia del entrenamiento.

    La convergencia es el momento donde el modelo deja de mejorar
    significativamente.

    Calcula:

    1) La mejora media por época:
       Δ accuracy = accuracy_t - accuracy_(t-1)

    2) La época estimada de convergencia para cada experimento,
       según el criterio definido en compute_convergence_epoch

    :param histories: Lista de historiales de entrenamiento
    :type histories: List[Dict[str, Any]]

    :return: Diccionario con mejoras medias y estadísticas
             de convergencia.
    :rtype: Dict[str, Any]
    """
    convergence_points = []
    all_improvements = []

    # Itera sobre cada experimento
    for h in histories:
        acc = np.array(h["accuracies"])

        # Detecta época de convergencia
        conv = compute_convergence_epoch(h["accuracies"])

        # Caso 1: Se detectó convergencia. Suma 1 para pasar de índice 0-based a época 1-based.
        # Caso 2: La convergencia nunca ocurrió. Toma la última época como referencia
        convergence_points.append(conv + 1 if conv >= 0 else len(acc))

        # np.diff calcula diferencias consecutivas
        all_improvements.append(np.diff(acc))

    # Convierte a matriz (filas = experimentos, columnas = épocas-1)
    imp_matrix = np.array(all_improvements)

    # Promedio de mejora por época
    mean_improvements = np.mean(imp_matrix, axis=0).tolist()

    return {
        "x": list(range(2, len(mean_improvements) + 2)),  # Épocas desde la 2 para eje x
        "y": mean_improvements,  # Mejora promedio por época
        "convergence_epochs": convergence_points,  # Épocas de convergencia por experimento
        "mean_convergence": float(
            np.mean(convergence_points)
        ),  # Promedio de convergencia
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

    - Precisión final = último valor de la curva
    - np.histogram = divide en bins (contenedores) para construir la distribución
    - mean/std/min/max -> estadísticas descriptivas

    :param histories: Lista de historiales.
    :type histories: List[Dict[str, Any]]

    :return: Diccionario con bins, frecuencias y estadísticas.
    :rtype: Dict[str, Any]
    """
    # Precisión final de cada experimento
    final_accs = np.array([h["accuracies"][-1] for h in histories])

    # Número de bins = mínimo entre 10 y número de experimentos
    num_bins = min(10, len(final_accs))

    # Calcula histograma:
    # counts = cuántos valores caen en cada intervalo
    # bin_edges = límites de los intervalos
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
# PRECISIÓN POR EXPERIMENTO + RSD (PANEL 3)
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
        "y": stats["per_experiment"],  # Precisión final por experimento
        "mean": stats["mean"],  # promedio global
        "std": stats["std"],  # desviación estándar global
        "rsd": stats["rsd"],  # desviación relativa (%)
        "upper": stats["mean"] + stats["std"],  # banda superior
        "lower": max(0.0, stats["mean"] - stats["std"]),  # banda inferior
        "min": stats["min"],
        "max": stats["max"],
        "range": stats["range"],
        "interpretation": stats["interpretation"],  # texto interpretativo
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

    :param all_results: Lista de resultados completos de cada configuración de entrenamiento
    :type all_results: List[Dict[str, Any]]

    :param config_labels: Etiquetas opcionales para cada configuración
    :type config_labels: List[str] | None

    :return: Diccionario estructurado para múltiples curvas
    :rtype: Dict[str, Any]
    """
    # Si no hay etiquetas, genera nombres por defecto
    if config_labels is None:
        config_labels = [f"Config {i + 1}" for i in range(len(all_results))]

    data: Dict[str, Any] = {
        "configurations": [],
        "title": "Comparación de Configuraciones",
        "xlabel": "Época",
        "ylabel": "Precisión (%)",
    }

    # Itera sobre cada configuración
    for result, label in zip(all_results, config_labels):
        stats = compute_epoch_statistics(result["all_histories"])
        data["configurations"].append(
            {
                "label": label,
                "x": list(
                    range(1, len(stats["mean"]) + 1)
                ),  # Número de épocas para eje x
                "y": stats["mean"],  # Promedio por época
                "std": stats["std"],  # Std por época
                "final_accuracy": result.get("final_mean_accuracy", 0.0),
                "final_std": result.get("final_std_accuracy", 0.0),
            }
        )

    return data


def prepare_benchmark_data(benchmark_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepara datos para visualizar la comparación secuencial vs paralelo.

    Extrae los tiempos por experimento de ambos modos y las métricas
    comparativas calculadas por ``run_benchmark_comparison``.

    Produce dos estructuras:
        - ``"times"``: datos para un gráfico de barras con los tiempos
          individuales de cada experimento en ambos modos.
        - ``"summary"``: métricas de la comparación (speedup, eficiencia,
          overhead, delta de precisión) para una tabla o anotaciones.

    :param benchmark_result: Resultado de ``run_benchmark_comparison``.
    :type benchmark_result: Dict[str, Any]

    :return: Diccionario listo para renderizar en matplotlib.
    :rtype: Dict[str, Any]
    """
    seq_bm = benchmark_result["sequential"]["benchmark"]
    par_bm = benchmark_result["parallel"]["benchmark"]
    comp = benchmark_result["comparison"]
    n = len(seq_bm["times"])

    return {
        # Tiempos por experimento (eje X = índice del experimento)
        "times": {
            "x": list(range(1, n + 1)),
            "seq": seq_bm["times"],
            "par": par_bm["times"],
            "seq_mean": seq_bm["mean_time"],
            "par_mean": par_bm["mean_time"],
            "title": "Tiempo por Experimento: Secuencial vs Paralelo",
            "xlabel": "Experimento",
            "ylabel": "Tiempo (s)",
        },
        # Métricas resumen para anotaciones
        "summary": {
            "speedup": comp["speedup"],
            "efficiency_pct": comp["efficiency_pct"],
            "overhead_sec": comp["overhead_sec"],
            "seq_mean_accuracy": comp["seq_mean_accuracy"],
            "par_mean_accuracy": comp["par_mean_accuracy"],
            "accuracy_delta": comp["accuracy_delta"],
        },
    }


"""
NOTAS DE ESTADÍSTICA:

Media (Mean) = μ
Desviación Estándar (std) = σ
Relative Standard Deviation (RSD) = (σ / μ)*100

------------------------------------------------------------------------------

La "Desviación Estándar" mide qué tan dispersos están los valores
respecto al promedio.

* Baja desviación → todos los experimentos se parecen
* Alta desviación → resultados muy variables

------------------------------------------------------------------------------

La "Desviación Estándar Relativa" es una medida estadística que
nos dice qué tan grande es la dispersión de los datos en comparación
con el promedio. Es, básicamente, la desviación estándar expresada
como un porcentaje.

Mientras que la desviación estándar común te da un número absoluto
(ej. ±5 metros), la RSD te da una perspectiva de proporción
(ej. un error del 2%).

------------------------------------------------------------------------------

La "Dispersión Más o Menos una Sigma" (±1σ) se refiere al intervalo
que abarca una desviación estándar por encima y por debajo del promedio
(la media) en un conjunto de datos.

En estadística, bajo una distribución normal (la famosa campana de Gauss),
este rango tiene un significado muy específico: 

* Cobertura de datos: Aproximadamente el 68.2% de todos los valores de un
  conjunto de datos se encuentran dentro de este intervalo.
* Significado: Indica que la mayoría de los eventos o mediciones son "normales"
  o esperados. Si un dato cae fuera de este rango de ±1σ, se empieza a considerar
  menos común, aunque todavía es muy frecuente (ocurre el 32% de las veces).

------------------------------------------------------------------------------

La "Convergencia" se define como el momento en que el modelo deja de mejorar de
forma significativa.

Cuando se entrena un modelo, en cada época este mejora un poco.
* Al principio mejora mucho.
* Luego mejora menos.
* Llega un punto donde casi no mejora.
Entonces, la convergencia es cuando el modelo ya casi no mejora más.

------------------------------------------------------------------------------

El "Threshold" o "Umbral Mínimo de Mejora" es la vara de medir que utilizamos
para declarar que un algoritmo ha "terminado" su trabajo con éxito.

Si se establece que threshold = 0.01, eso significa que: si el modelo mejora
menos de 1% en promedio, consideramos que ya no está mejorando realmente.

En otras palabras: Si la mejora es menor a 0.01, ya no vale la pena seguir
entrenando.

------------------------------------------------------------------------------

El "Window" es el periodo de tiempo (medido en épocas o iteraciones) que el
algoritmo observa para decidir si los cambios cumplen con el threshold.

Por ejemplo, si window = 3, eso significa: Vamos a mirar las últimas 3 mejoras
(épocas) seguidas para decidir si ya convergió.

En otras palabras: no se decide con una sola época, sino con varias seguidas.
"""
