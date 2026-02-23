"""
Analytics/statistics_engine.py

Módulo de funciones estadísticas para el análisis de experimentos
de entrenamiento de redes neuronales.

Todas las operaciones están implementadas utilizando NumPy por
eficiencia numérica.

Se usa estadística poblacional, porque estamos analizando todos
los experimentos generados, no una muestra parcial. Se establece
el parámetro ddof=0 para que NumPy trate a los datos como una
Población completa y no como una muestra.
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
    # Si hay menos de 2 valores, no existe dispersión real
    if len(values) < 2:
        return 0.0

    # np.std calcula la desviación estándar
    return float(np.std(values, ddof=0))


def compute_epoch_statistics(histories: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Calcula estadísticas descriptivas por época sobre múltiples experimentos.

    Cada elemento de ``histories`` debe contener la clave:

        "accuracies": List[float]

    Donde cada lista representa la evolución de precisión
    por época para un experimento independiente.

    Se construye una matriz bidimensional con forma:

        (num_experiments, num_epochs)

    A partir de esta matriz se calculan métricas por columna (por época).

    Estructura esperada:\n
        histories = [
            {"accuracies": [acc_e1_ep1, acc_e1_ep2, ...]},
            {"accuracies": [acc_e2_ep1, acc_e2_ep2, ...]},
            ...
        ]

    :param histories: Lista de historiales de experimentos.
    :type histories: List[Dict[str, Any]]

    :return: Diccionario con listas por época:\n
             - mean: promedio
             - std: desviación estándar poblacional
             - min: mínimo
             - max: máximo
    :rtype: Dict[str, List[float]]
    """
    if not histories:
        return {"mean": [], "std": [], "min": [], "max": []}

    # Construye una matriz 2D:
    # Filas -> experimentos
    # Columnas -> épocas
    acc_matrix = np.array([h["accuracies"] for h in histories])

    # Usa axis=0 para operar por columna (por época)
    return {
        "mean": np.mean(
            acc_matrix, axis=0
        ).tolist(),  # Promedio de cada época entre experimentos
        "std": np.std(
            acc_matrix, axis=0, ddof=0
        ).tolist(),  # Desviación estándar poblacional por época
        "min": np.min(acc_matrix, axis=0).tolist(),  # Valor mínimo por época
        "max": np.max(acc_matrix, axis=0).tolist(),  # Valor máximo por época
    }


# ================================================================
# ESTADÍSTICAS POR PARTICIÓN
# ================================================================


def compute_partition_statistics(histories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calcula estadísticas por partición y por época.

    Se espera que cada elemento de ``histories`` contenga
    la clave ``"partition_accuracies"`` con forma:

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

    # Convierte lista Python a arreglo NumPy en 3-D:
    # Eje 0 -> experimento
    # Eje 1 -> época
    # Eje 2 -> partición
    pa = np.array([h["partition_accuracies"] for h in histories])
    num_partitions = pa.shape[2]
    num_epochs = pa.shape[1]

    by_partition = []

    # Recorre cada partición
    for p_idx in range(num_partitions):
        epoch_stats = []

        # Recorremos cada época
        for epoch in range(num_epochs):
            # Extraemos todas las precisiones de:
            # - todos los experimentos (:)
            # - época fija
            # - partición fija
            #
            # Resultado:
            # accs.shape = (num_experiments,)
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

    :return: Diccionario con:\n
             - per_experiment: lista de precisiones finales (una por experimento)\n
             - mean: promedio global
             - std: desviación estándar poblacional
             - rsd: desviación estándar relativa (%)\n
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

    # Consigue precisión final por experimento
    final_accs = np.array([h["accuracies"][-1] for h in histories])
    mean = float(np.mean(final_accs))
    std = float(np.std(final_accs, ddof=0))

    # Calcula rsd evitando división por cero
    rsd = (std / mean) * 100.0 if mean > 0 else 0.0

    # Interpretación cualitativa
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

    :param accuracies: Lista de valores de accuracy por época.
    :type accuracies: List[float]

    :param threshold: Umbral mínimo de mejora media absoluta para considerar convergencia.
    :type threshold: float

    :param window: Número de épocas consecutivas usadas para calcular la mejora media.
    :type window: int

    :return: Índice (0-based) de convergencia, o -1 si no converge.
    :rtype: int
    """
    accs = np.array(accuracies)

    # Necesita al menos window + 1 valores
    # len(diffs) >= window  ->  len(diffs) = len(accs) - 1  ->  len(accs) - 1 >= window  ->  len(accs) >= window + 1
    if len(accs) < window + 1:
        return -1

    # Calcula diferencias entre épocas
    diffs = np.diff(accs)

    # Recorre usando una ventana móvil
    # i = Índice donde termina la ventana actual.
    # Si necesitamos W elementos consecutivos, entonces empieza en: i - (Window - 1)
    # Queremos recorrer todas las ventanas posibles de tamaño W.
    # Como no podemos ir antes del índice 0, i - (W - 1) ≥ 0. Despejamos: i ≥ W - 1 (punto de inicio para i)
    # El máximo valor que puede tomar i es len(diffs) -1, pero como range(a, b) llega solo hasta b - 1, eso ya se cumple.
    for i in range(window - 1, len(diffs)):
        window_mean = np.mean(np.abs(diffs[i - window + 1 : i + 1]))
        if window_mean < threshold:
            return i + 1  # +1 porque diff reduce la longitud en 1

    return -1


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
