"""
Funciones matemáticas puras para la red neuronal.

Incluye funciones de activación, álgebra lineal básica y las operaciones
especializadas para el entrenamiento federado (promedio de parámetros y
acumulación eficiente de gradientes).
"""

import math
from typing import Any, Dict, List

import numpy as np

# =========================
# FUNCIONES DE ACTIVACIÓN
# =========================


def sigmoid(z: float) -> float:
    """
    Función sigmoide: σ(z) = 1 / (1 + e^(−z)).

    Incluye protección contra overflow para valores muy negativos.

    :param z: Valor de entrada
    :type z: float

    :return: Valor en el rango (0, 1)
    :rtype: float
    """
    if z < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_derivative_from_activation(a: float) -> float:
    """
    Derivada de la sigmoide expresada en términos de la activación ya calculada.

    σ'(z) = σ(z) · (1 − σ(z)) = a · (1 − a)

    :param a: Valor de activación (resultado previo de sigmoid)
    :type a: float

    :return: Derivada evaluada en z
    :rtype: float
    """
    return a * (1.0 - a)


def softmax(z_list: List[float]) -> List[float]:
    """
    Función softmax: convierte logits a probabilidades que suman 1.

    Aplica estabilización numérica restando el máximo antes de exponenciar.

    :param z_list: Lista de logits
    :type z_list: List[float]

    :return: Lista de probabilidades
    :rtype: List[float]
    """
    z = np.array(z_list, dtype=np.float64)

    # Estabilización numérica
    z -= z.max()

    e = np.exp(z)
    return (e / e.sum()).tolist()


# =============================
# ÁLGEBRA LINEAL — VECTORES
# =============================


def vector_add(v1: List[float], v2: List[float]) -> List[float]:
    """
    Suma elemento a elemento de dos vectores del mismo tamaño.

    :param v1: Primer vector
    :type v1: List[float]

    :param v2: Segundo vector
    :type v2: List[float]

    :return: Vector suma
    :rtype: List[float]

    :raises ValueError: Si los vectores tienen longitudes distintas
    """
    if len(v1) != len(v2):
        raise ValueError(f"Vectores de distinto tamaño: {len(v1)} vs {len(v2)}")
    return (np.asarray(v1) + np.asarray(v2)).tolist()


def vector_subtract(v1: List[float], v2: List[float]) -> List[float]:
    """
    Resta elemento a elemento: v1 − v2.

    :param v1: Minuendo
    :type v1: List[float]

    :param v2: Sustraendo
    :type v2: List[float]

    :return: Vector diferencia
    :rtype: List[float]

    :raises ValueError: Si los vectores tienen longitudes distintas
    """
    if len(v1) != len(v2):
        raise ValueError(f"Vectores de distinto tamaño: {len(v1)} vs {len(v2)}")
    return (np.asarray(v1) - np.asarray(v2)).tolist()


def vector_zeros(size: int) -> List[float]:
    """
    Crea un vector de ceros.

    :param size: Longitud del vector
    :type size: int

    :return: Vector de ceros
    :rtype: List[float]
    """
    return [0.0] * size


# =============================
# ÁLGEBRA LINEAL — MATRICES
# =============================


def matrix_vector_multiply(
    matrix: List[List[float]], vector: List[float]
) -> List[float]:
    """
    Multiplicación matriz × vector.

    :param matrix: Matriz de tamaño m×n
    :type matrix: List[List[float]]

    :param vector: Vector de tamaño n
    :type vector: List[float]

    :return: Vector resultado de tamaño m
    :rtype: List[float]

    :raises ValueError: Si las dimensiones son incompatibles
    """
    if not matrix:
        return []
    if len(matrix[0]) != len(vector):
        raise ValueError(
            f"Dimensiones incompatibles: matriz {len(matrix)}×{len(matrix[0])}, "
            f"vector {len(vector)}"
        )
    return (np.asarray(matrix) @ np.asarray(vector)).tolist()


def matrix_transpose(matrix: List[List[float]]) -> List[List[float]]:
    """
    Transpone una matriz.

    :param matrix: Matriz de entrada de tamaño m×n
    :type matrix: List[List[float]]

    :return: Matriz transpuesta de tamaño n×m
    :rtype: List[List[float]]
    """
    if not matrix:
        return []
    return np.asarray(matrix).T.tolist()


def outer_product(v_col: List[float], v_row: List[float]) -> List[List[float]]:
    """
    Producto externo de dos vectores: resultado[i][j] = v_col[i] · v_row[j].

    :param v_col: Vector columna (define las filas)
    :type v_col: List[float]

    :param v_row: Vector fila (define las columnas)
    :type v_row: List[float]

    :return: Matriz de tamaño len(v_col) × len(v_row)
    :rtype: List[List[float]]
    """
    return np.outer(v_col, v_row).tolist()


def matrix_add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """
    Suma elemento a elemento de dos matrices.

    :param A: Primera matriz
    :type A: List[List[float]]

    :param B: Segunda matriz
    :type B: List[List[float]]

    :return: Matriz suma
    :rtype: List[List[float]]

    :raises ValueError: Si las matrices tienen dimensiones distintas
    """
    a, b = np.asarray(A), np.asarray(B)
    if a.shape != b.shape:
        raise ValueError("Las matrices deben tener las mismas dimensiones")
    return (a + b).tolist()


# ==============================
# INICIALIZACIÓN DE PARÁMETROS
# ==============================


def xavier_initialization(fan_in: int, fan_out: int) -> List[List[float]]:
    """
    Inicialización Xavier/Glorot para pesos de red neuronal.

    Los pesos se distribuyen con media 0 y desviación estándar
    sqrt(2 / (fan_in + fan_out)), lo que ayuda a mantener la varianza
    de las activaciones estable a través de las capas.

    :param fan_in: Número de neuronas de entrada (columnas de la matriz)
    :type fan_in: int

    :param fan_out: Número de neuronas de salida (filas de la matriz)
    :type fan_out: int

    :return: Matriz de pesos de tamaño fan_out × fan_in
    :rtype: List[List[float]]
    """
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0.0, std, (fan_out, fan_in)).tolist()


# ===========================================
# UTILIDADES PARA EL ALGORITMO DE DIEGO
# ===========================================


def average_vectors(vectors: List[List[float]]) -> List[float]:
    """
    Calcula el promedio elemento a elemento de una lista de vectores.

    :param vectors: Lista de vectores del mismo tamaño
    :type vectors: List[List[float]]

    :return: Vector promedio
    :rtype: List[float]

    :raises ValueError: Si la lista está vacía o los tamaños difieren
    """
    if not vectors:
        raise ValueError("No se puede promediar una lista vacía")

    size = len(vectors[0])
    if any(len(v) != size for v in vectors):
        raise ValueError("Todos los vectores deben tener el mismo tamaño")

    return np.mean(np.array(vectors, dtype=np.float64), axis=0).tolist()


def average_matrices(matrices: List[List[List[float]]]) -> List[List[float]]:
    """
    Calcula el promedio elemento a elemento de una lista de matrices.

    :param matrices: Lista de matrices con las mismas dimensiones
    :type matrices: List[List[List[float]]]

    :return: Matriz promedio
    :rtype: List[List[float]]

    :raises ValueError: Si la lista está vacía o las dimensiones difieren
    """
    if not matrices:
        raise ValueError("No se puede promediar una lista vacía")

    rows = len(matrices[0])
    if rows == 0:
        raise ValueError("Las matrices no pueden estar vacías")

    cols = len(matrices[0][0])

    # Valida dimensiones y forma rectangular
    for m in matrices:
        if len(m) != rows:
            raise ValueError("Todas las matrices deben tener las mismas dimensiones")
        for row in m:
            if len(row) != cols:
                raise ValueError(
                    "Todas las matrices deben tener las mismas dimensiones"
                )

    return np.mean(np.array(matrices, dtype=np.float64), axis=0).tolist()


def average_network_parameters(parameters_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Promedia los parámetros de múltiples redes neuronales.

    Recibe una lista de diccionarios con parámetros (W1, b1, W2, b2)
    y devuelve un diccionario con cada parámetro promediado.
    Es el núcleo del algoritmo de Diego: los parámetros de cada partición
    se promedian para obtener los nuevos parámetros globales.

    :param parameters_list: Lista de diccionarios de parámetros
    :type parameters_list: List[Dict[str, Any]]

    :return: Diccionario con parámetros promediados
    :rtype: Dict[str, Any]

    :raises ValueError: Si la lista está vacía o los diccionarios tienen claves distintas
    """
    if not parameters_list:
        raise ValueError("No se puede promediar una lista vacía de parámetros")

    keys = parameters_list[0].keys()
    if any(set(p.keys()) != set(keys) for p in parameters_list):
        raise ValueError("Todos los diccionarios deben tener las mismas claves")

    averaged: Dict[str, Any] = {}
    for key in keys:
        first = parameters_list[0][key]
        values = [p[key] for p in parameters_list]

        if isinstance(first, list) and first and isinstance(first[0], list):
            averaged[key] = average_matrices(values)
        elif isinstance(first, list):
            averaged[key] = average_vectors(values)
        else:
            averaged[key] = float(np.mean(values))

    return averaged


# ============================================================
# ACUMULACIÓN EFICIENTE DE GRADIENTES (IN-PLACE)
# ============================================================


def accumulate_outer_inplace(acc: Any, v_col: List[float], v_row: List[float]) -> None:
    """
    Acumula el producto externo v_col ⊗ v_row directamente sobre acc.

    Equivale a ``acc += outer_product(v_col, v_row)``, pero fusiona ambas
    operaciones en un único pase sin crear matrices intermedias.
    Para matrices grandes (ej. dW1 de 30×784) iteradas miles de veces
    por batch, elimina dos allocations de lista por ejemplo.

    :param acc: Acumulador modificado in-place (m×n)
    :type acc: List[List[float]] | np.ndarray

    :param v_col: Vector columna de longitud m
    :type v_col: List[float]

    :param v_row: Vector fila de longitud n
    :type v_row: List[float]
    """
    np.add(acc, np.outer(v_col, v_row), out=acc)


def accumulate_vector_inplace(acc: Any, v: List[float]) -> None:
    """
    Acumula v sobre acc directamente: ``acc[i] += v[i]`` para todo i.

    Equivale a ``vector_add`` pero sin crear una nueva lista.

    :param acc: Vector acumulador modificado in-place
    :type acc: List[float] | np.ndarray

    :param v: Vector a sumar
    :type v: List[float]
    """
    np.add(acc, v, out=acc)


# ======================
# UTILIDADES GENERALES
# ======================


def argmax(vector: List[float]) -> int:
    """
    Devuelve el índice del valor máximo en un vector.

    :param vector: Vector de entrada
    :type vector: List[float]
    :return: Índice del valor máximo
    :rtype: int
    """
    return int(np.argmax(vector))


def compute_one_hot(label: int, num_classes: int) -> List[float]:
    """
    Crea un vector one-hot para una etiqueta dada.

    :param label: Índice de la clase (0 a num_classes−1)
    :type label: int

    :param num_classes: Número total de clases
    :type num_classes: int

    :return: Vector con 1.0 en la posición label y 0.0 en el resto
    :rtype: List[float]

    :raises ValueError: Si label está fuera del rango válido
    """
    if label < 0 or label >= num_classes:
        raise ValueError(f"Etiqueta {label} fuera del rango [0, {num_classes})")
    one_hot = np.zeros(num_classes, dtype=np.float64)
    one_hot[label] = 1.0
    return one_hot.tolist()
