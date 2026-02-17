"""
Este módulo proporciona funciones matemáticas básicas.\n
Incluye funciones de activación, álgebra lineal y operaciones para
el manejo de gradientes y parámetros.
"""

import math
from typing import List

# =========================
# FUNCIONES DE ACTIVACIÓN
# =========================


def sigmoid(z: float) -> float:
    """
    Función sigmoide: σ(z) = 1 / (1 + e^(-z))\n
    Transforma cualquier número a un valor entre 0 y 1.\n
    Incluye protección contra overflow para valores muy negativos.

    :param z: Valor de entrada
    :type z: float
    :return: Valor entre 0 y 1
    :rtype: float
    """
    if z < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_derivative_from_activation(a: float) -> float:
    """
    Derivada de sigmoide: σ'(z) = σ(z) * (1 - σ(z))\n
    Como ya tenemos 'a' (que es σ(z)), usamos: a * (1 - a)

    :param a: Valor de activación (resultado de sigmoid)
    :type a: float
    :return: Derivada evaluada
    :rtype: float
    """
    return a * (1.0 - a)


def softmax(z_list: List[float]) -> List[float]:
    """
    Softmax: convierte logits a probabilidades que suman 1.\n
    Usa estabilización numérica restando el máximo antes de exponenciar.

    :param z_list: Lista de valores (logits)
    :type z_list: List[float]
    :return: Lista de probabilidades que suman 1
    :rtype: List[float]
    """
    max_z = max(z_list)
    exp_z = [math.exp(z - max_z) for z in z_list]
    sum_exp = sum(exp_z)
    return [e / sum_exp for e in exp_z]


# =============================
# ÁLGEBRA LINEAL - VECTORES
# =============================


def vector_add(v1: List[float], v2: List[float]) -> List[float]:
    """
    Suma elemento a elemento de dos vectores.

    :param v1: Primer vector
    :type v1: List[float]
    :param v2: Segundo vector
    :type v2: List[float]
    :return: Vector suma
    :rtype: List[float]
    :raises ValueError: Si los vectores tienen diferente longitud
    """
    if len(v1) != len(v2):
        raise ValueError(f"Vectores de diferente tamaño: {len(v1)} vs {len(v2)}")
    return [v1[i] + v2[i] for i in range(len(v1))]


def vector_subtract(v1: List[float], v2: List[float]) -> List[float]:
    """
    Resta elemento a elemento: v1 - v2.

    :param v1: Vector minuendo
    :type v1: List[float]
    :param v2: Vector sustraendo
    :type v2: List[float]
    :return: Vector diferencia
    :rtype: List[float]
    :raises ValueError: Si los vectores tienen diferente longitud
    """
    if len(v1) != len(v2):
        raise ValueError(f"Vectores de diferente tamaño: {len(v1)} vs {len(v2)}")
    return [v1[i] - v2[i] for i in range(len(v1))]


def vector_scale(vector: List[float], scalar: float) -> List[float]:
    """
    Multiplica cada elemento del vector por un escalar.

    :param vector: Vector a escalar
    :type vector: List[float]
    :param scalar: Valor escalar
    :type scalar: float
    :return: Vector escalado
    :rtype: List[float]
    """
    return [v * scalar for v in vector]


def vector_dot(v1: List[float], v2: List[float]) -> float:
    """
    Producto punto (producto escalar) de dos vectores.

    :param v1: Primer vector
    :type v1: List[float]
    :param v2: Segundo vector
    :type v2: List[float]
    :return: Producto escalar
    :rtype: float
    :raises ValueError: Si los vectores tienen diferente longitud
    """
    if len(v1) != len(v2):
        raise ValueError(f"Vectores de diferente tamaño: {len(v1)} vs {len(v2)}")
    return sum(v1[i] * v2[i] for i in range(len(v1)))


# =============================
# ÁLGEBRA LINEAL - VECTORES
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
    :raises ValueError: Si las dimensiones no son compatibles
    """
    if not matrix:
        return []
    if len(matrix[0]) != len(vector):
        raise ValueError(
            f"Dimensiones incompatibles: matriz {len(matrix)}×{len(matrix[0])}, vector {len(vector)}"
        )
    return [vector_dot(row, vector) for row in matrix]


def outer_product(v_col: List[float], v_row: List[float]) -> List[List[float]]:
    """
    Producto externo: crea una matriz a partir de dos vectores.

    Resultado[i][j] = v_col[i] * v_row[j]

    :param v_col: Vector columna
    :type v_col: List[float]
    :param v_row: Vector fila
    :type v_row: List[float]
    :return: Matriz de tamaño len(v_col) × len(v_row)
    :rtype: List[List[float]]
    """
    return [[vc * vr for vr in v_row] for vc in v_col]


def matrix_transpose(matrix: List[List[float]]) -> List[List[float]]:
    """
    Transpone una matriz (convierte filas en columnas).

    :param matrix: Matriz de entrada
    :type matrix: List[List[float]]
    :return: Matriz transpuesta
    :rtype: List[List[float]]
    """
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]
