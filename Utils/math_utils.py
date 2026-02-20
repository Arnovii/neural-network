"""
Utils/math_utils.py
Funciones matemáticas para la red neuronal — implementación NumPy nativa.

Todas las funciones operan directamente con np.ndarray.
No se usan listas de Python como tipo de dato principal.
"""

import numpy as np
from typing import Any, Dict, List


# =========================
# FUNCIONES DE ACTIVACIÓN
# =========================


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Función sigmoide vectorizada: σ(z) = 1 / (1 + e^(−z)).

    :param z: Array de entrada (escalar o N-dimensional)
    :type z: np.ndarray

    :return: Array con valores en el rango (0, 1)
    :rtype: np.ndarray
    """
    z_safe = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_safe))


def sigmoid_derivative_from_activation(a: np.ndarray) -> np.ndarray:
    """
    Derivada de la sigmoide a partir de la activación: a · (1 − a).

    σ'(z) = σ(z) · (1 − σ(z)) = a · (1 − a)

    :param a: Activación (resultado previo de sigmoid)
    :type a: np.ndarray

    :return: Derivada evaluada en z
    :rtype: np.ndarray
    """
    return a * (1.0 - a)


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Función softmax: convierte logits a probabilidades que suman 1.

    :param z: Array 1-D de logits
    :type z: np.ndarray

    :return: Array 1-D de probabilidades
    :rtype: np.ndarray
    """
    z_stable = z - np.max(z)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z)


# =============================
# ÁLGEBRA LINEAL — VECTORES
# =============================


def vector_add(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Suma elemento a elemento de dos vectores.

    :param v1: Primer vector
    :type v1: np.ndarray

    :param v2: Segundo vector
    :type v2: np.ndarray

    :return: Vector suma
    :rtype: np.ndarray
    """
    return v1 + v2


def vector_subtract(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Resta elemento a elemento: v1 − v2.

    :param v1: Vector minuendo
    :type v1: np.ndarray

    :param v2: Vector sustraendo
    :type v2: np.ndarray

    :return: Vector resultado
    :rtype: np.ndarray
    """
    return v1 - v2


def vector_zeros(size: int) -> np.ndarray:
    """
    Crea un vector de ceros.

    :param size: Tamaño del vector
    :type size: int

    :return: Vector de ceros
    :rtype: np.ndarray
    """
    return np.zeros(size)


# =============================
# ÁLGEBRA LINEAL — MATRICES
# =============================


def matrix_vector_multiply(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Multiplicación matriz × vector.

    :param matrix: Matriz 2-D
    :type matrix: np.ndarray

    :param vector: Vector compatible
    :type vector: np.ndarray

    :return: Vector resultado
    :rtype: np.ndarray
    """
    return matrix @ vector


def matrix_transpose(matrix: np.ndarray) -> np.ndarray:
    """
    Transpone una matriz.

    :param matrix: Matriz original
    :type matrix: np.ndarray

    :return: Matriz transpuesta
    :rtype: np.ndarray
    """
    return matrix.T


def outer_product(v_col: np.ndarray, v_row: np.ndarray) -> np.ndarray:
    """
    Producto externo de dos vectores.

    :param v_col: Vector columna
    :type v_col: np.ndarray

    :param v_row: Vector fila
    :type v_row: np.ndarray

    :return: Matriz resultado
    :rtype: np.ndarray
    """
    return np.outer(v_col, v_row)


def matrix_add(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Suma elemento a elemento de dos matrices.

    :param A: Primera matriz
    :type A: np.ndarray

    :param B: Segunda matriz
    :type B: np.ndarray

    :return: Matriz suma
    :rtype: np.ndarray
    """
    return A + B


# ==============================
# INICIALIZACIÓN DE PARÁMETROS
# ==============================


def xavier_initialization(fan_in: int, fan_out: int) -> np.ndarray:
    """
    Inicialización Xavier/Glorot: media 0, std = sqrt(2 / (fan_in + fan_out)).

    Los pesos se distribuyen con media 0 y desviación estándar
    sqrt(2 / (fan_in + fan_out)), lo que ayuda a mantener la varianza
    de las activaciones estable a través de las capas.

    :param fan_in: Neuronas de entrada (columnas)
    :type fan_in: int

    :param fan_out: Neuronas de salida (filas)
    :type fan_out: int

    :return: Matriz (fan_out × fan_in)
    :rtype: np.ndarray
    """
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0.0, std, size=(fan_out, fan_in))


# ===========================================
# UTILIDADES PARA EL ALGORITMO DE DIEGO
# ===========================================


def average_network_parameters(parameters_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Promedia los parámetros de múltiples redes neuronales.

    :param parameters_list: Lista de diccionarios con W1, b1, W2, b2
    :type parameters_list: List[Dict[str, Any]]

    :return: Diccionario con parámetros promediados
    :rtype: Dict[str, Any]
    """
    if not parameters_list:
        raise ValueError("No se puede promediar una lista vacía de parámetros")

    averaged: Dict[str, Any] = {}
    for key in parameters_list[0]:
        stacked = np.array([p[key] for p in parameters_list])
        averaged[key] = np.mean(stacked, axis=0)
    return averaged


def accumulate_outer_inplace(
    acc: np.ndarray, v_col: np.ndarray, v_row: np.ndarray
) -> None:
    """
    Acumula el producto externo v_col ⊗ v_row directamente sobre acc.

    :param acc: Matriz acumuladora
    :type acc: np.ndarray

    :param v_col: Vector columna
    :type v_col: np.ndarray

    :param v_row: Vector fila
    :type v_row: np.ndarray
    """
    acc += np.outer(v_col, v_row)


def accumulate_vector_inplace(acc: np.ndarray, v: np.ndarray) -> None:
    """
    Acumula v sobre acc in-place.

    :param acc: Vector acumulador
    :type acc: np.ndarray

    :param v: Vector a acumular
    :type v: np.ndarray
    """
    acc += v


# ======================
# UTILIDADES GENERALES
# ======================


def argmax(vector: np.ndarray) -> int:
    """
    Devuelve el índice del valor máximo.

    :param vector: Vector de entrada
    :type vector: np.ndarray

    :return: Índice del máximo
    :rtype: int
    """
    return int(np.argmax(vector))


def compute_one_hot(label: int, num_classes: int) -> np.ndarray:
    """
    Crea un vector one-hot.

    :param label: Índice de la clase (0 a num_classes−1)
    :type label: int

    :param num_classes: Número total de clases
    :type num_classes: int

    :return: Vector con 1.0 en la posición label
    :rtype: np.ndarray
    """
    if label < 0 or label >= num_classes:
        raise ValueError(f"Etiqueta {label} fuera del rango [0, {num_classes})")
    one_hot = np.zeros(num_classes)
    one_hot[label] = 1.0
    return one_hot
