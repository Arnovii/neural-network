"""
Utils/math_utils.py

Funciones matemáticas para la red neuronal — implementación NumPy nativa.

La razón de usar NumPy es que permite trabajar con vectores y matrices
de manera eficiente y rápida usando operaciones matemáticas optimizadas en C.

Todo trabaja directamente con np.ndarray (arreglos de NumPy),
no con listas normales de Python.
"""

import numpy as np
from typing import Any, Dict, List


# =========================
# FUNCIONES DE ACTIVACIÓN
# =========================


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Función sigmoide vectorizada: σ(z) = 1 / (1 + e^(−z)).

    - Recibe un número o un arreglo de números.
    - Aplica la fórmula elemento por elemento.
    - Devuelve un arreglo del mismo tamaño.

    :param z: Array de entrada (escalar o N-dimensional)
    :type z: np.ndarray

    :return: Array con valores en el rango (0, 1)
    :rtype: np.ndarray
    """
    # np.clip recorta valores fuera del rango entre -500 y 500
    z_safe = np.clip(z, -500, 500)

    # np.exp calcula e^x para cada elemento del arreglo
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
    # Multiplicación elemento a elemento del arreglo
    return a * (1.0 - a)


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Función softmax:
    Convierte un vector de valores arbitrarios (logits)
    en probabilidades que suman exactamente 1.

    :param z: Array 1-D de logits
    :type z: np.ndarray

    :return: Array 1-D de probabilidades
    :rtype: np.ndarray
    """
    # np.max obtiene el valor máximo del arreglo
    # Restarlo mejora estabilidad numérica
    z_stable = z - np.max(z)

    exp_z = np.exp(z_stable)

    # np.sum suma todos los elementos del arreglo
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
    # El operador @ hace multiplicación matricial
    return matrix @ vector


def matrix_transpose(matrix: np.ndarray) -> np.ndarray:
    """
    Transpone una matriz.

    Transponer significa intercambiar filas por columnas.

    :param matrix: Matriz original
    :type matrix: np.ndarray

    :return: Matriz transpuesta
    :rtype: np.ndarray
    """
    # .T es el atributo de NumPy para transponer
    return matrix.T


def outer_product(v_col: np.ndarray, v_row: np.ndarray) -> np.ndarray:
    """
    Producto externo de dos vectores.

    Si:
        v_col tiene tamaño (m)
        v_row tiene tamaño (n)

    El resultado es una matriz de tamaño (m x n).

    :param v_col: Vector columna
    :type v_col: np.ndarray

    :param v_row: Vector fila
    :type v_row: np.ndarray

    :return: Matriz resultado
    :rtype: np.ndarray
    """
    # np.outer calcula el producto externo
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
    Inicialización Xavier/Glorot:\n

    Los pesos se distribuyen con media 0 y desviación estándar (std):
    sqrt(2 / (fan_in + fan_out)), lo que ayuda a mantener la varianza
    de las activaciones estable a través de las capas, y evita que los
    gradientes exploten o desaparezcan.

    :param fan_in: Neuronas de entrada (columnas)
    :type fan_in: int

    :param fan_out: Neuronas de salida (filas)
    :type fan_out: int

    :return: Matriz (fan_out × fan_in)
    :rtype: np.ndarray
    """
    std = np.sqrt(2.0 / (fan_in + fan_out))

    # np.random.normal genera números aleatorios
    # con distribución normal (media 0, desviación std)
    return np.random.normal(0.0, std, size=(fan_out, fan_in))


# ===========================================
# UTILIDADES PARA EL ALGORITMO DE DIEGO
# ===========================================


def average_network_parameters(parameters_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Promedia múltiples conjuntos de parámetros de redes neuronales.

    :param parameters_list: Lista de diccionarios con W1, b1, W2, b2
    :type parameters_list: List[Dict[str, Any]]

    :return: Diccionario con parámetros promediados
    :rtype: Dict[str, Any]
    """
    if not parameters_list:
        raise ValueError("No se puede promediar una lista vacía de parámetros")

    averaged: Dict[str, Any] = {}

    # Recorre las claves del primer modelo ("W1", "b1", etc.)
    for key in parameters_list[0]:
        # np.array convierte la lista de Python en un array de NumPy, añadiendo una dimensión
        # La nueva dimensión (axis=0) representa el modelo
        stacked = np.array([p[key] for p in parameters_list])

        # np.mean calcula el promedio a lo largo del eje 0
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
    # np.argmax devuelve el índice del máximo
    return int(np.argmax(vector))


def compute_one_hot(label: int, num_classes: int) -> np.ndarray:
    """
    Crea un vector one-hot.

    Un vector one-hot es un vector lleno de ceros
    excepto en la posición correspondiente a la clase,
    donde se coloca un 1.

    :param label: Índice de la clase (0 a num_classes−1)
    :type label: int

    :param num_classes: Número total de clases
    :type num_classes: int

    :return: Vector con 1.0 en la posición label
    :rtype: np.ndarray
    """
    if label < 0 or label >= num_classes:
        raise ValueError(f"Etiqueta {label} fuera del rango [0, {num_classes}]")

    one_hot = np.zeros(num_classes)

    # Se coloca un 1.0 en la posición de la clase
    one_hot[label] = 1.0
    return one_hot
