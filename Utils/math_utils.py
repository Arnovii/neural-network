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
    Función softmax con estabilización numérica.

    Funciona tanto con un vector 1-D (un solo ejemplo) como con una
    matriz 2-D de forma ``(clases, N)`` donde N es el tamaño del batch.

    - Vector 1-D ``(clases,)``: reduce sobre el único eje.
    - Matriz 2-D ``(clases, N)``: reduce por columna (``axis=0``), de
      modo que cada columna (cada ejemplo) produce sus propias
      probabilidades independientes.

    En ambos casos se resta el máximo antes de exponenciar para evitar
    desbordamiento numérico. El resultado tiene la misma forma que la entrada.

    :param z: Logits. Array de forma ``(clases,)`` o ``(clases, N)``
    :type z: np.ndarray

    :return: Probabilidades con la misma forma que ``z``
    :rtype: np.ndarray
    """
    # Define el eje según el número de dimensiones de z
    axis = 0 if z.ndim > 1 else None

    # keepdims=True preserva las dimensiones originales para que la resta
    # y la división sean compatibles tanto en 1-D como en 2-D
    z_stable = z - np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z_stable)

    # np.sum suma todos los elementos del arreglo
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


# =============================
# ÁLGEBRA LINEAL — VECTORES
# =============================


def vector_zeros(size: int) -> np.ndarray:
    """
    Crea un vector de ceros.

    :param size: Tamaño del vector
    :type size: int

    :return: Vector de ceros
    :rtype: np.ndarray
    """
    return np.zeros(size)


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
