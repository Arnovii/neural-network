"""
Este módulo proporciona funciones matemáticas básicas.\n
Incluye funciones de activación, álgebra lineal y operaciones para
el manejo de gradientes y parámetros.
"""

import math
import random
from typing import Any, Dict, List

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


def vector_zeros(size: int) -> List[float]:
    """
    Crea un vector de ceros.

    :param size: Tamaño del vector
    :type size: int

    :return: Vector de ceros
    :rtype: List[float]
    """
    return [0.0] * size


# =============================
# ÁLGEBRA LINEAL - MATRICES
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


def matrix_add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """
    Suma elemento a elemento de dos matrices.

    :param A: Primera matriz
    :type A: List[List[float]]

    :param B: Segunda matriz
    :type B: List[List[float]]

    :return: Matriz suma
    :rtype: List[List[float]]

    :raises ValueError: Si las matrices tienen dimensiones diferentes
    """
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Las matrices deben tener las mismas dimensiones")
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def matrix_scale(matrix: List[List[float]], scalar: float) -> List[List[float]]:
    """
    Multiplica cada elemento de la matriz por un escalar.

    :param matrix: Matriz a escalar
    :type matrix: List[List[float]]

    :param scalar: Valor escalar
    :type scalar: float

    :return: Matriz escalada
    :rtype: List[List[float]]
    """
    return [[elem * scalar for elem in row] for row in matrix]


def matrix_zeros(rows: int, cols: int) -> List[List[float]]:
    """
    Crea una matriz de ceros.

    :param rows: Número de filas
    :type rows: int

    :param cols: Número de columnas
    :type cols: int

    :return: Matriz de ceros
    :rtype: List[List[float]]
    """
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def matrix_random_normal(
    rows: int, cols: int, mean: float = 0.0, std: float = 1.0
) -> List[List[float]]:
    """
    Crea una matriz con valores aleatorios de distribución normal.

    :param rows: Número de filas
    :type rows: int

    :param cols: Número de columnas
    :type cols: int

    :param mean: Media de la distribución
    :type mean: float

    :param std: Desviación estándar
    :type std: float

    :return: Matriz con valores aleatorios
    :rtype: List[List[float]]
    """
    return [[random.gauss(mean, std) for _ in range(cols)] for _ in range(rows)]


def xavier_initialization(fan_in: int, fan_out: int) -> List[List[float]]:
    """
    Inicialización Xavier/Glorot para pesos de red neuronal.

    Los pesos se inicializan con distribución normal de media 0 y\n
    desviación estándar sqrt(2 / (fan_in + fan_out)).

    :param fan_in: Número de neuronas de entrada
    :type fan_in: int

    :param fan_out: Número de neuronas de salida
    :type fan_out: int

    :return: Matriz de pesos inicializada
    :rtype: List[List[float]]
    """
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return matrix_random_normal(fan_out, fan_in, mean=0.0, std=std)


# ===================================================
# OPERACIONES PARA REDES NEURONALES - GRADIENTES
# ===================================================


def compute_one_hot(label: int, num_classes: int) -> List[float]:
    """
    Crea un vector one-hot para una etiqueta.

    :param label: Índice de la clase (0 a num_classes-1)
    :type label: int

    :param num_classes: Número total de clases
    :type num_classes: int

    :return: Vector one-hot
    :rtype: List[float]

    :raises ValueError: Si label está fuera de rango
    """
    if label < 0 or label >= num_classes:
        raise ValueError(f"Etiqueta {label} fuera de rango [0, {num_classes})")
    one_hot = [0.0] * num_classes
    one_hot[label] = 1.0
    return one_hot


def initialize_gradient_accumulators(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inicializa acumuladores de gradientes en cero basándose en la estructura de parámetros.

    :param parameters: Diccionario con parámetros de la red
    :type parameters: Dict[str, Any]

    :return: Diccionario con gradientes inicializados en cero
    :rtype: Dict[str, Any]
    """
    accumulators = {}

    for key, value in parameters.items():
        if isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], list):
                # Matriz
                rows = len(value)
                cols = len(value[0])
                accumulators[key] = matrix_zeros(rows, cols)
            else:
                # Vector
                accumulators[key] = vector_zeros(len(value))
        else:
            # Escalar
            accumulators[key] = 0.0

    return accumulators


# ===============================================================
# OPERACIONES PARA ALGORITMO DE DIEGO - PROMEDIO DE PARÁMETROS
# ===============================================================


def average_vectors(vectors: List[List[float]]) -> List[float]:
    """
    Calcula el promedio de múltiples vectores.

    :param vectors: Lista de vectores (todos del mismo tamaño)
    :type vectors: List[List[float]]

    :return: Vector promedio
    :rtype: List[float]

    :raises ValueError: Si la lista está vacía o los vectores tienen tamaños diferentes
    """
    if not vectors:
        raise ValueError("No se puede promediar una lista vacía de vectores")

    n = len(vectors)
    size = len(vectors[0])

    # Verifica que todos tengan el mismo tamaño
    for v in vectors:
        if len(v) != size:
            raise ValueError("Todos los vectores deben tener el mismo tamaño")

    # Suma todos los vectores
    result = vector_zeros(size)
    for v in vectors:
        result = vector_add(result, v)

    # Divide por n
    return vector_scale(result, 1.0 / n)


def average_matrices(matrices: List[List[List[float]]]) -> List[List[float]]:
    """
    Calcula el promedio de múltiples matrices.

    :param matrices: Lista de matrices (todas del mismo tamaño)
    :type matrices: List[List[List[float]]]

    :return: Matriz promedio
    :rtype: List[List[float]]

    :raises ValueError: Si la lista está vacía o las matrices tienen dimensiones diferentes
    """
    if not matrices:
        raise ValueError("No se puede promediar una lista vacía de matrices")

    n = len(matrices)
    rows = len(matrices[0])
    cols = len(matrices[0][0]) if rows > 0 else 0

    # Verifica que todas tengan las mismas dimensiones
    for m in matrices:
        if len(m) != rows or (rows > 0 and len(m[0]) != cols):
            raise ValueError("Todas las matrices deben tener las mismas dimensiones")

    # Suma todas las matrices
    result = matrix_zeros(rows, cols)
    for m in matrices:
        result = matrix_add(result, m)

    # Divide por n
    return matrix_scale(result, 1.0 / n)


def average_network_parameters(parameters_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Promedia los parámetros de múltiples redes neuronales.

    Recibe una lista de diccionarios con parámetros (W1, b1, W2, b2, ...)\n
    y devuelve un diccionario con los parámetros promediados.

    Estructura esperada de cada diccionario:\n
    {
        'W1': List[List[float]],  # Pesos capa 1\n
        'b1': List[float],        # Biases capa 1\n
        'W2': List[List[float]],  # Pesos capa 2\n
        'b2': List[float],        # Biases capa 2\n
        ...\n
    }

    :param parameters_list: Lista de diccionarios con parámetros
    :type parameters_list: List[Dict[str, Any]]

    :return: Diccionario con parámetros promediados
    :rtype: Dict[str, Any]

    :raises ValueError: Si la lista está vacía o los diccionarios tienen claves diferentes
    """
    if not parameters_list:
        raise ValueError("No se puede promediar una lista vacía de parámetros")

    # Obtiene las claves del primer diccionario
    keys = parameters_list[0].keys()

    # Verifica que todos tengan las mismas claves
    for params in parameters_list:
        if set(params.keys()) != set(keys):
            raise ValueError(
                "Todos los diccionarios de parámetros deben tener las mismas claves"
            )

    averaged_params = {}

    for key in keys:
        # Determina si es matriz o vector basándose en el primer elemento
        first_value = parameters_list[0][key]

        if isinstance(first_value, list):
            if len(first_value) > 0 and isinstance(first_value[0], list):
                # Es una matriz
                matrices = [params[key] for params in parameters_list]
                averaged_params[key] = average_matrices(matrices)
            else:
                # Es un vector
                vectors = [params[key] for params in parameters_list]
                averaged_params[key] = average_vectors(vectors)
        else:
            # Es un escalar u otro tipo, simplemente promedia
            values = [params[key] for params in parameters_list]
            averaged_params[key] = sum(values) / len(values)

    return averaged_params


# =========================================
# UTILIDADES PARA GRADIENTES ACUMULADOS
# =========================================


def accumulate_gradients(
    acc_gradients: Dict[str, Any], new_gradients: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Acumula gradientes sumándolos a los acumuladores existentes.

    :param acc_gradients: Diccionario con gradientes acumulados
    :type acc_gradients: Dict[str, Any]

    :param new_gradients: Diccionario con nuevos gradientes a sumar
    :type new_gradients: Dict[str, Any]

    :return: Diccionario con gradientes acumulados actualizados
    :rtype: Dict[str, Any]
    """
    result = {}

    for key in acc_gradients:
        if key not in new_gradients:
            result[key] = acc_gradients[key]
            continue

        acc_val = acc_gradients[key]
        new_val = new_gradients[key]

        if isinstance(acc_val, list):
            if len(acc_val) > 0 and isinstance(acc_val[0], list):
                # Matriz
                result[key] = matrix_add(acc_val, new_val)
            else:
                # Vector
                result[key] = vector_add(acc_val, new_val)
        else:
            # Escalar
            result[key] = acc_val + new_val

    return result


def scale_gradients(gradients: Dict[str, Any], scale: float) -> Dict[str, Any]:
    """
    Escala todos los gradientes por un factor.

    Útil para aplicar la tasa de aprendizaje: gradiente * (lr / n)

    :param gradients: Diccionario con gradientes
    :type gradients: Dict[str, Any]

    :param scale: Factor de escala
    :type scale: float

    :return: Diccionario con gradientes escalados
    :rtype: Dict[str, Any]
    """
    result = {}

    for key, value in gradients.items():
        if isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], list):
                # Matriz
                result[key] = matrix_scale(value, scale)
            else:
                # Vector
                result[key] = vector_scale(value, scale)
        else:
            # Escalar
            result[key] = value * scale

    return result


# =======================
# MÉTRICAS Y EVALUACIÓN
# =======================


def argmax(vector: List[float]) -> int:
    """
    Encuentra el índice del valor máximo en un vector.

    :param vector: Vector de entrada
    :type vector: List[float]

    :return: Índice del valor máximo
    :rtype: int
    """
    return max(range(len(vector)), key=lambda i: vector[i])
