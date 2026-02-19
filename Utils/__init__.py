"""
Utils — Utilidades de soporte para NN_practica.

Expone las funciones matemáticas, el cargador de MNIST y el particionador
de datos que usan los módulos de red neuronal y análisis.
"""

from Utils.math_utils import (
    # Activaciones
    sigmoid,
    sigmoid_derivative_from_activation,
    softmax,
    # Vectores
    vector_add,
    vector_subtract,
    vector_zeros,
    # Matrices
    matrix_add,
    matrix_transpose,
    matrix_vector_multiply,
    outer_product,
    # Inicialización
    xavier_initialization,
    # Algoritmo de Diego
    average_network_parameters,
    accumulate_outer_inplace,
    accumulate_vector_inplace,
    # Generales
    argmax,
    compute_one_hot,
)

from Utils.mnist_loader import (
    load_mnist_train,
    load_mnist_test,
)

from Utils.data_partitioner import (
    partition_mnist_data_simple,
)

__all__ = [
    # math_utils
    "sigmoid",
    "sigmoid_derivative_from_activation",
    "softmax",
    "vector_add",
    "vector_subtract",
    "vector_zeros",
    "matrix_add",
    "matrix_transpose",
    "matrix_vector_multiply",
    "outer_product",
    "xavier_initialization",
    "average_network_parameters",
    "accumulate_outer_inplace",
    "accumulate_vector_inplace",
    "argmax",
    "compute_one_hot",
    # mnist_loader
    "load_mnist_train",
    "load_mnist_test",
    # data_partitioner
    "partition_mnist_data_simple",
]
