"""
Utilidades para carga y procesamiento de datos MNIST.
"""

from .mnist_loader import (
    load_mnist_train,
    load_mnist_test,
    load_mnist_complete,
    descargar_mnist,
    get_data_directory,
)

from .data_partitioner import (
    partition_mnist_data,
    partition_mnist_data_simple,
    get_partition_statistics,
    print_partition_summary,
    merge_partitions,
)

from .math_utils import (
    # Activaciones
    sigmoid,
    sigmoid_derivative_from_activation,
    softmax,
    # Vectores
    vector_add,
    vector_subtract,
    vector_dot,
    vector_scale,
    vector_zeros,
    # Matrices
    matrix_vector_multiply,
    matrix_transpose,
    outer_product,
    matrix_add,
    matrix_scale,
    matrix_zeros,
    matrix_random_normal,
    xavier_initialization,
    # Redes neuronales
    compute_one_hot,
    # Algoritmo de Diego
    average_vectors,
    average_matrices,
    average_network_parameters,
    # Gradientes
    accumulate_gradients,
    scale_gradients,
    initialize_gradient_accumulators,
    # Métricas
    argmax,
)

__all__ = [
    # MNIST Loader
    "load_mnist_train",
    "load_mnist_test",
    "load_mnist_complete",
    "descargar_mnist",
    "get_data_directory",
    # Data Partitioner
    "partition_mnist_data",
    "partition_mnist_data_simple",
    "get_partition_statistics",
    "print_partition_summary",
    "merge_partitions",
    # Math Utils
    "sigmoid",
    "sigmoid_derivative_from_activation",
    "softmax",
    "vector_add",
    "vector_subtract",
    "vector_dot",
    "vector_scale",
    "vector_zeros",
    "matrix_vector_multiply",
    "matrix_transpose",
    "outer_product",
    "matrix_add",
    "matrix_scale",
    "matrix_zeros",
    "matrix_random_normal",
    "xavier_initialization",
    "compute_one_hot",
    "average_vectors",
    "average_matrices",
    "average_network_parameters",
    "accumulate_gradients",
    "scale_gradients",
    "initialize_gradient_accumulators",
    "argmax",
]
