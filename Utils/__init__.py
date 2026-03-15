"""
Utils — Utilidades de soporte para aprendizaje distribuido.

Módulos
-------
math_utils        Funciones matemáticas: sigmoid, softmax, xavier_initialization, etc.
mnist_loader      Cargadores de datos MNIST.
cifar_loader      Cargadores de datos CIFAR-10.
data_partitioner  Partición de datos para múltiples workers.
results_exporter  Exportación de resultados a JSON.

Uso rápido
----------
    from Utils import (
        load_cifar10_train, load_cifar10_test, NUM_CLASSES,
        sigmoid, softmax, xavier_initialization,
        export_results, partition_mnist_data_simple
    )
"""

# Math utilities
from Utils.math_utils import (
    sigmoid,
    sigmoid_derivative_from_activation,
    softmax,
    vector_zeros,
    xavier_initialization,
    average_arrays_dict,
)

# MNIST loaders
from Utils.mnist_loader import (
    load_mnist_train,
    load_mnist_test,
    load_mnist_labels,
    get_data_directory,
)

# CIFAR-10 loaders
from Utils.cifar_loader import (
    load_cifar10_train,
    load_cifar10_test,
    NUM_CLASSES,
)

# Data partitioning
from Utils.data_partitioner import partition_mnist_data_simple

# Results export
from Utils.results_exporter import export_results

__all__ = [
    # Math utilities
    "sigmoid",
    "sigmoid_derivative_from_activation",
    "softmax",
    "vector_zeros",
    "xavier_initialization",
    "average_arrays_dict",
    # MNIST
    "load_mnist_train",
    "load_mnist_test",
    "load_mnist_labels",
    "get_data_directory",
    # CIFAR-10
    "load_cifar10_train",
    "load_cifar10_test",
    "NUM_CLASSES",
    # Data partitioning
    "partition_mnist_data_simple",
    # Results export
    "export_results",
]
