"""
Utils/data_partitioner.py
Particionamiento estratificado del dataset MNIST.
Opera nativamente con np.ndarray.
"""

import numpy as np
from typing import List, Tuple


def partition_mnist_data_simple(
    num_partitions: int,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    random_seed: int | None = None,
    verbose: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Divide datos en N particiones estratificadas por clase.

    :param num_partitions: Número de particiones (>= 1)
    :type num_partitions: int

    :param X_train: Imágenes de entrenamiento con forma (N, 784)
    :type X_train: np.ndarray

    :param Y_train: Etiquetas con forma (N,)
    :type Y_train: np.ndarray

    :param random_seed: Semilla para reproducibilidad
    :type random_seed: int | None

    :param verbose: Si True, muestra información por partición
    :type verbose: bool

    :return: Lista de tuplas (X_partition, Y_partition)
    :rtype: List[Tuple[np.ndarray, np.ndarray]]
    """
    if num_partitions < 1:
        raise ValueError("El número de particiones debe ser al menos 1")
    if len(X_train) != len(Y_train):
        raise ValueError("X_train e Y_train deben tener la misma longitud")

    if random_seed is not None:
        np.random.seed(random_seed)

    # Agrupa índices por clase y distribuye round-robin
    partitions_indices: List[List[int]] = [[] for _ in range(num_partitions)]

    for digit in range(10):
        indices = np.where(Y_train == digit)[0]
        np.random.shuffle(indices)
        for i, idx in enumerate(indices):
            partitions_indices[i % num_partitions].append(idx)

    # Mezcla cada partición
    result = []
    for i, part_idx in enumerate(partitions_indices):
        part_idx_arr = np.array(part_idx)
        np.random.shuffle(part_idx_arr)
        result.append((X_train[part_idx_arr], Y_train[part_idx_arr]))

    if verbose:
        print(f"✓ {num_partitions} particiones creadas:")
        for i, (X, _) in enumerate(result):
            print(f"  Partición {i + 1}: {len(X)} ejemplos")

    return result
