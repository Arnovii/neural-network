"""
Particionamiento estratificado del dataset MNIST.

Garantiza que cada partición tenga una distribución de clases
proporcional a la del dataset completo.
"""

import random
from typing import Dict, List, Tuple


def partition_mnist_data_simple(
    num_partitions: int,
    X_train: List[List[float]],
    Y_train: List[int],
    random_seed: int | None = None,
    verbose: bool = False,
) -> List[Tuple[List[List[float]], List[int]]]:
    """
    Divide datos ya cargados en N particiones estratificadas por clase.

    El particionamiento es estratificado: los índices de cada dígito (0-9)
    se mezclan aleatoriamente y se distribuyen de forma round-robin entre
    las particiones, asegurando que cada una tenga una proporción similar
    de cada clase. Esto es clave para que el algoritmo de Diego arranque
    con condiciones equivalentes en cada partición.

    :param num_partitions: Número de particiones (debe ser >= 1)
    :type num_partitions: int

    :param X_train: Lista de imágenes de entrenamiento
    :type X_train: List[List[float]]

    :param Y_train: Lista de etiquetas correspondientes
    :type Y_train: List[int]

    :param random_seed: Semilla para reproducibilidad
    :type random_seed: int | None

    :param verbose: Si True, muestra el número de ejemplos por partición
    :type verbose: bool

    :return: Lista de tuplas (X_partition, Y_partition)
    :rtype: List[Tuple[List[List[float]], List[int]]]

    :raises ValueError: Si num_partitions < 1 o X_train e Y_train tienen tamaños distintos
    """
    if num_partitions < 1:
        raise ValueError("El número de particiones debe ser al menos 1")
    if len(X_train) != len(Y_train):
        raise ValueError("X_train e Y_train deben tener la misma longitud")

    # Establece semilla si se especificó
    if random_seed is not None:
        random.seed(random_seed)

    # Agrupa índices por clase
    indices_by_class: Dict[int, List[int]] = {d: [] for d in range(10)}
    for idx, label in enumerate(Y_train):
        indices_by_class[label].append(idx)

    # Distribuye tipo round-robin por clase para garantizar estratificación
    partitions: List[List[int]] = [[] for _ in range(num_partitions)]
    for digit in range(10):
        indices = indices_by_class[digit][:]
        random.shuffle(indices)
        for i, idx in enumerate(indices):
            partitions[i % num_partitions].append(idx)

    # Mezcla cada partición para evitar que los ejemplos queden ordenados por clase
    for partition in partitions:
        random.shuffle(partition)

    # Crea resultados
    result = [
        ([X_train[i] for i in part], [Y_train[i] for i in part]) for part in partitions
    ]

    if verbose:
        print(f"✓ {num_partitions} particiones creadas:")
        for i, (X, _) in enumerate(result):
            print(f"  Partición {i + 1}: {len(X)} ejemplos")

    return result
