"""
Utils/mnist_loader.py
Descarga y carga del dataset MNIST usando torchvision.
Las imágenes se devuelven como np.ndarray normalizadas al rango [0, 1].
"""

import os
import numpy as np
from typing import Tuple


def get_data_directory() -> str:
    """Devuelve la ruta absoluta al directorio Data/ del proyecto."""
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(utils_dir)
    data_dir = os.path.join(project_root, "Data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def load_mnist_train(
    data_dir: str | None = None,
    n_train: int | None = None,
    download_if_missing: bool = True,
    verbose: bool = True,
    random_seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga el conjunto de entrenamiento de MNIST.

    :return: Tupla (X_train, Y_train) con X de forma (N, 784) y Y de forma (N,)
    """
    from torchvision import datasets

    if n_train is not None and n_train < 1:
        raise ValueError(f"n_train debe ser >= 1, recibido: {n_train}")
    if random_seed is not None:
        np.random.seed(random_seed)
    if data_dir is None:
        data_dir = get_data_directory()

    if verbose:
        print("=" * 60)
        print("CARGANDO DATOS MNIST — ENTRENAMIENTO")
        print("=" * 60)

    dataset = datasets.MNIST(root=data_dir, train=True,
                             download=download_if_missing, transform=None)

    total = len(dataset)
    if n_train is not None and n_train > total:
        raise ValueError(f"n_train ({n_train}) supera los datos disponibles ({total})")

    cantidad = n_train if n_train is not None else total

    if n_train is not None and n_train < total:
        indices = np.random.choice(total, n_train, replace=False)
    else:
        indices = np.arange(cantidad)

    X = np.zeros((len(indices), 784))
    Y = np.zeros(len(indices), dtype=int)

    for out_idx, ds_idx in enumerate(indices):
        imagen, etiqueta = dataset[int(ds_idx)]
        X[out_idx] = np.array(imagen).flatten() / 255.0
        Y[out_idx] = int(etiqueta)

    if verbose:
        print(f"\n✓ {len(X)} ejemplos de entrenamiento cargados")

    return X, Y


def load_mnist_test(
    data_dir: str | None = None,
    download_if_missing: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga el conjunto de prueba de MNIST (10,000 ejemplos).

    :return: Tupla (X_test, Y_test) con X de forma (N, 784) y Y de forma (N,)
    """
    from torchvision import datasets

    if data_dir is None:
        data_dir = get_data_directory()

    if verbose:
        print("=" * 60)
        print("CARGANDO DATOS MNIST — PRUEBA")
        print("=" * 60)

    dataset = datasets.MNIST(root=data_dir, train=False,
                             download=download_if_missing, transform=None)

    X = np.zeros((len(dataset), 784))
    Y = np.zeros(len(dataset), dtype=int)

    for i in range(len(dataset)):
        imagen, etiqueta = dataset[i]
        X[i] = np.array(imagen).flatten() / 255.0
        Y[i] = int(etiqueta)

    if verbose:
        print(f"\n✓ {len(X)} ejemplos de prueba cargados")

    return X, Y
