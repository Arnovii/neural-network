"""
Descarga y carga del dataset MNIST.

Utiliza torchvision para descargar y cargar los datos de forma sencilla.
Las imágenes se devuelven normalizadas al rango [0, 1].
"""

import os
import random
import numpy as np
from typing import List, Tuple


# =======================
# CONFIGURACIÓN DE RUTAS
# =======================


def get_data_directory() -> str:
    """
    Devuelve la ruta absoluta al directorio Data/ del proyecto.

    El directorio se crea automáticamente si no existe.

    :return: Ruta absoluta al directorio de datos
    :rtype: str
    """
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(utils_dir)
    data_dir = os.path.join(project_root, "Data")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Creado directorio de datos: {data_dir}")

    return data_dir


# ========================
# FUNCIONES PÚBLICAS
# ========================


def load_mnist_train(
    data_dir: str | None = None,
    n_train: int | None = None,
    download_if_missing: bool = True,
    verbose: bool = True,
    random_seed: int | None = None,
) -> Tuple[List[List[float]], List[int]]:
    """
    Carga el conjunto de entrenamiento de MNIST.

    :param data_dir: Directorio de datos. Si es None, usa Data/ por defecto.
    :type data_dir: str | None

    :param n_train: Número de ejemplos a cargar. Si es None, carga los 60 000.
    :type n_train: int | None

    :param download_if_missing: Si True, descarga los datos si no existen.
    :type download_if_missing: bool

    :param verbose: Si True, muestra mensajes de progreso.
    :type verbose: bool

    :param random_seed: Semilla para el submuestreo aleatorio.
    :type random_seed: int | None

    :return: Tupla (X_train, Y_train)
    :rtype: Tuple[List[List[float]], List[int]]

    :raises ValueError: Si n_train es menor que 1 o mayor que los datos disponibles.
    """
    from torchvision import datasets

    # Valida el n_train
    if n_train is not None and n_train < 1:
        raise ValueError(f"n_train debe ser >= 1, recibido: {n_train}")

    # Configura semilla si se proporciona
    if random_seed is not None:
        random.seed(random_seed)

    # Obtiene el directorio
    if data_dir is None:
        data_dir = get_data_directory()

    if verbose:
        print("=" * 60)
        print("CARGANDO DATOS MNIST — ENTRENAMIENTO")
        print("=" * 60)

    # Carga el dataset usando torchvision (transform=None: imágenes PIL, sin tensores)
    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download_if_missing,
        transform=None,
    )

    total_disponible = len(dataset)

    if n_train is not None and n_train > total_disponible:
        raise ValueError(
            f"n_train ({n_train}) supera los datos disponibles ({total_disponible})"
        )

    # Determina cuántas imágenes cargar
    cantidad = n_train if n_train is not None else total_disponible

    # Selecciona índices (submuestreo aleatorio si n_train < total)
    if n_train is not None and n_train < total_disponible:
        indices = random.sample(range(total_disponible), n_train)
    else:
        indices = list(range(cantidad))

    # Convierte cada imagen PIL a lista de 784 floats normalizados en [0, 1]
    X: List[List[float]] = []
    Y: List[int] = []

    for i in indices:
        imagen, etiqueta = dataset[i]
        # np.array(imagen) convierte PIL a array 28×28
        # .flatten() lo convierte a vector de 784
        # / 255.0 normaliza al rango [0, 1]
        pixeles = (np.array(imagen).flatten() / 255.0).tolist()
        X.append(pixeles)
        Y.append(int(etiqueta))

    if verbose:
        print(f"\n✓ {len(X)} ejemplos de entrenamiento cargados")

    return X, Y


def load_mnist_test(
    data_dir: str | None = None,
    download_if_missing: bool = True,
    verbose: bool = True,
) -> Tuple[List[List[float]], List[int]]:
    """
    Carga el conjunto de prueba de MNIST (10 000 ejemplos).

    :param data_dir: Directorio de datos. Si es None, usa Data/ por defecto.
    :type data_dir: str | None

    :param download_if_missing: Si True, descarga los datos si no existen.
    :type download_if_missing: bool

    :param verbose: Si True, muestra mensajes de progreso.
    :type verbose: bool

    :return: Tupla (X_test, Y_test)
    :rtype: Tuple[List[List[float]], List[int]]
    """
    from torchvision import datasets

    # Obtiene el directorio
    if data_dir is None:
        data_dir = get_data_directory()

    if verbose:
        print("=" * 60)
        print("CARGANDO DATOS MNIST — PRUEBA")
        print("=" * 60)

    # Carga el dataset de prueba usando torchvision
    dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download_if_missing,
        transform=None,
    )

    # Convierte cada imagen a lista de 784 floats normalizados
    X: List[List[float]] = []
    Y: List[int] = []

    for i in range(len(dataset)):
        imagen, etiqueta = dataset[i]
        pixeles = (np.array(imagen).flatten() / 255.0).tolist()
        X.append(pixeles)
        Y.append(int(etiqueta))

    if verbose:
        print(f"\n✓ {len(X)} ejemplos de prueba cargados")

    return X, Y
