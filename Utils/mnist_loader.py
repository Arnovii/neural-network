"""
Utils/mnist_loader.py

Descarga y carga del dataset MNIST usando torchvision.
Las imágenes se devuelven como np.ndarray normalizadas al rango [0, 1].
"""

import os
import numpy as np
from typing import Tuple

# =======================
# CONFIGURACIÓN DE RUTAS
# =======================


def get_data_directory() -> str:
    """
    Devuelve la ruta absoluta al directorio Data/ del proyecto.
    Si el directorio no existe, se crea automáticamente.
    """
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(utils_dir)
    data_dir = os.path.join(project_root, "Data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


# ==================
# DESCARGA DE DATOS
# ==================


def load_mnist_train(
    data_dir: str | None = None,
    n_train: int | None = None,
    download_if_missing: bool = True,
    verbose: bool = True,
    random_seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga el conjunto de entrenamiento de MNIST.

    :param data_dir: Directorio donde se almacenará/buscará el dataset.
                     Si es None, se usa el directorio Data/ del proyecto.
    :type data_dir: str | None

    :param n_train: Número de ejemplos de entrenamiento a cargar.
                    Si es None, se cargan todos los disponibles (60,000).
                    Si es menor al total, se selecciona una muestra aleatoria.
    :type n_train: int | None

    :param download_if_missing: Si True, descarga el dataset automáticamente
                                si aun no existe de manera local.
    :type download_if_missing: bool

    :param verbose: Si True, imprime información de progreso en consola.
    :type verbose: bool

    :param random_seed: Semilla para la generación aleatoria cuando se
                        selecciona un subconjunto (n_train < total).
                        Permite reproducibilidad.
    :type random_seed: int | None

    :raises ValueError:\n
        - Si n_train < 1\n
        - Si n_train supera el número total de ejemplos disponibles

    :return: Tupla (X_train, Y_train)\n
        - X_train: Array de forma (N, 784) con valores normalizados en [0, 1]
        - Y_train: Array de forma (N,) con etiquetas enteras en {0,...,9}
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    from torchvision import datasets

    if n_train is not None and n_train < 1:
        raise ValueError(f"n_train debe ser >= 1, recibido: {n_train}")

    # np.random.seed fija la semilla para que las selecciones aleatorias
    # sean reproducibles
    if random_seed is not None:
        np.random.seed(random_seed)

    if data_dir is None:
        data_dir = get_data_directory()

    if verbose:
        print("=" * 60)
        print("CARGANDO DATOS MNIST — ENTRENAMIENTO")
        print("=" * 60)

    dataset = datasets.MNIST(
        root=data_dir, train=True, download=download_if_missing, transform=None
    )

    # Devuelve el total de imágenes disponibles
    total = len(dataset)

    if n_train is not None and n_train > total:
        raise ValueError(f"n_train ({n_train}) supera los datos disponibles ({total})")

    # Determina cuántos ejemplos se usarán
    cantidad = n_train if n_train is not None else total

    if n_train is not None and n_train < total:
        # np.random.choice selecciona índices aleatorios sin repetición
        indices = np.random.choice(total, n_train, replace=False)
    else:
        # np.arange crea un arreglo [0, 1, 2, ..., cantidad-1]
        indices = np.arange(cantidad)

    X = np.zeros((len(indices), 784))
    Y = np.zeros(len(indices), dtype=int)

    for out_idx, ds_idx in enumerate(indices):
        imagen, etiqueta = dataset[int(ds_idx)]

        # flatten() convierte la matriz 28x28 en vector de 784
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

    :param data_dir: Directorio donde se almacenará/buscará el dataset.
                     Si es None, se usa el directorio Data/ del proyecto.
    :type data_dir: str | None

    :param download_if_missing: Si True, descarga el dataset automáticamente
                                si aun no existe de manera local.
    :type download_if_missing: bool

    :param verbose: Si True, imprime información de progreso en consola.
    :type verbose: bool

    :return: Tupla (X_test, Y_test)\n
        - X_test: Array de forma (N, 784) con valores normalizados en [0, 1]
        - Y_test: Array de forma (N,) con etiquetas enteras en {0,...,9}
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    from torchvision import datasets

    if data_dir is None:
        data_dir = get_data_directory()

    if verbose:
        print("=" * 60)
        print("CARGANDO DATOS MNIST — PRUEBA")
        print("=" * 60)

    dataset = datasets.MNIST(
        root=data_dir, train=False, download=download_if_missing, transform=None
    )

    X = np.zeros((len(dataset), 784))
    Y = np.zeros(len(dataset), dtype=int)

    for i in range(len(dataset)):
        imagen, etiqueta = dataset[i]

        # Conversión a vector normalizado
        X[i] = np.array(imagen).flatten() / 255.0

        Y[i] = int(etiqueta)

    if verbose:
        print(f"\n✓ {len(X)} ejemplos de prueba cargados")

    return X, Y


# ========================
# CARGA SOLO DE ETIQUETAS
# ========================


def load_mnist_labels(
    data_dir: str | None = None,
    n_train: int | None = None,
    download_if_missing: bool = True,
) -> np.ndarray:
    """
    Carga únicamente las etiquetas del conjunto de entrenamiento de MNIST.

    Lee el archivo IDX de etiquetas directamente (``train-labels-idx1-ubyte``,
    ~60 KB), sin tocar el archivo de imágenes (~47 MB). Útil cuando solo
    se necesitan las clases, por ejemplo para la partición estratificada
    del Parameter Server.

    Si el archivo todavía no existe en disco, lo descarga usando
    ``torchvision`` (que también descarga las imágenes en ese primer uso,
    pero eso solo ocurre una vez).

    Formato IDX de etiquetas:
        Bytes 0–3 : número mágico (0x00000801)
        Bytes 4–7 : número total de etiquetas (int32 big-endian)
        Bytes 8…  : una etiqueta por byte (uint8), valor en {0, …, 9}

    :param data_dir: Directorio raíz de MNIST (contiene ``MNIST/raw/``).
                     Si es None, se usa el directorio ``Data/`` del proyecto.
    :type data_dir: str | None

    :param n_train: Número de etiquetas a devolver.
                    Si es None se devuelven las 60 000.
                    Si es menor al total se devuelven las primeras ``n_train``
                    (mismas posiciones que usaría ``load_mnist_train``
                    sin semilla, es decir, en orden natural).
    :type n_train: int | None

    :param download_if_missing: Si True y el archivo no existe, lo descarga
                                mediante torchvision antes de leerlo.
    :type download_if_missing: bool

    :return: Array de forma ``(n_train,)`` con etiquetas en ``{0, …, 9}``,
             dtype ``int32``.
    :rtype: np.ndarray

    :raises FileNotFoundError: Si el archivo no existe y
                               ``download_if_missing=False``.
    :raises ValueError: Si ``n_train`` supera el total disponible.
    """
    import struct

    if data_dir is None:
        data_dir = get_data_directory()

    label_path = os.path.join(data_dir, "MNIST", "raw", "train-labels-idx1-ubyte")

    # Si el archivo no existe, usa torchvision solo para descargarlo
    if not os.path.exists(label_path):
        if not download_if_missing:
            raise FileNotFoundError(
                f"Archivo de etiquetas no encontrado: {label_path}\n"
                "Pasa download_if_missing=True para descargarlo."
            )
        from torchvision import datasets

        datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=None,
        )

    # Lee el archivo IDX directamente — sin cargar imágenes
    with open(label_path, "rb") as f:
        magic, total = struct.unpack(">II", f.read(8))
        if magic != 0x00000801:
            raise ValueError(
                f"Número mágico inesperado: {magic:#010x} (se esperaba 0x00000801)"
            )

        if n_train is not None and n_train > total:
            raise ValueError(
                f"n_train ({n_train}) supera los ejemplos disponibles ({total})"
            )

        count = n_train if n_train is not None else total
        raw = f.read(count)  # un byte por etiqueta

    return np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
