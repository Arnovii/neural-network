"""
Descarga y carga del dataset MNIST.

Los archivos se almacenan en el directorio Data/ relativo a la raíz del proyecto.
Las imágenes se devuelven normalizadas al rango [0, 1].
"""

import gzip
import os
import random
import struct
import urllib.request
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


# ==================
# DESCARGA DE DATOS
# ==================

_MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

_MNIST_URLS_FALLBACK = {
    "train-images-idx3-ubyte.gz": "https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz",
}


def _descargar_mnist(data_dir: str, verbose: bool) -> None:
    """
    Descarga los cuatro archivos MNIST al directorio indicado.

    Intenta primero el mirror de AWS; si falla, usa GitHub como respaldo.

    :param data_dir: Directorio de destino
    :type data_dir: str

    :param verbose: Si True, muestra mensajes de progreso
    :type verbose: bool

    :raises Exception: Si ningún mirror responde correctamente
    """
    for archivo, url in _MNIST_URLS.items():
        filepath = os.path.join(data_dir, archivo)
        if os.path.exists(filepath):
            if verbose:
                print(f"  ✓ {archivo} ya existe")
            continue

        if verbose:
            print(f"  Descargando {archivo}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            if verbose:
                print(f"  ✓ {archivo} descargado desde AWS")
        except Exception:
            if verbose:
                print("  Reintentando desde GitHub...")
            try:
                urllib.request.urlretrieve(_MNIST_URLS_FALLBACK[archivo], filepath)
                if verbose:
                    print(f"  ✓ {archivo} descargado desde GitHub")
            except Exception:
                raise Exception(f"No se pudo descargar {archivo} desde ningún mirror")


# ================================
# CARGA DE IMÁGENES Y ETIQUETAS
# ================================


def _cargar_imagenes(archivo_gz: str) -> List[List[float]]:
    """
    Carga imágenes desde un archivo MNIST en formato IDX comprimido.

    :param archivo_gz: Ruta al archivo .gz
    :type archivo_gz: str

    :return: Lista de imágenes; cada imagen es una lista de 784 floats en [0, 1]
    :rtype: List[List[float]]

    :raises ValueError: Si el archivo no tiene el magic number correcto
    """
    with gzip.open(archivo_gz, "rb") as f:
        magic, num, filas, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(
                f"Magic number incorrecto: esperado 2051, obtenido {magic}"
            )
        buffer = f.read()
        pixeles_por_imagen = filas * cols
        return [
            [
                b / 255.0
                for b in buffer[i * pixeles_por_imagen : (i + 1) * pixeles_por_imagen]
            ]
            for i in range(num)
        ]


def _cargar_etiquetas(archivo_gz: str) -> List[int]:
    """
    Carga etiquetas desde un archivo MNIST en formato IDX comprimido.

    :param archivo_gz: Ruta al archivo .gz
    :type archivo_gz: str

    :return: Lista de enteros en el rango [0, 9]
    :rtype: List[int]

    :raises ValueError: Si el archivo no tiene el magic number correcto
    """
    with gzip.open(archivo_gz, "rb") as f:
        magic, _ = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(
                f"Magic number incorrecto: esperado 2049, obtenido {magic}"
            )
        return list(f.read())


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

    # Descarga si es necesario
    if download_if_missing:
        _descargar_mnist(data_dir, verbose=verbose)

    # Carga datos desde las rutas de los archivos
    X = _cargar_imagenes(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    Y = _cargar_etiquetas(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))

    # Válida integridad
    if len(X) != len(Y):
        raise RuntimeError("Inconsistencia: número de imágenes y etiquetas no coincide")

    if n_train is None:
        if verbose:
            print(f"\n✓ {len(X)} ejemplos de entrenamiento cargados")
        return X, Y

    if n_train > len(X):
        raise ValueError(f"n_train ({n_train}) supera los datos disponibles ({len(X)})")

    # Selecciona de forma aleatoria valores para los subconjuntos
    indices = random.sample(range(len(X)), n_train)
    X_sub = [X[i] for i in indices]
    Y_sub = [Y[i] for i in indices]

    if verbose:
        print(f"\n✓ {n_train} ejemplos de entrenamiento seleccionados")

    return X_sub, Y_sub


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

    # Obtiene el directorio
    if data_dir is None:
        data_dir = get_data_directory()

    if verbose:
        print("=" * 60)
        print("CARGANDO DATOS MNIST — PRUEBA")
        print("=" * 60)

    # Descarga si es necesario
    if download_if_missing:
        _descargar_mnist(data_dir, verbose=verbose)

    X = _cargar_imagenes(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    Y = _cargar_etiquetas(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

    if verbose:
        print(f"\n✓ {len(X)} ejemplos de prueba cargados")

    return X, Y
