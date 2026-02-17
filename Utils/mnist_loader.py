"""
Este módulo proporciona funciones para descargar y cargar el dataset MNIST
de manera centralizadas.

Los datos se almacenan en el directorio 'Data/' relativo a la ubicación
de este archivo.
"""

import os
import gzip
import struct
import urllib.request
from typing import List, Tuple


# =======================
# CONFIGURACIÓN DE RUTAS
# =======================


def get_data_directory() -> str:
    """
    Obtiene la ruta absoluta al directorio de datos.
    El directorio de datos está ubicado en 'Data/' relativo al directorio
    padre de Utils/ (es decir, al directorio raíz del proyecto).

    :return: Ruta absoluta al directorio Data/
    """
    # Directorio donde está este archivo (Utils/)
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    # Directorio padre (raíz del proyecto)
    project_root = os.path.dirname(utils_dir)
    # Directorio de datos
    data_dir = os.path.join(project_root, "Data")

    # Crea directorio Data\ si no existe
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Creado directorio de datos: {data_dir}")

    return data_dir


# ==================
# DESCARGA DE DATOS
# ==================

MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

MNIST_URLS_FALLBACK = {
    "train-images-idx3-ubyte.gz": "https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz",
}


def descargar_mnist(data_dir: str | None = None, verbose: bool = True) -> str:
    """
    Descarga MNIST desde mirrors alternativos confiables.
    Fuentes: AWS Open Data (mirror oficial) o GitHub (fallback)

    :param data_dir: Directorio donde descargar los archivos. Si es None, se usa el directorio Data/ por defecto.
    :type data_dir: Optional[str]
    :param verbose: Si True, muestra mensajes de progreso.
    :type verbose: bool
    :return: Ruta al directorio donde se descargaron los archivos.
    :rtype: str
    """
    if data_dir is None:
        data_dir = get_data_directory()

    for archivo, url in MNIST_URLS.items():
        filepath = os.path.join(data_dir, archivo)

        if not os.path.exists(filepath):
            if verbose:
                print(f"Descargando {archivo}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                if verbose:
                    print(f"  ✓ {archivo} descargado desde AWS")
            except Exception as e:
                if verbose:
                    print(f"  Error en AWS, intentando GitHub...")
                try:
                    urllib.request.urlretrieve(MNIST_URLS_FALLBACK[archivo], filepath)
                    if verbose:
                        print(f"  ✓ {archivo} descargado desde GitHub")
                except Exception as e2:
                    if verbose:
                        print(f"  ERROR: No se pudo descargar {archivo}")
                    raise Exception(f"No se pudieron descargar los datos: {archivo}")
        else:
            if verbose:
                print(f"  ✓ {archivo} ya existe")

    return data_dir


# ===============================
# CARGA DE IMÁGENES Y ETIQUETAS
# ===============================


def cargar_imagenes(archivo_gz: str, verbose: bool = False) -> List[List[float]]:
    """
    Carga imágenes de MNIST desde archivo .gz (formato IDX).

    :param archivo_gz: Ruta al archivo .gz con las imágenes.
    :type archivo_gz: str
    :param verbose: Si True, muestra información de carga.
    :type verbose: bool
    :return: Lista de imágenes, donde cada imagen es una lista de 784 floats normalizados en el rango [0, 1].
    :rtype: List[List[float]]
    """
    if verbose:
        print(f"  Cargando imágenes desde: {archivo_gz}")

    with gzip.open(archivo_gz, "rb") as f:
        magic, num, filas, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(
                f"Magic number incorrecto: esperado 2051, obtenido {magic}"
            )

        buffer = f.read()
        imagenes = []

        for i in range(num):
            inicio = i * filas * cols
            fin = inicio + filas * cols
            img = [b / 255.0 for b in buffer[inicio:fin]]
            imagenes.append(img)

        if verbose:
            print(f"  ✓ Cargadas {num} imágenes ({filas}x{cols})")

        return imagenes


def cargar_etiquetas(archivo_gz: str, verbose: bool = False) -> List[int]:
    """
    Carga etiquetas de MNIST desde archivo .gz (formato IDX).

    :param archivo_gz: Ruta al archivo .gz con las etiquetas.
    :type archivo_gz: str
    :param verbose: Si True, muestra información de carga.
    :type verbose: bool
    :return: Lista de etiquetas (enteros 0-9).
    :rtype: List[int]

    """
    if verbose:
        print(f"  Cargando etiquetas desde: {archivo_gz}")

    with gzip.open(archivo_gz, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(
                f"Magic number incorrecto: esperado 2049, obtenido {magic}"
            )

        buffer = f.read()
        etiquetas = [b for b in buffer]

        if verbose:
            print(f"  ✓ Cargadas {num} etiquetas")

        return etiquetas


# ========================
# FUNCIONES DE ALTO NIVEL
# =========================


def load_mnist_train(
    data_dir: str | None = None,
    n_train: int = 5000,
    download_if_missing: bool = True,
    verbose: bool = True,
) -> Tuple[List[List[float]], List[int]]:
    """
    Carga el conjunto de entrenamiento de MNIST.

    :param data_dir: Directorio de datos. Si es None, usa Data/ por defecto.
    :type data_dir: str | None
    :param n_train: Cantidad de datos de entrenamiento
    :type n_train: int = 5000
    :param download_if_missing: Si True, descarga los datos si no existen.
    :type download_if_missing: bool
    :param verbose: Si True, muestra mensajes de progreso.
    :type verbose: bool
    :return: Tupla (X_train, Y_train)
    :rtype: Tuple[List[List[float]], List[int]]
    """
    if n_train < 1 or n_train > 10000:
        raise ValueError(f"n_train fuera de rango: {n_train}")

    if data_dir is None:
        data_dir = get_data_directory()

    if verbose:
        print("=" * 60)
        print("CARGANDO DATOS MNIST - ENTRENAMIENTO")
        print("=" * 60)

    # Descarga si es necesario
    if download_if_missing:
        descargar_mnist(data_dir, verbose=verbose)

    # Carga archivos
    train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")

    X_train = cargar_imagenes(train_images_path, verbose=verbose)
    Y_train = cargar_etiquetas(train_labels_path, verbose=verbose)

    X_subset = X_train[:n_train]
    Y_subset = Y_train[:n_train]

    if verbose:
        print(f"\n✓ Datos de entrenamiento seleccionados: {len(X_subset)} ejemplos")

    return X_subset, Y_subset


def load_mnist_test(
    data_dir: str | None = None, download_if_missing: bool = True, verbose: bool = True
) -> Tuple[List[List[float]], List[int]]:
    """
    Carga el conjunto de prueba de MNIST.

    :param data_dir: Directorio de datos. Si es None, usa Data/ por defecto.
    :type data_dir: str | None
    :param download_if_missing: Si True, descarga los datos si no existen.
    :type download_if_missing: bool
    :param verbose: Si True, muestra mensajes de progreso.
    :type verbose: bool
    :return: Tupla (X_test, Y_test)
    :rtype: Tuple[List[List[float]], List[int]]
    """
    if data_dir is None:
        data_dir = get_data_directory()

    if verbose:
        print("=" * 60)
        print("CARGANDO DATOS MNIST - PRUEBA")
        print("=" * 60)

    # Descarga si es necesario
    if download_if_missing:
        descargar_mnist(data_dir, verbose=verbose)

    # Carga archivos
    test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

    X_test = cargar_imagenes(test_images_path, verbose=verbose)
    Y_test = cargar_etiquetas(test_labels_path, verbose=verbose)

    if verbose:
        print(f"\n✓ Datos de prueba cargados: {len(X_test)} ejemplos")

    return X_test, Y_test


def load_mnist_complete(
    data_dir: str | None = None, download_if_missing: bool = True, verbose: bool = True
) -> Tuple[List[List[float]], List[int], List[List[float]], List[int]]:
    """
    Carga el dataset MNIST completo (entrenamiento y prueba).

    :param data_dir: Directorio de datos. Si es None, usa Data/ por defecto.
    :type data_dir: str | None
    :param download_if_missing: Si True, descarga los datos si no existen.
    :type download_if_missing: bool
    :param verbose: Si True, muestra mensajes de progreso.
    :type verbose: bool
    :return: Tupla (X_train, Y_train, X_test, Y_test)
    :rtype: Tuple[List[List[float]], List[int], List[List[float]], List[int]]
    """
    if data_dir is None:
        data_dir = get_data_directory()

    if verbose:
        print("=" * 60)
        print("CARGANDO DATASET MNIST COMPLETO")
        print("=" * 60)

    # Descarga si es necesario
    if download_if_missing:
        descargar_mnist(data_dir, verbose=verbose)
        if verbose:
            print()

    # Carga entrenamiento
    X_train, Y_train = load_mnist_train(
        data_dir, download_if_missing=False, verbose=verbose
    )

    if verbose:
        print()

    # Carga prueba
    X_test, Y_test = load_mnist_test(
        data_dir, download_if_missing=False, verbose=verbose
    )

    if verbose:
        print("\n" + "=" * 60)
        print(
            f"TOTAL: {len(X_train)} entrenamiento + {len(X_test)} prueba = {len(X_train) + len(X_test)} ejemplos"
        )
        print("=" * 60)

    return X_train, Y_train, X_test, Y_test


# ================
# EJEMPLO DE USO
# =================


def main():
    """
    Ejemplo de uso del cargador de MNIST.
    """
    print("\n" + "=" * 70)
    print("EJEMPLO DE USO: MNIST LOADER")
    print("=" * 70)

    # Ejemplo 1: Cargar solo entrenamiento
    print("\n--- Ejemplo 1: Cargar conjunto de entrenamiento ---")
    X_train, Y_train = load_mnist_train(verbose=True)
    print(f"\nPrimeras 5 etiquetas de entrenamiento: {Y_train[:5]}")
    print(f"Dimensiones de primera imagen: {len(X_train[0])} píxeles")

    # Ejemplo 2: Cargar solo prueba
    print("\n" + "-" * 70)
    print("\n--- Ejemplo 2: Cargar conjunto de prueba ---")
    X_test, Y_test = load_mnist_test(verbose=True)
    print(f"\nPrimeras 5 etiquetas de prueba: {Y_test[:5]}")

    # Ejemplo 3: Cargar todo
    print("\n" + "-" * 70)
    print("\n--- Ejemplo 3: Cargar dataset completo ---")
    X_tr, Y_tr, X_te, Y_te = load_mnist_complete(verbose=True)

    # Verificar consistencia
    print("\n" + "=" * 70)
    print("VERIFICACIÓN DE CONSISTENCIA")
    print("=" * 70)
    print(f"✓ X_train tiene {len(X_tr)} imágenes")
    print(f"✓ Y_train tiene {len(Y_tr)} etiquetas")
    print(f"✓ X_test tiene {len(X_te)} imágenes")
    print(f"✓ Y_test tiene {len(Y_te)} etiquetas")
    print(f"✓ Cada imagen tiene {len(X_tr[0])} píxeles (28x28)")

    # Distribución de clases
    print("\n" + "=" * 70)
    print("DISTRIBUCIÓN DE CLASES EN ENTRENAMIENTO")
    print("=" * 70)
    class_counts = {i: 0 for i in range(10)}
    for label in Y_tr:
        class_counts[label] += 1
    for digit, count in class_counts.items():
        bar = "█" * (count // 200)
        print(f"Dígito {digit}: {count:5d} ejemplos {bar}")


if __name__ == "__main__":
    main()
