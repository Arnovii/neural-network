"""
Utils/cifar_loader.py

Descarga y carga de CIFAR-10 para la pipeline CNN + MLP distribuida.

──────────────────────────────────────────────────────────────────
FORMATO DE SALIDA
──────────────────────────────────────────────────────────────────
Las imágenes se devuelven en formato NCHW:

    X: (N, 3, 32, 32)  float32  normalizado por canal
    Y: (N,)            int32    etiquetas en {0, …, 9}

El formato NCHW (batch, canales, alto, ancho) es el estándar de
PyTorch y lo que espera la CNN (Model/cnn_extractor.py).
Normalizar en el loader — y no en cada Worker — evita repetir el
cálculo en cada época.

──────────────────────────────────────────────────────────────────
NORMALIZACIÓN POR CANAL
──────────────────────────────────────────────────────────────────
Se usa la media y std calculadas sobre el conjunto de entrenamiento
completo de CIFAR-10 (valores estándar de la literatura):

    μ = [0.4914, 0.4822, 0.4465]   (R, G, B)
    σ = [0.2470, 0.2435, 0.2616]

──────────────────────────────────────────────────────────────────
CLASES
──────────────────────────────────────────────────────────────────
0: avión       1: automóvil   2: pájaro      3: gato
4: ciervo      5: perro       6: rana        7: caballo
8: barco       9: camión
"""

import os
from typing import Tuple

import warnings
import numpy as np

warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)

# ── Constantes exportadas ─────────────────────────────────────────
NUM_CLASSES = 10

_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


# ================================================================
# HELPERS INTERNOS
# ================================================================


def _default_data_dir() -> str:
    """
    Devuelve la ruta absoluta al directorio Data/ del proyecto.

    Si el directorio no existe, se crea automáticamente.

    :return: Ruta absoluta al directorio Data/.
    :rtype: str.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "Data")
    os.makedirs(path, exist_ok=True)
    return path


def _cache_path(data_dir: str, name: str) -> str:
    """
    Construye la ruta de archivo .npz para cachear un dataset.

    :param data_dir: Directorio raíz de datos.
    :type data_dir: str.
    :param name: Nombre del dataset (ej. "cifar10_train_nchw").
    :type name: str.
    :return: Ruta completa: data_dir/{name}.npz.
    :rtype: str.
    """
    return os.path.join(data_dir, f"{name}.npz")


def _to_nchw_normalized(dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convierte un torchvision CIFAR10 dataset a NumPy NCHW normalizado.

    Pasos:
        1. dataset.data es uint8 (N, 32, 32, 3) — NHWC.
        2. float32 / 255 → [0, 1].
        3. (x − μ) / σ  por canal (broadcast sobre axis=3).
        4. Transpone NHWC → NCHW y hace copia C-contigua para torch.

    Todo vectorizado sin bucles por imagen.

    :param dataset: Dataset CIFAR-10 de torchvision.
    :type dataset: torchvision.datasets.CIFAR10.
    :return: Tupla (X, Y) normalizada y transpuesta.
    :rtype: Tuple[np.ndarray, np.ndarray] donde X shape (N, 3, 32, 32) float32
            y Y shape (N,) int32.
    """
    X = dataset.data.astype(np.float32) / 255.0  # (N, 32, 32, 3)
    X = (X - _MEAN) / _STD  # (N, 32, 32, 3)
    X = X.transpose(0, 3, 1, 2).copy()  # (N, 3, 32, 32)
    Y = np.array(dataset.targets, dtype=np.int32)
    return X, Y


# ================================================================
# FUNCIONES PÚBLICAS
# ================================================================


def load_cifar10_train(
    data_dir: str | None = None,
    download_if_missing: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga el conjunto de entrenamiento completo de CIFAR-10 (50 000 imágenes).

    Devuelve X (50000, 3, 32, 32) float32 normalizado en NCHW e
    Y (50000,) int32. La subselección por época (n_train < 50 000)
    la realiza el Worker con _reconstruct_indices usando la semilla
    del PS — sin transmitir índices por red.

    Usa caché automático en formato .npz.

    :param data_dir: Raíz de datos. None → Data/ del proyecto.
    :type data_dir: str | None, default=None.
    :param download_if_missing: Descarga desde internet si no existe localmente.
    :type download_if_missing: bool, default=True.
    :param verbose: Imprime progreso de carga.
    :type verbose: bool, default=True.
    :return: Tupla (X_train, Y_train) con 50000 imágenes.
    :rtype: Tuple[np.ndarray, np.ndarray].
    """
    from torchvision import datasets as tvd

    if data_dir is None:
        data_dir = _default_data_dir()

    cache_file = _cache_path(data_dir, "cifar10_train_nchw")

    # ── Si existe cache, cargar directamente ──
    if os.path.exists(cache_file):
        if verbose:
            print("Cargando CIFAR-10 desde cache...")

        data = np.load(cache_file)
        return data["X"], data["Y"]

    # ── Carga normal (solo primera vez) ──
    if verbose:
        print("=" * 60)
        print("CARGANDO CIFAR-10 — ENTRENAMIENTO (50 000 imágenes)")
        print("=" * 60)

    dataset = tvd.CIFAR10(
        root=data_dir,
        train=True,
        download=download_if_missing,
        transform=None,
    )
    X, Y = _to_nchw_normalized(dataset)

    # ── Guarda cache ──
    np.savez_compressed(cache_file, X=X, Y=Y)

    if verbose:
        print("✓ Dataset cacheado para futuras ejecuciones")

    return X, Y


def load_cifar10_test(
    data_dir: str | None = None,
    download_if_missing: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga el conjunto de prueba de CIFAR-10 (10 000 imágenes).

    Solo el PS lo carga para evaluar el modelo global tras cada época.

    :param data_dir: Raíz de datos. None → Data/ del proyecto.
    :type data_dir: str | None, default=None.
    :param download_if_missing: Descarga desde internet si no existe localmente.
    :type download_if_missing: bool, default=True.
    :param verbose: Imprime progreso de carga.
    :type verbose: bool, default=True.
    :return: Tupla (X_test, Y_test) con 10000 imágenes y etiquetas.
    :rtype: Tuple[np.ndarray, np.ndarray] con shapes (10000, 3, 32, 32) float32
            y (10000,) int32.
    """
    from torchvision import datasets as tvd

    if data_dir is None:
        data_dir = _default_data_dir()

    if verbose:
        print("=" * 60)
        print("CARGANDO CIFAR-10 — PRUEBA (10 000 imágenes)")
        print("=" * 60)

    dataset = tvd.CIFAR10(
        root=data_dir,
        train=False,
        download=download_if_missing,
        transform=None,
    )
    X, Y = _to_nchw_normalized(dataset)

    if verbose:
        print(f"✓ {len(X)} imágenes  |  shape {X.shape}  |  dtype {X.dtype}")

    return X, Y
