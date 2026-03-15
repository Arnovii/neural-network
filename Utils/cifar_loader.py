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

import numpy as np

# ── Constantes exportadas ─────────────────────────────────────────
NUM_CLASSES = 10
INPUT_SHAPE = (3, 32, 32)  # NCHW sin dimensión de batch

_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

CIFAR10_CLASSES = [
    "avión",
    "automóvil",
    "pájaro",
    "gato",
    "ciervo",
    "perro",
    "rana",
    "caballo",
    "barco",
    "camión",
]


# ================================================================
# HELPERS INTERNOS
# ================================================================


def _default_data_dir() -> str:
    """
    Devuelve la ruta absoluta al directorio Data/ del proyecto.
    Si el directorio no existe, se crea automáticamente.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "Data")
    os.makedirs(path, exist_ok=True)
    return path


def _to_nchw_normalized(dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convierte un torchvision CIFAR10 dataset a NumPy NCHW normalizado.

    1. dataset.data es uint8 (N, 32, 32, 3) — NHWC.
    2. float32 / 255 → [0, 1].
    3. (x − μ) / σ  por canal (broadcast sobre axis=3).
    4. Transpone NHWC → NCHW y hace copia C-contigua para torch.

    Todo vectorizado; sin bucles por imagen.
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

    :param data_dir: Raíz de datos. None → Data/ del proyecto.
    :param download_if_missing: Descarga si no existe localmente.
    :param verbose: Imprime progreso.
    :return: (X_train, Y_train)
    """
    from torchvision import datasets as tvd

    if data_dir is None:
        data_dir = _default_data_dir()

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

    if verbose:
        print(f"✓ {len(X)} imágenes  |  shape {X.shape}  |  dtype {X.dtype}")

    return X, Y


def load_cifar10_test(
    data_dir: str | None = None,
    download_if_missing: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga el conjunto de prueba de CIFAR-10 (10 000 imágenes).

    Solo el PS lo carga, para evaluar el modelo global tras cada época.

    :param data_dir: Raíz de datos. None → Data/ del proyecto.
    :param download_if_missing: Descarga si no existe localmente.
    :param verbose: Imprime progreso.
    :return: (X_test, Y_test)  shapes (10000, 3, 32, 32) y (10000,)
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
