"""
Utils/imagenet_loader.py

Carga lazy de ImageNet para la pipeline CNN + MLP distribuida.

──────────────────────────────────────────────────────────────────
DIFERENCIAS CLAVE VS CIFAR-10
──────────────────────────────────────────────────────────────────
CIFAR-10:  170 MB  · 50k train · 10k test  · carga en RAM completa
ImageNet: ~150 GB  · 1.28M train · 50k test · carga LAZY por lotes

ImageNet NO cabe en RAM. Este módulo expone dos interfaces:

  1. get_imagenet_dataloader() → DataLoader de PyTorch
     Carga imágenes directamente del disco bajo demanda.
     Ideal para extracción de features por shards.

  2. load_imagenet_labels()    → np.ndarray de etiquetas
     Carga SOLO las etiquetas (sin imágenes) en RAM.
     Permite reconstruir el round-robin sin las imágenes.

──────────────────────────────────────────────────────────────────
ESTRUCTURA ESPERADA DEL DATASET
──────────────────────────────────────────────────────────────────
ImageNet sigue el formato torchvision estándar:

    Data/ImageNet/
        train/
            n01440764/   ← synset de cada clase
                img1.JPEG
                img2.JPEG
                ...
            n01443537/
            ...
        val/
            n01440764/
            ...

Si usas el script oficial de preparación de ILSVRC:
    python valprep.sh   (mueve val a subdirectorios por clase)

──────────────────────────────────────────────────────────────────
NORMALIZACIÓN IMAGENET ESTÁNDAR
──────────────────────────────────────────────────────────────────
μ = [0.485, 0.456, 0.406]   (R, G, B)
σ = [0.229, 0.224, 0.225]

Estos valores se calcularon sobre el training set completo
de ILSVRC-2012 y son los valores estándar en la literatura.
"""

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets as tvd, transforms as T

# ── Constantes exportadas ─────────────────────────────────────────
NUM_CLASSES = 1000
IMAGE_SIZE = 224
SHARD_SIZE = 50_000  # imágenes por shard de features

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _default_data_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "Data", "ImageNet")


def _train_transform() -> T.Compose:
    """
    Transformación estándar para train: resize + center crop.

    Nota: NO aplicamos RandomHorizontalFlip ni RandomCrop porque
    los features se extraen UNA sola vez y se cachean. Con
    aumentación aleatoria los features serían distintos cada vez
    que se reconstruye la caché, rompiendo la reproducibilidad.
    Para datos consistentes usamos la misma transformación que val.
    """
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=_MEAN, std=_STD),
        ]
    )


def _val_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=_MEAN, std=_STD),
        ]
    )


def get_imagenet_dataloader(
    split: str = "train",
    data_dir: Optional[str] = None,
    batch_size: int = 256,
    num_workers: int = 4,
    indices: Optional[np.ndarray] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Devuelve un DataLoader de ImageNet para extracción lazy de features.

    No carga las imágenes en RAM — las lee del disco bajo demanda.
    Usar num_workers > 0 paralela la E/S y la decodificación JPEG,
    reduciendo significativamente el tiempo de extracción.

    :param split: "train" o "val".
    :param data_dir: Ruta a Data/ImageNet/. None = Data/ImageNet/.
    :param batch_size: Imágenes por batch de extracción.
    :param num_workers: Procesos paralelos de carga. 4 es un buen
                        default para disco SSD. Con HDD usar 2.
    :param indices: Subconjunto de índices a cargar. None = todos.
    :param pin_memory: True si hay GPU disponible (acelera transferencia).
    :return: DataLoader configurado.
    """
    if data_dir is None:
        data_dir = _default_data_dir()

    split_dir = os.path.join(data_dir, "train" if split == "train" else "val")
    transform = _train_transform() if split == "train" else _val_transform()

    dataset = tvd.ImageFolder(root=split_dir, transform=transform)

    if indices is not None:
        dataset = Subset(dataset, indices)  # type: ignore[assignment]

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # NO barajar — mantenemos orden para shards
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def load_imagenet_labels(
    split: str = "train",
    data_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Carga SOLO las etiquetas de ImageNet sin leer las imágenes.

    Construye el dataset de torchvision (que indexa el disco pero no
    carga imágenes) y extrae targets como np.ndarray int32.
    Útil para reconstruir el round-robin sin abrir ningún JPEG.

    :param split: "train" o "val".
    :param data_dir: Ruta a Data/ImageNet/.
    :return: (N,) int32 con etiquetas en [0, 999].
    """
    if data_dir is None:
        data_dir = _default_data_dir()

    split_dir = os.path.join(data_dir, "train" if split == "train" else "val")
    # transform=None: solo necesitamos los targets, nunca abrimos JPEGs
    dataset = tvd.ImageFolder(root=split_dir, transform=None)
    return np.array(dataset.targets, dtype=np.int32)


def get_dataset_size(
    split: str = "train",
    data_dir: Optional[str] = None,
) -> int:
    """
    Devuelve el número de imágenes del split sin cargarlas.

    :param split: "train" o "val".
    :return: Número de imágenes.
    """
    if data_dir is None:
        data_dir = _default_data_dir()
    split_dir = os.path.join(data_dir, "train" if split == "train" else "val")
    dataset = tvd.ImageFolder(root=split_dir, transform=None)
    return len(dataset)
