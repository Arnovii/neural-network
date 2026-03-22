"""
Utils/imagenet_loader.py

Carga de ImageNet para la pipeline CNN + MLP distribuida.
Soporta dos modos transparentes al resto del código:

  MODO LOCAL (rápido, offline)
  ─────────────────────────────
  Requiere el dataset ya descargado en disco con estructura ImageFolder:
      Data/ImageNet/train/n01440764/img.JPEG
      Data/ImageNet/val/n01440764/img.JPEG
  Usa torchvision.DataLoader con lectura paralela del disco.

  MODO STREAM (sin descarga previa, requiere internet)
  ──────────────────────────────────────────────────────
  Lee desde HuggingFace Hub (ILSVRC/imagenet-1k) bajo demanda.
  Las imágenes nunca se guardan en disco — solo los features extraídos.
  Requiere token HuggingFace con acceso al dataset (licencia ILSVRC).

  Flujo stream + caché de features:
    1ª sesión: stream HF → CNN forward → guardar shards .npy (~2.6 GB)
    2ª sesión+: cargar shards .npy directamente, sin internet

  DETECCIÓN AUTOMÁTICA
  ──────────────────────
  detect_data_source(data_dir) → "local" | "stream"
  Si data_dir contiene train/ y val/ → local; si no → stream.
  Toda la lógica de selección queda aquí, no en worker_node.

NORMALIZACIÓN IMAGENET ESTÁNDAR
─────────────────────────────────
  μ = [0.485, 0.456, 0.406]   σ = [0.229, 0.224, 0.225]
  Calculados sobre ILSVRC-2012 train. Necesarios para que los pesos
  preentrenados de ResNet-18 produzcan features de calidad.
"""

from __future__ import annotations

import os
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, Subset
from torchvision import datasets as tvd, transforms as T

# ── Constantes exportadas ─────────────────────────────────────────
NUM_CLASSES = 1000
IMAGE_SIZE = 224
SHARD_SIZE = 50_000  # imágenes por shard de features (~100 MB c/u)

HF_DATASET = "ILSVRC/imagenet-1k"  # dataset oficial en HuggingFace

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


# ══════════════════════════════════════════════════════════════════
# UTILIDADES COMUNES
# ══════════════════════════════════════════════════════════════════


def _default_data_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "Data", "ImageNet")


def _imagenet_transform() -> T.Compose:
    """
    Transformación estándar ImageNet: resize → center crop → tensor → normalize.

    No usamos RandomHorizontalFlip porque los features se extraen UNA sola
    vez y se cachean. Con aumentación aleatoria los shards serían distintos
    cada vez que se reconstruyen, rompiendo la reproducibilidad.
    """
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=_MEAN, std=_STD),
        ]
    )


def detect_data_source(data_dir: Optional[str] = None) -> str:
    """
    Detecta si el dataset está disponible en disco local o hay que usar streaming.

    Regla: si data_dir contiene train/ y val/ con al menos un subdirectorio
    cada uno, se usa el modo local. En cualquier otro caso, streaming.

    :param data_dir: Ruta al directorio raíz de ImageNet. None = Data/ImageNet/.
    :return: "local" si el dataset está en disco, "stream" si no.
    """
    if data_dir is None:
        data_dir = _default_data_dir()

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    def _has_subdirs(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        return any(os.path.isdir(os.path.join(path, e)) for e in os.listdir(path))

    if _has_subdirs(train_dir) and _has_subdirs(val_dir):
        return "local"
    return "stream"


# ══════════════════════════════════════════════════════════════════
# MODO LOCAL — torchvision ImageFolder
# ══════════════════════════════════════════════════════════════════


def get_imagenet_dataloader(
    split: str = "train",
    data_dir: Optional[str] = None,
    batch_size: int = 256,
    num_workers: int = 4,
    indices: Optional[np.ndarray] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """
    DataLoader local de ImageNet (requiere dataset en disco).

    Lee imágenes del disco bajo demanda. num_workers > 0 paralela
    la E/S y la decodificación JPEG, reduciendo el tiempo de extracción.

    :param split: "train" o "val".
    :param data_dir: Ruta a Data/ImageNet/. None = Data/ImageNet/.
    :param batch_size: Imágenes por batch.
    :param num_workers: Procesos paralelos de carga. 4 para SSD, 2 para HDD.
    :param indices: Subconjunto de índices. None = todo el split.
    :param pin_memory: True si hay GPU disponible.
    :return: DataLoader configurado.
    """
    if data_dir is None:
        data_dir = _default_data_dir()

    split_dir = os.path.join(data_dir, "train" if split == "train" else "val")
    dataset = tvd.ImageFolder(root=split_dir, transform=_imagenet_transform())

    if indices is not None:
        dataset = Subset(dataset, indices)  # type: ignore[assignment]

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
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
    Carga SOLO las etiquetas de ImageNet local sin leer las imágenes.

    :param split: "train" o "val".
    :param data_dir: Ruta a Data/ImageNet/.
    :return: (N,) int32 con etiquetas en [0, 999].
    :raises FileNotFoundError: Si el directorio del split no existe.
    """
    if data_dir is None:
        data_dir = _default_data_dir()

    split_dir = os.path.join(data_dir, "train" if split == "train" else "val")
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"Directorio de ImageNet no encontrado: {split_dir}\n"
            f"Usa --data-dir para especificar la ruta correcta o\n"
            f"usa modo stream con --hf-token."
        )
    dataset = tvd.ImageFolder(root=split_dir, transform=None)
    return np.array(dataset.targets, dtype=np.int32)


def get_dataset_size(
    split: str = "train",
    data_dir: Optional[str] = None,
) -> int:
    """Número de imágenes del split sin cargarlas."""
    if data_dir is None:
        data_dir = _default_data_dir()
    split_dir = os.path.join(data_dir, "train" if split == "train" else "val")
    dataset = tvd.ImageFolder(root=split_dir, transform=None)
    return len(dataset)


# ══════════════════════════════════════════════════════════════════
# MODO STREAM — HuggingFace Hub
# ══════════════════════════════════════════════════════════════════

# Tamaños oficiales de ILSVRC-2012
_HF_SPLIT_SIZES = {
    "train": 1_281_167,
    "validation": 50_000,
}


def _hf_split_name(split: str) -> str:
    """HuggingFace usa 'validation', no 'val'."""
    return "validation" if split == "val" else split


class _HFStreamDataset(IterableDataset):
    """
    Dataset iterable que lee ImageNet desde HuggingFace en modo streaming.

    Cada elemento es un dict con "image" (PIL.Image) y "label" (int).
    Esta clase lo convierte a tensores (C, H, W) float32 normalizados
    con etiquetas int64, exactamente igual que ImageFolder.

    El sharding se aplica antes de crear el dataset:
    ds.shard(num_shards, index) garantiza que cada Worker
    procese una porción distinta sin solapamiento.

    :param hf_split: Split de HuggingFace ("train" o "validation").
    :param token: Token HuggingFace con acceso a ILSVRC/imagenet-1k.
    :param shard_index: Índice de este shard (0-based).
    :param num_shards: Total de shards.
    :param start_index: Primer elemento a procesar dentro del shard
                        (para reanudar si se interrumpió).
    """

    def __init__(
        self,
        hf_split: str,
        token: str,
        shard_index: int = 0,
        num_shards: int = 1,
        start_index: int = 0,
    ) -> None:
        super().__init__()
        self._hf_split = hf_split
        self._token = token
        self._shard_index = shard_index
        self._num_shards = num_shards
        self._start_index = start_index
        self._transform = _imagenet_transform()

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise ImportError(
                "El modo streaming requiere la librería 'datasets' de HuggingFace.\n"
                "Instala con: pip install datasets"
            ) from e

        ds = load_dataset(
            HF_DATASET,
            split=self._hf_split,
            token=self._token,
            streaming=True,
            trust_remote_code=True,
        )

        # Sharding: cada Worker toma su porción sin solapamiento
        if self._num_shards > 1:
            ds = ds.shard(
                num_shards=self._num_shards,
                index=self._shard_index,
            )

        # Saltar elementos ya procesados (reanudación)
        if self._start_index > 0:
            ds = ds.skip(self._start_index)

        for item in ds:
            img = item["image"]
            label = item["label"]

            # Asegurar que la imagen es RGB (algunas son escala de grises)
            if img.mode != "RGB":
                img = img.convert("RGB")

            tensor = self._transform(img)
            yield tensor, label


def get_imagenet_stream_dataloader(
    split: str = "train",
    token: str = "",
    batch_size: int = 64,
    shard_index: int = 0,
    num_shards: int = 1,
    start_index: int = 0,
) -> DataLoader:
    """
    DataLoader de ImageNet en modo streaming desde HuggingFace.

    No descarga el dataset completo — las imágenes llegan bajo demanda.
    Los features resultantes SÍ se cachean en disco (shards .npy),
    por lo que solo la primera sesión requiere internet.

    IMPORTANTE: num_workers=0 porque IterableDataset con HuggingFace
    no es compatible con multiprocessing de DataLoader. La paralelización
    la gestiona HuggingFace internamente.

    :param split: "train" o "val".
    :param token: Token HuggingFace. También acepta variable de entorno
                  HF_TOKEN si token="".
    :param batch_size: Imágenes por batch.
    :param shard_index: Índice de este Worker (0-based).
    :param num_shards: Total de Workers que procesan en paralelo.
    :param start_index: Elemento de inicio dentro del shard (para reanudación).
    :return: DataLoader iterable configurado.
    """
    # Leer token desde variable de entorno si no se pasó explícitamente
    resolved_token = token or os.environ.get("HF_TOKEN", "")
    if not resolved_token:
        raise ValueError(
            "Se requiere un token de HuggingFace para modo streaming.\n"
            "Opciones:\n"
            "  1. --hf-token <token> en la línea de comandos\n"
            "  2. Variable de entorno: export HF_TOKEN=<token>\n"
            "Obtén tu token en: https://huggingface.co/settings/tokens"
        )

    hf_split = _hf_split_name(split)
    dataset = _HFStreamDataset(
        hf_split=hf_split,
        token=resolved_token,
        shard_index=shard_index,
        num_shards=num_shards,
        start_index=start_index,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # IterableDataset + HF no admite multiprocessing
        pin_memory=torch.cuda.is_available(),
    )


def load_imagenet_labels_stream(
    split: str = "val",
    token: str = "",
) -> np.ndarray:
    """
    Carga etiquetas de ImageNet desde HuggingFace sin imágenes.

    Descarga solo los metadatos del split (mucho más pequeño que las imágenes).
    Para el split 'val' (50k ejemplos) esto es ~400 KB vs ~6 GB de imágenes.

    :param split: "train" o "val". En práctica solo se usa "val" para Y_test.
    :param token: Token HuggingFace.
    :return: (N,) int32 con etiquetas en [0, 999].
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise ImportError(
            "El modo streaming requiere 'datasets'. Instala con: pip install datasets"
        ) from e

    resolved_token = token or os.environ.get("HF_TOKEN", "")
    if not resolved_token:
        raise ValueError(
            "Se requiere HF_TOKEN para cargar etiquetas desde HuggingFace."
        )

    hf_split = _hf_split_name(split)
    print(f"[Loader] Descargando etiquetas de {hf_split} desde HuggingFace...")

    # Cargar solo la columna 'label' — evita descargar las imágenes
    ds = load_dataset(
        HF_DATASET,
        split=hf_split,
        token=resolved_token,
        streaming=False,  # descarga completa pero solo metadatos
        trust_remote_code=True,
    ).select_columns(["label"])

    labels = np.array(ds["label"], dtype=np.int32)
    print(f"[Loader] {len(labels):,} etiquetas cargadas.")
    return labels


def get_stream_shard_size(
    split: str = "train",
    num_shards: int = 1,
    shard_index: int = 0,
) -> int:
    """
    Estima el número de imágenes que corresponden a un shard.

    Usa los tamaños oficiales de ILSVRC-2012.
    El último shard puede tener ligeramente menos imágenes.

    :param split: "train" o "val".
    :param num_shards: Total de shards.
    :param shard_index: Índice del shard.
    :return: Número estimado de imágenes en el shard.
    """
    hf_split = _hf_split_name(split)
    total = _HF_SPLIT_SIZES.get(hf_split, 0)
    base = total // num_shards
    extra = 1 if shard_index < (total % num_shards) else 0
    return base + extra
