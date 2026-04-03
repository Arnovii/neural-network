"""
Utils/imagenet_streaming.py

Pipeline de datos ImageNet-1k con streaming desde HuggingFace.

PRINCIPIOS:
  - Streaming puro: nunca se descarga el dataset completo.
  - Sharding automático por Worker: cada proceso consume su porción
    sin solapamiento con otros Workers.
  - Prefetching con doble buffer: un hilo background llena una cola
    mientras el entrenamiento consume el batch anterior.
  - Reconexión automática ante errores de red.

DATASET:
  Primario  : 'ILSVRC/imagenet-1k'  (requiere token y licencia aceptada en HF)
  Alternativa: 'timm/imagenet-1k-wds' (formato WebDataset, acceso público)

TRANSFORMS:
  Train: RandomResizedCrop(224) + HorizontalFlip + Normalize(ImageNet stats)
  Val:   Resize(256) + CenterCrop(224) + Normalize(ImageNet stats)

LABEL EXTRACTION:
  Se usa is-None check (no or-chain) para evitar que label=0 (clase tench)
  sea tratado como falsy y sustituido por el campo alternativo.
"""

import io
import queue
import threading
import time
from typing import Generator, Iterator, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image


# ── Estadísticas estándar de ImageNet ────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 1000


# ================================================================
# TRANSFORMS
# ================================================================


def get_train_transform(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, antialias=True),
            T.RandomHorizontalFlip(),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )


def get_val_transform(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize(256, antialias=True),
            T.CenterCrop(image_size),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )


# ================================================================
# HELPERS
# ================================================================


def _extract_label(sample: dict) -> int:
    """
    Extrae el label de un sample de forma robusta.

    Usa is-None check en lugar de or-chain para evitar que label=0
    (clase 'tench', primera clase de ImageNet-1k) sea tratado como
    falsy y sustituido incorrectamente por el campo 'cls'.

    Ejemplos:
        label=0, cls=None → 0   ✓
        label=0, cls=5    → 0   ✓  (sin el fix: devolvería 5)
        label=None, cls=3 → 3   ✓
        label=None, cls=None → 0 ✓
    """
    lbl = sample.get("label")
    if lbl is not None:
        return int(lbl)
    cls = sample.get("cls")
    if cls is not None:
        return int(cls)
    return 0


# ================================================================
# STREAM ITERATOR (infinito para train)
# ================================================================


class ImageNetStream:
    """
    Iterador de batches ImageNet con streaming infinito desde HF.

    Produce (X, Y) indefinidamente. Cuando el split se agota,
    reinicia el stream automáticamente.

    :param dataset_name:   Nombre del dataset en HF Hub.
    :param worker_rank:    Índice de este Worker (para sharding).
    :param num_workers:    Total de Workers.
    :param batch_size:     Imágenes por batch.
    :param image_size:     Tamaño de crop final.
    :param shuffle_buffer: Imágenes en buffer de shuffle (0 = sin shuffle).
    :param seed:           Semilla del shuffle.
    :param hf_token:       Token HF.
    """

    def __init__(
        self,
        dataset_name: str = "ILSVRC/imagenet-1k",
        worker_rank: int = 0,
        num_workers: int = 1,
        batch_size: int = 64,
        image_size: int = 224,
        shuffle_buffer: int = 1000,
        seed: Optional[int] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.worker_rank = worker_rank
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.hf_token = hf_token
        self.transform = get_train_transform(image_size)
        self._dataset = None
        self._batches = 0
        self._samples = 0

    def _open_dataset(self):
        from datasets import load_dataset

        kw = {"streaming": True, "split": "train"}
        if self.hf_token:
            kw["token"] = self.hf_token
        ds = load_dataset(self.dataset_name, **kw)
        if self.num_workers > 1:
            ds = ds.shard(
                num_shards=self.num_workers,
                index=self.worker_rank,
                contiguous=True,
            )
        if self.shuffle_buffer > 0:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        return ds

    @staticmethod
    def _to_pil(raw) -> Optional[Image.Image]:
        """Convierte el valor imagen del sample a PIL RGB."""
        if isinstance(raw, bytes):
            try:
                img = Image.open(io.BytesIO(raw))
            except Exception:
                return None
        elif isinstance(raw, Image.Image):
            img = raw
        else:
            return None
        return img if img.mode == "RGB" else img.convert("RGB")

    def _generate(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        buf_X: list = []
        buf_Y: list = []
        while True:
            if self._dataset is None:
                self._dataset = self._open_dataset()
            try:
                for sample in self._dataset:
                    raw = sample.get("image") or sample.get("jpg") or sample.get("png")
                    label = _extract_label(sample)  # FIX: is-None check
                    img = self._to_pil(raw)
                    if img is None:
                        continue
                    try:
                        tensor = self.transform(img)
                    except Exception:
                        continue
                    buf_X.append(tensor.numpy())
                    buf_Y.append(label)
                    if len(buf_X) >= self.batch_size:
                        X = np.stack(buf_X[: self.batch_size])
                        Y = np.array(buf_Y[: self.batch_size], dtype=np.int64)
                        buf_X = buf_X[self.batch_size :]
                        buf_Y = buf_Y[self.batch_size :]
                        self._batches += 1
                        self._samples += self.batch_size
                        yield X, Y
                # Stream agotado → reiniciar
                self._dataset = None
            except Exception as e:
                print(
                    f"[Stream W{self.worker_rank}] Error: {e}. Reconectando en 5 s..."
                )
                time.sleep(5)
                self._dataset = None

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        self._gen = self._generate()
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "_gen") or self._gen is None:
            self._gen = self._generate()
        return next(self._gen)

    @property
    def stats(self) -> dict:
        return {"batches": self._batches, "samples": self._samples}


# ================================================================
# PREFETCH BUFFER
# ================================================================


class PrefetchBuffer:
    """
    Buffer asíncrono que pre-carga batches en un hilo background.

    Mientras el entrenamiento consume el batch N, el hilo background
    ya está preparando el batch N+1. Elimina el tiempo de espera de
    I/O y decodificación JPEG del loop de entrenamiento.

    Con buffer_size=4 y batch_size=64:
      RAM usada ≈ 4 × 64 × 3 × 224 × 224 × 4 bytes ≈ 385 MB

    :param source:      Iterador fuente (ImageNetStream).
    :param buffer_size: Número de batches a pre-cargar (cola máxima).
    """

    def __init__(self, source: ImageNetStream, buffer_size: int = 4) -> None:
        self._source = source
        self._q: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[Exception] = None

    def start(self) -> None:
        iter(self._source)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._fill,
            daemon=True,
            name=f"prefetch-W{self._source.worker_rank}",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        if self._thread:
            self._thread.join(timeout=5)

    def _fill(self) -> None:
        try:
            for batch in self._source:
                if self._stop.is_set():
                    break
                while not self._stop.is_set():
                    try:
                        self._q.put(batch, timeout=1.0)
                        break
                    except queue.Full:
                        continue
        except Exception as e:
            self._error = e
            try:
                self._q.put(None, timeout=2.0)
            except queue.Full:
                pass

    def __iter__(self) -> "PrefetchBuffer":
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._error:
            raise RuntimeError(f"Error en prefetch: {self._error}")
        while True:
            try:
                item = self._q.get(timeout=30.0)
            except queue.Empty:
                if self._stop.is_set():
                    raise StopIteration
                continue
            if item is None:
                raise RuntimeError(
                    str(self._error) if self._error else "Stream terminado"
                )
            return item

    @property
    def queue_size(self) -> int:
        return self._q.qsize()


# ================================================================
# VALIDACIÓN (un solo paso sobre el split completo)
# ================================================================


class ValidationStream:
    """
    Iterador de validación que recorre el split completo UNA vez.

    Usado por el PS para evaluación periódica del modelo global.
    No es infinito: StopIteration al agotar el split.

    :param dataset_name: Dataset HF.
    :param batch_size:   Imágenes por batch.
    :param image_size:   Tamaño de crop.
    :param max_batches:  Limitar a N batches (None = todos los 50,000 imgs).
    :param hf_token:     Token HF.
    """

    def __init__(
        self,
        dataset_name: str = "ILSVRC/imagenet-1k",
        batch_size: int = 256,
        image_size: int = 224,
        max_batches: Optional[int] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.hf_token = hf_token
        self.transform = get_val_transform(image_size)

    def iterate(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        from datasets import load_dataset

        kw = {"streaming": True, "split": "validation"}
        if self.hf_token:
            kw["token"] = self.hf_token
        ds = load_dataset(self.dataset_name, **kw)

        buf_X, buf_Y = [], []
        done = 0

        for sample in ds:
            if self.max_batches and done >= self.max_batches:
                break
            raw = sample.get("image") or sample.get("jpg")
            label = _extract_label(sample)  # FIX: is-None check
            img = ImageNetStream._to_pil(raw)
            if img is None:
                continue
            try:
                tensor = self.transform(img)
            except Exception:
                continue
            buf_X.append(tensor.numpy())
            buf_Y.append(label)
            if len(buf_X) >= self.batch_size:
                yield (
                    np.stack(buf_X[: self.batch_size]),
                    np.array(buf_Y[: self.batch_size], dtype=np.int64),
                )
                buf_X = buf_X[self.batch_size :]
                buf_Y = buf_Y[self.batch_size :]
                done += 1

        if buf_X:
            yield np.stack(buf_X), np.array(buf_Y, dtype=np.int64)


# ================================================================
# FACTORY
# ================================================================


def build_worker_stream(
    worker_rank: int,
    num_workers: int,
    batch_size: int = 64,
    dataset_name: str = "ILSVRC/imagenet-1k",
    image_size: int = 224,
    shuffle_buffer: int = 1000,
    prefetch_batches: int = 4,
    seed: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> PrefetchBuffer:
    """
    Construye el pipeline completo para un Worker.

    Devuelve un PrefetchBuffer listo para llamar .start() e iterar.
    """
    source = ImageNetStream(
        dataset_name=dataset_name,
        worker_rank=worker_rank,
        num_workers=num_workers,
        batch_size=batch_size,
        image_size=image_size,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        hf_token=hf_token,
    )
    return PrefetchBuffer(source, buffer_size=prefetch_batches)
