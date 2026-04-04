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
    """
    Crea transformaciones de imagen para entrenamiento en ImageNet.

    Retorna composición de transforms para datos de entrenamiento:
    RandomResizedCrop, RandomHorizontalFlip, conversión a tensor float32,
    y normalización de ImageNet.

    :param image_size: Tamaño de crop cuadrado en píxeles (default: 224).
    :type image_size: int

    :returns: Pipeline Compose de Torchvision con transforms de entrenamiento.
    :rtype: T.Compose
    """
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
    """
    Crea transformaciones de imagen para validación/inferencia en ImageNet.

    Retorna composición de transforms para datos de validación:
    Resize a 256, CenterCrop al tamaño destino, conversión a tensor float32,
    y normalización de ImageNet.

    :param image_size: Tamaño de crop cuadrado en píxeles (default: 224).
    :type image_size: int

    :returns: Pipeline Compose de Torchvision con transforms de validación.
    :rtype: T.Compose
    """
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
        """
        Carga perezosamente y divide el dataset de entrenamiento desde HuggingFace Hub.

        Carga dataset con streaming=True (sin caché local), aplica división por Worker
        (cada Worker obtiene muestras contiguas), y buffer de shuffle opcional para
        aleatorización dentro de la división.

        División de Workers:
          Con num_workers=4, worker_rank=2:
            - Cada worker obtiene 1/4 del dataset completo, sin solapamiento
            - Muestras asignadas por algoritmo de sharding contiguo de HF
            - Buffers de shuffle independientes por worker (si shuffle_buffer > 0)

        :returns: Iterador de dataset streaming, dividido y aleatorizado
        :rtype: datasets.IterableDataset

        :raises EnvironmentError: Si dataset requiere autenticación y falta el token
        :raises ConnectionError: Si no es posible conectar con HuggingFace Hub
        """
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
        """
        Convierte datos de imagen crudos a PIL Image en formato RGB.

        Maneja múltiples formatos de entrada:
          - bytes: JPEG/PNG comprimido, descomprimido via PIL.open(BytesIO)
          - PIL.Image: pasado directamente, modo convertido a RGB si es necesario
          - Otros: retorna None (muestra omitida)

        Las fallas de descompresión retornan None (muestra omitida, sin excepción).

        :param raw: Datos de imagen crudos (bytes, PIL.Image, u otro)
        :type raw: Union[bytes, PIL.Image.Image, Any]

        :returns: PIL Image en modo RGB, o None si la conversión falló
        :rtype: Optional[PIL.Image.Image]

        :raises None: Errores de descompresión capturados silenciosamente, retorna None
        """
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
        """
        Generador infinito de batches de entrenamiento con reinicio automático.

        Implementa iteración infinita: abre dataset perezosamente, agrupa muestras,
        maneja excepciones con reconexiones automáticas, reinicia automáticamente
        cuando se agota. Actualiza estadísticas _batches y _samples.
        Omite muestras con errores de decodificación.

        Loop del generador:
          1. Abre dataset (perezoso, reutiliza si ya está abierto)
          2. Itera muestras: extrae imagen/label, transforma, acumula
          3. Cuando buffer alcanza batch_size: yielda batch, actualiza counters
          4. Al agotarse dataset: limpia _dataset, reinicia (iteración infinita)
          5. En excepción: espera 5s, limpia _dataset, reinten(resiliencia de red)

        :returns: Generador con stream infinito de batches (imágenes, labels)
        :rtype: Generator[Tuple[np.ndarray, np.ndarray], None, None]

        :raises RuntimeError: Nunca elevada por el generador mismo (excepciones capturadas y reconectadas)
        :raises Exception: Propaga si _open_dataset() falla después de reconexiones
        """
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
        """
        Inicializa generador infinito de batches para entrenamiento.

        Crea nuevo generador _generate() y retorna self para protocolo de iteración.
        Permite uso en bucles for: for X, Y in stream: ...

        :returns: Self como iterador
        :rtype: Iterator[Tuple[np.ndarray, np.ndarray]]

        :raises None
        """
        self._gen = self._generate()
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtiene siguiente batch (imágenes, labels) del stream de entrenamiento infinito.

        Inicializa perezosamente _generate() en primera llamada. Retorna tupla (X, Y).
        Stream reinicia automáticamente cuando dataset se agota (iteración infinita).

        :returns: Tupla batch (imágenes: np.ndarray[batch_size, 3, image_size, image_size], \
                  labels: np.ndarray[batch_size])
        :rtype: Tuple[np.ndarray, np.ndarray]

        :raises StopIteration: Nunca elevada (generador infinito)
        :raises Exception: Si error de red al obtener de HuggingFace o decodificación JPEG falla
        """
        if not hasattr(self, "_gen") or self._gen is None:
            self._gen = self._generate()
        return next(self._gen)

    @property
    def stats(self) -> dict:
        """
        Obtiene estadísticas de streaming (batches y muestras procesadas).

        Retorna contadores rastreando total de batches producidos e imágenes procesadas.
        Úsen para monitorear progreso de entrenamiento y detectar reinicios.

        :returns: Diccionario con claves 'batches' (int), 'samples' (int)
        :rtype: dict

        :raises None
        """
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
        """
        Inicia el hilo de prefetch en background.

        Inicializa el iterador del origen, limpia flag de parada, e inicia hilo daemon
        que continuamente llena la cola desde el stream de datos del origen.

        :returns: None
        :rtype: None

        :raises RuntimeError: Si creación de hilo falla o cola está corrupta
        """
        iter(self._source)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._fill,
            daemon=True,
            name=f"prefetch-W{self._source.worker_rank}",
        )
        self._thread.start()

    def stop(self) -> None:
        """
        Detiene el hilo de prefetch en background y desvácía la cola.

        Establece flag de parada, vacía batches restantes en cola, y espera
        terminación del hilo background (máx timeout 5 segundos).

        :returns: None
        :rtype: None

        :raises ThreadError: Si hilo se termina forzosamente con excepción
        """
        self._stop.set()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        if self._thread:
            self._thread.join(timeout=5)

    def _fill(self) -> None:
        """
        Hilo worker que continuamente llena la cola en background.

        Itera batches del origen, coloca cada uno en la cola acotada.
        Respeta flag de parada e reinten cuando queue.Full. Captura excepciones y
        notifica consumidor via sentinela None en cola.

        :returns: None
        :rtype: None

        :raises Exception: Capturada y almacenada en self._error, None encolado al consumidor
        """
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
        """
        Retorna self como iterador.

        Habilita uso en bucles for: for batch in prefetch_buffer: ...

        :returns: Self (protocolo iterador)
        :rtype: PrefetchBuffer

        :raises None
        """
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtiene siguiente batch prefetcheado (imágenes, labels) de la cola.

        Bloquea con timeout de 30 segundos esperando siguiente batch del hilo background.
        Retorna tupla (X, Y) si está disponible. Eleva StopIteration si flag de parada establecido.
        Propaga excepciones del hilo via RuntimeError.

        :returns: Tupla batch (imágenes: np.ndarray, labels: np.ndarray)
        :rtype: Tuple[np.ndarray, np.ndarray]

        :raises StopIteration: Cuando flag de parada establecido y cola vaciada
        :raises RuntimeError: Si hilo background encontró excepción
        :raises queue.Empty: Si timeout de 30 segundos excedido (condición timeout)
        """
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
        """
        Obtiene número actual de batches esperando en cola prefetch.

        Consulta no-bloqueante de profundidad interna de la cola. Úsin para monitorear
        rendimiento de prefetch y detectar atascos.

        :returns: Número de batches actualmente en la cola
        :rtype: int

        :raises None
        """
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
        """
        Itera a través de split de validación (de paso único, sin sharding).

        Retorna generador produciendo batches de validación. Procesa split de validación
        completo una sola vez por llamada (sin reinicios infinitos como en entrenamiento).
        Se detiene en max_batches si está establecido. Almacena en búfer batches parciales,
        produce chunks de batch_size completos, produce último batch parcial si existe remainder.

        :returns: Generador produciendo batches (imágenes, labels) de validación
        :rtype: Generator[Tuple[np.ndarray, np.ndarray], None, None]

        :raises ConnectionError: Si no es posible conectar con HuggingFace Hub
        :raises EnvironmentError: Si split de validación requiere autenticación y falta token
        :raises IOError: Si descompresión JPEG falla para imagen de muestra
        """
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
    Construye pipeline completo de prefetch para un Worker distribuido.

    Crea ImageNetStream con sharding del worker y lo envuelve en PrefetchBuffer
    para prefetch en background. Retorna iterador listo para usar.

    :param worker_rank: Índice de este worker (0-based, para sharding)
    :type worker_rank: int
    :param num_workers: Número total de workers
    :type num_workers: int
    :param batch_size: Imágenes por batch (default: 64)
    :type batch_size: int
    :param dataset_name: Nombre de dataset en HuggingFace (default: ILSVRC/imagenet-1k)
    :type dataset_name: str
    :param image_size: Tamaño de crop en píxeles (default: 224)
    :type image_size: int
    :param shuffle_buffer: Imágenes para buffer de shuffle (default: 1000)
    :type shuffle_buffer: int
    :param prefetch_batches: Batches a prefetch en background (default: 4)
    :type prefetch_batches: int
    :param seed: Semilla RNG para reproducibilidad (default: None)
    :type seed: Optional[int]
    :param hf_token: Token HuggingFace para autenticación (default: None)
    :type hf_token: Optional[str]

    :returns: PrefetchBuffer listo para llamar .start() e iterar
    :rtype: PrefetchBuffer

    :raises EnvironmentError: Si dataset requiere autenticación y falta token
    :raises ConnectionError: Si no es posible conectar con HuggingFace Hub
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
