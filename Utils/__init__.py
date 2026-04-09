"""
Utils — Utilidades para streaming de datos, logging y exportación de resultados.

Proporciona funcionalidades esenciales para el pipeline de entrenamiento distribuido
asíncrono sobre ImageNet-1k que soporta DOS MODOS:
  1. SimpleCNN E2E: CNN + MLP entrenables (E2E backprop)
  2. ResNet-18 MLP-only: CNN congelada, solo MLP entrenable

Streaming y logging funcionan idénticamente en ambos modos:

1. **Streaming de ImageNet-1k**: Descarga bajo demanda desde HuggingFace con:

   - Streaming puro: Nunca descarga el dataset completo (~1.2M imágenes).
   - Sharding automático: Cada Worker consume su porción sin solapamiento.
   - Prefetching asíncrono: Hilo background llena cola mientras entrenamiento consume.
   - Normalización: ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   - Transforms: RandomResizedCrop + HorizontalFlip para train; CenterCrop para val.

   Datasets soportados:
   - ILSVRC/imagenet-1k (requiere token HF + licencia aceptada)
   - timm/imagenet-1k-wds (WebDataset público, alternativa)

2. **Logging centralizado**: FormattedLogger con:

   - Colorización ANSI (UNIX/Windows 10+)
   - Timestamps para cada mensaje
   - Tags etiquetados: [INFO], [WORKER], [PS], [TRAIN], [EVAL], etc.
   - Soporte para modo silencioso

3. **Exportación de resultados**: Función export_results() que serializa
   históricos de training a JSON con metadatos.

MÓDULOS
=======

imagenet_streaming : module
    Clases y funciones para streaming de ImageNet-1k:

    - ImageNetStream: Iterador que descarga batches bajo demanda
    - PrefetchBuffer: Buffer asíncrono con pre-fetching en hilo background
    - ValidationStream: Iterador de validación (un solo pase completo)
    - build_worker_stream(): Factory que retorna PrefetchBuffer configurado
    - get_train_transform(): Transforms de data augmentation para entrenamiento
    - get_val_transform(): Transforms neutros para validación

    Parámetros principales:
    - worker_rank, num_workers: Sharding del dataset por Worker
    - batch_size: Imágenes por batch
    - prefetch_batches: Tamaño de cola de prefetch (default=4, ~385MB RAM)
    - shuffle_buffer: Buffer de shuffle interno (default=1000)
    - image_size: Tamaño final de crop (default=224)

logging_util : module
    Módulo de logging estructurado:

    - FormattedLogger: Clase principal con etiquetas por fase
    - get_logger(use_colors=False): Retorna logger global singleton

    Métodos de FormattedLogger:
    - info(msg), warning(msg), error(msg), etc.
    - Cada método añade timestamp y colorización automática

EXPORTACIONES PRINCIPALES
==========================

ImageNetStream : class
    from Utils.imagenet_streaming import ImageNetStream

    Iterador de batches bajo demanda desde HF.
    Parámetros: dataset_name, worker_rank, num_workers, batch_size, etc.

PrefetchBuffer : class
    from Utils.imagenet_streaming import PrefetchBuffer

    Buffer asíncrono con pre-fetching en background.
    Métodos: start(), stop(), __iter__(), __next__()
    Propiedad: queue_size

ValidationStream : class
    from Utils.imagenet_streaming import ValidationStream

    Iterador de validación (un solo pase, no infinito).
    Método: iterate() → Generator de (X_batch, Y_batch)

build_worker_stream : function
    from Utils.imagenet_streaming import build_worker_stream

    Factory que construye y retorna PrefetchBuffer configurado.
    Parámetros: worker_rank, num_workers, batch_size, dataset_name, etc.

get_train_transform, get_val_transform : functions
    from Utils.imagenet_streaming import get_train_transform, get_val_transform

    Retornan pipelines de transforms (augmentation para train, neutros para val).

FormattedLogger : class
    from Utils.logging_util import FormattedLogger

    Logger estructurado con colorización y timestamps.

get_logger : function
    from Utils.logging_util import get_logger

    Retorna logger global singleton.
    Parámetro: use_colors (default False para GUI)

FLUJO TÍPICO
============

    # Worker: Crear stream con sharding
    from Utils.imagenet_streaming import build_worker_stream

    stream = build_worker_stream(
        worker_rank=0, num_workers=2,
        batch_size=64, dataset_name='ILSVRC/imagenet-1k',
        hf_token=token, prefetch_batches=4
    )
    stream.start()

    for X_batch, Y_batch in stream:  # Yield (64, 3, 224, 224), (64,)
        # Entrenar con batch
        loss, acc = train_step(X_batch, Y_batch)

    stream.stop()

    # PS: Validación periódica
    from Utils.imagenet_streaming import ValidationStream

    val_stream = ValidationStream(
        dataset_name='ILSVRC/imagenet-1k',
        batch_size=256, max_batches=50, hf_token=token
    )
    for X_val, Y_val in val_stream.iterate():
        # Evaluar modelo global
        acc, loss = evaluate(X_val, Y_val)

    # Logging
    from Utils.logging_util import get_logger

    logger = get_logger(use_colors=True)
    logger.info("Iniciando entrenamiento")
    logger.error("Error de conexión")
"""

from Utils.imagenet_streaming import (
    ImageNetStream,
    PrefetchBuffer,
    ValidationStream,
    build_worker_stream,
    get_train_transform,
    get_val_transform,
)
from Utils.logging_util import FormattedLogger, get_logger

__all__ = [
    "ImageNetStream",
    "PrefetchBuffer",
    "ValidationStream",
    "build_worker_stream",
    "get_train_transform",
    "get_val_transform",
    "FormattedLogger",
    "get_logger",
]
