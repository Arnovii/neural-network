"""
Utils — Utilidades para streaming de datos, logging y exportación de resultados.

Proporciona funcionalidades esenciales para el pipeline de entrenamiento distribuido
asíncrono sobre ImageNet-1k:

1. **Streaming de ImageNet-1k**: Descarga bajo demanda desde HuggingFace con:

   - Streaming puro: Nunca descarga el dataset completo (~1.2M imágenes).
   - Sharding automático: Cada Worker consume su porción de datos sin solapamiento.
   - Prefetching asíncrono: Hilo background llena cola mientras entrenamiento consume.
   - Reconexión automática: Reintentos ante errores de red.
   - Transforms: RandomResizedCrop + HorizontalFlip para train; CenterCrop para val.

   Datasets soportados:
   - ILSVRC/imagenet-1k (requiere token HF + licencia aceptada)
   - timm/imagenet-1k-wds (formato WebDataset, acceso público)

2. **Logging centralizado**: Clase FormattedLogger con:

   - Colorización ANSI independiente de SO
   - Timestamps para cada mensaje
   - Fases etiquetadas: [INFO], [TRAIN], [EVAL], [ERROR], etc.
   - Soporte para modo quiet

3. **Exportación de resultados**: JSON con historiales de training incluyendo
   metadatos (arquitectura, épocas, workers activos, etc.).

MÓDULOS
=======

imagenet_streaming : module
    ImageNetStream: Iterador que descarga batches desde HF bajo demanda.
    PrefetchBuffer: Buffer asíncrono con pre-fetching en hilo background.
    ValidationStream: Iterador de validación (un solo pase, no infinito).
    build_worker_stream(): Factory que retorna PrefetchBuffer listo para usar.

    Parámetros clave:
    - worker_rank, num_workers: Sharding del dataset
    - prefetch_batches: Tamaño de cola (default=4, ~385MB por batch)
    - shuffle_buffer: Buffer de shuffle interno (default=1000 imágenes)

logging_util : module
    Clase FormattedLogger con métodos para cada fase.
    get_logger(): Retorna logger global con colorización.

results_exporter : module
    save_results(): Serializa histórico a JSON con metadatos.
    load_results(): Recarga histórico desde JSON.

EXPORTACIONES PRINCIPALES
==========================

ImageNetStream : class
    from Utils.imagenet_streaming import ImageNetStream

PrefetchBuffer : class
    from Utils.imagenet_streaming import PrefetchBuffer

ValidationStream : class
    from Utils.imagenet_streaming import ValidationStream

build_worker_stream : function
    from Utils.imagenet_streaming import build_worker_stream

get_logger : function
    from Utils.logging_util import get_logger

save_results, load_results : functions
    from Utils.results_exporter import save_results, load_results

FLUJO TÍPICO
============

    # Worker: Crear stream con sharding
    stream = build_worker_stream(
        worker_rank=0, num_workers=2, batch_size=64,
        dataset_name='ILSVRC/imagenet-1k', hf_token=token
    )
    stream.start()

    for X_batch, Y_batch in stream:  # (64, 3, 224, 224), (64,)
        # entrenar con batch
        pass

    stream.stop()

    # PS: Validación periódica
    val_stream = ValidationStream(
        dataset_name='ILSVRC/imagenet-1k',
        batch_size=256, max_batches=50, hf_token=token
    )
    for X_val, Y_val in val_stream.iterate():
        # evaluar
        pass
"""

export_results : function
    Exporta historial de entrenamiento a JSON.

    :param history: Dict con "accuracies", "losses", etc.
    :type history: Dict[str, List[float]]

    :param filepath: Ruta donde guardar JSON.
    :type filepath: str

get_logger : function
    Retorna instancia global de FormattedLogger configurada.

    :param use_colors: Usa colores ANSI (default False para GUI).
    :type use_colors: bool

    :return: Instancia logger global.
    :rtype: FormattedLogger

Uso rápido
----------
    from Utils import load_cifar10_train, load_cifar10_test, export_results, get_logger

    # Cargar datos
    X_train, Y_train = load_cifar10_train()
    X_test, Y_test = load_cifar10_test()

    # Logging
    logger = get_logger(use_colors=True)
    logger.train("Epoch 1: loss=0.45, acc=0.82")

    # Exportar resultados
    history = {"accuracies": [0.5, 0.7, 0.82], "losses": [1.2, 0.8, 0.45]}
    export_results(history, "results/training_output.json")
"""

# CIFAR-10 loaders
from Utils.cifar_loader import (
    load_cifar10_train,
    load_cifar10_test,
    NUM_CLASSES,
)

# Results export
from Utils.results_exporter import export_results

__all__ = [
    # CIFAR-10
    "load_cifar10_train",
    "load_cifar10_test",
    "NUM_CLASSES",
    # Results export
    "export_results",
]
