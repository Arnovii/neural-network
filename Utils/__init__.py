"""
Utils — Utilidades para entrenamiento distribuido ImageNet CNN+MLP.

──────────────────────────────────────────────────────────────────
Módulos
──────────────────────────────────────────────────────────────────
feature_scaler    FeatureScaler: normalización StandardScaler para features CNN.
                  Calcula media/std offline sobre features de train
                  y los aplica en train y test.

imagenet_loader   Cargadores lazy de ImageNet (~150 GB).
                  Funciones: get_imagenet_dataloader, load_imagenet_labels,
                            get_dataset_size.
                  Constantes: NUM_CLASSES, IMAGE_SIZE, SHARD_SIZE.

results_exporter  Exportación de resultados de entrenamiento a JSON con
                  historial, configuración y resumen de métricas.

──────────────────────────────────────────────────────────────────
Uso rápido
──────────────────────────────────────────────────────────────────
    from Utils import (
        # Feature normalization
        FeatureScaler,
        # ImageNet dataloaders
        get_imagenet_dataloader, load_imagenet_labels, get_dataset_size,
        NUM_CLASSES, IMAGE_SIZE, SHARD_SIZE,
        # Results export
        export_results
    )
"""

# Feature normalization
from Utils.feature_scaler import FeatureScaler

# ImageNet loaders
from Utils.imagenet_loader import (
    get_imagenet_dataloader,
    load_imagenet_labels,
    get_dataset_size,
    NUM_CLASSES,
    IMAGE_SIZE,
    SHARD_SIZE,
)

# Results export
from Utils.results_exporter import export_results

__all__ = [
    # Feature scaling
    "FeatureScaler",
    # ImageNet
    "get_imagenet_dataloader",
    "load_imagenet_labels",
    "get_dataset_size",
    "NUM_CLASSES",
    "IMAGE_SIZE",
    "SHARD_SIZE",
    # Results
    "export_results",
]
