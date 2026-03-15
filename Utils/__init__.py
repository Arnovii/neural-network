"""
Utils — Utilidades para entrenamiento distribuido CIFAR-10 CNN+MLP.

Módulos
-------
cifar_loader      Cargadores de datos CIFAR-10 (train y test).
results_exporter  Exportación de resultados de entrenamiento a JSON.

Uso rápido
----------
    from Utils import (
        load_cifar10_train, load_cifar10_test, NUM_CLASSES,
        export_results
    )
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
