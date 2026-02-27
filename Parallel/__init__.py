"""
Parallel — Soporte de multiprocessing para el Algoritmo de Diego.

Permite ejecutar el entrenamiento de cada partición en un proceso
del sistema operativo independiente, aprovechando múltiples núcleos
de CPU para acelerar cada época.
"""

from Parallel.core_validator import get_physical_cores, validate_partition_count
from Parallel.worker import train_partition_worker

__all__ = [
    "train_partition_worker",
    "validate_partition_count",
    "get_physical_cores",
]
