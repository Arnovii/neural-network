"""
Parallel/core_validator.py

Validación del número de particiones contra los núcleos físicos
de la CPU usando psutil.

La regla de oro del Algoritmo de Diego con multiprocessing es que
el número de particiones nunca debe exceder la cantidad de núcleos
físicos disponibles, ya que cada partición se entrena en un proceso
independiente del sistema operativo.
"""

import psutil


def get_physical_cores() -> int:
    """
    Retorna la cantidad de núcleos físicos de la CPU.

    Usa ``psutil.cpu_count(logical=False)`` para obtener núcleos
    físicos reales (no hilos lógicos / hyperthreading).

    Si psutil no puede determinar los núcleos físicos, utiliza
    los lógicos como respaldo.

    :return: Número de núcleos físicos.
    :rtype: int
    """
    cores = psutil.cpu_count(logical=False)

    # Fallback: psutil no pudo determinar núcleos físicos
    if cores is None:
        cores = psutil.cpu_count(logical=True) or 1

    return cores


def validate_partition_count(num_partitions: int) -> None:
    """
    Valida que el número de particiones no exceda los núcleos físicos.

    :param num_partitions: Número de particiones solicitado.
    :type num_partitions: int

    :raises ValueError: Si num_partitions supera los núcleos físicos.
    """
    physical_cores = get_physical_cores()

    if num_partitions > physical_cores:
        raise ValueError(
            f"El número de particiones ({num_partitions}) excede el número "
            f"de núcleos físicos de la CPU ({physical_cores}). "
            f"Máximo permitido: {physical_cores}."
        )
