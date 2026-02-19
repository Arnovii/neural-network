"""
Este módulo proporciona funciones para particionar el dataset MNIST
en múltiples subconjuntos de manera uniforme y aleatoria.
"""

import random
import os
import traceback
from typing import List, Tuple, Dict

# =====================
# IMPORTS DE LOADER
# =====================

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mnist_loader import load_mnist_train, get_data_directory

# =====================
# FUNCIÓN DE PARTICIÓN
# =====================


def partition_mnist_data(
    num_partitions: int,
    n_train: int | None = None,
    data_dir: str | None = None,
    download_if_missing: bool = True,
    random_seed: int | None = None,
    verbose: bool = True,
) -> List[Tuple[List[List[float]], List[int]]]:
    """
    Carga los datos MNIST y realiza un particionamiento estratificado por clase.\n
    Garantiza que cada partición tendrá una proproción similar
    de cada dígito.

    :param num_partitions: Número de particiones deseadas (debe ser >= 1)
    :type num_partitions: int
    :param n_train: Cantidad de ejemplos de entrenamiento a usar (opcional).
    :type n_train: int | None
    :param data_dir: Directorio donde se encuentran/estarán los archivos MNIST (Por defecto Data/)
    :type data_dir: Optional[str]
    :param download_if_missing: Si True, descarga los datos si no existen.
    :type download_if_missing: bool
    :param random_seed: Semilla para reproducibilidad (opcional).
    :type random_seed: int | None
    :param verbose: Si True, muestra mensajes de progreso.
    :type verbose: bool
    :return: Lista de tuplas (X_partition, Y_partition)
    :rtype: List[Tuple[List[List[float]], List[int]]]
    """
    if num_partitions < 1:
        raise ValueError("El número de particiones debe ser al menos 1")

    # Establece la semilla si se especifica
    if random_seed is not None:
        random.seed(random_seed)

    # Obtiene directorio de datos si no se especificó
    if data_dir is None:
        data_dir = get_data_directory()

    # Avisos de progreso
    if verbose:
        print("=" * 70)
        print("PARTICIONAMIENTO DE DATOS MNIST")
        print("=" * 70)
        print(f"\nDirectorio de datos: {data_dir}")
        print(f"Número de particiones: {num_partitions}")
        if random_seed is not None:
            print(f"Semilla aleatoria: {random_seed}")

    # Carga datos de entrenamiento usando el loader
    if verbose:
        print("\nCargando datos de entrenamiento...")

    X_train, Y_train = load_mnist_train(
        data_dir=data_dir,
        n_train=n_train,
        download_if_missing=download_if_missing,
        verbose=verbose,
    )

    if verbose:
        print(f"\nTotal de ejemplos de entrenamiento: {len(X_train)}")

    # Organiza índices por clase (estratificación)
    indices_by_class: Dict[int, List[int]] = {i: [] for i in range(10)}
    for idx, label in enumerate(Y_train):
        indices_by_class[label].append(idx)

    # Muestra distribución original
    if verbose:
        print("\nDistribución por clase en el dataset completo:")
        for digit in range(10):
            count = len(indices_by_class[digit])
            percentage = 100.0 * count / len(Y_train)
            bar = "█" * int(percentage / 2)
            print(f"  Dígito {digit}: {count:5d} ejemplos ({percentage:5.2f}%) {bar}")

    # Inicializa particiones vacías
    partitions: List[List[int]] = [[] for _ in range(num_partitions)]

    # Distribuye índices de cada clase uniformemente entre particiones
    if verbose:
        print(
            f"\nDistribuyendo datos en {num_partitions} particiones estratificadas..."
        )

    for digit in range(10):
        indices = indices_by_class[digit][:]
        random.shuffle(indices)  # Mezcla aleatoriamente los índices de esta clase

        # Distribuye uniformemente usando round-robin
        for i, idx in enumerate(indices):
            partition_idx = i % num_partitions
            partitions[partition_idx].append(idx)

    # Mezcla cada partición para evitar orden por clase
    for partition in partitions:
        random.shuffle(partition)

    # Crea los conjuntos de datos finales
    result: List[Tuple[List[List[float]], List[int]]] = []

    if verbose:
        print(f"\n{'=' * 70}")
        print("RESULTADOS DEL PARTICIONAMIENTO")
        print("=" * 70)

    for i, partition_indices in enumerate(partitions):
        X_partition = [X_train[idx] for idx in partition_indices]
        Y_partition = [Y_train[idx] for idx in partition_indices]

        result.append((X_partition, Y_partition))

        # Muestra estadísticas de esta partición
        if verbose:
            class_counts = {digit: 0 for digit in range(10)}
            for label in Y_partition:
                class_counts[label] += 1

            print(
                f"\n  Partición {i + 1}: {len(X_partition)} ejemplos ({100.0 * len(X_partition) / len(X_train):.1f}%)"
            )
            print("    Distribución por clase: ", end="")
            for digit in range(10):
                print(f"{digit}:{class_counts[digit]} ", end="")
            print()

    if verbose:
        print("\n" + "=" * 70)
        print("✓ Particionamiento completado exitosamente")
        print("=" * 70)

    return result


def partition_mnist_data_simple(
    num_partitions: int,
    X_train: List[List[float]],
    Y_train: List[int],
    random_seed: int | None = None,
    verbose: bool = False,
) -> List[Tuple[List[List[float]], List[int]]]:
    """
    Versión simplificada que trabaja directamente con datos ya cargados.\n
    Realiza un particionamiento estratificado por clase.\n
    Garantiza que cada partición tendrá una proproción similar
    de cada dígito.

    :param num_partitions: Número de particiones deseadas (debe ser >= 1)
    :type num_partitions: int
    :param X_train: Lista de imágenes de entrenamiento
    :type X_train: List[List[float]]
    :param Y_train: Lista de etiquetas de entrenamiento
    :type Y_train: List[int]
    :param random_seed: Semilla para reproducibilidad (Opcional)
    :type random_seed: int | None
    :param verbose: Si True, muestra mensajes de progreso
    :type verbose: bool
    :return: Lista de tuplas (X_partition, Y_partition)
    :rtype: List[Tuple[List[List[float]], List[int]]]
    """
    if num_partitions < 1:
        raise ValueError("El número de particiones debe ser al menos 1")

    if len(X_train) != len(Y_train):
        raise ValueError("X_train y Y_train deben tener la misma longitud")

    if random_seed is not None:
        random.seed(random_seed)

    if verbose:
        print(
            f"Particionando {len(X_train)} ejemplos en {num_partitions} particiones..."
        )

    # Organiza índices por clase
    indices_by_class: Dict[int, List[int]] = {i: [] for i in range(10)}
    for idx, label in enumerate(Y_train):
        indices_by_class[label].append(idx)

    # Inicializa particiones
    partitions: List[List[int]] = [[] for _ in range(num_partitions)]

    # Distribuye índices de cada clase uniformemente
    for digit in range(10):
        indices = indices_by_class[digit][:]
        random.shuffle(indices)

        for i, idx in enumerate(indices):
            partition_idx = i % num_partitions
            partitions[partition_idx].append(idx)

    # Mezcla cada partición
    for partition in partitions:
        random.shuffle(partition)

    # Crea resultados
    result: List[Tuple[List[List[float]], List[int]]] = []
    for partition_indices in partitions:
        X_partition = [X_train[idx] for idx in partition_indices]
        Y_partition = [Y_train[idx] for idx in partition_indices]
        result.append((X_partition, Y_partition))

    if verbose:
        print(f"✓ Creadas {len(result)} particiones")

    return result


# =====================
# FUNCIONES AUXILIARES
# =====================


def get_partition_statistics(
    partitions: List[Tuple[List[List[float]], List[int]]],
) -> List[Dict]:
    """
    Calcula estadísticas para cada partición.

    :param partitions: Lista de particiones devuelta por partition_mnist_data
    :type partitions: List[Tuple[List[List[float]], List[int]]]
    :return: Lista de diccionarios con estadísticas de cada partición
    :rtype: List[Dict[Any, Any]]
    """
    stats = []

    for i, (X, Y) in enumerate(partitions):
        class_counts = {digit: 0 for digit in range(10)}
        for label in Y:
            class_counts[label] += 1

        partition_stats = {
            "partition_id": i,
            "total_examples": len(X),
            "class_distribution": class_counts,
            "image_size": len(X[0]) if X else 0,
        }
        stats.append(partition_stats)

    return stats


def print_partition_summary(partitions: List[Tuple[List[List[float]], List[int]]]):
    """
    Imprime un resumen formateado de las particiones.
    """
    print("=" * 70)
    print("RESUMEN DE PARTICIONES MNIST")
    print("=" * 70)

    total_examples = sum(len(X) for X, _ in partitions)

    print(f"\nNúmero total de particiones: {len(partitions)}")
    print(f"Total de ejemplos: {total_examples}")
    print(f"Promedio por partición: {total_examples / len(partitions):.1f}")

    for i, (X, Y) in enumerate(partitions):
        class_counts = {digit: 0 for digit in range(10)}
        for label in Y:
            class_counts[label] += 1

        print(f"\n--- Partición {i + 1} ---")
        print(f"  Ejemplos: {len(X)}")
        print(f"  Porcentaje del total: {100.0 * len(X) / total_examples:.1f}%")
        print("  Distribución de clases:")
        for digit in range(10):
            count = class_counts[digit]
            percentage = 100.0 * count / len(X) if X else 0
            bar = "█" * int(percentage / 5)
            print(f"    Dígito {digit}: {count:4d} ({percentage:5.1f}%) {bar}")

    print("\n" + "=" * 70)


def merge_partitions(
    partitions: List[Tuple[List[List[float]], List[int]]], indices_to_merge: List[int]
) -> Tuple[List[List[float]], List[int]]:
    """
    Combina múltiples particiones en una sola.

    :param partitions: Lista de todas las particiones disponibles
    :type partitions: List[Tuple[List[List[float]], List[int]]]
    :param indices_to_merge: Índices de las particiones a combinar
    :type indices_to_merge: List[int]
    :return: Tupla (X_merged, Y_merged) con los datos combinados
    :rtype: Tuple[List[List[float]], List[int]]
    """
    X_merged = []
    Y_merged = []

    for idx in indices_to_merge:
        X, Y = partitions[idx]
        X_merged.extend(X)
        Y_merged.extend(Y)

    return X_merged, Y_merged


# =================
# EJEMPLO DE USO
# =================


def main():
    """
    Ejemplo de uso del particionador de datos MNIST.
    """
    print("\n" + "=" * 70)
    print("UTILIDAD DE PARTICIONAMIENTO DE DATOS MNIST")
    print("=" * 70)

    # Ejemplo 1: Crear 5 particiones estratificadas
    print("\n" + "=" * 70)
    print("EJEMPLO 1: Crear 5 particiones estratificadas")
    print("=" * 70)

    try:
        partitions = partition_mnist_data(
            num_partitions=5,
            n_train=5000,
            download_if_missing=True,
            random_seed=42,  # Para reproducibilidad
            verbose=True,
        )

        # Imprimir resumen detallado
        print_partition_summary(partitions)

        # Ejemplo de cómo usar las particiones para validación cruzada
        print("\n" + "=" * 70)
        print("EJEMPLO DE USO: Validación Cruzada (5-fold)")
        print("=" * 70)
        print("\nCódigo para entrenar con validación cruzada:")
        print("-" * 70)
        print("""
from Utils.data_partitioner import partition_mnist_data, merge_partitions

# Crear 5 particiones
partitions = partition_mnist_data(num_partitions=5, random_seed=42)

for fold in range(5):
    # Usar una partición para validación
    X_val, Y_val = partitions[fold]
    
    # Combinar el resto para entrenamiento
    train_indices = [i for i in range(5) if i != fold]
    X_train, Y_train = merge_partitions(partitions, train_indices)
    
    print(f"Fold {fold + 1}: {len(X_train)} entrenamiento, {len(X_val)} validación")
    
    # Entrenar modelo con X_train, Y_train
    # Evaluar en X_val, Y_val
        """)
        print("-" * 70)

        # Ejemplo 2: Crear particiones para entrenamiento distribuido
        print("\n" + "=" * 70)
        print("EJEMPLO 2: Entrenamiento Distribuido (Algoritmo de Diego)")
        print("=" * 70)

        federated_partitions = partition_mnist_data(
            num_partitions=10,  # 10 clientes
            download_if_missing=False,  # Ya descargados
            random_seed=123,
            verbose=True,
        )

        print(f"\n✓ Creadas {len(federated_partitions)} particiones para 10 clientes")
        print("  Cada cliente recibe una partición representativa del dataset")

        print("\n  Distribución de ejemplos por cliente:")
        for i, (X, Y) in enumerate(federated_partitions):
            print(f"    Cliente {i + 1:2d}: {len(X):5d} ejemplos")

        # Ejemplo 3: Usar con datos ya cargados
        print("\n" + "=" * 70)
        print("EJEMPLO 3: Particionar datos ya cargados")
        print("=" * 70)

        X, Y = load_mnist_train(verbose=False)
        print(f"\nDatos cargados: {len(X)} ejemplos")

        # Particionar usando la versión simple
        sub_partitions = partition_mnist_data_simple(
            num_partitions=3,
            X_train=X[:10000],  # Solo primeros 10k para el ejemplo
            Y_train=Y[:10000],
            random_seed=42,
            verbose=True,
        )

        print(f"\n✓ Creadas {len(sub_partitions)} sub-particiones")
        for i, (X_sub, Y_sub) in enumerate(sub_partitions):
            print(f"  Partición {i + 1}: {len(X_sub)} ejemplos")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
