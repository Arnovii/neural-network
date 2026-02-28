"""
Parallel/worker.py

Función worker de nivel módulo para el entrenamiento paralelo
de particiones.

Esta función se ejecuta en un proceso hijo del sistema operativo
creado por ``multiprocessing.Pool``. Debe ser una función de nivel
módulo (no un método ni un closure) para que Python pueda
serializarla con pickle.

Cada worker:
    1. Recibe los parámetros globales de la red y una partición.
    2. Crea una instancia independiente de DiegoNeuronalNetwork.
    3. Establece los parámetros globales.
    4. Entrena sobre su partición con train_on_batch.
    5. Retorna los parámetros entrenados y las métricas.
"""

import os
import sys
from typing import Any, Dict, Tuple

import numpy as np

# Evita sobresubscripción de hilos BLAS dentro de cada proceso worker.
# Si NumPy usa internamente MKL/OpenBLAS con múltiples hilos, y tenemos
# N procesos worker, tendríamos N × M hilos compitiendo por la CPU.
# Forzar 1 hilo por worker garantiza que cada proceso use exactamente
# un núcleo físico.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def train_partition_worker(
    args: Tuple[
        np.ndarray,
        np.ndarray,
        Dict[str, np.ndarray],
        int,
        int,
        int,
        float,
        int,
    ],
) -> Tuple[Dict[str, Any], float, float, int]:
    """
    Entrena una partición en un proceso independiente.

    Crea una red neuronal nueva, establece los parámetros globales,
    entrena sobre la partición y retorna los parámetros actualizados.

    :param args: Tupla con:\n
        - X_part: Imágenes de la partición (N, 784)
        - Y_part: Etiquetas de la partición (N,)
        - global_params: Parámetros globales {W1, b1, W2, b2}
        - input_size: Neuronas de entrada
        - hidden_size: Neuronas en capa oculta
        - output_size: Neuronas de salida
        - learning_rate: Tasa de aprendizaje
        - partition_index: Índice de la partición (para orden)
    :type args: Tuple

    :return: Tupla (parámetros entrenados, loss, accuracy, índice)
    :rtype: Tuple[Dict[str, Any], float, float, int]
    """
    (
        X_part,
        Y_part,
        global_params,
        input_size,
        hidden_size,
        output_size,
        learning_rate,
        partition_index,
    ) = args

    # Asegura que la raíz del proyecto esté en sys.path.
    # En Windows, multiprocessing usa 'spawn', que re-importa módulos
    # en cada proceso hijo sin heredar el sys.path del padre.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Importa dentro del worker para que cada proceso tenga su propio estado
    from Networks.nn_diego import DiegoNeuronalNetwork

    # Crea una red en este proceso.
    # random_seed=None porque los parámetros se sobrescriben inmediatamente
    # con set_parameters (la inicialización Xavier se descarta).
    network = DiegoNeuronalNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        random_seed=None,
    )

    # Establece los parámetros globales recibidos del proceso principal
    network.set_parameters(global_params)

    # Entrena sobre esta partición
    loss, accuracy = network.train_on_batch(X_part, Y_part, learning_rate)

    # Retorna los parámetros actualizados y las métricas
    return (network.get_parameters(), loss, accuracy, partition_index)


"""
NOTAS SOBRE EL USO DE HILOS

NumPy usa internamente bibliotecas como:
    * MKL
    * OpenBLAS
    * OMP

Estas pueden usar múltiples hilos por proceso.

Por poner un ejemplo, si se tiene 8 procesos worker y
cada uno usa 8 hilos internos,  terminas con 64 hilos
compitiendo por CPU, lo que se traduce en un peor rendimiento.

Cuando se habla de "Sobresubscripción de hilos", nos referinos
a que hay más hilos que núcleos disponibles. Esto hace que el
SO empiece a pausar hilos, reanudarlos y cambiar entre ellos
constantemente; lo que reduce el rendimiento, consume más energía
y hace que todo vaya más lento.

Lo que hacemos para evitar ese problema, es decile explícitamente
a cada una de esas librerías que se limiten a usar un único hilo.


NOTAS SOBRE MULTIPROCESSING EN WINDOWS

Cuando se usa multiprocessing, Python tiene que crear nuevos procesos.
Existen distintas maneras de crear un proceso nuevo.

En Python existen 3 métodos principales para crear procesos:
    * fork
    * spawn
    * forkserver

En Windows, solo existe Spawn.

Spawn significa: crea un proceso nuevo desde cero. Es como si se
abriera un nuevo Python completamente limpio, así que no se hereda
la memoria del proceso padre. Esto significa que no se heredan los
imports ya cargados, ni las variables globales, ni el estado
interno.

Un caso distinto sería el Linux, donde se usa Fork normalmente, eso
significa: clonar el proceso actual. Ese método para crear procesos
permita al proceso hijo ser una fotocopia exacta de la memoria del
padre. Sin embargo, Windows no lo permite, y eso ya es cuestión del
sistema operativo.

La razón por la que se necesita este archivo a parte, es que si le
asignamos a un proceso la ejecución del train_diego(), eso ocasionará
un bucle de creación infinita de procesos. Es decir, el proceso padre
crea un hijo, el hijo vuelve a ejecutar la función, esa función vuelve
a crear un nuevo proceso, y así sucesivamente.
"""
