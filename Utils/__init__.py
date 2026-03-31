"""
Utils — Utilidades para manejo de datos y exportación de resultados.

Prove funcionalidades esenciales para el pipeline de entrenamiento distribuido:

1. **Carga de datos CIFAR-10**: Disponible en dos formatos:
   
   - Pickle (``cifar-10-batches-py/*``): Carga estándar from Keras/TensorFlow.
   - NPZ (``cifar10_train_nchw.npz``): Formato optimizado NCHW precompilado.
   
   Normalización automática: resta media (0.491, 0.482, 0.446), divide por
   std (0.247, 0.244, 0.261). Conversión a formato NCHW.

2. **Logging centralizado**: Clase FormattedLogger con colorización, timestamps,
   y fases de entrenamiento (load, prep, train, eval, cnn, ps, worker, etc).

3. **Exportación de resultados**: Serialización de historiales de training
   (pérdidas, accuracies) a JSON con metadatos (épocas, arquitectura, etc).

Módulos
-------
cifar_loader : modulo
    Funciones para cargar CIFAR-10 (train/test) con normalización.
    Soporta tanto formato Pickle como NPZ.
    Gestión automática de memoria y conversión de tensores.

logging_util : modulo
    Clase FormattedLogger con métodos para cada fase de entrenamiento.
    Colorización ANSI, timestamps, agregación de métricas.

results_exporter : modulo
    Funciones para guardar y cargar historiales de training en JSON.
    Preserva metadatos y permite reproducibilidad.

Exportaciones principales
-------------------------
load_cifar10_train : function
    Carga dataset de entrenamiento CIFAR-10.
    
    :return: (X_train, Y_train) ambos normalizados y en formato NCHW.
    :rtype: tuple[np.ndarray, np.ndarray]

load_cifar10_test : function
    Carga dataset de prueba CIFAR-10.
    
    :return: (X_test, Y_test) ambos normalizados y en formato NCHW.
    :rtype: tuple[np.ndarray, np.ndarray]

NUM_CLASSES : int
    Constante = 10 (número de clases CIFAR-10).

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
