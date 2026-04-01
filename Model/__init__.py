"""
Model — Lógica central de redes neuronales (CNN + MLP).

Este paquete encapsula la arquitectura de dos capas de la red:

1. **CNN (pytorch)**: Extractor de características que transforma imágenes (3, 32, 32)
   en representaciones vectoriales compactas. Usa ResNet18 preentrenado o arquitectura
   simple customizable. Soporta caching automático por hash MD5 de pesos.

2. **MLP (NumPy)**: Perceptrón multicapa que realiza clasificación sobre las features
   extraídas por CNN. Implementa forward pass, backpropagation y SGD con momentum
   de forma eficiente vectorizada.

Módulos
-------
cnn_extractor : modulo
    Clase CNNExtractor con soporte para caching de features y pesos.
    Características: MD5 hashing, batch processing, device management.

mlp : modulo
    Funciones NumPy para inicialización, forward pass, backward pass,
    evaluación y aplicación de gradientes. Optimizado para distribución.

Exportaciones principales
-------------------------
CNNExtractor : class
    Envoltorio de PyTorch CNN con gestión de dispositivo (CPU/GPU) y caching.

init_params : function
    Inicializa pesos MLP con distribución normal (He initialization).

forward_and_gradients : function
    Pase forward + backward de MLP. Retorna logits, loss y gradientes.

evaluate : function
    Evalúa logits vs etiquetas (accuracy y cross-entropy loss).

apply_gradients : function
    Actualiza pesos usando SGD + momentum.

Uso rápido
----------
    from Model import CNNExtractor, init_params, forward_and_gradients

    # Cargar/entrenar CNN
    cnn = CNNExtractor(arch="resnet18", device="cuda")
    features = cnn.extract_batched(images_batch)  # (N, feature_dim)

    # Entrenar MLP
    params = init_params(feature_dim=512, hidden1=128, hidden2=64, n_classes=10)
    logits, loss, grads = forward_and_gradients(X_features, Y_labels, params)
"""

from Model.cnn_extractor import CNNExtractor
from Model.mlp import init_params, forward_and_gradients, evaluate, apply_gradients

__all__ = [
    "CNNExtractor",
    "init_params",
    "forward_and_gradients",
    "evaluate",
    "apply_gradients",
]
