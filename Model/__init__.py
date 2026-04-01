"""
Model — Redes neuronales E2E para ImageNet-1k (CNN + MLP en PyTorch).

Este paquete encapsula la arquitectura de dos capas de la red distribuida:

1. **CNN (PyTorch)**: Extractor de características que transforma imágenes (3, 224, 224)
   en representaciones vectoriales de 512 dimensiones. Arquitecturas soportadas:

   - ResNet-18 preentrenado (ImageNet IMAGENET1K_V1): ~50M params (recomendado)
   - Simple CNN custom: 3 bloques Conv→BN→ReLU→MaxPool (~1M params)

   La CNN implementa:
   - _build(): Construye modelo según arquitectura
   - extract_batched(): Extrae features de batch de imágenes
   - _get_weights_bytes(): Serializa state_dict para TCP
   - load_weights_from_bytes(): Carga state_dict desde TCP

2. **MLP (PyTorch)**: Clasificador multicapa que realiza clasificación 1k-way
   sobre las features extraídas por CNN. Arquitectura:

   feature_dim (512) → fc1 (hidden1, ReLU) → fc2 (hidden2, ReLU) → fc3 (1000 clases)

   Implementa:
   - forward(): Pase forward (fc1→ReLU→fc2→ReLU→fc3)
   - state_dict_numpy(): Exporta pesos a Dict[str, np.ndarray] para PS
   - load_state_dict_numpy(): Carga pesos desde Dict[str, np.ndarray] del PS

MÓDULOS
=======

cnn_extractor : module
    Clase CNNExtractor: Extractor CNN con soporte ResNet-18 + Simple.
    Métodos principales:
    - __init__(arch, pretrained, device, seed)
    - _build(): Construcción de arquitectura
    - feature_dim: Propiedad (siempre 512)
    - extract_batched(): Forward pass de imágenes
    - _get_weights_bytes(): Serialización para TCP
    - load_weights_from_bytes(): Deserialización desde TCP

mlp_pytorch : module
    Clase MLPPyTorch(nn.Module): Clasificador MLP en PyTorch.
    Métodos principales:
    - __init__(feature_dim, hidden1, hidden2, n_classes)
    - forward(x): Pase forward (N, feature_dim) → (N, n_classes)
    - state_dict_numpy(): Exporta parámetros como Dict[str, np.ndarray]
    - load_state_dict_numpy(state): Carga parámetros desde Dict[str, np.ndarray]

EXPORTACIONES PRINCIPALES
==========================

CNNExtractor : class
    from Model.cnn_extractor import CNNExtractor

    Parámetros:
    - arch: 'resnet18' (recomendado) | 'simple'
    - pretrained: True (carga IMAGENET1K_V1 para resnet18)
    - device: 'cpu' | 'cuda' | 'cuda:0' | 'mps'
    - seed: int o None (para reproducibilidad)

MLPPyTorch : class
    from Model.mlp_pytorch import MLPPyTorch

    Parámetros:
    - feature_dim: Dimensión de entrada (512 para ResNet-18)
    - hidden1: Neuronas capa 1 (default 1024)
    - hidden2: Neuronas capa 2 (default 512)
    - n_classes: Clases salida (1000 para ImageNet-1k)

FLUJO TÍPICO
============

    # PS: Cargar CNN una sola vez
    cnn = CNNExtractor(arch='resnet18', pretrained=True)
    cnn_bytes = cnn._get_weights_bytes()  # serializar para enviar a Workers

    # Worker: Recibir y entrenar MLP
    mlp = MLPPyTorch(feature_dim=512, hidden1=1024, hidden2=512, n_classes=1000)

    # Loop de entrenamiento
    features = cnn.extract_batched(image_batch)  # (N, 512)
    logits = mlp(features)  # (N, 1000)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    # ... SGD update ...

    # Enviar parámetros al PS
    params = mlp.state_dict_numpy()
"""

from Model.cnn_extractor import CNNExtractor
from Model.mlp_pytorch import MLPPyTorch

__all__ = [
    "CNNExtractor",
    "MLPPyTorch",
]
