"""
Model — Redes neuronales E2E para ImageNet-1k (CNN + MLP en PyTorch).

Este paquete encapsula la arquitectura de dos capas de la red distribuida:

1. **CNN (PyTorch)**: Extractor de características que transforma imágenes (3, 224, 224)
   en representaciones vectoriales de 512 dimensiones. Arquitecturas soportadas:

   - ResNet-18 preentrenado (ImageNet IMAGENET1K_V1): ~50M params (recomendado)
   - Simple CNN custom: 3 bloques Conv→BN→ReLU→MaxPool (~1M params)

   La CNN se congela en eval() durante el entrenamiento distribuido (los pesos
   se envían por TCP pero solo el MLP recibe gradientes en Workers).

2. **MLP (PyTorch)**: Clasificador multicapa que realiza clasificación 1k-way
   sobre las features extraídas por CNN. Arquitectura:

   feature_dim (512) → hidden1 (1024 default) → hidden2 (512 default) → 1000 clases

   Implementa forward pass, backward, SGD local. Sincronización de parámetros
   mediante FedAvg asíncrono en PS.

MÓDULOS
=======

cnn_extractor : module
    Clase CNNExtractor con soporte para ResNet-18 (pretrained) y Simple CNN.
    - _build(): Construye modelo según arquitectura
    - extract_batched(): Extrae features de batch de imágenes
    - _get_weights_bytes(): Serializa state_dict para TCP
    - load_weights_from_bytes(): Carga state_dict desde TCP

mlp_pytorch : module
    Clase MLPPyTorch: MLP en PyTorch con forward pass y gradient support.
    - __init__(): Inicializa capas fc1, fc2, fc3 con dims configurables
    - forward(): Pase forward (fc1→ReLU→fc2→ReLU→fc3)
    - state_dict_numpy(): Exporta pesos a Dict[str, np.ndarray] para PS
    - Integración automática con torch.optim para SGD local

EXPORTACIONES PRINCIPALES
==========================

CNNExtractor : class
    from Model.cnn_extractor import CNNExtractor
    
    arch='resnet18' | arch='simple'
    pretrained=True (solo para resnet18)
    device='cpu' | 'cuda' | 'cuda:0' | 'mps'

MLPPyTorch : class
    from Model.mlp_pytorch import MLPPyTorch
    
    feature_dim: dimensión de entrada (512 para ResNet-18)
    hidden1, hidden2: capas ocultas (1024, 512 default)
    n_classes: 1000 (ImageNet-1k)

FLUJO TÍPICO
============

    # PS: Cargar CNN una sola vez
    cnn = CNNExtractor(arch='resnet18', pretrained=True, device='cpu')
    cnn_bytes = cnn._get_weights_bytes()  # serializar para enviar

    # Worker: Recibir y entrenar MLP
    mlp = MLPPyTorch(feature_dim=512, hidden1=1024, hidden2=512, n_classes=1000)
    # ... loop de entrenamiento ...
    params = mlp.state_dict_numpy()  # exportar para PS
"""
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
