"""Model — Redes neuronales para ImageNet-1k (CNN + MLP en PyTorch).

Este paquete encapsula la arquitectura de dos capas con DOS MODOS distintos:

1. **CNN (PyTorch)**: Extractor de características desde imágenes (3, 224, 224)
   a representaciones vectoriales de 512 dimensiones. Arquitecturas:

   - **ResNet-18 preentrenado** (ImageNet IMAGENET1K_V1): ~50M params
     → CONGELADA (requires_grad=False) → Actúa como extractor fijo, NO se entrena
     
   - **SimpleCNN custom**: 3 bloques Conv→BN→ReLU→MaxPool ~1M params
     → ENTRENABLE (requires_grad=True) → Participa en E2E backward pass

   La CNN implementa:
   - __init__(arch, pretrained, device, seed): Establece requires_grad según arquitectura
   - _build(): Construye modelo y congela/descongela según arquitectura
   - extract_batched(): Extrae features de batch de imágenes (forward solo, sin grad)
   - _get_weights_bytes(): Serializa state_dict para TCP
   - load_weights_from_bytes(): Carga state_dict desde TCP

2. **MLP (PyTorch)**: Clasificador multicapa que realiza clasificación 1k-way
   sobre las features extraídas por CNN. Arquitectura:

   feature_dim (512) → fc1 (hidden1, ReLU) → fc2 (hidden2, ReLU) → fc3 (1000 clases)

   SIEMPRE ENTRENABLE (requires_grad=True en ambos modos):
   - En SimpleCNN: CNN + MLP reciben gradientes ambos
   - En ResNet-18: Solo MLP recibe gradientes (CNN congelada permanentemente)

   Implementa:
   - forward(): Pase forward (fc1→ReLU→fc2→ReLU→fc3)
   - state_dict_numpy(): Exporta pesos a Dict[str, np.ndarray] para PS
   - load_state_dict_numpy(): Carga pesos desde Dict[str, np.ndarray] del PS

MÓDULOS
=======

cnn_extractor : module
    Clase CNNExtractor: Extractor CNN con soporte ResNet-18 (CONGELADA) + SimpleCNN (ENTRENABLE).
    Comportamiento ESTABLECIDO EN __init__, NO es dinámico:
    - __init__(arch, pretrained, device, seed): Establece requires_grad PERMANENTEMENTE
      * arch='resnet18' → requires_grad=False (CNN nunca recibe gradientes)
      * arch='simple' → requires_grad=True (CNN participa en E2E backprop)
    - _build(): Construcción de arquitectura + aplicación de requires_grad
    - feature_dim: Propiedad (siempre 512)
    - extract_batched(): Forward pass de imágenes (sin gradientes)
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
    - arch: 'resnet18' (CONGELADA, MLP-only) | 'simple' (ENTRENABLE, E2E)
    - pretrained: True (carga IMAGENET1K_V1 para resnet18)
    - device: 'cpu' | 'cuda' | 'cuda:0' | 'mps'
    - seed: Optional[int] (default None, para reproducibilidad en SimpleCNN)
    
    NOTA: requires_grad se establece en __init__ y NUNCA cambia durante entrenamiento.
    - ResNet-18: requires_grad=False PERMANENTEMENTE (feature extractor)
    - SimpleCNN: requires_grad=True PERMANENTEMENTE (participates in E2E training)

MLPPyTorch : class
    from Model.mlp_pytorch import MLPPyTorch

    Parámetros:
    - feature_dim: Dimensión de entrada (512 para ResNet-18)
    - hidden1: Neuronas capa 1 (default 1024)
    - hidden2: Neuronas capa 2 (default 512)
    - n_classes: Clases salida (1000 para ImageNet-1k)

FLUJO TÍPICO (SimpleCNN E2E)
============================

    # PS: Cargar CNN entrenable
    cnn = CNNExtractor(arch='simple', pretrained=False)
    cnn_bytes = cnn._get_weights_bytes()  # serializar para enviar a Workers

    # Worker: Recibir y entrenar CNN + MLP
    mlp = MLPPyTorch(feature_dim=512, hidden1=1024, hidden2=512, n_classes=1000)

    # Loop de entrenamiento E2E
    features = cnn.extract_batched(image_batch)  # (N, 512), CNN forward sin grad
    logits = mlp(features)  # (N, 1000)
    loss = F.cross_entropy(logits, labels)
    loss.backward()  # ← Backprop: MLP + CNN ambos reciben gradientes
    # ... SGD update en CNN y MLP ...

    # Enviar parámetros al PS
    cnn_params = cnn.state_dict_numpy()  # actualizados
    mlp_params = mlp.state_dict_numpy()  # actualizados

FLUJO TÍPICO (ResNet-18 MLP-only)
==================================

    # PS: Cargar CNN congelada
    cnn = CNNExtractor(arch='resnet18', pretrained=True)  # requires_grad=False
    cnn_bytes = cnn._get_weights_bytes()

    # Worker: Recibir y entrenar solo MLP
    mlp = MLPPyTorch(feature_dim=512, hidden1=1024, hidden2=512, n_classes=1000)

    # Loop de entrenamiento MLP-only
    features = cnn.extract_batched(image_batch)  # (N, 512), CNN frozen
    logits = mlp(features)  # (N, 1000)
    loss = F.cross_entropy(logits, labels)
    loss.backward()  # ← Backprop: Solo MLP recibe gradientes (CNN congelada)
    # ... SGD update solo en MLP ...

    # Enviar parámetros al PS
    cnn_params = cnn.state_dict_numpy()  # NO cambian (congelada)
    mlp_params = mlp.state_dict_numpy()  # actualizados (entrenada)
"""

from Model.cnn_extractor import CNNExtractor
from Model.mlp_pytorch import MLPPyTorch

__all__ = [
    "CNNExtractor",
    "MLPPyTorch",
]
